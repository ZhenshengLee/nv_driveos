/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * OpenWFD NvSci Sample
 *
 * This sample application demonstrates the usage of the OpenWFD APIs
 * for displaying NvSciBuf backed buffer objects.
 */

#include <iostream>
#include <thread>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <random>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>


// The extension string for WFD_NVX_create_source_from_nvscibuf
// needs to be defined before including the WFD headers
#define WFD_NVX_create_source_from_nvscibuf
#define WFD_WFDEXT_PROTOTYPES
#include <WF/wfd.h>
#include <WF/wfdext.h>

#include "nvscibuf.h"
#include "nvscisync.h"

#define BMP_SIGNATURE_OFFSET     0x00
#define BMP_DATA_START_OFFSET    0x0A
#define BMP_WIDTH_OFFSET         0x12
#define BMP_HEIGHT_OFFSET        0x16
#define BMP_BPP_OFFSET           0x1C

/* BMP Header size is 54 bytes although this sample is using only first 30 bytes */
#define BMP_HEADER_SIZE          0x36
#define BMP_SIGNATURE_SIZE       0x02

// Helper macro for WFD Error checking
#define CHECK_WFD_ERROR(device)                                             \
    {                                                                       \
        WFDErrorCode wfdErr = wfdGetError(device);                          \
        if (wfdErr) {                                                       \
            std::cerr << "WFD Error 0x" << std::hex << wfdErr << " at: " << \
                                        std::dec << std::endl;              \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;          \
        };                                                                  \
    }


// Utility function for splash screen
static int DrawBmp32(const char *file_name, unsigned int bmp_offset, void *ptr_disp,
                     int width, int height, int bpp, int pitch)
{
    unsigned int    bmp_data_offset;
    int             bmp_width;
    int             bmp_height;
    unsigned short  bmp_bpp;
    unsigned int    bmp_bytes_per_pixel;
    unsigned int    bmp_stride;
    unsigned char   *p = (unsigned char *)ptr_disp;
    unsigned char   *pSrc;
    unsigned int    *pDst;
    unsigned char   bmp_header[BMP_HEADER_SIZE] = { 0U };
    unsigned char   *pFullSrc = nullptr;
    unsigned char   *pFullSrc_original = nullptr;
    unsigned char   *pLineSrc = nullptr;
    unsigned int    *pLineDst = nullptr;
    unsigned short  signature;
    int             fd = -1;
    int             w;
    int             res = -1;


    /* This routine assumes display mode is 32 bpp */
    if (bpp != 32) {
        printf("Unsupported display mode with %d bpp\n", bpp);
        return res;
    }

    fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        printf("Failed to open file %s\n", file_name);
        goto fail;
    }
    if (lseek(fd, bmp_offset, SEEK_SET) < 0)
        goto fail;
    if (read(fd, bmp_header, BMP_HEADER_SIZE) < 0)
        goto fail;

    /* Extract pixel data offset, width, height and bpp from bmp file */
    memcpy (&signature, bmp_header + BMP_SIGNATURE_OFFSET, BMP_SIGNATURE_SIZE);
    if (signature != 0x4D42) {
        printf("Invalid bitmap file.\n");
        goto fail;
    }

    memcpy (&bmp_data_offset, bmp_header + BMP_DATA_START_OFFSET, sizeof(bmp_data_offset));
    memcpy (&bmp_width, bmp_header + BMP_WIDTH_OFFSET, sizeof(bmp_width));
    memcpy (&bmp_height, bmp_header + BMP_HEIGHT_OFFSET, sizeof(bmp_height));
    memcpy (&bmp_bpp, bmp_header + BMP_BPP_OFFSET, sizeof(bmp_bpp));
    if (bmp_bpp != 24 && bmp_bpp != 32) {
        printf("Unsupported bits-per-pixel: %d\n", bmp_bpp);
        goto fail;
    }

    if (bmp_height > height || bmp_width > width) {
        /* Bmp too large, bail out */
        printf("bmp too large, not supported.\n");
        printf("bmp     %d x %d\n", bmp_width, bmp_height);
        printf("display %d x %d\n", width, height);
        goto fail;
    }

    bmp_bytes_per_pixel = bmp_bpp >> 3;
    bmp_stride = bmp_width * bmp_bytes_per_pixel;
    bmp_stride += (4 - ((bmp_width * bmp_bytes_per_pixel) & 0x3)) & 0x3;

    /* Center bmp in display, point to bottom line (bmp is inverted) */
    p = (unsigned char *)ptr_disp + (pitch * (((height + bmp_height) / 2) - 1));
    p += ((width - bmp_width) / 2) * 4 /* bpp / 8 = 4 */;

    /* If bmp smaller than display, clear frame buffer */
    if (height > bmp_height || width > bmp_width)
        memset(ptr_disp, 0, height * pitch);

    /* Allocate buffer for source and destination data */
    pFullSrc = (unsigned char *)malloc(bmp_stride * bmp_height);
    if (!pFullSrc)
        goto fail;
    pFullSrc_original = pFullSrc;
    pLineSrc = (unsigned char *)malloc(bmp_stride);
    if (!pLineSrc)
        goto fail;
    pLineDst = (unsigned int *)malloc(bmp_width * 4);
    if (!pLineDst)
        goto fail;

    /* Copy whole pixel data */
    if (lseek (fd, bmp_offset + bmp_data_offset, SEEK_SET) < 0)
        goto fail;
    if (read (fd, pFullSrc, bmp_stride * bmp_height) < 0)
        goto fail;

    /* Draw the splash */
    while (bmp_height--) {
        memcpy(pLineSrc, pFullSrc, bmp_stride);
        pFullSrc += bmp_stride;
        pSrc = (unsigned char *)pLineSrc;
        pDst = pLineDst;
        w = bmp_width;
        while (w--) {
            memcpy(pDst, pSrc, bmp_bytes_per_pixel);
            if (bmp_bytes_per_pixel < 4)
                memset((unsigned char *)pDst + bmp_bytes_per_pixel, 0,
                    4 - bmp_bytes_per_pixel);
            pDst++;
            pSrc += bmp_bytes_per_pixel;
        }
        memcpy(p, pLineDst, 4 * bmp_width);
        p -= pitch;
    }
    res = 0;
    fail:
    if (fd >= 0)
        close(fd);
    if (pLineSrc)
        free(pLineSrc);
    if (pLineDst)
        free(pLineDst);
    if (pFullSrc_original)
        free(pFullSrc_original);

    return res;
}

// Helper for calculating the size of an array.
template <typename T, size_t N>
constexpr size_t ComputeArraySize(T (&inputArray)[N]) {
    return N;
}

static constexpr uint32_t MAX_SUPPORTED_PORTS = 4U;
static constexpr uint32_t MAX_SUPPORTED_PIPES_PER_PORT = 5U;
static constexpr uint32_t MAX_MODES = 40U;

// Static mode attributes
struct WFDPortModeData {
    WFDPortMode m_portMode {WFD_INVALID_HANDLE};
    WFDint m_width {0};
    WFDint m_height {0};
    WFDint m_refreshRate {0};

    WFDPortModeData(WFDPortMode portMode, WFDint width,
                    WFDint height, WFDint refreshRate) : m_portMode(portMode),
                                                         m_width(width),
                                                         m_height(height),
                                                         m_refreshRate(refreshRate) {}
    void PrintConfig();

    WFDPortModeData() = delete;
};

// Static WFDPort attributes
struct WFDPortData {
    WFDint m_portID {0};
    WFDint m_numBindablePipes {0};
    WFDint m_bindablePipeIds[MAX_SUPPORTED_PIPES_PER_PORT];
    WFDint m_numModes {0};
    std::vector<WFDPortModeData> m_portModeData;

    WFDPortData(WFDint portID) : m_portID(portID) {};
    void PrintConfig();
    bool GetIdxIntoPortModeData(const uint32_t width, const uint32_t height,
                                const uint32_t refreshRate, uint32_t &idxIntoPortModeData);

    WFDPortData() = delete;
};

// Static WFD configuration.
class StaticWFDData {
    public:
        static StaticWFDData* GetInstance();
        void PrintWFDConfig();

        std::vector<WFDPortData> m_portData;
        WFDint m_numPorts {0};
        bool m_inited {false};
    protected:
        StaticWFDData();
    private:
        static StaticWFDData* m_instance;
};

// Config to be used by applicable samples
struct SampleConfig {
    uint32_t portIdx;
    uint32_t layerIdx;
    uint32_t modeWidth;
    uint32_t modeHeight;
    uint32_t modeRefreshRate;
    uint32_t imageWidth;
    uint32_t imageHeight;
};

struct NvSciBufResources {
    NvSciBufModule module;
    NvSciBufAttrList unreconciledList;
    NvSciBufAttrList conflictList;
    NvSciBufAttrList reconciledList;
    NvSciBufAttrKeyValuePair reconciledBufAttrs[3];
    NvSciBufObj obj;
    uint8_t *cpuPtr;
    uint64_t allocationSize;
    uint32_t width;
    uint32_t height;
    uint64_t *pitch;
    bool inited;

    NvSciBufResources();
    ~NvSciBufResources();

    bool InitializeRGB(uint32_t imageWidth, uint32_t imageHeight);
    void FillARGB8888Buffer(uint8_t a, uint8_t r, uint8_t g, uint8_t b);
    bool FlushCpuCache(const uint64_t offset = 0U);
};

struct WFDResources {
    WFDDevice device;
    WFDint numPorts;
    WFDint numPipes;
    WFDint numModes;
    WFDint portId;
    WFDint pipeId;
    WFDPort port;
    WFDPipeline pipe;
    WFDPortMode portMode;
    WFDSource source;
    WFDint sourceRect[4];
    NvSciBufObj *bufObj;
    bool inited;

    WFDResources(const SampleConfig &config, NvSciBufObj *obj);

    ~WFDResources() {
        if (device && pipe) {
            // Perform a null flip
            wfdBindSourceToPipeline(device, pipe, (WFDSource)0, WFD_TRANSITION_AT_VSYNC, nullptr);
            wfdDeviceCommit(device, WFD_COMMIT_PIPELINE, pipe);
        }

        if (source) {
            wfdDestroySource(device, source);
        }

        if (pipe) {
            wfdDestroyPipeline(device, pipe);
        }

        if (port) {
            wfdDestroyPort(device, port);
        }

        if (device) {
            wfdDestroyDevice(device);
        }
    }
};

struct NvSciSyncResources {
    NvSciSyncModule module;
    NvSciSyncAttrList unreconciledList;
    NvSciSyncAttrList conflictList;
    NvSciSyncAttrList reconciledList;
    NvSciSyncObj obj;
    NvSciSyncFence fence;
    bool inited;

    NvSciSyncResources() : module(nullptr), fence(NvSciSyncFenceInitializer),
                           inited(false) {}

    ~NvSciSyncResources() {
        if (unreconciledList) {
            NvSciSyncAttrListFree(unreconciledList);
        }

        if (conflictList) {
            NvSciSyncAttrListFree(conflictList);
        }

        if (reconciledList) {
            NvSciSyncAttrListFree(reconciledList);
        }

        if (inited) {
            NvSciSyncFenceClear(&fence);
        }

        if (obj) {
            NvSciSyncObjFree(obj);
        }

        if (module) {
            NvSciSyncModuleClose(module);
        }
    }
};

// Function pointer typedef for the samples
typedef bool SampleFunction(SampleConfig config);

static const uint32_t imagePlaneCount = 1U;
static const uint32_t semiPlanarImageCount = 2U;
static const uint32_t planarImageCount = 3U;
static const uint32_t numFrames = 360U; // Number of frames to display
static const uint32_t pitchAlignment = 128U; // The minimum pitch alignment restriction in NvDisplay.

// Prototypes for all available samples.
static bool Sample1(SampleConfig);
static bool Sample2(SampleConfig);
static bool Sample3(SampleConfig);
static bool Sample4(SampleConfig);
static bool Sample5(SampleConfig);

// Below samples require a specific setup configuration
static bool FrozenFrameDetectionSample(SampleConfig config);
static bool DPMSTSample();
static bool SafeStateSample(SampleConfig config);

// Add a pointer to the sample's function here to enable it automatically.
SampleFunction *sampleFunctionList[] = {
    &Sample1,
    &Sample2,
    &Sample3,
    &Sample4,
    &Sample5
};

static const uint32_t numSamples = ComputeArraySize(sampleFunctionList);

StaticWFDData* StaticWFDData::m_instance = nullptr;

StaticWFDData* StaticWFDData::GetInstance() {

    if (m_instance == nullptr) {
        m_instance = new StaticWFDData;
    }

    return m_instance;
}

StaticWFDData::StaticWFDData() {
    WFDDevice device = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, nullptr);
    if (device) {
        m_numPorts = wfdEnumeratePorts(device, nullptr, 0, nullptr);
        if (m_numPorts > 0) {
            assert(m_numPorts <= static_cast<int32_t>(MAX_SUPPORTED_PORTS));
            std::vector<WFDint> portIDs(m_numPorts);
            m_numPorts = wfdEnumeratePorts(device, portIDs.data(), m_numPorts, nullptr);
            for (auto portID : portIDs) {
                auto tempPort = wfdCreatePort(device, portID, nullptr);
                if (tempPort) {
                    m_portData.emplace_back(portID);
                    m_portData.back().m_numBindablePipes = wfdGetPortAttribi(device, tempPort,
                                                                             WFD_PORT_PIPELINE_ID_COUNT);
                    wfdGetPortAttribiv(device, tempPort, WFD_PORT_BINDABLE_PIPELINE_IDS,
                                       m_portData.back().m_numBindablePipes, m_portData.back().m_bindablePipeIds);
                    m_portData.back().m_numModes = wfdGetPortModes(device, tempPort, nullptr, 0);
                    assert(m_portData.back().m_numModes <= static_cast<WFDint>(MAX_MODES));

                    WFDPortMode portModes[MAX_MODES];
                    m_portData.back().m_numModes = wfdGetPortModes(device, tempPort, portModes,
                                                                   MAX_MODES);
                    for (int32_t idx = 0U; idx < m_portData.back().m_numModes; idx++) {
                        m_portData.back().m_portModeData.emplace_back(portModes[idx],
                                                                      wfdGetPortModeAttribi(device,
                                                                                            tempPort,
                                                                                            portModes[idx],
                                                                                            WFD_PORT_MODE_WIDTH),
                                                                      wfdGetPortModeAttribi(device,
                                                                                            tempPort,
                                                                                            portModes[idx],
                                                                                            WFD_PORT_MODE_HEIGHT),
                                                                      wfdGetPortModeAttribi(device,
                                                                                            tempPort,
                                                                                            portModes[idx],
                                                                                            WFD_PORT_MODE_REFRESH_RATE));
                    }
                    wfdDestroyPort(device, tempPort);
                    m_inited = true;
                } else {
                    CHECK_WFD_ERROR(device);
                    std::cerr << "Failed to create WFDPort with ID: " << portID << "\n";
                }
            }
        } else {
            CHECK_WFD_ERROR(device);
            std::cerr << "No WFDPorts found\n";
        }
    } else {
        std::cerr << "Failed to create WFDDevice\n";
    }

    if (device) {
        wfdDestroyDevice(device);
    }

    return;
}

void StaticWFDData::PrintWFDConfig() {
    std::cout << "Number of available Ports: " << m_numPorts << "\n";
    std::cout << "Port Config: " << "\n\n";
    for (int32_t idx = 0U; idx < m_numPorts; idx++) {
        m_portData[idx].PrintConfig();
        std::cout << "\n";
    }
}

void WFDPortData::PrintConfig() {
    std::cout << "\t" << "Port ID " << m_portID << "\n";
    if (m_numBindablePipes == 0U) {
        std::cout << "    No bindable Pipes.";
    } else {
        std::cout << "   Number of bindable Pipes: " << m_numBindablePipes << "\n";
        std::cout << "   Bindable Pipe IDs:";
        for (int32_t idx = 0U; idx < m_numBindablePipes; idx++) {
            std::cout << " " << m_bindablePipeIds[idx];
        }
    }
    std::cout << "\n   Supported Modes: " << "\n";
    for (int32_t idx = 0U; idx < m_numModes; idx++) {
        m_portModeData[idx].PrintConfig();
    }
}

bool WFDPortData::GetIdxIntoPortModeData(const uint32_t width, const uint32_t height,
                                         const uint32_t refreshRate, uint32_t &idxIntoPortModeData) {
    bool ret = false;

    for (uint32_t idx = 0U; idx < m_portModeData.size(); idx++) {
        if ((m_portModeData[idx].m_width == static_cast<int32_t>(width)) &&
            (m_portModeData[idx].m_height == static_cast<int32_t>(height)) &&
            (m_portModeData[idx].m_refreshRate == static_cast<int32_t>(refreshRate))) {
            idxIntoPortModeData = idx;
            ret = true;
            break;
        }
    }

    return ret;
}

void WFDPortModeData::PrintConfig() {
    std::cout << "    " << m_width << "x" << m_height << "@" << m_refreshRate << "\n";
}

NvSciBufResources::NvSciBufResources() : module(nullptr), unreconciledList(nullptr),
                                         conflictList(nullptr), reconciledList(nullptr),
                                         cpuPtr(nullptr), allocationSize(0U), pitch(nullptr),
                                         inited(true) {
    NvSciError sciErr = NvSciBufModuleOpen(&module);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to open NvSciBuf module" << std::endl;
        inited = false;
        return;
    }

    sciErr = NvSciBufAttrListCreate(module, &unreconciledList);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to create unreconciled attribute list" << std::endl;
        inited = false;
        return;
    }

    WFDErrorCode err = wfdNvSciBufSetDisplayAttributesNVX(&unreconciledList);
    if (err) {
        std::cerr << "Failed to set Display attributes for NvSciBufObj\n";
        inited = false;
        return;
    }
}

NvSciBufResources::~NvSciBufResources() {
    if (unreconciledList) {
        NvSciBufAttrListFree(unreconciledList);
    }

    if (conflictList) {
        NvSciBufAttrListFree(conflictList);
    }

    if (obj) {
        NvSciBufObjFree(obj);
    }

    if (module) {
        NvSciBufModuleClose(module);
    }
}

bool NvSciBufResources::InitializeRGB(uint32_t imageWidth, uint32_t imageHeight) {
    NvSciError err = NvSciError_Success;
    // Default buffer attributes
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValColorFmt bufColorFormat = NvSciColor_A8R8G8B8;
    NvSciBufAttrValColorStd bufColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;
    width = imageWidth;
    height = imageHeight;

    // This app performs CPU writes on the buffer so we'll be needing
    // CPU access permissions for the buffer
    bool needCpuAccessFlag = true;
    NvSciBufAttrKeyValuePair bufAttrs[] = {
        {
            NvSciBufGeneralAttrKey_RequiredPerm,
            &bufPerm,
            sizeof(bufPerm)
        },

        {
            NvSciBufGeneralAttrKey_Types,
            &bufType,
            sizeof(bufType)
        },
        {
            NvSciBufGeneralAttrKey_NeedCpuAccess,
            &needCpuAccessFlag,
            sizeof(needCpuAccessFlag)
        },
        {
            NvSciBufImageAttrKey_Layout,
            &layout,
            sizeof(layout)
        },
        {
            NvSciBufImageAttrKey_PlaneCount,
            &imagePlaneCount,
            sizeof(imagePlaneCount)
        },
        {
            NvSciBufImageAttrKey_PlaneColorFormat,
            &bufColorFormat,
            sizeof(bufColorFormat)
        },
        {
            NvSciBufImageAttrKey_PlaneColorStd,
            &bufColorStd,
            sizeof(bufColorStd)
        },
        {
            NvSciBufImageAttrKey_PlaneWidth,
            &imageWidth,
            sizeof(imageWidth),
        },
        {
            NvSciBufImageAttrKey_PlaneHeight,
            &imageHeight,
            sizeof(imageHeight),
        },
        {
            NvSciBufImageAttrKey_ScanType,
            &bufScanType,
            sizeof(bufScanType)
        },
    };

    err = NvSciBufAttrListSetAttrs(unreconciledList, bufAttrs,
                                   sizeof(bufAttrs)/sizeof(NvSciBufAttrKeyValuePair));
    if (err != NvSciError_Success) {
        std::cerr << "Failed to set attribute list to the specified attributes" << std::endl;
        return false;
    }

    err = NvSciBufAttrListReconcileAndObjAlloc(&unreconciledList, 1U, &obj, &conflictList);
    if (err != NvSciError_Success) {
        std::cerr << "Failed to allocate bufObj. Err : " << err << std::endl;
        return false;
    }

    err = NvSciBufObjGetCpuPtr(obj, (void **)&cpuPtr);
    if (err != NvSciError_Success) {
        std::cerr << "Failed to get CPU VA Ptr for the bufObj. Err : " << err << std::endl;
        return false;
    }

    err = NvSciBufObjGetAttrList(obj, &reconciledList);
    if (err != NvSciError_Success) {
        std::cerr << "Failed to get reconciled list" << std::endl;
        return false;
    }

    reconciledBufAttrs[0].key = NvSciBufImageAttrKey_Size;
    reconciledBufAttrs[1].key = NvSciBufImageAttrKey_PlanePitch;

    err = NvSciBufAttrListGetAttrs(reconciledList, &reconciledBufAttrs[0], 2);
    if (err != NvSciError_Success) {
        std::cerr << "Failed to get attribute list from NvSciBufObj" << std::endl;
        return false;
    }

    const uint64_t *size = reinterpret_cast<const uint64_t *>(reconciledBufAttrs[0].value);
    allocationSize = *size;

    // Get the pitch and perform an explicit alignment check
    // since NvSciBuf cannot guarantee the alignment as the app
    // does not pass any engine info.
    const uint64_t *tempPitch = reinterpret_cast<const uint64_t *>(reconciledBufAttrs[1].value);
    if (tempPitch == nullptr) {
        std::cerr << "Failed to query pitch\n";
        return false;
    }

    if (*tempPitch % pitchAlignment != 0U) {
        std::cerr << "Buffer size " << imageWidth << "x" << imageHeight << \
            " not allowed due to unsupported pitch alignment\n";
        return false;
    }

    pitch = const_cast<uint64_t *>(tempPitch);
    return true;
}

bool NvSciBufResources::FlushCpuCache(const uint64_t offset) {
    bool ret = true;

    NvSciError err = NvSciBufObjFlushCpuCacheRange(obj, offset, allocationSize);
    if (err != NvSciError_Success) {
        std::cerr << "CPU Cache flush failed\n";
        ret = false;
    }

    return ret;
}

void NvSciBufResources::FillARGB8888Buffer(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(cpuPtr);
    for (uint32_t idx = 0U; idx < width * height; idx++) {
        uint32_t data = 0U;
        data = (data | (static_cast<uint32_t>(a) << 24U));
        data = (data | (static_cast<uint32_t>(r) << 16U));
        data = (data | (static_cast<uint32_t>(g) << 8U));
        data = (data | (static_cast<uint32_t>(b)));
        *ptr++ = data;
    }
    static_cast<void>(FlushCpuCache());
}

NvSciSyncResources syncRes;
NvSciSyncResources syncRes_2;

static bool initializeNvSciSync() {
    NvSciError sciErr = NvSciError_Success;
    bool cpuWaiter = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitSignal;

    sciErr = NvSciSyncModuleOpen(&syncRes.module);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to open NvSciSync Module" << std::endl;
        return false;
    }

    sciErr = NvSciSyncAttrListCreate(syncRes.module, &syncRes.unreconciledList);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to create Signaler Attr List" << std::endl;
        return false;
    }

    WFDErrorCode err = wfdNvSciSyncSetSignalerAttributesNVX(&syncRes.unreconciledList);
    if (err) {
        std::cerr << "Failed to set Signaler attributes\n";
        return false;
    }

    NvSciSyncAttrKeyValuePair syncAttrs[] = {
        {
            NvSciSyncAttrKey_NeedCpuAccess,
            (void*)&cpuWaiter,
             sizeof(cpuWaiter)
        },
        {
            NvSciSyncAttrKey_RequiredPerm,
            (void*)&cpuPerm,
            sizeof(cpuPerm)
        },
    };

    sciErr = NvSciSyncAttrListSetAttrs(syncRes.unreconciledList, syncAttrs,
                                       sizeof(syncAttrs)/sizeof(NvSciSyncAttrKeyValuePair));
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to set Signaler list's Attributes" << std::endl;
        return false;
    }

    sciErr = NvSciSyncAttrListReconcile(&syncRes.unreconciledList, 1, &syncRes.reconciledList,
                                        &syncRes.conflictList);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to reconcile attrs" << std::endl;
        return false;
    }

    sciErr = NvSciSyncObjAlloc(syncRes.reconciledList, &syncRes.obj);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to create NvSciSyncObj" << std::endl;
        return false;
    }

    syncRes.inited = true;

    return true;
}

WFDResources::WFDResources(const SampleConfig &config,
                           NvSciBufObj *obj) : device(WFD_INVALID_HANDLE), numPorts(0), numPipes(0),
                                               numModes(0), portId(WFD_INVALID_PORT_ID),
                                               pipeId(WFD_INVALID_PIPELINE_ID),
                                               port(WFD_INVALID_HANDLE),
                                               pipe(WFD_INVALID_HANDLE),
                                               portMode(WFD_INVALID_HANDLE),
                                               source(WFD_INVALID_HANDLE),
                                               sourceRect{0, 0, static_cast<int32_t>(config.imageWidth),
                                                          static_cast<int32_t>(config.imageHeight)},
                                               bufObj(obj), inited(true) {
    device = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, nullptr);
    if (!device) {
        std::cerr << "Failed to create WFDDevice handle. Check slog2info for any errors." << std::endl;
        inited = false;
        return;
    }

    port = wfdCreatePort(device, StaticWFDData::GetInstance()->m_portData[config.portIdx].m_portID, nullptr);
    if (!port) {
        CHECK_WFD_ERROR(device);
        std::cerr << "Failed to create WFDPort for Port ID " << portId << std::endl;
        inited = false;
        return;
    }

    uint32_t portModeIdx = 0U;
    StaticWFDData::GetInstance()->m_portData[config.portIdx].GetIdxIntoPortModeData(config.modeWidth,
                                                                                    config.modeHeight,
                                                                                    config.modeRefreshRate,
                                                                                    portModeIdx);

    wfdSetPortMode(device, port,
                   StaticWFDData::GetInstance()->m_portData[config.portIdx].m_portModeData[portModeIdx].m_portMode);
    CHECK_WFD_ERROR(device);

    wfdDeviceCommit(device, WFD_COMMIT_ENTIRE_PORT, port);
    CHECK_WFD_ERROR(device);

    pipe = wfdCreatePipeline(device,
                             StaticWFDData::GetInstance()->m_portData[config.portIdx].m_bindablePipeIds[config.layerIdx],
                             nullptr);
    if (!pipe) {
        CHECK_WFD_ERROR(device);
        std::cerr << "Failed to create pipeline" << std::endl;
        inited = false;
        return;
    }

    wfdBindPipelineToPort(device, port, pipe);
    CHECK_WFD_ERROR(device);

    wfdDeviceCommit(device, WFD_COMMIT_ENTIRE_PORT, port);
    CHECK_WFD_ERROR(device);

    // Now we can create a WFDSource out of the allocated NvSciBufObj and bind it
    // to a WFDPipeline
    source = wfdCreateSourceFromNvSciBufNVX(device, pipe, bufObj);
    if (!source) {
        CHECK_WFD_ERROR(device);
        std::cerr << "Failed to create WFDSource" << std::endl;
        inited = false;
        return;
    }


    wfdBindSourceToPipeline(device, pipe, source, WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(device);

    wfdSetPipelineAttribiv(device, pipe, WFD_PIPELINE_SOURCE_RECTANGLE, 4, sourceRect);
    CHECK_WFD_ERROR(device);

    wfdSetPipelineAttribiv(device, pipe, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, sourceRect);
    CHECK_WFD_ERROR(device);

    inited = true;

}

static void printSampleInfo(bool printUsageOnly = false) {
    if (!printUsageOnly) {
        std::cout << "\nOpenWFD Sample Application\n\n";
        std::cout << "This sample application exercies various features of the OpenWFD driver:\n";
        std::cout << "\nPrerequisites:\n";
#if defined(__QNX__)
        std::cout << "If the app is being run on QNX Standard builds\n";
        std::cout << "ensure QNX Screen is NOT running before launching openwfd_nvsci_sample.\n";
#else
        std::cout << "nvidia.ko and nvidia-modeset.ko must be loaded before launching openwfd_nvsci_sample.\n";
        std::cout << "nvidia-drm.ko should not be loaded and no display clients (Xorg, GDM etc.) should be running.\n";
#endif
        std::cout << "\nSamples:\n";
        std::cout << "  1. NvSciBuf sanity sample         - A White square should appear on the display.\n";
        std::cout << "  2. NvSciBuf sanity sample         - A White square should fade in and out on the display.\n";
        std::cout << "  3. NvSciSync sanity sample        - A White square should appear on the display after 4 seconds.\n";
        std::cout << "  4. Non blocking commit sample     - A White square should appear on the display after 10 seconds.\n";
        std::cout << "  5. Alpha Blending sample          - This sample flips two buffers:\n";
        std::cout << "                                      1. Green colored RGBA buffer to the left portion of the display.\n";
        std::cout << "                                      2. Red colored RGBA buffer to the right portion of the display.\n";
        std::cout << "                                      The transparency attributes WFD_TRANSPARENCY_GLOBAL_ALPHA and\n";
        std::cout << "                                      WFD_TRANSPARENCY_SOURCE_ALPHA are set when flipping the buffers.\n";
    }
    std::cout << "\nUsage: \n";
    std::cout << "--help / -h           : Print information about wfd_nvsci_sample.\n";
    std::cout << "--sample / -s         : Specify the sample number to run.\n";
    std::cout << "                        Example - --sample 1\n";
    std::cout << "--res / -r            : Set size of buffer. (Default: ModeWidth x ModeHeight)\n";
    std::cout << "                        Example - --res 1920x1080\n";
    std::cout << "                        Note: Not all buffer sizes are supported.\n";
    std::cout << "--config / -c         : Print the display H/W configuration and exit.\n";
    std::cout << "--mode / -m           : Specify the display mode resolution to be used. (Default: Preferred mode as reported by Display)\n";
    std::cout << "                        Example - --mode 1920x1080@60\n";
    std::cout << "--display / -d        : Specify the display to run the sample on. (Default: 0)\n";
    std::cout << "--layer / -l          : Specify the layer to run the sample on. (Default: 0)\n";
    std::cout << "--safestate           : Run a sample which exercises Display's Safe State entry.\n";
    std::cout << "                        Requires QNX and a serializer connected to the target.\n";
    std::cout << "--dpmst               : Run a sample which flips to multiple displays.\n";
    std::cout << "                        Requires a DP-MST setup.\n";
    std::cout << "--frozenframedetection: Run a sample which exercies Display's Frozen Frame Detection feature.\n";
    std::cout << "                        Requires a setup with Frozen Frame Detection enabled.\n";
    std::cout << "--splash              : Displays a splash screen for a specified amount of time.\n";
    std::cout << "--timeout / -t        : Timeout for the splash screen in seconds. Valid only with --splash option.\n";
    std::cout << "                        Default: 60.\n";
    std::cout << "--image / -i          : Splash screen bmp image to be displayed.\n";
    std::cout << "                        Default: White color image.\n";
}

enum class ParseCommandLineArgIdx : uint32_t {
    VALID_ARGUMENTS = 0U,
    EARLY_EXIT,
    SAMPLE_TO_RUN,
    DISPLAY_CONFIG_ONLY,
    SAMPLE_CONFIG,
    DP_MST,
    SAFE_STATE,
    FROZEN_FRAME,
    SPLASH_SCREEN,
    SPLASH_TIMEOUT,
    SPLASH_IMAGE,
    LAST = SPLASH_IMAGE
};

static bool ValidateConfig(SampleConfig &config) {
    uint32_t portModeIdx = 0U;

    if (config.portIdx >= static_cast<uint32_t>(StaticWFDData::GetInstance()->m_numPorts)) {
        std::cerr << "Invalid Display ID specified: " << config.portIdx << "\n";
        return false;
    }

    if (config.layerIdx >= static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[config.portIdx].m_numBindablePipes)) {
        std::cerr << "Invalid Layer ID specified: " << config.layerIdx << "\n";
        return false;
    }

    if ((config.modeWidth == 0U) || (config.modeHeight == 0U) || (config.modeRefreshRate == 0U)) {
        config.modeWidth =
            StaticWFDData::GetInstance()->m_portData[config.portIdx].m_portModeData[0].m_width;
        config.modeHeight =
            StaticWFDData::GetInstance()->m_portData[config.portIdx].m_portModeData[0].m_height;
        config.modeRefreshRate =
            StaticWFDData::GetInstance()->m_portData[config.portIdx].m_portModeData[0].m_refreshRate;
    }

    if (!StaticWFDData::GetInstance()->m_portData[config.portIdx].GetIdxIntoPortModeData(config.modeWidth,
                                                                                         config.modeHeight,
                                                                                         config.modeRefreshRate,
                                                                                         portModeIdx)) {
        std::cerr << "Mode " << config.modeWidth << "x" << config.modeHeight << "@" << config.modeRefreshRate \
            << " not supported on Display ID: " << config.portIdx << "\n";
        return false;
    }

    if ((config.imageWidth == 0U) || (config.imageHeight == 0U)) {
        config.imageWidth = config.modeWidth;
        config.imageHeight = config.modeHeight;
    }

    if (config.imageWidth > config.modeWidth) {
        std::cerr << "Specified Image width: " << config.imageWidth \
            << " is larger than the Mode width: " << config.modeWidth << "\n";
        return false;
    }

    if (config.imageHeight > config.modeHeight) {
        std::cerr << "Specified Image height: " << config.imageHeight \
            << " is larger than the Mode height: " << config.modeHeight << "\n";
        return false;
    }

    return true;
}

using ParseCommandLineArgsReturnType = std::tuple<bool, bool, uint32_t, bool, SampleConfig,
                                                  bool, bool, bool, bool, uint32_t, char*>;

static auto ParseCommandLineArgs(int argc, char **argv) -> ParseCommandLineArgsReturnType {
    bool validArgs = true;
    bool earlyExit = false;
    uint32_t sampleToRun = 0U;
    bool displayConfigOnly = false;
    SampleConfig config = {};
    bool runDpMstSample = false;
    bool runSafeStateSample = false;
    bool runFrozenFrameDetectionSample = false;
    bool runSplashScreen = false;
    uint32_t timeoutInSeconds = 60U;
    char *splashImage = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printSampleInfo();
            earlyExit = true;
        } else if ((!strcmp(argv[i], "-s") || !strcmp(argv[i], "--sample"))
                   && i != argc - 1) {
            sampleToRun = strtol(argv[++i], nullptr, 10);
            if (sampleToRun > numSamples || sampleToRun < 1) {
                validArgs = false;
            }
        } else if ((!strcmp(argv[i], "-c") || !strcmp(argv[i], "--config"))) {
            displayConfigOnly = true;
        } else if ((!strcmp(argv[i], "-r") || !strcmp(argv[i], "--res"))
                   && i != argc - 1) {
            if ((sscanf(argv[++i], "%ux%u", &config.imageWidth, &config.imageHeight) != 2)) {
                validArgs = false;
            }
        } else if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--mode"))
                   && i != argc - 1) {
            if ((sscanf(argv[++i], "%ux%u@%u", &config.modeWidth, &config.modeHeight,
                        &config.modeRefreshRate) != 3)) {
                validArgs = false;
            }
        } else if ((!strcmp(argv[i], "-d") || !strcmp(argv[i], "--display"))
                   && i != argc - 1) {
            config.portIdx = strtol(argv[++i], nullptr, 10);
        } else if ((!strcmp(argv[i], "-l") || !strcmp(argv[i], "--layer"))
                   && i != argc - 1) {
            config.layerIdx = strtol(argv[++i], nullptr, 10);
        } else if (!strcmp(argv[i], "--dpmst")) {
            runDpMstSample = true;
        } else if (!strcmp(argv[i], "--safestate")) {
            runSafeStateSample = true;
        } else if (!strcmp(argv[i], "--frozenframedetection")) {
            runFrozenFrameDetectionSample = true;
        } else if (!strcmp(argv[i], "--splash")) {
            runSplashScreen = true;
        } else if ((!strcmp(argv[i], "-t") || !strcmp(argv[i], "--timeout"))
                   && i != argc - 1) {
            timeoutInSeconds = strtol(argv[++i], nullptr, 10);
        } else if ((!strcmp(argv[i], "-i") || !strcmp(argv[i], "--image"))
                   && i != argc - 1) {
            splashImage = argv[++i];
        } else {
            validArgs = false;
        }
    }

    if (validArgs) {
        validArgs = ValidateConfig(config);
    }

    return std::make_tuple(validArgs, earlyExit, sampleToRun, displayConfigOnly,
                           config, runDpMstSample, runSafeStateSample,
                           runFrozenFrameDetectionSample, runSplashScreen,
                           timeoutInSeconds, splashImage);
}

static bool Sample1(SampleConfig config) {
    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    bufRes.FillARGB8888Buffer(255U, 255U, 255U, 255U);

    for (uint32_t i = 0U; i < numFrames; i++) {
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    return true;
}

static bool Sample2(SampleConfig config) {
    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    uint8_t pixelValue = 0U;

    for (uint32_t i = 0U; i < numFrames * 5; i++) {
        ++pixelValue;
        bufRes.FillARGB8888Buffer(255U, pixelValue, pixelValue, pixelValue);
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_PIPELINE, wfdRes.pipe);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    return true;
}

static bool Sample3(SampleConfig config) {
    auto signaler = [](void)
    {
        uint32_t sleepInSeconds = 4U;
        std::this_thread::sleep_for(std::chrono::seconds(sleepInSeconds));
        NvSciSyncObjSignal(syncRes.obj);
        std::cout << "Fence has been signaled. A white square should be visible on the display now" << std::endl;
    };

    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    if (!syncRes.inited && !initializeNvSciSync()) {
        std::cout << "Failed to initialize NvSciSync" << std::endl;
        return false;
    }

    WFDErrorCode err = WFD_ERROR_NONE;
    NvSciError sciErr = NvSciError_Success;

    sciErr = NvSciSyncObjGenerateFence(syncRes.obj, &syncRes.fence);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to get Fence obj" << std::endl;
        return false;
    }

    err = wfdBindNvSciSyncFenceToSourceNVX(wfdRes.device, wfdRes.source, &syncRes.fence);
    if (err != WFD_ERROR_NONE) {
        CHECK_WFD_ERROR(wfdRes.device);
        std::cerr << "Failed to bind Fence to WFDSource" << std::endl;
        return false;
    }

    bufRes.FillARGB8888Buffer(255U, 255U, 255U, 255U);

    std::thread signalThread(signaler);

    std::cout << "Scheduling flips on the main thread" << std::endl;
    for (uint32_t i = 0U; i < numFrames; i++) {
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_PIPELINE, wfdRes.pipe);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    signalThread.join();
    return true;
}

static bool Sample4(SampleConfig config) {
    auto signaler = [](void)
    {
        uint32_t sleepInSeconds = 10U;
        std::this_thread::sleep_for(std::chrono::seconds(sleepInSeconds));
        NvSciSyncObjSignal(syncRes.obj);
        std::cout << "Fence has been signaled. A white square should be visible on the display now" << std::endl;
    };

    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    if (!syncRes.inited && !initializeNvSciSync()) {
        std::cout << "Failed to initialize NvSciSync" << std::endl;
        return false;
    }

    NvSciError sciErr = NvSciError_Success;
    WFDErrorCode err;
    bool cpuWaiter = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitSignal;
    NvSciSyncCpuWaitContext cpuWaitContext;

    sciErr = NvSciSyncCpuWaitContextAlloc(syncRes.module, &cpuWaitContext);
    if (sciErr != NvSciError_Success) {
        std::cout << "Failed to create cpu wait context\n";
        return false;
    }

    sciErr = NvSciSyncAttrListCreate(syncRes.module, &syncRes_2.unreconciledList);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to create Signaler Attr List" << std::endl;
        return false;
    }

    err = wfdNvSciSyncSetWaiterAttributesNVX(&syncRes_2.unreconciledList);
    if (err) {
        std::cerr << "Failed to set Waiter attributes\n";
        return false;
    }

    NvSciSyncAttrKeyValuePair syncAttrs[] = {
        {
            NvSciSyncAttrKey_NeedCpuAccess,
            (void*)&cpuWaiter,
             sizeof(cpuWaiter)
        },
        {
            NvSciSyncAttrKey_RequiredPerm,
            (void*)&cpuPerm,
            sizeof(cpuPerm)
        },
    };

    sciErr = NvSciSyncAttrListSetAttrs(syncRes_2.unreconciledList, syncAttrs,
                                       sizeof(syncAttrs)/sizeof(NvSciSyncAttrKeyValuePair));
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to set Signaler list's Attributes" << std::endl;
        return false;
    }

    sciErr = NvSciSyncAttrListReconcile(&syncRes_2.unreconciledList, 1, &syncRes_2.reconciledList,
                                        &syncRes_2.conflictList);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to reconcile attrs" << std::endl;
        return false;
    }

    sciErr = NvSciSyncObjAlloc(syncRes_2.reconciledList, &syncRes_2.obj);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to create NvSciSyncObj" << std::endl;
        return false;
    }

    sciErr = NvSciSyncObjGenerateFence(syncRes.obj, &syncRes.fence);
    if (sciErr != NvSciError_Success) {
        std::cerr << "Failed to get Fence obj" << std::endl;
        return false;
    }

    WFDint nb = wfdGetPipelineAttribi(wfdRes.device, wfdRes.pipe, static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX));
    if (nb == WFD_FALSE) {
        std::cout << "Non blocking commit disabled. Enabling it\n";
        wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_TRUE);
    }

    err = wfdRegisterPostFlipNvSciSyncObjNVX(wfdRes.device, &syncRes_2.obj);
    if (err != WFD_ERROR_NONE) {
        std::cout << "Failed to register NvSciSyncObj\n";
        return false;
    }

    err = wfdBindNvSciSyncFenceToSourceNVX(wfdRes.device, wfdRes.source, &syncRes.fence);
    if (err != WFD_ERROR_NONE) {
        CHECK_WFD_ERROR(wfdRes.device);
        std::cerr << "Failed to bind Fence to WFDSource" << std::endl;
        return false;
    }

    bufRes.FillARGB8888Buffer(255, 255, 255, 255);

    std::thread signalThread(signaler);

    std::cout << "Scheduling " << numFrames * 2 << " flips on the main thread" << std::endl;

    for (uint32_t i = 0U; i < numFrames * 2; i++) {
        NvSciSyncFence postFlipFence = NvSciSyncFenceInitializer;

        wfdDeviceCommitWithNvSciSyncFenceNVX(wfdRes.device, WFD_COMMIT_PIPELINE, wfdRes.pipe, &postFlipFence);
        CHECK_WFD_ERROR(wfdRes.device);

        sciErr = NvSciSyncFenceWait(&postFlipFence, cpuWaitContext, -1);
        if (sciErr != NvSciError_Success) {
            std::cout << "Fence wait failed on post flip fence\n";
            NvSciSyncFenceClear(&postFlipFence);
            return -1;
        }
        NvSciSyncFenceClear(&postFlipFence);
    }

    std::cout << "Post flip fence signalled. Display is done with its work\n";

    signalThread.join();

    //Set the default value of WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX
    wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_FALSE);
    CHECK_WFD_ERROR(wfdRes.device);

    NvSciSyncCpuWaitContextFree(cpuWaitContext);
    return true;
}

static bool DPMSTSample() {
    // Initialize the first buffer resource. This gets
    // used on the first WFDPort.
    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_width,
                              StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_height)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    // Clear the first buffer resource to green
    bufRes.FillARGB8888Buffer(255, 0, 255, 0);

    // Create the first WFDPort and its dependent resources via WFDResources
    SampleConfig config = {
        0U /* portIdx */,
        0U /* layerIdx */,
        static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_width) /* modeWidth */,
        static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_height) /* modeHeight */,
        static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_refreshRate) /* modeRefreshRate */,
        static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_width) /* imageWidth */,
        static_cast<uint32_t>(StaticWFDData::GetInstance()->m_portData[0].m_portModeData[0].m_height) /* imageHeight */
    };
    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    // Explicitly create the second WFDPort.
    WFDPort secondPort = wfdCreatePort(wfdRes.device,
                                       StaticWFDData::GetInstance()->m_portData[1].m_portID,
                                       nullptr);
    if (!secondPort) {
        std::cerr << "Failed to create second WFDPort handle\n";
        CHECK_WFD_ERROR(wfdRes.device);
        return false;
    }

    // Perform a modeset on the second WFDPort.
    // We use the first WFDPortMode of the WFDPort returned by OpenWFD
    // by default as this would be the preferred mode
    // as reported by the attached Display.
    wfdSetPortMode(wfdRes.device, secondPort,
                   StaticWFDData::GetInstance()->m_portData[1].m_portModeData[0].m_portMode);
    CHECK_WFD_ERROR(wfdRes.device);

    // Initialize the second buffer resource.
    // This gets used on the second WFDPort.
    NvSciBufResources secondBufRes;

    if (!secondBufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!secondBufRes.InitializeRGB(StaticWFDData::GetInstance()->m_portData[1].m_portModeData[0].m_width,
                                    StaticWFDData::GetInstance()->m_portData[1].m_portModeData[0].m_height)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    // Clear the second buffer resource to blue
    secondBufRes.FillARGB8888Buffer(255, 0, 0, 255);

    // Create a WFDPipeline handle for the first layer on the second WFDPort.
    WFDPipeline secondPipe =
        wfdCreatePipeline(wfdRes.device,
                          StaticWFDData::GetInstance()->m_portData[1].m_bindablePipeIds[0],
                          nullptr);
    if (!secondPipe) {
        CHECK_WFD_ERROR(wfdRes.device);
        std::cerr << "Failed to create second WFDPipeline handle\n";
        return false;
    }

    wfdBindPipelineToPort(wfdRes.device, secondPort, secondPipe);
    CHECK_WFD_ERROR(wfdRes.device);

    WFDSource secondSource = wfdCreateSourceFromNvSciBufNVX(wfdRes.device,
                                                            secondPipe,
                                                            &secondBufRes.obj);
    if (!secondSource) {
        CHECK_WFD_ERROR(wfdRes.device);
        std::cerr << "Failed to create second WFDSource handle\n";
        return false;
    }

    wfdBindSourceToPipeline(wfdRes.device, secondPipe, secondSource,
                            WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);

    WFDint secondSourceRect[4] = {
        0 /* x offset */,
        0 /* y offset */,
        static_cast<int32_t>(StaticWFDData::GetInstance()->m_portData[1].m_portModeData[0].m_width) /* width */,
        static_cast<int32_t>(StaticWFDData::GetInstance()->m_portData[1].m_portModeData[0].m_height) /* height */
    };

    wfdSetPipelineAttribiv(wfdRes.device, secondPipe, WFD_PIPELINE_SOURCE_RECTANGLE,
                           4, secondSourceRect);
    CHECK_WFD_ERROR(wfdRes.device);

    wfdSetPipelineAttribiv(wfdRes.device, secondPipe, WFD_PIPELINE_DESTINATION_RECTANGLE,
                           4, secondSourceRect);
    CHECK_WFD_ERROR(wfdRes.device);

    for (uint32_t i = 0U; i < numFrames * 3U; i++) {
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
        CHECK_WFD_ERROR(wfdRes.device);
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, secondPort);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    // Perform a null flip.
    wfdBindSourceToPipeline(wfdRes.device, wfdRes.pipe, static_cast<WFDSource>(0),
                            WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdBindSourceToPipeline(wfdRes.device, secondPipe, static_cast<WFDSource>(0),
                            WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);

    wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, secondPort);
    CHECK_WFD_ERROR(wfdRes.device);

    // Destroy the resources created as part of this sample
    wfdDestroySource(wfdRes.device, secondSource);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdDestroyPipeline(wfdRes.device, secondPipe);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdDestroyPort(wfdRes.device, secondPort);
    CHECK_WFD_ERROR(wfdRes.device);

    return true;
}

static bool Sample5(SampleConfig config) {
    NvSciBufResources bufRes;
    NvSciBufResources bufRes_2;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    if (!bufRes_2.inited) {
        std::cout << "Failed to initialize NvSciBuf 2\n";
        return false;
    }

    if (!bufRes_2.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cout << "Failed to create a RGB NvSciBufObj 2\n";
        return false;
    }

    // Override the user specified layerIdx to 0 since this sample uses two overlays
    // and if the user specified layerIdx was a value other than 0, we try to create the same WFDPipeline handle
    // twice.
    config.layerIdx = 0U;
    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    if (StaticWFDData::GetInstance()->m_portData[config.portIdx].m_numBindablePipes < 2) {
        std::cerr << "Specified Display: " << config.portIdx << " has less than two pipes.\n";
        std::cerr << "Blending sample requires atleast two pipes. Bailing\n";
        return false;
    }

    WFDPipeline newPipe = wfdCreatePipeline(wfdRes.device, StaticWFDData::GetInstance()->m_portData[config.portIdx].m_bindablePipeIds[1], nullptr);
    CHECK_WFD_ERROR(wfdRes.device);
    if (!newPipe) {
        std::cerr << "Failed to create pipe\n";
       return false;
    }

    wfdBindPipelineToPort(wfdRes.device, wfdRes.port, newPipe);
    CHECK_WFD_ERROR(wfdRes.device);
    WFDSource newSource = wfdCreateSourceFromNvSciBufNVX(wfdRes.device, newPipe, &bufRes_2.obj);
    CHECK_WFD_ERROR(wfdRes.device);
    if (!newSource) {
        std::cerr << "Failed to create new WFDSource object\n";
        return false;
    }

    // Position the rects so that portions of the buffer overlap.
    WFDint sourceWidth = static_cast<WFDint>(config.imageWidth) / 2;
    WFDint newSourceOffset = sourceWidth / 2;
    WFDint sourceRect[4] = {0, 0, sourceWidth, static_cast<WFDint>(config.imageHeight)};
    WFDint newSourceRect[4] = {newSourceOffset,0, sourceWidth, static_cast<WFDint>(config.imageHeight)};
    wfdBindSourceToPipeline(wfdRes.device, newPipe, newSource, WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribiv(wfdRes.device, newPipe, WFD_PIPELINE_SOURCE_RECTANGLE, 4, sourceRect);
    CHECK_WFD_ERROR(wfdRes.device);

    wfdSetPipelineAttribiv(wfdRes.device, newPipe, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, sourceRect);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribiv(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_SOURCE_RECTANGLE, 4, newSourceRect);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribiv(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, newSourceRect);
    CHECK_WFD_ERROR(wfdRes.device);

    bufRes.FillARGB8888Buffer(255, 255, 0, 0);
    bufRes_2.FillARGB8888Buffer(255, 0, 255, 0);

    std::cout << "Setting WFD_TRANSPARENCY_SOURCE_ALPHA\n";
    wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_TRANSPARENCY_ENABLE, WFD_TRANSPARENCY_SOURCE_ALPHA);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribi(wfdRes.device, newPipe, WFD_PIPELINE_TRANSPARENCY_ENABLE, WFD_TRANSPARENCY_SOURCE_ALPHA);
    CHECK_WFD_ERROR(wfdRes.device);
    uint8_t alpha = 0U;
    for (uint32_t i = 0U; i < numFrames; i++) {
        bufRes_2.FillARGB8888Buffer(alpha++, 0, 255, 0);
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    std::cout << "Setting WFD_TRANSPARENCY_SOURCE_ALPHA | WFD_TRANSPARENCY_GLOBAL_ALPHA\n";
    wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_TRANSPARENCY_ENABLE, WFD_TRANSPARENCY_SOURCE_ALPHA | WFD_TRANSPARENCY_GLOBAL_ALPHA);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribi(wfdRes.device, newPipe, WFD_PIPELINE_TRANSPARENCY_ENABLE, WFD_TRANSPARENCY_SOURCE_ALPHA | WFD_TRANSPARENCY_GLOBAL_ALPHA);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_GLOBAL_ALPHA, 0);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdSetPipelineAttribi(wfdRes.device, newPipe, WFD_PIPELINE_GLOBAL_ALPHA, 255);
    CHECK_WFD_ERROR(wfdRes.device);
    bufRes.FillARGB8888Buffer(128, 255, 0, 0);
    bufRes_2.FillARGB8888Buffer(128, 0, 255, 0);
    uint8_t globalAlpha = 0U;
    for (uint32_t i = 0U; i < numFrames; i++) {
        wfdSetPipelineAttribi(wfdRes.device, newPipe, WFD_PIPELINE_GLOBAL_ALPHA, globalAlpha);
        CHECK_WFD_ERROR(wfdRes.device);
        wfdSetPipelineAttribi(wfdRes.device, wfdRes.pipe, WFD_PIPELINE_GLOBAL_ALPHA, globalAlpha++);
        CHECK_WFD_ERROR(wfdRes.device);
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    // Perform a null flip on both the pipes.
    wfdBindSourceToPipeline(wfdRes.device, newPipe, static_cast<WFDSource>(0), WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdBindSourceToPipeline(wfdRes.device, wfdRes.pipe, static_cast<WFDSource>(0), WFD_TRANSITION_AT_VSYNC, nullptr);
    CHECK_WFD_ERROR(wfdRes.device);
    wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
    CHECK_WFD_ERROR(wfdRes.device);

    // Destroy resources created as part of this sample.
    wfdDestroySource(wfdRes.device, newSource);
    CHECK_WFD_ERROR(wfdRes.device);

    wfdDestroyPipeline(wfdRes.device, newPipe);
    CHECK_WFD_ERROR(wfdRes.device);

    return true;
}

static bool FrozenFrameDetectionSample(SampleConfig config) {
#if defined(__QNX__)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0U, 255U);

    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cerr << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cerr << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    // Enable frozen frame detection
    std::cout << "Enabling frozen frame detection\n";
    wfdSetPortAttribi(wfdRes.device, wfdRes.port, static_cast<WFDPortConfigAttrib>(WFD_PORT_DETECT_FROZEN_FRAME_NV), WFD_TRUE);
    CHECK_WFD_ERROR(wfdRes.device);

    std::cout << "Flipping buffers with alternating colors\n";
    for (uint32_t i = 0U; i < numFrames; i++) {
        bufRes.FillARGB8888Buffer(255, dist(gen), dist(gen), dist(gen));
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_PIPELINE, wfdRes.pipe);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    std::cout << "Flipping buffers with a static white. Display should start detecting frozen frames now\n";
    bufRes.FillARGB8888Buffer(255, 255, 255, 255);
    for (uint32_t i = 0U; i < numFrames; i++) {
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_PIPELINE, wfdRes.pipe);
        CHECK_WFD_ERROR(wfdRes.device);
    }
#else
    std::cout << "Frozen frame detection is not supported on non-QNX platforms. Skipping sample\n";
#endif

    return true;
}

static bool SafeStateSample(SampleConfig config) {
#if defined(__QNX__)
    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cerr << "Failed to initialize NvSciBuf" << std::endl;
        return false;
    }

    if (!bufRes.InitializeRGB(config.imageWidth, config.imageHeight)) {
        std::cerr << "Failed to create a RGB NvSciBufObj" << std::endl;
        return false;
    }

    WFDResources wfdRes(config, &bufRes.obj);
    if (!wfdRes.inited) {
        std::cerr << "Failed to initialize WFD\n";
        return false;
    }

    bufRes.FillARGB8888Buffer(255, 255, 255, 255);

    std::cout << "Flipping buffers cleared to white to display\n";

    for (uint32_t i = 0U; i < numFrames; i++) {
        wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
        CHECK_WFD_ERROR(wfdRes.device);
    }

    std::cout << "Setting display to safe state\n";

    auto err = wfdDeviceSetSafeStateNV(wfdRes.device);

    if (err == WFD_ERROR_NONE) {
        std::cout << "Display should be blank now.\n";
        for (uint32_t i = 0U; i < numFrames; i++) {
            wfdDeviceCommit(wfdRes.device, WFD_COMMIT_ENTIRE_PORT, wfdRes.port);
            CHECK_WFD_ERROR(wfdRes.device);
        }
    } else {
        std::cout << "Unable to transition display to safe state\n";
        std::cout << "Display safe state transition is supported only on QNX with Serializer\n";
        // Do not fail here as it could be possible that the sample is running on a setup without serializer
        // connected.
    }
#else
    std::cout << "Display safe state is not supported on non-QNX platforms. Skipping sample\n";
#endif

    return true;
}

static bool SplashProgram(uint32_t timeoutInSeconds, const char *imagePath)
{
    std::mutex m;
    std::condition_variable cv;
    bool status = true;
    std::vector<uint32_t> portIdxToUse;
    auto timer = [&](void)
    {
        std::this_thread::sleep_for(std::chrono::seconds(timeoutInSeconds));
        cv.notify_one();
    };

    std::thread timeoutThread(timer);

    std::vector<WFDPort> portHandles;
    std::vector<WFDPipeline> pipeHandles;
    std::vector<WFDSource> sourceHandles;
    WFDint sourceRect[4];
    WFDDevice device = WFD_INVALID_HANDLE;

    NvSciBufResources bufRes;

    if (!bufRes.inited) {
        std::cout << "Failed to initialize NvSciBuf" << std::endl;
        status = false;
        goto fail;
    }

    device = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, nullptr);
    if (device == WFD_INVALID_HANDLE) {
        std::cerr << "Failed to create WFDDevice handle\n";
        status = false;
        goto fail;
    }

    for (int32_t idx = 0; idx < StaticWFDData::GetInstance()->m_numPorts; idx++) {
        if (StaticWFDData::GetInstance()->m_portData[idx].m_numBindablePipes > 0) {
            portIdxToUse.push_back(idx);
        }
    }

    if (!bufRes.InitializeRGB(StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_width,
                              StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_height)) {
        std::cerr << "Failed to initialize RGB buffer\n";
        status = false;
        goto fail;
    }

    if (!imagePath) {
        std::memset(bufRes.cpuPtr, 255, bufRes.allocationSize);
    } else {
        auto ret = DrawBmp32(imagePath, 0, bufRes.cpuPtr, StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_width,
                             StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_height, 32, *bufRes.pitch);
        if (ret == -1) {
            std::cerr << "Failed to decode bmp image. Falling back to using a white image for splash screen\n";
            std::memset(bufRes.cpuPtr, 255, bufRes.allocationSize);
        }
    }
    sourceRect[0] = sourceRect[1] = 0;
    sourceRect[2] = StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_width;
    sourceRect[3] = StaticWFDData::GetInstance()->m_portData[portIdxToUse[0]].m_portModeData[0].m_height;

    for (auto portIdx : portIdxToUse) {
        auto portHandle = wfdCreatePort(device, StaticWFDData::GetInstance()->m_portData[portIdx].m_portID, nullptr);
        if (portHandle != WFD_INVALID_HANDLE) {
            portHandles.push_back(portHandle);
        } else {
            CHECK_WFD_ERROR(device);
            status = false;
            std::cerr << "Failed to create WFDPort handle with ID : " << StaticWFDData::GetInstance()->m_portData[portIdx].m_portID << "\n";
            goto fail;
        }

        auto pipeHandle = wfdCreatePipeline(device, StaticWFDData::GetInstance()->m_portData[portIdx].m_bindablePipeIds[0], nullptr);
        if (pipeHandle != WFD_INVALID_HANDLE) {
            pipeHandles.emplace_back(pipeHandle);
        } else {
            CHECK_WFD_ERROR(device);
            status = false;
            std::cerr << "Failed to create WFDPipeline handle with ID : " << StaticWFDData::GetInstance()->m_portData[portIdx].m_bindablePipeIds[0] << "\n";
            goto fail;
        }

        wfdBindPipelineToPort(device, portHandle, pipeHandle);
        CHECK_WFD_ERROR(device);

        auto source = wfdCreateSourceFromNvSciBufNVX(device, pipeHandle, &bufRes.obj);
        if (!source) {
            CHECK_WFD_ERROR(device);
            std::cerr << "Failed to create WFDSource handle\n";
            status = false;
            goto fail;
        } else {
            sourceHandles.emplace_back(source);
        }

        wfdBindSourceToPipeline(device, pipeHandle, source, WFD_TRANSITION_AT_VSYNC, nullptr);
        CHECK_WFD_ERROR(device);
        wfdSetPipelineAttribiv(device, pipeHandle, WFD_PIPELINE_SOURCE_RECTANGLE, 4, sourceRect);
        CHECK_WFD_ERROR(device);
        wfdSetPipelineAttribiv(device, pipeHandle, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, sourceRect);
        CHECK_WFD_ERROR(device);

        wfdDeviceCommit(device, WFD_COMMIT_ENTIRE_PORT, portHandle);
        CHECK_WFD_ERROR(device);
    }

    {
        std::unique_lock<std::mutex> scopeLock(m);
        cv.wait(scopeLock);
    }

fail:
    timeoutThread.join();

    for (auto pipe : pipeHandles) {
        if (pipe != WFD_INVALID_HANDLE) {
            wfdBindSourceToPipeline(device, pipe, static_cast<WFDSource>(0), WFD_TRANSITION_AT_VSYNC, nullptr);
            CHECK_WFD_ERROR(device);
        }
    }

    for (auto port : portHandles) {
        if (port != WFD_INVALID_HANDLE) {
            wfdDeviceCommit(device, WFD_COMMIT_ENTIRE_PORT, port);
            CHECK_WFD_ERROR(device);
        }
    }

    for (auto source : sourceHandles) {
        if (source != WFD_INVALID_HANDLE) {
            wfdDestroySource(device, source);
            CHECK_WFD_ERROR(device);
        }
    }

    if (device != WFD_INVALID_HANDLE) {
        wfdDestroyDevice(device);
    }

    return status;

}

static bool GetDisplayConfiguration() {
    auto instance = StaticWFDData::GetInstance();
    return instance->m_inited;
}

int main(int argc, char **argv) {
    if (!GetDisplayConfiguration()) {
        std::cout << "Failed to initialize OpenWFD\n";
        return -1;
    }

    auto parseResult = ParseCommandLineArgs(argc, argv);
    if (!std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::VALID_ARGUMENTS)>(parseResult)) {
        std::cerr << "\nInvalid arguments specified.\n";
        printSampleInfo(true);
        return -1;
    }

    if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::EARLY_EXIT)>(parseResult)) {
        return 0;
    }

    if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::DISPLAY_CONFIG_ONLY)>(parseResult)) {
        StaticWFDData::GetInstance()->PrintWFDConfig();
        return 0;
    }

    if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_TO_RUN)>(parseResult)) {
        std::cout << "Running sample : " << std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_TO_RUN)>(parseResult) << std::endl;
        bool rv = sampleFunctionList[std::get<2>(parseResult) - 1](std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_CONFIG)>(parseResult));
        if (!rv) {
            std::cout << "Sample " << std::get<2>(parseResult) << " Failed!";
            return -1;
        }
    } else {
        if ((!std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::DP_MST)>(parseResult)) &&
            (!std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAFE_STATE)>(parseResult)) &&
            (!std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::FROZEN_FRAME)>(parseResult)) &&
            (!std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SPLASH_SCREEN)>(parseResult))) {
            for (uint32_t i = 0; i < numSamples; i++) {
                std::cout << "Running sample : " << i+1 << std::endl;
                bool rv = sampleFunctionList[i](std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_CONFIG)>(parseResult));
                if (!rv) {
                    std::cout << "Sample " << i+1 << " Failed!";
                    return -1;
                }
            }
        } else {
            if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::DP_MST)>(parseResult)) {
                std::cout << "Running DP-MST sample\n";
                bool rv = DPMSTSample();
                if (!rv) {
                    std::cout << "DP-MST sample failed\n";
                    return -1;
                }
            }
            if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAFE_STATE)>(parseResult)) {
                std::cout << "Running Safe State sample\n";
                bool rv = SafeStateSample(std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_CONFIG)>(parseResult));
                if (!rv) {
                    std::cout << "SafeState sample failed\n";
                    return -1;
                }
            }
            if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::FROZEN_FRAME)>(parseResult)) {
                bool rv = FrozenFrameDetectionSample(std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SAMPLE_CONFIG)>(parseResult));
                if (!rv) {
                    std::cout << "Frozen frame detection sample failed\n";
                    return -1;
                }
            }
            if (std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SPLASH_SCREEN)>(parseResult)) {
                bool rv = SplashProgram(std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SPLASH_TIMEOUT)>(parseResult),
                                        std::get<static_cast<uint32_t>(ParseCommandLineArgIdx::SPLASH_IMAGE)>(parseResult));
                if (!rv) {
                    std::cerr << "Splash screen failed\n";
                    return -1;
                }
            }
        }
    }
}

