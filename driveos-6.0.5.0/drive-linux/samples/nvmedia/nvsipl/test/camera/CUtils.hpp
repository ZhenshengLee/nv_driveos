/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "nvmedia_core.h"
#include "nvscierror.h"
#include "nvscibuf.h"

#include <iostream>
#include <memory>
#include <cstdarg>
#include <vector>

#include "NvSIPLCommon.hpp"
#include "NvSIPLCapStructs.h"
#include "nvscibuf.h"
#include "nvscisync.h"

using namespace nvsipl;

#ifndef CUTILS_HPP
#define CUTILS_HPP

static const uint32_t MAX_SUPPORTED_DISPLAYS = 2U;
static const uint32_t NUM_CAPTURE_BUFFERS_PER_POOL = 6U;
static const uint32_t NUM_ISP_BUFFERS_PER_POOL = 4U;

struct CloseNvSciBufAttrList {
    void operator()(NvSciBufAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciBufAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciBufObj {
    void operator()(NvSciBufObj *bufObj) const {
        if (bufObj != nullptr) {
            if ((*bufObj) != nullptr) {
                NvSciBufObjFree(*bufObj);
            }
            delete bufObj;
        }
    }
};

struct CloseNvSciSyncAttrList {
    void operator()(NvSciSyncAttrList *attrList) const {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciSyncAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncObj {
    void operator()(NvSciSyncObj *syncObj) const {
        if (syncObj != nullptr) {
            if ((*syncObj) != nullptr) {
                NvSciSyncObjFree(*syncObj);
            }
            delete syncObj;
        }
    }
};

// Interface class for a single display interface
class IDisplayInterface
{
public:
    virtual SIPLStatus Init(uint32_t &uWidth,
                            uint32_t &uHeight,
                            NvSciBufModule &bufModule,
                            NvSciBufAttrList &bufAtrList) = 0;
    virtual NvSciBufObj & GetBuffer() = 0;
    virtual SIPLStatus WfdFlip() = 0;
    virtual ~IDisplayInterface() = default;
};

// Interface class for a display manager
class IDisplayManager
{
public:
    virtual SIPLStatus Init(uint32_t uNumDisplays) = 0;
    virtual SIPLStatus GetDisplayInterface(uint32_t uDispId, IDisplayInterface * &pDispIf) = 0;
    virtual ~IDisplayManager() = default;
};

/** Helper MACROS */
#define CHK_PTR_AND_RETURN(ptr, api) \
    if ((ptr) == nullptr) { \
        LOG_ERR("%s failed\n", (api)); \
        return NVSIPL_STATUS_OUT_OF_MEMORY; \
    }

#define CHK_PTR_AND_RETURN_BADARG(ptr, name) \
    if ((ptr) == nullptr) { \
        LOG_ERR("%s is null\n", (name)); \
        return NVSIPL_STATUS_BAD_ARGUMENT; \
    }

#define CHK_PTR_QUIT_AND_RETURN(ptr, quit, api) \
    if ((ptr) == nullptr) { \
        LOG_ERR("%s failed\n", (api)); \
        *(quit) = true; \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_STATUS_AND_RETURN(status, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        LOG_ERR("%s failed. status: %u\n", (api), (status)); \
        return (status); \
    }

#define CHK_STATUS_QUIT_AND_RETURN(status, quit, api) \
    if ((status) != NVSIPL_STATUS_OK) { \
        LOG_ERR("%s failed. status: %u\n", (api), (status)); \
        *(quit) = true; \
        return (status); \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        LOG_ERR("%s failed. nvmStatus: %u\n", (api), (nvmStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_NVMSTATUS_AND_EXIT(nvmStatus, api) \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) { \
        LOG_ERR("%s failed. nvmStatus: %u\n", (api), (nvmStatus)); \
        return; \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api) \
    if ((nvSciStatus) != NvSciError_Success) { \
        LOG_ERR("%s failed. nvSciStatus: %u\n", (api), (nvSciStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_NVSCISTATUS_QUIT_AND_RETURN(nvSciStatus, quit, api) \
    if ((nvSciStatus) != NvSciError_Success) { \
        LOG_ERR("%s failed. nvSciStatus: %u\n", (api), (nvSciStatus)); \
        *(quit) = true; \
        return NVSIPL_STATUS_ERROR; \
    }

#define CHK_WFDSTATUS_AND_RETURN(wfdStatus, api) \
    if (wfdStatus) { \
        LOG_ERR("%s failed, status: %u\n", (api), (wfdStatus)); \
        return NVSIPL_STATUS_ERROR; \
    }

#define GET_WFDERROR_AND_RETURN(device) \
    { \
        WFDErrorCode wfdErr = wfdGetError(device); \
        if (wfdErr) { \
            LOG_ERR("WFD error %x, line: %u\n", wfdErr, __LINE__); \
            return NVSIPL_STATUS_ERROR; \
        } \
    }

#define LINE_INFO __FUNCTION__, __LINE__

//! Quick-log a message at debugging level
#define LOG_DBG(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at preset level
#define LOG_MSG(...) \
    CLogger::GetInstance().LogMessage(__VA_ARGS__)

#define LEVEL_NONE CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG CLogger::LogLevel::LEVEL_DEBUG

//! \brief Logger utility class
//! This is a singleton class - at most one instance can exist at all times.
class CLogger
{
public:
    //! enum describing the different levels for logging
    enum LogLevel
    {
        /** no log */
        LEVEL_NO_LOG = 0,
        /** error level */
        LEVEL_ERROR,
        /** warning level */
        LEVEL_WARNING,
        /** info level */
        LEVEL_INFORMATION,
        /** debug level */
        LEVEL_DEBUG
    };

    //! enum describing the different styles for logging
    enum LogStyle
    {
        LOG_STYLE_NORMAL = 0,
        LOG_STYLE_FUNCTION_LINE = 1
    };

    //! Get the logging instance.
    //! \return Reference to the Logger object.
    static CLogger& GetInstance();

    //! Set the level for logging.
    //! \param[in] eLevel The logging level.
    void SetLogLevel(LogLevel eLevel);

    //! Get the level for logging.
    LogLevel GetLogLevel(void);

    //! Set the style for logging.
    //! \param[in] eStyle The logging style.
    void SetLogStyle(LogStyle eStyle);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] pszFormat Format string as a cstring.
    void LogLevelMessage(LogLevel eLevel,
                         const char *pszFunctionName,
                         uint32_t sLineNumber,
                         const char *pszFormat,
                         ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] sFormat Format string as a C++ string.
    void LogLevelMessage(LogLevel eLevel,
                         std::string sFunctionName,
                         uint32_t sLineNumber,
                         std::string sFormat,
                         ...);

    //! Log a message (cstring) at preset level.
    //! \param[in] pszFormat Format string as a cstring.
    void LogMessage(const char *pszFormat,
                    ...);

    //! Log a message (C++ string) at preset level.
    //! \param[in] sFormat Format string as a C++ string.
    void LogMessage(std::string sFormat,
                    ...);

private:
    //! Need private constructor because this is a singleton.
    CLogger() = default;
    LogLevel m_level = LEVEL_ERR;
    LogStyle m_style = LOG_STYLE_NORMAL;

    void LogLevelMessageVa(LogLevel eLevel,
                           const char *pszFunctionName,
                           uint32_t sLineNumber,
                           const char *pszFormat,
                           va_list ap);
    void LogMessageVa(const char *pszFormat,
                      va_list ap);
};
// CLogger class

class CUtils final
{
public:
    static SIPLStatus CreateRgbaBuffer(NvSciBufModule &bufModule,
                                       NvSciBufAttrList &bufAttrList,
                                       uint32_t width,
                                       uint32_t height,
                                       NvSciBufObj *pBufObj);
    static SIPLStatus ConvertRawToRgba(NvSciBufObj srcBufObj,
                                       uint8_t *pSrcBuf,
                                       NvSciBufObj dstBufObj,
                                       uint8_t *pDstBuf);
    static SIPLStatus IsRawBuffer(NvSciBufObj bufObj, bool &bIsRaw);
    static uint8_t * CreateImageBuffer(NvSciBufObj bufObj);
    static bool GetBpp(uint32_t buffBits, uint32_t *buffBytesVal);
};

SIPLStatus LoadNITOFile(std::string folderPath,
                        std::string moduleName,
                        std::vector<uint8_t>& nito);


#define MAX_NUM_SURFACES (3U)

#define FENCE_FRAME_TIMEOUT_MS (100UL)

typedef struct {
    NvSciBufType bufType;
    uint64_t size;
    uint32_t planeCount;
    NvSciBufAttrValImageLayoutType layout;
    uint32_t planeWidths[MAX_NUM_SURFACES];
    uint32_t planeHeights[MAX_NUM_SURFACES];
    uint32_t planePitches[MAX_NUM_SURFACES];
    uint32_t planeBitsPerPixels[MAX_NUM_SURFACES];
    uint32_t planeAlignedHeights[MAX_NUM_SURFACES];
    uint64_t planeAlignedSizes[MAX_NUM_SURFACES];
    uint8_t planeChannelCounts[MAX_NUM_SURFACES];
    uint64_t planeOffsets[MAX_NUM_SURFACES];
    uint64_t topPadding[MAX_NUM_SURFACES];
    uint64_t bottomPadding[MAX_NUM_SURFACES];
    bool needSwCacheCoherency;
    NvSciBufAttrValColorFmt planeColorFormats[MAX_NUM_SURFACES];
} BufferAttrs;

typedef enum {
    FMT_LOWER_BOUND,
    FMT_YUV_420SP_UINT8_BL,
    FMT_YUV_420SP_UINT8_PL,
    FMT_YUV_420SP_UINT16_BL,
    FMT_YUV_420SP_UINT16_PL,
    FMT_YUV_444SP_UINT8_BL,
    FMT_YUV_444SP_UINT8_PL,
    FMT_YUV_444SP_UINT16_BL,
    FMT_YUV_444SP_UINT16_PL,
    FMT_VUYX_UINT8_BL,
    FMT_VUYX_UINT8_PL,
    FMT_VUYX_UINT16_PL,
    FMT_LUMA_UINT16_PL,
    FMT_RGBA_FLOAT16_PL,
    FMT_UPPER_BOUND
} ISPOutputFormats;

SIPLStatus PopulateBufAttr(const NvSciBufObj& sciBufObj, BufferAttrs &bufAttrs);

#if !NV_IS_SAFETY
SIPLStatus PrintParameterSetID(uint8_t const *const IDArray, size_t const IDArrayLength);
SIPLStatus PrintParameterSetSchemaHash(uint8_t const *const schemaHashArray, size_t const schemaHashArrayLength);
SIPLStatus PrintParameterSetDataHash(uint8_t const *const dataHashArray, size_t const dataHashArrayLength);
#endif // !NV_IS_SAFETY

#endif
