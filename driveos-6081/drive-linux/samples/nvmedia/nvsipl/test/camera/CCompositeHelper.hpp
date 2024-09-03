/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <unistd.h>

#include "nvmedia_2d_sci.h"
#include "nvmedia_2d.h"

#include "CNvWfd.hpp"
#include "CUtils.hpp"

#ifndef CCOMPOSITEHELPER_HPP
#define CCOMPOSITEHELPER_HPP

static const uint32_t NUM_OF_GROUPS = 4U;
static const uint32_t NUM_OF_COLS = 4U;
static const uint32_t NUM_OF_ROWS = 4U;
static const uint32_t NUM_OF_INPUTS = NUM_OF_GROUPS * NUM_OF_COLS * NUM_OF_ROWS;

struct DestroyNvMedia2DDevice
{
    void operator ()(NvMedia2D *p) const
    {
        NvMedia2DDestroy(p);
    }
};

// Interface class for composition/display helper
class ICompositeHelper
{
public:
    virtual SIPLStatus Init(NvSiplRect *pRect,
                            NvSciBufModule bufModule,
                            NvSciSyncCpuWaitContext cpuWaitContext,
                            bool soloInput) = 0;
    virtual SIPLStatus Start() = 0;
    virtual SIPLStatus Stop() = 0;
    virtual SIPLStatus Deinit() = 0;
    virtual SIPLStatus FillNvSciSyncAttrList(uint32_t id,
                                             NvSciSyncAttrList &attrList,
                                             NvMediaNvSciSyncClientType clientType) = 0;
    virtual SIPLStatus RegisterNvSciSyncObj(uint32_t id,
                                            NvMediaNvSciSyncObjType syncObjType,
                                            NvSciSyncObj &syncObj) = 0;
    virtual SIPLStatus UnregisterNvSciSyncObj(uint32_t id, NvSciSyncObj &syncObj) = 0;
    virtual SIPLStatus FillNvSciBufAttrList(NvSciBufAttrList &attrList) = 0;
    virtual SIPLStatus RegisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj) = 0;
    virtual SIPLStatus UnregisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj) = 0;

    // Display
    uint32_t m_uGroupIndex = -1U; // ID of current active group
    std::atomic<uint32_t> m_uNewGroupIndex; // ID of new active group
    IDisplayInterface *m_dispIf {nullptr};

    // 2D
    std::unique_ptr<NvMedia2D, DestroyNvMedia2DDevice> m_p2dDevices[NUM_OF_ROWS] {nullptr};
        // Note: Creating a 2D context for each row (camera module) to get around the limit on the
        // number of surfaces that can be used with a single 2D context

    // Output
    uint32_t m_outWidth = 0U;
    uint32_t m_outHeight = 0U;
    bool m_soloInput = false;
};

// Synchronization data for an input from a SIPL producer
class CInputSync
{
public:
    uint32_t m_id;

    SIPLStatus SetSyncObj(std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj,
                          NvMediaNvSciSyncObjType syncObjType) {
        if ((syncObjType != NVMEDIA_PRESYNCOBJ) && (syncObjType != NVMEDIA_EOFSYNCOBJ)) {
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        const uint32_t syncObjIdx = static_cast<uint32_t>(syncObjType);
        m_syncObjs[syncObjIdx] = std::move(syncObj);
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus GetSyncObj(NvSciSyncObj &syncObj, NvMediaNvSciSyncObjType syncObjType) {
        if ((syncObjType != NVMEDIA_PRESYNCOBJ) && (syncObjType != NVMEDIA_EOFSYNCOBJ)) {
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }
        const uint32_t syncObjIdx = static_cast<uint32_t>(syncObjType);
        if (m_syncObjs[syncObjIdx] == nullptr) {
            return NVSIPL_STATUS_ERROR;
        }
        syncObj = *m_syncObjs[syncObjIdx];
        return NVSIPL_STATUS_OK;
    }

private:
    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> m_syncObjs[2];
};

// Buffer data for an input from a SIPL producer
class CInputBuffers
{
public:
    bool m_scratchRegisteredWith2d {false};

    SIPLStatus AllocateImageBuffers(NvSciBufModule &bufModule,
                                    NvSciBufObj &rawBufObj,
                                    NvSciBufAttrList &bufAttrList) {
        if (m_scratchBuffer == nullptr) {
            // Allocate scratch RGBA buffer for raw to RGBA conversion
            m_scratchBuffer.reset(new NvSciBufObj());
            CHK_PTR_AND_RETURN(m_scratchBuffer.get(), "NvSciBufObj creation");
            bool isRGBIR = false;
            SIPLStatus status = CUtils::IsRGBIRBuffer(rawBufObj, isRGBIR);
            CHK_STATUS_AND_RETURN(status, "IsRGBIRBuffer");
            BufferAttrs rawBufAttrs;
            status = PopulateBufAttr(rawBufObj, rawBufAttrs);
            CHK_STATUS_AND_RETURN(status, "PopulateBufAttr");
            status = CUtils::CreateScratchBuffer(bufModule,
                                                 bufAttrList,
                                                 rawBufAttrs.planeWidths[0] / 2U,
                                                 rawBufAttrs.planeHeights[0] / 2U,
                                                 m_scratchBuffer.get(),
                                                 isRGBIR);
            CHK_STATUS_AND_RETURN(status, "CUtils::CreateScratchBuffer");
            CHK_PTR_AND_RETURN(*m_scratchBuffer, "CUtils::CreateScratchBuffer");
        }
        if (m_rawImageBuf == nullptr) {
            // Allocate CPU buffer for raw image
            m_rawImageBuf.reset(CUtils::CreateImageBuffer(rawBufObj));
            CHK_PTR_AND_RETURN(m_rawImageBuf, "CUtils::CreateImageBuffer");
        }
        if (m_scratchImageBuf == nullptr) {
            // Allocate CPU buffer for RGBA image
            m_scratchImageBuf.reset(CUtils::CreateImageBuffer(*m_scratchBuffer));
            CHK_PTR_AND_RETURN(m_scratchImageBuf, "CUtils::CreateImageBuffer");
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus ConvertRaw(NvSciBufObj &rawBufObj) {
        if (m_scratchBuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        return CUtils::ConvertRaw(rawBufObj,
                                  m_rawImageBuf.get(),
                                  *m_scratchBuffer,
                                  m_scratchImageBuf.get());
    }

    SIPLStatus GetScratchBuffer(NvSciBufObj &bufObj) {
        if (m_scratchBuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        bufObj = *m_scratchBuffer;
        return NVSIPL_STATUS_OK;
    }

private:
    std::unique_ptr<NvSciBufObj, CloseNvSciBufObj> m_scratchBuffer {nullptr};
    std::unique_ptr<uint8_t[]> m_rawImageBuf {nullptr};
    std::unique_ptr<uint8_t[]> m_scratchImageBuf {nullptr};
};

// Data related to input from a SIPL producer; a locking queue with some enhancements
template <class T>
class CInputInfo : public CInputBuffers, public CInputSync
{
public:
    NvSiplRect m_rect;
    bool m_isSimulatorMode {false};

    void QueueAdd(T item) {
        std::unique_lock<std::mutex> lck(m_queueMut);
        m_queue.push(item);
    }

    bool IsQueueEmpty() {
        std::unique_lock<std::mutex> lck(m_queueMut);
        return m_queue.empty();
    }

    SIPLStatus QueueGet(T &item) {
        std::unique_lock<std::mutex> lck(m_queueMut);
        if (m_queue.empty()) {
            LOG_INFO("Queue is empty for input:%u\n", m_id);
            return NVSIPL_STATUS_ERROR;
        }
        item = m_queue.front();
        m_queue.pop();
        return NVSIPL_STATUS_OK;
    }

    void DequeueAndReleaseAll() {
        std::unique_lock<std::mutex> lck(m_queueMut);
        while (!m_queue.empty()) {
            // Release is performed automatically in destructor of queue item
            m_queue.pop();
        }
    }

    void SetRect(uint32_t group, bool soloInput, uint32_t outputWidth, uint32_t outputHeight) {
        uint16_t startX = 0U;
        uint16_t startY = 0U;
        uint16_t endX = outputWidth;
        uint16_t endY = outputHeight;
        uint32_t subId = m_id - (group * NUM_OF_COLS * NUM_OF_ROWS);
        uint32_t modIndex = subId / NUM_OF_COLS;
        uint32_t outIndex = subId % NUM_OF_COLS;

        // If this is the only input in the group, use the full screen
        // If there are other inputs in the group, calculate this input's destination rectangle
        if (!soloInput) {
            uint16_t xStep = outputWidth / NUM_OF_COLS;
            uint16_t yStep = outputHeight / NUM_OF_ROWS;
            startX = outIndex * xStep;
            startY = modIndex * yStep;
            endX = startX + xStep;
            endY = startY + yStep;
        }

        m_rect = { startX, startY, endX, endY };
        LOG_INFO("Rectangle for group:%u link:%u output:%u is \n", group, modIndex, outIndex);
        PRINT_RECT(m_rect);
    }

private:
    std::mutex m_queueMut;
    std::queue<T> m_queue;
};

// Interface class for group management
class IGroupInfo
{
public:
    uint32_t m_id;
    std::atomic<bool> m_bGroupInUse;

    virtual SIPLStatus AddInput(uint32_t id, bool isSimulatorMode) = 0;
    virtual bool GetInput(uint32_t id, CInputSync * &pInput) = 0;
    virtual bool GetInput(uint32_t id, CInputBuffers * &pInput) = 0;
    virtual void GetAllInputs(std::vector<CInputSync *> &pInputs) = 0;
    virtual void DequeueAndReleaseAll() = 0;
    virtual bool CheckInputQueues() = 0;
    virtual void SetRects(uint32_t outputWidth, uint32_t outputHeight) = 0;
    virtual bool HasInputs() = 0;
};

// Group of inputs from a single deserializer, composited to a single frame and displayed together
template <class T>
class CGroupInfo : public IGroupInfo
{
public:
    std::vector<std::unique_ptr<CInputInfo<T>>> m_vInputs;

    SIPLStatus AddInput(uint32_t id, bool isSimulatorMode) override {
        std::unique_ptr<CInputInfo<T>> upInputInfo(new (std::nothrow) CInputInfo<T>());
        CHK_PTR_AND_RETURN(upInputInfo, "CInputInfo creation");
        upInputInfo->m_id = id;
        upInputInfo->m_isSimulatorMode = isSimulatorMode;
        m_vInputs.push_back(std::move(upInputInfo));
        return NVSIPL_STATUS_OK;
    }

    bool GetInput(uint32_t id, CInputSync * &pInput) {
        for (std::unique_ptr<CInputInfo<T>> &upInputIter : m_vInputs) {
            if (upInputIter->m_id == id) {
                pInput = static_cast<CInputSync *>(upInputIter.get());
                return true;
            }
        }
        return false;
    }

    bool GetInput(uint32_t id, CInputBuffers * &pInput) {
        for (std::unique_ptr<CInputInfo<T>> &upInputIter : m_vInputs) {
            if (upInputIter->m_id == id) {
                pInput = static_cast<CInputBuffers *>(upInputIter.get());
                return true;
            }
        }
        return false;
    }

    CInputInfo<T> * GetInput(uint32_t id) {
        for (std::unique_ptr<CInputInfo<T>> &upInputIter : m_vInputs) {
            if (upInputIter->m_id == id) {
                return upInputIter.get();
            }
        }
        LOG_ERR("Failed to find input information structure\n");
        return nullptr;
    }

    bool GetIfActive(uint32_t id, CInputInfo<T> * &pInput) {
        if (m_bGroupInUse) {
            for (std::unique_ptr<CInputInfo<T>> &upInputIter : m_vInputs) {
                if (upInputIter->m_id == id) {
                    pInput = upInputIter.get();
                    return true;
                }
            }
        }
        return false;
    }

    void GetAllInputs(std::vector<CInputSync *> &pInputs) override {
        pInputs.clear();
        for (std::unique_ptr<CInputInfo<T>> &upInputIter : m_vInputs) {
            pInputs.push_back(static_cast<CInputSync *>(upInputIter.get()));
        }
    }

    void DequeueAndReleaseAll() override {
        for (std::unique_ptr<CInputInfo<T>> &upInput : m_vInputs) {
            upInput->DequeueAndReleaseAll();
        }
    }

    bool CheckInputQueues() override {
        bool anyReady = false;
        for (std::unique_ptr<CInputInfo<T>> &upInput : m_vInputs) {
            if (!upInput->IsQueueEmpty()) {
                anyReady = true;
                break;
            }
        }
        return anyReady;
    }

    void SetRects(uint32_t outputWidth, uint32_t outputHeight) override {
        for (std::unique_ptr<CInputInfo<T>> &upInput : m_vInputs) {
            upInput->SetRect(m_id, (m_vInputs.size() == 1U), outputWidth, outputHeight);
        }
    }

    bool HasInputs() override {
        return (m_vInputs.size() != 0U);
    }
};

// Composition/display helper, takes a group of inputs and sends them to a display
template <class T>
class CCompositeHelper : public ICompositeHelper
{
public:
    CCompositeHelper(CGroupInfo<T> *pGroupInfos) {
        m_pGroupInfos = pGroupInfos;
    }

    SIPLStatus Init(NvSiplRect *pRect,
                    NvSciBufModule bufModule,
                    NvSciSyncCpuWaitContext cpuWaitContext,
                    bool soloInput) override {
        if (m_pGroupInfos == nullptr) {
            LOG_ERR("CompositeHelper: m_pGroupInfos is null\n");
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        m_soloInput = soloInput;
        m_cpuWaitContext = cpuWaitContext;

        // Create 2D for each input
        for (uint32_t row = 0U; row < NUM_OF_ROWS; row++) {
            NvMedia2D *p2dHandle = nullptr;
            NvMedia2DAttributes attr2d {
                1U,   // numComposeParameters
                256U, // maxRegisteredBuffers
                33U,  // maxRegisteredSyncs
                0U,   // maxFilterBuffers
                0U,   // flags
            };
            NvMediaStatus nvmStatus = NvMedia2DCreate(&p2dHandle, &attr2d);
            m_p2dDevices[row].reset(p2dHandle);
            CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCreate");
            if (!m_p2dDevices[row]) {
                LOG_ERR("CompositeHelper: NvMedia2DCreate failed for source:%u\n", row);
                return NVSIPL_STATUS_ERROR;
            }
        }

        if (pRect != nullptr) {
            m_outWidth = (pRect->x1 - pRect->x0);
            m_outHeight = (pRect->y1 - pRect->y0);
            if (m_outWidth == 0 || m_outHeight == 0) {
                LOG_ERR("CompositeHelper: Invalid output resolution specified\n");
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }
        }

        NvSciBufAttrList bufAttrList2d;
        NvSciError sciErr = NvSciBufAttrListCreate(bufModule, &bufAttrList2d);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");
        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufAttrKeyValuePair attrKvp = {
            NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType),
        };
        sciErr = NvSciBufAttrListSetAttrs(bufAttrList2d, &attrKvp, 1U);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
        SIPLStatus status = FillNvSciBufAttrList(bufAttrList2d);
        CHK_STATUS_AND_RETURN(status, "FillNvSciBufAttrList");

        status = m_dispIf->Init(m_outWidth, m_outHeight, bufModule, bufAttrList2d);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CompositeHelper: Unable to initialize display interface\n");
            return status;
        }

        m_dispIf->GetBuffers(m_vDstBuffer);
        for (NvSciBufObj bufObj : m_vDstBuffer) {
            for (uint32_t row = 0U; row < NUM_OF_ROWS; row++) {
                NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_p2dDevices[row].get(),
                                                                       bufObj);
                CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
            }
        }

        // Clear display
        status = m_dispIf->WfdClearDisplay();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CompositeHelper: Failed to clear display\n");
            return status;
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Start() override {
        if (m_uGroupIndex == -1U) {
            LOG_ERR("CompositeHelper: No input registered\n");
            return NVSIPL_STATUS_INVALID_STATE;
        }

        m_pGroupInfos[m_uGroupIndex].SetRects(m_outWidth, m_outHeight);

        // Start the thread
        LOG_INFO("Starting compositor helper thread\n");
        m_pthread.reset(new std::thread(&CCompositeHelper<T>::ThreadFunc, this));
        if (m_pthread == nullptr) {
            LOG_ERR("Failed to create compositor helper thread\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }
        LOG_INFO("Created compositor thread ID:%u\n", m_pthread->get_id());

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Stop() override {
        // Signal thread to stop
        m_bRunning = false;

        // Wait for the thread
        if ((m_pthread != nullptr) && (m_pthread->joinable())) {
            LOG_INFO("Waiting to join compositor helper thread ID:%u\n", m_pthread->get_id());
            m_pthread->join();
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Deinit() override {
        SIPLStatus ret = NVSIPL_STATUS_OK;

        // Clear display
        SIPLStatus status = m_dispIf->WfdClearDisplay();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CompositeHelper: Failed to clear display\n");
            ret = status;
        }

        if (m_cpuWaitContext != nullptr) {
            m_cpuWaitContext = nullptr; // Dont free, owned by calling class
        }

        for (NvSciBufObj& bufObj : m_vDstBuffer) {
            if (bufObj != nullptr) {
                for (uint32_t row = 0U; row < NUM_OF_ROWS; row++) {
                    NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_p2dDevices[row].get(),
                                                                            bufObj);
                    if (nvmStatus != NVMEDIA_STATUS_OK) {
                        LOG_ERR("NvMedia2DUnregisterNvSciBufObj failed. nvmStatus: %u\n", nvmStatus);
                        ret = NVSIPL_STATUS_ERROR;
                    }
                }
            }
        }

        return ret;
    }

    static inline uint32_t GetRowFromId(uint32_t id) {
        return ((id % (NUM_OF_ROWS * NUM_OF_COLS)) / NUM_OF_COLS);
    }

    SIPLStatus FillNvSciSyncAttrList(uint32_t id,
                                     NvSciSyncAttrList &attrList,
                                     NvMediaNvSciSyncClientType clientType) override {
        const uint32_t row = GetRowFromId(id);
        NvMediaStatus nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_p2dDevices[row].get(),
                                                                 attrList,
                                                                 clientType);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus RegisterNvSciSyncObj(uint32_t id,
                                    NvMediaNvSciSyncObjType syncObjType,
                                    NvSciSyncObj &syncObj) override {
        const uint32_t row = GetRowFromId(id);
        NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_p2dDevices[row].get(),
                                                                syncObjType,
                                                                syncObj);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus UnregisterNvSciSyncObj(uint32_t id, NvSciSyncObj &syncObj) override {
        const uint32_t row = GetRowFromId(id);
        NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_p2dDevices[row].get(), syncObj);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciSyncObj");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus FillNvSciBufAttrList(NvSciBufAttrList &attrList) override {
        NvMediaStatus nvmStatus = NvMedia2DFillNvSciBufAttrList(m_p2dDevices[0].get(), attrList);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciBufAttrList");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus RegisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj) override {
        const uint32_t row = GetRowFromId(id);
        NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_p2dDevices[row].get(), bufObj);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus UnregisterNvSciBufObj(uint32_t id, NvSciBufObj &bufObj) override {
        const uint32_t row = GetRowFromId(id);
        NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_p2dDevices[row].get(), bufObj);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciBufObj");
        return NVSIPL_STATUS_OK;
    }

    virtual ~CCompositeHelper() = default;

protected:
    bool CheckInputQueues() {
        return m_pGroupInfos[m_uGroupIndex].CheckInputQueues();
    }

    SIPLStatus PreSync(T item, uint32_t row, NvMedia2DComposeParameters &params) {
        // Get the post-fence from the item
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        SIPLStatus status = item.GetFence(&fence);
        CHK_STATUS_AND_RETURN(status, "GetFence");

        // Set the pre-fence for NvMedia 2D
        NvMediaStatus nvmStatus = NvMedia2DInsertPreNvSciSyncFence(m_p2dDevices[row].get(),
                                                                   params,
                                                                   &fence);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DInsertPreNvSciSyncFence");
        NvSciSyncFenceClear(&fence);

        return status;
    }

    SIPLStatus PostSync(T item,
                        uint32_t row,
                        NvMedia2DComposeResult *composeResult) {
        // Get the end-of-frame fence from NvMedia 2D
        NvSciSyncFenceClear(&m_fence);
        m_fence = NvSciSyncFenceInitializer;
        NvMediaStatus nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_p2dDevices[row].get(),
                                                                composeResult,
                                                                &m_fence);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");

        SIPLStatus status = item.AddFence(m_fence);
        CHK_STATUS_AND_RETURN(status, "AddFence");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus FenceWait(void) {
        NvSciError sciErr = NvSciSyncFenceWait(&m_fence,
                                               m_cpuWaitContext,
                                               NvSciSyncFenceMaxTimeout);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");

        return NVSIPL_STATUS_OK;
    }

    void ThreadFunc() {
        pthread_setname_np(pthread_self(), "CCompositeHelper");

        m_bRunning = true;

        uint32_t sleepTimeMs = uint32_t(1000 / (60.0f)); // Assume refresh rate is 60 fps

        // To store buffers in case of frame not ready
        T storedItems[MAX_OUTPUTS_PER_SENSOR*MAX_SENSORS];
        // To keep track of whether something was written to a destination buffer for a particular rectangle
        bool dstBufState[BUFFER_NUM][MAX_OUTPUTS_PER_SENSOR*MAX_SENSORS] {{false}};

        while (m_bRunning) {
            // Check if there is a pending group change
            if (m_uNewGroupIndex != m_uGroupIndex) {
                // Clear display
                SIPLStatus status = m_dispIf->WfdClearDisplay();
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("CompositeHelper: Failed to clear display\n");
                    return;
                }

                for (uint32_t i = 0U; i < (MAX_OUTPUTS_PER_SENSOR*MAX_SENSORS); i++) {
                    // Release all stored items
                    storedItems[i].ReleaseItem();
                }
                // Reset destination buffer state
                memset(&dstBufState[0][0], false, sizeof(dstBufState) );

                // Release buffers from input queues for current group
                m_pGroupInfos[m_uGroupIndex].DequeueAndReleaseAll();

                // Indicate that old group index is no longer being used
                bool bGroupInUse = true;
                if (!m_pGroupInfos[m_uGroupIndex].m_bGroupInUse.compare_exchange_strong(bGroupInUse, false)) {
                    LOG_ERR("CompositeHelper: Group is being used without being owner:%u\n", m_uGroupIndex);
                    return;
                }
                // Update current group index
                m_uGroupIndex = m_uNewGroupIndex;
                // Adjust new group's rectangle sizes to match this compositor/display
                m_pGroupInfos[m_uGroupIndex].SetRects(m_outWidth, m_outHeight);
            }

            // Check for input readiness
            bool anyReady = CheckInputQueues();
            if (!anyReady) {
                LOG_INFO("CompositeHelper: No inputs available yet\n");
                // Sleep for refresh rate
                std::this_thread::sleep_for(std::chrono::milliseconds(sleepTimeMs));
                continue;
            }

            // Get full buffer from input queue and composite to output image
            for (std::unique_ptr<CInputInfo<T>> &upInput : m_pGroupInfos[m_uGroupIndex].m_vInputs) {
                bool dupBuff = false;
                NvSciBufObj srcBuf;
                T item;
                SIPLStatus status = upInput->QueueGet(item);
                if (status == NVSIPL_STATUS_OK) {
                    srcBuf = item.GetBuffer();
                    if (srcBuf == nullptr) {
                        LOG_ERR("CompositeHelper: Couldn't get buffer from item\n");
                        return;
                    }
                    storedItems[upInput->m_id].ReleaseItem(); // Release stored old item
                    storedItems[upInput->m_id] = item; // Store Item in case its needed for duplication
                } else {
                    if (dstBufState[m_uDstBufferIndex][upInput->m_id]) {
                        // Destination Buffer needs duplication?
                        srcBuf = storedItems[upInput->m_id].GetBuffer();
                        dstBufState[m_uDstBufferIndex][upInput->m_id] = false;
                        dupBuff = true;
                    } else {
                        // Destination Buffer already duplicated
                        // Release item to avoid frame drops
                        storedItems[upInput->m_id].ReleaseItem();
                        continue;
                    }
                }

                // Convert RAW to RGBA if necessary
                bool isRaw = false;
                status = CUtils::IsRawBuffer(srcBuf, isRaw);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("CompositeHelper: CUtils::IsRawBuffer failed for source:%u\n", upInput->m_id);
                    return;
                }
                if (isRaw) {
                    status = upInput->ConvertRaw(srcBuf);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("CompositeHelper: ConvertRawToRgba failed for source:%u\n",
                                upInput->m_id);
                        return;
                    }
                    status = upInput->GetScratchBuffer(srcBuf);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("CompositeHelper: GetScratchBuffer failed for source:%u\n",
                                upInput->m_id);
                        return;
                    }
                }

                // Blit
                uint32_t row = (upInput->m_id - (m_uGroupIndex * NUM_OF_COLS * NUM_OF_ROWS)) / NUM_OF_COLS;
                // Acquire an empty NvMedia 2D parameters object
                NvMedia2DComposeParameters params;
                NvMediaStatus nvmStatus = NvMedia2DGetComposeParameters(m_p2dDevices[row].get(), &params);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DGetComposeParameters");

                // Set the NvSciSyncObj to use for end-of-frame synchronization
                NvSciSyncObj eofSyncObj;
                status = upInput->GetSyncObj(eofSyncObj, NVMEDIA_EOFSYNCOBJ);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("CompositeHelper: GetSyncObj failed for source:%u\n", upInput->m_id);
                    return;
                }
                nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_p2dDevices[row].get(), params, eofSyncObj);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

                // Set the source layer parameters for layer zero
                nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_p2dDevices[row].get(), params, 0U, srcBuf);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");

                NvMediaRect nvmRect { upInput->m_rect.x0,
                                      upInput->m_rect.y0,
                                      upInput->m_rect.x1,
                                      upInput->m_rect.y1 };
                nvmStatus = NvMedia2DSetSrcGeometry(m_p2dDevices[row].get(),
                                                    params,
                                                    0U,
                                                    NULL,
                                                    &nvmRect,
                                                    NVMEDIA_2D_TRANSFORM_NONE);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetSrcGeometry");

                // Set the destination surface
                nvmStatus = NvMedia2DSetDstNvSciBufObj(m_p2dDevices[row].get(),
                                                       params,
                                                       m_vDstBuffer[m_uDstBufferIndex]);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

                status = PreSync(storedItems[upInput->m_id], row, params);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                // Submit the compose operation
                NvMedia2DComposeResult composeResult;
                nvmStatus = NvMedia2DCompose(m_p2dDevices[row].get(), params, &composeResult);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DCompose");

                status = PostSync(storedItems[upInput->m_id], row, &composeResult);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                dstBufState[m_uDstBufferIndex][upInput->m_id] = !dupBuff;
            }

            SIPLStatus status = FenceWait();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CompositeHelper: Fence Wait failed! status: %u\n", status);
                return;
            }

            // Send composited output image to display
            status = m_dispIf->WfdFlip(m_vDstBuffer[m_uDstBufferIndex]);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CompositeHelper: WfdFlip failed! status: %u\n", status);
                return;
            }

            m_uDstBufferIndex = (m_uDstBufferIndex + 1U) % m_vDstBuffer.size();
        } // while ()

        return;
    }

    // Output
    uint32_t m_uDstBufferIndex = 0U;
    std::vector<NvSciBufObj> m_vDstBuffer;
    NvSciSyncCpuWaitContext m_cpuWaitContext;
    NvSciSyncFence m_fence = NvSciSyncFenceInitializer;

    // Threading
    std::unique_ptr<std::thread> m_pthread {nullptr};
    std::atomic<bool> m_bRunning; // Flag indicating if thread is running
    CGroupInfo<T> *m_pGroupInfos = {nullptr};
};

#endif // CCOMPOSITEHELPER_HPP
