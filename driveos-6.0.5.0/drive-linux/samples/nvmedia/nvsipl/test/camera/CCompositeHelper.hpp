/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
                            NvSciSyncModule syncModule) = 0;
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
    bool m_rgbaRegisteredWith2d {false};

    SIPLStatus AllocateImageBuffers(NvSciBufModule &bufModule,
                                    NvSciBufObj &rawBufObj,
                                    NvSciBufAttrList &bufAttrList) {
        if (m_scratchRGBABuffer == nullptr) {
            // Allocate scratch RGBA buffer for raw to RGBA conversion
            m_scratchRGBABuffer.reset(new NvSciBufObj());
            CHK_PTR_AND_RETURN(m_scratchRGBABuffer.get(), "NvSciBufObj creation");
            BufferAttrs rawBufAttrs;
            SIPLStatus status = PopulateBufAttr(rawBufObj, rawBufAttrs);
            CHK_STATUS_AND_RETURN(status, "PopulateBufAttr");
            status = CUtils::CreateRgbaBuffer(bufModule,
                                              bufAttrList,
                                              rawBufAttrs.planeWidths[0] / 2U,
                                              rawBufAttrs.planeHeights[0] / 2U,
                                              m_scratchRGBABuffer.get());
            CHK_STATUS_AND_RETURN(status, "CUtils::CreateRgbaBuffer");
            CHK_PTR_AND_RETURN(*m_scratchRGBABuffer, "CUtils::CreateRgbaBuffer");
        }
        if (m_rawImageBuf == nullptr) {
            // Allocate CPU buffer for raw image
            m_rawImageBuf.reset(CUtils::CreateImageBuffer(rawBufObj));
            CHK_PTR_AND_RETURN(m_rawImageBuf, "CUtils::CreateImageBuffer");
        }
        if (m_rgbaImageBuf == nullptr) {
            // Allocate CPU buffer for RGBA image
            m_rgbaImageBuf.reset(CUtils::CreateImageBuffer(*m_scratchRGBABuffer));
            CHK_PTR_AND_RETURN(m_rgbaImageBuf, "CUtils::CreateImageBuffer");
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus ConvertRawToRgba(NvSciBufObj &rawBufObj) {
        if (m_scratchRGBABuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        return CUtils::ConvertRawToRgba(rawBufObj,
                                        m_rawImageBuf.get(),
                                        *m_scratchRGBABuffer,
                                        m_rgbaImageBuf.get());
    }

    SIPLStatus GetRGBABuffer(NvSciBufObj &bufObj) {
        if (m_scratchRGBABuffer == nullptr) {
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }
        bufObj = *m_scratchRGBABuffer;
        return NVSIPL_STATUS_OK;
    }

private:
    std::unique_ptr<NvSciBufObj, CloseNvSciBufObj> m_scratchRGBABuffer {nullptr};
    std::unique_ptr<uint8_t[]> m_rawImageBuf {nullptr};
    std::unique_ptr<uint8_t[]> m_rgbaImageBuf {nullptr};
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
        LOG_INFO("Rectangle for group:%u link:%u output:%u is ", group, modIndex, outIndex);
        LOG_INFO("[(%u, %u) : (%u, %u)]\n", m_rect.x0, m_rect.y0, m_rect.x1, m_rect.y1);
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
                    NvSciSyncModule syncModule) override {
        if (m_pGroupInfos == nullptr) {
            LOG_ERR("CompositeHelper: m_pGroupInfos is null\n");
            return NVSIPL_STATUS_NOT_INITIALIZED;
        }

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

        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> bufAttrList2d;
        bufAttrList2d.reset(new NvSciBufAttrList());
        NvSciError sciErr = NvSciBufAttrListCreate(bufModule, bufAttrList2d.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");

        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufAttrKeyValuePair attrKvp = {
            NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType),
        };
        sciErr = NvSciBufAttrListSetAttrs(*bufAttrList2d, &attrKvp, 1U);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

        SIPLStatus status = FillNvSciBufAttrList(*bufAttrList2d);
        CHK_STATUS_AND_RETURN(status, "FillNvSciBufAttrList");

        status = m_dispIf->Init(m_outWidth, m_outHeight, bufModule, *bufAttrList2d);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CompositeHelper: Unable to initialize display interface\n");
            return status;
        }

        m_dstBuffer = m_dispIf->GetBuffer();
        if (m_dstBuffer == nullptr) {
            LOG_ERR("CompositeHelper: m_dstBuffer is null\n");
            return NVSIPL_STATUS_RESOURCE_ERROR;
        }

        for (uint32_t row = 0U; row < NUM_OF_ROWS; row++) {
            NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_p2dDevices[row].get(),
                                                                   m_dstBuffer);
            CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");
        }

        // Create a blank NvSciBufObj to clear the frames during transitions (RGBA PL, CPU access)
        status = CUtils::CreateRgbaBuffer(bufModule,
                                          *bufAttrList2d,
                                          m_outWidth,
                                          m_outHeight,
                                          &m_blankFrame);
        CHK_STATUS_AND_RETURN(status, "CUtils::CreateRgbaBuffer");
        if (m_blankFrame == nullptr) {
            LOG_ERR("CompositeHelper: Failed to create clear frame\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }

        // Memset the NvSciBufObj (assumes the blank frame to be RGBA)
        const uint32_t buffSize = m_outWidth * m_outHeight * 4U;
        const uint32_t buffPitch = m_outWidth * 4U;
        uint8_t *buff = new (std::nothrow) uint8_t[buffSize];
        if (buff == nullptr) {
            LOG_ERR("CompositeHelper: Failed to allocate buffer\n");
            return NVSIPL_STATUS_OUT_OF_MEMORY;
        }
        std::fill(buff, buff + buffSize, 0x00);
        sciErr = NvSciBufObjPutPixels(m_blankFrame,
                                      nullptr,
                                      (const void **)(&buff),
                                      &buffSize,
                                      &buffPitch);
        delete [] buff;
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjPutPixels");

        status = RegisterNvSciBufObj(0U, m_blankFrame);
        CHK_STATUS_AND_RETURN(status, "RegisterNvSciBufObj");

        // Create an NvSciSyncObj to perform a CPU wait after the blank frame transitions
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> reconciledAttrList;
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> conflictAttrList;
        reconciledAttrList.reset(new NvSciSyncAttrList());
        conflictAttrList.reset(new NvSciSyncAttrList());
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrList2d;
        attrList2d.reset(new NvSciSyncAttrList());
        sciErr = NvSciSyncAttrListCreate(syncModule, attrList2d.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "2D NvSciSyncAttrListCreate");
        status = FillNvSciSyncAttrList(0U, *attrList2d, NVMEDIA_SIGNALER);
        CHK_STATUS_AND_RETURN(status, "2D FillNvSciSyncAttrList");
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> attrListCpuWaiter;
        attrListCpuWaiter.reset(new NvSciSyncAttrList());
        sciErr = NvSciSyncAttrListCreate(syncModule, attrListCpuWaiter.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU NvSciSyncAttrListCreate");
        NvSciSyncAttrKeyValuePair keyVals[2];
        bool cpuWaiter = true;
        NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
        keyVals[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
        keyVals[0].value = (void *)&cpuWaiter;
        keyVals[0].len = sizeof(cpuWaiter);
        keyVals[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
        keyVals[1].value = (void*)&cpuPerm;
        keyVals[1].len = sizeof(cpuPerm);
        sciErr = NvSciSyncAttrListSetAttrs(*attrListCpuWaiter, keyVals, 2U);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");
        NvSciSyncAttrList attrListsForReconcile[2] = { *attrList2d, *attrListCpuWaiter };
        sciErr = NvSciSyncAttrListReconcile(attrListsForReconcile,
                                            2U,
                                            reconciledAttrList.get(),
                                            conflictAttrList.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");
        m_blankFrameSyncObj.reset(new NvSciSyncObj());
        CHK_PTR_AND_RETURN(m_blankFrameSyncObj, "NvSciSyncObj creation");
        sciErr = NvSciSyncObjAlloc(*reconciledAttrList, m_blankFrameSyncObj.get());
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");
        status = RegisterNvSciSyncObj(0U, NVMEDIA_EOFSYNCOBJ, *m_blankFrameSyncObj);
        CHK_STATUS_AND_RETURN(status, "2D RegisterNvSciSyncObj");

        // Create an NvSciSyncCpuWaitContext to perform the actual CPU waits
        sciErr = NvSciSyncCpuWaitContextAlloc(syncModule, &m_cpuWaitContext);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

        ClearOutputBuffer();

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

        SIPLStatus status = ClearOutputBuffer();
        if (status != NVSIPL_STATUS_OK) {
            ret = status;
        }

        if (m_blankFrameSyncObj != nullptr) {
            status = UnregisterNvSciSyncObj(0U, *m_blankFrameSyncObj);
            if (status != NVSIPL_STATUS_OK) {
                ret = status;
            }
            NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
        }

        if (m_dstBuffer != nullptr) {
            for (uint32_t row = 0U; row < NUM_OF_ROWS; row++) {
                NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_p2dDevices[row].get(),
                                                                         m_dstBuffer);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    LOG_ERR("NvMedia2DUnregisterNvSciBufObj failed. nvmStatus: %u\n", nvmStatus);
                    ret = NVSIPL_STATUS_ERROR;
                }
            }
        }

        if (m_blankFrame != nullptr) {
            status = UnregisterNvSciBufObj(0U, m_blankFrame);
            if (status != NVSIPL_STATUS_OK) {
                ret = status;
            }
            NvSciBufObjFree(m_blankFrame);
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

    // Copy blank frame to destination buffer
    SIPLStatus ClearOutputBuffer() {
        // Acquire an empty NvMedia 2D parameters object
        NvMedia2DComposeParameters params;
        NvMediaStatus nvmStatus = NvMedia2DGetComposeParameters(m_p2dDevices[0].get(), &params);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetComposeParameters");

        // Set the NvSciSyncObj to use for end-of-frame synchronization
        nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_p2dDevices[0].get(),
                                                   params,
                                                   *m_blankFrameSyncObj);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

        // Set the source layer parameters for layer zero
        nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_p2dDevices[0].get(), params, 0U, m_blankFrame);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");
        nvmStatus = NvMedia2DSetSrcGeometry(m_p2dDevices[0].get(),
                                            params,
                                            0U,
                                            nullptr,
                                            nullptr,
                                            NVMEDIA_2D_TRANSFORM_NONE);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcGeometry");

        // Set the destination surface
        nvmStatus = NvMedia2DSetDstNvSciBufObj(m_p2dDevices[0].get(), params, m_dstBuffer);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

        // Submit the compose operation
        NvMedia2DComposeResult composeResult;
        nvmStatus = NvMedia2DCompose(m_p2dDevices[0].get(), params, &composeResult);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCompose");

        // Get the end-of-frame fence from NvMedia 2D and perform a CPU wait on it
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_p2dDevices[0].get(), &composeResult, &fence);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");
        NvSciError sciErr = NvSciSyncFenceWait(&fence, m_cpuWaitContext, NvSciSyncFenceMaxTimeout);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
        NvSciSyncFenceClear(&fence);

        // Update the display with that frame
        SIPLStatus status = m_dispIf->WfdFlip();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CompositeHelper: WfdFlip (ClearOutputBuffer) failed! status: %u\n", status);
            return status;
        }

        return NVSIPL_STATUS_OK;
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

    SIPLStatus PostSync(T item, uint32_t row, NvMedia2DComposeResult *composeResult) {
        // Get the end-of-frame fence from NvMedia 2D
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        NvMediaStatus nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_p2dDevices[row].get(),
                                                                composeResult,
                                                                &fence);
        CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");

        // Add end-of-frame fence as pre-fence to the item
        SIPLStatus status = item.AddFence(fence);
        CHK_STATUS_AND_RETURN(status, "AddFence");
        NvSciSyncFenceClear(&fence);

        return NVSIPL_STATUS_OK;
    }

    void ThreadFunc() {
        pthread_setname_np(pthread_self(), "CCompositeHelper");

        m_bRunning = true;

        uint32_t sleepTimeMs = uint32_t(1000 / (60.0f)); // Assume refresh rate is 60 fps

        while (m_bRunning) {
            // Check if there is a pending group change
            if (m_uNewGroupIndex != m_uGroupIndex) {
                // Clear output buffer
                ClearOutputBuffer();

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
                T item;
                SIPLStatus status = upInput->QueueGet(item);
                if (status != NVSIPL_STATUS_OK) {
                    continue;
                }
                NvSciBufObj srcBuf = item.GetBuffer();
                if (srcBuf == nullptr) {
                    LOG_ERR("CompositeHelper: Couldn't get buffer from item\n");
                    return;
                }

                // Convert RAW to RGBA if necessary
                bool isRaw = false;
                status = CUtils::IsRawBuffer(srcBuf, isRaw);
                if (status != NVSIPL_STATUS_OK) {
                    LOG_ERR("CompositeHelper: CUtils::IsRawBuffer failed for source:%u\n", upInput->m_id);
                    return;
                }
                if (isRaw) {
                    status = upInput->ConvertRawToRgba(srcBuf);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("CompositeHelper: ConvertRawToRgba failed for source:%u\n", upInput->m_id);
                        return;
                    }
                    status = upInput->GetRGBABuffer(srcBuf);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("CompositeHelper: GetRGBABuffer failed for source:%u\n", upInput->m_id);
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
                                                       m_dstBuffer);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

                status = PreSync(item, row, params);
                if (status != NVSIPL_STATUS_OK) {
                    return;
                }

                // Submit the compose operation
                NvMedia2DComposeResult composeResult;
                nvmStatus = NvMedia2DCompose(m_p2dDevices[row].get(), params, &composeResult);
                CHK_NVMSTATUS_AND_EXIT(nvmStatus, "NvMedia2DCompose");

                status = PostSync(item, row, &composeResult);
                if (status != NVSIPL_STATUS_OK) {
                   return;
                }

                // Automatically release underlying data (buffer or packet) when item is destroyed
            }

            // Send composited output image to display
            SIPLStatus status = m_dispIf->WfdFlip();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CompositeHelper: WfdFlip failed! status: %u\n", status);
                return;
            }
        } // while ()

        return;
    }

    // Output
    NvSciBufObj m_dstBuffer {nullptr};
    NvSciBufObj m_blankFrame {nullptr};
    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> m_blankFrameSyncObj;
    NvSciSyncCpuWaitContext m_cpuWaitContext;

    // Threading
    std::unique_ptr<std::thread> m_pthread {nullptr};
    std::atomic<bool> m_bRunning; // Flag indicating if thread is running
    CGroupInfo<T> *m_pGroupInfos = {nullptr};
};

#endif // CCOMPOSITEHELPER_HPP
