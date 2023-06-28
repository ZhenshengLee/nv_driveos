/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "NvSIPLClient.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "CUtils.hpp"
#include "nvscisync.h"
#include <mutex>

#ifndef CFRAMEFEEDER_HPP
#define CFRAMEFEEDER_HPP

using namespace nvsipl;

class CSemaphore
{
public:
    CSemaphore(uint32_t const uMaxCount) {
        m_uMaxCount = uMaxCount;
        m_upCondVar.reset(new std::condition_variable());
    }

    CSemaphore() = delete;

    void Increment() {
        bool bNotify = false;
        std::unique_lock<std::mutex> lck(m_mutex);
        if (m_uCount < m_uMaxCount) {
            bNotify = (m_uCount == 0U);
            m_uCount++;
        }
        lck.unlock();

        if (bNotify) {
            m_upCondVar->notify_all();
        }
    }

    void Decrement() {
        std::unique_lock<std::mutex> lck(m_mutex);
        while (m_uCount == 0U) {
            m_upCondVar->wait(lck);
        }
        m_uCount--;
        lck.unlock();
    }

private:
    std::mutex m_mutex {};
    std::unique_ptr<std::condition_variable> m_upCondVar {nullptr};
    uint32_t m_uCount {0U};
    uint32_t m_uMaxCount {0U};
};

// CFrameFeeder: Supplies frames captured by one pipeline to another pipeline for ISP processing
class CFrameFeeder: public NvSIPLImageGroupWriter
{
public:
    virtual ~CFrameFeeder() {
        Deinit();
    }

    SIPLStatus Init(NvSciSyncModule syncModule,
                    std::atomic<bool> *pQuitFlag) {
        CHK_PTR_AND_RETURN_BADARG(pQuitFlag, "Quit flag pointer");
        m_pQuitFlag = pQuitFlag;

        return NVSIPL_STATUS_OK;
    }

    void Stop() {
        m_bStop = true;
        return;
    }

    void Deinit() {
        INvSIPLClient::INvSIPLBuffer *pBuffer = nullptr;
        std::unique_lock<std::mutex> inLck(m_inputQueueMut);
        while (m_inputQueue.size() > 0U) {
            m_currBufferSemaphore.Decrement();
            pBuffer = m_inputQueue.front();
            m_inputQueue.pop();
            pBuffer->Release();
        }
        inLck.unlock();
        std::unique_lock<std::mutex> outLck(m_outputQueueMut);
        while (m_outputQueue.size() > 0U) {
            pBuffer = m_outputQueue.front();
            m_outputQueue.pop();
            pBuffer->Release();
        }
        outLck.unlock();
        return;
    }

    SIPLStatus SetInputFrame(INvSIPLClient::INvSIPLBuffer *pBuffer) {
        CHK_PTR_AND_RETURN_BADARG(pBuffer, "SetInputFrame pointer");
        std::unique_lock<std::mutex> inLck(m_inputQueueMut);
        if (m_inputQueue.size() >= NUM_CAPTURE_BUFFERS_PER_POOL) {
            LOG_ERR("Buffer input queue is full, not able to accept input frame\n");
            return NVSIPL_STATUS_ERROR;
        }
        m_inputQueue.push(pBuffer);
        inLck.unlock();
        pBuffer->AddRef();
        m_currBufferSemaphore.Increment();

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus DeliverOutputFrame(INvSIPLClient::INvSIPLBuffer *pBuffer) {
        // Confirm that output buffer pointer is valid
        CHK_PTR_AND_RETURN_BADARG(pBuffer, "DeliverOutputFrame pointer");

        // Get buffer from queue
        std::unique_lock<std::mutex> outLck(m_outputQueueMut);
        if (m_outputQueue.empty()) {
            LOG_ERR("Buffer output queue is empty, not able to accept output frame\n");
            return NVSIPL_STATUS_ERROR;
        }
        INvSIPLClient::INvSIPLBuffer *pCurrBuffer = m_outputQueue.front();
        m_outputQueue.pop();
        outLck.unlock();
        CHK_PTR_AND_RETURN_BADARG(pCurrBuffer, "Current output buffer pointer");

        // Set ISP buffer post-fence as capture buffer pre-fence
        NvSciSyncFence fence = NvSciSyncFenceInitializer;
        SIPLStatus status = pBuffer->GetEOFNvSciSyncFence(&fence);
        CHK_STATUS_AND_RETURN(status, "GetEOFNvSciSyncFence");
        status = pCurrBuffer->AddNvSciSyncPrefence(fence);
        NvSciSyncFenceClear(&fence);
        CHK_STATUS_AND_RETURN(status, "AddNvSciSyncPrefence");

        // Release capture buffer
        status = pCurrBuffer->Release();
        CHK_STATUS_AND_RETURN(status, "INvSIPLClient::INvSIPLBuffer::Release");

        return NVSIPL_STATUS_OK;
    }

    // Implement the callback function
    SIPLStatus FillRawBuffer(RawBuffer &oRawBuffer) final
    {
        // Check if being told to exit
        if (m_bStop) {
            return NVSIPL_STATUS_EOF;
        }

        // Wait for a buffer to have been provided
        m_currBufferSemaphore.Decrement();

        // Get pointer to buffer
        std::unique_lock<std::mutex> inLck(m_inputQueueMut);
        if (m_inputQueue.empty()) {
            LOG_ERR("Buffer input queue is empty, not able to fill buffer for reprocessing\n");
            *m_pQuitFlag = true;
            return NVSIPL_STATUS_ERROR;
        }
        INvSIPLClient::INvSIPLBuffer *pCurrBuffer = m_inputQueue.front();
        inLck.unlock();
        CHK_PTR_QUIT_AND_RETURN(pCurrBuffer, m_pQuitFlag, "Current input buffer pointer");

        // Check that the image in the input buffer is the same as the image in the stored buffer
        INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer =
                (INvSIPLClient::INvSIPLNvMBuffer *)pCurrBuffer;
        CHK_PTR_QUIT_AND_RETURN(pNvMBuffer, m_pQuitFlag, "Converted current buffer pointer");
        if (pNvMBuffer->GetNvSciBufImage() != oRawBuffer.image) {
            // The stored buffer doesn't match the input, inject a frame drop to try again
            oRawBuffer.dropBuffer = true;
            m_currBufferSemaphore.Increment();
            return NVSIPL_STATUS_OK;
        }

        // Capture buffers aren't delivered until they are ready, no need to wait or check here

        // Set the timestamps in the input buffer to the timestamps in the stored buffer
        const INvSIPLClient::ImageMetaData &md = pNvMBuffer->GetImageData();
        oRawBuffer.frameCaptureTSC = md.frameCaptureTSC;
        oRawBuffer.frameCaptureStartTSC = md.frameCaptureStartTSC;

        // Move buffer from input queue to output queue
        inLck.lock();
        m_inputQueue.pop();
        inLck.unlock();
        std::unique_lock<std::mutex> outLck(m_outputQueueMut);
        if (m_outputQueue.size() >= NUM_CAPTURE_BUFFERS_PER_POOL) {
            LOG_ERR("Buffer output queue is full, not able to move buffer to output queue\n");
            *m_pQuitFlag = true;
            return NVSIPL_STATUS_ERROR;
        }
        m_outputQueue.push(pCurrBuffer);
        outLck.unlock();

        LOG_INFO("Fed frame: %u\n", m_uNumFedFrames++);
        return NVSIPL_STATUS_OK;
    }

private:
    std::atomic<bool> *m_pQuitFlag {nullptr};
    bool m_bStop {false};
    CSemaphore m_currBufferSemaphore {NUM_CAPTURE_BUFFERS_PER_POOL};
    std::queue<INvSIPLClient::INvSIPLBuffer *> m_inputQueue {};
    std::mutex m_inputQueueMut {};
    std::queue<INvSIPLClient::INvSIPLBuffer *> m_outputQueue {};
    std::mutex m_outputQueueMut {};
    uint32_t m_uNumFedFrames {0U};
};

#endif // CFRAMEFEEDER_HPP
