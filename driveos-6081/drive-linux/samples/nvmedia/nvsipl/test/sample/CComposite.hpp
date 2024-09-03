/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "nvmedia_2d_sci.h"
#include "NvSIPLClient.hpp" // NvSIPL Client header for input buffers

#include "nvscibuf.h"
#include "CImageManager.hpp"
#include "CWFDConsumer.hpp"

#ifndef CCOMPOSITE_HPP
#define CCOMPOSITE_HPP

using namespace nvsipl;

class CComposite
{
  public:
    // Destructor
    virtual ~CComposite()
    {
        if (m_bRunning) {
            Stop();
            Deinit();
        }
    }
    // Initializes compositor
    SIPLStatus Init(NvSciBufModule& bufModule,
                    NvSciSyncModule& syncModule,
                    CImageManager* pImageManager,
                    std::vector<uint32_t>& vSensorIds);
    SIPLStatus FillNvSciSyncAttrList(NvSciSyncAttrList &attrList,
                                     NvMediaNvSciSyncClientType clientType);
    SIPLStatus RegisterNvSciSyncObj(uint32_t uSensorId,
                                    NvMediaNvSciSyncObjType syncObjType,
                                    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> syncObj);
    SIPLStatus Post(uint32_t sensorId, INvSIPLClient::INvSIPLNvMBuffer *pBuffer);
    SIPLStatus Start();
    SIPLStatus Stop();
    virtual SIPLStatus Deinit();

  protected:
    virtual void ThreadFunc(void);
    virtual bool CheckInputQueues(void);

  protected:
    struct DestroyNvMedia2DDevice
    {
        void operator ()(NvMedia2D *p) const
        {
            NvMedia2DDestroy(p);
        }
    };


  struct ReleaseNvMBuffer {
      void operator()(INvSIPLClient::INvSIPLNvMBuffer *pNvMBuffer) const {
          if (pNvMBuffer != nullptr) {
              SIPLStatus status = pNvMBuffer->Release();
              if (status != NVSIPL_STATUS_OK) {
                  LOG_ERR("Buffer release failed\n");
              }
          }
      }
  };

  private:
    SIPLStatus UnregisterNvSciSyncObjs(void);
    SIPLStatus RegisterImages(void);
    SIPLStatus UnregisterImages(void);
    SIPLStatus ComputeInputRects(uint32_t uWidth, uint32_t uHeight);
    SIPLStatus FenceWait(void);

    std::unique_ptr<NvMedia2D, DestroyNvMedia2DDevice> m_up2DDevice;

    NvMediaRect m_oInputRects[MAX_NUM_SENSORS];

    std::unique_ptr<CWFDConsumer> m_upWfdConsumer = nullptr;
    // Thread stuff
    std::unique_ptr<std::thread> m_upthread {nullptr};
    std::atomic<bool> m_bRunning; // Flag indicating if compositor is running.

    std::mutex m_vInputQueueMutex[MAX_NUM_SENSORS];
    std::queue<INvSIPLClient::INvSIPLNvMBuffer*> m_vInputQueue[MAX_NUM_SENSORS];
    uint32_t m_uDstBufferIndex = 0U;
    std::vector<NvSciBufObj> m_vDstBuffer;
    std::unique_ptr<NvSciSyncObj, CloseNvSciSyncObj> m_syncObjs[MAX_NUM_SENSORS];
    std::vector<uint32_t> m_vSensorIds;
    CImageManager* m_pImageManager;

    bool m_dstBufState[BUFFER_NUM][MAX_NUM_SENSORS] {{false}};
    std::unique_ptr<INvSIPLClient::INvSIPLNvMBuffer, ReleaseNvMBuffer> m_bufStorage[MAX_NUM_SENSORS];

    NvSciSyncCpuWaitContext m_cpuWaitContext;
    NvSciSyncFence m_fence = NvSciSyncFenceInitializer;
};

#endif // CCOMPOSITE_HPP