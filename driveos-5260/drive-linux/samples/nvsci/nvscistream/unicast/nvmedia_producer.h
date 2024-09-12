//
// NvMedia Producer client declaration.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef NVMEDIA_PRODUCER_H
#define NVMEDIA_PRODUCER_H

#include "client_common.h"

// nvmedia includes
#include "nvmedia_core.h"
#include "nvmedia_common.h"
#include "nvmedia_image.h"
#include "nvmedia_image_nvscibuf.h"
#include "nvmedia_2d.h"
#include "nvmedia_2d_nvscisync.h"

namespace NvScistreamSample
{
    class NvMediaProducer:
        public ClientCommon
    {
    public:
        NvMediaProducer() = delete;
        NvMediaProducer(NvSciStreamBlock blockHandle,
                        uint32_t numSyncs = NUM_PROD_SYNCS);
        virtual ~NvMediaProducer(void);

        // Buffer setup functions
        virtual void createBufAttrList(NvSciBufModule bufModule);
        virtual void mapPacket(
            NvSciStreamCookie &cookie,
            NvSciStreamPacket packtHandle);
        virtual void mapBuffers(NvSciStreamCookie cookie);
        virtual void unmapBuffers(NvSciStreamCookie cookie);

        // Sync object setup functions
        virtual void createSyncAttrLists(NvSciSyncModule syncModule);
        virtual void reconcileAndAllocSyncObjs(void);
        virtual void mapSyncObjs(void);
        virtual void initFences(NvSciSyncModule syncModule);
        virtual void unmapSyncObjs(void);

        // Event handling
        void handleResourceSetupEvents(NvSciStreamEvent &event);

        // Streaming functions
        void getPacket(NvSciStreamCookie &cookie);
        virtual void processPayload(NvSciStreamCookie cookie);
        void sendPacket(NvSciStreamCookie cookie);

    private:
        NvSciSyncCpuWaitContext waitContext;
        uint32_t numPacketsPresented;

        NvMediaDevice      *nvmdevice;
        NvMedia2D          *nvm2d;
        NvSciBufObj        bufobj_in[NUM_PACKETS];
        NvMediaImage       *nvmimg_in[NUM_PACKETS];
        NvSciBufObj        bufObj[NUM_PACKETS][NUM_ELEMENTS_PER_PACKET];
        NvMediaImage       *nvmimage[NUM_PACKETS][NUM_ELEMENTS_PER_PACKET];

        void setupBuffers(NvSciBufModule bufModule);
    };
}

#endif
