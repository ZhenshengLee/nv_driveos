//
// NvMedia Producer client definition.
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <string.h>
#include "nvmedia_producer.h"

namespace NvScistreamSample
{
    NvMediaProducer::NvMediaProducer(NvSciStreamBlock blockHandle, uint32_t numSyncs) :
        ClientCommon(blockHandle, numSyncs),
        waitContext(nullptr),
        numPacketsPresented(0U),
        nvmdevice(nullptr),
        nvm2d(nullptr)
    {
        nvmdevice = NvMediaDeviceCreate();
        if (nvmdevice == NULL) {
            LOG_ERR_EXIT("Failed to create NvMedia device");
        }

        nvm2d = NvMedia2DCreate(nvmdevice);
        if (nvm2d == NULL) {
            LOG_ERR_EXIT("Failed to create 2D NvMedia object");
        }

        LOG_DEBUG("NvMedia producer created");
    }

    NvMediaProducer::~NvMediaProducer(void)
    {
        if (waitContext != nullptr) {
            NvSciSyncCpuWaitContextFree(waitContext);
        }

        for (uint32_t i = 0; i < numPackets; i++) {
            NvSciStreamCookie cookie = getCookieAtIndex(i);
            unmapBuffers(cookie);
        }
        unmapSyncObjs();

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            NvMedia2DUnregisterNvSciSyncObj(nvm2d, syncObjs[i]);
        }
        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            NvMediaImageDestroy(nvmimg_in[i]);
            NvSciBufObjFree(bufobj_in[i]);
        }
        NvMediaImageNvSciBufDeinit();
        NvMedia2DDestroy(nvm2d);
        NvMediaDeviceDestroy(nvmdevice);
    }

    // Buffer setup functions
    void NvMediaProducer::createBufAttrList(NvSciBufModule bufModule)
    {
        // Setup NvMedia buffets
        setupBuffers(bufModule);

        // create attr requirements
        for (uint32_t i = 0U; i < numElements; i++) {
            CHECK_NVSCIERR(NvSciBufAttrListCreate(bufModule,
                                                  &bufAttrLists[i]));
            LOG_DEBUG("Create NvSciBuf attribute list of element " << i << ".");

            NvSciBufAttrList attrList = bufAttrLists[i];
            NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
            NvSciBufAttrKeyValuePair attrKvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                                &access_perm,
                                                sizeof(access_perm)};
            NvMediaSurfaceType   nvmsurfType;
            NVM_SURF_FMT_DEFINE_ATTR(surfFormatAttrs);
            NvMediaSurfAllocAttr surfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
            uint32_t numSurfAllocAttrs = 0;

            // Create YUV 422 PL images with cpu mapping pointer
            NVM_SURF_FMT_SET_ATTR_YUV(surfFormatAttrs, YUYV, 422, PACKED, UINT, 8, PL);

            nvmsurfType = NvMediaSurfaceFormatGetType(surfFormatAttrs,
                            NVM_SURF_FMT_ATTR_MAX);

            surfAllocAttrs[0].type = NVM_SURF_ATTR_WIDTH;
            surfAllocAttrs[0].value = WIDTH;
            surfAllocAttrs[1].type = NVM_SURF_ATTR_HEIGHT;
            surfAllocAttrs[1].value = HEIGHT;
            surfAllocAttrs[2].type = NVM_SURF_ATTR_CPU_ACCESS;
            surfAllocAttrs[2].value = NVM_SURF_ATTR_CPU_ACCESS_CACHED;
            numSurfAllocAttrs = 3;

            CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(attrList, &attrKvp,1));

            CHECK_NVMEDIAERR(NvMediaImageFillNvSciBufAttrs(nvmdevice,
                                                           nvmsurfType,
                                                           surfAllocAttrs,
                                                           numSurfAllocAttrs,
                                                           0,
                                                           attrList));

            LOG_DEBUG("Set attribute value of element " << i << ".");
        }
    }

    // Map the received packet in its own space and assigns cookie to it.
    void NvMediaProducer::mapPacket(NvSciStreamCookie &cookie,
                                   NvSciStreamPacket packetHandle)
    {
        ClientCommon::mapPacket(cookie, packetHandle);
    }

    // Create client buffer objects from NvSciBufObj
    void NvMediaProducer::mapBuffers(NvSciStreamCookie cookie)
    {
        uint32_t id = ClientCommon::getIndexFromCookie(cookie);
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("Invalid packet for cookie.");
        }

        uint8_t *mapped_ptr = NULL;
        NvMediaImageSurfaceMap surfaceMap;
        uint32_t size = 0U;

        for (uint32_t i = 0U; i < numReconciledElements; i++) {
            LOG_DEBUG("Producer Mapping element: " << i << " of cookie " << std::hex << cookie << ".");
            CHECK_NVSCIERR(NvSciBufObjDup(packet->buffers[i], &bufObj[id][i]));
            CHECK_NVMEDIAERR(NvMediaImageCreateFromNvSciBuf(nvmdevice,
                                                            bufObj[id][i],
                                                            &nvmimage[id][i]));

            CHECK_NVMEDIAERR(NvMediaImageLock(nvmimage[id][i], NVMEDIA_IMAGE_ACCESS_WRITE, &surfaceMap));
            // Clear buffers
            // This code assumes YUV 422 PL images with cpu mapping pointers are used in the test
            mapped_ptr = (uint8_t*)surfaceMap.surface[0].mapping;
            size = surfaceMap.surface[0].pitch * surfaceMap.height;
            (void) memset(mapped_ptr, 0, size);
            NvMediaImageUnlock(nvmimage[id][i]);
        }
    }

    // destroy client buffer objects created from NvSciBufObj
    void NvMediaProducer::unmapBuffers(NvSciStreamCookie cookie)
    {
        uint32_t id = ClientCommon::getIndexFromCookie(cookie);

        for (uint32_t i = 0U; i < numReconciledElements; i++) {
            NvMediaImageDestroy(nvmimage[id][i]);
            NvSciBufObjFree(bufObj[id][i]);
        }
    }

    // Create and set CPU signaler and waiter attribute lists.
    void NvMediaProducer::createSyncAttrLists(NvSciSyncModule syncModule)
    {
        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &signalerAttrList));
        LOG_DEBUG("Create signaler's sync attribute list.");
        CHECK_NVMEDIAERR(NvMedia2DFillNvSciSyncAttrList(nvm2d, signalerAttrList, NVMEDIA_SIGNALER));
        LOG_DEBUG("Set nvmedia-signaler attribute value.");

        CHECK_NVSCIERR(NvSciSyncAttrListCreate(syncModule, &waiterAttrList));
        LOG_DEBUG("Create waiter's sync attribute list.");
        CHECK_NVMEDIAERR(NvMedia2DFillNvSciSyncAttrList(nvm2d, waiterAttrList, NVMEDIA_WAITER));
        LOG_DEBUG("Set nvmedia-waiter attribute value.");
    }

    void NvMediaProducer::handleResourceSetupEvents(NvSciStreamEvent &event)
    {
        switch (event.type)
        {
            case NvSciStreamEventType_SyncAttr:
                recvSyncObjAttrs(event);
                LOG_DEBUG("Producer received producer attributes");
                reconcileAndAllocSyncObjs();
                LOG_DEBUG("Producer reconciled producer attributes and creates sync object:\n"
                        "\tsending sync objects to consumer...");
                sendSyncObjs();
                break;

            case NvSciStreamEventType_SyncCount:
                recvSyncObjCount(event.count);
                break;

            case NvSciStreamEventType_SyncDesc:
                recvSyncObj(event);
                break;

            case NvSciStreamEventType_PacketElementCount:
                recvReconciledPacketElementCount(event.count);
                break;

            case NvSciStreamEventType_PacketAttr:
                recvReconciledPacketAttr(event);
                break;

            case NvSciStreamEventType_PacketCreate:
            {
                NvSciStreamCookie cookie = 0U;
                NvSciStreamPacket packetHandle;
                recvPacket(event, packetHandle);
                mapPacket(cookie, packetHandle);
                registerPacket(cookie);
            }
                break;

            case NvSciStreamEventType_PacketElement:
                recvPacketElement(event);
                break;

            default:
                break;
        }
    }

    // Reconciles its own sync object attribute and the received sync
    // object object attribute. Then it recates a sync object based on the
    // reconciled attribute list.
    void NvMediaProducer::reconcileAndAllocSyncObjs(void)
    {
        ClientCommon::reconcileAndAllocSyncObjs();

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            CHECK_NVMEDIAERR(NvMedia2DRegisterNvSciSyncObj(nvm2d, NVMEDIA_EOFSYNCOBJ, syncObjs[i]));
        }
        CHECK_NVMEDIAERR(NvMedia2DSetNvSciSyncObjforEOF(nvm2d, syncObjs[0]));
    }

    // Map the recived sync object into its own space
    void NvMediaProducer::mapSyncObjs(void)
    {
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            CHECK_NVMEDIAERR(NvMedia2DRegisterNvSciSyncObj(nvm2d, NVMEDIA_PRESYNCOBJ, waiterSyncObjs[i]));
        }
    }

    // unmap the recived sync objects
    void NvMediaProducer::unmapSyncObjs(void)
    {
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            CHECK_NVMEDIAERR(NvMedia2DUnregisterNvSciSyncObj(nvm2d, waiterSyncObjs[i]));
        }
    }

    // Producer receives PacketReady event and gets the next available
    //  packet to write.
    void NvMediaProducer::getPacket(NvSciStreamCookie &cookie)
    {
        NvSciStreamEvent event;
        uint32_t timeouts = 0U;

        while (true) {
            NvSciError err = NvSciStreamBlockEventQuery(handle, QUERY_TIMEOUT, &event);
            if (err == NvSciError_Success) {
                if (event.type != NvSciStreamEventType_PacketReady) {
                    LOG_ERR_EXIT("Producer Failed to receive PACKET_READY.");
                }
                LOG_DEBUG("Producer received PACKET_READY event.");

                // Clear prefences
                for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                    NvSciSyncFenceClear(&prefences[i]);
                }

                cookie = 0U;
                CHECK_NVSCIERR(NvSciStreamProducerPacketGet(handle,
                                                            &cookie,
                                                            prefences));
                if (cookie == 0U) {
                    LOG_ERR_EXIT("Producer Failed to get a packet from pool.");
                }
                LOG_DEBUG("Producer obtained a packet (cookie = " << std::hex << cookie << ") from pool.");

                // Assign prefences value to the corresponding packet
                Packet *packet = getPacketByCookie(cookie);
                if (packet == nullptr) {
                    LOG_ERR_EXIT("Invalid packet from cookie.");
                }

                for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
                    NvSciSyncFenceDup(&prefences[i], &packet->fences[i]);
                    NvSciSyncFenceClear(&prefences[i]);
                }

                break;
            } else if (err == NvSciError_Timeout) {
                // if query timeouts - keep waiting for event
                // until wait threshold is reached
                if (timeouts < MAX_QUERY_TIMEOUTS) {
                    timeouts++;
                } else {
                    LOG_ERR_EXIT("Producer Query waits seem to be taking forever!");
                }
            } else {
                LOG_ERR("NvSciStreamBlockEventQuery Failed:");
                CHECK_NVSCIERR(err);
            }
        }
    }

    // Init the fence object.
    void NvMediaProducer::initFences(NvSciSyncModule syncModule)
    {

    }

    // Producer waits for prefences, write buffer data, and then generates
    //  postfences.
    void NvMediaProducer::processPayload(NvSciStreamCookie cookie)
    {
        LOG_DEBUG("Process payload (cookie = " << std::hex << cookie << ").");

        uint32_t id = ClientCommon::getIndexFromCookie(cookie);
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("Invalid packet from cookie.");
        }

        // insert for prefences
        for (uint32_t i = 0U; i < numRecvSyncObjs; i++) {
            CHECK_NVMEDIAERR(NvMedia2DInsertPreNvSciSyncFence(nvm2d, &packet->fences[i]));
            NvSciSyncFenceClear(&packet->fences[i]);
        }

        CHECK_NVMEDIAERR(NvMedia2DBlitEx(nvm2d,
                                         nvmimage[id][0],
                                         NULL,
                                         nvmimg_in[id],
                                         NULL,
                                         NULL,
                                         NULL));

        CHECK_NVMEDIAERR(NvMedia2DGetEOFNvSciSyncFence(nvm2d,
                                                       syncObjs[0],
                                                       &packet->fences[0]));
        for (uint32_t i = 1U; i < numSyncObjs; i++) {
            NvSciSyncFenceClear(&packet->fences[i]);
        }
    }

    // Producer sends a packet to the consumer.
    void NvMediaProducer::sendPacket(NvSciStreamCookie cookie)
    {
        // Get the buffer by producer cookie.
        Packet *packet = getPacketByCookie(cookie);
        if (packet == nullptr) {
            LOG_ERR_EXIT("Invalid packet from cookie.");
        }

        CHECK_NVSCIERR(NvSciStreamProducerPacketPresent(handle,
                                                        packet->handle,
                                                        packet->fences));

        for (uint32_t i = 0U; i < numSyncObjs; i++) {
            NvSciSyncFenceClear(&packet->fences[i]);
        }

        LOG_DEBUG("Send the packet (cookie = " << packet->cookie << ", handle = " << packet->handle << ").");
    }

    void NvMediaProducer::setupBuffers(NvSciBufModule bufModule)
    {
        NvMediaSurfaceType   nvmsurfType;
        NVM_SURF_FMT_DEFINE_ATTR(surfFormatAttrs);
        NvMediaSurfAllocAttr surfAllocAttrs[NVM_SURF_ALLOC_ATTR_MAX];
        uint32_t numSurfAllocAttrs = 0;
        NvMediaImageSurfaceMap inp_surfaceMap;
        uint8_t *mapped_ptr = NULL;

        NvSciBufAttrList unreconciled_attrlist = NULL;
        NvSciBufAttrList reconciled_attrlist = NULL;
        NvSciBufAttrList conflictlist = NULL;
        NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attr_kvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                            &access_perm,
                                            sizeof(access_perm)};

        // Create YUV 422 PL images with cpu mapping pointer
        NVM_SURF_FMT_SET_ATTR_YUV(surfFormatAttrs, YUYV, 422, PACKED, UINT, 8, PL);

        nvmsurfType = NvMediaSurfaceFormatGetType(surfFormatAttrs,
                        NVM_SURF_FMT_ATTR_MAX);

        surfAllocAttrs[0].type = NVM_SURF_ATTR_WIDTH;
        surfAllocAttrs[0].value = WIDTH;
        surfAllocAttrs[1].type = NVM_SURF_ATTR_HEIGHT;
        surfAllocAttrs[1].value = HEIGHT;
        surfAllocAttrs[2].type = NVM_SURF_ATTR_CPU_ACCESS;
        surfAllocAttrs[2].value = NVM_SURF_ATTR_CPU_ACCESS_CACHED;
        numSurfAllocAttrs = 3;


        // CHECK_NVSCIERR(NvSciBufModuleOpen(&module));
        CHECK_NVSCIERR(NvSciBufAttrListCreate(bufModule, &unreconciled_attrlist));
        CHECK_NVSCIERR(NvSciBufAttrListSetAttrs(unreconciled_attrlist, &attr_kvp, 1));

        CHECK_NVMEDIAERR(NvMediaImageNvSciBufInit());
        CHECK_NVMEDIAERR(NvMediaImageFillNvSciBufAttrs(nvmdevice,
                                                       nvmsurfType,
                                                       surfAllocAttrs,
                                                       numSurfAllocAttrs,
                                                       0,
                                                       unreconciled_attrlist));
        CHECK_NVSCIERR(NvSciBufAttrListReconcile(&unreconciled_attrlist,
                                                 1,
                                                 &reconciled_attrlist,
                                                 &conflictlist));

        for (uint32_t i = 0U; i < NUM_PACKETS; i++) {
            CHECK_NVSCIERR(NvSciBufObjAlloc(reconciled_attrlist, &bufobj_in[i]));

            CHECK_NVMEDIAERR(NvMediaImageCreateFromNvSciBuf(nvmdevice,
                                                            bufobj_in[i],
                                                            &nvmimg_in[i]));

            CHECK_NVMEDIAERR(NvMediaImageLock(nvmimg_in[i],
                                              NVMEDIA_IMAGE_ACCESS_WRITE,
                                              &inp_surfaceMap));

            // Fill predefined data
            uint32_t nWidth = 4U;
            // This code assumes YUV 422 PL images with cpu mapping pointers
            mapped_ptr = (uint8_t*)inp_surfaceMap.surface[0].mapping;
            for (uint32_t y = 0U; y < inp_surfaceMap.height; y++) {
                for (uint32_t x = 0U; x < inp_surfaceMap.width * nWidth; x++) {
                    mapped_ptr[x] = (i + ((x % 32)+ (y % 32) )) % (1 << 8);
                }
                mapped_ptr += inp_surfaceMap.surface[0].pitch;
            }
            NvMediaImageUnlock(nvmimg_in[i]);
        }

        if (reconciled_attrlist != NULL) {
            NvSciBufAttrListFree(reconciled_attrlist);
        }
        if (unreconciled_attrlist != NULL) {
            NvSciBufAttrListFree(unreconciled_attrlist);
        }
        if (conflictlist != NULL) {
            NvSciBufAttrListFree(conflictlist);
        }
    }
}
