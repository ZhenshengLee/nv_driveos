//
// NvSciStream Common Client usage
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CLIENT_COMMON_H
#define CLIENT_COMMON_H

#include "nvscistream.h"
#include "util.h"

namespace NvScistreamSample
{
    constexpr NvSciStreamCookie cookieBase = 0xC00C1E4U;

    class ClientCommon {
    public:
        ClientCommon() = delete;
        ClientCommon(NvSciStreamBlock blockHandle, uint32_t numSyncs);
        virtual ~ClientCommon(void);

        // Buffer setup functions
        virtual void createBufAttrList(NvSciBufModule bufModule) = 0;
        void sendPacketAttr(void);
        void recvReconciledPacketElementCount(uint32_t count);
        void recvReconciledPacketAttr(NvSciStreamEvent &event);
        // void recvReconciledPacketAttr(void);
        void recvPacket(NvSciStreamEvent &event,
                        NvSciStreamPacket &packetHandle);
        virtual void mapPacket(
            NvSciStreamCookie &cookie,
            NvSciStreamPacket packetHandle);
        void registerPacket(
            NvSciStreamCookie cookie,
            NvSciError status = NvSciError_Success);
        void recvPacketElement(NvSciStreamEvent &event);
        // void recvPacketElements(NvSciStreamCookie cookie);
        void registerPacketElements(
            NvSciStreamCookie cookie,
            NvSciError status = NvSciError_Success);
        bool allPacketInfoReceived(void);
        virtual void mapBuffers(NvSciStreamCookie cookie) = 0;
        virtual void unmapBuffers(NvSciStreamCookie cookie) = 0;

        // Sync objects setup functons
        virtual void createSyncAttrLists(NvSciSyncModule syncModule) = 0;
        void sendSyncAttr(void);
        void recvSyncObjAttrs(NvSciStreamEvent &event);
        void validateNumSyncObjs(void);
        virtual void reconcileAndAllocSyncObjs(void);
        void sendSyncObjs(void);
        void recvSyncObjCount(uint32_t count);
        void recvSyncObj(NvSciStreamEvent &event);
        // void recvSyncObjs(void);
        bool allSyncInfoReceived(void);
        virtual void mapSyncObjs(void) = 0;
        virtual void initFences(NvSciSyncModule syncModule) = 0;
        virtual void unmapSyncObjs(void) = 0;

        uint32_t getNumRecvSyncObjs() { return numRecvSyncObjs; }

        static inline NvSciStreamCookie getCookieAtIndex(uint32_t id)
        {
            if (id >= NUM_PACKETS) {
                LOG_ERR_EXIT("invalid packet index");
            }
            return (cookieBase + static_cast<NvSciStreamCookie>(id + 1U));
        }

        static inline uint32_t getIndexFromCookie(NvSciStreamCookie cookie)
        {
            if (cookie <= cookieBase) {
                LOG_ERR_EXIT("invalid cookie assignment");
            }
            uint32_t id = static_cast<uint32_t>(cookie - cookieBase) - 1U;
            return id;
        }

    protected:
        // Decide the cookie for the new packet
        inline NvSciStreamCookie assignPacketCookie(void)
        {
            NvSciStreamCookie cookie = cookieBase +
                static_cast<NvSciStreamCookie>(numPackets);
            if (cookie <= cookieBase || numPackets == 0U) {
                // no packet was received. So this is invalid
                LOG_ERR_EXIT("invalid cookie assignment");
            }
            return cookie;
        }
        inline uint32_t packetCookie2Id(const NvSciStreamCookie& cookie)
        {
            if (cookie <= cookieBase || numPackets == 0U) {
                LOG_ERR_EXIT("invalid cookie assignment");
            }

            return static_cast<uint32_t>(cookie - cookieBase) - 1U;
        }

        inline Packet* getPacketByCookie(const NvSciStreamCookie& cookie)
        {
            uint32_t id = packetCookie2Id(cookie);
            return &(packets[id]);
        }

        NvSciStreamBlock        handle;

        // sync objects
        bool                    hasRecvSyncAttr;
        bool                    hasRecvSyncCount;

        NvSciSyncAttrList       signalerAttrList;
        NvSciSyncAttrList       waiterAttrList;

        NvSciSyncAttrList       recvSyncAttrList;

        uint32_t                numSyncObjs = 0U;
        NvSciSyncObj            syncObjs[MAX_NUM_SYNCS];

        uint32_t                numRecvSyncObjs;
        uint32_t                numSyncObjsRecvd;
        NvSciSyncObj           *waiterSyncObjs;
        NvSciSyncFence         *prefences;

        // packet elements (buffer)
        uint32_t                numElements;
        NvSciBufAttrList        bufAttrLists[NUM_ELEMENTS_PER_PACKET];

        uint32_t                numReconciledElements;
        uint32_t                numReconciledElementsRecvd;
        PacketElementAttr      *reconciledElementAttrs;

        // packets
        uint32_t                numPackets;
        Packet                  packets[NUM_PACKETS];
    };
}

#endif
