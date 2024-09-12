//
// Pool manager client declaration.
//
// Copyright (c) 2019 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef PACKET_POOL_MGR_H
#define PACKET_POOL_MGR_H

#include "nvscistream.h"
#include "util.h"

namespace NvScistreamSample
{
    class PacketPoolManager {
    public:
        PacketPoolManager(NvSciStreamBlock blockHandle);
        virtual ~PacketPoolManager(void);

        // Buffer setup functions
        void recvPacketAttrs(NvSciStreamEvent &event);
        bool recvAllPacketAttrs(void);
        virtual void reconcilePacketAttrs(void);
        void sendReconciledPacketAttrs(void);
        virtual void allocBuffers(NvSciStreamCookie &cookie);
        void sendPacket(NvSciStreamCookie cookie);
        void reconcileAndMapPackets(void);
        // void recvPacketStatus(void);
        void recvPacketStatus(NvSciStreamEvent &event);
        bool recvAllPacketStatus(void);

    protected:
        // Decide the cookie for the new packet
        inline NvSciStreamCookie assignPacketCookie(void)
        {
            NvSciStreamCookie cookie =
                static_cast<NvSciStreamCookie>(numPacketsCreated + 1U);
            if (cookie == 0U) {
                LOG_ERR_EXIT("invalid cookie assignment");
            }
            return cookie;
        }
        inline Packet* getPacketByCookie(NvSciStreamCookie cookie)
        {
            if (cookie == 0U) {
                LOG_ERR_EXIT("invalid cookie request");
            }
            uint32_t id = static_cast<uint32_t>(cookie) - 1U;
            return &packets[id];
        }

        NvSciStreamBlock        handle;

        // Producer packet element attributue
        uint32_t                numProdElements;
        uint32_t                numProdElementsRecv;
        PacketElementAttr      *prodElementAttrs;

        // Consumer packet element attributue
        uint32_t                numConsElements;
        uint32_t                numConsElementsRecv;
        PacketElementAttr      *consElementAttrs;

        // Reconciled packet element atrribute
        uint32_t                numElements;
        PacketElementAttr      *elementAttrs;

        // Packet element descriptions
        uint32_t                numPackets;
        Packet                  packets[NUM_PACKETS];
        uint32_t                numPacketsCreated;

        // Packet + Element Status
        uint32_t numProdPacketStatusRecv;
        uint32_t numConsPacketStatusRecv;
        uint32_t numProdElementStatusRecv;
        uint32_t numConsElementStatusRecv;
    };
}

#endif
