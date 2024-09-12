//
// Pool manager client declaration.
//
// Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef PACKET_POOL_MGR_H
#define PACKET_POOL_MGR_H

#include "nvscistream.h"
#include "constants.h"
#include "log.h"

// Define attributes of a packet element.
typedef struct {
    uint32_t userData;
    NvSciStreamElementMode syncMode;
    NvSciBufAttrList bufAttrList = nullptr;
} PacketElementAttr;

// Define Packet struct which is used by the pool
typedef struct {
    // The client's handle for the packet.
    NvSciStreamCookie cookie = 0U;
    // The NvSciStream's handle for the packet
    NvSciStreamPacket handle = 0U;
    // An array of elements/buffers in the packet
    NvSciBufObj      *buffers = nullptr;
} PoolPacket;

class PacketPoolManager
{
public:
    PacketPoolManager(NvSciStreamBlock blockHandle);
    ~PacketPoolManager(void);

    // Buffer setup functions
    void recvPacketAttrs(NvSciStreamEvent &event);
    bool recvAllPacketAttrs(void);
    void reconcilePacketAttrs(void);
    void sendReconciledPacketAttrs(void);
    void allocBuffers(NvSciStreamCookie &cookie);
    void sendPacket(NvSciStreamCookie cookie);
    void reconcileAndMapPackets(void);
    void recvPacketStatus(NvSciStreamEvent &event);
    bool recvAllPacketStatus(void);

protected:
    // Decide the cookie for the new packet
    inline NvSciStreamCookie assignPacketCookie(void)
    {
        NvSciStreamCookie cookie =
            static_cast<NvSciStreamCookie>(numPackets + 1U);
        if (cookie == 0U) {
            LOG_ERR_EXIT("Invalid cookie assignment");
        }
        return cookie;
    }
    inline PoolPacket* getPacketByCookie(NvSciStreamCookie cookie)
    {
        if (cookie == 0U) {
            LOG_ERR_EXIT("Invalid cookie request");
        }
        uint32_t id = static_cast<uint32_t>(cookie) - 1U;
        return &packets[id];
    }

    NvSciStreamBlock    handle = 0U;

    // Producer packet element attributue
    uint32_t            numProdElements = 0U;
    uint32_t            numProdElementsRecv = 0U;
    PacketElementAttr  *prodElementAttrs = nullptr;

    // Consumer packet element attributue
    uint32_t            numConsElements = 0U;
    uint32_t            numConsElementsRecv = 0U;
    PacketElementAttr  *consElementAttrs = nullptr;

    // Reconciled packet element atrribute
    uint32_t            numElements = 0U;
    PacketElementAttr  *elementAttrs = nullptr;

    // Packet element descriptions
    uint32_t            numPackets = 0U;
    PoolPacket          packets[NUM_PACKETS];

    // Packet + Element Status
    uint32_t            numProdPacketStatusRecv = 0U;
    uint32_t            numConsPacketStatusRecv = 0U;
    uint32_t            numProdElementStatusRecv = 0U;
    uint32_t            numConsElementStatusRecv = 0U;
};

#endif
