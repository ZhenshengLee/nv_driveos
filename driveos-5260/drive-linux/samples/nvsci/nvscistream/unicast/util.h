//
// Utility functions
//
// Copyright (c) 2019-2020 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef UTIL_H
#define UTIL_H

#include "log.h"
#include "constants.h"
#include <map>

namespace NvScistreamSample
{
    // Define attributes of a packet element.
    typedef struct {
        uint32_t userData;
        NvSciStreamElementMode syncMode;
        NvSciBufAttrList bufAttrList;
    } PacketElementAttr;

    // Define Packet struct which is used by the client
    typedef struct {
        // The client's handle for the packet.
        NvSciStreamCookie cookie;
        // The NvSciStream's Handle for the packet
        NvSciStreamPacket handle;
        // An array of elements/buffers in the packet
        NvSciBufObj      *buffers;
        // An array of pre-allocated prefences for this packet.
        NvSciSyncFence    fences[MAX_NUM_SYNCS];
    } Packet;

    // Define end point info for ipc channel
    typedef struct {
        // channel name
        char chname[15];
        // NvIPC handle
        NvSciIpcEndpoint endpoint;
        // channel info
        NvSciIpcEndpointInfo info;
    } Endpoint;

    // Fill in CPU signaler/waiter attribute list
    inline void setCpuSyncAttrList(
        NvSciSyncAccessPerm cpuPerm,
        NvSciSyncAttrList attrList)
    {
        NvSciSyncAttrKeyValuePair keyValue[2];
        bool cpuSync = true;
        keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
        keyValue[0].value = (void*)&cpuSync;
        keyValue[0].len = sizeof(cpuSync);
        keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
        keyValue[1].value = (void*)&cpuPerm;
        keyValue[1].len = sizeof(cpuPerm);
        CHECK_NVSCIERR(NvSciSyncAttrListSetAttrs(attrList, keyValue, 2));
    }

    inline void freeSyncObjs(
        NvSciSyncObj* syncObjs,
        uint32_t count)
    {
        if (syncObjs != nullptr) {
            for (uint32_t i = 0U; i < count; i++) {
                if (syncObjs[i] != nullptr) {
                    NvSciSyncObjFree(syncObjs[i]);
                }
            }
            free(syncObjs);
            syncObjs = nullptr;
        }
    }

    inline void freeElementAttrs(
        PacketElementAttr* attrs,
        uint32_t count)
    {
        if (attrs != nullptr) {
            for (uint32_t i = 0U; i < count; i++) {
                if (attrs[i].bufAttrList != nullptr) {
                    NvSciBufAttrListFree(attrs[i].bufAttrList);
                    attrs[i].bufAttrList = nullptr;
                }
            }
            free(attrs);
            attrs = nullptr;
        }
    }

}

#endif
