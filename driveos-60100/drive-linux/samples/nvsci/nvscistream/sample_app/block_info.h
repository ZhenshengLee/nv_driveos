/* NvSciStream Safety Sample App - block abstraction
 *
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _BLOCK_INFO_H
#define _BLOCK_INFO_H 1

#include <pthread.h>
#include "nvscistream.h"

/* Maximum number of consumers */
#define MAX_CONSUMERS 4

/* Maximum number of packets supported */
#define MAX_PACKETS 32

/* Maximum number of elements supported */
#define MAX_ELEMS 4

/* Memory size of endpoint inforamtion */
#define INFO_SIZE 50

/* NvSci modules for all objects */
extern NvSciSyncModule sciSyncModule;
extern NvSciBufModule  sciBufModule;

/* NvSciIpc Endpoint */
extern NvSciIpcEndpoint ipcsrcEndpoint;
extern NvSciIpcEndpoint ipcdstEndpoint;

/* Options for producer */
typedef struct {
    uint32_t block;
    bool isProxyApp;
} ProdArgs;

typedef struct {
    uint32_t block;
    bool isProxyApp;
} ConsArgs;

typedef struct {
    uint32_t block;
    uint32_t numPacket;
    bool isC2cPool;
} PoolArgs;

/* Options for each consumer */
typedef struct {
    uint32_t numPacket;
    uint32_t numCons;
    uint32_t limit;
    int32_t timeout;
    char c2csrcChannel[32];
    char c2cdstChannel[32];
} TestArgs;
extern TestArgs testArgs;
/* Structure to track packet element attributes */
typedef struct {
    /* The application's name for the element */
    uint32_t         userName;
    /* Attribute list for element */
    NvSciBufAttrList attrList;
} ElemAttr;

/*
 * Some block types that do not require direct interaction will share a
 *   common private data structure and event handling functon.
 */

/* Common block private data */
typedef struct {
    NvSciStreamBlock  block;
    int64_t           waitTime;
    char              name[32];
    void            (*deleteFunc)(void*);
} BlockData;

extern int32_t handleQMProxyProcess (void);
extern int32_t handleQMProcess (void);
extern int32_t handleASILProcess (void);
extern void* handleProducer(void *args);
extern void* handleConsumer(void *args);
extern void* handlePool(void *args);
extern NvSciStreamBlock producer;
extern NvSciStreamBlock consumer;
extern NvSciStreamBlock staticPool;
extern NvSciStreamBlock queue[2];
extern NvSciStreamBlock ipcSrc[2];
extern NvSciStreamBlock c2cSrc;
extern NvSciStreamBlock multicast;
extern NvSciStreamBlock c2cDst;
extern NvSciStreamBlock ipcDst[2];
extern NvSciStreamBlock returnSync;
extern NvSciStreamBlock limiter;

#endif // _BLOCK_INFO_H
