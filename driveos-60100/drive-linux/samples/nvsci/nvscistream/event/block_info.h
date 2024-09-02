/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NvSciStream Event Loop Driven Sample App - block abstraction
 */

#ifndef _BLOCK_INFO_H
#define _BLOCK_INFO_H 1
#include <stdatomic.h>
#include <pthread.h>
#include "nvscistream.h"

/* Maximum number of consumers */
#define MAX_CONSUMERS 4

/* Maximum number of blocks */
#define MAX_BLOCKS 100

/* Maximum number of packets supported */
#define MAX_PACKETS 32

/* Maximum number of elements supported */
#define MAX_ELEMS 8

/* Memory size of endpoint inforamtion */
#define INFO_SIZE 50

/* NvSci modules for all objects */
extern NvSciSyncModule sciSyncModule;
extern NvSciBufModule  sciBufModule;

/* Flag used to terminate the thread that
* was spawned to handle the late/re-attached
* consumer connections upon stream disconnect
*/
extern atomic_int streamDone;

/* Number of registered blocks for streaming */
extern int32_t numBlocks;

/* Number of active blocks */
extern uint32_t numAlive;

/* variables for synchronization */
extern pthread_mutex_t mutex;
extern pthread_cond_t cond;

/* Thread for handling late/re-attached consumer
* connections
*/
extern pthread_t dispatchThread;

/* Common options for all blocks */
typedef struct {
    /* Indicate whether the producer/consumer sets endpoint info */
    bool endInfo;
    /* Indicate whether the producer uses yuv format */
    bool yuv;
    /* Indicate whether the extern event service is used */
    bool useExtEventService;
    /* Indicates the number of late consumers for late/re-attach usecase */
    uint32_t numLateConsumer;
    /* Total number of consumers */
    uint32_t numConsumer;
    /* Indicates c2c usecase */
    bool c2cMode;
    /* Indicates consumer connection is late/reattach*/
    bool lateAttach;
} CommonOptions;

extern CommonOptions opts;

/* Endpoint data structure for tracking the
*  IPC/C2C channels
*/
typedef struct {
    /* Holds the IPC endpoint corresponding to an IPC channel */
    NvSciIpcEndpoint ipcEndpoint;
    /* Holds the C2C endpoint corresponding to an C2C channel */
    NvSciIpcEndpoint c2cEndpoint;
    /* named IPC channel */
    char ipcChannel[32];
    /* named IPC channel used for handsking between
    * producer and late/re-attached consumer connection
    */
    char ipcChannelForHandshake[32];
    /* named c2c channel */
    char c2cChannel[32];
    /* IPC/C2C block created for handling the late/re-attached
    * consumer connections*/
    NvSciStreamBlock ipcBlock;
    /* Queue block that is needed for c2c usecase for a c2c
    * consumer late/reattach connections
    */
    NvSciStreamBlock queue;
    /* ReturnSync block that is needed for c2c usecase for a c2c
    * consumer late/reattach connections
    */
    NvSciStreamBlock returnSync;

    /* Indicates the connect state of IPC channel */
    bool ipcConnected;
    /* Indicates the connect state of C2C channel */
    bool c2cConnected;
    /* Indicates the Open state of IPC channel */
    bool ipcOpened;
    /* Indicates the Open state of C2C channel */
    bool c2cOpened;
    /* QNX channel ID for communication */
    int32_t chid;
    /* QNX channel connection ID */
    int32_t coid;
} Endpoint;

extern Endpoint ipcEP[MAX_CONSUMERS];

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

typedef int32_t (*BlockFunc)(void* data, uint32_t wait);

/* Structure to track block info */
typedef struct {
    NvSciStreamBlock    handle;
    void*               data;
    BlockFunc           func;
    NvSciEventNotifier* notifier;
    bool                isAlive;
    bool                retry;
} BlockEventData;

extern BlockEventData       blocks[MAX_BLOCKS];
extern BlockEventData*      blocksAlive[MAX_BLOCKS];

/* Create data structure for common blocks */
extern BlockData* createCommon(
    char const* name,
    size_t      size);

/* Handle event for common block */
extern int32_t handleCommon(
    void*       data,
    uint32_t    wait);

/* Delete common block */
extern void deleteCommon(
    void*       data);

/*
 * Functions for setting up each kind of block
 */

extern int32_t createIpcDst(
    NvSciStreamBlock* ipcDst,
    const char*       channel,
    bool              useExternalEventService);

extern int32_t createIpcSrc(
    NvSciStreamBlock* ipcSrc,
    const char*       channel,
    bool              useExternalEventService);

extern int32_t createIpcSrc2(
    NvSciStreamBlock* ipcsrc,
    NvSciIpcEndpoint endpoint,
    bool              useExtEventService);

extern int32_t createC2cSrc2(
    NvSciStreamBlock* c2cSrc,
    NvSciIpcEndpoint endpoint,
    NvSciStreamBlock  queue);

extern int32_t createC2cDst(
    NvSciStreamBlock* c2cDst,
    const char*       channel,
    NvSciStreamBlock  pool);

extern int32_t createC2cSrc(
    NvSciStreamBlock* c2cSrc,
    const char*       channel,
    NvSciStreamBlock  queue);

extern int32_t createLimiter(
    NvSciStreamBlock* limiter,
    uint32_t          limit);

extern int32_t createPresentSync(
    NvSciStreamBlock* presentSync);

extern int32_t createReturnSync(
    NvSciStreamBlock* returnSync);

extern int32_t createMulticast(
    NvSciStreamBlock* multicast,
    uint32_t          numConsumer);

extern int32_t createPool_Common(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool);

extern int32_t createQueue(
    NvSciStreamBlock* queue,
    uint32_t          useMailbox);

extern int32_t (*createProducer)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t numFrames);

extern int32_t (*createConsumer)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t frames);

extern int32_t (createProducer_Usecase1)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t          numFrames);

extern int32_t (createConsumer_Usecase1)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t          frames);

extern int32_t (createProducer_Usecase2)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t          numFrames);

extern int32_t (createConsumer_Usecase2)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t          frames);

extern int32_t(createProducer_Usecase3)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool,
    uint32_t          numFrames);

extern int32_t(createConsumer_Usecase3)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index,
    uint32_t          frames);

extern int32_t createPool_Usecase3(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool);

extern void* handleLateConsumerThreadFunc(void*);

#endif // _BLOCK_INFO_H
