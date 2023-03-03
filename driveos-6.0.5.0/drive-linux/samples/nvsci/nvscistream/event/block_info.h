/* NvSciStream Event Loop Driven Sample App - block abstraction
 *
 * Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _BLOCK_INFO_H
#define _BLOCK_INFO_H 1

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
extern NvSciIpcEndpoint ipcEndpoint;

/* Common options for all blocks */
typedef struct {
    bool endInfo;
} CommonOptions;

extern CommonOptions opts;

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
    const char*       channel);

extern int32_t createIpcSrc(
    NvSciStreamBlock* ipcSrc,
    const char*       channel);

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

extern int32_t createPool(
    NvSciStreamBlock* pool,
    uint32_t          numPacket,
    bool              isC2cPool);

extern int32_t createQueue(
    NvSciStreamBlock* queue,
    uint32_t          useMailbox);

extern int32_t (*createProducer)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool);

extern int32_t (*createConsumer)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index);

extern int32_t (createProducer_Usecase1)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool);

extern int32_t (createConsumer_Usecase1)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index);

extern int32_t (createProducer_Usecase2)(
    NvSciStreamBlock* producer,
    NvSciStreamBlock  pool);

extern int32_t (createConsumer_Usecase2)(
    NvSciStreamBlock* consumer,
    NvSciStreamBlock  queue,
    uint32_t          index);


#endif // _BLOCK_INFO_H
