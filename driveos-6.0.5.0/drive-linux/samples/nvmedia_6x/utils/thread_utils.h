/*
 * Copyright (c) 2021, NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_THREAD_UTILS_H_
#define _NVMEDIA_TEST_THREAD_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "nvmedia_core.h"

#define NV_TIMEOUT_INFINITE 0xFFFFFFFF
#define NV_THREAD_PRIORITY_NORMAL 0

typedef void NvEvent;
typedef void NvMutex;
typedef void NvThread;
typedef void NvSemaphore;
typedef void NvQueue;

NvMediaStatus NvMutexCreate(NvMutex **ppMutex);
NvMediaStatus NvMutexDestroy(NvMutex *pMutex);
NvMediaStatus NvMutexAcquire(NvMutex *pMutex);
NvMediaStatus NvMutexRelease(NvMutex *pMutex);

NvMediaStatus NvThreadCreate(NvThread **ppThread, uint32_t (*pFunc)(void* pParam), void* pParam, int sPriority);
NvMediaStatus NvThreadPriorityGet(NvThread *pThread, int *psPriority);
NvMediaStatus NvThreadPrioritySet(NvThread *pThread, int sPriority);
NvMediaStatus NvThreadNameSet(NvThread *pThread, char *name);
NvMediaStatus NvThreadDestroy(NvThread *pThread);
NvMediaStatus NvThreadYield(void);
int           NvThreadGetPid(NvThread *pThread);

NvMediaStatus NvEventCreate(NvEvent **ppEvent, int bManual, int bSet);
NvMediaStatus NvEventWait(NvEvent *pEvent, uint32_t uTimeoutMs);
NvMediaStatus NvEventSet(NvEvent *pEvent);
NvMediaStatus NvEventReset(NvEvent *pEvent);
NvMediaStatus NvEventDestroy(NvEvent *pEvent);

NvMediaStatus NvSemaphoreCreate(NvSemaphore** ppSemaphore, uint32_t uInitCount, uint32_t uMaxCount);
NvMediaStatus NvSemaphoreIncrement(NvSemaphore *pSem);
NvMediaStatus NvSemaphoreDecrement(NvSemaphore *pSem, uint32_t uTimeoutMs);
NvMediaStatus NvSemaphoreDestroy(NvSemaphore *pSem);

NvMediaStatus NvQueueCreate(NvQueue **ppQueue, uint32_t uQueueSize, uint32_t uItemSize);
NvMediaStatus NvQueueDestroy(NvQueue *pQueue);
NvMediaStatus NvQueueGet(NvQueue *pQueue, void *pItem, uint32_t uTimeout);
NvMediaStatus NvQueuePeek(NvQueue *pQueue, void *pItem, uint32_t *puItems);
NvMediaStatus NvQueuePut(NvQueue *pQueue, void *pItem, uint32_t uTimeout);
NvMediaStatus NvQueuePutFront(NvQueue *pQueueApp, void *pItem, uint32_t uTimeout);
NvMediaStatus NvQueueGetSize(NvQueue *pQueue, uint32_t *puSize);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_THREAD_UTILS_H_ */
