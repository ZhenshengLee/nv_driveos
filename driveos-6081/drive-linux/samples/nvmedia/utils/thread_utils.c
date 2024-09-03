/*
 * Copyright (c) 2013-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */
#define _GNU_SOURCE

#include <dirent.h>
#include <fnmatch.h>

#include <sched.h>
#include <pthread.h>
#include "os_common.h"
#include <sys/types.h>
#include <unistd.h>

#if !defined(NVMEDIA_GHSI)
#include <sys/resource.h>
#include <sys/time.h>

#include <sys/wait.h>

#if !defined(NVMEDIA_QNX)
#include <sys/syscall.h>
#endif

#include <time.h>

#endif

#include "thread_utils.h"
#include "log_utils.h"

typedef struct tagNvThread {
    uint32_t (*pFunc)(void*);
    pthread_cond_t      condition;
    pthread_mutex_t     mutex;
    pthread_mutexattr_t mutexattr;
    void*               pParam;
    pthread_t           thread;
    pthread_attr_t      thread_attr;
    pid_t               pid;
    int                 priority;
#ifdef NVMEDIA_GHSI
    Task                hTask;
#endif

    int                 schedPolicy;
    int                 schedPriorityMin;
    int                 schedPriorityMax;
    int                 schedPriorityBase;
} NvThreadCtx;

typedef struct tagNvEvent {
    pthread_cond_t      condition;
    pthread_mutex_t     mutex;
    int                 signaled;
    int                 manual;
} NvEventCtx;

typedef struct tagNvMutexData {
    pthread_mutexattr_t mutexattr;
    pthread_mutex_t     mutex;
} NvMutexCtx;

typedef struct tagNvSemaphore {
    pthread_cond_t      condition;
    pthread_mutex_t     mutex;
    uint32_t            maxCount;
    uint32_t            count;
} NvSemaphoreCtx;

typedef struct tagNvQueue {
    uint32_t            uNextGet;
    uint32_t            uNextPut;
    uint8_t            *pQueueData;
    NvSemaphore        *pSemGet;
    NvSemaphore        *pSemPut;
    NvMutex            *pMutex;
    uint32_t            uItems;
    uint32_t            uItemSize;
    uint32_t            uQueueSize;
} NvQueueCtx;

static void CalculateTimeoutTime(uint32_t uTimeoutMs, struct timespec *pTimeOut);
static void *ThreadFunc(void *pParam);

NvMediaStatus NvMutexCreate(NvMutex **ppMutexApp)
{
    NvMutexCtx *pMutex = (NvMutexCtx *)malloc(sizeof(NvMutexCtx));
    int iReturnCode = 0;

    *ppMutexApp = 0;
    if(pMutex){
        memset(pMutex, 0, sizeof(NvMutexCtx));

        iReturnCode = pthread_mutexattr_init(&pMutex->mutexattr);
        if (iReturnCode) {
            LOG_ERR("pthread_mutexattr_init failed (%d)", iReturnCode);
            free(pMutex);
            return NVMEDIA_STATUS_ERROR;
        }

#ifndef NVMEDIA_QNX
        // iReturnCode = pthread_mutexattr_settype(&pMutex->mutexattr, PTHREAD_MUTEX_RECURSIVE_NP);
        if (iReturnCode) {
            LOG_ERR("pthread_mutexattr_settype failed (%d)", iReturnCode);
            pthread_mutexattr_destroy(&pMutex->mutexattr);
            free(pMutex);
            return NVMEDIA_STATUS_ERROR;
        }
#endif
        iReturnCode = pthread_mutex_init(&pMutex->mutex, &pMutex->mutexattr);
        if (iReturnCode) {
            pthread_mutexattr_destroy(&pMutex->mutexattr);
            LOG_ERR("pthread_mutex_init failed (%d)", iReturnCode);
            free(pMutex);
            return NVMEDIA_STATUS_ERROR;
        }

        *ppMutexApp = (NvMutex *)pMutex;
        return NVMEDIA_STATUS_OK;
    }
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvMutexDestroy(NvMutex *pMutexApp)
{
    NvMutexCtx *pMutex = (NvMutexCtx *)pMutexApp;
    int iReturnCode1 = 0, iReturnCode2 = 0;

    if(pMutex){
        iReturnCode1 = pthread_mutex_destroy(&pMutex->mutex);
        if(iReturnCode1){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode1);
        }
        iReturnCode2 = pthread_mutexattr_destroy(&pMutex->mutexattr);
        if(iReturnCode2){
            LOG_ERR("pthread_mutexattr_destroy failed (%d)", iReturnCode2);
        }
        free(pMutex);
        return (!iReturnCode1 && !iReturnCode2) ? NVMEDIA_STATUS_OK : NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvMutexAcquire(NvMutex *pMutexApp)
{
    NvMutexCtx *pMutex = (NvMutexCtx *)pMutexApp;
    int iReturnCode = 0;

    if(pMutex){
        iReturnCode = pthread_mutex_lock(&pMutex->mutex);
        if (iReturnCode) {
            LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
        }
        return iReturnCode ? NVMEDIA_STATUS_ERROR : NVMEDIA_STATUS_OK;
    }
    LOG_ERR("Invalid handle failed (%d)", iReturnCode);
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvMutexRelease(NvMutex *pMutexApp)
{
    NvMutexCtx *pMutex = (NvMutexCtx *)pMutexApp;
    int iReturnCode = 0;

    if(pMutex){
        iReturnCode = pthread_mutex_unlock(&pMutex->mutex);
        if (iReturnCode) {
            LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        }
        return iReturnCode ? NVMEDIA_STATUS_ERROR : NVMEDIA_STATUS_OK;
    }
    LOG_ERR("Invalid handle failed (%d)", iReturnCode);
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvEventCreate(NvEvent **ppEventApp, int bManual, int bSet)
{
    int iReturnCode = 0;
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_CLOCK_SELECTION) && _POSIX_CLOCK_SELECTION >= 0
    pthread_condattr_t oCondAttr;
#endif
    pthread_condattr_t *pCondAttr = 0;
    *ppEventApp = 0;

    NvEventCtx *pEvent = (NvEventCtx *)malloc(sizeof(NvEventCtx));
    if (!pEvent) {
        return NVMEDIA_STATUS_ERROR;
    }

    pEvent->manual = bManual;

    iReturnCode = pthread_mutex_init(&pEvent->mutex, 0);
    if (iReturnCode) {
        LOG_ERR("pthread_mutex_init failed (%d)", iReturnCode);
        free(pEvent);
        return NVMEDIA_STATUS_ERROR;
    }

    // If monotonic clock and clock selection are supported
    // then create condattr and set the clock type to CLOCK_MONOTONIC
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_CLOCK_SELECTION) && _POSIX_CLOCK_SELECTION >= 0
    iReturnCode = pthread_condattr_init(&oCondAttr);
    if (iReturnCode) {
        LOG_ERR("pthread_condattr_init failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_destroy(&pEvent->mutex);
        if (iReturnCode){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        free(pEvent);
        return NVMEDIA_STATUS_ERROR;
    }
    iReturnCode = pthread_condattr_setclock(&oCondAttr, CLOCK_MONOTONIC);
    if (iReturnCode) {
        LOG_ERR("pthread_condattr_setclock failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_destroy(&pEvent->mutex);
        if (iReturnCode){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        free(pEvent);
        return NVMEDIA_STATUS_ERROR;
    }
    pCondAttr = &oCondAttr;
#endif

    iReturnCode = pthread_cond_init(&pEvent->condition, pCondAttr);
    if (iReturnCode) {
        LOG_ERR("pthread_cond_init failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_destroy(&pEvent->mutex);
        if (iReturnCode){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        free(pEvent);
        return NVMEDIA_STATUS_ERROR;
    }

    *ppEventApp = (NvEvent *)pEvent;

    if (bSet) {
        NvEventSet(pEvent);
    }
    else {
        NvEventReset(pEvent);
    }
    return NVMEDIA_STATUS_OK;
}

#if !defined(NVMEDIA_QNX) && !defined(NVMEDIA_ANDROID)
#ifndef  NVMEDIA_GHSI
#if !defined(__GLIBC__) || !defined(__GLIBC_PREREQ) || !__GLIBC_PREREQ(2, 30)
static pid_t gettid(void)
{
    return (pid_t)syscall(__NR_gettid);
}
#endif
#else
static pid_t gettid(void)
{
    return getpid();
}
#endif
#endif

static void *ThreadFunc(void *pParam)
{
    NvThreadCtx *pThread = (NvThreadCtx *)(pParam);
    int iReturnCode = 0, iReturnCode2 = 0;

    if (pThread->pid == 0){
        iReturnCode = pthread_mutex_lock(&pThread->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
            return NULL;
        }
        pThread->pid = gettid();
#ifdef NVMEDIA_GHSI
        pThread->hTask = CurrentTask();
#endif
        iReturnCode = pthread_cond_signal(&pThread->condition);
        if(iReturnCode){
            LOG_ERR("pthread_cond_signal failed (%d)", iReturnCode);
            iReturnCode2 = pthread_mutex_unlock(&pThread->mutex);
            if(iReturnCode2){
                LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode2);
                return NULL;
            }
            return NULL;
        }
        iReturnCode = pthread_mutex_unlock(&pThread->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
            return NULL;
        }
    }
    (*pThread->pFunc)(pThread->pParam);
    return NULL;
}

int NvThreadGetPid(NvThread *pThreadApp)
{
    NvThreadCtx *pThread = (NvThreadCtx *)pThreadApp;
    return (int)pThread->pid;
}

NvMediaStatus NvThreadCreate(NvThread **ppThreadApp, uint32_t (*pFunc)(void *pParam), void *pParam, int sPriority)
{
    struct sched_param oSched;
    NvThreadCtx *pThread = (NvThreadCtx *)calloc(1, sizeof(NvThreadCtx));
    int iReturnCode = 0;

    if(pThread){
        pThread->pFunc = pFunc;
        pThread->pParam = pParam;
        pThread->pid = 0;

        iReturnCode = pthread_getschedparam(pthread_self(), &pThread->schedPolicy, &oSched);
        if (iReturnCode == 0){
            if (pThread->schedPolicy == SCHED_OTHER) {
#ifdef NVMEDIA_GHSI
                // sched_get_XXX are unsupported, using integrity's static limits
                pThread->schedPriorityBase = oSched.sched_priority;
                pThread->schedPriorityMin = 0;
                pThread->schedPriorityMax = 255;
#else
                pThread->schedPriorityBase = getpriority(PRIO_PROCESS, 0);
                pThread->schedPriorityMin = -20;
                pThread->schedPriorityMax =  19;
#endif
            } else {
                pThread->schedPriorityBase = oSched.sched_priority;
                pThread->schedPriorityMin = sched_get_priority_min(pThread->schedPolicy);
                pThread->schedPriorityMax = sched_get_priority_max(pThread->schedPolicy);
            }
        } else {
            LOG_ERR("pthread_getschedparam failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }

        iReturnCode = pthread_attr_init(&pThread->thread_attr);

        if(iReturnCode){
            LOG_ERR("pthread_attr_init failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }
#ifndef NVMEDIA_ANDROID
        iReturnCode = pthread_attr_setinheritsched(&pThread->thread_attr, PTHREAD_INHERIT_SCHED);

        if(iReturnCode){
            LOG_ERR("pthread_attr_setinheritsched failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }
#endif
        /* Additional 32K stack size on top of PTHREAD_STACK_MIN */
        iReturnCode = pthread_attr_setstacksize(&pThread->thread_attr, (PTHREAD_STACK_MIN + 0x1C000));
        if(iReturnCode){
            LOG_ERR("pthread_attr_setstacksize failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }

        iReturnCode = pthread_mutex_init(&pThread->mutex, 0);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_init failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }

        iReturnCode = pthread_cond_init(&pThread->condition, 0);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_init failed (%d)", iReturnCode);
            free(pThread);
            return NVMEDIA_STATUS_ERROR;
        }

        *ppThreadApp = (NvThread *)pThread;

        iReturnCode = pthread_create(&pThread->thread, &pThread->thread_attr, ThreadFunc, (void*)pThread);
        if (iReturnCode) {
            LOG_ERR("pthread_create failed (%d)", iReturnCode);
            free(pThread);
            *ppThreadApp = NULL;
            return NVMEDIA_STATUS_ERROR;
        }

        iReturnCode = pthread_mutex_lock(&pThread->mutex);
        if(iReturnCode){
            free(pThread);
            *ppThreadApp = NULL;
            LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
            return NVMEDIA_STATUS_ERROR;
        }

        while (pThread->pid == 0){
            iReturnCode = pthread_cond_wait(&pThread->condition, &pThread->mutex);
            if(iReturnCode){
                free(pThread);
                *ppThreadApp = NULL;
                LOG_ERR("pthread_cond_wait failed (%d)", iReturnCode);
                return NVMEDIA_STATUS_ERROR;
            }
        }
        iReturnCode = pthread_mutex_unlock(&pThread->mutex);
        if(iReturnCode){
            free(pThread);
            *ppThreadApp = NULL;
            LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
            return NVMEDIA_STATUS_ERROR;
        }
        NvThreadPrioritySet(pThread, sPriority);
        return NVMEDIA_STATUS_OK;
    }
    return NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvThreadPriorityGet(NvThread *pThreadApp, int *psPriority)
{
    NvThreadCtx *pThread = (NvThreadCtx *)pThreadApp;
    int iSchedPolicy;
    int iReturnCode;
    struct sched_param oSched;

    if(!psPriority)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    if (pThread->pid == 0)
         return NVMEDIA_STATUS_ERROR;

    iReturnCode = pthread_getschedparam(pThread->thread, &iSchedPolicy, &oSched);
    if (iReturnCode){
        LOG_ERR("pthread_getschedparam failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    if (iSchedPolicy == -1) {
        LOG_ERR("sched_getscheduler failed (errno=%d)", errno);
        return NVMEDIA_STATUS_ERROR;
    }

    if (iSchedPolicy == SCHED_OTHER) {
#ifndef NVMEDIA_GHSI
        *psPriority = getpriority(PRIO_PROCESS, pThread->pid);
#else
        {
            Error E;
            Value TP;
            E = GetTaskPriority(pThread->hTask, &TP);
            if (E) {
                LOG_MSG("GetTaskPriority failed (code=%d)", E);
                return NVMEDIA_STATUS_ERROR;
            } else
            {
                *psPriority = (int)TP;
            }
        }
#endif
    } else {
        *psPriority = oSched.sched_priority;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvThreadPrioritySet(NvThread *pThreadApp, int sPriority)
{
    NvThreadCtx *pThread = (NvThreadCtx *)pThreadApp;
    int iSchedPolicy = -1;
    int iReturnCode;
    struct sched_param oSched;

    if (pThread->pid == 0)
         return NVMEDIA_STATUS_ERROR;

    iReturnCode = pthread_getschedparam(pThread->thread, &iSchedPolicy, &oSched);
    if (iReturnCode){
        LOG_ERR("pthread_getschedparam failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    if (iSchedPolicy == -1) {
        LOG_ERR("pthread_getschedparam failed (errno=%d)", errno);
        return NVMEDIA_STATUS_ERROR;
    }

    if (iSchedPolicy == SCHED_OTHER)
        pThread->priority = pThread->schedPriorityBase - sPriority;
    else
        pThread->priority = pThread->schedPriorityBase + sPriority;

    // Ensure priority is within limits
    if (pThread->priority < pThread->schedPriorityMin)
        pThread->priority = pThread->schedPriorityMin;
    else if (pThread->priority > pThread->schedPriorityMax)
        pThread->priority = pThread->schedPriorityMax;

    if (iSchedPolicy == SCHED_OTHER){
#ifndef NVMEDIA_GHSI
        iReturnCode = setpriority(PRIO_PROCESS, pThread->pid, pThread->priority);
#else
        iReturnCode = (int)SetTaskPriority(pThread->hTask, pThread->priority, false);
#endif
        if (iReturnCode){
            LOG_ERR("setpriority failed (%d)", iReturnCode);
            return NVMEDIA_STATUS_ERROR;
        }
    } else {
        struct sched_param oSched;
        oSched.sched_priority = pThread->priority;
        iReturnCode = pthread_setschedparam(pThread->thread, iSchedPolicy, &oSched);
        if (iReturnCode){
            LOG_ERR("pthread_setschedparam failed (%d)", iReturnCode);
            return NVMEDIA_STATUS_ERROR;
        }
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvThreadNameSet(NvThread *pThreadApp, char *name)
{
    int iReturnCode;
    NvThreadCtx *pThread = (NvThreadCtx *)pThreadApp;

    if (pThread->pid == 0)
         return NVMEDIA_STATUS_ERROR;
#ifdef NVMEDIA_GHSI
    iReturnCode = (int)SetTaskName(pThread->hTask, name, strlen(name));
#else
    iReturnCode = pthread_setname_np(pThread->thread, name);
#endif
    if (iReturnCode != 0) {
        LOG_ERR("pthread_setname_np failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvThreadDestroy(NvThread *pThreadApp)
{
    NvThreadCtx *pThread = (NvThreadCtx *)pThreadApp;
    int iReturnCode1 = 0, iReturnCode2 = 0, iReturnCode3 = 0;

    if (pthread_join(pThread->thread, 0)) {
        return NVMEDIA_STATUS_ERROR;
    }
    iReturnCode1 = pthread_attr_destroy(&pThread->thread_attr);
    if(iReturnCode1){
        LOG_ERR("pthread_attr_destroy failed (%d)", iReturnCode1);
    }
    iReturnCode2 = pthread_cond_destroy(&pThread->condition);
    if(iReturnCode2){
        LOG_ERR("pthread_cond_destroy failed (%d)", iReturnCode2);
    }
    iReturnCode3 = pthread_mutex_destroy(&pThread->mutex);
    if(iReturnCode3){
        LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode3);
    }
    free(pThread);
    return (!iReturnCode1 && !iReturnCode2 && !iReturnCode3) ? NVMEDIA_STATUS_OK : NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvThreadYield(void)
{
    int iReturnCode;

    iReturnCode = sched_yield();
    if(iReturnCode) {
        LOG_ERR("sched_yield: %d\n", errno);
        perror("sched_yield");
    }
    return (!iReturnCode) ? NVMEDIA_STATUS_OK : NVMEDIA_STATUS_ERROR;
}


static uint64_t GetClock(void)
{
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0
    struct timespec tv;
#else
    struct timeval tv;
#endif

    // Get current time
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_MONOTONIC_CLOCK) && _POSIX_MONOTONIC_CLOCK >= 0 && _POSIX_TIMERS > 0
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (uint64_t)tv.tv_sec * (uint64_t)1000000 + (uint64_t)(tv.tv_nsec / 1000);
#else
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * (uint64_t)1000000 + (uint64_t)tv.tv_usec;
#endif
}

void CalculateTimeoutTime(uint32_t uTimeoutMs, struct timespec *pTimeOut)
{
    uint64_t currentTimeuSec;

    // Get current time
    currentTimeuSec = GetClock();

    // Add Timeout (in micro seconds)
    currentTimeuSec += (uTimeoutMs * 1000);

    // Split to Seconds and Nano Seconds
    pTimeOut->tv_sec  = currentTimeuSec / 1000000;
    pTimeOut->tv_nsec = (currentTimeuSec % 1000000) * 1000;
}

NvMediaStatus NvEventWait(NvEvent *pEventApp, uint32_t uTimeoutMs)
{
    struct timespec timeout;
    NvEventCtx    *pEvent = (NvEventCtx *)pEventApp;
    int iReturnCode = 0;

    CalculateTimeoutTime(uTimeoutMs, &timeout);

    iReturnCode = pthread_mutex_lock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }

    if (uTimeoutMs == 0) {
        if (pEvent->signaled) {
            if (!pEvent->manual) {
                pEvent->signaled = 0;
            }
        } else {
            iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
            if(iReturnCode){
                LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
            }
            return NVMEDIA_STATUS_ERROR;
        }
    } else if (uTimeoutMs == NV_TIMEOUT_INFINITE) {
        while (!pEvent->signaled) {
            iReturnCode = pthread_cond_wait(&pEvent->condition, &pEvent->mutex);
            if(iReturnCode){
                LOG_ERR("pthread_cond_wait failed (%d)", iReturnCode);
            }
        }
        if (!pEvent->manual) {
            pEvent->signaled = 0;
        }
    } else {
        while (!pEvent->signaled) {
            iReturnCode = pthread_cond_timedwait(&pEvent->condition, &pEvent->mutex, &timeout);
            if (iReturnCode == ETIMEDOUT) {
                iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
                if(iReturnCode){
                    LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                }
                return NVMEDIA_STATUS_TIMED_OUT;
            } else if(iReturnCode) {
                LOG_ERR("pthread_cond_timedwait failed (%d)", iReturnCode);
                iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
                if(iReturnCode){
                    LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                }
                return NVMEDIA_STATUS_ERROR;
            }
        }
        if (!pEvent->manual) {
            pEvent->signaled = 0;
        }
    }
    iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvEventSet(NvEvent *pEventApp)
{
    NvEventCtx *pEvent = (NvEventCtx *)pEventApp;
    int iReturnCode = 0;

    iReturnCode = pthread_mutex_lock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    pEvent->signaled = 1;
    iReturnCode = pthread_cond_signal(&pEvent->condition);
    if(iReturnCode){
        LOG_ERR("pthread_cond_signal failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        }
        return NVMEDIA_STATUS_ERROR;
    }
    iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvEventReset(NvEvent *pEventApp)
{
    NvEventCtx *pEvent = (NvEventCtx *)pEventApp;
    int iReturnCode = 0;

    iReturnCode = pthread_mutex_lock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    pEvent->signaled = 0;
    iReturnCode = pthread_mutex_unlock(&pEvent->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvEventDestroy(NvEvent *pEventApp)
{
    NvEventCtx *pEvent = (NvEventCtx *)pEventApp;
    int iReturnCode1 = 0, iReturnCode2 = 0, iReturnCode3 = 0, iReturnCode4 = 0;

    iReturnCode1 = pthread_mutex_lock(&pEvent->mutex);
    if(iReturnCode1){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode1);
    }

    iReturnCode2 = pthread_cond_destroy(&pEvent->condition);
    if(iReturnCode2){
        LOG_ERR("pthread_cond_destroy failed (%d)", iReturnCode2);
    }

    iReturnCode3 = pthread_mutex_unlock(&pEvent->mutex);
    if(iReturnCode3){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode3);
    }

    iReturnCode4 = pthread_mutex_destroy(&pEvent->mutex);
    if(iReturnCode4){
        LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode4);
    }
    free(pEvent);
    return (!iReturnCode1 && !iReturnCode2 && !iReturnCode3 && !iReturnCode4) ? NVMEDIA_STATUS_OK : NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvSemaphoreCreate(NvSemaphore **ppSemaphoreApp, uint32_t uInitCount, uint32_t uMaxCount)
{
    NvSemaphoreCtx *pSem = (NvSemaphoreCtx *)malloc(sizeof(NvSemaphoreCtx));
    int iReturnCode = 0;
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_CLOCK_SELECTION) && _POSIX_CLOCK_SELECTION >= 0
    pthread_condattr_t oCondAttr;
#endif
    pthread_condattr_t *pCondAttr = 0;

    *ppSemaphoreApp = 0;
    if(!pSem) {
        return NVMEDIA_STATUS_ERROR;
    }

    if (uInitCount > uMaxCount) {
        uInitCount = uMaxCount;
    }

    pSem->maxCount = uMaxCount;
    pSem->count = uInitCount;

    iReturnCode = pthread_mutex_init(&pSem->mutex, 0);
    if (iReturnCode) {
        LOG_ERR("pthread_mutex_init failed (%d)", iReturnCode);
        free(pSem);
        return NVMEDIA_STATUS_ERROR;
    }

// If monotonic clock and clock selection are supported
// then create condattr and set the clock type to CLOCK_MONOTONIC
#if defined(CLOCK_MONOTONIC) && defined(_POSIX_CLOCK_SELECTION) && _POSIX_CLOCK_SELECTION >= 0
    iReturnCode = pthread_condattr_init(&oCondAttr);
    if (iReturnCode) {
        LOG_ERR("pthread_condattr_init failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_destroy(&pSem->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        free(pSem);
        return NVMEDIA_STATUS_ERROR;
    }
    iReturnCode = pthread_condattr_setclock(&oCondAttr, CLOCK_MONOTONIC);
    if (iReturnCode) {
        LOG_ERR("pthread_condattr_setclock failed (%d)", iReturnCode);
        iReturnCode = pthread_mutex_destroy(&pSem->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        free(pSem);
        return NVMEDIA_STATUS_ERROR;
    }
    pCondAttr = &oCondAttr;
#endif

    iReturnCode = pthread_cond_init(&pSem->condition, pCondAttr);
    if (iReturnCode) {
        if(pthread_mutex_destroy(&pSem->mutex)){
            LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode);
        }
        LOG_ERR("pthread_cond_init failed (%d)", iReturnCode);
        free(pSem);
        return NVMEDIA_STATUS_ERROR;
    }

    *ppSemaphoreApp = (NvSemaphore *)pSem;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvSemaphoreIncrement(NvSemaphore *pSemApp)
{
    NvSemaphoreCtx *pSem = (NvSemaphoreCtx *)pSemApp;
    int iReturnCode = 0;

    iReturnCode = pthread_mutex_lock(&pSem->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    pSem->count++;
    if(pSem->count > pSem->maxCount) {
        pSem->count = pSem->maxCount;
    } else {
        iReturnCode = pthread_cond_broadcast(&pSem->condition);
        if(iReturnCode){
            LOG_ERR("pthread_cond_broadcast failed (%d)", iReturnCode);
            iReturnCode = pthread_mutex_unlock(&pSem->mutex);
            if(iReturnCode){
                LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
            }
            return NVMEDIA_STATUS_ERROR;
        }
    }
    iReturnCode = pthread_mutex_unlock(&pSem->mutex);
    if(iReturnCode){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        return NVMEDIA_STATUS_ERROR;
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvSemaphoreDecrement(NvSemaphore *pSemApp, uint32_t uTimeoutMs)
{
    struct timespec timeout;
    NvSemaphoreCtx *pSem = (NvSemaphoreCtx *)pSemApp;
    int iReturnCode = 0;

    CalculateTimeoutTime(uTimeoutMs, &timeout);

    while (1) {
        iReturnCode = pthread_mutex_lock(&pSem->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode);
            return NVMEDIA_STATUS_ERROR;
        }
        if (pSem->count > 0) {
           pSem->count--;
           iReturnCode = pthread_mutex_unlock(&pSem->mutex);
           if(iReturnCode){
                LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                return NVMEDIA_STATUS_ERROR;
           }
           break;
        }

        if (uTimeoutMs == 0) {
            iReturnCode = pthread_mutex_unlock(&pSem->mutex);
            if(iReturnCode){
                LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
            }
            return NVMEDIA_STATUS_ERROR;
        } else if (uTimeoutMs == NV_TIMEOUT_INFINITE) {
            iReturnCode = pthread_cond_wait(&pSem->condition, &pSem->mutex);
            if(iReturnCode){
                iReturnCode = pthread_mutex_unlock(&pSem->mutex);
                if(iReturnCode){
                    LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                }
                LOG_ERR("pthread_cond_wait failed (%d)", iReturnCode);
                return NVMEDIA_STATUS_ERROR;
           }
        } else {
            iReturnCode = pthread_cond_timedwait(&pSem->condition, &pSem->mutex, &timeout);
            if (iReturnCode == ETIMEDOUT) {
                iReturnCode = pthread_mutex_unlock(&pSem->mutex);
                if(iReturnCode){
                    LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                }
                return NVMEDIA_STATUS_TIMED_OUT;
            } else if(iReturnCode){
                LOG_ERR("pthread_cond_timedwait failed (%d)", iReturnCode);
                iReturnCode = pthread_mutex_unlock(&pSem->mutex);
                if(iReturnCode){
                    LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
                }
                return NVMEDIA_STATUS_ERROR;
            }
        }
        iReturnCode = pthread_mutex_unlock(&pSem->mutex);
        if(iReturnCode){
            LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode);
        }
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvSemaphoreDestroy(NvSemaphore *pSemApp)
{
    NvSemaphoreCtx *pSem = (NvSemaphoreCtx *)pSemApp;
    int iReturnCode1 = 0, iReturnCode2 = 0, iReturnCode3 = 0, iReturnCode4 = 0;

    iReturnCode1 = pthread_mutex_lock(&pSem->mutex);
    if(iReturnCode1){
        LOG_ERR("pthread_mutex_lock failed (%d)", iReturnCode1);
    }

    iReturnCode2 = pthread_cond_destroy(&pSem->condition);
    if(iReturnCode2){
        LOG_ERR("pthread_cond_destroy failed (%d)", iReturnCode2);
    }

    iReturnCode3 = pthread_mutex_unlock(&pSem->mutex);
    if(iReturnCode3){
        LOG_ERR("pthread_mutex_unlock failed (%d)", iReturnCode3);
    }

    iReturnCode4 = pthread_mutex_destroy(&pSem->mutex);
    if(iReturnCode4){
        LOG_ERR("pthread_mutex_destroy failed (%d)", iReturnCode4);
    }
    free(pSem);
    return (!iReturnCode1 && !iReturnCode2 && !iReturnCode3 && !iReturnCode4) ? NVMEDIA_STATUS_OK : NVMEDIA_STATUS_ERROR;
}

NvMediaStatus NvQueueCreate(NvQueue **ppQueueApp, uint32_t uQueueSize, uint32_t uItemSize)
{
    NvQueueCtx *pQueue =(NvQueueCtx *)malloc(sizeof(NvQueueCtx));
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;
    *ppQueueApp = pQueue;
    if(pQueue) {
        memset(pQueue, 0, sizeof(NvQueueCtx));
        pQueue->uNextPut        = 0;
        pQueue->uNextGet        = 0;
        pQueue->uQueueSize      = uQueueSize;
        pQueue->uItemSize       = uItemSize;
        pQueue->uItems          = 0;
        pQueue->pQueueData      = malloc(uQueueSize * uItemSize);
        nr = NvSemaphoreCreate(&pQueue->pSemGet, 0, uQueueSize);
        nr = NvSemaphoreCreate(&pQueue->pSemPut, uQueueSize, uQueueSize);
        nr = NvMutexCreate(&pQueue->pMutex);
    }
    return nr;
}

NvMediaStatus NvQueueDestroy(NvQueue *pQueueApp)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    if(pQueue) {
        free(pQueue->pQueueData);
        NvSemaphoreDestroy(pQueue->pSemGet);
        NvSemaphoreDestroy(pQueue->pSemPut);
        NvMutexDestroy(pQueue->pMutex);
        free(pQueue);
    }
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus NvQueueGet(NvQueue *pQueueApp, void *pItem, uint32_t uTimeout)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;

    if(pQueue) {
        nr = NvSemaphoreDecrement(pQueue->pSemGet, uTimeout);
        if(NVMEDIA_STATUS_OK == nr) {
            nr = NvMutexAcquire(pQueue->pMutex);
            pQueue->uNextGet = (pQueue->uNextGet + 1) % pQueue->uQueueSize;
            pQueue->uItems--;
            memcpy(pItem, pQueue->pQueueData + (pQueue->uNextGet * pQueue->uItemSize), pQueue->uItemSize);
            nr = NvMutexRelease(pQueue->pMutex);
            nr = NvSemaphoreIncrement(pQueue->pSemPut);
        }
    }
    return nr;
}

NvMediaStatus NvQueuePut(NvQueue *pQueueApp, void *pItem, uint32_t uTimeout)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;

    if(pQueue) {
        nr = NvSemaphoreDecrement(pQueue->pSemPut, uTimeout);
        if(NVMEDIA_STATUS_OK == nr) {
            nr = NvMutexAcquire(pQueue->pMutex);
            pQueue->uNextPut = (pQueue->uNextPut + 1) % pQueue->uQueueSize;
            memcpy(pQueue->pQueueData + (pQueue->uNextPut * pQueue->uItemSize), pItem, pQueue->uItemSize);
            pQueue->uItems++;
            nr = NvMutexRelease(pQueue->pMutex);
            nr = NvSemaphoreIncrement(pQueue->pSemGet);
        }
    }
    return nr;
}

NvMediaStatus NvQueuePutFront(NvQueue *pQueueApp, void *pItem, uint32_t uTimeout)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;

    if(pQueue) {
        nr = NvSemaphoreDecrement(pQueue->pSemPut, uTimeout);
        if(NVMEDIA_STATUS_OK == nr) {
            nr = NvMutexAcquire(pQueue->pMutex);
            memcpy(pQueue->pQueueData + (pQueue->uNextGet * pQueue->uItemSize), pItem, pQueue->uItemSize);
            pQueue->uNextGet = (pQueue->uNextGet - 1) % pQueue->uQueueSize;
            pQueue->uItems++;
            nr = NvMutexRelease(pQueue->pMutex);
            nr = NvSemaphoreIncrement(pQueue->pSemGet);
        }
    }
    return nr;
}

NvMediaStatus NvQueueGetSize(NvQueue *pQueueApp, uint32_t *puSize)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;

    if(pQueue) {
        nr = NvMutexAcquire(pQueue->pMutex);
        *puSize = pQueue->uItems;
        nr = NvMutexRelease(pQueue->pMutex);
    }
    return nr;
}


NvMediaStatus NvQueuePeek(NvQueue *pQueueApp, void *pItem, uint32_t *puItems)
{
    NvQueueCtx *pQueue = (NvQueueCtx *)pQueueApp;
    NvMediaStatus nr = NVMEDIA_STATUS_ERROR;

    uint32_t uNextGet;
    if(pQueue) {
        nr = NvMutexAcquire(pQueue->pMutex);
        *puItems = pQueue->uItems;
        if(pQueue->uItems) {
            uNextGet = (pQueue->uNextGet + 1) % pQueue->uQueueSize;
            memcpy(pItem, pQueue->pQueueData + (uNextGet * pQueue->uItemSize), pQueue->uItemSize);
        }
        nr = NvMutexRelease(pQueue->pMutex);
    }
    return nr;
}
