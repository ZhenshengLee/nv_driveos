/*
 * Copyright (c) 2016 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

#ifndef NVRM_GPUSCHED_H
#define NVRM_GPUSCHED_H

#include <sys/types.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef uint32_t NvRmGpuSchedTsgId;
typedef struct NvRmGpuSchedSessionRec NvRmGpuSchedSession;
typedef struct NvRmGpuSchedTsgRec NvRmGpuSchedTsg;
typedef struct NvRmGpuSchedTsgSetRec NvRmGpuSchedTsgSet;

/**
 * The set of all possible error codes.
 */
typedef enum {
    /** The operation completed successfully; no error. */
    NVRM_GPUSCHED_STATUS_OK = 0,
    /** Bad parameter was passed. */
    NVRM_GPUSCHED_STATUS_BAD_PARAMETER,
    /** Operation has not finished yet. */
    NVRM_GPUSCHED_STATUS_PENDING,
    /** Operation timed out. */
    NVRM_GPUSCHED_STATUS_TIMED_OUT,
    /** Out of memory. */
    NVRM_GPUSCHED_STATUS_OUT_OF_MEMORY,
    /** Not initialized. */
    NVRM_GPUSCHED_STATUS_NOT_INITIALIZED,
    /** Not supported. */
    NVRM_GPUSCHED_STATUS_NOT_SUPPORTED,
    /** A catch-all error, used when no other error code applies. */
    NVRM_GPUSCHED_STATUS_ERROR,
    /** No operation is pending. */
    NVRM_GPUSCHED_STATUS_NONE_PENDING,
    /** Insufficient buffering. */
    NVRM_GPUSCHED_STATUS_INSUFFICIENT_BUFFERING,
    /** Invalid size. */
    NVRM_GPUSCHED_STATUS_INVALID_SIZE,
    /** Incompatible version */
    NVRM_GPUSCHED_STATUS_INCOMPATIBLE_VERSION,
} NvRmGpuSchedStatus;

/**
 * Bitfield of TSG events.
 */
typedef enum
{
    /** At least one TSG has been opened */
    NVRM_GPUSCHED_TSG_EVENT_OPEN = (1ULL << 0),
} NvRmGpuSchedTsgEvents;

typedef enum {
    NVRM_GPUSCHED_RUNLIST_INTERLEAVE_LOW = 1,
    NVRM_GPUSCHED_RUNLIST_INTERLEAVE_MEDIUM = 2,
    NVRM_GPUSCHED_RUNLIST_INTERLEAVE_HIGH = 3,
} NvRmGpuSchedRunlistInterleave;

typedef struct NvRmGpuSchedTsgParamsRec {
    NvRmGpuSchedTsgId tsgid;
    NvRmGpuSchedRunlistInterleave interleave;
    uint32_t timesliceUs;
    pid_t pid;
} NvRmGpuSchedTsgParams;

typedef struct NvRmGpuSchedSessionAttrRec
{
    uint32_t reserved;
} NvRmGpuSchedSessionAttr;

#define NVRM_GPUSCHED_DEFINE_SESSION_ATTR(x) \
    NvRmGpuSchedSessionAttr x = { 0 }

#define NVRM_GPUSCHED_WAIT_INFINITE	((uint32_t)-1)

/**
 * Open GPU scheduling control session
 *
 * @param attr - session attributes
 * @param phSession - address of GPU sched session handle to be allocated
 *
 * @returns NVRM_GPUSCHED_STATUS_OK Indicates that GPU scheduling session is opened.
 */
NvRmGpuSchedStatus NvRmGpuSchedSessionOpen(
    const NvRmGpuSchedSessionAttr *attr,
    NvRmGpuSchedSession **phSession);

/**
 * Close GPU scheduling control session
 *
 * @param hSession - session handle
 */
void NvRmGpuSchedSessionClose(NvRmGpuSchedSession *hSession);

/**
 * Blocking wait on any TSG event.
 *
 * @param session - session handle
 * @param events - output bitfield of TSG event that occurred. can be NULL, if
 *     caller does not need this information.
 * @param timeout - timeout value in milliseconds.
 *     NVRM_GPUSCHED_WAIT_INFINITE to wait infinitely.
 *
 * @returns NVRM_GPUSCHED_STATUS_OK indicates that we received events. 'events'
 *     contains theTSG events that occurred.
 */
NvRmGpuSchedStatus NvRmGpuSchedWaitTsgEvents(
    NvRmGpuSchedSession *hSession,
    NvRmGpuSchedTsgEvents *events,
    uint32_t timeout);

/**
 * Get list of all TSGs
 *
 * @param hTsgSet - handle to TSG set. will contain active TSGs on success.
 *
 * @returns NVRM_GPUSCHED_STATUS_OK when all active TSGs could be retrieved.
 */
NvRmGpuSchedStatus NvRmGpuSchedGetAllActiveTsgs(
    NvRmGpuSchedTsgSet *hTsgSet);

/**
 * Get list of recent TSGs
 *
 * The first time this function is invoked, it returns the complete list of
 * TSGs. Afterwards, it returns the list of TSGs that were opened since last
 * invocation of this function.
 *
 * @param hTsgSet - handle to TSG set. will contain delta TSGs on success.
 *
 * @returns NVRM_GPUSCHED_STATUS_OK when recent TSGs could be retrieved (note
 *     that the list can be empty if no TSG was allocated since last invocation
 *     of this function)
 */
NvRmGpuSchedStatus NvRmGpuSchedGetDeltaTsgs(
    NvRmGpuSchedTsgSet *hTsgSet);

/**
 * Get list of TSGs for a given pid
 *
 * @param hTsgSet - handle to TSG set. will contain process's TSGs on success.
 * @param pid - process identifier
 *
 * @returns NVRM_GPUSCHED_STATUS_OK when list of TSG could be retrieved (note
 *     that the list can be empty if no TSG was found for this pid)
 */
NvRmGpuSchedStatus NvRmGpuSchedGetTsgsByPid(
    NvRmGpuSchedTsgSet *hTsgSet,
    pid_t pid);

/**
 * Get next TSG identifier from a TSG set.
 *
 * TSG set has an internal iterator which is incremented until all TSGs
 * in the set are iterated.
 *
 * @param hTsgSet - handle to TSG set
 * @param tsgid - a TSG identifier
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if one TSG could be fetched from the TSG set.
 * @returns NVRM_GPUSCHED_NONE_PENDING if TSG is empty.
 */
NvRmGpuSchedStatus NvRmGpuSchedGetNextTsgId(
    NvRmGpuSchedTsgSet *hTsgSet,
    NvRmGpuSchedTsgId *tsgid);

/**
 * Get a handle to a TSG.
 *
 * Acquiring the TSG ensures it is not re-allocated while setting scheduling
 * parameters. After setting scheduling parameters, this handle must be
 * released using NvRmGpuSchedReleaseTsgHandle.
 *
 * @param hSession - session handle
 * @param tsgid - a TSG identifier
 * @param phTsg - address of TSG handle to be allocated
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if TSG could be acquired.
 */
NvRmGpuSchedStatus NvRmGpuSchedGetTsgHandle(
    NvRmGpuSchedSession *hSession,
    NvRmGpuSchedTsgId tsgid,
    NvRmGpuSchedTsg **phTsg);

/**
 * Get current GPU scheduling parameters for a TSG
 *
 * @param hTsg - handle to TSG
 * @param params - GPU scheduling parameters
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if params could be retrieved for this TSG.
 */
NvRmGpuSchedStatus NvRmGpuSchedGetTsgParams(
    NvRmGpuSchedTsg *hTsg,
    NvRmGpuSchedTsgParams *params);

/**
 * Set runlist interleave for a TSG
 *
 * @param hTsg - handle to TSG
 * @param interleave - interleave level
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if interleave could be set for this TSG.
 */
NvRmGpuSchedStatus NvRmGpuSchedSetTsgInterleave(
    NvRmGpuSchedTsg *hTsg,
    NvRmGpuSchedRunlistInterleave interleave);

/**
 * Set timeslice for a TSG
 *
 * @param hTsg - handle to TSG
 * @param timeslice - timeslice in usecs.
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if timeslice could be set for this TSG.
 */
NvRmGpuSchedStatus NvRmGpuSchedSetTsgTimeslice(
    NvRmGpuSchedTsg *hTsg,
    uint32_t timeslice);

/**
 * Release TSG handle
 *
 * @param hTsg - handle to TSG
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if TSG handle could be released
 */
NvRmGpuSchedStatus NvRmGpuSchedReleaseTsgHandle(
    NvRmGpuSchedTsg *hTsg);

/**
 * Lock GPU scheduling control, to prevent other application to change
 * their own scheduling parameters using the per-TSG API. Locking is non
 * sticky: it ends when the session is closed, or when the app manager
 * exits.
 *
 * @param hSession - session handle
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if GPU scheduling control could be locked
 */
NvRmGpuSchedStatus NvRmGpuSchedLockControl(
    NvRmGpuSchedSession *hSession);

/**
 * Unlock GPU scheduling control, to allow other application to change
 * their own scheduling parameters using the per-TSG API.
 *
 * @param hSession - pointer to session
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if GPU scheduling control could be unlocked
 */
NvRmGpuSchedStatus NvRmGpuSchedUnlockControl(
    NvRmGpuSchedSession *hSession);

/**
 * Allocate a TSG set
 *
 * TSG set contains an internal iterator  which is reset every time the set
 * is used with a query functions like NvRmGpuSchedGetAllActiveTsgs,
 * NvRmGpuSchedGetDeltaTsgs or NvRmGpuSchedGetTsgByPids. TSG set is recyclable,
 * i.e. it can be used several times with query functions.
 *
 * @param session - pointer to session
 * @param phSet - address of TSG set handle to be allocated
 *
 * @returns NVRM_GPUSCHED_STATUS_OK if TSG set could be allocated.
 */
NvRmGpuSchedStatus NvRmGpuSchedAllocateTsgSet(
    NvRmGpuSchedSession *hSession,
    NvRmGpuSchedTsgSet **phSet);

/**
 * Free TSG set
 *
 * @param hSet - TSG set handle
 */
void NvRmGpuSchedFreeTsgSet(
    NvRmGpuSchedTsgSet *hSet);


#ifdef __cplusplus
};  /* end of extern "C" */
#endif /* __cplusplus */

#endif /* NVRM_GPUSCHED_H */
