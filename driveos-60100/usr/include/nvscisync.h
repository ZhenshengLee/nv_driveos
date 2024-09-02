/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */
/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSciSync </b>
 *
 * The NvSciSync library allows applications to manage synchronization
 * objects which coordinate when sequences of operations begin and end.
 */
#ifndef INCLUDED_NVSCISYNC_H
#define INCLUDED_NVSCISYNC_H

/**
 * @defgroup nvsci_sync Synchronization APIs
 *
 * @ingroup nvsci_group_stream
 * @{
 *
 * The NvSciSync library allows applications to manage synchronization
 * objects which coordinate when sequences of operations begin and end.
 *
 * The following constants are defined and have type @c unsigned @c int:
 * * @c NvSciSyncMajorVersion
 * * @c NvSciSyncMinorVersion
 *
 * In C and C++ these constants are guaranteed to be defined as global
 * const variables.

 * Upon each new release of NvSciSync:
 * * If the new release changes behavior of NvSciSync that could in any way
 *   prevent an application compliant with previous versions of this
 *   specification from functioning correctly, or that could prevent
 *   interoperability with earlier versions of NvSciSync, the major version is
 *   increased and the minor version is reset to zero.
 *   @note This is typically done only after a deprecation period.
 * * Otherwise, if the new release changes NvSciSync in any application
 *   observable manner (e.g., new features, or bug fixes), the major version is
 *   kept the same and the minor version is increased.
 * * Otherwise, the major and minor version are both kept the same.
 *
 * This version of this specification corresponds to NvSciSync version 1.0
 * (major version 1, minor version 0).
 *
 * Different processes using the NvSciSync inter-process APIs may use different
 * minor versions of NvSciSync within the same major version, provided that if
 * a process uses a feature newer than another processor's NvSciSync version,
 * the latter process does not import an unreconciled NvSciSyncAttrList (directly
 * or indirectly) from the former process.
 *
 * In general, an NvSciSync API call will not return an error code if it has
 * caused any side effects other than allocating resources and subsequently
 * freeing those resources.
 *
 * In general, unless specified otherwise, if a NULL pointer is passed to an
 * NvSciSync API call, the API call will either return ::NvSciError_BadParameter
 * or (if there are other applicable error conditions as well) an error code
 * corresponding to another error.
 *
 * Each NvSciSyncAttrList is either unreconciled or reconciled.
 * It is unreconciled if it was:
 * - Created by NvSciSyncAttrListCreate(),
 * - Created by a failed NvSciSyncAttrListReconcile() call,
 * - Created by a failed NvSciSyncAttrListReconcileAndObjAlloc() call, or
 * - An import of an export of one or more unreconciled NvSciSyncAttrLists.
 *
 * It is reconciled if it was:
 * - Created by a successful NvSciSyncAttrListReconcile() call,
 * - Provided by NvSciSyncObjGetAttrList(), or
 * - An import of an export of another reconciled NvSciSyncAttrList.
 */

/**
 * \page nvscisync_page_blanket_statements NvSciSync blanket statements
 * \section nvscisync_in_out_params Input/Output parameters
 * - NvSciSyncFence passed as input parameter to an API is valid input if
 * it was first initialized to all zeroes
 * or modified by any successful NvSciSync API accepting NvSciSyncFence.
 * - NvSciSyncObj passed as an input parameter to an API is valid input if it
 * is returned from a successful call to NvSciSyncObjAlloc(), NvSciSyncObjDup(),
 * NvSciSyncObjIpcImport(), NvSciSyncIpcImportAttrListAndObj(),
 * NvSciSyncAttrListReconcileAndObjAlloc(), or NvSciSyncFenceGetSyncObj() and
 * has not yet been deallocated using NvSciSyncObjFree().
 * - NvSciSyncCpuWaitContext passed as an input parameter to an API is valid
 * input if it is returned from a successful call to
 * NvSciSyncCpuWaitContextAlloc() and has not been deallocated using
 * NvSciSyncCpuWaitContextFree().
 * - NvSciSyncModule passed as input parameter to an API is valid input if it is
 * returned from a successful call to NvSciSyncModuleOpen() and has not yet
 * been deallocated using NvSciSyncModuleClose().
 * - NvSciIpcEndpoint passed as input parameter to an API is valid if it is
 * obtained from successful call to NvSciIpcOpenEndpoint() and has not yet been
 * freed using NvSciIpcCloseEndpointSafe().
 * - Unreconciled NvSciSyncAttrList is valid if it is obtained from successful
 * call to NvSciSyncAttrListCreate() or if it is obtained from successful call to
 * NvSciSyncAttrListClone() where input to NvSciSyncAttrListClone() is valid
 * unreconciled NvSciSyncAttrList or if it is obtained from successful call to
 * NvSciSyncAttrListIpcImportUnreconciled() and has not been deallocated using
 * NvSciSyncAttrListFree().
 * - Reconciled NvSciSyncAttrList is valid if it is obtained from successful call
 * to NvSciSyncAttrListReconcile() or if it is obtained from successful call to
 * NvSciSyncAttrListClone() where input to NvSciSyncAttrListClone() is valid
 * reconciled NvSciSyncAttrList or if it is obtained from successful call to
 * NvSciSyncAttrListIpcImportReconciled() and has not been deallocated using
 * NvSciSyncAttrListFree() or has been obtained from a successful call to
 * NvSciSyncObjGetAttrList() and the input NvSciSyncObj to this call has not
 * yet been deallocated using NvSciSyncObjFree().
 * - If the valid range for the input parameter is not explicitly mentioned in
 * the API specification or in the blanket statements then it is considered that
 * the input parameter takes any value from the entire range corresponding to
 * its datatype as the valid value. Please note that this also applies to the
 * members of a structure if the structure is taken as an input parameter.
 * - NvSciSyncModule is not sharable accross processes. Users must create a new
 * NvSciSyncModule using NvSciSyncModuleOpen() in every process.
 * - Applications need to ensure that any non-NULL NvSciSync structures, such as
 * NvSciSyncModule, NvSciSyncAttrList, NvSciSyncObj, NvSciSync AttrList/Object
 * transport descriptors passed as an input parameter to NvSciSync APIs were
 * returned as an output parameter by a prior successful invocation of an
 * NvSciSync API and those structures were not freed.
 * - Every NvSciSyncFence needs to be intialized to all zeros after its storage
 * is allocated and before it is passed to an API for the first time.
 *
 * \section nvscisync_out_params Output parameters
 * - In general, output parameters are passed by reference through pointers.
 * Also, since a null pointer cannot be used to convey an output parameter, API
 * functions typically return an error code if a null pointer is supplied for a
 * required output parameter unless otherwise stated explicitly. Output
 * parameter is valid only if error code returned by an API is
 * NvSciError_Success unless otherwise stated explicitly.
 *
 * \section nvscisync_concurrency Concurrency
 * - Every individual function can be called concurrently with itself
 * without any side-effects unless otherwise stated explicitly in
 * the interface specifications.
 * - The conditions for combinations of functions that cannot be called
 * concurrently or calling them concurrently leads to side effects are
 * explicitly stated in the interface specifications.
 *
 * \section nvscisync_fence_states Fence states
 * - A zero initialized NvSciSyncFence or one fed to NvSciSyncFenceClear()
 *   becomes cleared.
 * - NvSciSyncFence becomes not cleared if it is modified by a successful
 * NvSciSyncObjGenerateFence() or NvSciSyncFenceDup().
 * - NvSciSyncFence filled by successful NvSciSyncIpcImportFence() is cleared
 * if and only if the input fence descriptor was created from a cleared
 * NvSciSyncFence.
 *
 * \implements{18839709}
 */

#if !defined (__cplusplus)
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#endif

#include "nvscierror.h"
#include "nvsciipc.h"
#include "nvscibuf.h"

#if defined (__cplusplus)
extern "C"
{
#endif

#if defined __GNUC__
    #define PACK_SYNC( __Declaration__, ... ) __Declaration__,##__VA_ARGS__ __attribute__((packed))
#else
    #define PACK_SYNC( __Declaration__, ... ) __pragma(pack(push, 1)) __Declaration__,##__VA_ARGS__ __pragma(pack(pop))
#endif

/**
 * \brief NvSciSync major version number.
 *
 * \implements{18840177}
 */
static const uint32_t NvSciSyncMajorVersion = 2U;

/**
 * \brief NvSciSync minor version number.
 *
 * \implements{18840180}
 */
static const uint32_t NvSciSyncMinorVersion = 8U;

/**
 * Maximum supported timeout value.
 * NvSciSyncFenceWait() can wait for at most this many microseconds.
 * This value also corresponds to infinite timeout.
 */
static const int64_t NvSciSyncFenceMaxTimeout = (0x7fffffffffffffff / 1000);

/**
 * \brief Represents an instance of the NvSciSyncModule. Any NvSciSyncAttrList
 * created or imported using a particular NvSciSyncModule is bound to that
 * module instance, along with any NvSciSyncObjs created or imported using those
 * NvSciSyncAttrLists and any NvSciSyncFences created or imported using those
 * NvSciSyncObjs.
 *
 * @note For any NvSciSync API call that has more than one input of type
 * NvSciSyncModule, NvSciSyncAttrList, NvSciSyncObj, and/or NvSciSyncFence, all
 * such inputs must agree on the NvSciSyncModule instance.
 */
typedef struct NvSciSyncModuleRec* NvSciSyncModule;

/**
 * \brief Represents the right to perform a CPU wait on an NvSciSyncFence. It
 * holds resources necessary to perform a CPU wait using NvSciSyncFenceWait().
 * It can be used to wait on NvSciSyncFence(s) associated with the same
 * NvSciSyncModule that this NvSciSyncCpuWaitContext is associated with.
 * An NvSciSyncCpuWaitContext can be used to wait on only one
 * NvSciSyncFence at a time. However, a single NvSciSyncCpuWaitContext
 * can be used to wait on different NvSciSyncFences at different
 * instants of time.
 */
typedef struct NvSciSyncCpuWaitContextRec* NvSciSyncCpuWaitContext;

/**
 * \brief Defines the opaque NvSciSyncFence.
 *
 * This structure describes a synchronization primitive snapshot
 * that must be reached to release the waiters.
 *
 * Unlike NvSciSyncAttrList and NvSciSyncObj objects, applications are
 * responsible for the memory management of NvSciSyncFences.
 *
 * Every NvSciSyncFence must be initialized to all zeros after its storage
 * is allocated before the first time it is passed to any NvSciSync API
 * calls. NvSciSyncFenceInitializer can be used for this, or
 * memset(), or any other mechanism for zeroing memory.
 *
 * Every NvSciSyncFence not in a cleared state holds a reference to
 * the NvSciSyncObj it is related to, preventing that NvSciSyncObj
 * from being deleted. It also contains
 * id and value of the synchronization primitive corresponding
 * to the desired snapshot.
 *
 * In case if the corresponding NvSciSyncObj supports timestamps,
 * this structure also contains information about the memory location
 * of the timestamp of the event unblocking the NvSciSyncFence.
 *
 * This structure also contains information about the memory location
 * of the status of the task triggering the event unblocking
 * the NvSciSyncFence.
 */
/**
 * \implements{18840156}
 */
typedef struct NvSciSyncFence {
    uint64_t payload[6];
} NvSciSyncFence;

/**
 * \brief Defines the value used to zero-initialize the NvSciSyncFence object.
 *  An NvSciSyncFence that is all zeroes is in a cleared state.
 *
 * \implements{18840183}
 */
static const NvSciSyncFence NvSciSyncFenceInitializer = {{0U}};

/**
 * \brief Defines the exported form of NvSciSyncFence intended to be shared
 * across an NvSciIpc channel.
 *
 * \implements{18840195}
 */
typedef struct {
    uint64_t payload[7];
} NvSciSyncFenceIpcExportDescriptor;

/**
 * \brief Defines the exported form of NvSciSyncObj intended to be shared
 * across an NvSciIpc channel.
 *
 * \implements{18840198}
 */
typedef struct {
    /** Exported data (blob) for NvSciSyncObj */
    uint64_t payload[128];
} NvSciSyncObjIpcExportDescriptor;

/**
 * A Synchronization Object is a container holding the reconciled
 * NvSciSyncAttrList defining constraints of the Fence and the handle of the
 * actual Primitive, with access permissions being enforced by the
 * NvSciSyncAttrKey_RequiredPerm and NvSciSyncAttrKey_NeedCpuAccess Attribute
 * Keys.
 *
 * If Timestamps have been requested prior to Reconciliation via the
 * NvSciSyncAttrKey_WaiterRequireTimestamps key, this will also hold the
 * Timestamp Buffer allocated by NvSciBuf.
 *
 * The object will also hold a buffer containing the task statuses
 * of the tasks signaling this Object.
 */

/**
 * \brief A reference to a particular Synchronization Object.
 *
 * @note Every NvSciSyncObj that has been created but not freed
 * holds a reference to the NvSciSyncModule, preventing the module
 * from being de-initialized.
 */
typedef struct NvSciSyncObjRec* NvSciSyncObj;

/**
 * \brief A reference, that is not modifiable,
 *  to a particular Synchronization Object.
 */
typedef const struct NvSciSyncObjRec* NvSciSyncObjConst;

/**
 * \brief A container constituting an NvSciSyncAttrList which contains:
 * - set of NvSciSyncAttrKey attributes defining synchronization object
 *   constraints
 * - slot count defining number of slots in an NvSciSyncAttrList
 * - flag specifying if NvSciSyncAttrList is reconciled or unreconciled.
 *
 * @note Every NvSciSyncAttrList that has been created but not freed
 * holds a reference to the NvSciSyncModule, preventing the module
 * from being de-initialized.
 */
typedef struct NvSciSyncAttrListRec* NvSciSyncAttrList;

/**
 * \brief Describes NvSciSyncObj access permissions.
 *
 * \implements{18840171}
 */
typedef uint64_t NvSciSyncAccessPerm;

/**
 * This represents the capability to wait on an NvSciSyncObj as it
 * progresses through points on its sync timeline.
 */
#if defined(__cplusplus)
    #define NvSciSyncAccessPerm_WaitOnly (static_cast<uint64_t>(1U) << 0U)
#else
    #define NvSciSyncAccessPerm_WaitOnly ((uint64_t)1U << 0U)
#endif

/**
 * This represents the capability to advance an NvSciSyncObj to its
 * next point on its sync timeline.
 */
#if defined(__cplusplus)
    #define NvSciSyncAccessPerm_SignalOnly (static_cast<uint64_t>(1U) << 1U)
#else
    #define NvSciSyncAccessPerm_SignalOnly ((uint64_t)1U << 1U)
#endif

/**
  * This represents the capability to advance an NvSciSyncObj to its
  * next point on its sync timeline and also wait until that next point is
  * reached.
  */
#define NvSciSyncAccessPerm_WaitSignal (NvSciSyncAccessPerm_WaitOnly | NvSciSyncAccessPerm_SignalOnly)
/**
 * Usage of Auto permissions is restricted only for export/import APIs and
 * shouldn't be used as valid value for NvSciSyncAttrKey_RequiredPerm
 * Attribute.
 */
#if defined(__cplusplus)
    #define NvSciSyncAccessPerm_Auto (static_cast<uint64_t>(1U) << 63U)
#else
    #define NvSciSyncAccessPerm_Auto ((uint64_t)1U << 63U)
#endif

/**
 * \brief Status of the signaler's task that signals a particular
 *     NvSciSyncFence.
 *
 * This is defined as an enum but will be kept in a 16-bit field
 * of a slot in the shared task status buffer.
 *
 * A value outside of range defined here is an engine specific failure.
 *
 * \implements{}
 */
typedef enum {
    /** The task has completed successfully */
    NvSciSyncTaskStatus_Success = 0U,
    /** The task has failed */
    NvSciSyncTaskStatus_Failure = 1U,
    /** The signaler did not report any task status.
        The default value set by NvSciSync when a new slot is requested. */
    NvSciSyncTaskStatus_Invalid = UINT16_MAX,
} NvSciSyncTaskStatusVal;

/**
 * \brief A single slot in the task status buffer.
 *
 * \implements{}
 */
PACK_SYNC(typedef struct {
    /** Used to get timestamp of Task Status */
    uint64_t timestamp;
    /** unused */
    uint32_t statusEngine;
    /** unused */
    uint16_t subframe;
    /** A status word filled with NvSciSyncTaskStatusVal values.
     * A value from beyond NvSciSyncTaskStatusVal's range signifies
     * an engine specific error */
    uint16_t status;
}) NvSciSyncTaskStatus;

/**
 * \brief Types of synchronization primitives.
 *
 * \implements{}
 */
enum NvSciSyncAttrValPrimitiveTypeRec {
    /** For NvSciSync internal use only */
    NvSciSyncAttrValPrimitiveType_LowerBound,
    /**
     * Syncpoint.
     * Supported only on Tegra platforms.
     */
    NvSciSyncAttrValPrimitiveType_Syncpoint,
    /**
     * 16 bytes semaphore backed by system memory.
     * Contains space for 8-byte timestamp and 4-byte payload.
     * Supported on Tegra and x86_64 platforms.
     */
    NvSciSyncAttrValPrimitiveType_SysmemSemaphore,
    /**
     * 16 bytes semaphore backed by video memory.
     * Contains space for 8-byte timestamp and 4-byte payload.
     * Currently not supported.
     */
    NvSciSyncAttrValPrimitiveType_VidmemSemaphore,
    /**
     * 16 bytes semaphore backed by system memory.
     * Contains space for 8-byte timestamp and 8-byte payload.
     * Supported on Tegra and x86_64 platforms.
     */
    NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b,
    /**
     * 16 bytes semaphore backed by video memory.
     * Contains space for 8-byte timestamp and 8-byte payload.
     * Currently not supported.
     */
    NvSciSyncAttrValPrimitiveType_VidmemSemaphorePayload64b,
    /** For NvSciSync internal use only */
    NvSciSyncAttrValPrimitiveType_UpperBound,
};

/**
 * \brief Alias for enum NvSciSyncAttrValPrimitiveTypeRec
 *
 * \implements{}
 */
typedef enum NvSciSyncAttrValPrimitiveTypeRec NvSciSyncAttrValPrimitiveType;

/**
 * \brief Describes the NvSciSync public attribute keys holding the
 * corresponding values specifying synchronization object constraints. Input
 * attribute keys specify desired synchronization object constraints and can be
 * set/retrieved from the unreconciled NvSciSyncAttrList using
 * NvSciSyncAttrListSetAttrs()/NvSciSyncAttrListGetAttrs() respectively. Output
 * attribute keys specify actual constraints computed by NvSciSync if
 * reconciliation succeeds. Output attribute keys can be retrieved from a
 * reconciled NvSciSyncAttrList using NvSciSyncAttrListGetAttrs().
 *
 * \implements{18840165}
 */
typedef enum {
    /** Specifies the lower bound - for NvSciSync internal use only. */
    NvSciSyncAttrKey_LowerBound,
    /** (bool, inout) Specifies if CPU access is required.
     *
     * During reconciliation, reconciler sets value of this key to true in the
     * reconciled NvSciSyncAttrList if any of the unreconciled
     * NvSciSyncAttrList(s) involved in reconciliation that is owned by the
     * reconciler has this key set to true, otherwise it is set to false in
     * reconciled NvSciSyncAttrList.
     * If the user sets it to true in the unreconciled list but does not set
     * NvSciSyncAttrKey_PrimitiveInfo, then NvSciSync will add the following values
     * to NvSciSyncAttrKey_PrimitiveInfo attribute just before reconciliation or
     * exporting unreconciled attribute list:
     * - NvSciSyncAttrValPrimitiveType_Syncpoint
     * - NvSciSyncAttrValPrimitiveType_SysmemSemaphore
     * - NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b
     *
     * When importing the reconciled NvSciSyncAttrList NvSciSync will set the key
     * to OR of values of this key in unreconciled NvSciSyncAttrList(s) relayed by
     * the peer.
     *
     * During validation of reconciled NvSciSyncAttrList against input
     * unreconciled NvSciSyncAttrList(s), validation succeeds if value of this
     * attribute in the reconciled NvSciSyncAttrList is true provided any of the
     * input unreconciled NvSciSyncAttrList(s) owned by the peer set it to
     * true OR if value of this attribute in the reconciled NvSciSyncAttrList is
     * false provided all of the input unreconciled NvSciSyncAttrList(s) owned by
     * the peer set it to false.
     */
    NvSciSyncAttrKey_NeedCpuAccess,
    /**
     * (NvSciSyncAccessPerm, inout) Specifies the required access permissions.
     * If @ref NvSciSyncAttrKey_NeedCpuAccess is true, the CPU will be offered
     * at least these permissions.
     * Any hardware accelerators that contribute to this NvSciSyncAttrList will be
     * offered at least these permissions.
     */
    NvSciSyncAttrKey_RequiredPerm,
    /**
     * (NvSciSyncAccessPerm, out) Actual permission granted after reconciliation.
     * @note This key is read-only.
     *
     * It represents the cumulative permissions of the
     * NvSciSyncAttrKey_RequiredPerm in all NvSciSyncAttrLists being reconciled.
     * The reconciliation fails if any of the following conditions are met:
     * - no NvSciSyncAttrList with NvSciSyncAttrKey_RequiredPerm being set to
     * NvSciSyncAccessPerm_SignalOnly,
     * - more than one NvSciSyncAttrList with NvSciSyncAttrKey_RequiredPerm
     * being set to NvSciSyncAccessPerm_SignalOnly,
     * - no NvSciSyncAttrList with NvSciSyncAttrKey_RequiredPerm
     * being set to NvSciSyncAccessPerm_WaitOnly.
     *
     * If NvSciSyncObj is obtained by calling NvSciSyncObjAlloc(),
     * NvSciSyncAttrKey_ActualPerm is set to NvSciSyncAccessPerm_WaitSignal
     * in the reconciled NvSciSyncAttrList corresponding to it since allocated
     * NvSciSyncObj gets wait-signal permissions by default.
     *
     * For any peer importing the NvSciSyncObj, this key is set in the reconciled
     * NvSciSyncAttrList to the sum of NvSciSyncAttrKey_RequiredPerm requested
     * by the peer and all peers relaying their NvSciSyncAttrList export
     * descriptors via it.
     *
     * During validation of reconciled NvSciSyncAttrList against input
     * unreconciled NvSciSyncAttrList(s), validation succeeds only if
     * NvSciSyncAttrKey_ActualPerm in reconciled is bigger or equal than
     * NvSciSyncAttrKey_RequiredPerm of all the input unreconciled
     * NvSciSyncAttrLists.
     */
    NvSciSyncAttrKey_ActualPerm,
    /**
     * (bool, inout) Importing and then exporting an
     * NvSciSyncFenceIpcExportDescriptor has no side effects and yields an
     * identical NvSciSyncFenceIpcExportDescriptor even if the
     * NvSciIpcEndpoint(s) used for import and export are different from ones
     * used for exporting/importing NvSciSyncAttrList(s).
     *
     * If this attribute key is set to false, this indicates that the
     * NvSciSyncFenceIpcExportDescriptor must be exported through the same IPC
     * path as the NvSciSyncObj. Otherwise if set to true, this indicates that
     * the NvSciSyncFenceIpcExportDescriptor must be exported via NvSciIpc
     * through the first peer that was part of the IPC path travelled through
     * by the NvSciSyncObj (but not necessarily an identical path).
     *
     * During reconciliation, this key is set to true in reconciled
     * NvSciSyncAttrList if any one of the input NvSciSyncAttrList has this set
     * to true.
     */
    NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports,
    /** (bool, inout) Specifies if timestamps are required. If the waiter
     * wishes to read timestamps then the NvSciSyncAttrKey_NeedCpuAccess key
     * should be set to true.
     *
     * During reconciliation, reconciler sets value of this key to true in the
     * reconciled NvSciSyncAttrList if any of the unreconciled
     * NvSciSyncAttrList(s) involved in reconciliation has NvSciSyncAttrKey_RequiredPerm
     * set to NvSciSyncAccessPerm_WaitOnly/NvSciSyncAccessPerm_WaitSignal and
     * NvSciSyncAttrKey_NeedCpuAccess is set to true, otherwise it is set to false in
     * reconciled NvSciSyncAttrList.
     */
    NvSciSyncAttrKey_WaiterRequireTimestamps,
    /**
     * (bool, inout) Specifies if deterministic primitives are required.
     * This allows for the possibility of generating fences on the waiter's
     * side without needing to import them. This means that the Signaler will
     * increment the instance 0 of the primitive in NvSciSyncObj by 1 at each
     * submission.
     *
     * During reconciliation, this key is set to true in the reconciled
     * NvSciSyncAttrList if any one of the input NvSciSyncAttrList(s) has this
     * set to true.
     */
    NvSciSyncAttrKey_RequireDeterministicFences,
    /**
     * (uint32_t, out) Specifies the total number of slots in the timestamps buffer.
     * This allows the user to iterate over the slots by incrementing
     * the current slot index modulo _NumTimestampsSlots.
     *
     * During reconciliation, the value of this key is set to the number of
     * slots in the timestamps buffer. If timestamps are not required by this
     * attribute list, then the value of this key is undefined.
     */
    NvSciSyncAttrKey_NumTimestampSlots,
    /**
     * (uint32_t, out) Specifies the total number of slots in the task status buffer.
     * This allows the user to iterate over the slots by incrementing
     * the current slot index modulo _NumTaskStatusSlots.
     *
     * During reconciliation, the value of this key is set to the number of
     * slots in the task status buffer. If task status is not required by this
     * attribute list, then the value of this key will be 0.
     */
    NvSciSyncAttrKey_NumTaskStatusSlots,
    /**
     * (uint64_t, out) Specifies the maximum value of the reconciled primitive.
     *
     * During reconciliation, value of this key is set to UINT64_MAX if the
     * reconciled primitive type is 64bit-SysmemSemaphore, otherwise value of
     * the key is set to UINT32_MAX
     */
    NvSciSyncAttrKey_MaxPrimitiveValue,
    /**
     * (NvSciSyncAttrValPrimitiveType[], inout) supported primitive
     * types.
     *
     * If set, the value of this key is used in reconciliation and determines
     * the primitive type backing the NvSciSyncObj.
     * If this key is not set, then it has no impact on reconciliation.
     * During reconciliation, the reconciler sets the value of this key to the
     * reconciled primitive type.
     *
     * The reconciliation will fail if this key is set to value which does not
     * intersect with values of primitive type set internally by user mode drivers
     * or NvSciSync.
     *
     * During validation of a reconciled NvSciSyncAttrList against input
     * unreconciled NvSciSyncAttrList(s), validation succeeds if the
     * NvSciSyncAttrValPrimitiveType in
     * NvSciSyncAttrKey_PrimitiveInfo of the reconciled
     * NvSciSyncAttrList is present in NvSciSyncAttrKey_PrimitiveInfo or any of
     * the internal attributes related to NvSciSyncAttrValPrimitiveType
     * of the input unreconciled NvSciSyncAttrLists.
     */
    NvSciSyncAttrKey_PrimitiveInfo,

    /** (NvSciBufPeerLocationInfo[], inout) An attribute indicating location
     * information of late peer which are going to gain access to the allocated
     * NvSciSyncObj using NvSciSyncObjAttachPeer() API.
     */
    NvSciSyncAttrKey_PeerLocationInfo,
    /**
     * (NvSciRmGpuId[], inout)
     * GpuID of the GPU in the system that will access the semaphore
     * buffer.
     */
    NvSciSyncAttrKey_GpuId,
    /**
     * (NvSciBufPeerHwEngine[], inout) An attribute indicating engine
     * information of late peer which are going to gain access to the allocated
     * NvSciSyncObj using NvSciSyncObjAttachPeer() API.
     */
    NvSciSyncAttrKey_PeerHwEngineArray,
    /** Specifies the upper bound - for NvSciSync internal use only. */
    NvSciSyncAttrKey_UpperBound,
} NvSciSyncAttrKey;

/**
 * \brief This structure defines a key/value pair used to get or set
 * the NvSciSyncAttrKey(s) and their corresponding values from or to
 * NvSciSyncAttrList.
 *
 * \implements{18840168}
 */
typedef struct {
    /** NvSciSyncAttrKey for which value needs to be set/retrieved. This member
     * is initialized to any defined value of the NvSciSyncAttrKey other than
     * NvSciSyncAttrKey_LowerBound and NvSciSyncAttrKey_UpperBound */
    NvSciSyncAttrKey attrKey;
    /** Memory which contains the value corresponding to the key */
    const void* value;
    /** Length of the value in bytes */
    size_t len;
} NvSciSyncAttrKeyValuePair;

/**
 * @brief Modes of CPU waiting.
 */
typedef enum {
    /** same behavior as with NvSciSyncFenceWait() */
    NvSciSyncWaitMode_Default = 0U,
    /** Polling with processor yielding */
    NvSciSyncWaitMode_BusyWithYield = 1U,
    /** Polling with spin, no yielding */
    NvSciSyncWaitMode_BusyNoYield = 2U,
    /** Blocked wait, released by a trigger, like an interrupt */
    NvSciSyncWaitMode_Blocking = 3U,
} NvSciSyncWaitMode;

/**
 * \brief Initializes and returns a new NvSciSyncModule with no
 * NvSciSyncAttrLists, NvSciSyncCpuWaitContexts, NvSciSyncObjs or
 * NvSciSyncFences bound to it.
 *
 * @note A process may call this function multiple times. Each successful
 * invocation will yield a new NvSciSyncModule instance.
 *
 * \param[out] newModule The new NvSciSyncModule.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if @a newModule is NULL.
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_ResourceError if system drivers are not available or
 *    resources other then memory are unavailable.
 *
 * @pre None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncModuleOpen(
    NvSciSyncModule* newModule);

/**
 * \brief Closes an instance of the NvSciSyncModule that was
 * obtained through an earlier call to NvSciSyncModuleOpen(). Once an
 * NvSciSyncModule is closed and all NvSciSyncAttrLists, NvSciSyncObjs,
 * NvSciSyncCpuWaitContexts, NvSciSyncFences bound to that module instance are
 * freed, the NvSciSyncModule instance will be de-initialized in the calling
 * process. Until then the NvSciSyncModule will still be accessible from those
 * objects still referencing it.
 *
 * \note Every owner of the NvSciSyncModule must call NvSciSyncModuleClose()
 * only after all the functions invoked by the owner with NvSciSyncModule as an
 * input are completed.
 *
 * \param[in] module The NvSciSyncModule instance to close. The calling process
 * must not pass this module to another NvSciSync API call.
 *
 * \return void
 * - Panics if:
 *   - Init Mode API is called in Runtime Mode.
 *   - @a module is invalid
 *
 * @pre @id{NvSciSyncModuleClose_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the input
 *        NvSciSyncModule @a module
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void NvSciSyncModuleClose(
    NvSciSyncModule module);

/**
 * \brief Allocates a new NvSciSyncCpuWaitContext.
 *
 * \param[in] module NvSciSyncModule instance with which to associate
 *            the new NvSciSyncCpuWaitContext.
 * \param[out] newContext The new NvSciSyncCpuWaitContext.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if @a module is NULL or @a newContext is NULL.
 * - ::NvSciError_InvalidState if failed to associate @a module with @a
 *   newContext.
 * - ::NvSciError_InsufficientMemory if not enough system memory to create a
 *   new context.
 * - ::NvSciError_ResourceError if not enough system resources.
 * - Panics if @a module is not valid.
 *
 * @pre @id{NvSciSyncCpuWaitContextAlloc_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
*/
NvSciError NvSciSyncCpuWaitContextAlloc(
    NvSciSyncModule module,
    NvSciSyncCpuWaitContext* newContext);

/**
 * \brief Releases the NvSciSyncCpuWaitContext.
 *
 * \param[in] context NvSciSyncCpuWaitContext to be freed.
 *
 * \return void
 * - Panics if:
 *   - Init Mode API is called in Runtime Mode.
 *   - NvSciSyncModule associated with @a context is not valid.
 *
 * @pre @id{NvSciSyncCpuWaitContextFree_PreCond_001} @asil{QM}
 *  Valid NvSciSyncCpuWaitContext is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the input
 *        NvSciSyncCpuWaitContext @a context
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
*/
void NvSciSyncCpuWaitContextFree(
    NvSciSyncCpuWaitContext context);

/**
 * \brief Creates a new, single-slot unreconciled NvSciSyncAttrList
 * associated with the input NvSciSyncModule with empty NvSciSyncAttrKeys.
 *
 * \param[in] module The NvSciSyncModule instance with which to associate the
 * new NvSciSyncAttrList.
 * \param[out] attrList The new NvSciSyncAttrList.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any argument is NULL.
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 *   to create a NvSciSyncAttrList.
 * - ::NvSciError_InvalidState if no more references can be taken for
 *     input NvSciSyncModule to create the new NvSciSyncAttrList.
 * - Panics if @a module is not valid.
 *
 * @pre @id{NvSciSyncAttrListCreate_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListCreate(
    NvSciSyncModule module,
    NvSciSyncAttrList* attrList);

/**
 * \brief Frees the NvSciSyncAttrList and removes its association with the
 * NvSciSyncModule with which it was created.
 *
 * \param[in] attrList The NvSciSyncAttrList to be freed.
 * \return void
 * - Panics if:
 *   - Init Mode API is called in Runtime Mode.
 *   - @a attrList is not valid.
 *
 * @pre @id{NvSciSyncAttrListFree_PreCon_001} @asil{QM}
 *  Valid NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the input
 *        NvSciSyncAttrList @a attrList
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void NvSciSyncAttrListFree(
    NvSciSyncAttrList attrList);

/**
 * \brief Checks whether the NvSciSyncAttrList is reconciled
 *
 * \param[in] attrList NvSciSyncAttrList to check.
 * \param[out] isReconciled A pointer to a boolean to store whether the
 * @a attrList is reconciled or not.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any argument is NULL.
 * - Panics if @a attrList is not valid.
 *
 * @pre @id{NvSciSyncAttrListIsReconciled_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListIsReconciled(
    NvSciSyncAttrList attrList,
    bool* isReconciled);

/**
 * \brief Validates a reconciled NvSciSyncAttrList against a set of input
 * unreconciled NvSciSyncAttrLists.
 *
 * \param[in] reconciledAttrList Reconciled NvSciSyncAttrList to be validated.
 * \param[in] inputUnreconciledAttrListArray Array containing the unreconciled
 * NvSciSyncAttrLists used for validation.
 * Valid value: Array of valid unreconciled NvSciSyncAttrLists
 * \param[in] inputUnreconciledAttrListCount number of elements/indices in
 * @a inputUnreconciledAttrListArray
 * Valid value: [1, SIZE_MAX]
 * \param[out] isReconciledListValid A pointer to a boolean to store whether
 * the reconciled NvSciSyncAttrList satisfies the parameters of set of
 * unreconciled NvSciSyncAttrList(s) or not.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - an NvSciSyncAttrList appears multiple times in
 *           @a inputUnreconciledAttrListArray
 *         - @a isReconciledListValid is NULL
 *         - any of the input NvSciSyncAttrLists in
 *           inputUnreconciledAttrListArray are not unreconciled
 *         - @a reconciledAttrList is NULL or not reconciled
 *         - not all the NvSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *           and the @a reconciledAttrList are bound to the same NvSciSyncModule
 *           instance.
 *         - reconciled NvSciSyncAttrList does not satisfy the unreconciled
 *           NvSciSyncAttrLists requirements.
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory
 *   to create temporary data structures
 * - ::NvSciError_Overflow if internal integer overflow occurs.
 * - Panics if @a reconciledAttrList or any of the input unreconciled
 *   NvSciSyncAttrList are not valid.
 *
 * @pre @id{NvSciSyncAttrListValidateReconciled_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrList is obtained.
 * @pre @id{NvSciSyncAttrListValidateReconciled_PreCond_002} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListValidateReconciled(
    NvSciSyncAttrList reconciledAttrList,
    const NvSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    bool* isReconciledListValid);

/**
 * \brief Sets the values for NvSciSyncAttrKey(s) in slot 0 of the
 * input NvSciSyncAttrList.
 *
 * \param[in] attrList An unreconciled NvSciSyncAttrList containing the attribute
 * key and value to set.
 * \param[in] pairArray Array of NvSciSyncAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and
 * key member of every NvSciSyncAttrKeyValuePair in the array is an input or
 * input/output attribute and it is > NvSciSyncAttrKey_LowerBound and <
 * NvSciSyncAttrKey_UpperBound and value member of every NvSciSyncAttrKeyValuePair
 * in the array is not NULL.
 * \param[in] pairCount The number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a pairArray is NULL
 *         - @a attrList is NULL
 *         - @a attrList is not unreconciled and/or not writable,
 *         - @a pairCount is 0
 *         - @a pairArray has duplicate keys
 *         - any of the keys in @a pairArray is not a supported public key
 *         - any of the values in @a pairArray is NULL
 *         - any of the len(s) in @a pairArray is invalid for a given attribute
 *         - any of the attributes to be written is non-writable in attrList
 * - Panics if @a attrList is not valid
 *
 * @pre @id{NvSciSyncAttrListSetAttrs_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListSetAttrs_PostCond_001}
 *  pairArray values are copied to attrList. Set attributes are made
 * non-writeable in attrList.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListSetAttrs(
    NvSciSyncAttrList attrList,
    const NvSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Gets the value of NvSciSyncAttrKey from slot 0 of the input
 * NvSciSyncAttrList.
 *
 *
 * \param[in] attrList NvSciSyncAttrList to retrieve the value for given
 * NvSciSyncAttrKey(s) from
 * \param[in,out] pairArray A pointer to the array of NvSciSyncAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every NvSciSyncAttrKeyValuePair in the array > NvSciSyncAttrKey_LowerBound
 * and < NvSciSyncAttrKey_UpperBound.
 * \param[in] pairCount The number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - @a pairCount is 0
 *         - any of the keys in @a pairArray is not a supported NvSciSyncAttrKey
 * - Panics if @a attrList is not valid
 *
 * @pre @id{NvSciSyncAttrListGetAttrs_PreCond_001} @asil{B}
 *  Valid NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListGetAttrs_PostCond_001}
 *  If an attribute was never set, the corresponding value will be set to NULL and length to 0.
 * @arr @id{NvSciSyncAttrListGetAttrs_RES_001} @asil{D}
 *  The return values, stored in NvSciSyncAttrKeyValuePair, consist of
 * const void* pointers to the attribute values from NvSciSyncAttrList.
 * The application must not write to this data.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListGetAttrs(
    NvSciSyncAttrList attrList,
    NvSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Gets the slot count of the given NvSciSyncAttrList.
 *
 * \param[in] attrList NvSciSyncAttrList to get the slot count from.
 * \return Number of slots or 0 if attrList is NULL or panic if attrList
 *  is not valid
 *
 * @pre @id{NvSciSyncAttrListGetSlotCount_PreCond_001} @asil{B}
 *  Valid NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
size_t NvSciSyncAttrListGetSlotCount(
    NvSciSyncAttrList attrList);

/**
 * \brief Appends multiple unreconciled NvSciSyncAttrLists together, forming a
 * single new unreconciled NvSciSyncAttrList with a slot count equal to the sum
 * of all the slot counts of NvSciSyncAttrList(s) in the input array which is
 * no longer writable.
 *
 * \param[in] inputUnreconciledAttrListArray Array containing the unreconciled
 * NvSciSyncAttrList(s) to be appended together.
 * Valid value: Array of unreconciled NvSciSyncAttrList(s) where the array
 * size is at least 1.
 * \param[in] inputUnreconciledAttrListCount Number of unreconciled
 * NvSciSyncAttrList(s) in @a inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if it
 * is non-zero.
 * \param[out] newUnreconciledAttrList Appended NvSciSyncAttrList created out of
 * the input unreconciled NvSciSyncAttrList(s). The output NvSciSyncAttrList is
 * non-writable.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a inputUnreconciledAttrListArray is NULL
 *         - @a inputUnreconciledAttrListCount is 0
 *         - an NvSciSyncAttrList appears multiple times in
 *           @a inputUnreconciledAttrListArray
 *         - @a newUnreconciledAttrList is NULL
 *         - any of the input NvSciSyncAttrLists in
 *           @a inputUnreconciledAttrListArray are not unreconciled
 *         - not all the NvSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *           are bound to the same NvSciSyncModule instance.
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory to
 *   create the new unreconciled NvSciSyncAttrList.
 * - ::NvSciError_Overflow if the combined slot counts of all the input
 *   NvSciSyncAttrLists exceeds UINT64_MAX
 * - ::NvSciError_InvalidState if no more references can be taken for
 *   NvSciSyncModule associated with the NvSciSyncAttrList in @a
 *   inputUnreconciledAttrListArray to create the new NvSciSyncAttrList.
 * - Panics if any of the input NvSciSyncAttrLists are not valid
 *
 * @pre @id{NvSciSyncAttrListAppendUnreconciled_PreCond_001} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 * @post @id{NvSciSyncAttrListAppendUnreconciled_PostCond_001}
 *  Appended NvSciSyncAttrList is read-only and the attribute values in the
 * list cannot be modified using set attribute APIs.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListAppendUnreconciled(
    const NvSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    NvSciSyncAttrList* newUnreconciledAttrList);

/**
 * \brief Clones an NvSciSyncAttrList. The cloned NvSciSyncAttrList will
 * contain slot count, reconciliation type and all the attribute values of the
 * original NvSciSyncAttrList.
 *
 * \param[in] origAttrList NvSciSyncAttrList to be cloned.
 * \param[out] newAttrList The new NvSciSyncAttrList.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if @a origAttrList or @a newAttrList is NULL.
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory
 * - ::NvSciError_InvalidState if no more references can be taken for
 *     NvSciSyncModule associated with @a origAttrList to create the new
 *     NvSciSyncAttrList.
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 *   to create a NvSciSyncAttrList.
 * - Panic if @a origAttrList is not valid
 *
 * @pre @id{NvSciSyncAttrListClone_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListClone_PostCond_001}
 *  If the original NvSciSyncAttrList is unreconciled, then modification will be
 * allowed on the cloned NvSciSyncAttrList using set attributes APIs even if the
 * attributes had been set in the original NvSciSyncAttrList, but the calls to
 * set attributes in either NvSciSyncAttrList will not affect the other.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListClone(
    NvSciSyncAttrList origAttrList,
    NvSciSyncAttrList* newAttrList);

/**
 * \brief Gets the value(s) of NvSciSyncAttrKey(s) from an NvSciSyncAttrList at
 * given slot index in a multi-slot unreconciled NvSciSyncAttrList.
 *
 *
 * \param[in] attrList NvSciSyncAttrList to retrieve the NvSciSyncAttrKey and value
 * from.
 * \param[in] slotIndex Index in the NvSciSyncAttrList.
 * Valid value: 0 to slot count of NvSciSyncAttrList - 1.
 * \param[in,out] pairArray Array of NvSciSyncAttrKeyValuePair. Holds the
 * NvSciSyncAttrKey(s) passed into the function and returns an array of
 * NvSciSyncAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every NvSciSyncAttrKeyValuePair in the array > NvSciSyncAttrKey_LowerBound
 * and < NvSciSyncAttrKey_UpperBound.
 * \param[in] pairCount Indicates the number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a slotIndex is not a valid slot in @a attrList
 *         - @a attrList is NULL
 *         - @a pairArray is NULL
 *         - @a pairCount is 0
 *         - any of the keys in @a pairArray is not a NvSciSyncAttrKey
 * - Panics if @a attrList is not valid
 *
 * @pre @id{NvSciSyncAttrListSlotGetAttrs_PreCond_001} @asil{B}
 *  Valid NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListSlotGetAttrs_PostCond_001}
 *  If an attribute was never set, the corresponding value will be set to NULL
 * and length to 0.
 * @arr @id{NvSciSyncAttrListSlotGetAttrs_RES_001} @asil{D}
 *  The returned pairArray consists of const void* pointers to the actual attribute
 * values from NvSciSyncAttrList. The application must not overwrite this data.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListSlotGetAttrs(
    NvSciSyncAttrList attrList,
    size_t slotIndex,
    NvSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * \brief Reconciles the input unreconciled NvSciSyncAttrLists into a new
 * reconciled NvSciSyncAttrList.
 *
 */
#if (NV_IS_SAFETY == 0)
/**
 * On reconciliation failure, this API call allocates memory for the conflicting
 * NvSciSyncAttrList which has to be freed by the caller using
 * NvSciSyncAttrListFree().
 *
 */
#endif
/**
 * \param[in] inputArray Array containing unreconciled NvSciSyncAttrLists to be
 * reconciled.
 * Valid value: Array of valid NvSciSyncAttrLists where the array size is at least 1
 * \param[in] inputCount The number of unreconciled NvSciSyncAttrLists in
 * @a inputArray.
 * Valid value: inputCount is valid input if is non-zero.
 * \param[out] newReconciledList Reconciled NvSciSyncAttrList.
 */
#if (NV_IS_SAFETY == 0)
/**
 * \param[out] newConflictList Unreconciled NvSciSyncAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * \param[out] newConflictList Unused.
 */
#endif
/**
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - inputCount is 0
 *         - any of the input NvSciSyncAttrLists are not unreconciled
 *         - not all the NvSciSyncAttrLists in inputArray are bound to the same
 *           NvSciSyncModule instance.
 *         - any of the attributes in any of the input NvSciSyncAttrLists has
 *           an invalid value for that attribute
 *         - inputArray is NULL
 *         - newReconciledList is NULL
 *         - an NvSciSyncAttrList appears multiple times in inputArray
 */
#if (NV_IS_SAFETY == 0)
/**        - newConflictList is NULL
 */
#endif
/**
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if a new NvSciSyncAttrList cannot be associated
 *   with the NvSciSyncModule associated with the NvSciSyncAttrList(s) in the
 *   given @a inputArray to create a new reconciled NvSciSyncAttrList
 * - ::NvSciError_Overflow if internal integer overflow is detected.
 * - ::NvSciError_ReconciliationFailed if reconciliation failed because
 *     of conflicting attributes
 * - ::NvSciError_UnsupportedConfig if any of the following occurs:
 *      - there is an attribute mismatch between signaler and waiters
 *      - an unsuported combination of attributes is requested
 * - Panics if any of the input NvSciSyncAttrLists is not valid
 *
 * @pre @id{NvSciSyncAttrListReconcile_PreCond_001} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 * @post @id{NvSciSyncAttrListReconcile_PostCond_001}
 *  newReconciledList is populated only if the reconciliation is successful.
 * @arr @id{NvSciSyncAttrListReconcile_REC_001} @asil{QM}
 *  On success, this API call allocates memory for the reconciled NvSciSyncAttrList
 * which has to be freed by the caller using NvSciSyncAttrListFree().
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListReconcile(
    const NvSciSyncAttrList inputArray[],
    size_t inputCount,
    NvSciSyncAttrList* newReconciledList,
    NvSciSyncAttrList* newConflictList);

#if (NV_IS_SAFETY == 0)
/**
 * @if (SWDOCS_NVSCISYNC_NOTSUPPORT)
 *
 * \brief Dumps the NvSciSyncAttrList into a binary descriptor.
 *
 * @note This API can be used for debugging purpose.
 *
 * \param[in] attrList NvSciSyncAttrList to create the blob from.
 * \param[out] buf A pointer to binary descriptor buffer.
 * \param[out] len The length of the binary descriptor buffer created.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 * - ::NvSciError_InsufficientMemory if memory allocation failed
 * - Panics if attrList is not valid
 *
 * @note This API is deprecated and will be removed in a future version. This
 * prototype is only provided to not break compiliation of older code. Its use
 * is not supported. Do not rely on using this API.
 *
 * @pre
 * - Valid NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @endif
 */
/// @cond (SWDOCS_NVSCISYNC_NOTSUPPORT)
NvSciError NvSciSyncAttrListDebugDump(
    NvSciSyncAttrList attrList,
    void** buf,
    size_t* len);
/// @endcond
#endif

/**
 * \brief Transforms the input unreconciled NvSciSyncAttrList(s) to an exportable
 * unreconciled NvSciSyncAttrList descriptor that can be transported by the
 * application to any remote process as a serialized set of bytes over an
 * NvSciIpc channel.
 *
 * \param[in] unreconciledAttrListArray NvSciSyncAttrList(s) to be exported.
 * Valid value: Array of valid NvSciSyncAttrList(s) where the array
 * size is at least 1.
 * \param[in] unreconciledAttrListCount Number of NvSciSyncAttrList(s) in
 * @a unreconciledAttrListArray.
 * Valid value: unreconciledAttrListCount is valid input if it
 * is non-zero.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller may
 * send the exported unreconciled NvSciSyncAttrList descriptor.
 * \param[out] descBuf A pointer to the new unreconciled NvSciSyncAttrList
 * descriptor, which the caller can deallocate later using
 * NvSciSyncAttrListFreeDesc().
 * \param[out] descLen The size of the new unreconciled NvSciSyncAttrList
 * descriptor.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a unreconciledAttrListCount is 0
 *      - @a unreconciledAttrListArray is NULL
 *      - @a ipcEndpoint is not a valid NvSciIpcEndpoint
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 *      - any of the input NvSciSyncAttrLists is not unreconciled
 *      - Not all of the NvSciSyncAttrLists in @a unreconciledAttrListArray
 *        are bound to the same NvSciSyncModule instance.
 *      - an NvSciSyncAttrList appears multiple times in
 *        @a unreconciledAttrListArray
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if no more references can be taken for the
 *     NvSciSyncModule associated with @a unreconciledAttrListArray.
 * - ::NvSciError_Overflow if the combined slot count of all the unreconciled
 *   NvSciSyncAttrList exceeds UINT64_MAX
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 */
#if (NV_IS_SAFETY == 0)
/**
 *   or there was a problem with NvSciIpc
*/
#endif
/**
 * - ::NvSciError_NoSpace if no space is left in transport buffer to append the
 *        key-value pair.
 * - Panic if any of the input NvSciSyncAttrLists is not valid.
 *
 * @pre @id{NvSciSyncAttrListIpcExportUnreconciled_PreCond_001} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 * @post @id{NvSciSyncAttrListIpcExportUnreconciled_PostCond_001}
 *  When exporting an array containing multiple unreconciled
 * NvSciSyncAttrLists, the importing endpoint still imports just one unreconciled
 * NvSciSyncAttrList. This unreconciled NvSciSyncAttrList is referred to as a
 * multi-slot NvSciSyncAttrList. It logically represents an array of
 * NvSciSyncAttrLists, where each key has an array of values, one per slot.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListIpcExportUnreconciled(
    const NvSciSyncAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    NvSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * \brief Transforms the reconciled NvSciSyncAttrList to an exportable reconciled
 * NvSciSyncAttrList descriptor that can be transported by the application to any
 * remote process as a serialized set of bytes over an NvSciIpc channel.
 *
 * \param[in] reconciledAttrList The NvSciSyncAttrList to be exported.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller may
 * send the exported reconciled NvSciSyncAttrList descriptor.
 * \param[out] descBuf A pointer to the new reconciled NvSciSyncAttrList
 * descriptor, which the caller can deallocate later using
 * NvSciSyncAttrListFreeDesc().
 * \param[out] descLen The size of the new reconciled NvSciSyncAttrList
 * descriptor.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - @a ipcEndpoint is not a valid NvSciIpcEndpoint
 *         - @a reconciledAttrList does not correspond to a waiter or a signaler
 *         - @a reconciledAttrList is not a reconciled NvSciSyncAttrList
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - ::NvSciError_InvalidState if no more references can be taken for
 *     NvSciSyncModule associated with @a reconciledAttrList to create the new
 *     NvSciSyncAttrList.
 * - Panic if @a reconciledAttrList is not valid
 *
 * @pre @id{NvSciSyncAttrListIpcExportReconciled_PreCond_001} @asil{QM}
 *  Valid reconciled NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListIpcExportReconciled(
    const NvSciSyncAttrList reconciledAttrList,
    NvSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * \brief Transforms an exported unreconciled NvSciSyncAttrList descriptor
 * (potentially received from any process) into an unreconciled
 * NvSciSyncAttrList which is no longer writable.
 *
 * \param[in] module The NvSciSyncModule instance with which to associate the
 * imported NvSciSyncAttrList.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller receives
 * the exported unreconciled NvSciSyncAttrList descriptor.
 * \param[in] descBuf The unreconciled NvSciSyncAttrList descriptor to be
 * translated into an unreconciled NvSciSyncAttrList. It should be the result of
 * NvSciSyncAttrListIpcExportUnreconciled
 * Valid value: descBuf is valid input if it is non-NULL.
 * \param[in] descLen The size of the unreconciled NvSciSyncAttrList descriptor.
 * Valid value: descLen is valid input if it is not 0.
 * \param[out] importedUnreconciledAttrList Imported unreconciled NvSciSyncAttrList.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - @a descLen is 0
 *         - @a ipcEndpoint is not a valid NvSciIpcEndpoint
 *         - array of bytes indicated by @a descBuf and @a descLen
 *           do not constitute a valid exported NvSciSyncAttrList descriptor
 *           for an unreconciled NvSciSyncAttrList
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_Overflow if internal integer overflow is detected.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a module is not valid
 *
 * @pre @id{NvSciSyncAttrListIpcImportUnreconciled_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 * @post @id{NvSciSyncAttrListIpcImportUnreconciled_PostCond_001}
 *  Imported NvSciSyncAttrList is no longer writable.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListIpcImportUnreconciled(
    NvSciSyncModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    NvSciSyncAttrList* importedUnreconciledAttrList);

/**
 * \brief Translates an exported reconciled NvSciSyncAttrList descriptor
 * (potentially received from any process) into a reconciled NvSciSyncAttrList.
 *
 * \param[in] module The NvSciSyncModule instance with which to associate the
 * imported NvSciSyncAttrList.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller
 *            receives the exported reconciled NvSciSyncAttrList descriptor.
 * \param[in] descBuf The reconciled NvSciSyncAttrList descriptor to be
 * transformed into a reconciled NvSciSyncAttrList.
 * Valid value: descBuf is valid if it is non-NULL.
 * \param[in] descLen The size of the reconciled NvSciSyncAttrList descriptor.
 * Valid value: descLen is valid if it is not 0.
 * \param[in] inputUnreconciledAttrListArray The array of NvSciSyncAttrLists against
 * which the new NvSciSyncAttrList is to be validated.
 * Valid value: Array of valid NvSciSyncAttrList(s)
 * \param[in] inputUnreconciledAttrListCount The number of NvSciSyncAttrLists in
 * inputUnreconciledAttrListArray. If inputUnreconciledAttrListCount is
 * non-zero, then this operation will fail with an error unless all the
 * constraints of all the NvSciSyncAttrLists in inputUnreconciledAttrListArray are
 * met by the imported NvSciSyncAttrList.
 * Valid value: [0, SIZE_MAX]
 * \param[out] importedReconciledAttrList Imported NvSciSyncAttrList.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is not a valid NvSciIpcEndpoint
 *      - the array of bytes indicated by @a descBuf and @a descLen do not
 *        constitute a valid exported NvSciSyncAttrList descriptor for a
 *        reconciled NvSciSyncAttrList
 *      - @a inputUnreconciledAttrListArray is NULL but
 *        @a inputUnreconciledAttrListCount is not 0
 *      - any of the NvSciSyncAttrLists in inputUnreconciledAttrListArray are
 *        not unreconciled
 *      - any of the NvSciSyncAttrLists in @a inputUnreconciledAttrListArray
 *        is not bound to @a module.
 *      - an NvSciSyncAttrList appears multiple times in
 *        @a inputUnreconciledAttrListArray
 *      - @a importedReconciledAttrList is NULL
 * - ::NvSciError_AttrListValidationFailed if the NvSciSyncAttrList to be
 *   imported either would not be a reconciled NvSciSyncAttrList or would not meet
 *   at least one of constraints in one of the input unreconciled
 *   NvSciSyncAttrLists.
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if no more references can be taken on the
 *     NvSciSyncModule.
 * - ::NvSciError_Overflow if internal integer overflow is detected.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a module or any of the input NvSciSyncAttrLists are not valid
 *
 * @pre @id{NvSciSyncAttrListIpcImportReconciled_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 * @pre @id{NvSciSyncAttrListIpcImportReconciled_PreCond_002} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 * @post @id{NvSciSyncAttrListIpcImportReconciled_PostCond_001}
 *  NvSciSyncAttrListIpcImportReconciled() also validates that the
 * reconciled NvSciSyncAttrList to be imported will
 * be a reconciled NvSciSyncAttrList that is consistent with the constraints in
 * an array of input unreconciled NvSciSyncAttrList(s).
 * @arr @id{NvSciSyncAttrListIpcImportReconciled_REC_001} @asil{QM}
 *  It is recommended to provide the unreconciled NvSciSyncAttrList(s) to
 * this API so NvSciSync can validate.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListIpcImportReconciled(
    NvSciSyncModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const NvSciSyncAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    NvSciSyncAttrList* importedReconciledAttrList);

/**
 * \brief Frees an exported NvSciSyncAttrList descriptor previously returned by
 * any NvSciSyncAttrList exporting function.
 *
 * \param[in] descBuf The exported NvSciSyncAttrList descriptor to be freed.
 * The valid value is non-NULL.
 * \return void
 * - Panics if Init Mode API is called in Runtime Mode.
 *
 * @pre @id{NvSciSyncAttrListFreeDesc_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrList export descriptor is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the input @a descBuf
 *        to be freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void NvSciSyncAttrListFreeDesc(
    void* descBuf);

/**
 * \brief Frees any resources allocated for the NvSciSyncFence.
 *
 * \param[in,out] syncFence A pointer to NvSciSyncFence.
 * \return void
 * - Panics if the NvSciSyncObj associated with @a syncFence is not valid
 *
 * @pre @id{NvSciSyncFenceClear_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @post @id{NvSciSyncFenceClear_PostCond_001}
 *  The memory pointed to by the NvSciSyncFence is guaranteed to be all
 * zeros and thus the NvSciSyncFence is returned to the cleared state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the @a syncFence to
 *        be cleared
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
void NvSciSyncFenceClear(
    NvSciSyncFence* syncFence);

/**
 * \brief Duplicates the given NvSciSyncFence, such that any wait on duplicated
 * NvSciSyncFence will complete at the same time as a wait on given
 * NvSciSyncFence.
 *
 * \param[in] srcSyncFence NvSciSyncFence to duplicate.
 * \param[out] dstSyncFence duplicated NvSciSyncFence.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any argument is NULL or both parameters
     point to the same NvSciSyncFence.
 * - ::NvSciError_InvalidState if no more references can be taken
 *      for NvSciSyncObj associated with @a srcSyncFence
 * - Panics if the NvSciSyncObj associated with @a srcSyncFence
 *   or @a dstSyncFence are not valid
 *
 * @pre @id{NvSciSyncFenceDup_PreCond_001} @asil{B}
 *  Both arguments are pointers to valid NvSciSyncFences.
 * @post @id{NvSciSyncFenceDup_PostCond_001}
 * The given NvSciSyncFence will be cleared before the duplication.
 * @post @id{NvSciSyncFenceDup_PostCond_002}
 *  If the given NvSciSyncFence holds any reference on a NvSciSyncObj, then the
 * duplicated NvSciSyncFence will create an additional reference on it.
 * @post @id{NvSciSyncFenceDup_PostCond_003}
 *  If the given NvSciSyncFence is in a cleared state, then so
 * also will be the duplicated NvSciSyncFence.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the @a dstSyncFence
 *        if it had previously been associated with an NvSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceDup(
    const NvSciSyncFence* srcSyncFence,
    NvSciSyncFence* dstSyncFence);

/**
 * \brief Extracts the id and value from the input NvSciSyncFence.
 *
 * \param[in] syncFence NvSciSyncFence from which the id and value should be
 * retrieved
 * \param[out] id NvSciSyncFence id
 * \param[out] value NvSciSyncFence value
 *
 * \return NvSciError
 * - NvSciError_Success if successful
 * - NvSciError_BadParameter if syncFence is NULL or invalid or id/value are NULL
 * - NvSciError_ClearedFence if syncFence is a valid cleared NvSciSyncFence
 * - Panics if the NvSciSyncObj associated with the syncFence is invalid
 *
 * @pre @id{NvSciSyncFenceExtractFence_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceExtractFence(
    const NvSciSyncFence* syncFence,
    uint64_t* id,
    uint64_t* value);

/**
 * \brief Populates the input NvSciSyncFence based on the input id and value.
 *
 * \param[in] syncObj valid NvSciSyncObj
 * \param[in] id NvSciSyncFence identifier
 * Valid value: [0, UINT32_MAX-1] for NvSciSyncAttrValPrimitiveType_Syncpoint.
 * [0, value returned by NvSciSyncObjGetNumPrimitives()-1] for
 * NvSciSyncAttrValPrimitiveType_SysmemSemaphore and
 * NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in] value NvSciSyncFence value
 * Valid value: [0, UINT32_MAX] for NvSciSyncAttrValPrimitiveType_Syncpoint
 * and NvSciSyncAttrValPrimitiveType_SysmemSemaphore.
 * [0, UINT64_MAX] for
 * NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b.
 * \param[in,out] syncFence NvSciSyncFence to populate
 *
 * \return NvSciError
 * - NvSciError_Success if successful.
 * - NvSciError_BadParameter if syncObj or syncFence is NULL.
 * - NvSciError_Overflow if id is invalid.
 * - NvSciError_Overflow if value is invalid.
 * - NvSciError_InvalidState if no more references can be taken on
 *   the syncObj
 * - Panics if syncObj or NvSciSyncObj initially associated with syncFence
 *   is invalid
 *
 * @pre @id{NvSciSyncFenceUpdateFence_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 * @pre @id{NvSciSyncFenceUpdateFence_PreCond_002} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @post @id{NvSciSyncFenceUpdateFence_PostCond_001}
 *  This new NvSciSyncFence is associated with the input NvSciSyncObj. The input
 * NvSciSyncFence is cleared before being populated with the new data.
 * @arr @id{NvSciSyncFenceUpdateFence_REC_001} @asil{QM}
 *  The task status slot associated with this fence will be 0. It is recommended
 * to not use this interface when using task status buffer.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the @a syncFence to
 *        be updated
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceUpdateFence(
    NvSciSyncObj syncObj,
    uint64_t id,
    uint64_t value,
    NvSciSyncFence* syncFence);

/**
 * \brief Sets the current timestamp slot index in the NvSciSyncFence
 *
 * \param[in,out] syncFence object of type NvSciSyncFence
 * \param[in] timestampSlot index of the timestamp slot to set in NvSciSyncFence.
 * Valid value: [0, number-of-timestamp-slots - 1]
 *
 * \return ::NvSciError
 * - ::NvSciError_Success if successful
 * - ::NvSciError_ClearedFence if @a syncFence is cleared
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a syncFence is NULL
 *         - timestamps are not supported by NvSciSyncObj associated with the @a syncFence
 *         - @a timestampSlot is invalid slot index
 * - Panics if NvSciSyncObj associated with @a syncFence is not valid.
 *
 * @pre @id{NvSciSyncFenceAddTimestampSlot_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceAddTimestampSlot(
    NvSciSyncFence* syncFence,
    uint32_t timestampSlot);

/**
 * \brief Sets the current task status slot index to the NvSciSyncFence
 *
 * \param[in,out] syncFence object of type NvSciSyncFence
 * \param[in] taskStatusSlot index of the task status slot to set in NvSciSyncFence.
 * Valid value: [0, number-of-task-status-slots - 1]
 *
 * \return ::NvSciError
 * - ::NvSciError_Success if successful
 * - ::NvSciError_ClearedFence if @a syncFence is cleared
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a syncFence is NULL
 *         - task status is not supported by NvSciSyncObj associated with @a syncFence
 *         - @a taskStatusSlot is invalid slot index
 * - Panics if NvSciSyncObj associated with @a syncFence is not valid.
 *
 * @pre @id{NvSciSyncFenceAddTaskStatusSlot_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceAddTaskStatusSlot(
    NvSciSyncFence* syncFence,
    uint32_t taskStatusSlot);

/**
 * \brief Reads the current timestamp slot index from the NvSciSyncFence
 *
 * \param[in] syncFence object of type NvSciSyncFence
 * \param[out] timestampSlot index of the timestmp slot in NvSciSyncFence.
 *
 * \return ::NvSciError
 * - ::NvSciError_Success if successful
 * - ::NvSciError_ClearedFence if @a syncFence is cleared
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - timestamps are not supported by NvSciSyncObj associated with
 *         - @a syncFence does not support timestamps
 * - Panics if NvSciSyncObj associated with @a syncFence is not valid.
 *
 * @pre @id{NvSciSyncFenceExtractTimestampSlot_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceExtractTimestampSlot(
    NvSciSyncFence* syncFence,
    uint32_t* timeStampSlot);

/**
 * \brief Allocates and initializes a @ref NvSciSyncObj that meets all the
 * constraints specified in the given reconciled NvSciSyncAttrList.
 *
 * \param[in] reconciledList A reconciled NvSciSyncAttrList.
 * \param[out] syncObj The allocated @ref NvSciSyncObj.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_InvalidState if not enough references remain on the
 *      reconciled NvSciSyncAttrList
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - @a reconciledList is not a reconciled NvSciSyncAttrList
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a reconciledList is not valid
 *
 * @pre @id{NvSciSyncObjAlloc_PreCond_001} @asil{QM}
 *  Valid reconciled NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncObjAlloc_PostCond_001}
 *  The resulting object will hold a buffer of 1024 slots
 * for holding tasks status.
 * @arr @id{NvSciSyncObjAlloc_REC_001} @asil{QM}
 *  This function does not take ownership of the reconciled NvSciSyncAttrList.
 * The caller remains responsible for freeing the reconciled NvSciSyncAttrList.
 * The caller may free the reconciled NvSciSyncAttrList any time after this
 * function is called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjAlloc(
    NvSciSyncAttrList reconciledList,
    NvSciSyncObj* syncObj);

/**
 * \brief Creates a new @ref NvSciSyncObj holding a reference to the original
 * resources to which the input @ref NvSciSyncObj holds reference to.
 *
 * \param[in] syncObj NvSciSyncObj to duplicate.
 * \param[out] dupObj Duplicated NvSciSyncObj.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any argument is NULL.
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if the number of references to the
 *   synchronization object of the input NvSciSyncObj is INT32_MAX and the
 *   newly duplicated NvSciSyncObj tries to take one more reference using this
 *   API.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncObjDup_PreCond_001} Valid NvSciSyncObj is obtained. @asil{QM}
 * @post @id{NvSciSyncObjDup_PostCond_001}
 *  The duplicated NvSciSyncObj is not a completely new NvSciSyncObj. Therefore,
 * signaling and generating NvSciSyncFences from one affects the state of the
 * other, because it is the same underlying NvSciSyncObj.
 * @arr @id{NvSciSyncObjDup_REC_001} @asil{QM}
 *  The resulting NvSciSyncObj must be freed separately by the user.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjDup(
    NvSciSyncObj syncObj,
    NvSciSyncObj* dupObj);

/**
 * \brief Retrieves the reconciled NvSciSyncAttrList associated with an
 * input NvSciSyncObj.
 *
 * \param[in] syncObj Handle corresponding to NvSciSyncObj from which the
 * NvSciSyncAttrList has to be retrieved.
 * \param[out] syncAttrList pointer to the retrieved NvSciSyncAttrList.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if @a syncObj or @a syncAttrList is NULL.
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncObjGetAttrList_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 * @post @id{NvSciSyncObjGetAttrList_PostCond_001}
 *  The retrieved NvSciSyncAttrList from an NvSciSyncObj is read-only
 * and the attribute values in the list cannot be modified using set attribute APIs.
 * @arr @id{NvSciSyncObjGetAttrList_RES_001} @asil{D}
 *  The retrieved NvSciSyncAttrList must not be freed with NvSciSyncAttrListFree().
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncObjGetAttrList(
    NvSciSyncObj syncObj,
    NvSciSyncAttrList* syncAttrList);

/**
 * \brief Destroys a valid @ref NvSciSyncObj and frees any resources that were
 * allocated for it.
 *
 * \param[in] syncObj NvSciSyncObj to be freed.
 *
 * \return void
 * - Panics if:
 *   - Init Mode API is called in Runtime Mode.
 *   - @a syncObj is not a valid @ref NvSciSyncObj
#if 0
 *   or there was an unexpected freeing error from C2C
#endif
 *
 * @pre @id{NvSciSyncObjFree_PreCond_001} @asil{QM}
 *  Valid NvSciSyncObj is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the
 *        NvSciSyncAttrList obtained from NvSciSyncObjGetAttrList() to be
 *        freed, since the lifetime of that reconciled NvSciSyncAttrList is
 *        tied to the associated NvSciSyncObj
 *      - Provided there is no active operation involving the input
 *        NvSciSyncObj @a syncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \implements{11273441}
 */
void NvSciSyncObjFree(
    NvSciSyncObj syncObj);

/**
 * \brief Exports an NvSciSyncObj into an NvSciIpc-transferable object
 * binary descriptor.
 *
 * The binary descriptor can be transferred to a Waiter to create a matching
 * NvSciSyncObj.
 *
 * \param[in] syncObj A NvSciSyncObj to export.
 * \param[in] permissions Flag indicating the expected NvSciSyncAccessPerm.
 * Valid value: any value of NvSciSyncAccessPerm
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller
 *            intends to transfer the exported NvSciSyncObj descriptor.
 * \param[out] desc NvSciSync fills in this caller-supplied descriptor with
 *             the exported form of NvSciSyncObj that is to be shared across
 *             an NvSciIpc channel.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_ResourceError if something went wrong with NvSciIpc
 * - ::NvSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a desc is NULL
 *      - @a syncObj is NULL
 *      - @a permissions is invalid
 *      - @a permissions contains larger permissions than those set on
 *        NvSciSyncAttrKey_ActualPerm on the reconciled NvSciSyncAttrList
 *        associated with the NvSciSyncObj granted to this peer
 *      - @a permissions contains smaller permissions than the expected
 *        permissions requested by the receiving peer
 *      - @a ipcEndpoint is invalid
 *      - @a ipcEndpoint does not lead to a peer in the topology tree
 *        of this NvSciSyncObj
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 */
#if (NV_L4T == 1)
/**
 * - ::NvSciError_NotSupported if trying to export syncpoint signaling
 *   over a C2C Ipc channel.
 */
#endif
/**
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncObjIpcExport_PreCond_001} @asil{QM}
 *  Valid NvSciSyncObj is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjIpcExport(
    NvSciSyncObj syncObj,
    NvSciSyncAccessPerm permissions,
    NvSciIpcEndpoint ipcEndpoint,
    NvSciSyncObjIpcExportDescriptor* desc);

/**
 * \brief Creates and returns an @ref NvSciSyncObj based on the supplied binary
 * descriptor describing an exported @ref NvSciSyncObj.
 *
 * This function is called from the waiter after it receives the binary
 * descriptor from the signaler who has created and exported the binary
 * descriptor.
 *
 * \param[in] ipcEndpoint The @ref NvSciIpcEndpoint through which the caller
 *            received the exported NvSciSyncObj descriptor.
 * \param[in] desc The exported form of @ref NvSciSyncObj received through the
 *            NvSciIpc channel.
 * Valid value: desc is valid if it is non-NULL
 * \param[in] inputAttrList The reconciled NvSciSyncAttrList returned by
 *            @ref NvSciSyncAttrListIpcImportReconciled.
 * \param[in] permissions NvSciSyncAccessPerm indicating the expected access
 * permissions.
 * Valid value: any value of NvSciSyncAccessPerm
 * \param[in] timeoutUs Unused
 * \param[out] syncObj The Waiter's NvSciSyncObj.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_ResourceError if something went wrong with NvSciIpc
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a ipcEndpoint is invalid
 *      - @a desc is invalid
 *      - @a inputAttrList is NULL
 *      - @a permissions is invalid
 *      - @a syncObj is NULL
 *      - @a permissions is NvSciSyncAccessPerm_Auto but permissions granted
 *        in @a desc are not enough to satisfy expected permissions stored in
 *        @a inputAttrList
 *      - the NvSciSyncObjIpcExportDescriptor export descriptor corresponds to
 *        a descriptor generated by an incompatible NvSciSync library version
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if the imported NvSciSyncObj cannot be associated
 *   with the NvSciSyncModule associated with the reconciled input
 *   NvSciSyncAttrList.
 * - ::NvSciError_Overflow if @a desc is too big to be imported.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - ::NvSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panics if @a inputAttrList is not valid
 *
 * @pre @id{NvSciSyncObjIpcImport_PreCond_001} @asil{QM}
 *  Valid reconciled NvSciSyncAttrList is obtained.
 * @arr @id{NvSciSyncObjIpcImport_REC_001} @asil{QM}
 *  NvSciSyncObjIpcImport() does not take ownership of input NvSciSyncAttrList.
 * The caller remains responsible for freeing input NvSciSyncAttrList. The caller
 * may free the input NvSciSyncAttrList any time after this function is called.
 * @arr @id{NvSciSyncObjIpcImport_RES_002} @asil{B}
 *  Input NvSciSyncObjIpcExportDescriptor can be used to import NvSciSyncObj only once
 * i.e the same desc cannot be used to import syncObj multiple times.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjIpcImport(
    NvSciIpcEndpoint ipcEndpoint,
    const NvSciSyncObjIpcExportDescriptor* desc,
    NvSciSyncAttrList inputAttrList,
    NvSciSyncAccessPerm permissions,
    int64_t timeoutUs,
    NvSciSyncObj* syncObj);

/**
 * \brief Exports the input NvSciSyncFence into a binary descriptor shareable
 * across the NvSciIpc channel.
 *
 * The resulting descriptor of a non-cleared NvSciSyncFence is associated with
 * NvSciSyncObj associated with the NvSciSyncFence. After transporting
 * the descriptor via an Ipc path, NvSciSync will be able to recognize
 * that the NvSciSyncFence is associated with this NvSciSyncObj if NvSciSyncObj
 * traversed the same Ipc path.
 *
 * \param[in] syncFence A pointer to NvSciSyncFence object to be exported.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller may
 *            send the exported fence descriptor.
 * \param[out] desc The exported form of NvSciSyncFence shared across
 *             an NvSciIpc channel.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - any argument is NULL
 *      - ipcEndpoint is not a valid NvSciIpcEndpoint
 * - Panics if @a syncFence is associated an invalid NvSciSyncObj
 *
 * @pre @id{NvSciSyncIpcExportFence_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncIpcExportFence(
    const NvSciSyncFence* syncFence,
    NvSciIpcEndpoint ipcEndpoint,
    NvSciSyncFenceIpcExportDescriptor* desc);

/**
 * \brief Fills in the NvSciSyncFence based on the input
 * binary descriptor. If the NvSciSyncFence descriptor does not describe a
 * cleared NvSciSyncFence, then NvSciSync will validate if it corresponds to the
 * NvSciSyncObj and it will associate the out NvSciSyncFence with the
 * NvSciSyncObj.
 *
 * \param[in] syncObj The NvSciSyncObj.
 * \param[in] desc The exported form of NvSciSyncFence.
 *  Valid value: A binary descriptor produced by NvSciSyncIpcExportFence.
 * \param[out] syncFence A pointer to NvSciSyncFence object.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - any argument is NULL
 *      - fence descriptor does not describe a cleared NvSciSyncFence but
 *        it is associated with an NvSciSyncObj different from @a syncObj
 *      - fence descriptor's value exceeds allowed range for syncObj's primitive
 * - ::NvSciError_InvalidState if @a syncObj cannot take more references.
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncIpcImportFence_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 * @pre @id{NvSciSyncIpcImportFence_PreCond_002} @asil{B}
 *  Valid NvSciSyncFence is obtained.
 * @post @id{NvSciSyncIpcImportFence_PostCond_001}
 *  The NvSciSyncFence will be cleared first, removing any previous reference to a NvSciSyncObj.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the @a syncFence if
 *        it had previously been associated with an NvSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncIpcImportFence(
    NvSciSyncObj syncObj,
    const NvSciSyncFenceIpcExportDescriptor* desc,
    NvSciSyncFence* syncFence);

/**
 * @brief Validates the NvSciSyncObj satisfies the constraints of the
 * NvSciSyncAttrList that it is associated with.
 */
#if (NV_IS_SAFETY == 0)
/**
 * \param[in] syncObj Unused
 *
 * \return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success
 *
 * @note This is a no-op.
 */
#else
/**
 * \param[in] syncObj The NvSciSyncObj to validate
 *
 * \return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a syncObj is NULL
 *      - @a syncObj is invalid
 *      - reconciled list associated with @a syncObj is not validated.
 *      - @a syncObj does not satisfy the constraints of the NvSciSyncAttrList
 *        that it is associated with
 * - ::NvSciError_Revalidation_Success if any of the following occurs:
 *      - the API is called after the provided @a syncObj has already been
 *        validated
 */
#endif
/**
 * @pre @id{NvSciSyncObjValidate_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 * @post @id{NvSciSyncObjValidate_PostCond_001}
 *  NvSciSyncObj is marked as validated if successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjValidate(
    NvSciSyncObj syncObj);

/**
 * \brief Generates next point on sync timeline of an NvSciSyncObj and fills
 * in the supplied NvSciSyncFence object.
 *
 * This function can be used when the CPU is the Signaler.
 *
 * \param[in] syncObj A valid NvSciSyncObj.
 * \param[out] syncFence NvSciSyncFence to be filled
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - @a syncObj is not a CPU signaler
 */
#if (NV_IS_SAFETY == 0)
/**
 *              and not a C2C signaler
 */
#endif
/**         - @a syncObj does not own the backing primitive
 * - ::NvSciError_InvalidState if the newly created NvSciSyncFence cannot be
 *   associated with the synchronization object of the given NvSciSyncObj.
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncObjGenerateFence_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 * @pre @id{NvSciSyncObjGenerateFence_PreCond_002} @asil{B}
 *  Valid NvSciSyncFence is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *      - Provided there is no active operation involving the @a syncFence if
 *        it had previously been associated with an NvSciSyncObj
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncObjGenerateFence(
    NvSciSyncObj syncObj,
    NvSciSyncFence* syncFence);

/**
 * \brief Signals the @ref NvSciSyncObj using the reconciled primitive that
 * was allocated along with the NvSciSyncObj.
 */
#if (NV_IS_SAFETY == 0)
/**
 * If the signal operation fails, then the timestamp value is undefined.
 */
#endif
/**
 * This function is called when the CPU is the Signaler.
 *
 * \param[in] syncObj A valid NvSciSyncObj to signal.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a syncObj is NULL
 *         - @a syncObj is not a CPU Signaler
 *         - @a syncObj does not own the backing primitive
 * - ::NvSciError_ResourceError if the signal operation fails.
 * - Panics if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncObjSignal_PreCond_001} @asil{B}
 *  Valid NvSciSyncObj is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncObjSignal(
    NvSciSyncObj syncObj);

/**
 * \brief Performs a synchronous wait on the @ref NvSciSyncFence object until the
 * NvSciSyncFence has been signaled or the timeout expires. Any
 * NvSciSyncCpuWaitContext may be used for waiting on any NvSciSyncFence provided
 * they were created in the same NvSciSyncModule context.
 *
 * \param[in] syncFence The NvSciSyncFence to wait on.
 * \param[in] context NvSciSyncCpuWaitContext holding resources needed
 *            to perform waiting.
 * \param[in] timeoutUs Timeout to wait for in micro seconds, -1 for infinite
 *            wait.
 *  Valid value: [-1, NvSciSyncFenceMaxTimeout]
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if any of the following occurs:
 *         - @a syncFence is cleared
 *         - @a syncFence is expired
 *         - @a syncFence has been signaled within the given timeout
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a syncFence is NULL
 *         - @a context is NULL
 *         - @a syncFence and @a context are associated with different
 *           NvSciSyncModule
 *         - @a timeoutUs is invalid
 *         - if caller doesn't have CPU wait permissions in NvSciSyncObj
 *           associated with @a syncFence
 *         - the module reference associated with @a context
 *           is NULL
 * - ::NvSciError_ResourceError if wait operation did not complete
 *   successfully.
 * - ::NvSciError_Timeout if wait did not complete in the given timeout.
 * - ::NvSciError_Overflow if the NvSciSyncFence's id or value are not in range
 *   supported by the primitive this NvSciSyncFence corresponds to.
 * - Panics if any NvSciSyncObj associated with @a syncFence or @a context
 *   is not valid
 *
 * @pre @id{NvSciSyncFenceWait_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @pre @id{NvSciSyncFenceWait_PreCond_002} @asil{B}
 *  Valid NvSciSyncCpuWaitContext is obtained.
 * @post @id{NvSciSyncFenceWait_PostCond_001}
 *  Waiting on a cleared and expired NvSciSyncFence is always not blocking.
 * @arr @id{NvSciSyncFenceWait_RES_003} @asil{B}
 *  One NvSciSyncCpuWaitContext can be used to wait on only one NvSciSyncFence at
 * a time but it can be used to wait on a different NvSciSyncFence at a different time.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceWait(
    const NvSciSyncFence* syncFence,
    NvSciSyncCpuWaitContext context,
    int64_t timeoutUs);

/**
 * \brief Read the timestamp associated with the NvSciSyncFence
 *
 * This function can be used when the CPU is the waiter.
 *
 * \param[in] syncFence object of type NvSciSyncFence
 * \param[out] timestampUS time (in microseconds) when the NvSciSyncFence expired.
 *
 * \return ::NvSciError
 * - ::NvSciError_Success if successful
 * - ::NvSciError_ClearedFence if @a syncFence is cleared
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - any argument is NULL
 *         - timestamps are not supported by NvSciSyncObj associated with
 *           @a syncFence
 *         - @a syncFence does not support timestamps
 *         - the NvSciSyncAttrList associated with the @a syncFence has not
 *           requested CPU access
 * - Panics if NvSciSyncObj associated with @a syncFence is not valid.
 *
 * @pre @id{NvSciSyncFenceGetTimestamp_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @pre @id{NvSciSyncFenceGetTimestamp_PreCond_002} @asil{D}
 *  NvSciSyncFence pointed by syncFence is expired.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceGetTimestamp(
    const NvSciSyncFence* syncFence,
    uint64_t* timestampUS);

/**
 * \brief Reads the task status associated with the NvSciSyncFence and stores
 * it in the user provided out parameter.
 *
 * \param[in] syncFence object of type NvSciSyncFence
 * \param[out] taskStatus user provided struct to store the result
 *
 * \return ::NvSciError
 * - ::NvSciError_Success if successful
 * - ::NvSciError_ClearedFence if @a syncFence is cleared
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a syncFence is NULL or has no non NULL NvSciSyncObj associated
 *         - @a taskStatus is NULL
 *         - syncFence points to an invalid task status slot
 * - Panics if NvSciSyncObj associated with @a syncFence is not valid
 *
 * @pre @id{NvSciSyncFenceGetTaskStatus_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @pre @id{NvSciSyncFenceGetTaskStatus_PreCond_002} @asil{D}
 *  NvSciSyncFence pointed by syncFence is expired.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncFenceGetTaskStatus(
    const NvSciSyncFence* syncFence,
    NvSciSyncTaskStatus* taskStatus);

/*
 * NvSciSync Utility functions
 */

/**
 * \brief Gets the attribute value from the slot 0 of the passed NvSciSyncAttrList
 * with the given NvSciSyncAttrKey.
 *
 * \param[in] attrList NvSciSyncAttrList to retrieve the NvSciSyncAttrKey and
 * value from.
 * \param[in] key NvSciSyncAttrKey for which value to retrieve.
 * Valid value: key is a valid input if it is an input or input/output attribute
 * and it is > NvSciSyncAttrKey_LowerBound and < NvSciSyncAttrKey_UpperBound
 * \param[out] value A pointer to the location where the attribute value is written.
 * \param[out] len Length of the value.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a attrList is NULL
 *         - @a value is NULL
 *         - @a len is NULL
 *         - @a key is not a supported public key
 * - Panics if @a attrList is not valid
 *
 * @pre @id{NvSciSyncAttrListGetAttr_PreCond_001} @asil{B}
 *  Valid NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListGetAttr_PostCond_001}
 *  If an NvSciSyncAttrKey was not set, this function will set *value to NULL and *len to 0.
 * @arr @id{NvSciSyncAttrListGetAttr_RES_001} @asil{D}
 *  The retrieved value must not be freed.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListGetAttr(
    NvSciSyncAttrList attrList,
    NvSciSyncAttrKey key,
    const void** value,
    size_t* len);

/**
 * \brief Reconciles the input unreconciled NvSciSyncAttrList(s) into a new
 * reconciled NvSciSyncAttrList and allocates NvSciSyncObj that meets all the
 * constraints in the reconciled NvSciSyncAttrList. If successful, a reconciled
 * NvSciSyncAttrList will be associated with a newly-allocated @ref NvSciSyncObj
 * that satisfies all the constraints specified in the reconciled
 * NvSciSyncAttrList.
 *
 * Note: This function serves as a convenience function that combines calls to
 * NvSciSyncAttrListReconcile and NvSciSyncObjAlloc.
 *
 * \param[in] inputArray Array containing the unreconciled NvSciSyncAttrList(s)
 *            to reconcile.
 * Valid value: Array of valid NvSciSyncAttrLists where the array size is at least 1
 * \param[in] inputCount Number of unreconciled NvSciSyncAttrLists in
 *            @a inputArray.
 * Valid value: inputCount is valid input if is non-zero.
 * \param[out] syncObj The new NvSciSyncObj.
 */
#if (NV_IS_SAFETY == 0)
/**
 * \param[out] newConflictList unreconciled NvSciSyncAttrList consisting of the
 *             key-value pairs which caused the reconciliation failure.
 * Valid value: This parameter is a valid output parameter only if the return
 *     code is ::NvSciError_ReconciliationFailed
 */
#else
/**
 * \param[out] newConflictList Unused
 */
#endif
/**
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a inputCount is 0
 *      - @a inputArray is NULL
 *      - @a syncObj is NULL
 *      - not all the NvSciSyncAttrLists in inputArray are bound to the same
 *        NvSciSyncModule instance
 *      - any of the attributes in any of the input NvSciSyncAttrLists has an
 *        invalid value for that attribute
 *      - if any of the NvSciSyncAttrList in inputArray are not unreconciled
 *      - an NvSciSyncAttrList appears multiple times in inputArray
 */
#if (NV_IS_SAFETY == 0)
/**      - @a newConflictList is NULL
 */
#endif
/**
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if the newly created NvSciSyncObj cannot be
 *   associated with the NvSciSyncModule with which the NvSciSyncAttrList(s) in
 *   @a inputArray are associated.
 * - ::NvSciError_ReconciliationFailed if reconciliation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - ::NvSciError_UnsupportedConfig if there is an NvSciSyncAttrList mismatch between
 *   Signaler and Waiters.
 * - Panics if any of the input NvSciSyncAttrLists are not valid
 *
 * @pre @id{NvSciSyncAttrListReconcileAndObjAlloc_PreCond_001} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListReconcileAndObjAlloc(
    const NvSciSyncAttrList inputArray[],
    size_t inputCount,
    NvSciSyncObj* syncObj,
    NvSciSyncAttrList* newConflictList);

/**
 * \brief Exports an NvSciSyncAttrList and NvSciSyncObj into an
 * NvSciIpc-transferable object binary descriptor pointed to by @a data.
 *
 * The binary descriptor can subsequently be transferred to Waiters to create
 * a matching NvSciSyncObj.
 *
 * Note: This function serves as a convenience function that combines calls to
 * NvSciSyncAttrListIpcExportReconciled and NvSciSyncObjIpcExport.
 *
 * \param[in] syncObj NvSciSyncObj to export.
 * \param[in] permissions Flag indicating the expected NvSciSyncAccessPerm.
 * Valid value: permissions is valid if it is set to NvSciSyncAccessPerm_WaitOnly
 * or NvSciSyncAccessPerm_Auto.
 * \param[in] ipcEndpoint The NvSciIpcEndpoint through which the caller may send
 *            the exported NvSciSyncAttrList and NvSciSyncObj descriptor.
 * \param[out] attrListAndObjDesc Exported form of NvSciSyncAttrList and
 * NvSciSyncObj shareable across an NvSciIpc channel.
 * \param[out] attrListAndObjDescSize Size of the exported blob.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDescSize is NULL
 *      - @a syncObj is NULL
 *      - @a permissions flag contains signaling rights
 *      - @a ipcEndpoint is invalid.
 *      - @a ipcEndpoint does not lead to a peer in the topology tree
 *        of this NvSciSyncObj
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 *     or something went wrong with NvSciIpc
 * - ::NvSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panic if @a syncObj is not valid
 *
 * @pre @id{NvSciSyncIpcExportAttrListAndObj_PreCond_001} @asil{QM}
 *  Valid NvSciSyncObj obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncIpcExportAttrListAndObj(
    NvSciSyncObj syncObj,
    NvSciSyncAccessPerm permissions,
    NvSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize);

/**
 * \brief Frees an @ref NvSciSyncIpcExportAttrListAndObj descriptor
 * returned by a successful call to @ref NvSciSyncIpcExportAttrListAndObj.
 *
 * Does nothing for NULL.
 *
 * \param[in] attrListAndObjDescBuf Exported @ref NvSciSyncIpcExportAttrListAndObj
 * descriptor to be freed.
 * \return void
 * - Panics if Init Mode API is called in Runtime Mode.
 *
 * @pre @id{NvSciSyncAttrListAndObjFreeDesc_PreCond_001} @asil{QM}
 *  Valid NvSciSyncAttrListAndObj export descriptor is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *      - Provided there is no active operation involving the input
 *        @a attrListAndObjDescBuf to be freed
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void NvSciSyncAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf);

/**
 * \brief Creates an NvSciSyncObj based on the supplied binary descriptor
 * returned from a successful call to @ref NvSciSyncIpcExportAttrListAndObj
 * that has not yet been freed via @ref NvSciSyncAttrListAndObjFreeDesc. It
 * also validates reconciled NvSciSyncAttrList against input
 * unreconciled NvSciSyncAttrLists to ensure that the reconciled
 * NvSciSyncAttrList satisfies the constraints of all the given unreconciled
 * NvSciSyncAttrLists.
 *
 * This function is called from the Waiter after it receives the binary
 * descriptor from the Signaler who has created the binary descriptor.
 * Waiter will create its own NvSciSyncObj and return as output.
 *
 * Note: This function serves as a convenience function that combines calls to
 * NvSciSyncAttrListIpcImportReconciled and NvSciSyncObjIpcImport.
 *
 * \param[in] module A @ref NvSciSyncModule to associate the imported
 *            @ref NvSciSyncAttrList with.
 * \param[in] ipcEndpoint The @ref NvSciIpcEndpoint through which the caller
 *            receives the exported NvSciSyncAttrList and NvSciSyncObj descriptor.
 * \param[in] attrListAndObjDesc Exported form of NvSciSyncAttrList and
 *            NvSciSyncObj received through NvSciIpc channel.
 * Valid value: attrListAndObjDesc is valid if it is non-NULL.
 * \param[in] attrListAndObjDescSize Size of the exported blob.
 * Valid value: attrListAndObjDescSize is valid if it is bigger or equal
 *              sizeof(NvSciSyncObjIpcExportDescriptor).
 * \param[in] attrList The array of unreconciled NvSciSyncAttrLists
 *            against which the new NvSciSyncAttrList is to be validated.
 * Valid value: Array of valid NvSciSyncAttrList(s)
 * \param[in] attrListCount Number of unreconciled NvSciSyncAttrLists in the
 *            @a attrList array.
 * Valid value: [0, SIZE_MAX]
 * \param[in] minPermissions Flag indicating the expected NvSciSyncAccessPerm.
 * Valid value: NvSciSyncAccessPerm_WaitOnly and NvSciSyncAccessPerm_Auto
 * \param[in] timeoutUs Unused
 * \param[out] syncObj Waiter's NvSciSyncObj.
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a ipcEndpoint is invalid
 *      - @a attrListAndObjDesc is invalid
 *      - @a attrListAndObjDescSize is invalid
 *      - @a minPermissions is invalid
 *      - @a syncObj is NULL
 *      - input unreconciled NvSciSyncAttrLists' constraints
 *        are not satisfied by attributes of imported NvSciSyncObj.
 *      - @a attrList is NULL but @a attrListCount is not 0
 *      - if any of the NvSciSyncAttrList in attrList are not unreconciled
 *      - an NvSciSyncAttrList appears multiple times in @a attrList
 *      - @a minPermissions is NvSciSyncAccessPerm_Auto but permissions granted
 *        in the object part of @a attrListAndObjDesc are not enough to satisfy
 *        expected permissions stored in the attribute list part
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 *     or something went wrong with NvSciIpc
 * - ::NvSciError_TryItAgain if current operation needs to be retried by the
 *   user. This error is returned only when communication boundary is chip to
 *   chip (C2c).
 * - Panics if any of the following occurs:
 *      - @a module is not valid
 *      - any of the input unreconciled NvSciSyncAttrLists are not valid
 *
 * @pre @id{NvSciSyncIpcImportAttrListAndObj_PreCond_001} @asil{QM}
 *  Valid NvSciSyncModule is obtained.
 * @pre @id{NvSciSyncIpcImportAttrListAndObj_PreCond_002} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) are obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncIpcImportAttrListAndObj(
    NvSciSyncModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    NvSciSyncAttrList const attrList[],
    size_t attrListCount,
    NvSciSyncAccessPerm minPermissions,
    int64_t timeoutUs,
    NvSciSyncObj* syncObj);

/**
 * \brief Checks if the loaded library version is compatible with the version
 * the application was compiled against.
 *
 * This function checks the version of all dependent libraries and sets the
 * output variable to true if all libraries are compatible and all in parameters
 * valid, else sets output to false.
 *
 * \param[in] majorVer build major version.
 * Valid value: valid if set to NvSciSyncMajorVersion
 * \param[in] minorVer build minor version.
 * Valid value: valid if set to <= NvSciSyncMinorVersion
 * \param[out] isCompatible pointer to the bool holding the result.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a isCompatible is NULL
 *      - @a failed to check dependent library versions.
 *
 * @pre None
 * @post None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

/**
 * \brief Allows remote peer NvSciIpcEndpoint to gain access to already
 * allocated NvSciSyncObj. Provided that the allocated NvSciSyncObj
 * meets the requirements provided by the input unreconciled attribute
 * list of remote peer interested in gaining access to input NvSciSyncObj.
 *
 * \param[in] syncObj The NvSciSyncObj whose access needs to be granted
 * \param[in] inputArray list of unreconciled NvSciSyncAttrList imported
 * from remote peers who wants access to the input NvSciSyncObj
 * \param[in] inputCount Count of unreconciled NvSciSyncAttrList provided in
 * input unreconciledLists
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a syncObj is NULL
 *      - @a inputArray is NULL
 *      - any of NvSciSyncAttrLists in @a inputArray are NULL
 *      - any of NvSciSyncAttrLists in @a inputArray are not unreconciled
 *      - not all the NvSciSyncAttrLists in @a inputArray are bound to the same
 *        NvSciSyncModule instance as that of @a syncObj.
 *      - @a inputCount is 0
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - Panic if @a syncObj or NvSciSyncAttrLists inside @a inputArray are not valid
 *
 * @pre @id{NvSciSyncObjAttachPeer_PreCond_001} @asil{QM}
 *  Valid NvSciSyncObj is allocated.
 * @pre @id{NvSciSyncObjAttachPeer_PreCond_002} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList(s) inside @a inputArray.
 * @post @id{NvSciSyncObjAttachPeer_PostCond_001}
 *  NvSciSyncObj and reconciled list associated with @a syncObj is exportable
 * to remote peers NvSciIpcEndpoints whose unreconciled NvSciSyncAttrList was provided
 * as input to this function if the operation is successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncObjAttachPeer(
    NvSciSyncObj syncObj,
    const NvSciSyncAttrList inputArray[],
    size_t inputCount);

/**
 * \brief Fills appropriate attributes for C2C copy in the
 * input NvSciSyncAttrList
 *
 * \param[in] unrecAttrList attribute list to be filled
 * \param[in] permissions Permissions to be set in the attribute list
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *           - @a unrecAttrList is NULL,
 *           - @a unrecAttrList is not unreconciled and/or not writable,
 *           - any of NvSciSyncAttrKey_RequiredPerm,
 *             and equivalent of NvSciSyncAttrKey_PrimitiveInfo,
 *             is already set in @a unrecAttrList.
 * - Panics if @a unrecAttrList is not valid.
 *
 * @pre @id{NvSciSyncFillC2cAttrs_PreCond_001} @asil{QM}
 *  Valid unreconciled NvSciSyncAttrList is obtained.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncFillC2cAttrs(
    NvSciSyncAttrList unrecAttrList,
    NvSciSyncAccessPerm permissions);

/**
 * \brief Validates @a reconciledAttrList against the set of input attributes
 * that the user has set in the unreconciled NvSciSyncAttrList(s).
 * This API provides the safety mechanism to detect and report
 * any reconciliation errors
 */
#if (NV_IS_SAFETY == 0)
/**
 * \param[in] reconciledAttrList Unused
 * \param[in] pairArray Unused
 * \param[in] pairCount Unused
 * \param[in] permissions Unused
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success
 *
 * @note This is a no-op.
 */
#else
/**
 * \param[in] reconciledAttrList reconciled NvSciSyncAttrList to be validated
 * \param[in] pairArray Array of NvSciSyncAttrKeyValuePair structures that user
 * has used to set in the unreconciled NvSciSyncAttrList.
 * \param[in] pairCount Number of elements/entries in @a pairArray
 * \param[in] permissions Permissions to be used for @a reconciledAttrList validation
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful
 * - ::NvSciError_InvalidState if the Init Mode API check fails.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *         - @a reconciledAttrList is null
 *         - @a reconciledAttrList is not reconciled
 *         - any of the input attributes in @a pairArray is not a valid attribute
 *         - @a reconciledAttrList is not internally consistent
 *         - @a reconciledAttrList does not satisfy at least one of
 *           the attributes in @a pairArray
 *         - @a permissions is not one of the following: NvSciSyncAccessPerm_WaitOnly
 *           NvSciSyncAccessPerm_SignalOnly, NvSciSyncAccessPerm_WaitSignal
 * - ::NvSciError_AttrListValidationFailed indicates that reconciled
 *    NvSciSyncAttrList is not valid against the user provided key value pair(s).
 * - Panics if @a reconciledAttrList is not valid
 */
#endif
/**
 * @pre @id{NvSciSyncAttrListValidateReconciledAgainstAttrs_PreCond_001} @asil{B}
 *  Valid reconciled NvSciSyncAttrList is obtained.
 * @post @id{NvSciSyncAttrListValidateReconciledAgainstAttrs_PostCond_001}
 *  reconciledAttrList is marked as validated if operation is successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciSyncAttrListValidateReconciledAgainstAttrs(
    NvSciSyncAttrList reconciledAttrList,
    const NvSciSyncAttrKeyValuePair* pairArray,
    size_t pairCount,
    NvSciSyncAccessPerm permissions);

/**
 * \brief Performs a synchronous wait on the @ref NvSciSyncFence object until the
 * NvSciSyncFence has been signaled or the timeout expires. Any
* NvSciSyncCpuWaitContext may be used for waiting on any NvSciSyncFence provided
* they were created in the same NvSciSyncModule context.
 *
 * The user can choose details of the waiting procedure for better performance.
 * Syncpoints only support NvSciSyncWaitMode_Blocking which is also the default.
 * Semaphores support:
 * - NvSciSyncWaitMode_BusyWithYield - The default, suitable when expected waiting time
 * is long, the thread goes to sleep when waiting.
 * - NvSciSyncWaitMode_BusyNoYield - Suitable when expected waiting time is short. The thread
 * spins and does not go to sleep.
 *
 * \param[in] syncFence The NvSciSyncFence to wait on.
 * \param[in] context NvSciSyncCpuWaitContext holding resources needed
 * to perform waiting.
 * \param[in] timeoutUs Timeout to wait for in micro seconds, -1 for infinite
 * wait.
 * Valid value: [-1, NvSciSyncFenceMaxTimeout]
 * \param[in] waitMode Chooses the method of waiting.
 * Valid value: One of NvSciSyncWaitMode values
 *
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if any of the following occurs:
 *     - @a syncFence is cleared
 *     - @a syncFence is expired
 *     - @a syncFence has been signaled within the given timeout
 * - ::NvSciError_BadParameter if any of the following occurs:
 *     - @a syncFence is NULL
 *     - @a context is NULL
 *     - @a syncFence and @a context are associated with different
 *       NvSciSyncModule
 *     - @a timeoutUs is invalid
 *     - if caller doesn't have CPU wait permissions in NvSciSyncObj
 *       associated with @a syncFence
 *     - the module reference associated with @a context
 *       is NULL
 *     - @a waiMode is invalid
 * - ::NvSciError_ResourceError if wait operation did not complete
 *     successfully.
 * - ::NvSciError_Timeout if wait did not complete in the given timeout.
 * - ::NvSciError_Overflow if the NvSciSyncFence's id or value are not in range
 *     supported by the primitive this NvSciSyncFence corresponds to.
 * - ::NvSciError_UnsupportedConfig if @a waitMode is not supported by
 *     the primitive associated with @a syncFence's NvSciSyncObj
 * - Panics if any NvSciSyncObj associated with @a syncFence or @a context
 *     is not valid
 *
 * @pre @id{NvSciSyncFenceWaitWithMode_PreCond_001} @asil{B}
 *  Valid NvSciSyncFence* is obtained.
 * @pre @id{NvSciSyncFenceWaitWithMode_PreCond_002} @asil{B}
 *  Valid NvSciSyncCpuWaitContext is obtained.
 * @post @id{NvSciSyncFenceWaitWithMode_PostCond_001}
 *  Waiting on a cleared and expired NvSciSyncFence is always not blocking.
 * @arr @id{NvSciSyncFenceWaitWithMode_RES_001} @asil{B}
 *  One NvSciSyncCpuWaitContext can be used to wait on only one NvSciSyncFence
 * at a time but it can be used to wait on a different NvSciSyncFence at a different time.
 *
 * @usage
 * - Allowed context for the API call
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 * - Init: No
 * - Runtime: Yes
 * - De-Init: No
 */
NvSciError NvSciSyncFenceWaitWithMode(
    const NvSciSyncFence* syncFence,
    NvSciSyncCpuWaitContext context,
    int64_t timeoutUs,
    NvSciSyncWaitMode waitMode);

#if defined(__cplusplus)
}
#endif // __cplusplus
 /** @} */
#endif // INCLUDED_NVSCISYNC_H
