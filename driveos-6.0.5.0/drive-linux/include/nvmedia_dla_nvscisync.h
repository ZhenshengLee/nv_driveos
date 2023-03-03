/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * \file
 *
 * \brief <b> NVIDIA Media Interface: DLA NvSciSync </b>
 *
 * @b Description: This file contains the NvMediaDla and NvSciSync related APIs.
 *
 */

#ifndef NVMEDIA_DLA_NVSCISYNC_H
#define NVMEDIA_DLA_NVSCISYNC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvmedia_dla.h"

/**
 * @defgroup nvmedia_dla_nvscisync_api Deep Learning Accelerator Synchronization
 *
 * The NvMedia DLA NvSciSync API encompasses all NvMediaDla
 * NvSciSync handling functions.
 *
 * @ingroup nvmedia_dla_top
 * @{
 */

/** \brief Major version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_MAJOR   1
/** \brief Minor version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_MINOR   6
/** \brief Patch version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_PATCH   0

/**
 * NvMediaDlaInsertPreNvSciSyncFence API can be called at most
 * NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES times before each Dla submit call.
*/
#define NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES  (8U)

/**
 * \brief Returns the version information for the NvMedia DLA NvSciSync library.
 *
 * \param[in, out] version A pointer to an NvMediaVersion structure
 *                 filled by the DLA NvSciSync library.
 *                 @inputrange A non-null pointer to an %NvMediaVersion.
 *                 @outputrange A non-null pointer to an %NvMediaVersion if
 *                 successful, otherwise the value pointed to by @a version
 *                 remains unchanged.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a version is NULL.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaNvSciSyncGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Fills the NvMediaDla specific NvSciSync attributes.
 *
 * This API assumes that @a attrlist is a valid \ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - \ref NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * \param[in] dla           An NvMedia DLA device handle.
 *                          @inputrange A non-null pointer to an \ref NvMediaDla
 *                          created with NvMediaDlaCreate().
 * \param[in, out] attrlist A pointer to an \ref NvSciSyncAttrList structure
 *                          where NvMedia places NvSciSync attributes.
 *                          @inputrange A non-null pointer created by
 *                          NvSciSyncAttrListCreate().
 *                          @outputrange A non-null pointer to an NvSciSyncAttrList.
 * \param[in] clienttype    Indicates whether the @a attrlist is requested for
 *                          an %NvMediaDla signaler or an %NvMediaDla waiter.
 *                          @inputrange Any enum value defined by
 *                          \ref NvMediaNvSciSyncClientType.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the call is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL,
 *         or any of the above listed public attributes are already set,
 *         or if client type is invalid.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
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
NvMediaStatus
NvMediaDlaFillNvSciSyncAttrList(
    const NvMediaDla                *dla,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Fills the NvMediaDla specific NvSciSync deterministic attributes.
 *
 * This API assumes that @a attrlist is a valid \ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - \ref NvSciSyncAttrKey_RequiredPerm
 * - \ref NvSciSyncAttrKey_RequireDeterministicFences
 *
 * The application must not set this attribute.
 *
 * \param[in] dla           An NvMedia DLA device handle.
 *                          @inputrange A non-null pointer to an \ref NvMediaDla
 *                          created with NvMediaDlaCreate().
 * \param[in, out] attrlist A pointer to an \ref NvSciSyncAttrList structure
 *                          where NvMedia places NvSciSync attributes.
 *                          @inputrange A non-null pointer created by
 *                          NvSciSyncAttrListCreate().
 *                          @outputrange A non-null pointer to an NvSciSyncAttrList.
 * \param[in] clienttype    Indicates whether the @a attrlist is requested for
 *                          an %NvMediaDla signaler or an %NvMediaDla waiter.
 *                          @inputrange Any enum value defined by
 *                          \ref NvMediaNvSciSyncClientType.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the call is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL,
 *         or any of the above listed public attributes are already set,
 *         or if client type is invalid.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
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
NvMediaStatus
NvMediaDlaFillNvSciSyncDeterministicAttrList(
    const NvMediaDla* dla,
    NvSciSyncAttrList attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Registers an \ref NvSciSyncObj with \ref NvMediaDla.
 *
 * Every NvSciSyncObj(even duplicate objects) used by %NvMediaDla
 * must be registered by a call to this function before it is used.
 * Only the exact same registered \ref NvSciSyncObj can be passed to the run time APIs.
 *
 * For a given NvMediaDla handle, one NvSciSyncObj can be registered
 * as one \ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObj objects can
 * be registered.
 *
 * \param[in] dla         An NvMedia DLA device handle.
 *                        @inputrange A non-null pointer to an %NvMediaDla
 *                        created with NvMediaDlaCreate().
 * \param[in] syncobjtype Determines how @a nvscisync is used by @a dla.
 *                        @inputrange Any enum value defined by
 *                        \ref NvMediaNvSciSyncObjType.
 * \param[in] nvscisync   The %NvSciSyncObj to be registered
 *                        with @a dla.
 *                        @inputrange A non-null pointer created by
 *                        NvSciSyncObjAlloc().
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a dla is NULL or
 *        @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *        compatible %NvSciSyncObj which %NvMediaDla can support.
 * - \ref NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObj objects
 *        are already registered for the given @a syncobjtype, OR
 *        if @a nvscisync is already registered with the same @a dla
 *        handle for a different @a syncobjtype.
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
NvMediaStatus
NvMediaDlaRegisterNvSciSyncObj(
    NvMediaDla                *dla,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * \brief Unregisters an \ref NvSciSyncObj with NvMediaDla.
 *
 * During teardown, every %NvSciSyncObj registered with NvMediaDla must be
 * unregistered before calling NvMediaDlaDestroy().
 *
 * Before the application calls this function, it must ensure that the
 * application is in teardown mode, and any NvMediaDla operation using
 * this @a nvscisync has completed.
 * If the function is called while @a nvscisync is still in use by any
 * NvMediaDla operations, the behavior is undefined.
 *
 * \param[in] dla         An NvMedia DLA device handle.
 *                        @inputrange A non-null pointer to an \ref NvMediaDla
 *                        created with NvMediaDlaCreate().
 * \param[in] nvscisync   An \ref NvSciSyncObj to be unregistered with @a dla.
 *                        @inputrange A non-null pointer created by
 *                        NvSciSyncObjAlloc().
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a dla is NULL, or
 *        @a nvscisync is not registered with @a dla.
 * - \ref NVMEDIA_STATUS_ERROR if @a dla is destroyed before this function is
 *        called.
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
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaUnregisterNvSciSyncObj(
    NvMediaDla                *dla,
    NvSciSyncObj               nvscisync
);

/**
 * \brief Sets the \ref NvSciSyncObj to be used for a Start of Frame (SOF)
 * \ref NvSciSyncFence.
 *
 * \note This API is not applicable for NvSciSyncObj backed by deterministic
 *      primitive.
 *
 * To use NvMediaDlaGetSOFNvSciSyncFence(), the application must call
 * this function before the first DLA submit API.
 * NvMedia DlA currently accepts only one SOF \ref NvSciSyncObj.
 *
 * \param[in] dla          An NvMedia DLA device handle.
 *                         @inputrange A non-null pointer to an \ref NvMediaDla
 *                         created with NvMediaDlaCreate().
 * \param[in] nvscisyncSOF A registered NvSciSyncObj to be
 *                         associated with SOF %NvSciSyncFence.
 *                         @inputrange A non-null pointer created by
 *                         NvSciSyncObjAlloc().
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a dla is NULL, or if @a nvscisyncSOF
 *        is not registered with @a dla as either type
 *        \ref NVMEDIA_SOFSYNCOBJ or type \ref NVMEDIA_SOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisyncSOF is backed by
 *          deterministic primitive.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
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
NvMediaStatus
NvMediaDlaSetNvSciSyncObjforSOF(
    NvMediaDla                *dla,
    NvSciSyncObj               nvscisyncSOF
);

/**
 * \brief Sets an \ref NvSciSyncObj to be used for a End of Frame (EOF)
 * \ref NvSciSyncFence.
 *
 * \note This API is not applicable for NvSciSyncObj backed by deterministic
 *      primitive.
 *
 * To use NvMediaDlaGetEOFNvSciSyncFence(), the application must call
 * this function before the calling the first DLA submit API.
 * NvMedia DLA currently accepts only one EOF %NvSciSyncObj.
 *
 * \param[in] dla          An NvMedia DLA device handle.
 *                         @inputrange A non-null pointer to an \ref NvMediaDla
 *                         created with NvMediaDlaCreate().
 * \param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                         associated with EOF %NvSciSyncFence.
 *                         @inputrange A non-null pointer created by
 *                         NvSciSyncObjAlloc().
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if @a dla is NULL, OR if @a nvscisyncEOF
 *        is not registered with @a dla as either type \ref NVMEDIA_EOFSYNCOBJ
 *        or type \ref NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisyncEOF is backed by
 *          deterministic primitive.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
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
NvMediaStatus
NvMediaDlaSetNvSciSyncObjforEOF(
    NvMediaDla                *dla,
    NvSciSyncObj               nvscisyncEOF
);

/**
 * \brief Sets an \ref NvSciSyncFence as a prefence for a DLA submit operation.
 *
 * If you use %NvMediaDlaInsertPreNvSciSyncFence(), the application must call it
 * before calling a DLA submit API. The following DLA submit operation
 * is started only after the expiry of the @a prenvscisyncfence.
 *
 * You can set a maximum of \ref NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaDlaInsertPreNvSciSyncFence() before calling a DLA
 * submit function.
 *
 * After a call to the DLA submit function, all NvSciSyncFences previously inserted
 * by %NvMediaDlaInsertPreNvSciSyncFence() are cleared, and they are not
 * reused for subsequent DLA submit calls.
 *
 * \param[in] dla               An NvMedia DLA device handle.
 *                              @inputrange A non-null pointer to an
 *                              \ref NvMediaDla created with NvMediaDlaCreate().
 * \param[in] prenvscisyncfence A pointer to NvSciSyncFence.
 *                              @inputrange A non-null pointer to
 *                              %NvSciSyncFence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the following occurs:
 *        - @a dla is not a valid NvMediaDla handle, or
 *        - @a prenvscisyncfence is NULL, or
 *        - @a prenvscisyncfence is not generated with an \ref NvSciSyncObj
 *          that was registered with @a dla as either type
 *          \ref NVMEDIA_PRESYNCOBJ or type \ref NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaDlaInsertPreNvSciSyncFence is
 *        already called at least %NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES times with
 *        the same @a dla NvMediaDla handle before a DLA submit call.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
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
NvMediaStatus
NvMediaDlaInsertPreNvSciSyncFence(
    NvMediaDla                *dla,
    const NvSciSyncFence      *prenvscisyncfence
);

/**
 * \brief Gets an SOF \ref NvSciSyncFence for a DLA submit operation.
 *
 * \note This API is not applicable for NvSciSyncObj backed by deterministic
 *      primitive.
 *
 * An SOF %NvSciSyncFence is associated with a DLA submit operation, and its
 * expiry indicates that the corresponding DLA submit operation has started.
 * NvMediaDlaGetSOFNvSciSyncFence() returns the SOF %NvSciSyncFence associated
 * with the last DLA submit call.
 *
 * If you use %NvMediaDlaGetSOFNvSciSyncFence(),
 * you must call it after calling the DLA submit function.
 *
 * \param[in] dla                An NvMedia DLA device handle.
 *                               @inputrange A non-null pointer to an
 *                               \ref NvMediaDla created with NvMediaDlaCreate().
 * \param[in] sofnvscisyncobj    The SOF \ref NvSciSyncObj associated with
 *                               the \ref NvSciSyncFence being requested.
 *                               This structure will be modified by this function.
 *                               @inputrange A non-null pointer created by
 *                               NvSciSyncObjAlloc().
 * \param[in, out] sofnvscisyncfence A pointer to the SOF %NvSciSyncFence.
 *                               @inputrange A non-null pointer to
 *                               %NvSciSyncFence.
 *                               @outputrange A non-null pointer to
 *                               SOF %NvSciSyncFence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the following occurs:
 *        - @a dla is not a valid NvMediaDla handle, or
 *        - @a sofnvscisyncfence is NULL, or
 *        - @a sofnvscisyncobj is not registered with @a dla as type
 *          \ref NVMEDIA_SOFSYNCOBJ or type \ref NVMEDIA_SOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if @a sofnvscisyncobj is backed by
 *          deterministic primitive.
 * - \ref NVMEDIA_STATUS_ERROR, if the function is called before setting
 *         the loadable as current or if there is a failure while updating
 *         the sofnvscisyncfence with the fence from the sofnvscisyncobj
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
NvMediaStatus
NvMediaDlaGetSOFNvSciSyncFence(
    const NvMediaDla                *dla,
    NvSciSyncObj               sofnvscisyncobj,
    NvSciSyncFence            *sofnvscisyncfence
);

/**
 * \brief Gets an EOF \ref NvSciSyncFence for a DLA submit operation.
 *
 * \note This API is not applicable for NvSciSyncObj backed by deterministic
 *      primitive.
 *
 * An EOF %NvSciSyncFence is associated with a DLA submit operation and its expiry
 * indicates that the corresponding DLA submit operation has finished.
 * \ref NvMediaDlaGetEOFNvSciSyncFence returns the EOF %NvSciSyncFence associated
 * with the last DLA submit call.
 *
 * If you use %NvMediaDlaGetEOFNvSciSyncFence(),
 * you must call it after calling a DLA submit function.
 *
 * \param[in] dla                An NvMedia DLA device handle.
 *                               @inputrange A non-null pointer to an
 *                               \ref NvMediaDla created with NvMediaDlaCreate().
 * \param[in] eofnvscisyncobj    An EOF \ref NvSciSyncObj associated with
 *                               the \ref NvSciSyncFence being requested.
 *                               This structure will be modified by this function.
 *                               @inputrange A non-null pointer created by
 *                               NvSciSyncObjAlloc().
 * \param[in, out] eofnvscisyncfence A pointer to the EOF %NvSciSyncFence.
 *                               @inputrange A non-null pointer to
 *                               %NvSciSyncFence.
 *                               @outputrange A non-null pointer to
 *                               %NvSciSyncFence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the following occurs:
 *        - @a dla is not a valid NvMediaDla handle, or
 *        - @a eofnvscisyncfence is NULL, or
 *        - @a eofnvscisyncobj is not registered with @a dla as type
 *          \ref NVMEDIA_EOFSYNCOBJ or type \ref NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if @a eofnvscisyncobj is backed by
 *          deterministic primitive.
 *        calling the DLA submit API.
 * - \ref NVMEDIA_STATUS_ERROR, if the function is called before setting
 *         the loadable as current or if there is a failure while updating
 *         the eofnvscisyncfence with the fence from the eofnvscisyncobj
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
NvMediaStatus
NvMediaDlaGetEOFNvSciSyncFence(
    const NvMediaDla                *dla,
    NvSciSyncObj               eofnvscisyncobj,
    NvSciSyncFence            *eofnvscisyncfence
);

/**
 * \brief Sets \ref NvSciSyncObj as a SOF for a DLA submit operation.
 *
 * If you use %NvMediaDlaInsertSOFNvSciSyncObj(), the application must call it
 * before calling a DLA submit API & NvSciSyncObj gets signaled by DLA prior to
 * start of execution.
 *
 * You can set a maximum of \ref <TODO> prefences by calling
 * %NvMediaDlaInsertSOFNvSciSyncObj() before calling a DLA submit function.
 *
 * After a call to the DLA submit function, all NvSciSyncFences previously
 * inserted by %%NvMediaDlaInsertSOFNvSciSyncObj() are cleared, and they are not
 * reused for subsequent DLA submit calls.
 *
 * \param[in] dla               An NvMedia DLA device handle.
 *                              @inputrange A non-null pointer to an
 *                              \ref NvMediaDla created with NvMediaDlaCreate().
 * \param[in] syncObj           NvSciSyncObj that needs to be used as SOF for
 *                              current submission. @inputrange A non-null
 *                              NvSciSyncObj that is already registered.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the following occurs:
 *        - @a dla is not a valid NvMediaDla handle.
 * - \ref NVMEDIA_STATUS_ERROR if any of the following occurs:
 *        - current loadable is not set.
 *        - if syncObj is not registered with NvMediaDla for SOF operation.
 *        - if function fails to set syncObj as active SOF event for the current
 *          submission.
 *  @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 **/
NvMediaStatus
NvMediaDlaInsertSOFNvSciSyncObj(
    NvMediaDla* dla,
    NvSciSyncObj syncObj
);

/**
 * \brief Sets \ref NvSciSyncObj as a EOF for a DLA submit operation.
 *
 * If you use %NvMediaDlaInsertEOFNvSciSyncObj(), the application must call it
 * before calling a DLA submit API & NvSciSyncObj gets signaled by DLA
 * immediately after task execution completion.
 *
 * You can set a maximum of \ref <TODO> prefences by calling
 * %NvMediaDlaInsertEOFNvSciSyncObj() before calling a DLA submit function.
 *
 * After a call to the DLA submit function, all NvSciSyncFences previously
 * inserted by %%NvMediaDlaInsertEOFNvSciSyncObj() are cleared, and they are not
 * reused for subsequent DLA submit calls.
 *
 * \param[in] dla               An NvMedia DLA device handle.
 *                              @inputrange A non-null pointer to an
 *                              \ref NvMediaDla created with NvMediaDlaCreate().
 * \param[in] syncObj           NvSciSyncObj that needs to be used as EOF for
 *                              current submission. @inputrange A non-null
 *                              NvSciSyncObj that is already registered.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the following occurs:
 *        - @a dla is not a valid NvMediaDla handle.
 * - \ref NVMEDIA_STATUS_ERROR if any of the following occurs:
 *        - current loadable is not set.
 *        - if syncObj is not registered with NvMediaDla for EOF operation.
 *        - if function fails to set syncObj as active EOF event for the current
 *          submission.
 *  @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 **/
NvMediaStatus
NvMediaDlaInsertEOFNvSciSyncObj(
    NvMediaDla* dla,
    NvSciSyncObj syncObj
);

/*
 * \defgroup history_nvmedia_dla_nvscisync History
 * Provides change history for the NvMedia Dla NvSciSync API
 *
 * \section history_nvmedia_dla_nvscisync Version History
 *
 * <b> Version 1.0 </b> March 14, 2019
 * - Initial release
 *
 * <b> Version 1.1 </b> April 11, 2019
 * - Add new API NvMediaDlaSetNvSciSyncObjforSOF and NvMediaDlaGetEOFNvSciSyncFence
 * - Rename NvMediaDlaUnRegisterNvSciSyncObj to NvMediaDlaUnregisterNvSciSyncObj
 *
 * <b> Version 1.2 </b> Jan 22, 2020
 * - Disable NvMediaDlaSetNvSciSyncObjforSOF and NvMediaDlaGetSOFNvSciSyncFence in
 *   safety build as they are currently unsupported.
 *
 * <b> Version 1.3 </b> Jul 20, 2020
 * - Added support for NvSciSyncObj backed by deterministic primitive.
 * - Currently timestamp feature is disabled with NvSciSyncObj backed by
 *  deterministic primitive.
 * - Added new APIs: NvMediaDlaInsertEOFNvSciSyncObj,
 *      NvMediaDlaInsertSOFNvSciSyncObj (disabled in safety),
 *      NvMediaDlaFillNvSciSyncDeterministicAttrList
 *
 * <b> Version 1.4 </b> July 26, 2021
 * - Update comments for NvMediaDlaGetEOFNvSciSyncFence and NvMediaDlaGetSOFNvSciSyncFence
 *
 * <b> Version 1.5 </b> August 20, 2021
 * - Update doxygen comments for All APIs to have Thread safety information and API Group information
 *
 * <b> Version 1.6 </b> October 25, 2021
 * - Enable SOF feature in safety builds.
 * - Enable timestamp support for all primitives.
 *
 * <b> Version 1.6.0 </b> May 10, 2022
 * - Added patch version number macro: NVMEDIA_DLA_NVSCISYNC_VERSION_PATCH.
 *
 */

/** @} <!-- Ends nvmedia_dla_nvscisync_api DLA Synchronization --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_DLA_NVSCISYNC_H */
