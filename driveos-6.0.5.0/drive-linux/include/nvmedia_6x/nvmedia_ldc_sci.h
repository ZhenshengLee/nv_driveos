/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/// @file
/// @brief <b> NVIDIA Media Interface: LDC NvSci </b>
///
/// @b Description: This file contains the NvMedia LDC and NvSci related APIs.
///

#ifndef NVMEDIA_LDC_SCI_H
#define NVMEDIA_LDC_SCI_H

#include "nvmedia_core.h"
#include "nvmedia_ldc.h"
#include "nvscibuf.h"
#include "nvscisync.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_nvmedia_ldc_sci_api LDC Surface and Synchronization
///
/// The NvMedia LDC NvSci API encompasses all NvMedia LDC handling for NvSciBuf
/// NvSciSync related functions.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Fills the NvMedia LDC specific NvSciBuf attributes.
///
/// @param[in]  handle      Not used. Can be NULL.
/// @param[out] attrList    Pointer to an #NvSciBufAttrList struct that will
///                         be populated with the NvSciBuf attributes.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Attributes filled successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  The attributes have already been
///                                       filled, or attrList is invalid.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to populate the attribute list.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvMediaStatus
NvMediaLdcFillNvSciBufAttrList(NvMediaLdc *const handle, NvSciBufAttrList const attrList);

/// @brief Fills the NvMedia LDC specific NvSciSync attributes.
///
/// This function sets the public attributes:
/// - #NvSciSyncAttrKey_RequiredPerm
///
/// The application must not set this attribute.
///
/// @param[in]  handle      Pointer to the NvMediaLdc context.
/// @param[out] attrList    Pointer to an #NvSciSyncAttrList struct that will
///                         be populated with the NvSciSync attributes.
/// @param[in]  clientType  An #NvMediaNvSciSyncClientType, to indicate whether
///                         the attributes filled should be for a waiter or a
///                         signaler. The value should be either:
///                         - NVMEDIA_WAITER
///                         - NVMEDIA_SIGNALER
///                         - NVMEDIA_SIGNALER_WAITER.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Attributes filled successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  The attributes have already been
///                                       filled, or one of the parameters has
///                                       an invalid value. This could be:
///                                       - @a handle is NULL
///                                       - @a attrList is invalid
///                                       - @a clientType is invalid.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to populate the attribute list.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvMediaStatus
NvMediaLdcFillNvSciSyncAttrList(NvMediaLdc *const handle,
                                NvSciSyncAttrList const attrList,
                                NvMediaNvSciSyncClientType const clientType);

/// @brief Register an #NvSciBufObj with NvMedia LDC.
///
/// Every NvSciBufObj (even duplicate objects) used by NvMedia LDC
/// must be registered by a call to this function before it is used.
/// Only the exact same registered NvSciBufObj can be passed to
/// NvMediaLdcSetSrcSurface(), NvMediaLdcSetDstSurface(),
/// NvMediaLdcSetPreviousSurface(), NvMediaLdcSetXSobelDstSurface(),
/// and NvMediaLdcSetDownsampledXSobelDstSurface() functions.
///
/// @param[in] handle  Pointer to the NvMediaLdc context.
/// @param[in] bufObj  The #NvSciBufObj to be registered.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Buffer registered
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           Returned when:
///                                                - @a handle is NULL
///                                                - @a bufObj is invalid
///                                                - @a bufObj has already been
///                                                  registered
///                                                - duplicate of @a bufObj was
///                                                  previously registered with
///                                                  more strict read-only
///                                                  permissions.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of registered
///                                                buffers has been reached.
/// @retval NVMEDIA_STATUS_INVALID_STATE           The function was called in
///                                                incorrect system state.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to register the
///                                                buffer.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMediaLdcUnregisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcRegisterNvSciBufObj(NvMediaLdc *const handle, NvSciBufObj const bufObj);

/// @brief Register an #NvSciSyncObj with NvMedia LDC.
///
/// Every NvSciSyncObj (even duplicate objects) used by NvMedia LDC
/// must be registered by a call to this function before it is used.
///
/// Only the exact same registered NvSciSyncObj can be passed to
/// NvMediaLdcSetNvSciSyncObjforEOF() or NvMediaLdcUnregisterNvSciSyncObj().
/// Also the NvSciSyncObjs associated with any NvSciSyncFences passed to
/// NvMediaLdcInsertPreNvSciSyncFence() must be registered.
///
/// For a given NvMediaLdc handle, one NvSciSyncObj can be registered as one
/// #NvMediaNvSciSyncObjType only.
///
/// @param[in] handle       Pointer to the NvMediaLdc context.
/// @param[in] syncObjType  An #NvMediaNvSciSyncClientType, to indicate what
///                         event the sync object will represent.
/// @param[in] syncObj      The #NvSciSyncObj to be registered.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Sync object registered
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           One of the parameters has an
///                                                invalid value. This could be:
///                                                - @a handle is NULL
///                                                - @a syncObjType is invalid
///                                                - @a syncObj is invalid,
///                                                or the sync object has been
///                                                already registered.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of registered
///                                                sync objects has been
///                                                reached.
/// @retval NVMEDIA_STATUS_INVALID_STATE           The function was called in
///                                                incorrect system state.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to register the
///                                                sync object.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMediaLdcUnregisterNvSciSyncObj()
/// @sa NvMediaLdcAttributes.maxRegisteredSyncs
///
NvMediaStatus
NvMediaLdcRegisterNvSciSyncObj(NvMediaLdc *const handle,
                               NvMediaNvSciSyncObjType const syncObjType,
                               NvSciSyncObj const syncObj);

/// @brief Unregisters an #NvSciBufObj with #NvMediaLdc.
///
/// Every #NvSciBufObj registered with #NvMediaLdc by
/// NvMediaLdcRegisterNvSciBufObj() must be unregistered before you call
/// NvMediaLdcDestroy().
///
/// @param[in] handle  Pointer to the NvMediaLdc context.
/// @param[in] bufObj  The #NvSciBufObj to be unregistered.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Buffer unregistered successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a bufObj is invalid,
///                                       or the buffer was not registered.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        The buffer is still being used by a
///                                       pending operation.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to unregister the buffer.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcUnregisterNvSciBufObj(NvMediaLdc *const handle, NvSciBufObj const bufObj);

/// @brief Unregisters an #NvSciSyncObj with #NvMediaLdc.
///
/// Every #NvSciSyncObj registered with #NvMediaLdc by
/// NvMediaLdcRegisterNvSciSyncObj() must be unregistered before you call
/// NvMediaLdcDestroy().
///
/// Before the application calls this function, it must ensure that any
/// #NvMediaLdcProcess() operation that uses the #NvSciSyncObj has completed.
/// If this function is called while #NvSciSyncObj is still in use by any
/// NvMediaLdcProcess() operation, behavior is undefined.
///
/// @param[in] handle   Pointer to the NvMediaLdc context.
/// @param[in] syncObj  The #NvSciSyncObj to be unregistered.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Sync object unregistered successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a syncObj is invalid,
///                                       or the sync object was not registered.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        The sync object is still being used by
///                                       a pending operation.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to unregister the sync object.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @sa NvMediaLdcRegisterNvSciSyncObj()
///
NvMediaStatus
NvMediaLdcUnregisterNvSciSyncObj(NvMediaLdc *const handle, NvSciSyncObj const syncObj);

/// @brief Specifies the #NvSciSyncObj to be used for EOF event.
///
/// @param[in] handle   Pointer to the NvMediaLdc context.
/// @param[in] params   An NvMediaLdcParameters handle.
/// @param[in] syncObj  The #NvSciSyncObj to be used for the EOF fence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             EOF fence set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  The sync object is not registered as
///                                       an EOF event type with NvMedia LDC, or
///                                       one of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a syncObj is invalid.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcRegisterNvSciSyncObj()
///
NvMediaStatus
NvMediaLdcSetNvSciSyncObjforEOF(NvMediaLdc *const handle,
                                NvMediaLdcParameters const params,
                                NvSciSyncObj const syncObj);

/// @brief Insert an #NvSciSyncFence as a pre-fence.
///
/// This function inserts the specified #NvSciSyncFence as a pre-fence to the
/// LDC operation. The #NvMediaLdcProcess() operation is started only after
/// the expiry of the @a syncFence. The pre-fences need to be set separately for
/// each #NvMediaLdcProcess() call, even if the same NvMediaLdcParameters handle
/// is used.
///
/// For example, in this sequence of code:
/// @code
/// NvMediaLdcInsertPreNvSciSyncFence(handle, params, syncFence);
/// NvMediaLdcProcess(handle, params, NULL);
/// @endcode
/// the #NvMediaLdcProcess() operation is assured to start only after the expiry
/// of @a syncFence.
///
/// You can set a maximum of 16 prefences by calling
/// #NvMediaLdcInsertPreNvSciSyncFence().
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] params     An NvMediaLdcParameters handle.
/// @param[in] syncFence  Pointer to an #NvSciSyncFence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Pre-fence inserted
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           The sync object is not
///                                                registered as an PRESYNC type
///                                                with NvMedia LDC, or one of
///                                                the parameters has an invalid
///                                                value. This could be:
///                                                - @a handle is NULL
///                                                - @a params is invalid
///                                                - @a syncFence is NULL.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of registered
///                                                pre-fences has been reached.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to insert the
///                                                pre-fence.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcRegisterNvSciSyncObj()
///
NvMediaStatus
NvMediaLdcInsertPreNvSciSyncFence(NvMediaLdc *const handle,
                                  NvMediaLdcParameters const params,
                                  NvSciSyncFence const *const syncFence);

/// @brief Gets an EOF #NvSciSyncFence for an NvMediaLdcProcess() operation.
///
/// The expiry of an EOF #NvSciSyncFence associated with an #NvMediaLdcProcess()
/// operation indicates that the corresponding #NvMediaLdcProcess() operation
/// has finished.
///
/// To be able to get the EOF fence using this function, the #NvSciSyncObj to be
/// used for EOF event needs to have been set for the #NvMediaLdcParameters
/// handle used with the #NvMediaLdcProcess() operation with
/// #NvMediaLdcSetNvSciSyncObjforEOF().
///
/// For example, in this sequence of code:
/// @code
/// NvMediaLdcSetNvSciSyncObjforEOF(handle, params, eofSyncObj);
/// NvMediaLdcProcess(handle, params, &result);
/// NvMediaLdcGetEOFNvSciSyncFence(handle, &result, &syncFence);
/// @endcode
/// expiry of @a syncFence indicates that the preceding #NvMediaLdcProcess()
/// operation has finished.
///
/// @param[in]  handle     Pointer to the NvMediaLdc context.
/// @param[in]  result     Pointer to the NvMediaLdcResult struct.
/// @param[out] syncFence  Pointer to an #NvSciSyncFence that will be populated
///                        with the EOF fence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             EOF fence returned successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  No #NvSciSyncObj was set for the
///                                       operation EOF event, or one of the
///                                       parameters has an invalid value. This
///                                       could be:
///                                       - @a handle is NULL
///                                       - @a result is NULL or invalid
///                                       - @a syncFence is NULL.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to get the fence object.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcRegisterNvSciSyncObj()
/// @sa NvMediaLdcSetNvSciSyncObjforEOF()
///
NvMediaStatus
NvMediaLdcGetEOFNvSciSyncFence(NvMediaLdc *const handle,
                               NvMediaLdcResult const *const result,
                               NvSciSyncFence *const syncFence);

/// @brief Sets the source surface.
///
/// The surface needs to be set separately for each #NvMediaLdcProcess() call,
/// even if the same NvMediaLdcParameters handle is used.
///
/// @param[in] handle      Pointer to the NvMediaLdc context.
/// @param[in] params      An NvMediaLdcParameters handle.
/// @param[in] surface     The #NvSciBufObj to be used for the source surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a surface is invalid,
///                                       or the surface was not registered.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcSetSrcSurface(NvMediaLdc *const handle,
                        NvMediaLdcParameters const params,
                        NvSciBufObj const surface);

/// @brief Sets the destination surface.
///
/// The surface needs to be set separately for each #NvMediaLdcProcess() call,
/// even if the same NvMediaLdcParameters handle is used.
///
/// @param[in] handle      Pointer to the NvMediaLdc context.
/// @param[in] params      An NvMediaLdcParameters handle.
/// @param[in] surface     The #NvSciBufObj to be used for the destination
///                        surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a surface is invalid,
///                                       or the surface was not registered.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcSetDstSurface(NvMediaLdc *const handle,
                        NvMediaLdcParameters const params,
                        NvSciBufObj const surface);

/// @brief Sets the previous surface for TNR operation.
///
/// The previous surface is the surface used as the destination surface with the
/// previous #NvMediaLdcProcess() call.
///
/// The surface needs to be set separately for each #NvMediaLdcProcess() call,
/// even if the same NvMediaLdcParameters handle is used.
///
/// @param[in] handle      Pointer to the NvMediaLdc context.
/// @param[in] params      An NvMediaLdcParameters handle.
/// @param[in] surface     The #NvSciBufObj to be used as a previous
///                        surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a surface is invalid,
///                                       or the surface was not registered.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcSetPreviousSurface(NvMediaLdc *const handle,
                             NvMediaLdcParameters const params,
                             NvSciBufObj const surface);

/// @brief Sets the surface for xSobel output.
///
/// The xSobel output can only be used if TNR processing has not been enabled
/// with #NvMediaLdcSetTnrParameters().
///
/// The surface needs to be set separately for each #NvMediaLdcProcess() call,
/// even if the same NvMediaLdcParameters handle is used.
///
/// @param[in] handle      Pointer to the NvMediaLdc context.
/// @param[in] params      An NvMediaLdcParameters handle.
/// @param[in] surface     The #NvSciBufObj to be used for the destination
///                        xSobel surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a surface is invalid,
///                                       or the surface was not registered.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcSetXSobelDstSurface(NvMediaLdc *const handle,
                              NvMediaLdcParameters const params,
                              NvSciBufObj const surface);

/// @brief Sets the surface for 4x4 downsampled xSobel output.
///
/// The xSobel output can only be used if TNR processing has not been enabled
/// with #NvMediaLdcSetTnrParameters().
///
/// The surface needs to be set separately for each #NvMediaLdcProcess() call,
/// even if the same NvMediaLdcParameters handle is used.
///
/// @param[in] handle      Pointer to the NvMediaLdc context.
/// @param[in] params      An NvMediaLdcParameters handle.
/// @param[in] surface     The #NvSciBufObj to be used for the destination
///                        downsampled xSobel surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a surface is invalid,
///                                       or the surface was not registered.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
/// @sa NvMediaLdcRegisterNvSciBufObj()
///
NvMediaStatus
NvMediaLdcSetDownsampledXSobelDstSurface(NvMediaLdc *const handle,
                                         NvMediaLdcParameters const params,
                                         NvSciBufObj const surface);

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif
