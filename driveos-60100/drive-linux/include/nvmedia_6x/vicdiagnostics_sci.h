/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/// @file
/// @brief <b> NVIDIA Media Interface: Vic Diagnostics NvSci </b>
///
/// @b Description: This file contains the Vic Diagnostics and NvSci
///                 related APIs.
///

#ifndef VICDIAGNOSTICS_SCI_H
#define VICDIAGNOSTICS_SCI_H

#include "nvmedia_core.h"
#include "vicdiagnostics.h"
#include "nvscisync.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_vicdiagnostics_sci_api Vic Diagnostics Synchronization
///
/// The Vic Diagnostics NvSci API encompasses all Vic Diagnostics handling for
/// NvSciSync related functions.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Fills the Vic Diagnostics specific NvSciSync attributes.
///
/// This function updates the input NvSciSyncAttrList with values equivalent to
/// the following public attribute key-values:
///
/// NvSciSyncAttrKey_RequiredPerm set to
/// - NvSciSyncAccessPerm_WaitOnly for @ref clientType NVMEDIA_WAITER
/// - NvSciSyncAccessPerm_SignalOnly for @ref clientType NVMEDIA_SIGNALER
/// - NvSciSyncAccessPerm_WaitSignal for @ref clientType NVMEDIA_SIGNALER_WAITER
///
/// NvSciSyncAttrKey_PrimitiveInfo set to
/// - NvSciSyncAttrValPrimitiveType_Syncpoint
///
/// The application must not set these attributes for the same NvSciSyncAttrList
/// that is passed to this function.
///
/// When @a clientType is NVMEDIA_SIGNALER or NVMEDIA_SIGNALER_WAITER, some of
/// the attribute values returned in @a attrList are tied to the specific
/// VicDiagnostics context instance passed in @a handle. NvSciSyncObjs created
/// using such attribute values must not be used after the VicDiagnostics
/// context instance has been destroyed.
///
/// @param[in]  handle      Pointer to the VicDiagnostics context.
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
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a attrList must be valid NvSciSyncAttrList handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvMediaStatus
VicDiagnosticsFillNvSciSyncAttrList(VicDiagnostics const * const handle,
                                    NvSciSyncAttrList const attrList,
                                    NvMediaNvSciSyncClientType const clientType);

/// @brief Register an #NvSciSyncObj with Vic Diagnostics.
///
/// Every NvSciSyncObj used by Vic Diagnostics must be registered by a call to
/// this function before it is used. Also the NvSciSyncObjs associated with any
/// NvSciSyncFences passed to VicDiagnosticsInsertPreNvSciSyncFence() must be
/// registered.
///
/// @param[in] handle       Pointer to the VicDiagnostics context.
/// @param[in] syncObjType  An #NvMediaNvSciSyncClientType, to indicate what
///                         event the sync object will represent. Must be one of
///                         - NVMEDIA_PRESYNCOBJ
///                         - NVMEDIA_EOFSYNCOBJ
///                         - NVMEDIA_EOF_PRESYNCOBJ
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
///                                                or the sync object or its
///                                                duplicate has been already
///                                                registered.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of registered
///                                                sync objects has been
///                                                reached.
/// @retval NVMEDIA_STATUS_INVALID_STATE           The function was called in
///                                                incorrect system state.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to register the
///                                                sync object.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a syncObj must be valid NvSciSyncObj handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa VicDiagnosticsUnregisterNvSciSyncObj()
/// @sa VicDiagnosticsAttributes.maxRegisteredSyncs
///
NvMediaStatus
VicDiagnosticsRegisterNvSciSyncObj(VicDiagnostics const * const handle,
                                   NvMediaNvSciSyncObjType const syncObjType,
                                   NvSciSyncObj const syncObj);

/// @brief Unregisters an #NvSciSyncObj with #VicDiagnostics.
///
/// Every #NvSciSyncObj registered with #VicDiagnostics by
/// VicDiagnosticsRegisterNvSciSyncObj() must be unregistered before you
/// call VicDiagnosticsDestroy().
///
/// Before the application calls this function, it must ensure that any
/// #VicDiagnosticsExecute() operation that uses the #NvSciSyncObj has
/// completed. If this function is called while #NvSciSyncObj is still in use by
/// any VicDiagnosticsExecute() operation, an error will be returned.
///
/// @param[in] handle   Pointer to the VicDiagnostics context.
/// @param[in] syncObj  The #NvSciSyncObj to be unregistered.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Sync object unregistered successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a syncObj is invalid,
///                                       or the sync object or its duplicate
///                                       was not registered.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        The sync object is still being used by
///                                       a pending operation.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to unregister the sync object.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a syncObj must be valid NvSciSyncObj handle.
/// - @a syncObj must have been previously registered with
///   VicDiagnosticsRegisterNvSciSyncObj().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @sa VicDiagnosticsRegisterNvSciSyncObj()
///
NvMediaStatus
VicDiagnosticsUnregisterNvSciSyncObj(VicDiagnostics const * const handle,
                                     NvSciSyncObj const syncObj);

/// @brief Specifies the #NvSciSyncObj to be used for EOF event.
///
/// @param[in] handle   Pointer to the VicDiagnostics context.
/// @param[in] params   An VicDiagnosticsParameters handle.
/// @param[in] syncObj  The #NvSciSyncObj to be used for the EOF fence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             EOF fence set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  The sync object or its duplicate is
///                                       not registered as an EOF event type
///                                       with Vic Diagnostics, or one of the
///                                       parameters has an invalid value. This
///                                       could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a syncObj is invalid.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a params must be valid VicDiagnosticsParameters handle created with
///   VicDiagnosticsGetParameters().
/// - @a syncObj must be valid NvSciSyncObj handle.
/// - @a syncObj must have been previously registered with
///   VicDiagnosticsRegisterNvSciSyncObj() using an EOF event type.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa VicDiagnosticsRegisterNvSciSyncObj()
///
NvMediaStatus
VicDiagnosticsSetNvSciSyncObjforEOF(VicDiagnostics const * const handle,
                                    VicDiagnosticsParameters const params,
                                    NvSciSyncObj const syncObj);

/// @brief Insert an #NvSciSyncFence as a pre-fence.
///
/// This function inserts the specified #NvSciSyncFence as a pre-fence to the
/// diagnostic operation. The #VicDiagnosticsExecute() operation is started
/// only after the expiry of the @a syncFence.
///
/// For example, in this sequence of code:
/// @code
/// VicDiagnosticsInsertPreNvSciSyncFence(handle, params, syncFence);
/// VicDiagnosticsExecute(handle, params, NULL);
/// @endcode
/// the #VicDiagnosticsExecute () operation is assured to start only after
/// the expiry of @a syncFence.
///
/// You can set a maximum of 16 prefences by calling
/// #VicDiagnosticsInsertPreNvSciSyncFence().
///
/// @param[in] handle     Pointer to the VicDiagnostics context.
/// @param[in] params     An VicDiagnosticsParameters handle.
/// @param[in] syncFence  Pointer to an #NvSciSyncFence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Pre-fence inserted
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           The sync object or its
///                                                duplicate is not registered
///                                                as an PRESYNC type with
///                                                Vic Diagnostics, or one of
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
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a params must be valid VicDiagnosticsParameters handle created with
///   VicDiagnosticsGetParameters().
/// - @a syncFence must be a valid NvSciSyncFence handle.
/// - The sync object associated with @a syncFence must have been previously
///   registered with VicDiagnosticsRegisterNvSciSyncObj() using a PRESYNC event
///   type.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa VicDiagnosticsRegisterNvSciSyncObj()
///
NvMediaStatus
VicDiagnosticsInsertPreNvSciSyncFence(VicDiagnostics const * const handle,
                                      VicDiagnosticsParameters const params,
                                      NvSciSyncFence const * const syncFence);

/// @brief Gets an EOF #NvSciSyncFence for an VicDiagnosticsExecute()
/// operation.
///
/// The expiry of an EOF #NvSciSyncFence associated with an
/// #VicDiagnosticsExecute() operation indicates that the corresponding
/// #VicDiagnosticsExecute() operation has finished.
///
/// To be able to get the EOF fence using this function, the #NvSciSyncObj to be
/// used for EOF event needs to have been set for the
/// #VicDiagnosticsParameters handle used with the
/// #VicDiagnosticsExecute() operation with
/// #VicDiagnosticsSetNvSciSyncObjforEOF().
///
/// For example, in this sequence of code:
/// @code
/// VicDiagnosticsSetNvSciSyncObjforEOF(handle, params, eofSyncObj);
/// VicDiagnosticsExecute(handle, params, &result);
/// VicDiagnosticsGetEOFNvSciSyncFence(handle, &result, &syncFence);
/// @endcode
/// expiry of @a syncFence indicates that the preceding
/// #VicDiagnosticsExecute() operation has finished.
///
/// For a given NvSciSyncObj used for the EOF event, the EOF fence for a
/// VicDiagnosticsExecute() operation can be queried using this function only
/// until the next operation using the same NvSciSyncObj for the EOF event is
/// submitted. When more operations are submitted, the VicDiagnosticsResult
/// structs for the previous operations are no longer considered valid by this
/// function.
///
/// The fence returned in @a syncFence is tied to the specific VicDiagnostics
/// context instance passed in @a handle and must not be used after the
/// VicDiagnostics context instance has been destroyed.
///
/// @param[in]  handle     Pointer to the VicDiagnostics context.
/// @param[in]  result     Pointer to the VicDiagnosticsResult struct.
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
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a result must be valid VicDiagnosticsResult handle returned by
///   VicDiagnosticsExecute().
/// - The operation that produced @a result must have had an EOF sync object
///   configured with VicDiagnosticsSetNvSciSyncObjforEOF().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa VicDiagnosticsRegisterNvSciSyncObj()
/// @sa VicDiagnosticsSetNvSciSyncObjforEOF()
///
NvMediaStatus
VicDiagnosticsGetEOFNvSciSyncFence(VicDiagnostics const * const handle,
                                   VicDiagnosticsResult const * const result,
                                   NvSciSyncFence * const syncFence);

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // VICDIAGNOSTICS_SCI_H
