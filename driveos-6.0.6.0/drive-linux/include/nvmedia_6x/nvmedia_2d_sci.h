/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/// @file
/// @brief <b> NVIDIA Media Interface: 2D NvSci </b>
///
/// @b Description: This file contains the NvMedia 2D and NvSci related APIs.
///

#ifndef NVMEDIA_2D_SCI_H
#define NVMEDIA_2D_SCI_H

#include "nvmedia_core.h"
#include "nvmedia_2d.h"
#include "nvscibuf.h"
#include "nvscisync.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_nvmedia_2d_sci_api 2D Surface and Synchronization
///
/// The NvMedia 2D NvSci API encompasses all NvMedia 2D handling for NvSciBuf
/// NvSciSync related functions.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Fills the NvMedia 2D specific NvSciBuf attributes.
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
/// @pre
/// - @a attrList must be valid NvSciBufAttrList handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvMediaStatus
NvMedia2DFillNvSciBufAttrList(NvMedia2D * const handle,
                              NvSciBufAttrList const attrList);

/// @brief Fills the NvMedia 2D specific NvSciSync attributes.
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
/// @param[in]  handle      Pointer to the NvMedia2D context.
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
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a attrList must be valid NvSciSyncAttrList handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvMediaStatus
NvMedia2DFillNvSciSyncAttrList(NvMedia2D * const handle,
                               NvSciSyncAttrList const attrList,
                               NvMediaNvSciSyncClientType const clientType);

/// @brief Register an #NvSciBufObj with NvMedia 2D.
///
/// Every NvSciBufObj (even duplicate objects) used by NvMedia 2D
/// must be registered by a call to this function before it is used.
/// Only the exact same registered NvSciBufObj can be passed to
/// NvMedia2DSetSrcNvSciBufObj() and NvMedia2DSetDstNvSciBufObj() functions.
///
/// @param[in] handle  Pointer to the NvMedia2D context.
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
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a bufObj must be valid NvSciBufObj handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMedia2DUnregisterNvSciBufObj()
///
NvMediaStatus
NvMedia2DRegisterNvSciBufObj(NvMedia2D * const handle,
                             NvSciBufObj const bufObj);

/// @brief Register an #NvSciSyncObj with NvMedia 2D.
///
/// Every NvSciSyncObj (even duplicate objects) used by NvMedia 2D
/// must be registered by a call to this function before it is used.
///
/// Only the exact same registered NvSciSyncObj can be passed to
/// NvMedia2DSetNvSciSyncObjforEOF() or NvMedia2DUnregisterNvSciSyncObj(). Also
/// the NvSciSyncObjs associated with any NvSciSyncFences passed to
/// NvMedia2DInsertPreNvSciSyncFence() must be registered.
///
/// For a given NvMedia2D handle, one NvSciSyncObj can be registered as one
/// #NvMediaNvSciSyncObjType only.
///
/// @param[in] handle       Pointer to the NvMedia2D context.
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
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a syncObj must be valid NvSciSyncObj handle.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMedia2DUnregisterNvSciSyncObj()
/// @sa NvMedia2DAttributes.maxRegisteredSyncs
///
NvMediaStatus
NvMedia2DRegisterNvSciSyncObj(NvMedia2D * const handle,
                              NvMediaNvSciSyncObjType const syncObjType,
                              NvSciSyncObj const syncObj);

/// @brief Unregisters an #NvSciBufObj with #NvMedia2D.
///
/// Every #NvSciBufObj registered with #NvMedia2D by
/// NvMedia2DRegisterNvSciBufObj() must be unregistered before you call
/// NvMedia2DDestroy().
///
/// @param[in] handle  Pointer to the NvMedia2D context.
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
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a bufObj must be valid NvSciBufObj handle.
/// - @a bufObj must have been previously registered with
///   NvMedia2DRegisterNvSciBufObj().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @sa NvMedia2DRegisterNvSciBufObj()
///
NvMediaStatus
NvMedia2DUnregisterNvSciBufObj(NvMedia2D * const handle,
                               NvSciBufObj bufObj);

/// @brief Unregisters an #NvSciSyncObj with #NvMedia2D.
///
/// Every #NvSciSyncObj registered with #NvMedia2D by
/// NvMedia2DRegisterNvSciSyncObj() must be unregistered before you call
/// NvMedia2DDestroy().
///
/// Before the application calls this function, it must ensure that any
/// #NvMedia2DCompose() operation that uses the #NvSciSyncObj has completed.
/// If this function is called while #NvSciSyncObj is still in use by any
/// NvMedia2DCompose() operation, behavior is undefined.
///
/// @param[in] handle   Pointer to the NvMedia2D context.
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
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a syncObj must be valid NvSciSyncObj handle.
/// - @a syncObj must have been previously registered with
///   NvMedia2DRegisterNvSciSyncObj().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @sa NvMedia2DRegisterNvSciSyncObj()
///
NvMediaStatus
NvMedia2DUnregisterNvSciSyncObj(NvMedia2D * const handle,
                                NvSciSyncObj syncObj);

/// @brief Specifies the #NvSciSyncObj to be used for EOF event.
///
/// @param[in] handle   Pointer to the NvMedia2D context.
/// @param[in] params   An NvMedia2DComposeParameters handle.
/// @param[in] syncObj  The #NvSciSyncObj to be used for the EOF fence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             EOF fence set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  The sync object is not registered as
///                                       an EOF event type with NvMedia 2D, or
///                                       one of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a syncObj is invalid.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
/// - @a syncObj must be valid NvSciSyncObj handle.
/// - @a syncObj must have been previously registered with
///   NvMedia2DRegisterNvSciSyncObj() using an EOF event type.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DRegisterNvSciSyncObj()
///
NvMediaStatus
NvMedia2DSetNvSciSyncObjforEOF(NvMedia2D * const handle,
                               NvMedia2DComposeParameters const params,
                               NvSciSyncObj syncObj);

/// @brief Insert an #NvSciSyncFence as a pre-fence.
///
/// This function inserts the specified #NvSciSyncFence as a pre-fence to the
/// compose operation. The #NvMedia2DCompose() operation is started only after
/// the expiry of the @a syncFence.
///
/// For example, in this sequence of code:
/// @code
/// NvMedia2DInsertPreNvSciSyncFence(handle, params, syncFence);
/// NvMedia2DCompose(handle, params, NULL);
/// @endcode
/// the #NvMedia2DCompose () operation is assured to start only after the expiry
/// of @a syncFence.
///
/// You can set a maximum of 16 prefences by calling
/// #NvMedia2DInsertPreNvSciSyncFence().
///
/// @param[in] handle     Pointer to the NvMedia2D context.
/// @param[in] params     An NvMedia2DComposeParameters handle.
/// @param[in] syncFence  Pointer to an #NvSciSyncFence.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Pre-fence inserted
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           The sync object is not
///                                                registered as an PRESYNC type
///                                                with NvMedia 2D, or one of
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
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
/// - @a syncFence must be a valid NvSciSyncFence handle.
/// - The sync object associated with @a syncFence must have been previously
///   registered with NvMedia2DRegisterNvSciSyncObj() using a PRESYNC event
///   type.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DRegisterNvSciSyncObj()
///
NvMediaStatus
NvMedia2DInsertPreNvSciSyncFence(NvMedia2D * const handle,
                                 NvMedia2DComposeParameters const params,
                                 NvSciSyncFence const * const syncFence);

/// @brief Gets an EOF #NvSciSyncFence for an NvMedia2DCompose() operation.
///
/// The expiry of an EOF #NvSciSyncFence associated with an #NvMedia2DCompose()
/// operation indicates that the corresponding #NvMedia2DCompose() operation
/// has finished.
///
/// To be able to get the EOF fence using this function, the #NvSciSyncObj to be
/// used for EOF event needs to have been set for the
/// #NvMedia2DComposeParameters handle used with the #NvMedia2DCompose()
/// operation with #NvMedia2DSetNvSciSyncObjforEOF().
///
/// For example, in this sequence of code:
/// @code
/// NvMedia2DSetNvSciSyncObjforEOF(handle, params, eofSyncObj);
/// NvMedia2DCompose(handle, params, &result);
/// NvMedia2DGetEOFNvSciSyncFence(handle, &result, &syncFence);
/// @endcode
/// expiry of @a syncFence indicates that the preceding #NvMedia2DCompose()
/// operation has finished.
///
/// @param[in]  handle     Pointer to the NvMedia2D context.
/// @param[in]  result     Pointer to the NvMedia2DResult struct.
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
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a result must be valid NvMedia2DComposeResult handle returned by
///   NvMedia2DCompose().
/// - The compose operation that produced @a result must have had an EOF sync
///   object configured with NvMedia2DSetNvSciSyncObjforEOF().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DRegisterNvSciSyncObj()
/// @sa NvMedia2DSetNvSciSyncObjforEOF()
///
NvMediaStatus
NvMedia2DGetEOFNvSciSyncFence(NvMedia2D * const handle,
                              NvMedia2DComposeResult const * const result,
                              NvSciSyncFence * const syncFence);

/// @brief Sets the surface for a source layer.
///
/// @param[in] handle      Pointer to the NvMedia2D context.
/// @param[in] params      An NvMedia2DComposeParameters handle.
/// @param[in] index       Index of source layer to configure.
/// @param[in] srcSurface  The #NvSciBufObj to be used for the source surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a index is out of range,
///                                       or the buffer was not registered.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
/// - @a srcSurface must be valid NvSciBufObj handle.
/// - @a srcSurface must have been previously registered with
///   NvMedia2DRegisterNvSciBufObj().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DComposeParameters
/// @sa NvMedia2DCompose()
/// @sa NvMedia2DRegisterNvSciBufObj()
///
NvMediaStatus
NvMedia2DSetSrcNvSciBufObj(NvMedia2D * const handle,
                           NvMedia2DComposeParameters const params,
                           uint32_t const index,
                           NvSciBufObj srcSurface);

/// @brief Sets the surface for the destination.
///
/// @param[in] handle      Pointer to the NvMedia2D context.
/// @param[in] params      An NvMedia2DComposeParameters handle.
/// @param[in] dstSurface  The #NvSciBufObj to be used for the destination
///                        surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid,
///                                       or the buffer was not registered.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
/// - @a dstSurface must be valid NvSciBufObj handle.
/// - @a dstSurface must have been previously registered with
///   NvMedia2DRegisterNvSciBufObj().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DComposeParameters
/// @sa NvMedia2DCompose()
/// @sa NvMedia2DRegisterNvSciBufObj()
///
NvMediaStatus
NvMedia2DSetDstNvSciBufObj(NvMedia2D * const handle,
                           NvMedia2DComposeParameters const params,
                           NvSciBufObj dstSurface);

///
/// @}
///

#ifdef __cplusplus
}
#endif

#endif // NVMEDIA_2D_SCI_H
