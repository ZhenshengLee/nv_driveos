/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// @brief <b> NVIDIA Media Interface: VIC HW Diagnostics Control </b>
///
/// @b Description: This file contains the "VIC HW Diagnostics API."
///

#ifndef VICDIAGNOSTICS_H
#define VICDIAGNOSTICS_H

#include "nvmedia_core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_vicdiagnostics_api VIC HW Diagnostics
///
/// The VIC HW Diagnostics API encompasses functionality to run diagnostic
/// tests on VIC hardware.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Major version number of VIC Diagnostics header.
///
/// This defines the major version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #VicDiagnosticsGetVersion() function.
///
/// @sa VicDiagnosticsGetVersion()
///
#define VICDIAGNOSTICS_VERSION_MAJOR   1

/// @brief Minor version number of VIC Diagnostics header.
///
/// This defines the minor version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #VicDiagnosticsGetVersion() function.
///
/// @sa VicDiagnosticsGetVersion()
///
#define VICDIAGNOSTICS_VERSION_MINOR   0

/// @brief Patch version number of VIC Diagnostics header.
///
/// This defines the patch version of the API defined in this header.
///
/// @sa VicDiagnosticsGetVersion()
///
#define VICDIAGNOSTICS_VERSION_PATCH   1

/// @brief Attributes structure for #VicDiagnosticsCreate().
///
/// This type holds the attributes to control the behaviour of the
/// VicDiagnostics context. These attributes take effect during the call
/// to #VicDiagnosticsCreate().
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa VicDiagnosticsCreate()
///
typedef struct
{
    /// @brief Number of diagnostics parameters objects to allocate.
    ///
    /// Must be in range [1, 16].
    ///
    /// @sa VicDiagnosticsParameters
    /// @sa VicDiagnosticsGetParameters()
    ///
    uint32_t numDiagnosticsParameters;

    /// @brief Maximum number of sync objects that can be registered.
    ///
    /// Must be in range [1, 256].
    ///
    /// @sa VicDiagnosticsRegisterNvSciSyncObj()
    ///
    uint32_t maxRegisteredSyncs;

    /// @brief Internal use only.
    uint32_t flags;

} VicDiagnosticsAttributes;

/// @brief Stores configuration for the VicDiagnosticsExecute() operation.
///
/// This object stores the information needed to configure the diagnostic
/// operation that is executed inside the VicDiagnosticsExecute() function.
///
/// The underlying object cannot be instantiated directly by the client. Instead
/// use the #VicDiagnosticsGetParameters() function to retrieve a handle to
/// an available instance.
///
/// Value 0 is never a valid handle value, and can be used to initialize a
/// VicDiagnosticsParameters handle to a known value.
///
/// @sa VicDiagnosticsGetParameters()
/// @sa VicDiagnosticsInsertPreNvSciSyncFence()
/// @sa VicDiagnosticsSetNvSciSyncObjforEOF()
/// @sa VicDiagnosticsExecute()
///
typedef uint32_t VicDiagnosticsParameters;

/// @brief Stores information returned from VicDiagnosticsExecute().
///
/// This type holds the information about an operation that has been submitted
/// to VicDiagnosticsExecute() for execution.
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa VicDiagnosticsExecute()
///
typedef struct
{
    /// @brief ID number for operation that was submitted to VicDiagnosticsExecute().
    ///
    /// The number will wrap once the uint64_t range has been exceeded. A value
    /// of 0 indicates that no operation was submitted.
    ///
    uint64_t operationId;
} VicDiagnosticsResult;

/// @brief Returns the version number of the NvMedia Vic Diagnostics library.
///
/// This function returns the major and minor version number of the
/// Vic Diagnostics library. The client must pass an #NvMediaVersion struct
/// to this function, and the version information will be returned in this
/// struct.
///
/// This allows the client to verify that the version of the library matches
/// and is compatible with the the version number of the header file they are
/// using.
///
/// @param[out] version  Pointer to an #NvMediaVersion struct that will be
///                      populated with the version information.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Version information returned
///                                       successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a version is NULL.
///
/// @pre None.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaVersion object
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: Yes
///   - De-Init: Yes
///
/// @sa VICDIAGNOSTICS_VERSION_MAJOR
/// @sa VICDIAGNOSTICS_VERSION_MINOR
/// @sa VICDIAGNOSTICS_VERSION_PATCH
/// @sa NvMediaVersion
///
NvMediaStatus
VicDiagnosticsGetVersion(NvMediaVersion * const version);

/// @brief VicDiagnostics Context.
///
/// This type represents a context for the VicDiagnostics library.
/// This context is an opaque data type that encapsulates the state needed to
/// service the VicDiagnostics API calls.
///
/// @sa VicDiagnosticsCreate()
///
typedef struct VicDiagnostics VicDiagnostics;

/// @brief Creates a new VicDiagnostics context.
///
/// This function creates a new instance of an VicDiagnostics context, and
/// returns a pointer to that context. Ownership of this context is passed to
/// the caller. When no longer in use, the caller must destroy the context
/// using the #VicDiagnosticsDestroy() function.
///
/// Default attributes (when not specified by caller):
/// - numDiagnosticsParameters: 1
/// - maxRegisteredSyncs: 16
/// - flags: 0.
///
/// @param[out] handle  Pointer to receive the handle to the new
///                     VicDiagnostics context.
/// @param[in]  attr    Pointer to VicDiagnosticsAttributes struct, or
///                     NULL for default attributes.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context created successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL, or @a attr has bad
///                                       attribute values.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  NvMedia Vic Diagnostics is not
///                                       supported on this hardware platform.
/// @retval NVMEDIA_STATUS_OUT_OF_MEMORY  Memory allocation failed for internal
///                                       data structures or device memory
///                                       buffers.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to create the context.
///
/// @pre None.
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
/// @sa VicDiagnostics
/// @sa VicDiagnosticsDestroy()
///
NvMediaStatus
VicDiagnosticsCreate(VicDiagnostics ** const handle,
                     VicDiagnosticsAttributes const * const attr);

/// @brief Destroys the VicDiagnostics context.
///
/// This function destroys the specified VicDiagnostics context.
///
/// Before calling this function, the caller must ensure:
/// - There are no NvSciSync objects still registered against the
///   VicDiagnostics context.
/// - All previous Vic Diagnostics operations submitted using
///   VicDiagnosticsExecute() have completed.
///
/// @param[in] handle  Pointer to the VicDiagnostics context.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context destroyed successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        There are still some NvSciSync objects
///                                       registered against the
///                                       VicDiagnostics context.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to destroy the context. The
///                                       context is in state where the only
///                                       valid operation is to attempt to
///                                       destroy it again.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
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
/// @sa VicDiagnostics
/// @sa VicDiagnosticsCreate()
///
NvMediaStatus
VicDiagnosticsDestroy(VicDiagnostics const * const handle);

/// @brief Returns an #VicDiagnosticsParameters instance.
///
/// This functions returns a handle to an VicDiagnosticsParameters object.
/// The object will be initialized and ready to use. The caller takes ownership
/// of this handle. Ownership will be passed back to the VicDiagnostics
/// context when it is subsequently used in the #VicDiagnosticsExecute()
/// operation.
///
/// The handle returned in @a params is tied to the specific VicDiagnostics
/// context instance passed in @a handle and cannot be used with other context
/// instances.
///
/// @param[in]  handle  Pointer to the VicDiagnostics context.
/// @param[out] params  Pointer to an #VicDiagnosticsarameters, which will
///                     be populated with the handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Parameters instance is
///                                                initialized successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           One of the parameters has an
///                                                invalid value, either:
///                                                - @a handle is NULL
///                                                - @a params is NULL.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  There is no free instance
///                                                available.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to retrieve the
///                                                parameters object.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
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
/// @sa VicDiagnosticsParameters
/// @sa VicDiagnosticsExecute()
///
NvMediaStatus
VicDiagnosticsGetParameters(VicDiagnostics const * const handle,
                            VicDiagnosticsParameters * const params);

/// @brief Task status value indicating the test checksum did not match the
///        expected one.
#define VICDIAGNOSTICS_TASK_STATUS_CHECKSUM_MISMATCH 41u

/// @brief Performs a VIC HW diagnostic operation.
///
/// A diagnostic operation performs a pre-defined hardware diagnostic test based
/// on calculating a checksum for an operation and comparing it to an expected
/// value. The test to be run can be chosen with VicDiagnosticsSetTest().
///
/// To get full diagnostic coverage of the VIC HW, all of the available tests
/// need to be executed periodically. The possibility to select a single test to
/// be run independently through VicDiagnosticsSetTest() is offered to allow for
/// more fine-grained scheduling of the test execution.
///
/// The result of the diagnostic test is indicated in the task status that can
/// be queried from the EOF fence:
/// - NvSciSyncTaskStatus_Success: test passed
/// - VICDIAGNOSTICS_TASK_STATUS_CHECKSUM_MISMATCH: calculated checksum did not
///   match the expected one
/// - Other: an internal failure
///
/// Because of this, EOF sync object must always be set with
/// VicDiagnosticsSetNvSciSyncObjforEOF() prior to calling
/// VicDiagnosticsExecute().
///
/// This example runs a diagnostic test with id 0.
///
/// @code
///     VicDiagnosticsParameters params;
///     VicDiagnosticsGetParameters(handle, &params);
///     VicDiagnosticsSetTest(handle, params, 0);
///     VicDiagnosticsSetNvSciSyncObjforEOF(handle, params, eofSyncObj);
///     VicDiagnosticsExecute(handle, params, NULL);
/// @endcode
///
/// @anchor VicDiagnosticsExecuteConcurrencyRestrictions
///
/// Restrictions on concurrency:
/// - There can be a maximum of 16 operations submitted through the same
///   VicDiagnostics handle pending simultaneously.
///
/// If any of the restrictions are violated, this function will fail with an
/// error code.
///
/// The result info returned in @a result is tied to the specific VicDiagnostics
/// context instance passed in @a handle and cannot be used with other context
/// instances.
///
/// @param[in]  handle  Pointer to the VicDiagnostics context.
/// @param[in]  params  An VicDiagnosticsParameters handle.
/// @param[out] result  Pointer to VicDiagnosticsResult struct that will
///                     be populated with result info. May be NULL.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Operation submitted successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - EOF sync object has not been set
///                                       - some of the parameters configured
///                                         through @a params have invalid
///                                         values.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  Requested operation is not supported
///                                       by current platform.
/// @retval NVMEDIA_STATUS_TIMED_OUT      No space available in the command
///                                       buffer for this operation, because
///                                       previous operations are still
///                                       pending (see @ref
///                                       VicDiagnosticsExecuteConcurrencyRestrictions
///                                       "restrictions on concurrency"). The
///                                       caller should wait for the least
///                                       recently submitted operation to
///                                       complete and then try again.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to perform the diagnostic
///                                       operation. This error indicates the
///                                       system is potentially in an
///                                       unrecoverable state.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a params must be valid VicDiagnosticsParameters handle created with
///   VicDiagnosticsGetParameters().
/// - @a params must have a valid EOF sync object set with
///   VicDiagnosticsSetNvSciSyncObjforEOF().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different VicDiagnostics handle
///   - Re-entrant: No
///   - Async/Sync: Async
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa VicDiagnosticsParameters
/// @sa VicDiagnosticsGetParameters()
/// @sa VicDiagnosticsInsertPreNvSciSyncFence()
/// @sa VicDiagnosticsSetNvSciSyncObjforEOF()
/// @sa VicDiagnosticsGetEOFNvSciSyncFence()
///
NvMediaStatus
VicDiagnosticsExecute(VicDiagnostics const * const handle,
                      VicDiagnosticsParameters const params,
                      VicDiagnosticsResult * const result);

/// @brief Gets the number of available diagnostics tests.
///
/// This function returns the number of available diagnostics tests.
///
/// @param[in] handle        Pointer to the VicDiagnostics context.
/// @param[out] numTests     A uint32_t to receive the number of available tests.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Test id was set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a numTests is NULL.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
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
/// @sa VicDiagnosticsSetTest()
///
NvMediaStatus
VicDiagnosticsGetNumTests(VicDiagnostics const * const handle,
                          uint32_t * const numTests);

/// @brief Sets the test ID for an #VicDiagnosticsParameters instance.
///
/// This function updates the VicDiagnosticsParameters instance with a
/// test ID to be run when VicDiagnosticsExecute() is called. Test ID must
/// be in the range [0, @a VicDiagnosticsGetNumTests()[
///
/// @param[in] handle        Pointer to the VicDiagnostics context.
/// @param[in] params        An #VicDiagnosticsParameters handle.
/// @param[in] testId        A testId to run.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Test id was set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a params is invalid.
///                                       - @a testId is out of range.
///
/// @pre
/// - @a handle must be valid VicDiagnostics handle created with
///   VicDiagnosticsCreate().
/// - @a params must be valid VicDiagnosticsParameters handle created with
///   VicDiagnosticsGetParameters().
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
/// @sa VicDiagnosticsGetNumTests
///
NvMediaStatus
VicDiagnosticsSetTest(VicDiagnostics const * const handle,
                      VicDiagnosticsParameters const params,
                      uint32_t const testId);

///
/// @}
///

//
// Version History
//
// Version 1.0 August 10, 2022
// - Initial release
//
// Version 1.0.1 September 2, 2022
// - Always treat parameters handle value 0 as invalid
//

#ifdef __cplusplus
}
#endif

#endif // VICDIAGNOSTICS_H
