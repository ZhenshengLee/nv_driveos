/*
 * Copyright (c) 2014-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef INCLUDED_nvrm_gpu_H
#define INCLUDED_nvrm_gpu_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "nvcommon.h"
#include "nverror.h"

#if defined(__cplusplus)
extern "C"
{
#endif

#if defined(NVRM_GPU_BUILD_VARIANT)
namespace nvrm_gpu
{
#endif

/// @file
/// @brief <b>NVIDIA GPU access API</b>
///
/// @ifnot NVRM_GPU_PRIVATE
/// @b Description: This file contains API for GPU library management, clock
/// controls, sensors, and device events.
/// @endif
///
/// Refer to @ref nvrm_gpu_group for general documentation.
///
/// @defgroup nvrm_gpu_group GPU access API
///
/// GPU management and access API.
///
/// @ifnot NVRM_GPU_PRIVATE
/// For related information, see
/// <a href="../Clock Freq and Thermal/discrete_GPU.html" target="_blank">
/// Discrete GPU</a> in the _Development Guide_.
/// @endif
///
/// @section nvrm_gpu_api_general_design General design of nvrm_gpu
///
/// @begin_swad
///
/// @sphinx_anchor{nvrm_gpu_interface_ood}
///
/// @subsection nvrm_gpu_api_general_design_ood Object-oriented design
///
/// nvrm_gpu follows object-oriented design. In general, objects are created by
/// API functions ending with "Open" or "Create" and they are destroyed by API
/// functions ending with "Close". For example NvRmGpuLibOpen() creates the
/// library object and NvRmGpuLibClose() destroys the object.
///
/// The objects typically have a real-world counterpart or they represent a
/// logical construct. For example, a device object represents an NVIDIA GPU.
///
/// All objects are referred by handles. A handle is a pointer-to-struct type
/// where the struct provides typing. For example, a library object is
/// referred by a variable of type `NvRmGpuLib *`. Handles cannot be
/// dereferenced by the nvrm_gpu user.
///
/// The handle lifetime is the same as the underlying object lifetime. That is,
/// a valid handle is created by an Open/Create function and a Close function
/// will invalidate the handle as well as destroy the object. For example,
/// NvRmGpuDeviceOpen() creates a device object and returns a handle to the
/// device. NvRmGpuDeviceClose() destroys the device object and invalidates the
/// device handle.
///
/// In general, before destroying an object, any possible child object must be
/// destroyed first. For example, when closing the library with
/// NvRmGpuLibClose(), all devices opened within the library scope must be
/// closed first with NvRmGpuDeviceClose(). Failing to do so will produce
/// undefined behavior in form of dangling pointers in nvrm_gpu internal objects
/// and data structures.
///
/// Unless otherwise stated, it is illegal to cross-reference handles under
/// different #NvRmGpuDevice instances. For instance, a channel belonging to one
/// device object cannot be bound to the address space belonging to another
/// device object.
///
/// @subsection nvrm_gpu_api_general_design_error_codes Common error codes
///
/// nvrm_gpu uses the following common error codes throughout the API:
///
/// <table>
///   <tr>
///     <th>Error code</th>
///     <th>Build configurations</th>
///     <th>Description</th>
///     <th>Remarks</th>
///   </tr>
///   <tr>
///     <td>NvError_GpuInvalidHandle</td>
///     <td>Safety builds only</td>
///     <td>Invalid API object handle</td>
///     <td>nvrm_gpu standard does not perform handle validation for performance reasons.</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuInvalidDvmsState</td>
///     <td>QNX builds only</td>
///     <td>API function not available in the current DVMS state</td>
///     <td>None</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuHwError</td>
///     <td>All</td>
///     <td>Error encountered during direct communication with the HW.</td>
///     <td>Examples: Unexpected register read value; unexpected control buffer value</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuOutOfOrderFree</td>
///     <td>Safety builds only</td>
///     <td>A parent (or other dependency) object freed before children (or other dependencies).</td>
///     <td>None</td>
///   </tr>
///   <tr>
///     <td>NvError_InsufficientMemory</td>
///     <td>All</td>
///     <td>Insufficient memory to perform the operation.</td>
///     <td>None</td>
///   </tr>
///   <tr>
///     <td>NvError_ResourceError</td>
///     <td>All</td>
///     <td>Error communicating with a kernel-mode driver (Linux) or a resource manager (QNX)</td>
///     <td>None</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuFatalLockdown</td>
///     <td>Safety builds only</td>
///     <td>nvrm_gpu is in lockdown mode due to a previous fatal error.</td>
///     <td>
///         When fatal error lockdown mode is enabled:
/// @if NVRM_GPU_SC_PRIVATE
///         (See @ref nvrm_gpu_sc_arch_error_handling)
/// @endif
///         - Once a fatal error has been encountered, most API calls return this error. The exceptions
///           are API calls that never fail (e.g. NvRmGpuLibGetVersionInfo()) and API calls that always fail
///           (e.g., calls disabled in the build). When a fatal error has been encountered, the integrity of
///           %nvrm_gpu is considered lost. The lockdown mode prevents any further undefined behavior.
///
///         Otherwise:
///         - This error code is never returned.
///     </td>
///   </tr>
///   <tr>
///     <td>NvError_GpuFatalConsistencyError</td>
///     <td>Safety builds only</td>
///     <td>Internal state consistency check failed.</td>
///     <td>Potential sources include memory corruption or a programming defect.</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuFatalLogicError</td>
///     <td>Safety builds only</td>
///     <td>Internal logic error detected.</td>
///     <td>
///         These errors are generally triggered due defensive mechanisms. For
///         example: internal out-of-bounds check failed; integer overflow check
///         failed; predicate check failed.
///     </td>
///   </tr>
///   <tr>
///     <td>NvError_GpuFatalOsError</td>
///     <td>Safety builds only</td>
///     <td>
///         An OS call or a library call that should never fail with correct programming returned an error.
///     </td>
///     <td>Example: pthread_mutex_lock() or pthread_mutex_unlock() returned an error.</td>
///   </tr>
///   <tr>
///     <td>NvError_GpuFatalUncheckedException</td>
///     <td>Safety builds only</td>
///     <td>An unexpected (unchecked) exception was caught.</td>
///     <td>None</td>
///   </tr>
/// </table>
///
/// @sphinx_anchor{nvrm_gpu_general_interface_contract}
///
/// @subsection nvrm_gpu_api_general_design_interface_contract General interface contract
///
/// Parameter sanity checking:
///
///  - nvrm_gpu will do general sanity checking for all received parameters with
///    the exception of pointer validity. The sanity checking includes basic
///    range checking for integer values, validity checking for enumeration
///    values, and so on. Sanity checking may be delegated to the underlying
///    kernel driver, other shared libraries used by nvrm_gpu, or the GPU itself.
///
///  - Pointers passed as parameters are not validated. nvrm_gpu is a userspace
///    component and it has no efficient way of validating pointers. This
///    includes also the nvrm_gpu object handles. All non-@NULL pointers are
///    expected to be valid.
///
/// Objects and their relations:
///
///  - In object creation, the first parameter specifies the parent. Some objects
///    may be created for parents of different types. When this is the case, it
///    is specifically stated in the API.
///
///  - It is a fatal error to close a parent or otherwise related object before
///    the child object, unless otherwise stated. The parent and other related
///    objects are either referenced in the Open/Create functions or specific
///    other functions that bind objects together.
///
///  - The Close functions can always be supplied with @NULL handles, unless
///    otherwise stated. Closing a @NULL handle is a no-operation. This allows
///    the nvrm_gpu users to unconditionally close all handles in
///    deinitialization, regardless of whether they refer to valid objects or
///    @NULL. (See NvRmGpuLibClose() for an example.)
///
/// Handles:
///
///  - nvrm_gpu generally assumes that all nvrm_gpu handles passed to
///    the nvrm_gpu API functions are valid, correctly typed, and non-@NULL.
///    Individual API functions may relax the non-@NULL requirement by marking
///    input handles as optional in the API function documentation, in which case
///    a @NULL pointer may be passed, instead.
///
///  - Generally, a valid handle is obtained from an API function named
///    "create" or "open". Valid handle is invalidated by calling an API
///    function named "close".
///
/// Pointers:
///
///  - All passed pointers to memory are expected to be pointers to the regular
///    memory type. On AArch64, this is the Normal memory type. In particular,
///    Device memory types (e.g., Device-GRE) are excluded. All dereferenceable
///    pointers received from nvrm_gpu are regular memory types unless otherwise
///    stated.
///
///  - All passed pointers are expected to be aligned by the C platform ABI
///    rules.
///
///  - Some nvrm_gpu API functions return const pointers as return values. In
///    general, the pointers are valid as long as the related object is valid
///    unless otherwise stated. And in particular, the caller shall not attempt
///    to free them. For example, the device list returned by
///    NvRmGpuLibListDevices() is valid until the related library handle is
///    closed.
///
/// Forwards and backwards compatibility:
///
///  - Many nvrm_gpu functions expect a pointer to an attribute struct. The
///    attribute struct provides room for future extensibility while maintaining
///    source-level backwards compatibility. Generally, a NULL pointer is also
///    accepted in which case nvrm_gpu assumes default attributes.  The nvrm_gpu
///    users should define the attribute structs with macros provided by the
///    nvrm_gpu API. (See NvRmGpuDeviceOpen() and
///    NVRM_GPU_DEFINE_DEVICE_OPEN_ATTR() for instance.)
///
///  - In general, nvrm_gpu attempts to maintain source-level backwards
///    compatibility. This means that the user may upgrade nvrm_gpu without
///    changes in the nvrm_gpu users' sources. However, a recompilation may be
///    needed, as nvrm_gpu does not provide binary compatibility.
///
///  - The reasons for breaking the backwards compatibility are usually either
///    due to security reasons or that the functionality has been completely
///    removed.
///
/// @sphinx_anchor{nvrm_gpu_concurrency_and_thread_safety}
///
/// @subsection nvrm_gpu_api_general_design_concurrency Concurrency and thread safety
///
/// nvrm_gpu is thread-safe with the following rule: an object may be closed
/// only if there are no concurrent operations on-going on it. Attempting to
/// close an object in one thread when another thread is still accessing it is a
/// fatal error.
///
/// nvrm_gpu internally uses fine-grain locking to promote high-performance
/// multi-threaded programming. The nvrm_gpu implementation uses the partial lock
/// ordering technique to avoid deadlocks related to nested locking.
///
/// @end_swad
///
/// @subsection nvrm_gpu_api_general_design_safety_subset Safety-certified subset
///
/// nvrm_gpu is subject to safety certification for specific releases. The
/// nvrm_gpu library in the safety-certified release comes in form of special
/// build called the <em>safety build</em> of nvrm_gpu. The safety build
/// supports a subset of the nvrm_gpu API functionality. This is referred as the
/// <em>safety subset</em>.
///
/// The safety subset available in the safety build is denoted by API groups
/// that have "safety subset" in their name. Functionality that is not within
/// the "safety subset" groups is not available in the safety build and must not
/// be attempted to be used in a safety-critical context. The top-level safety
/// subset API group is @ref nvrm_gpu_safety_group.
///
/// The use of nvrm_gpu in a safety-critical context is further subject to
/// certain assumptions, restrictions, and recommendations. These are described
/// in the safety manual provided in the safety-certified release.
///
/// The safety build of nvrm_gpu has additional internal checks enabled that are
/// not available in the regular builds for performance reasons.

// -------------------------------------------------
// --------------- API Groups ----------------------
// -------------------------------------------------

/// @defgroup nvrm_gpu_safety_group GPU access API (safety subset)

/// @defgroup nvrm_gpu_lib_group GPU access API: Library
/// @ingroup nvrm_gpu_group
///
/// @brief Library management and device discovery.

/// @defgroup nvrm_gpu_lib_safety_group GPU access API: Library (safety subset)
/// @ingroup nvrm_gpu_lib_group
/// @ingroup nvrm_gpu_safety_group

/// @defgroup nvrm_gpu_device_group GPU access API: Device management
/// @ingroup nvrm_gpu_group
///
/// @brief Device control, device capabilities, and device memory management.

/// @defgroup nvrm_gpu_device_safety_group GPU access API: Device management (safety subset)
/// @ingroup nvrm_gpu_device_group
/// @ingroup nvrm_gpu_safety_group

// -------------------------------------------------
// --------------- Handles -------------------------
// -------------------------------------------------

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Library handle
///
/// @sa NvRmGpuLibOpen()
/// @sa NvRmGpuLibGetInfo()
/// @sa NvRmGpuLibClose()
///
typedef struct NvRmGpuLibRec NvRmGpuLib;

/// @ingroup nvrm_gpu_device_safety_group
/// @brief Device handle
///
/// @sa NvRmGpuDeviceOpen()
/// @sa NvRmGpuDeviceGetInfo()
/// @sa NvRmGpuDeviceClose()
///
typedef struct NvRmGpuDeviceRec NvRmGpuDevice;

/// @ingroup nvrm_gpu_device_group
/// @brief Device event session handle
///
/// @sa NvRmGpuDeviceEventSessionOpen()
/// @sa NvRmGpuDeviceEventSessionClose()
typedef struct NvRmGpuDeviceEventSessionRec NvRmGpuDeviceEventSession;


// -------------------------------------------------
// --------------- Library functions ---------------
// -------------------------------------------------

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief The version information structure returned by
/// NvRmGpuLibGetVersionInfo().
///
/// @since 0.1.0
typedef struct NvRmGpuLibVersionInfoRec
{
    /// @brief The library major version
    ///
    /// @since 0.1.0
    uint32_t major;

    /// @brief The library minor version
    ///
    /// @since 0.1.0
    uint32_t minor;

    /// @brief The library patch level
    ///
    /// @since 0.1.0
    uint32_t patch;

    /// @brief Version string suffix (always non-@NULL)
    ///
    /// When the version string suffix is:
    /// - An empty string, the version is a general release.
    /// - A non-empty string, the version is a special release and non-standard
    ///   compatibility rules may apply. For example:
    ///   - @c "-dev" : Internal development release
    ///   - @c "-rel34" : Version numbering is specific to NVIDIA Tegra release branch 34
    ///
    /// @since 0.1.0
    const char *suffix;

} NvRmGpuLibVersionInfo;

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Returns the library version information.
///
/// @remark This function may be called without opening the library.
///
/// @remark This function is guaranteed to never break backwards compatibility,
/// making it safe to call with the dlopen()/dlsym() pattern.
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_lib_get_version_info
/// @endif
///
/// @pre None.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
/// @since 0.1.0
const NvRmGpuLibVersionInfo *NvRmGpuLibGetVersionInfo(void);

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Extensible attribute structure for #NvRmGpuLibOpen()
///
/// This structure specifies the attributes for opening the nvrm_gpu
/// library. Use #NVRM_GPU_DEFINE_LIB_OPEN_ATTR() to define the attribute struct
/// with defaults.
///
/// Example:
///
///     // define libOpenAttr with default values
///     NVRM_GPU_DEFINE_LIB_OPEN_ATTR(libOpenAttr);
///
///     // open the library
///     NvRmGpuLib *hLib = NvRmGpuLibOpen(&libOpenAttr);
///
/// @sa NvRmGpuLibOpen()
///
typedef struct NvRmGpuLibOpenAttrRec
{
    /// @brief Dummy field for C/C++ ABI compatibility
    uint32_t reserved;

} NvRmGpuLibOpenAttr;

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Definer macro for #NvRmGpuLibOpenAttr.
///
/// This macro defines a variable of type #NvRmGpuLibOpenAttr with
/// the default values.
///
#define NVRM_GPU_DEFINE_LIB_OPEN_ATTR(x) NvRmGpuLibOpenAttr x = { 0U }

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Opens a new instance of the nvrm_gpu library.
///
/// This function creates a new library handle and initializes the library if
/// necessary. After the library is no longer used, the library handle should be
/// closed with NvRmGpuLibClose() to avoid memory leaks.
///
/// @param[in]  attr  Extensible library open attributes, or @NULL for defaults.
///                   Currently unused.
///
/// @return Library handle, or @NULL if the library could not be initialized.
///
/// @remark There can be multiple concurrent instances of the library in the
///         process. Global shared data used by the library is internally
///         reference counted: the first instance will initialize the shared
///         resources; and when the last instance is closed, the shared
///         resources are freed.
///
/// @remark If the library initialization fails, an error message is printed on
///         stderr with an error code for diagnostics.
///
/// **Example:**
///
/// @code
/// // open the library
/// NvRmGpuLib *hLib = NvRmGpuLibOpen(NULL);
///
/// if (hLib != NULL)
/// {
///     NvRmGpuDevice *hDevice = NULL;
///     NvError err;
///
///     err = NvRmGpuDeviceOpen(hLib, NVRM_GPU_DEVICE_INDEX_DEFAULT, NULL, &hDevice);
///     if (err == NvSuccess)
///     {
///         // use the device
///         ...
///
///         // all done, close the device
///         NvRmGpuDeviceClose(hDevice);
///     }
///     else
///     {
///         // deal with the error
///     }
///
///     // all done, close the library
///     NvRmGpuLibClose(hLib);
/// }
/// else
/// {
///     // deal with the error
/// }
/// @endcode
///
/// @sa NvRmGpuLibClose()
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_lib_open
/// @endif
///
/// @pre None.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvRmGpuLib *NvRmGpuLibOpen(const NvRmGpuLibOpenAttr *attr);

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Closes the library and releases all resources.
///
/// @param[in] hLib    Library handle. May be @NULL, in which case this function
///                    is a no-op.
///
/// @return The usual NvError code
/// @retval NvSuccess  The library was closed and all related resources were
///                    freed successfully
/// @retval NvError_*  Unspecified error. The error code is returned for
///                    diagnostic purposes. The library object is closed regardless
///                    but some resources may have failed to close gracefully.
///
/// @remark Every resource attached to the library must be closed before closing
/// the library to avoid leaks and dangling pointers.
///
/// @sa NvRmGpuLibOpen()
/// @sa NvRmGpuDeviceClose()
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_lib_close
/// @endif
///
/// @pre Library is opened successfully and all resources originating from the library
/// are closed.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes, with the following conditions:
///     - No concurrent operations on the Library object ongoing in other threads.
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
NvError NvRmGpuLibClose(NvRmGpuLib *hLib);

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Device attachment state.
///
/// @sa NvRmGpuLibListDevices(), NvRmGpuLibDeviceListEntry
/// @sa NvRmGpuDeviceOpen()
///
typedef enum
{
    /// @brief Device is attached and may be opened with NvRmGpuDeviceOpen()
    NvRmGpuLibDeviceState_Attached,

    /// @brief Device exists, but not enough privileges to access.
    NvRmGpuLibDeviceState_InsufficientPrivileges,

    /// @brief Device state is not known. Prober failed to determine device
    /// state.
    NvRmGpuLibDeviceState_Unknown,

} NvRmGpuLibDeviceState;


/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Device list entry
///
/// @sa NvRmGpuLibListDevices()
///
typedef struct NvRmGpuLibDeviceListEntryRec
{
    /// @brief Internal device index. Used in NvRmGpuDeviceOpen()
    int deviceIndex;

    /// @brief Device attachment state.
    ///
    /// @sa NvRmGpuDeviceState
    NvRmGpuLibDeviceState deviceState;

    /// @brief Informative device name
    ///
    /// This is the 'probe' name of the device. The name is backend-specific. Examples:
    /// - nouveau:/dev/dri/renderD128
    /// - nvgpu:/dev/nvhost-gpu
    /// - nvgpu:/dev/nvgpu-pci/card-0001:15:00.0
    ///
    /// @remark The path is informational only. If probe-time device
    /// identification is required, please file a feature request.
    const char *name;
} NvRmGpuLibDeviceListEntry;

/// @ingroup nvrm_gpu_lib_safety_group
/// @brief Returns the list of probed GPUs
///
/// Returns the list of probed GPUs. The list is valid until the library handle
/// is closed.
///
/// @param[in]  hLib         Library handle
/// @param[out] pNumDevices  Non-@NULL Pointer to receive the number of entries in the list
///
/// @return     Pointer to the list of probed GPUs (C array). The caller must
///             not attempt to free the pointer.
///
/// @remark The first device listed is considered the primary GPU.
///
/// @remark The device index numbers returned are non-negative, unique and in
///   ascending order. Numbering may be discontiguous, and specifically, the
///   index numbers will likely not start at 0.
///
/// @sa NvRmGpuDeviceOpen()
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_lib_list_devices
/// @endif
///
/// @pre Library is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
const NvRmGpuLibDeviceListEntry *NvRmGpuLibListDevices(NvRmGpuLib *hLib, size_t *pNumDevices);



/// @ingroup nvrm_gpu_device_safety_group
/// @brief Pseudo-index for the default (primary) device.
///
/// @remark By default, this is the first GPU enumerated by
/// NvRmGpuLibListDevices(). This can be overridden with environment variable
/// <tt>NVRM_GPU_DEFAULT_DEVICE_INDEX</tt>.
///
/// @sa NvRmGpuDeviceOpen()
///
#define NVRM_GPU_DEVICE_INDEX_DEFAULT (-1)

/// @ingroup nvrm_gpu_device_safety_group
/// @brief Inter-engine synchronization type for GPU jobs
///
/// The usual GPU channels (also known as KMD kickoff channels) support
/// attaching pre-fences and post-fences with job submission. The pre-fence is a
/// synchronization condition that must be met before the job execution can
/// begin. Respectively, the post-fence is a synchronization condition that will
/// be triggered after the job execution has completed. This allows a larger
/// task to be split into parts where GPU and other engines seamlessly process
/// the data in multiple stages. For example:
/// - camera produces a frame
/// - GPU processes the frame after camera has produced it
/// - Display controller displays the frame after GPU has processed it
///
/// Depending on the operating system and HW capabilities, different
/// synchronization object types are available:
///
/// - <b>Tegra HOST 1X syncpoint</b> --- Syncpoint is a hardware register
///   provided by the SoC (generally, 32-bit integer with wrap-around safe
///   semantics). Pre-sync condition waits until the syncpoint value reaches a
///   threshold, and post-sync condition increases the syncpoint value.
///
/// - <b>Android/Linux sync fd</b> --- Synchronization fence backed up by a file
///   (sync_file). The synchronization fence has two stages: untriggered
///   (initial state) and triggered. Pre-sync condition always waits for the
///   fence to become triggered, and post-sync condition triggers the fence.
///
/// @remark This is not to be confused with GPU semaphores. GPU semaphores are
/// usually used to synchronize jobs that are executed within a single device,
/// or between multiple GPUs, or sometimes between the GPU and the CPU. GPU
/// semaphore is simply a memory location with semantics similar to Tegra HOST1X
/// syncpoints. Generally, waiters wait until the value at the memory location
/// reaches a specific threshold, and waiters are released by setting the
/// semaphore to the threshold value or above (but there are other modes, too.)
///
/// @sa https://www.kernel.org/doc/Documentation/sync_file.txt
/// @sa NvRmGpuChannelKickoffPb()
///
typedef enum
{
    /// @brief Default sync type
    ///
    /// @remark Depending on the context, this is platform default,
    /// device-default, or channel-default.
    ///
    /// @sa NvRmGpuDeviceInfo::defaultSyncType
    /// @sa NvRmGpuChannelInfo::syncType
    NvRmGpuSyncType_Default,

    /// @brief Synchronization type is Android/Linux sync fd.
    NvRmGpuSyncType_SyncFd,

    /// @brief Synchronization type is Tegra HOST1X syncpoint.
    NvRmGpuSyncType_Syncpoint,
} NvRmGpuSyncType;


/// @ingroup nvrm_gpu_device_safety_group
/// @brief Extensible attribute structure for #NvRmGpuDeviceOpen()
///
/// @remark Use NVRM_GPU_DEFINE_DEVICE_OPEN_ATTR() to define the attribute
/// variable with defaults.
///
/// @sa NvRmGpuDeviceReadTimeNs()
///
typedef struct NvRmGpuDeviceOpenAttrRec
{
    /// @brief The default sync type for this device.
    ///
    /// @deprecated This field should be left with the default value. Use
    /// NvRmGpuChannelAttr::defaultSyncType. NvRmGpuChannelKickoffPbAttr() also
    /// accepts mixing the kickoff sync types with
    /// NvRmGpuChannelKickoffPbAttr::completionSyncType .
    ///
    /// Default: #NvRmGpuSyncType_Default
    ///
    /// @remark It is a fatal error to request an unsupported sync type.
    ///
    /// @remark This field should be removed.
    NvRmGpuSyncType syncType;

    /// @brief Ignored field
    ///
    /// @deprecated This field is not in use anymore. It used to specify between
    /// sandboxable channels (in Android web browser context) and regular
    /// channels. Sandboxable channels used to require extra resources, but that
    /// is not true anymore and channels are always sandbox-friendly.
    ///
    /// Default: @false
    ///
    /// @remark This field should be removed.
    bool sandboxFriendlyChannels;
} NvRmGpuDeviceOpenAttr;

/// @ingroup nvrm_gpu_device_safety_group
/// @brief Definer macro for #NvRmGpuDeviceOpen().
///
/// This macro defines a variable of type #NvRmGpuDeviceOpenAttr with
/// the default values.
///
/// @sa NvRmGpuDeviceOpen()
///
#define NVRM_GPU_DEFINE_DEVICE_OPEN_ATTR(x) \
    NvRmGpuDeviceOpenAttr x = { NvRmGpuSyncType_Default, false }


/// @ingroup nvrm_gpu_device_safety_group
/// @brief Opens a GPU device.
///
/// @param[in]  hLib         Library handle
/// @param[in]  deviceIndex  Device index (NvRmGpuLibDeviceListEntry::deviceIndex) or
///                          #NVRM_GPU_DEVICE_INDEX_DEFAULT for the default device.
/// @param[in]  attr         Pointer to device open attributes or @NULL for defaults.
/// @param[out] phDevice     Pointer to receive the device handle.
///
/// @return The usual NvError code
/// @retval NvSuccess                Device opened successfully.
/// @retval NvError_BadValue         Bad device index
/// @retval NvError_DeviceNotFound   Device node not found
/// @retval NvError_AccessDenied     Not enough privileges to access the device
/// @retval NvError_*                Unspecified error. Error code returned for diagnostic purposes.
///
/// @remark Only attached GPUs can be opened. See NvRmGpuLibDeviceListEntry::deviceState.
///
/// @remark See #NVRM_GPU_DEVICE_INDEX_DEFAULT for the discussion on default device.
///
/// @remark On QNX, this API call and any subsequent operations with device-derived resources may mandate
/// the user to possess the ability "nvgpu/gpu-access".
///
/// @sa NvRmGpuDeviceGetInfo()
/// @sa NvRmGpuDeviceOpenAttr, NVRM_GPU_DEFINE_DEVICE_OPEN_ATTR()
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_dev_open
/// @endif
///
/// @pre Library is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceOpen(NvRmGpuLib *hLib, int deviceIndex, const NvRmGpuDeviceOpenAttr *attr,
                          NvRmGpuDevice **phDevice);

/// @ingroup nvrm_gpu_device_safety_group
/// @brief Closes the GPU device.
///
/// @param[in]  hDevice  Device handle to close. May be @NULL.
///
/// @return The usual NvError code
/// @retval NvSuccess               Device closed and all related resources released successfully,
///                                 or device handle was @NULL.
/// @retval NvError_*               Unspecified error. Device handle is closed, but some resources
///                                 may be left unreleased. Error code is returned only for diagnostic
///                                 purposes.
///
/// @remark Every resource attached to the device must be closed before closing
/// the device to avoid leaks and dangling pointers.
///
/// @sa NvRmGpuDeviceOpen()
/// @sa NvRmGpuAddressSpaceClose(), NvRmGpuChannelClose(),
/// NvRmGpuCtxSwTraceClose(), NvRmGpuTaskSchedulingGroupClose(),
/// NvRmGpuRegOpsSessionClose(), NvRmGpuDeviceEventSessionClose()
///
/// @if NVRM_GPU_SC_PRIVATE
/// @sa @ref nvrm_gpu_sc_arch_sequences_dev_close
/// @endif
///
/// @pre Device is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes, with the following conditions:
///     - No concurrent operations on the Device object ongoing in other threads.
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: Yes
///
NvError NvRmGpuDeviceClose(NvRmGpuDevice *hDevice);


#if NVOS_IS_LINUX || NVOS_IS_QNX

/// @ingroup nvrm_gpu_device_group
/// @brief Format macro for printf for printing #NvRmGpuClockAsyncReqHandle
///
/// The use is similar to PRIu64 for printing uint64_t.
#define NVRM_GPU_CLOCK_ASYNC_REQ_HANDLE_PRIFMT "d"

/// @ingroup nvrm_gpu_device_group
/// @brief Invalid handle value
///
/// @sa NvRmGpuClockAsyncReqHandle
#define NVRM_GPU_CLOCK_ASYNC_REQ_INVALID_HANDLE (-1)

/// @ingroup nvrm_gpu_device_group
/// @brief OS-specific type of asynchronous clock request handle.
///
/// Asynchronous clock request handle is a waitable handle for clock change
/// requests. This allows one to issue multiple clock change requests
/// concurrently (e.g., GPU clock and memory clock for multiple devices) and
/// then wait for all of them to complete.
///
/// Use #NVRM_GPU_CLOCK_ASYNC_REQ_INVALID_HANDLE to safely initialize a handle.
///
/// @remark This is a file descriptor on QNX and Linux.
typedef int NvRmGpuClockAsyncReqHandle;
#else

/// @ingroup nvrm_gpu_device_group
/// @brief Format macro for printf for printing #NvRmGpuClockAsyncReqHandle
///
/// The use is similar to PRIu64 for printing uint64_t.
#define NVRM_GPU_CLOCK_ASYNC_REQ_HANDLE_PRIFMT "p"

/// @ingroup nvrm_gpu_device_group
/// @brief Invalid handle value
///
/// @sa NvRmGpuClockAsyncReqHandle
#define NVRM_GPU_CLOCK_ASYNC_REQ_INVALID_HANDLE (NULL)

/// @struct NvRmGpuClockAsyncNotImplemented
/// @ingroup nvrm_gpu_device_group
/// @brief OS-specific type of asynchronous clock request handle (unimplemented).
///
/// Marker for unimplemented handle type.
///
///
/// @typedef NvRmGpuClockAsyncReqHandle
/// @ingroup nvrm_gpu_device_group
/// @brief OS-specific type of asynchronous clock request handle.
///
/// Asynchronous clock request handle is a waitable handle for clock change
/// requests. This allows one to issue multiple clock change requests
/// concurrently (e.g., GPU clock and memory clock for multiple devices) and
/// then wait for all of them to complete.
///
/// Use #NVRM_GPU_CLOCK_ASYNC_REQ_INVALID_HANDLE to safely initialize a handle.
///
/// @remark This is void pointer on operating systems that do not support
/// asynchronous clock request handles. HOS does not have the support.
typedef struct NvRmGpuClockAsyncNotImplemented *NvRmGpuClockAsyncReqHandle;
#endif

/// @ingroup nvrm_gpu_device_group
/// @brief Clock domains
///
/// The GPU has different clock domains that can be queried or requested
/// separately. These include the memory clock and the graphics clock.
///
/// @sa NvRmGpuClockGetDomains(), NvRmGpuDeviceInfo::clockDomains
typedef enum
{
    /// @brief Memory clock
    NvRmGpuClockDomain_MCLK = 0,

    /// @brief Main graphics core clock
    NvRmGpuClockDomain_GPCCLK,

    /// @brief Number of clock domains
    NvRmGpuClockDomain_Count
} NvRmGpuClockDomain;

/// @ingroup nvrm_gpu_device_group
/// @brief Request type for clock get.
///
/// @sa NvRmGpuClockGet()
typedef enum NvRmGpuClockType
{
    /// @brief Target clock frequency requested by the user.
    ///
    /// This is the minimum frequency requested by the user. The programmed
    /// frequency may differ.
    NvRmGpuClockType_Target = 1,

    /// @brief Clock frequency programmed to the HW (including PLL constraints).
    ///
    /// @remark This is called the "Actual" clock frequency as this frequency is
    /// the one that is actually programmed.
    NvRmGpuClockType_Actual = 2,

    /// @brief Effective clock as measured from hardware.
    NvRmGpuClockType_Effective = 3
} NvRmGpuClockType;


/// @ingroup nvrm_gpu_device_group
/// @brief Entry for clock get request
///
/// @sa NvRmGpuClockGet()
typedef struct NvRmGpuClockGetEntryRec
{
    /// @brief \b (IN) Domain for the clock request
    ///
    /// @remark This is input parameter. NvRmGpuClockGet() will not modify this
    /// field.
    NvRmGpuClockDomain domain;

    /// @brief \b (IN) Request type
    ///
    /// @remark This is input parameter. NvRmGpuClockGet() will not modify this
    /// field.
    NvRmGpuClockType type;

    /// @brief \b (OUT) Frequency in Hz
    ///
    /// @remark This is output parameter. NvRmGpuClockGet() will modify this
    /// field on #NvSuccess. It may also modify this field on error.
    uint64_t freqHz;

} NvRmGpuClockGetEntry;

/// @ingroup nvrm_gpu_device_group
/// @brief Entry for clock set request
///
/// @sa NvRmGpuClockSet()
typedef struct NvRmGpuClockSetEntryRec
{
    /// @brief Domain for clock request
    NvRmGpuClockDomain domain;

    /// @brief Frequency for clock request
    uint64_t freqHz;
} NvRmGpuClockSetEntry;

/// @ingroup nvrm_gpu_device_group
/// @brief Frequency range for clock domain
///
/// @sa NvRmGpuClockGetDomains()
typedef struct NvRmGpuClockRangeRec
{
    uint64_t minHz;
    uint64_t maxHz;
} NvRmGpuClockRange;

/// @ingroup nvrm_gpu_device_group
/// @brief Clock voltage/frequency point
///
/// @sa NvRmGpuClockGetPoints()
typedef struct NvRmGpuClockPointRec
{
    uint64_t freqHz;
} NvRmGpuClockPoint;

/// @ingroup nvrm_gpu_device_group
/// @brief Clock domain info
///
/// @sa NvRmGpuClockGetDomains()
typedef struct NvRmGpuClockDomainInfoRec
{
    /// @brief Clock domain
    NvRmGpuClockDomain domain;

    /// @brief Frequency range of the clock domain
    NvRmGpuClockRange range;

    /// @brief Maximum number of voltage/frequency points returned by
    /// NvRmGpuClockGetPoints()
    size_t maxVfPoints;

} NvRmGpuClockDomainInfo;

/// @ingroup nvrm_gpu_device_group
/// @brief Returns available GPU clock domains for the device.
///
/// @param[in]  hDevice      Device handle.
/// @param[out] infos        Array of available clock domains. This list is valid
///                          during the life-time of the @a hDevice handle. The returned
///                          pointer must not be freed by the caller.
/// @param[out] pNumDomains  Number of domains in array.
///
/// @return The usual #NvError return code.
/// @retval NvSuccess              Successful request.
/// @retval NvError_NotSupported   Clock controls API not supported by this
///                                device. Capability
///                                NvRmGpuDeviceInfo::hasClockControls for this
///                                device is @false.
/// @retval NvError_*              Unspecified error. Error code returned for
///                                diagnostics.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls. This function can be
///         used to probe the capability. If \a NvError_NotSupported is
///         returned, then NvRmGpuDeviceInfo::hasClockControls is @false.
///
/// @remark There may be more actual clock domains in the GPU HW than returned
/// by this function. This function returns the domains that can be queried or
/// requested.
///
/// @sa NvRmGpuClockGetPoints()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockGetDomains(NvRmGpuDevice *hDevice,
                               const NvRmGpuClockDomainInfo **infos,
                               size_t *pNumDomains);

/// @ingroup nvrm_gpu_device_group
/// @brief Retrieves voltage/frequency (VF) points for a given clock domain.
/// For information about VF points, see @ref NvRmGpuClockAsyncReqHandle.
///
/// Each clock domain has VF points, defined as frequencies for which voltage is
/// optimal. In general, the clock arbiter will try to program frequencies which
/// correspond to VF points.
///
/// @param[in]  hDevice     Device handle.
/// @param[in]  domain      Clock domain to query.
/// @param[out] pClkPoints  Pointer to receive the array of optimal VF
///                         points. The allocated array must contain space for
///                         at least NvRmGpuClockDomainInfo::maxVfPoints (as
///                         retrieved by NvRmGpuClockGetDomains()).
/// @param[out] pNumPoints  Number of VF points. May vary depending on thermal
///                         conditions, and will be at most
///                         NvRmGpuClockDomainInfo::maxVfPoints.
///
/// @return The usual NvError return code.
/// @retval NvSuccess              Successful request.
/// @retval NvError_NotSupported   Clock controls API not supported by this device.
/// @retval NvError_*              Unspecified error. Error code returned for diagnostics.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls
///
/// @sa NvRmGpuClockGet(), NvRmGpuClockSet()
/// @sa NvRmGpuClockGetDomains()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockGetPoints(NvRmGpuDevice *hDevice,
                              NvRmGpuClockDomain domain,
                              NvRmGpuClockPoint *pClkPoints,
                              size_t *pNumPoints);

/// @ingroup nvrm_gpu_device_group
/// @brief Requests minimum clocks for one or more clock domains.
///
/// This function allows the caller to request minimum GPU clocks for one or
/// more GPU clock domains. Both synchronous and asynchronous requests are
/// supported.
///
/// In the asynchronous case, a waitable clock handle is returned. When setting
/// clocks on multiple devices, the asynchronous mode allows one to perform all
/// requests concurrently, and then wait until the clocks have been set. Clock
/// stabilization can take a while.
///
/// When another request is made for a clock domain in the same #NvRmGpuDevice
/// instance, the previous requests are overridden.
///
/// The actually programmed clock frequency may not be exactly the requested
/// frequency. Generally, the clock arbiter chooses the optimal
/// voltage-frequency point that is at least as high as the highest frequency
/// requested by any GPU user. However, depending on the thermal and other
/// conditions, the actual GPU frequency may be also lower. Use
/// NvRmGpuClockGet() to query the programmed and the effective
/// frequencies. #NvRmGpuDeviceEventSession can be also used to monitor changes
/// in the GPU frequencies.
///
///
/// @param[in]  hDevice        Device handle.
///
/// @param[in] pClkSetEntries  Array of request entries. Each entry requests
///                            target frequency for one clock domain. If a clock
///                            domain appears multiple times in one call (not
///                            recommended), then only the last entry will be
///                            taken into account.
///
/// @param[in]  numEntries     Number of entries in the @a pClkSetEntries array.
///
/// @param[out] phReq          Pointer to asynchronous request handle or @NULL
///                            for a synchronous request
///                            - If @NULL, the request is synchronous and
///                              function returns only after all clocks are
///                              programmed.
///                            - If non-@NULL, the request is asynchronous and a
///                              waitable request completion handle is returned
///                              on success. Use NvRmGpuClockWaitAsyncReq() to wait
///                              using the handle. The request handle must be
///                              closed by the caller using
///                              NvRmGpuClockCloseAsyncReq().
///
/// @return The usual NvError code. In case of error, the asynchronous request
///         handle is not returned.
///
/// @retval NvSuccess            The request was successfully made. In the
///                              synchronous case, the wait was also
///                              successful. In the asynchronous case, the
///                              request handle is returned.
///
/// @retval NvError_NotSupported Clock controls API not supported by this device.
///
/// @retval NvError_Busy         A temporary blocking condition when submitting
///                              the asynchronous request. The user should try
///                              again.
///
/// @retval NvError_*            Unspecified error. The error code is returned
///                              for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls
///
/// @remark The synchronous wait case is equivalent with performing the
///         asynchronous wait request followed immediately by calls to
///         NvRmGpuClockWaitAsyncReq() and NvRmGpuClockCloseAsyncReq().
///
/// @remark A subsequent clock request to a domain supersedes the previous
///         request (per #NvRmGpuDevice instance)
///
/// @remark The lifespan of GPU clock requests is tied to the #NvRmGpuDevice
///         instance. All requests by the user are canceled when the
///         #NvRmGpuDevice handle is closed.
///
/// @remark The clock requests of all #NvRmGpuDevice instances are coalesced.
///         Generally, given that thermal and power limits are not exceeded, the
///         actual clock frequency will be at least the greatest requested. The
///         exact selection algorithm depends on the global clock management
///         driver policy.  The selection algorithm is run by the "clock
///         arbiter" within the KMD component.
///
/// @remark Actual frequency might differ depending on global policies, requests
///         from other NvRmGpuDevice instances, or thermal conditions.
///
/// @remark If specified target frequency is not a VF point, clock arbiter will
///         generally try to program the clocks with first VF point that is
///         greater than or equal to specified target frequency (assuming a
///         single application)
///
/// @sa NvRmGpuClockGet()
/// @sa NvRmGpuClockGetDomains()
/// @sa NvRmGpuClockWaitAsyncReq()
/// @sa NvRmGpuClockCloseAsyncReq()
/// @sa NvRmGpuDeviceEventSessionOpen()
/// @sa NvRmGpuClockWaitAnyEvent()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Async/Sync
///     - Async if @a phReq was provided; and
///     - Sync if @a phReq was @NULL.
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockSet(NvRmGpuDevice *hDevice,
                        const NvRmGpuClockSetEntry *pClkSetEntries,
                        size_t numEntries,
                        NvRmGpuClockAsyncReqHandle *phReq);

/// @ingroup nvrm_gpu_device_group
/// @brief Waits for the completion of one or more asynchronous clock requests.
///
/// @param[in] hDevice     Device handle
/// @param[in] phReqs      Array of request handles.
/// @param[in] numEntries  Number of entries in the request array.
/// @param[in] timeoutMs   Wait timeout in milliseconds. Use as follows:
///                        - #NV_WAIT_INFINITE: No timeout, indefinite wait
///                        - `>=0`: Timeout specified. The function returns when the
///                          request completes or when the timeout expires,
///                          whichever comes first.
///                        - `0`: Peek. The function returns immediately.
///
/// @return The usual #NvError return value
/// @retval NvSuccess        All requests have completed.
/// @retval NvError_Timeout  Timeout was reached before all
///                          requests were completed.
/// @retval NvError_BadValue One or more of the handles is invalid.
/// @retval NvError_*        Unspecified error. The error code is returned
///                          for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls
///
/// @remark When this functions returns an error, the state of the request
///         handles in <tt>phReq</tt> is left undefined. The handles should be
///         closed without any further operation on them.
///
/// @sa NvRmGpuClockSet()
///
/// @pre Asynchronous requests were created successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockWaitAsyncReq(NvRmGpuDevice *hDevice,
                                 const NvRmGpuClockAsyncReqHandle *phReqs,
                                 size_t numEntries,
                                 uint32_t timeoutMs);


/// @ingroup nvrm_gpu_device_group
/// @brief Closes an asynchronous clock request handle.
///
/// Frees all resources related to an asynchronous request created with
/// NvRmGpuClockSet(). It is not mandatory to wait for request completion before
/// closing the handle.
///
/// @param[in] hDevice Device handle
/// @param[in] hReq    Asynchronous request handle to close. May be @NULL.
///
/// @return The usual  #NvError code
/// @retval NvSuccess  Handle closed successfully, or @NULL handle was provided.
/// @retval NvError_*  Unspecified error. The error code is returned
///                    for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls
///
/// @sa NvRmGpuClockSet()
/// @sa NvRmGpuClockWaitAsyncReq()
///
/// @pre Asynchronous requests were created successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockCloseAsyncReq(NvRmGpuDevice *hDevice,
                                  NvRmGpuClockAsyncReqHandle hReq);


/// @ingroup nvrm_gpu_device_group
/// @brief This function is not implemented and it should be deleted.
///
/// @pre N/A.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockWaitAnyEvent(NvRmGpuDevice *hDevice,
                                 uint32_t timeoutMs);


/// @ingroup nvrm_gpu_device_group
/// @brief Request one or more clock domain frequency state.
///
/// This function can be used to request control frequencies on clock
/// domains. This function accepts one or more requests. The request is targeted
/// for:
/// - clock domain (e.g., GPC core clock, memory clock). See
///   #NvRmGpuClockDomain.
/// - control information type: application target frequency, programmed
///   frequency, measured frequency. See #NvRmGpuClockType.
///
/// @param[in]     hDevice        Device handle
///
/// @param[in,out] pClkGetEntries Array of clock request entries. For each
///                               entry, the clock domain and clock types must
///                               be set. Upon a successful call the associated
///                               frequency will be returned on a successful
///                               call. It is allowed to mix several clocks
///                               domains and clock request types in the same
///                               request.
///
/// @param[in]     numEntries     Number of request entries.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_NotSupported  Clock controls API not supported.
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasClockControls
///
/// @sa NvRmGpuClockSet()
/// @sa NvRmGpuClockGetDomains()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuClockGet(NvRmGpuDevice *hDevice,
                        NvRmGpuClockGetEntry *pClkGetEntries,
                        size_t numEntries);

/// @ingroup nvrm_gpu_device_group
/// @brief Voltage sensors
typedef enum
{
    /// @brief Core GPU voltage
    NvRmGpuDeviceVoltage_Core = 1,

    /// @brief SRAM voltage
    NvRmGpuDeviceVoltage_SRAM,

    /// @brief Bus voltage
    NvRmGpuDeviceVoltage_Bus
} NvRmGpuDeviceVoltage;

/// @ingroup nvrm_gpu_device_group
/// @brief Returns the list of available voltage sensors for the device.
///
/// @param[in]  hDevice      Device handle.
/// @param[out] pSensors     Non-@NULL pointer to receive a pointer to the array of
///                          available sensors. The returned pointer may be
///                          @NULL if no sensors are available. The
///                          returned pointer is valid for the life-time of \a
///                          hDevice and it must not be freed by the caller.
/// @param[out] pNumSensors  Non-@NULL pointer to receive the number of available sensors.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @sa NvRmGpuDeviceGetVoltage()
///
/// @pre Device was opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceListVoltageSensors(NvRmGpuDevice *hDevice,
                                        const NvRmGpuDeviceVoltage **pSensors,
                                        size_t *pNumSensors);


/// @ingroup nvrm_gpu_device_group
/// @brief Retrieves the voltage sensor reading.
///
/// @param[in]  hDevice            Device handle
/// @param[in]  which              The voltage sensor to query
/// @param[out] pVoltageMicroVolt  non-@NULL pointer to receive the voltage in microvolts.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_NotSupported  Not supported on this device
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasDeviceSensorInfo
///
/// @remark See NvRmGpuDeviceListVoltageSensors() for the available voltage
/// sensors for the device.
///
/// @sa NvRmGpuDeviceListVoltageSensors()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceGetVoltage(NvRmGpuDevice *hDevice,
                                NvRmGpuDeviceVoltage which,
                                uint64_t *pVoltageMicroVolt);

/// @ingroup nvrm_gpu_device_group
/// @brief Electric current sensors
typedef enum
{
    /// @brief Bus current
    NvRmGpuDeviceCurrent_Bus = 1,
} NvRmGpuDeviceCurrent;


/// @ingroup nvrm_gpu_device_group
/// @brief Returns the list of available electric current sensors for the
/// device.
///
/// @param[in]  hDevice      Device handle.
/// @param[out] pSensors     Non-@NULL pointer to receive a pointer to the array of
///                          available sensors. The returned pointer may be
///                          @NULL if no sensors are available. The
///                          returned pointer is valid for the life-time of \a
///                          hDevice and it must not be freed by the caller.
/// @param[out] pNumSensors  Non-@NULL pointer to receive the number of available sensors.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @sa NvRmGpuDeviceGetCurrent()
///
/// @pre Device was opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceListCurrentSensors(NvRmGpuDevice *hDevice,
                                        const NvRmGpuDeviceCurrent **pSensors,
                                        size_t *pNumSensors);

/// @ingroup nvrm_gpu_device_group
/// @brief Retrieves the electric current reading.
///
/// @param[in]  hDevice              Device handle.
/// @param[in]  which                The current sensor to query.
/// @param[out] pCurrentMicroAmpere  Pointer to receive the current in microamperes.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasDeviceSensorInfo
///
/// @remark See NvRmGpuDeviceListCurrentSensors() for the available electric
/// current sensors for the device.
///
/// @sa NvRmGpuDeviceListCurrentSensors()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceGetCurrent(NvRmGpuDevice *hDevice,
                                NvRmGpuDeviceCurrent which,
                                uint64_t *pCurrentMicroAmpere);

/// @ingroup nvrm_gpu_device_group
/// @brief Electric power sensors
typedef enum
{
    /// @brief Power consumed at the regulator
    NvRmGpuDevicePower_Bus = 1
} NvRmGpuDevicePower;

/// @ingroup nvrm_gpu_device_group
/// @brief Returns the list of available power sensors for the device.
///
/// @param[in]  hDevice      Device handle.
/// @param[out] pSensors     Non-@NULL pointer to receive a pointer to the array of
///                          available sensors. The returned pointer may be
///                          @NULL if no sensors are available. The
///                          returned pointer is valid for the life-time of \a
///                          hDevice and it must not be freed by the caller.
/// @param[out] pNumSensors  Non-@NULL pointer to receive the number of available sensors.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @sa NvRmGpuDeviceGetPower()
///
/// @pre Device is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceListPowerSensors(NvRmGpuDevice *hDevice,
                                      const NvRmGpuDevicePower **pSensors,
                                      size_t *pNumSensors);

/// @ingroup nvrm_gpu_device_group
/// @brief Retrieves the power sensor reading.
///
/// @param[in]  hDevice          Device handle.
/// @param[in]  which            The power sensor to query.
/// @param[out] pPowerMicroWatt  Power in microwatts.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_NotSupported  Not supported on this device
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark See NvRmGpuDeviceListPowerSensors() for the available electric
/// power sensors for the device.
///
/// @remark Requires NvRmGpuDeviceInfo::hasDeviceSensorInfo
///
/// @sa NvRmGpuDeviceListPowerSensors()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceGetPower(NvRmGpuDevice *hDevice,
                              NvRmGpuDevicePower which,
                              uint64_t *pPowerMicroWatt);


/// @ingroup nvrm_gpu_device_group
/// @brief Temperature sensors
typedef enum
{
    /// @brief The internal GPU temperature sensor
    NvRmGpuDeviceTemperature_InternalSensor = 1
} NvRmGpuDeviceTemperature;

/// @ingroup nvrm_gpu_device_group
/// @brief Returns the list of available temperature sensors for the device.
///
/// @param[in]  hDevice      Device handle.
/// @param[out] pSensors     Non-@NULL pointer to receive a pointer to the array of
///                          available sensors. The returned pointer may be
///                          @NULL if no sensors are available. The
///                          returned pointer is valid for the life-time of \a
///                          hDevice and it must not be freed by the caller.
/// @param[out] pNumSensors  Non-@NULL pointer to receive the number of available sensors.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_NotSupported  Not supported on this device
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @sa NvRmGpuDeviceGetTemperature()
///
/// @pre Device is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceListTemperatureSensors(NvRmGpuDevice *hDevice,
                                            const NvRmGpuDeviceTemperature **pSensors,
                                            size_t *pNumSensors);

/// @ingroup nvrm_gpu_device_group
/// @brief Retrieves the temperature sensor reading.
///
/// @param[in]  hDevice                   Device handle.
/// @param[in]  which                     Temperature sensor to query.
/// @param[out] pTemperatureMilliCelsius  Pointer to receive the temperature reading in millidegrees Celsius.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_NotSupported  Not supported on this device
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasDeviceSensorInfo
///
/// @remark See NvRmGpuDeviceListPowerSensors() for the available temperature
/// sensors for the device.
///
/// @sa NvRmGpuDeviceListTemperatureSensors()
/// @sa NvRmGpuDeviceThermalAlertSetLimit()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceGetTemperature(NvRmGpuDevice *hDevice,
                                    NvRmGpuDeviceTemperature which,
                                    int32_t *pTemperatureMilliCelsius);

/// @ingroup nvrm_gpu_device_group
/// @brief Sets the thermal alert limit.
///
/// @param hDevice         Device handle.
/// @param temperature_mC  Thermal temperature alert threshold in millidegrees Celsius.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful operation
/// @retval NvError_NotSupported  Operation not supported for the device.
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDeviceInfo::hasDeviceThermalAlert
///
/// @sa NvRmGpuDeviceGetTemperature()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceThermalAlertSetLimit(NvRmGpuDevice *hDevice,
                                          int32_t temperature_mC);


/// @ingroup nvrm_gpu_device_group
/// GPU device event type
///
/// @sa NvRmGpuDeviceEventSessionOpen()
typedef enum
{
    // @brief Frequency change event
    ///
    /// Voltage/frequency update occurred for a clock domain. This can be
    /// because of:
    /// - an nvrm_gpu user issued a target frequency request; or
    /// - because of a change in thermal or power conditions.
    ///
    /// This is an informational event.
    NvRmGpuDeviceEventId_VfUpdate = 0,

    /// @brief A clock domain frequency is below target.
    ///
    /// This indicates that the GPU may be operating at lower than expected
    /// performance.
    NvRmGpuDeviceEventId_AlarmTargetVfNotPossible,

    /// @brief A clock domain frequency is below local target frequency
    /// requested by a session.
    ///
    /// This indicates that the GPU may be operating at lower than expected
    /// performance.
    NvRmGpuDeviceEventId_AlarmLocalTargetVfNotPossible,

    /// @brief The clock arbiter has failed.
    ///
    /// This is a system failure. Frequency change requests may not be honored
    /// anymore.
    NvRmGpuDeviceEventId_AlarmClockArbiterFailed,

    /// @brief VF table update failed.
    ///
    /// VF table update is typically related to operating condition
    /// change. Something went wrong and VF tables could not be updated.
    ///
    /// This is a system failure.
    NvRmGpuDeviceEventId_AlarmVfTableUpdateFailed,

    /// @brief Temperature above threshold.
    ///
    /// The GPU temperature is above threshold. Measures may have to be taken to
    /// prevent thermal throttling. For instance, target frequencies may need to
    /// be lowered.
    ///
    /// @sa NvRmGpuDeviceThermalAlertSetLimit()
    NvRmGpuDeviceEventId_AlarmThermalAboveThreshold,

    /// @brief Power above threshold.
    ///
    /// The GPU power drain is above threshold. Measures may have to be taken to
    /// remedy the condition. For instance, target frequencies may need to be
    /// lowered.
    NvRmGpuDeviceEventId_AlarmPowerAboveThreshold,

    /// @brief Device lost.
    ///
    /// The GPU device is lost. This may be due to number of reasons, such as
    /// bus failure, power failure, hardware failure, GPU hang/reboot, firmware
    /// failure, or a programming failure due to KMD.
    ///
    /// This is a system failure. The nvrm_gpu user should close all resources.
    ///
    /// @remark NvRmGpuDeviceClose()
    NvRmGpuDeviceEventId_AlarmGpuLost,

    /// @brief Number of events.
    NvRmGpuDeviceEventId_Count

} NvRmGpuDeviceEventId;

/// @ingroup nvrm_gpu_device_group
/// @brief Extensible attribute structure for
/// #NvRmGpuDeviceEventSessionOpen().
///
/// @remark Use NVRM_GPU_DEFINE_DEVICE_EVENT_SESSION_ATTR() to define the
/// attribute variable with defaults.
///
/// @sa NvRmGpuDeviceEventSessionOpen()
/// @sa NVRM_GPU_DEFINE_DEVICE_EVENT_SESSION_ATTR()
typedef struct NvRmGpuDeviceEventSessionOpenAttrRec
{
    /// @brief List of events to listen.
    ///
    /// @remark Use NvRmGpuDeviceEventSessionOpenAttrSetAllEvents() to listen to
    /// all events.
    const NvRmGpuDeviceEventId *filterList;

    /// @brief Number of entries in the event list.
    size_t filterListSize;

} NvRmGpuDeviceEventSessionOpenAttr;

/// @ingroup nvrm_gpu_device_group
/// @brief Definer macro for #NvRmGpuDeviceEventSessionOpenAttr.
///
/// This macro defines a variable of type #NvRmGpuDeviceEventSessionOpenAttr
/// with the default values.
///
/// @sa NvRmGpuDeviceEventSessionOpen()
#define NVRM_GPU_DEFINE_DEVICE_EVENT_SESSION_ATTR(x)            \
    NvRmGpuDeviceEventSessionOpenAttr x = { NULL, 0 }

/// @ingroup nvrm_gpu_device_group
/// @brief Assigns device events attribute structure with a list of all events
/// to listen to.
///
/// @param[out]  attr Non-@NULL pointer to the device events attribute struct.
///
/// @sa #NvRmGpuDeviceEventSessionOpenAttr
///
/// @pre None.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
static inline void NvRmGpuDeviceEventSessionOpenAttrSetAllEvents(NvRmGpuDeviceEventSessionOpenAttr *attr)
{
    static const NvRmGpuDeviceEventId allEvents[] =
    {
        NvRmGpuDeviceEventId_VfUpdate,
        NvRmGpuDeviceEventId_AlarmTargetVfNotPossible,
        NvRmGpuDeviceEventId_AlarmLocalTargetVfNotPossible,
        NvRmGpuDeviceEventId_AlarmClockArbiterFailed,
        NvRmGpuDeviceEventId_AlarmVfTableUpdateFailed,
        NvRmGpuDeviceEventId_AlarmThermalAboveThreshold,
        NvRmGpuDeviceEventId_AlarmPowerAboveThreshold,
        NvRmGpuDeviceEventId_AlarmGpuLost
    };
    attr->filterList = allEvents;
    attr->filterListSize = NV_ARRAY_SIZE(allEvents);
}

/// @ingroup nvrm_gpu_device_group
/// @brief Opens a session to monitor device events.
///
/// @param[in]  hDevice    Device handle.
/// @param[in]  attr       Event session attributes. The attribute structure
///                        contains the device event filter list.
/// @param[out] phSession  Pointer to receive the event session handle on success.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Device event session created successfully.
/// @retval NvError_NotSupported  This device does not support device events.
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @remark Requires NvRmGpuDevice::hasDeviceEvents
///
/// @sa NvRmGpuDeviceEventSessionClose()
///
/// @pre Device is opened successfully and supports the feature. See remarks
/// for the condition.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceEventSessionOpen(NvRmGpuDevice *hDevice,
                                      const NvRmGpuDeviceEventSessionOpenAttr *attr,
                                      NvRmGpuDeviceEventSession **phSession);

/// @ingroup nvrm_gpu_device_group
/// @brief GPU device event.
///
/// Data type for a single GPU device event. This contains the timestamp and the
/// event type.
///
/// @sa NvRmGpuDeviceEventSessionRead()
/// @sa NvRmGpuDeviceReadTimeNs()
typedef struct NvRmGpuDeviceEventInfoRec
{
    /// @brief Event type
    NvRmGpuDeviceEventId eventId;

    /// @brief GPU time (in nanoseconds)
    ///
    /// This is the unscaled GPU PTIMER timestamp at the occurrence of the
    /// event.
    ///
    /// @remark Certain integrated Tegra GPUs require GPU timestamp
    /// scaling. These GPUs are T210 and T214. See the discussion in
    /// NvRmGpuDeviceInfo::ptimerScaleNumerator for further details.
    ///
    /// @sa NvRmGpuDeviceReadTimeNs()
    uint64_t timeNs;

} NvRmGpuDeviceEventInfo;

/// @ingroup nvrm_gpu_device_group
/// @brief Read next device event
///
/// @param[in]  hSession    Event session handle
/// @param[out] pEventInfo  Pointer to receive the event on success
/// @param[in]  timeoutMs   Timeout value in milliseconds. Special values:
///                         - `0`: non-blocking peek
///                         - #NV_WAIT_INFINITE: wait indefinitely for the next event
///
/// @return NvSuccess indicates that one event occurred, and detailed
///         information has been updated in @a pEventInfo.
///         NvError_Timeout indicates that timeout was reached.
///
/// @remark When an event occurs while there is a previous pending event of the
/// same type, the events are merged. In this case, only one event is reported.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Successful request
/// @retval NvError_Timeout       Timeout occurred before an event was available
/// @retval NvError_*             Unspecified error. The error code is returned
///                               for diagnostic purposes.
///
/// @sa NvRmGpuDeviceEventSessionOpen()
/// @sa NvRmGpuDeviceEventInfo
///
/// @pre Device event session is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceEventSessionRead(NvRmGpuDeviceEventSession *hSession,
                                      NvRmGpuDeviceEventInfo *pEventInfo,
                                      uint32_t timeoutMs);

/// @ingroup nvrm_gpu_device_group
/// @brief Closes the device event session.
///
/// @param[in] hSession  Device event session handle to close. May be @NULL.
///
/// @return The usual #NvError return code
/// @retval NvSuccess             Event session closed successfully or \a hSession was @NULL.
/// @retval NvError_*             Unspecified error while closing the
///                               session. The session is closed, regardless.
///                               The error code is returned for diagnostic
///                               purposes.
///
/// @remark Regardless of possible errors in deinitialization, the object will
/// be closed.
///
/// @pre Device event session is opened successfully.
///
/// @usage
/// - Allowed context for the API call
///   - Thread-safe: Yes
///   - Interrupt handler: No
///   - Signal handler: No
///   - Re-entrant: Yes
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: No
///   - De-Init: No
///
NvError NvRmGpuDeviceEventSessionClose(NvRmGpuDeviceEventSession *hSession);

#if defined(NVRM_GPU_BUILD_VARIANT)
} // namespace nvrm_gpu
#endif

#if defined(__cplusplus)
}
#endif

#if !defined(NV_SDK_BUILD)
#include "nvrm_gpu_priv.h"
#endif

#endif
