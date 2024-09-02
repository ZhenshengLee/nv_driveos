/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// @brief <b> NVIDIA Media Interface: 2D Processing Control </b>
///
/// @b Description: This file contains the #image_2d_api "Image 2D
/// Processing API."
///

#ifndef NVMEDIA_2D_H
#define NVMEDIA_2D_H

#include "nvmedia_core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_nvmedia_2d_api 2D Processing
///
/// The 2D Processing API encompasses all NvMedia 2D image processing
/// related functionality.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Major version number of NvMedia 2D header.
///
/// This defines the major version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #NvMedia2DGetVersion() function.
///
/// @sa NvMedia2DGetVersion()
///
#define NVMEDIA_2D_VERSION_MAJOR   8

/// @brief Minor version number of NvMedia 2D header.
///
/// This defines the minor version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #NvMedia2DGetVersion() function.
///
/// @sa NvMedia2DGetVersion()
///
#define NVMEDIA_2D_VERSION_MINOR   0

/// @brief Patch version number of NvMedia 2D header.
///
/// This defines the patch version of the API defined in this header.
///
/// @sa NvMedia2DGetVersion()
///
#define NVMEDIA_2D_VERSION_PATCH   0

/// @brief 2D filter mode.
///
/// This enum describes the filter modes that are supported by NvMedia 2D.
///
/// @sa NvMedia2DSetFilterBuffer()
///
typedef enum
{
    /// @brief Filtering is disabled.
    ///
    /// This mode results in nearest-neighbor filtering being used. This is the
    /// default mode.
    ///
    NVMEDIA_2D_FILTER_OFF = 0x1,

    /// @brief Low quality filtering.
    ///
    /// This mode results in bilinear filtering being used.
    ///
    NVMEDIA_2D_FILTER_LOW,

    /// @brief Medium quality filtering.
    ///
    /// This mode results in 5-tap filtering being used. By default, the filter
    /// used is a Lanczos filter. The coefficients can be overriden with
    /// NvMedia2DSetFilterBuffer().
    ///
    NVMEDIA_2D_FILTER_MEDIUM,

    /// @brief Highest quality filtering.
    ///
    /// This mode results in 10-tap filtering being used. By default, the filter
    /// used is a Lanczos filter. The coefficients can be overriden with
    /// NvMedia2DSetFilterBuffer().
    ///
    NVMEDIA_2D_FILTER_HIGH

} NvMedia2DFilter;

/// @brief 2D rotation/transform.
///
/// This enum describes the 2D transform to apply during a blit operation.
///
/// Transformations are used to rotate and mirror the source surface of
/// a blit operation. The destination rectangle is not affected by any
/// transformation settings.
///
/// @ref NvMediaTransform identifies the transformations that can be applied
/// as a combination of rotation and mirroring.
///
/// Specifically, given a hypothetical 2x2 image, applying these operations
/// would yield the following results:
/// @code
///
///       INPUT                      OUTPUT
///
///      +---+---+                  +---+---+
///      | 0 | 1 |     IDENTITY     | 0 | 1 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 2 | 3 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |    ROTATE_90     | 1 | 3 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 0 | 2 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |    ROTATE_180    | 3 | 2 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 1 | 0 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |    ROTATE_270    | 2 | 0 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 3 | 1 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |  FLIP_HORIZONTAL | 1 | 0 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 3 | 2 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |  INVTRANSPOSE    | 3 | 1 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 2 | 0 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |  FLIP_VERTICAL   | 2 | 3 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 0 | 1 |
///      +---+---+                  +---+---+
///
///      +---+---+                  +---+---+
///      | 0 | 1 |    TRANSPOSE     | 0 | 2 |
///      +---+---+ ---------------> +---+---+
///      | 2 | 3 |                  | 1 | 3 |
///      +---+---+                  +---+---+
///
/// @endcode
///
typedef enum
{
    /// @brief No transform.
    NVMEDIA_2D_TRANSFORM_NONE = 0x0,

    /// @brief Rotate 90 degrees anti-clockwise.
    NVMEDIA_2D_TRANSFORM_ROTATE_90,

    /// @brief Rotate 180 degrees.
    NVMEDIA_2D_TRANSFORM_ROTATE_180,

    /// @brief Rotate 270 degrees anti-clockwise.
    NVMEDIA_2D_TRANSFORM_ROTATE_270,

    /// @brief Flip horizontally (mirror along Y axis).
    NVMEDIA_2D_TRANSFORM_FLIP_HORIZONTAL,

    /// @brief Apply inverse transpose (mirror along diagonal axis from
    /// top-right to bottom-left).
    NVMEDIA_2D_TRANSFORM_INV_TRANSPOSE,

    /// @brief Flip vertically (mirror along X axis).
    NVMEDIA_2D_TRANSFORM_FLIP_VERTICAL,

    /// @brief Apply transpose (mirror along diagonal axis from top-left to
    /// bottom-right).
    NVMEDIA_2D_TRANSFORM_TRANSPOSE

} NvMedia2DTransform;

/// @brief Blending to use when compositing surfaces.
///
typedef enum
{
    /// @brief Disable blending.
    ///
    /// The color and alpha channels of the source surfaces are copied to the
    /// destination surface as is. The constant alpha factor is ignored. This is
    /// the default mode.
    ///
    NVMEDIA_2D_BLEND_MODE_DISABLED,

    /// @brief Ignore the alpha channel values.
    ///
    /// Ignore the alpha channel values in the pixel data and only use the
    /// constant alpha factor that was passed when configuring blending.
    ///
    /// The color channels are blended with:
    ///
    /// `result = src * constantAlpha + dst * (1 - constantAlpha)`
    ///
    NVMEDIA_2D_BLEND_MODE_CONSTANT_ALPHA,

    /// @brief Treat color and alpha channels independently.
    ///
    /// Use the product of the alpha channel values in the pixel data and the
    /// constant alpha factor to interpolate between color channel values of the
    /// blended surfaces.
    ///
    /// The color channels are blended with:
    ///
    /// `result = src * srcAlpha * constantAlpha + dst * (1 - srcAlpha * constantAlpha)`
    ///
    NVMEDIA_2D_BLEND_MODE_STRAIGHT_ALPHA,

    /// @brief Treat color channels as having the alpha channel premultiplied.
    ///
    /// As NVMEDIA_2D_BLEND_MODE_STRAIGHT_ALPHA, but assume color channel values
    /// have been premultiplied by the alpha channel value.
    ///
    /// The color channels are blended with:
    ///
    /// `result = src * constantAlpha + dst * (1 - srcAlpha * constantAlpha)`
    ///
    NVMEDIA_2D_BLEND_MODE_PREMULTIPLIED_ALPHA

} NvMedia2DBlendMode;

/// @brief Attributes structure for #NvMedia2DCreate().
///
/// This type holds the attributes to control the behaviour of the NvMedia2D
/// context. These attributes take effect during the call to #NvMedia2DCreate().
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMedia2DCreate()
///
typedef struct
{
    /// @brief Number of compose parameters objects to allocate.
    ///
    /// Must be in range [1, 16].
    ///
    /// @sa NvMedia2DComposeParameters
    /// @sa NvMedia2DGetComposeParameters()
    ///
    uint32_t numComposeParameters;

    /// @brief Maximum number of buffers that can be registered.
    ///
    /// Must be in range [1, 1024].
    ///
    /// @sa NvMedia2DRegisterNvSciBufObj()
    ///
    uint32_t maxRegisteredBuffers;

    /// @brief Maximum number of sync objects that can be registered.
    ///
    /// Must be in range [1, 256].
    ///
    /// @sa NvMedia2DRegisterNvSciSyncObj()
    ///
    uint32_t maxRegisteredSyncs;

    /// @brief Maximum number of filter buffers that can be created.
    ///
    /// Must be in range [0, 16].
    ///
    /// @sa NvMedia2DFilterBuffer
    /// @sa NvMedia2DCreateFilterBuffer()
    uint32_t maxFilterBuffers;

    /// @brief Internal use only.
    uint32_t flags;

} NvMedia2DAttributes;

/// @brief Stores configuration for the NvMedia2DCompose() operation.
///
/// This object stores the information needed to configure the 2D operation that
/// is executed inside the NvMedia2DCompose() function.
///
/// The underlying object cannot be instantiated directly by the client. Instead
/// use the #NvMedia2DGetComposeParameters() function to retrieve a handle to
/// an available instance.
///
/// Value 0 is never a valid handle value, and can be used to initialize an
/// NvMedia2DComposeParameters handle to a known value.
///
/// @sa NvMedia2DGetComposeParameters()
/// @sa NvMedia2DInsertPreNvSciSyncFence()
/// @sa NvMedia2DSetNvSciSyncObjforEOF()
/// @sa NvMedia2DSetSrcGeometry()
/// @sa NvMedia2DSetSrcFilter()
/// @sa NvMedia2DSetSrcNvSciBufObj()
/// @sa NvMedia2DSetDstNvSciBufObj()
/// @sa NvMedia2DCompose()
///
typedef uint32_t NvMedia2DComposeParameters;

/// @brief Stores a filter buffer which coefficients can be configured.
///
/// The underlying object cannot be instantiated directly by the client. Instead
/// use the #NvMedia2DCreateFilterBuffer() function to create and retrieve
/// a handle to an available instance with resources allocated.
/// Use #NvMedia2DDestroyFilterBuffer() function to deallocate resources
/// allocated by #NvMedia2DCreateFilterBuffer() and release an instance.
///
/// Value 0 is never a valid handle value, and can be used to initialize an
/// NvMedia2DFilterBuffer handle to a known value.
///
/// @sa NvMedia2DCreateFilterBuffer
/// @sa NvMedia2DDestroyFilterBuffer
///
typedef uint32_t NvMedia2DFilterBuffer;

/// @brief Stores information returned from NvMedia2DCompose().
///
/// This type holds the information about an operation that has been submitted
/// to NvMedia2DCompose() for execution.
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMedia2DCompose()
///
typedef struct
{
    /// @brief ID number for operation that was submitted to NvMedia2DCompose().
    ///
    /// The number will wrap once the uint64_t range has been exceeded. A value
    /// of 0 indicates that no operation was submitted.
    ///
    uint64_t operationId;

} NvMedia2DComposeResult;

/// @brief Coefficients values structure for 5-tap custom filter.
///
/// This type holds the list of coefficients values for a 1D polyphasic
/// filter with 5 taps and 32 phases.
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMedia2DComputeFilterCoefficients5Tap()
///
typedef struct
{
    /// @brief Array of coefficients values, ordered by phase, then by tap.
    ///
    /// The coefficient values are interpreted as 10-bit signed binary fixed
    /// point numbers in two's complement format, with (from least to most
    /// significant bits):
    /// - 8 fraction bits
    /// - 1 integer bit
    /// - 1 sign bit
    ///
    /// Other bits of the values are discarded.
    ///
    int16_t coeffs[32][5];
} NvMedia2DFilterCoefficients5Tap;

/// @brief Coefficients values structure for 10-tap custom filter.
///
/// This type holds the list of coefficients values for a 1D polyphasic
/// filter with 10 taps and 32 phases.
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMedia2DComputeFilterCoefficients10Tap()
///
typedef struct
{
    /// @brief Array of coefficients values, ordered by phase, then by tap.
    ///
    /// The coefficient values are interpreted as 10-bit signed binary fixed
    /// point numbers in two's complement format, with (from least to most
    /// significant bits):
    /// - 8 fraction bits
    /// - 1 integer bit
    /// - 1 sign bit
    ///
    /// Other bits of the values are discarded.
    ///
    int16_t coeffs[32][10];
} NvMedia2DFilterCoefficients10Tap;

/// @brief Returns the version number of the NvMedia 2D library.
///
/// This function returns the major and minor version number of the
/// NvMedia 2D library. The client must pass an #NvMediaVersion struct
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
/// @sa NVMEDIA_2D_VERSION_MAJOR
/// @sa NVMEDIA_2D_VERSION_MINOR
/// @sa NVMEDIA_2D_VERSION_PATCH
/// @sa NvMediaVersion
///
NvMediaStatus
NvMedia2DGetVersion(NvMediaVersion * const version);

/// @brief NvMedia2D Context.
///
/// This type represents a context for the NvMedia2D library.
/// This context is an opaque data type that encapsulates the state needed to
/// service the NvMedia2D API calls.
///
/// @sa NvMedia2DCreate()
///
typedef struct NvMedia2D NvMedia2D;

/// @brief Creates a new NvMedia2D context.
///
/// This function creates a new instance of an NvMedia2D context, and returns a
/// pointer to that context. Ownership of this context is passed to the caller.
/// When no longer in use, the caller must destroy the context using the
/// #NvMedia2DDestroy() function.
///
/// Default attributes (when not specified by caller):
/// - numComposeParameters: 1
/// - maxRegisteredBuffers: 64
/// - maxRegisteredSyncs: 16
/// - maxFilterBuffers: 0
/// - flags: 0.
///
/// @param[out] handle  Pointer to receive the handle to the new NvMedia2D
///                     context.
/// @param[in]  attr    Pointer to NvMedia2DAttributes struct, or NULL for
///                     default attributes.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context created successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL, or @a attr has bad
///                                       attribute values.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  NvMedia 2D is not supported on this
///                                       hardware platform.
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
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMedia2D
/// @sa NvMedia2DDestroy()
///
NvMediaStatus
NvMedia2DCreate(NvMedia2D ** const handle,
                NvMedia2DAttributes const * const attr);

/// @brief Destroys the NvMedia2D context.
///
/// This function destroys the specified NvMedia2D context.
///
/// Before calling this function, the caller must ensure:
/// - There are no NvSciSync or NvSyncBuf objects still registered against the
///   NvMedia2D context.
/// - All previous 2D operations submitted using NvMedia2DCompose() have
//    completed.
///
/// @param[in] handle  Pointer to the NvMedia2D context.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context destroyed successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        There are still some NvSciSync or
///                                       NvSciBuf objects registered against
///                                       the NvMedia2D context.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to destroy the context. The
///                                       context is in state where the only
///                                       valid operation is to attempt to
///                                       destroy it again.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
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
/// @sa NvMedia2D
/// @sa NvMedia2DCreate()
///
NvMediaStatus
NvMedia2DDestroy(NvMedia2D const * const handle);

/// @brief Returns an #NvMedia2DComposeParameters instance.
///
/// This functions returns a handle to an NvMedia2DComposeParameters object.
/// The object will be initialized and ready to use. The caller takes ownership
/// of this handle. Ownership will be passed back to the NvMedia2D context when
/// it is subsequently used in the #NvMedia2DCompose() operation.
///
/// The handle returned in @a params is tied to the specific NvMedia2D context
/// instance passed in @a handle and cannot be used with other context
/// instances.
///
/// The object will be initialized with these default values:
/// - source rectangle: set to the dimensions of the source surface
/// - destination rectangle: set to the dimensions of the destination surface
/// - filter: NVMEDIA_2D_FILTER_OFF
/// - transform: NVMEDIA_2D_TRANSFORM_NONE.
///
/// @param[in]  handle  Pointer to the NvMedia2D context.
/// @param[out] params  Pointer to an #NvMedia2DComposeParameters, which will
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
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
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
///
NvMediaStatus
NvMedia2DGetComposeParameters(NvMedia2D const * const handle,
                              NvMedia2DComposeParameters * const params);

/// @brief Performs a 2D compose operation.
///
/// A compose operation transfers pixels from a set of source surfaces to a
/// destination surface, applying a variety of transformations to the pixel
/// values on the way. The surfaces can have different pixel formats. NvMedia 2D
/// does the necessary conversions between the formats.
///
/// @note For a YUV surface type with 16-bit depth, only scale and crop are
/// supported. Pixel format conversion, transformations or multi-surface
/// composition is not supported.
///
/// If the dimensions of the source rectangle do not match the dimensions of the
/// destination rectangle, the operation scales the pixels to fit the
/// destination rectangle. The filtering mode for scale defaults to
/// #NVMEDIA_2D_FILTER_OFF. Additional filtering modes can be used by setting
/// the corresponding parameter using #NvMedia2DSetSrcFilter().
///
/// This example performs a straight pixel copy between surfaces of the same
/// dimensions (but not necessarily the same bit depth or color format):
///
/// @code
///     NvMedia2DComposeParameters params;
///     NvMedia2DGetComposeParameters(handle, &params);
///     NvMedia2DSetSrcNvSciBufObj(handle, params, 0, srcBuf);
///     NvMedia2DSetDstNvSciBufObj(handle, params, dstBuf);
///     NvMedia2DCompose(handle, params, NULL);
/// @endcode
///
/// @anchor NvMedia2DComposeParameterRestrictions
///
/// Restrictions on dimensions for input and output surfaces:
/// - For 16-bit YUV/YUVX/Y formats:
///   - Width must be within the range: [16, 8192]
///   - Height must be within the range: [16, 8192]
/// - For any other supported format:
///   - Width must be within the range: [16, 16384]
///   - Height must be within the range: [16, 16384]
///
/// Additional restrictions for chroma sub-sampled YUV formats:
/// - 444:
///   - Width: no restriction
///   - Height: no restriction
/// - 422:
///   - Width:
///     - must be a multiple of 2
///     - must be at least 32 for 16-bit YUV formats
///   - Height: no restriction
/// - 422R:
///   - Width: no restriction
///   - Height:
///     - must be a multiple of 2
///     - must be at least 32 for 16-bit YUV formats
/// - 420:
///   - Width:
///     - must be a multiple of 2
///     - must be at least 32 for 16-bit YUV formats
///   - Height:
///     - must be a multiple of 2
///     - must be at least 32 for 16-bit YUV formats
///
/// Restrictions on the source rectangle:
/// - Must be within the bounds of the source surface.
/// - Width and height must be greater than zero.
///
/// Restrictions on the destination rectangle:
/// - Must be within the bounds of the destination surface.
/// - Width and height must be greater than zero.
/// - For 16-bit formats, top-left corner must be zero.
/// - For any YUV format with chroma subsampling different than 444, all corners
///   must be aligned to a a multiple of 2.
///
/// @anchor NvMedia2DComposeConcurrencyRestrictions
///
/// Restrictions on concurrency:
/// - There can be a maximum of 16 operations submitted through the same
///   NvMedia2D handle pending simultaneously.
///
/// If any of the restrictions are violated, this function will fail with an
/// error code.
///
/// Performance considerations:
/// - Using block linear memory layout for surfaces generally provides better
///   performance than pitch linear layout.
/// - Using (semi-)planar YUV formats generally provides better performance than
///   interleaved (Y8U8Y8V8 etc.) formats.
/// - The hardware accelerator utilized by NvMedia 2D accesses the destination
///   surface in 64x16 (width x height) pixel tiles. Using destination surface
///   size and destination rectangle dimensions that are exact multiples of the
///   tile size most optimally aligns with this hardware access pattern.
///
/// The result info returned in @a result is tied to the specific NvMedia2D
/// context instance passed in @a handle and cannot be used with other context
/// instances.
///
/// @param[in]  handle  Pointer to the NvMedia2D context.
/// @param[in]  params  An NvMedia2DComposeParameters handle.
/// @param[out] result  Pointer to NvMedia2DComposeResult struct that will be
///                     populated with result info. May be NULL.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Operation submitted successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - some of the compose parameters
///                                         configured through @a params have
///                                         invalid values (see @ref
///                                         NvMedia2DComposeParameterRestrictions
///                                         "restrictions on parameters").
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  Requested operation is not supported
///                                       by current platform (see @ref
///                                       NvMedia2DComposeParameterRestrictions
///                                       "restrictions on parameters").
/// @retval NVMEDIA_STATUS_TIMED_OUT      No space available in the command
///                                       buffer for this operation, because
///                                       previous operations are still
///                                       pending (see @ref
///                                       NvMedia2DComposeConcurrencyRestrictions
///                                       "restrictions on concurrency"). The
///                                       caller should wait for the least
///                                       recently submitted operation to
///                                       complete and then try again.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to perform the compose
///                                       operation. This error indicates the
///                                       system is potentially in an
///                                       unrecoverable state.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMedia2D handle
///   - Re-entrant: No
///   - Async/Sync: Async
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DComposeParameters
/// @sa NvMedia2DGetComposeParameters()
/// @sa NvMedia2DInsertPreNvSciSyncFence()
/// @sa NvMedia2DSetNvSciSyncObjforEOF()
/// @sa NvMedia2DSetSrcGeometry()
/// @sa NvMedia2DSetSrcFilter()
/// @sa NvMedia2DSetSrcBlendMode()
/// @sa NvMedia2DSetSrcNvSciBufObj()
/// @sa NvMedia2DSetDstNvSciBufObj()
/// @sa NvMedia2DGetEOFNvSciSyncFence()
///
NvMediaStatus
NvMedia2DCompose(NvMedia2D const * const handle,
                 NvMedia2DComposeParameters const params,
                 NvMedia2DComposeResult * const result);

/// @brief Sets the geometry for a source layer.
///
/// @param[in] handle     Pointer to the NvMedia2D context.
/// @param[in] params     An NvMedia2DComposeParameters handle.
/// @param[in] index      Index of source layer to configure. Must be in range
///                       [0, 15].
/// @param[in] dstRect    Pointer to an NvMediaRect that contains the
///                       destination rectangle, or NULL for default rectangle.
/// @param[in] srcRect    Pointer to an NvMediaRect that contains the source
///                       rectangle, or NULL for default rectangle.
/// @param[in] transform  An #NvMedia2DTransform to apply the content region.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a index is out of range
///                                         - @a srcRect is invalid
///                                         - @a dstRect is invalid
///                                         - @a transform is invalid.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED    Requested operation is not supported
///                                         by current platform.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
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
///
NvMediaStatus
NvMedia2DSetSrcGeometry(NvMedia2D const * const handle,
                        NvMedia2DComposeParameters const params,
                        uint32_t const index,
                        NvMediaRect const * const srcRect,
                        NvMediaRect const * const dstRect,
                        NvMedia2DTransform const transform);

/// @brief Sets the filter mode for a source layer.
///
/// @param[in] handle  Pointer to the NvMedia2D context.
/// @param[in] params  An NvMedia2DComposeParameters handle.
/// @param[in] index   Index of source layer to configure. Must be in range
///                    [0, 15].
/// @param[in] filter  An #NvMedia2DFilter to use when reading the the layer's
///                    source surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a index is out of range
///                                         - @a filter is invalid.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
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
///
NvMediaStatus
NvMedia2DSetSrcFilter(NvMedia2D const * const handle,
                      NvMedia2DComposeParameters const params,
                      uint32_t const index,
                      NvMedia2DFilter const filter);

/// @brief Sets the blend mode for a source layer.
///
/// @param[in] handle         Pointer to the NvMedia2D context.
/// @param[in] params         An NvMedia2DComposeParameters handle.
/// @param[in] index          Index of source layer to configure. Must be in
///                           range [0, 15].
/// @param[in] blendMode      Blend mode to set.
/// @param[in] constantAlpha  Constant alpha factor to use in blending.
///                           Must be in range [0, 1].
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a index is out of range
///                                         - @a blendMode is invalid
///                                         - @a constantAlpha is out of range
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
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
/// @sa NvMedia2DBlendMode
/// @sa NvMedia2DCompose()
///
NvMediaStatus
NvMedia2DSetSrcBlendMode(NvMedia2D const * const handle,
                         NvMedia2DComposeParameters const params,
                         uint32_t const index,
                         NvMedia2DBlendMode const blendMode,
                         float const constantAlpha);

/// @brief Creates and returns an #NvMedia2DFilterBuffer instance.
///
/// This functions returns a handle to an NvMedia2DFilterBuffer object. The
/// filter buffer can be used to provide custom 5-tap and 10-tap filter
/// coefficients for a compose operation.
///
/// The handle returned in @a filterBuffer is tied to the specific NvMedia2D
/// context instance passed in @a handle and cannot be used with other context
/// instances.
///
/// The buffer instance must be destroyed with #NvMedia2DDestroyFilterBuffer()
/// during the De-Init stage.
///
/// @param[in]  handle        Pointer to the NvMedia2D context.
/// @param[out] filterBuffer  Pointer to an #NvMedia2DFilterBuffer, which will
///                           be populated with the handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Buffer created successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           One of the parameters has an
///                                                invalid value, either:
///                                                - @a handle is NULL
///                                                - @a filterBuffer is NULL.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of buffers has
///                                                been created.
/// @retval NVMEDIA_STATUS_INVALID_STATE           The function was called in
///                                                incorrect system state.
/// @retval NVMEDIA_STATUS_OUT_OF_MEMORY           Failed to allocate memory for
///                                                the buffer.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to create the
///                                                buffer.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
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
/// @sa NvMedia2DAttributes
/// @sa NvMedia2DFilterBuffer
/// @sa NvMedia2DDestroyFilterBuffer()
///
NvMediaStatus
NvMedia2DCreateFilterBuffer(NvMedia2D const * const handle,
                            NvMedia2DFilterBuffer * const filterBuffer);

/// @brief Destroys an #NvMedia2DFilterBuffer instance.
///
/// This functions destroys an NvMedia2DFilterBuffer object.
///
/// @param[in] handle        Pointer to the NvMedia2D context.
/// @param[in] filterBuffer  An #NvMedia2DFilterBuffer handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Buffer are destroyed successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a filterBuffer is invalid.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        The buffer is still being used by a
///                                       pending operation.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to destroy the buffer. The
///                                       buffer is in state where the only
///                                       valid operation is to attempt to
///                                       destroy it again.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a filterBuffer must be valid NvMedia2DFilterBuffer handle created with
///   NvMedia2DCreateFilterBuffer().
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
/// @sa NvMedia2DFilterBuffer
/// @sa NvMedia2DCreateFilterBuffer()
///
NvMediaStatus
NvMedia2DDestroyFilterBuffer(NvMedia2D const * const handle,
                             NvMedia2DFilterBuffer const filterBuffer);

/// @brief Sets the filter buffer for an #NvMedia2DComposeParameters instance.
///
/// This function updates the NvMedia2DComposeParameters instance to indicate
/// that the specified filter buffer object shall be used to provide the custom
/// filter coefficients for the compose operation.
///
/// After this function returns successfully, there are a few additional
/// limitations on the compose operation:
/// - Only the first 5 source layers can be used (indexes 0 to 4).
/// - If a source layer's NvMedia2DFilter is set to NVMEDIA_2D_FILTER_MEDIUM or
///   NVMEDIA_2D_FILTER_HIGH, the custom 5-tap or 10-tap coefficients,
///   respectively, for such layer shall have been properly set in the
///   NvMedia2DFilterBuffer.
///
/// Due to the filter buffer object being read-only from the compose operation
/// perspective, there is no limitation for the same filter buffer object to
/// be set for multiple NvMedia2DComposeParameters instances.
///
/// @param[in] handle        Pointer to the NvMedia2D context.
/// @param[in] params        An #NvMedia2DComposeParameters handle.
/// @param[in] filterBuffer  An #NvMedia2DFilterBuffer handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Filter buffer was set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a params is invalid.
///                                       - @a filterBuffer is invalid.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a params must be valid NvMedia2DComposeParameters handle created with
///   NvMedia2DGetComposeParameters().
/// - @a filterBuffer must be valid NvMedia2DFilterBuffer handle created with
///   NvMedia2DCreateFilterBuffer().
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
/// @sa NvMedia2DFilterBuffer
/// @sa NvMedia2DCreateFilterBuffer()
/// @sa NvMedia2DFilter
/// @sa NvMedia2DSetSrcFilter()
/// @sa NvMedia2DComputeFilterCoefficients5Tap()
/// @sa NvMedia2DComputeFilterCoefficients10Tap()
///
NvMediaStatus
NvMedia2DSetFilterBuffer(NvMedia2D const * const handle,
                         NvMedia2DComposeParameters const params,
                         NvMedia2DFilterBuffer const filterBuffer);

/// @brief Computes the 5-tap filter coefficients for an #NvMedia2DFilterBuffer.
///
/// This function computes the filter coefficients values for a specific source
/// layer based on the contents of four #NvMedia2DFilterCoefficients5Tap
/// structures.
///
/// There is no restriction on multiple parameters pointing to the same
/// #NvMedia2DFilterCoefficients5Tap structure
///
/// @param[in] handle           Pointer to the NvMedia2D context.
/// @param[in] filterBuffer     An #NvMedia2DFilterBuffer handle.
/// @param[in] index            Index of source layer to configure. Must be in
///                             range [0, 4].
/// @param[in] lumaX,lumaY      Pointers to #NvMedia2DFilterCoefficients5Tap.
///                             These configure to the luma component for YUV
///                             formats, or all the components for RGB formats.
///                             There is one pointer for the horizontal
///                             direction, and one pointer for the vertical
///                             direction.
/// @param[in] chromaX,chromaY  Pointers to #NvMedia2DFilterCoefficients5Tap.
///                             These configure the chroma component for YUV
///                             formats. There is one pointer for the
///                             horizontal direction, and one pointer for the
///                             vertical direction.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Filter buffer was updated successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a filterBuffer is invalid.
///                                       - @a index is out of range.
///                                       - @a lumaX is NULL.
///                                       - @a lumaY is NULL.
///                                       - @a chromaX is NULL.
///                                       - @a chromaY is NULL.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a filterBuffer must be valid NvMedia2DFilterBuffer handle created with
///   NvMedia2DCreateFilterBuffer().
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
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DFilterBuffer
/// @sa NvMedia2DFilterCoefficients5Tap
/// @sa NvMedia2DCreateFilterBuffer()
///
NvMediaStatus
NvMedia2DComputeFilterCoefficients5Tap(NvMedia2D const * const handle,
                                       NvMedia2DFilterBuffer const filterBuffer,
                                       uint32_t const index,
                                       NvMedia2DFilterCoefficients5Tap const * const lumaX,
                                       NvMedia2DFilterCoefficients5Tap const * const lumaY,
                                       NvMedia2DFilterCoefficients5Tap const * const chromaX,
                                       NvMedia2DFilterCoefficients5Tap const * const chromaY);

/// @brief Computes the 10-tap filter coefficients for an #NvMedia2DFilterBuffer.
///
/// This function computes the filter coefficients values for a specific source
/// layer based on the contents of four #NvMedia2DFilterCoefficients10Tap
/// structures.
///
/// There is no restriction on multiple parameters pointing to the same
/// #NvMedia2DFilterCoefficients10Tap structure
///
/// @param[in] handle           Pointer to the NvMedia2D context.
/// @param[in] filterBuffer     An #NvMedia2DFilterBuffer handle.
/// @param[in] index            Index of source layer to configure. Must be in
///                             range [0, 4].
/// @param[in] lumaX,lumaY      Pointers to #NvMedia2DFilterCoefficients10Tap.
///                             These configure to the luma component for YUV
///                             formats, or all the components for RGB formats.
///                             There is one pointer for the horizontal
///                             direction, and one pointer for the vertical
///                             direction.
/// @param[in] chromaX,chromaY  Pointers to #NvMedia2DFilterCoefficients10Tap.
///                             These configure the chroma component for YUV
///                             formats. There is one pointer for the
///                             horizontal direction, and one pointer for the
///                             vertical direction.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Filter buffer was updated successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL.
///                                       - @a filterBuffer is invalid.
///                                       - @a index is out of range.
///                                       - @a lumaX is NULL.
///                                       - @a lumaY is NULL.
///                                       - @a chromaX is NULL.
///                                       - @a chromaY is NULL.
///
/// @pre
/// - @a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
/// - @a filterBuffer must be valid NvMedia2DFilterBuffer handle created with
///   NvMedia2DCreateFilterBuffer().
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
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMedia2DFilterBuffer
/// @sa NvMedia2DFilterCoefficients10Tap
/// @sa NvMedia2DCreateFilterBuffer()
///
NvMediaStatus
NvMedia2DComputeFilterCoefficients10Tap(NvMedia2D const * const handle,
                                        NvMedia2DFilterBuffer const filterBuffer,
                                        uint32_t const index,
                                        NvMedia2DFilterCoefficients10Tap const * const lumaX,
                                        NvMedia2DFilterCoefficients10Tap const * const lumaY,
                                        NvMedia2DFilterCoefficients10Tap const * const chromaX,
                                        NvMedia2DFilterCoefficients10Tap const * const chromaY);

///
/// @}
///

//
// Version History
//
// Version 1.1 February 1, 2016
// - Initial release
//
// Version 1.2 May 11, 2016
// - Added #NvMedia2DCheckVersion API
//
// Version 1.3 May 5, 2017
// - Removed compositing, blending and alpha related defines and structures
//
// Version 2.0 May 11, 2017
// - Deprecated NvMedia2DBlit API
// - Deprecated NvMedia2DCheckVersion API
// - Deprecated NvMedia2DColorStandard, NvMedia2DColorRange and
//   NvMedia2DColorMatrix types
// - Added #NvMedia2DGetVersion API
//
// Version 2.1 May 17, 2017
// - Moved transformation to nvmedia_common.h
// - Renamed NvMedia2DTransform to #NvMediaTransform
//
// Version 2.2 September 4, 2018
// - Added deprecated warning message for #NvMedia2DCopyPlane,
//   NvMedia2DWeave
// - Added APIs #NvMedia2DCopyPlaneNew, #NvMedia2DWeaveNew
//
// Version 3.0 October 30, 2018
// - Deprecated #NvMedia2DCopyPlane API
// - Deprecated #NvMedia2DWeave API
//
// Version 3.1 December 11, 2018
// - Fixed MISRA-C Rule 21.1 and 21.2 Violations
//
// Version 3.2 January 21, 2019
// - Moved #NvMediaTransform from nvmedia_common.h to this header
//
// Version 3.3 Feb 21, 2019
// - Changed #NvMedia2D type from void to struct
//
// Version 3.4 March 5, 2019
// - Fixed MISRA-C Rule 8.13 Violations
//
// Version 3.5 March 14, 2019
// - Removing NvMedia2DBlitFlags enum definition
// - updated #NvMedia2DBlitParametersOut structure definition
//
// Version 3.6 March 18, 2019
// - Added APIs #NvMedia2DImageRegister, #NvMedia2DImageUnRegister
//
// Version 3.7 March 22, 2019
// - Unnecessary header include nvmedia_common.h has been removed
//
// Version 3.8 May 18, 2020
// - Changes related to MISRA-C Rule 8.13 Violations fixes.
//
// Version 3.9 Nov 12, 2020
// - Improved comments and documentation
// - Introduce NvMedia2DDestroyEx, which returns an error unlike NvMedia2DDestroy
// - NvMedia2DDestroy is marked as deprecated
// - NVMEDIA_STATUS_UNDEFINED_STATE is returned
//   instead of NVMEDIA_STATUS_BAD_PARAMETER if error happens
//   after submit is started
//
// Version 3.10 January 25, 2021
// - Remove NvMedia2DWeaveNew API.
//
// Version 4.0 September 23, 2021
// - Remove NvMedia2DCopyPlaneNew API.
// - Remove NvMedia2DBlitEx API.
// - Remove NvMedia2DImageRegister API.
// - Remove NvMedia2DImageUnregister API.
// - Remove NvMedia2DDestroyEx API.
// - Remove NvMedia2DNvSciSyncGetVersion API.
// - Remove NVMEDIA_2D_NVSCISYNC_VERSION_MAJOR token.
// - Remove NVMEDIA_2D_NVSCISYNC_VERSION_MINOR token.
// - Change prototype for NvMedia2DCreate API.
// - Change prototype for NvMedia2DDestroy API.
// - Change prototype for NvMedia2DSetNvSciSyncObjforEOF API.
// - Change prototype for NvMedia2DInsertPreNvSciSyncFence API.
// - Change prototype for NvMedia2DGetEOFNvSciSyncFence API.
// - Rename NvMedia2DStretchFilter to NvMedia2DFilter.
// - Rename NvMediaTransform to NvMedia2DTransform.
// - Add NVMEDIA_2D_VERSION_PATCH token.
// - Add NvMedia2DCompose API.
// - Add NvMedia2DFillNvSciBufAttrList API.
// - Add NvMedia2DRegisterNvSciBufObj API.
// - Add NvMedia2DUnregisterNvSciBufObj API.
// - Add NvMedia2DSetSrcNvSciBufObj API.
// - Add NvMedia2DSetDstNvSciBufObj API.
// - Set default filter mode to NVMEDIA_2D_FILTER_OFF.
//
// Version 4.1 November 15, 2021
// - Add NvMedia2DSetSrcBlendMode API.
//
// Version 4.2 November 29, 2021
// - Add refcounting to NvMedia2DRegisterNvSciBufObj/UnregisterNvSciBufObj API.
//
// Version 4.3 March 8, 2022
// - Add NvMedia2DCreateFilterBuffer API.
// - Add NvMedia2DDestroyFilterBuffer API.
// - Add NvMedia2DSetFilterBuffer API.
// - Add NvMedia2DComputeFilterCoefficients5Tap API.
// - Add NvMedia2DComputeFilterCoefficients10Tap API.
//
// Version 5.0.0 March 28, 2022
// - Add support for NvSciSync task statuses
// - Max pre-fence count changed from 32 to 16
//
// Version 6.0.0 June 3, 2022
// - Change default for maxRegisteredBuffers attribute from 256 to 64
// - Forbid registering same buffer multiple times
// - Error codes changed for multiple APIs
//
// Version 7.0.0 July 8, 2022
// - New error NVMEDIA_STATUS_INVALID_STATE added for multiple APIs
//
// Version 7.0.1 August 25, 2022
// - Allow NULL context handle in NvMedia2DFillNvSciBufAttrList
//
// Version 7.0.2 September 2, 2022
// - Always treat compose parameters and filter buffer handle value 0 as invalid
//
// Version 8.0.0 October 17, 2023
// - Update the logic to compute the filter buffer coefficients values
//

#ifdef __cplusplus
}
#endif

#endif // NVMEDIA_2D_H
