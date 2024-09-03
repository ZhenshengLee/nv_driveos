/*
 * Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/// @file
/// @brief <b> NVIDIA Media Interface: Lens Distortion Correction
///        and Temporal Noise Reduction </b>
///
/// @b Description: This file contains the #image_ldc_api "Image LDC API".
///

#ifndef NVMEDIA_LDC_H
#define NVMEDIA_LDC_H

#include "nvmedia_core.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @defgroup 6x_nvmedia_ldc_api Lens Distortion Correction
///
/// The LDC API encompasses all NvMedia LDC/TNR image processing
/// related functionality.
///
/// @ingroup 6x_nvmedia_top
/// @{
///

/// @brief Major version number of NvMedia LDC header.
///
/// This defines the major version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #NvMediaLdcGetVersion() function.
///
/// @sa NvMediaLdcGetVersion()
///
#define NVMEDIA_LDC_VERSION_MAJOR 8

/// @brief Minor version number of NvMedia LDC header.
///
/// This defines the minor version of the API defined in this header. This
/// is intended to be used when validating the version number returned by the
/// #NvMediaLdcGetVersion() function.
///
/// @sa NvMediaLdcGetVersion()
///
#define NVMEDIA_LDC_VERSION_MINOR 1

/// @brief Patch version number of NvMedia LDC header.
///
/// This defines the patch version of the API defined in this header. This
/// might be used to validate the version number returned by the
/// #NvMediaLdcGetVersion() function in case when the client needs
/// a revision with a particular patch included.
///
/// @sa NvMediaLdcGetVersion()
///
#define NVMEDIA_LDC_VERSION_PATCH 0

/// @brief NvMediaLdc context.
///
/// This type represents a context for the NvMediaLdc library.
/// This context is an opaque data type that encapsulates the state needed to
/// service the NvMediaLdc API calls.
///
/// @sa NvMediaLdcCreate()
///
typedef struct NvMediaLdc NvMediaLdc;

/// @brief Stores configuration for the NvMediaLdcProcess() operation.
///
/// This object stores the information needed to configure the LDC operation
/// that is executed inside the NvMediaLdcProcess() function.
///
/// The underlying object cannot be instantiated directly by the client. Instead
/// use the #NvMediaLdcCreateParameters() function to create and retrieve
/// a handle to an available instance with resources allocated
/// in respect to used #NvMediaLdcParametersAttributes.
/// Use #NvMediaLdcDestroyParameters() function to deallocate resources
/// allocated by #NvMediaLdcCreateParameters() and release an instance.
///
/// Value 0 is never a valid handle value, and can be used to initialize an
/// NvMediaLdcParameters handle to a known value.
///
/// @sa NvMediaLdcCreateParameters()
/// @sa NvMediaLdcDestroyParameters()
/// @sa NvMediaLdcSetFilter()
/// @sa NvMediaLdcSetGeometry()
/// @sa NvMediaLdcSetIptParameters()
/// @sa NvMediaLdcSetWarpMapParameters()
/// @sa NvMediaLdcSetMaskMapParameters()
/// @sa NvMediaLdcSetTnrParameters()
/// @sa NvMediaLdcProcess()
///
typedef uint32_t NvMediaLdcParameters;

/// @brief Attributes structure for #NvMediaLdcCreate().
///
/// This type holds the attributes to control the behaviour of the NvMediaLdc
/// context. These attributes take effect during the call
/// to #NvMediaLdcCreate().
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMediaLdcCreate()
///
typedef struct
{
    /// @brief Number of parameters objects to allocate.
    ///        It is the maximum number of NvMediaLdcParameters objects
    ///        that the client can create.
    ///
    /// @sa NvMediaLdcParameters
    /// @sa NvMediaLdcCreateParameters()
    ///
    uint32_t maxParameters;

    /// @brief Maximum number of buffers that can be registered.
    ///
    /// @sa NvMediaLdcRegisterNvSciBufObj()
    ///
    uint32_t maxRegisteredBuffers;

    /// @brief Maximum number of sync objects that can be registered.
    ///
    /// @sa NvMediaLdcRegisterNvSciSyncObj()
    ///
    uint32_t maxRegisteredSyncs;

    /// @brief Internal use only.
    uint32_t flags;
} NvMediaLdcAttributes;

/// @brief Attributes that specify resources needed
///        for #NvMediaLdcParameters instance.
///
/// @sa #NvMediaLdcCreateParameters()
///
typedef struct
{
     /// @brief Maximum allowed number of control points horizontally
     ///        in a warp map. This controls the amount of memory allocated
     ///        for warp map. Use 0 if warp map is not used
     ///        to disable any memory allocation for it.
    uint32_t maxWarpMapWidth;

     /// @brief Maximum allowed number of control points vertically
     ///        in a warp map. This controls the amount of memory allocated
     ///        for warp map. Use 0 if warp map is not used
     ///        to disable any memory allocation for it.
    uint32_t maxWarpMapHeight;

    /// @brief Whether to allocate memory for data structures required
    ///        for TNR processing.
    bool enableTnr;

    /// @brief Whether to allocate memory for Mask Map.
    bool enableMaskMap;

    /// @brief Maximum allowed destination rectangle width.
    ///        These control the amount of memory allocated for
    ///        - TNR,
    ///        - Mask Map.
    ///        Not used if none of these features are enabled.
    uint32_t maxDstWidth;

    /// @brief Maximum allowed destination rectangle height.
    ///        These control the amount of memory allocated for
    ///        - TNR,
    ///        - Mask Map.
    ///        Not used if none of these features are enabled.
    uint32_t maxDstHeight;
} NvMediaLdcParametersAttributes;

/// @brief Stores information returned from NvMediaLdcProcess().
///
/// This type holds the information about an operation that has been submitted
/// to NvMediaLdcProcess() for execution.
///
/// This struct itself can be considered a POD type, so it does not have any
/// functions to create/destroy it.
///
/// @sa NvMediaLdcProcess()
///
typedef struct
{
    /// @brief ID number for operation that was submitted
    ///        to NvMediaLdcProcess().
    ///
    /// The number will wrap once the uint64_t range has been exceeded. A value
    /// of 0 indicates that no operation was submitted.
    ///
    uint64_t operationId;
} NvMediaLdcResult;

/// @brief VIC filter mode.
///
/// This enum describes the filter modes that are supported by NvMedia LDC.
///
typedef enum
{
    /// @brief Filtering is disabled.
    NVMEDIA_LDC_FILTER_OFF,

    /// @brief Low quality filtering.
    NVMEDIA_LDC_FILTER_LOW,

    /// @brief Medium quality filtering.
    NVMEDIA_LDC_FILTER_MEDIUM
} NvMediaLdcFilter;

/// @brief Maximum number of horizontal regions.
#define NVMEDIA_LDC_MAX_REGIONS_X 4u

/// @brief Maximum number of vertical regions.
#define NVMEDIA_LDC_MAX_REGIONS_Y 4u

/// @brief Holds the NvMedia LDC region configuration.
///
/// This structure defines the layout of the control points
/// in the destination image.
///
/// The control points are used as the basis for geometric transformation from
/// source image to destination image. The remaining points are transformed
/// based on the interpolation. Thus the density of the control points controls
/// the quality of the geometric transformation.
///
/// This is an example of defining regions in the image:
/// @code
///
///           (dstRect->x1 - dstRect->x0)
///        /                              \'
///       /                                \'
///      /                                  \'
///
///  regionWidth[0]             regionWidth[numRegionsX -1]
///      /     \                     /      \'
///     |-------|                   |--------|
///     --------------------------------------                                      \'
///     |******** ******** ...      ******** |--                                     \'
///     |* +  + * *      * ...      *      * |   \                                    \'
///     |* +  + * *      *          *      * | regionHeight[0]                         \'
///     |* +  + * *      *          *      * |   /                                      \'
///     |******** ********          ******** |--                                         \'
///     |..                         ..       |                                            \'
///     |..                         ..       |
///     |..                         ..       |                           (dstRect->y1 - dstRect->y0)
///     |                                    |
///     |                                    |                                            /
///     |******** ********...       ******** |--                                         /
///     |*      * *      *...       *      * |  \                                       /
///     |*      * *      *          *      * | regionHeight[numRegionsY -1]            /
///     |*      * *      *          *      * |  /                                     /
///     |******** ********          ******** |--                                     /
///     --------------------------------------
///
/// @endcode
///
/// This is an example of defining control points in one region:
/// @code
///     *********
///     *  +  + *-- \'
///     *       *     (1 << controlPointYSpacingLog2)
///     *  +  + *-- /
///     *       *
///     *********
///        |--|
///      (1 << controlPointXSpacingLog2)
///
/// @endcode
/// See #NvMediaLdcWarpMapParameters for additional details of how
/// the control points are organized.
///
/// Restrictions
///
/// * numRegionsX cannot exceed #NVMEDIA_LDC_MAX_REGIONS_X.
/// * numRegionsY cannot exceed #NVMEDIA_LDC_MAX_REGIONS_Y.
/// * Alignment restrictions:
///   - regionWidth[N] must be equal or greater than and a multiple of 64 for
///     each N, except N = numRegionsX - 1.
///   - regionHeight[N] must be equal or greater than and a multiple of 16 for
///     each N, except N = numRegionsY - 1.
/// * Region width/height restrictions:
///  - The sum of regionWidth[] must be equal to the width of the dstRect
///    argument passed to #NvMediaLdcSetGeometry().
///  - The sum of regionHeight[] must be equal to the height of dstRect
///    argument passed to #NvMediaLdcSetGeometry().
///
/// @sa NvMediaLdcIptParameters
/// @sa NvMediaLdcWarpMapParameters
/// @sa NvMediaLdcSetIptParameters()
/// @sa NvMediaLdcSetWarpMapParameters()
///
typedef struct
{
    /// @brief Holds the number of horizontal regions.
    ///        Allowed values are [1, 4], inclusive.
    uint32_t numRegionsX;

    /// @brief Holds the number of vertical regions.
    ///        Allowed values are [1, 4], inclusive.
    uint32_t numRegionsY;

    /// @brief Holds the width of regions.
    uint32_t regionWidth[NVMEDIA_LDC_MAX_REGIONS_X];

    /// @brief Holds the height of regions.
    uint32_t regionHeight[NVMEDIA_LDC_MAX_REGIONS_Y];

    /// @brief Holds the horizontal interval between the control points
    ///        in each region in log2 space.
    uint32_t controlPointXSpacingLog2[NVMEDIA_LDC_MAX_REGIONS_X];

    /// @brief Holds the vertical interval between the control points
    ///        in each region in log2 space.
    uint32_t controlPointYSpacingLog2[NVMEDIA_LDC_MAX_REGIONS_Y];
} NvMediaLdcRegionParameters;

/// @brief Holds inverse perspective transformation configuration.
///
/// @sa NvMediaLdcSetIptParameters()
///
typedef struct
{
    /// @brief Specifies the regions parameters.
    NvMediaLdcRegionParameters regionParams;

    /// @brief Holds the perspective matrix,
    ///        defined in the following way:
    ///            |p[0][0]  p[0][1]  p[0][2]|
    ///            |p[1][0]  p[1][1]  p[1][2]|
    ///            |p[2][0]  p[2][1]  p[2][2]|
    float matrixCoeffs[3][3];
} NvMediaLdcIptParameters;

/// @brief Represents a control point.
typedef struct
{
    /// @brief Horizontal coordinate.
    float x;

    /// @brief Vertical coordinate.
    float y;
} NvMediaLdcControlPoint;

/// @brief Holds the NvMedia LDC definition of a Warp Map.
///
/// Warp Map stores the mappings of each control point from destination
/// image to input image. The coordinates of control points in a destination
/// image are defined by regionParams.
///
/// If warp map region N has controlPointXSpacingLog2[N] = 0 it is considered to
/// have "full columns", and with controlPointYSpacingLog2[N] = 0 "full rows".
/// Otherwise, the region has "sparse columns/rows".
///
/// The warp map processing always operates on 64x16 pixel tiles, so when
/// determining the number of control points in warp map region N,
/// regionWidth[N] is rounded up to the next multiple of 64 and regionHeight[N]
/// to the next multiple of 16.
///
/// In a warp map region with full columns/rows, the number of columns/rows of
/// control points equals the 64x16 rounded up width/height of the region.
///
/// In a warp map region with sparse columns/rows,
/// - First control point column/row is at position 0 within the region.
/// - There's always an odd number of columns/rows of control points.
/// - The position of the last control point column/row is always outside the
///   64x16 rounded up width/height of the region.
///
/// For example, if region width is 192 and control point interval is 64, it
/// will result in 5 columns of control points with X-coordinates 0, 64, 128,
/// 192 and 256.
///
/// The code below shows how to calculate the exact number of control points
/// for a single region:
///
/// @code
///     NvMediaLdcRegionParameters regionInfo;
///     ...
///     w = RoundUp(regionInfo.regionWidth[i], 64);
///     dx = 1 << regionInfo.controlPointXSpacingLog2[i];
///     numColumns = RoundUp(w, 2 * dx) / dx + (dx == 1 ? 0 : 1);
///
///     h = RoundUp(regionInfo.regionHeight[i], 16);
///     dy = 1 << regionInfo.controlPointYSpacingLog2[i];
///     numRows = RoundUp(h, 2 * dy) / dy + (dy == 1 ? 0 : 1);
/// @endcode
///
/// Where `RoundUp()` rounds the first argument up to the nearest multiple of
/// the second argument.
///
/// @sa NvMediaLdcRegionParameters
/// @sa NvMediaLdcSetWarpMapParameters()
///
typedef struct
{
    /// @brief Specifies the regions paramters.
    NvMediaLdcRegionParameters regionParams;

    /// @brief Number of control points.
    uint32_t numControlPoints;

    /// @brief Array of control points across all the regions
    ///        stored in a row-major order.
    NvMediaLdcControlPoint const *controlPoints;
} NvMediaLdcWarpMapParameters;

/// @brief Holds the Mask Map information.
///
/// With the destination rectangle, this mask map surface defines the region of
/// interest in the destination image.
/// The dstRect argument passed to #NvMediaLdcSetGeometry() defines the
/// destination rectangle.
///
/// @sa NvMediaLdcSetMaskMapParameters()
///
typedef struct
{
    /// @brief Holds the width in pixels of the mask map surface,
    ///        which must be equal to the width of the destination rectangle.
    uint32_t width;

    /// @brief Holds the height in pixels of the mask map surface,
    ///        which must be equal to the height of the destination rectangle.
    uint32_t height;

    /// @brief Indicates whether to fill the masked pixel with
    ///        the mask color.
    bool useMaskColor;

    /// @brief Holds the Y channel value of the mask color.
    float maskColorY;

    /// @brief Holds the U channel value of the mask color.
    float maskColorU;

    /// @brief Holds the V channel value of the mask color.
    float maskColorV;

    /// @brief Holds the value for the Mask Map surface.
    ///
    /// The Mask Map surface is stored row by row. Each stored bool value is
    /// used to indicate whether this pixel has been masked or not.
    /// A true value means that the pixel is not to be masked.
    /// The buffer needs to hold width * height bool objects.
    bool const *pixelMasks;
} NvMediaLdcMaskMapParameters;

/// @brief Holds the TNR3 initialization parameters.
///
/// @sa NvMediaLdcSetTnrParameters()
///
typedef struct
{
    /// @brief Holds the sigma of the luma for spatial filter.
    uint32_t spatialSigmaLuma;

    /// @brief Holds the sigma of the chroma for spatial filter.
    uint32_t spatialSigmaChroma;

    /// @brief Holds the sigma of the luma for range filter.
    uint32_t rangeSigmaLuma;

    /// @brief Holds the sigma of the chroma for range filter.
    uint32_t rangeSigmaChroma;

    /// @brief Holds the SAD multiplier parameter.
    float sadMultiplier;

    /// @brief Holds the weight of luma when calculating SAD.
    float sadWeightLuma;

    /// @brief Holds a flag which enables or disables the spatial alpha smooth.
    bool alphaSmoothEnable;

    /// @brief Holds the temporal alpha restrict increase capablility.
    float alphaIncreaseCap;

    /// @brief Holds the alpha scale IIR for strength.
    float alphaScaleIIR;

    /// @brief Holds the max luma value in Alpha Clip Calculation.
    float alphaMaxLuma;

    /// @brief Holds the min luma value in Alpha Clip Calculation.
    float alphaMinLuma;

    /// @brief Holds the max chroma value in Alpha Clip Calculation.
    float alphaMaxChroma;

    /// @brief Holds the min chroma value in Alpha Clip Calculation.
    float alphaMinChroma;

    /// @brief Holds parameter BetaX1 in Beta Calculation.
    float betaX1;

    /// @brief Holds parameter BetaX2 in Beta Calculation.
    float betaX2;

    /// @brief Holds parameter MaxBeta threshold in Beta Calculation.
    float maxBeta;
    /// @brief Holds parameter BetaX2 in Beta Calculation.
    float minBeta;
} NvMediaLdcTnrParameters;

/// @brief Number of bytes in NvMedia LDC checksums.
#define NVMEDIA_LDC_CHECKSUM_NUM_BYTES 24u

/// @brief Represents a checksum.
typedef struct
{
    /// @brief The checksum data.
    uint8_t data[NVMEDIA_LDC_CHECKSUM_NUM_BYTES];
} NvMediaLdcChecksum;

/// @brief NvMedia LDC checksum mode.
///
/// This enum describes the different checksum calculation modes that are
/// supported by NvMedia LDC.
///
typedef enum
{
    /// @brief No checksum calculated. This is the default behavior.
    NVMEDIA_LDC_CHECKSUM_MODE_DISABLED,

    /// @brief Checksum calculated from source surface pixel data.
    NVMEDIA_LDC_CHECKSUM_MODE_SRC_SURFACE,
} NvMediaLdcChecksumMode;

/// @brief Returns the version number of the NvMedia LDC library.
///
/// This function returns the major, minor, and patch version number of the
/// NvMedia LDC library. The client must pass an #NvMediaVersion struct
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
/// @sa NVMEDIA_LDC_VERSION_MAJOR
/// @sa NVMEDIA_LDC_VERSION_MINOR
/// @sa NVMEDIA_LDC_VERSION_PATCH
/// @sa NvMediaVersion
///
NvMediaStatus
NvMediaLdcGetVersion(NvMediaVersion *const version);

/// @brief Creates a new NvMediaLdc context.
///
/// This function creates a new instance of an NvMediaLdc context, and returns a
/// pointer to that context. Ownership of this context is passed to the caller.
/// When no longer in use, the caller must destroy the context using the
/// #NvMediaLdcDestroy() function.
///
/// Default attributes (when not specified by caller):
/// - maxParameters: 1
/// - maxRegisteredBuffers: 16
/// - maxRegisteredSyncs: 16
/// - flags: 0.
///
/// @param[out] handle  Pointer to receive the handle to the new NvMediaLdc
///                     context.
/// @param[in]  attr    Pointer to NvMediaLdcAttributes struct, or NULL for
///                     default attributes.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context created successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL, or @a attr has bad
///                                       attribute values.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  NvMedia LDC is not supported on this
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
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: No
///   - De-Init: No
///
/// @sa NvMediaLdc
/// @sa NvMediaLdcDestroy()
///
NvMediaStatus
NvMediaLdcCreate(NvMediaLdc **const handle, NvMediaLdcAttributes const *const attr);

/// @brief Destroys the NvMediaLdc context.
///
/// This function destroys the specified NvMediaLdc context.
///
/// Before calling this function, the caller must ensure:
/// - There are no NvSciSync or NvSyncBuf objects still registered against the
///   NvMediaLdc context.
/// - All previous LDC operations submitted using NvMediaLdcProcess() have
///    completed.
///
/// @param[in] handle  Pointer to the NvMediaLdc context.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Context destroyed successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  @a handle is NULL.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        There are still some NvSciSync or
///                                       NvSciBuf objects registered against
///                                       the NvMediaLdc context.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to destroy the context. The
///                                       context is in state where the only
///                                       valid operation is to attempt to
///                                       destroy it again.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
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
/// @sa NvMediaLdc
/// @sa NvMediaLdcCreate()
///
NvMediaStatus
NvMediaLdcDestroy(NvMediaLdc *const handle);

/// @brief Creates and returns an #NvMediaLdcParameters instance.
///
/// This functions returns a handle to an NvMediaLdcParameters object.
/// The object will be initialised and ready to use.
/// The attributes attr define resources preallocated per parameters instance.
/// The parameters instance must be destroyed with #NvMediaLdcDestroyParameters()
/// during the De-Init stage.
///
/// The object will be initialised with these default values:
/// - source rectangle: set to the dimensions of the source surface
/// - destination rectangle: set to the dimensions of the destination surface
/// - filter: NVMEDIA_LDC_FILTER_OFF
/// - Temporal Noise Reduction: Disabled
/// - Inverse Perspective Transform: Disabled
/// - Warp Map mode: Disabled
/// - Mask Map: Disabled
///
/// @param[in]  handle  Pointer to the NvMediaLdc context.
/// @param[in]  attr    Pointer to #NvMediaLdcParametersAttributes.
///                     The pointer can be NULL, in which case TNR, Warp Map,
///                     and Mask Map mode will be disabled, and no memory
///                     resources will be pre-allocated.
/// @param[out] params  Pointer to an #NvMediaLdcParameters, which will
///                     be populated with the handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK                      Parameters are created
///                                                successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER           One of the parameters has an
///                                                invalid value, either:
///                                                - @a handle is NULL
///                                                - @a attr has bad attribute
///                                                  values
///                                                - @a params is NULL.
/// @retval NVMEDIA_STATUS_INSUFFICIENT_BUFFERING  Maximum number of parameters
///                                                objects has been created.
/// @retval NVMEDIA_STATUS_INVALID_STATE           The function was called in
///                                                incorrect system state.
/// @retval NVMEDIA_STATUS_OUT_OF_MEMORY           Failed to allocate memory for
///                                                the parameters.
/// @retval NVMEDIA_STATUS_ERROR                   An internal failure occurred
///                                                when trying to create the
///                                                parameters object.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
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
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcParametersAttributes
/// @sa NvMediaLdcDestroyParameters()
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcCreateParameters(NvMediaLdc *const handle,
                           NvMediaLdcParametersAttributes const *const attr,
                           NvMediaLdcParameters *const params);

/// @brief Destroys an #NvMediaLdcParameters instance.
///
/// This functions destroys an NvMediaLdcParameters object.
///
/// @param[in]  handle  Pointer to the #NvMediaLdc context.
/// @param[out] params  An #NvMediaLdcParameters handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters are destroyed successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value, either:
///                                       - @a handle is NULL
///                                       - @a params is invalid.
/// @retval NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect
///                                       system state.
/// @retval NVMEDIA_STATUS_PENDING        The parameters object is still being
///                                       used by a pending operation. The
///                                       object is in state where the only
///                                       valid operation is to attempt to
///                                       destroy it again.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to destroy the parameters
///                                       object. The object is in state where
///                                       the only valid operation is to attempt
///                                       to destroy it again.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcCreateParameters()
///
NvMediaStatus
NvMediaLdcDestroyParameters(NvMediaLdc *const handle, NvMediaLdcParameters const params);

/// @brief Sets the filter mode to use for LDC pixel interpolation.
///
/// @param[in] handle  Pointer to the NvMediaLdc context.
/// @param[in] params  An NvMediaLdcParameters handle.
/// @param[in] filter  An #NvMediaLdcFilter to use when reading the the layer's
///                    source surface.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a filter value is invalid
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcFilter
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetFilter(NvMediaLdc *const handle,
                    NvMediaLdcParameters const params,
                    NvMediaLdcFilter const filter);

/// @brief Sets source and destination rectangles for LDC operation.
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] params     An NvMediaLdcParameters handle.
/// @param[in] srcRect    Pointer to an NvMediaRect that contains the source
///                       rectangle, or NULL for default rectangle.
/// @param[in] dstRect    Pointer to an NvMediaRect that contains the
///                       destination rectangle, or NULL for default rectangle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcSetTnrParameters()
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetGeometry(NvMediaLdc *const handle,
                      NvMediaLdcParameters const params,
                      NvMediaRect const *const srcRect,
                      NvMediaRect const *const dstRect);

/// @brief Sets parameters for LDC Inverse Perspective Transform (IPT)
///        operation.
///
/// This enables VIC geotrans processing in IPT mode.
/// To disable IPT mode iptParams must be set to NULL.
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] params     An NvMediaLdcParameters handle.
/// @param[in] iptParams  Pointer to #NvMediaLdcIptParameters.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a iptParams is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcIptParameters
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetIptParameters(NvMediaLdc *const handle,
                           NvMediaLdcParameters const params,
                           NvMediaLdcIptParameters const *const iptParams);

/// @brief Sets parameters for LDC Warp Map.
///
/// This enables VIC geotrans processing in Warp Map mode.
/// To disable Warp Map mode warpMapParams must be set to NULL.
///
/// @param[in] handle         Pointer to the NvMediaLdc context.
/// @param[in] params         An NvMediaLdcParameters handle.
/// @param[in] warpMapParams  Pointer to #NvMediaLdcWarpMapParameters.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a warpMapParams is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcWarpMapParameters
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetWarpMapParameters(NvMediaLdc *const handle,
                               NvMediaLdcParameters const params,
                               NvMediaLdcWarpMapParameters const *const warpMapParams);

/// @brief Sets parameters for LDC Mask Map.
///
/// This enables VIC Mask Map feature. This feature can be used only if IPT
/// or Warp Map mode is enabled.
/// To disable the feature maskMapParams must be set to NULL.
///
/// @param[in] handle         Pointer to the NvMediaLdc context.
/// @param[in] params         An NvMediaLdcParameters handle.
/// @param[in] maskMapParams  Pointer to #NvMediaLdcMaskMapParameters.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a maskMapParams is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcMaskMapParameters
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetMaskMapParameters(NvMediaLdc *const handle,
                               NvMediaLdcParameters const params,
                               NvMediaLdcMaskMapParameters const *const maskMapParams);

/// @brief Sets Temporal Noise Reduction (TNR) parameters.
///
/// This enables VIC TNR3 processing.
/// To disable TNR3 tnrParams must set to NULL.
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] params     An NvMediaLdcParameters handle.
/// @param[in] tnrParams  Pointer to #NvMediaLdcTnrParameters.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - @a tnrParams is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcTnrParameters
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcProcess()
///
NvMediaStatus
NvMediaLdcSetTnrParameters(NvMediaLdc *const handle,
                           NvMediaLdcParameters const params,
                           NvMediaLdcTnrParameters const *const tnrParams);

/// @brief Resets TNR algorithm state.
///
/// This function resets NvMedia LDC internal TNR algorithm state. After a
/// successful call to this function, the next frame submitted for TNR
/// processing with NvMediaLdcProcess() will be treated by the algorithm as the
/// first one and no previous surface needs to be set for it with
/// NvMediaLdcSetPreviousSurface().
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] params     An NvMediaLdcParameters handle.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               State reset successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
NvMediaStatus
NvMediaLdcResetTnr(NvMediaLdc *const handle, NvMediaLdcParameters const params);

/// @brief Sets checksum calculation mode.
///
/// This function configures the checksums to calculate for the
/// NvMediaLdcProcess() operations performed using @a params.
///
/// NvMedia LDC maintains the checksums from 16 most recent NvMediaLdcProcess()
/// operations that had the checksum calculation configured using this function.
/// When more operations are performed, the oldest checksums are overwritten.
/// The client must ensure the checksums are read with NvMediaLdcGetChecksum()
/// before they are overwritten by subsequent operations.
///
/// @param[in] handle        Pointer to the NvMediaLdc context.
/// @param[in] params        An NvMediaLdcParameters handle.
/// @param[in] checksumMode  The checksum to calculate.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Parameters set successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a params is invalid
///                                       - @a checksumMode is invalid.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
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
/// @sa NvMediaLdcChecksumMode
/// @sa NvMediaLdcGetChecksum()
///
NvMediaStatus
NvMediaLdcSetChecksumMode(NvMediaLdc *const handle,
                          NvMediaLdcParameters const params,
                          NvMediaLdcChecksumMode const checksumMode);

/// @brief Gets a checksum calculated for an NvMediaLdcProcess() operation.
///
/// This function gets a checksum calculated for an NvMediaLdcProcess()
/// operation. The operation must have had checksum calculation configured with
/// NvMediaLdcSetChecksumMode().
///
/// @param[in] handle     Pointer to the NvMediaLdc context.
/// @param[in] result     Pointer to the NvMediaLdcResult object of the
///                       operation.
/// @param[out] checksum  Pointer to an NvMediaLdcChecksum object that will be
///                       populated with the checksum.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK             Checksum returned successfully in
///                                       @a checksum.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER  One of the parameters has an invalid
///                                       value. This could be:
///                                       - @a handle is NULL
///                                       - @a result is NULL
///                                       - @a result indicates no operation was
///                                         submitted
///                                       - No checksum was found for operation
///                                         represented by @a result
///                                       - @a checksum is NULL.
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED  Checksums are not supported on current
///                                       platform.
/// @retval NVMEDIA_STATUS_PENDING        The operation is still pending.
/// @retval NVMEDIA_STATUS_ERROR          An internal failure occurred when
///                                       trying to read the checksum.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a result must be valid NvMediaLdcResult handle returned by
///   NvMediaLdcProcess().
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
/// @sa NvMediaLdcChecksum
/// @sa NvMediaLdcSetChecksumMode()
///
NvMediaStatus
NvMediaLdcGetChecksum(NvMediaLdc *const handle,
                      NvMediaLdcResult const *const result,
                      NvMediaLdcChecksum *const checksum);

/// @brief Performs LDC operation.
///
/// LDC performs transformations depending on the parameters set:
///
/// 1. LDC performs a geometric transformation if #NvMediaLdcSetIptParameters()
///    was called with proper IPT parameters or
///    if #NvMediaLdcSetWarpMapParameters() was called with proper
///    Warp Map parameters.
///    It fetches the pixels in the source image and renders onto the
///    destination rectangle of the destination image.
///    The source image and the destination image must have the same format.
///    LDC bypasses the transform stage if no IPT or Warp Map parameters
///    are set.
///
///    The region of interest in the destination image is defined by:
///    - Destination rectangle and
///    - Mask Map
///      (if a Mask Map set with #NvMediaLdcSetMaskMapParameters()).
///
/// 2. LDC outputs xSobel if #setXSobelDstSurface() was called prior.
///    The xSobel image must have the same bit-depth as the source image.
///
/// 3. LDC outputs 4x4 downsampled xSobel output if
///    #NvMediaLdcSetDownsampledXSobelDstSurface() was called prior.
///    The downsampled image must have the same bit depth as the source image.
///
///    In order to produce xSobel output, xSobel surface must be explicitly
///    set prior to every #NvMediaLdcProcess() call, just like all other
///    surfaces. If xSobel surface is not explicitly set
///    between #NvMediaLdcProcess() calls, the xSobel output will be disabled.
///    The same applies to downsampled xSobel.
///
/// 4. LDC performs Temporal Noise Reduction is #NvMediaLdcSetTnrParameters()
///    was called with proper TNR parameters.
///
/// @anchor NvMediaLdcProcessParameterRestrictions
///
/// Restrictions on dimensions for source and destination images:
/// - Width must be even and within the range: [64, 16384]
/// - Height must be even and within the range: [16, 16384]
///
/// Restrictions on the source rectangle:
/// - Must be within the bounds of the source image.
/// - Width and height must be greater than zero.
/// - For any YUV format with chroma subsampling different than 444, all corners
///   must be aligned to a a multiple of 2.
///
/// Restrictions on the destination rectangle:
/// - Top-left corner must be (0, 0).
/// - Must be within the bounds of the destination image.
/// - Width and height must be greater than zero.
/// - For any YUV format with chroma subsampling different than 444, all corners
///   must be aligned to a a multiple of 2.
///
/// @anchor NvMediaLdcProcessConcurrencyRestrictions
///
/// Restrictions on concurrency:
/// - There can be a maximum of 16 operations submitted through the same
///   NvMediaLdc handle pending simultaneously.
///
/// If any of the restrictions are violated, this function will fail with an
/// error code.
///
/// @param[in]  handle  Pointer to the NvMediaLdc context.
/// @param[in]  params  An NvMediaLdcParameters handle.
/// @param[out] result  Pointer to NvMediaLdcResult struct that will be
///                     populated with result info. May be NULL.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK               Operation submitted successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER    One of the parameters has an invalid
///                                         value. This could be:
///                                         - @a handle is NULL
///                                         - @a params is invalid
///                                         - some of the parameters configured
///                                           through @a params have invalid
///                                           values (see @ref
///                                           NvMediaLdcProcessParameterRestrictions
///                                           "restrictions on parameters").
/// @retval NVMEDIA_STATUS_NOT_SUPPORTED    Requested operation is not supported
///                                         by current platform or some of the
///                                         parameters combinations
///                                         are not supported by the current
///                                         implementation (see @ref
///                                         NvMediaLdcProcessParameterRestrictions
///                                         "restrictions on parameters").
/// @retval NVMEDIA_STATUS_TIMED_OUT        No space available in the command
///                                         buffer for this operation, because
///                                         previous operations are still
///                                         pending (see @ref
///                                         NvMediaLdcProcessConcurrencyRestrictions
///                                         "restrictions on concurrency"). The
///                                         caller should wait for the least
///                                         recently submitted operation to
///                                         complete and then try again.
/// @retval NVMEDIA_STATUS_ERROR            An internal failure occurred when
///                                         trying to perform the LDC operation.
///                                         This error indicates the system is
///                                         potentially in an unrecoverable
///                                         state.
///
/// @pre
/// - @a handle must be valid NvMediaLdc handle created with NvMediaLdcCreate().
/// - @a params must be valid NvMediaLdcParameters handle created with
///   NvMediaLdcCreateParameters().
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different NvMediaLdc handle
///   - Re-entrant: No
///   - Async/Sync: Async
/// - Required privileges: None
/// - API group
///   - Init: No
///   - Runtime: Yes
///   - De-Init: No
///
/// @sa NvMediaLdcParameters
/// @sa NvMediaLdcCreateParameters()
/// @sa NvMediaLdcSetFilter()
/// @sa NvMediaLdcSetGeometry()
/// @sa NvMediaLdcSetIptParameters()
/// @sa NvMediaLdcSetMaskMapParameters()
/// @sa NvMediaLdcSetTnrParameters()
/// @sa NvMediaLdcSetWarpMapParameters()
/// @sa NvMediaLdcInsertPreNvSciSyncFence()
/// @sa NvMediaLdcSetNvSciSyncObjforEOF()
/// @sa NvMediaLdcGetEOFNvSciSyncFence()
/// @sa NvMediaLdcSetSrcSurface()
/// @sa NvMediaLdcSetDstSurface()
/// @sa NvMediaLdcSetPreviousSurface()
/// @sa NvMediaLdcSetXSobelDstSurface()
/// @sa NvMediaLdcSetDownsampledXSobelDstSurface()
///
NvMediaStatus
NvMediaLdcProcess(NvMediaLdc *const handle,
                  NvMediaLdcParameters const params,
                  NvMediaLdcResult *const result);

///
/// @}
///

//
// Version History
//
// Version 1.0 May 1, 2017
// - Initial release.
//
// Version 1.1 March 16, 2018
// - Add support of TNR2
//
// Version 1.2 September 4, 2018
// - New member variable srcSurfaceType is added to NvMediaLDCInitParams.
//
// Version 1.3 December 26, 2018
// - Adding unsigned tag to macro constants to fix MISRA 10.4 violation.
// - Fixing MISRA 21.1 violations
//
// Version 1.4 January 2, 2019
// - Added deprecated warning message for NvMediaLDCCreate.
// - Added API NvMediaLDCCreateNew
//
// Version 1.5 March 6, 2019
// - Fixing MISRA 8.13 violations.
//
// Version 1.6 March 12, 2019
// - Added required header includes nvmedia_core.h and nvmedia_surface.h
//
// Version 2.0 March 29, 2019
// - Deprecated NvMediaLDCCreate API.
//
// Version 2.1 January 23, 2020
// - Limited destination surface rectangle's top, left corner to (0, 0).
//
// Version 3.0 December 26, 2021
// - Major rework of API, NvMediaImage support was deprecated and
//   all functionality related to NvMediaImage has been updated
//   in order to support NvSciBuf instead of NvMediaImage.
// - The workflow was changed from setting the parameters
//   at the Init stage to using a subset of API that set parameters
//   prior to NvMediaLdcProcess() call.
//
// Version 4.0.0 March 28, 2022
// - Add support for NvSciSync task statuses
// - Max pre-fence count changed from 32 to 16
//
// Version 5.0.0 May 10, 2022
// - Re-added the destination rectangle top-left (0, 0) limitation.
//
// Version 6.0.0 June 7, 2022
// - Forbid registering same buffer multiple times
// - Error codes changed for multiple APIs
//
// Version 7.0.0 July 8, 2022
// - New error NVMEDIA_STATUS_INVALID_STATE added for multiple APIs
//
// Version 7.0.1 August 25, 2022
// - Allow NULL context handle in NvMediaLdcFillNvSciBufAttrList
//
// Version 7.0.2 September 2, 2022
// - Always treat parameters handle value 0 as invalid
//
// Version 8.0.0 March 10, 2023
// - NvMediaLdcSetGeometry() and NvMediaLdcSetTnrParameters() no longer reset
//   TNR algorithm state
// - New API NvMediaLdcResetTnr() added for explicitly resetting the state
//
// Version 8.1.0 June 2, 2023
// - New checksum API added
//

#ifdef __cplusplus
}
#endif

#endif
