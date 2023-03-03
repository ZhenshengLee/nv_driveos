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
/// @brief <b> NVIDIA Media Interface: LDC Utility functions </b>
///

#ifndef NVMEDIA_LDC_UTIL_H
#define NVMEDIA_LDC_UTIL_H

#include "nvmedia_core.h"
#include "nvmedia_ldc.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Supported lens models used to generate a Warp Map.
///
/// @sa NvMediaLdcGenWarpMap
///
typedef enum
{
    /// @brief Specifies a polynomial distortion model.
    NVMEDIA_LDC_LENS_MODEL_POLYNOMIAL_DISTORTION,

    /// @brief Specifies a fisheye model: r = 2ftan(theta/2), where theta
    ///        is the angle from the optical axis, f is the focal length,
    ///        and r is the distance of a pixel from the image center.
    NVMEDIA_LDC_LENS_MODEL_FISHEYE_EQUIDISTANT,

    /// @brief Specifies a fisheye model: r = f*theta, where theta
    ///        is the angle from the optical axis, f is the focal length,
    ///        and r is the distance of a pixel from the image center.
    NVMEDIA_LDC_LENS_MODEL_FISHEYE_EQUISOLID,

    /// @brief Specifies a fisheye model: r = 2fsin(theta/2), where theta
    ///        is the angle from the optical axis, f is the focal length,
    ///        and r is the distance of a pixel from the image center.
    NVMEDIA_LDC_LENS_MODEL_FISHEYE_ORTHOGRAPHIC,

    /// @brief Specifies a fisheye model: r = fsin(theta), where theta
    ///        is the angle from the optical axis, f is the focal length,
    ///        and r is the distance of a pixel from the image center.
    NVMEDIA_LDC_LENS_MODEL_FISHEYE_STEREOGRAPHIC
} NvMediaLdcLensModel;

/// @brief Instrinsic camera parameters.
///
/// Camera model parameters that encompass
/// focal length and optical center.
///
/// @sa NvMediaLdcGenWarpMap
///
typedef struct
{
    /// @brief Matrix of camera intrinsic parameters.
    ///
    /// Parameters matrix is defined as follow:
    ///             | -f/s_x     S      o_x  |
    ///     M_int = |    0    -f/s_y    o_y  |
    ///             |    0       0       1   |
    ///     where f is a focal length,
    ///     (o_x, o_y) are principal point coordinates,
    ///     S is axis skew,
    ///     and s_x, s_y is the size of the pixels.
    /// The last row is excluded from the matrixCoeffs.
    float matrixCoeffs[2][3];
} NvMediaLdcCameraIntrinsic;

/// @brief Extrinsic camera parameters.
///
/// Camera model parameters that define a transform from real world coordinates
/// to camera coordinates.
///
/// @sa NvMediaLdcGenWarpMap
///
typedef struct
{
    /// @brief Rotation matrix.
    float R[3][3];
    /// @brief Translation vector.
    float T[3];
} NvMediaLdcCameraExtrinsic;

/// @brief Distortion coefficients for the lens model.
///
/// Distortion coefficients are defined in the following way:
/// - k1, k2, k3, k4, k5, k6 are radial distortion coeffcients.
/// - p1 and p2 are tangential distortion coeffcients.
///
/// Setting any coefficient to 0 implies that it is not used in computation.
///
/// If we denote a point without distortion as [x, y, 1] and the corresponding
/// point with distortion as [xd, yd, 1], then the distortion model is defined
/// as follows:
///
/// When #NvMediaLdcLensModel is #NVMEDIA_LDC_LENS_MODEL_POLYNOMIAL_DISTORTION,
/// the control parameters are k1, k2, k3, k4, k5, k6, p1, and p2.
/// - \f$r = sqrt (x ^ 2 + y ^ 2)\f$
/// - \f$kr = (1 + k1 * r^2 + k2 * r^4 + k3 * r^6) /
///           (1 + k4 * r^2 + k5 * r^4 + k6 * r^6)\f$
/// - \f$xd = x * kr + p1 * (2 * x * y) + p2 * (r^2 + 2 * x^2)\f$
/// - \f$yd = y * kr + p1 * (r^2 + 2 * y^2) + p2 * (2 * x * y)\f$
///
/// When NvMediaLensModel is #NVMEDIA_LDC_LENS_MODEL_FISHEYE_EQUIDISTANT,
/// #NVMEDIA_LDC_LENS_MODEL_FISHEYE_EQUISOLID,
/// #NVMEDIA_LDC_LENS_MODEL_FISHEYE_ORTHOGRAPHIC, or
/// #NVMEDIA_LDC_LENS_MODEL_FISHEYE_STEREOGRAPHIC,
/// the control parameters are k1, k2, k3, and k4.
/// - \f$r = sqrt (x ^ 2 + y ^ 2)\f$
/// - \f$theta = atan(r)\f$
/// - \f$theta_d = theta * (1 + k1 * theta^2 +
///                             k2 * theta^4 +
///                             k3 * theta^6 +
///                             k4 * theta^8)\f$
///
/// @sa NvMediaLdcGenWarpMap
///
typedef struct
{
    /// @brief Camera model.
    NvMediaLdcLensModel model;

    /// @brief Holds the radial distortion coefficient.
    float k1;
    /// @brief Holds the radial distortion coefficient.
    float k2;
    /// @brief Holds the radial distortion coefficient.
    float k3;
    /// @brief Holds the radial distortion coefficient.
    float k4;
    /// @brief Holds the radial distortion coefficient.
    float k5;
    /// @brief Holds the radial distortion coefficient.
    float k6;
    /// @brief Holds the tangential distortion coefficient.
    float p1;
    /// @brief Holds the tangential distortion coefficient.
    float p2;
} NvMediaLdcLensDistortion;

/// @brief Helper function to calculate the number of control points
///        from the defined sparse warp map region parameters.
///
/// @param[in] regionParams Pointer to #NvMediaLdcRegionParameters
///                         that contains parameters for regions of a Warp Map.
///
/// @param[out] numControlPoints Number of Control Points calculated.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK Number of Control Points has been calculated
///                           successfully.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER One of parameters is invalid.
///                                      This could be either:
///                                      - @a regionParams is NULL
///                                      - @a numControlPoints is NULL
///                                      - @a regionParams has values that
///                                           out of acceptable domain.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different numControlPoints object
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: Yes
///   - De-Init: Yes
///
NvMediaStatus
NvMediaLdcGetNumControlPoints(NvMediaLdcRegionParameters const *const regionParams,
                              uint32_t *numControlPoints);

/// @brief Helper function to fills the given warp map with an identity mapping.
///
/// This function is useful if the user wants to specify their own mapping.
/// It sets the control points coordinates to the destination coordinates as
/// defined implicitly by the #NvMediaLdcRegionParameters.
/// The user then can iterate through these points and apply a custom
/// mapping function to each one.
///
/// @param[in] dstRect Pointer to the destination rectangle.
/// @param[in] regionParams Pointer to #NvMediaLdcRegionParameters for Warp Map.
/// @param[in] numControlPoints Number of control points in Warp Map.
/// @param[in] controlPoints Pointer to the control points array.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK Warp Map was successfully configured.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER One of parameters is invalid.
///                                      This could be either:
///                                      - @a dstRect is NULL
///                                      - @a regionParams is NULL
///                                      - @a numControlPoints is 0
///                                      - @a controlPoints is NULL
///                                      - @a regionParams has values that
///                                           out of acceptable domain.
///                                      - @a number of control points
///                                           is inconsistent with
///                                           regions config.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different controlPoints object
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: Yes
///   - De-Init: Yes
///
NvMediaStatus
NvMediaLdcGenWarpMapIdentity(NvMediaRect const *const dstRect,
                             NvMediaLdcRegionParameters const *const regionParams,
                             uint32_t numControlPoints,
                             NvMediaLdcControlPoint *const controlPoints);

/// @brief Generates Warp Map for a particular lens model.
///
/// @param[in] Kin Pointer to #NvMediaLdcCameraIntrinsic parameters.
/// @param[in] X Pointer to #NvMediaLdcCameraExtrinsic parameters.
/// @param[in] Kout Pointer to #NvMediaLdcCameraIntrinsic parameters.
/// @param[in] distModel Pointer to #NvMediaLdcLensDistortion struct.
/// @param[in] dstRect Pointer to the destination rectangle.
/// @param[in] regionParams Pointer to #NvMediaLdcRegionParameters for Warp Map.
/// @param[in] numControlPoints Number of control points in Warp Map.
/// @param[in] controlPoints Pointer to the control points array.
///
/// @return An #NvMediaStatus return code.
/// @retval NVMEDIA_STATUS_OK Warp Map is successfully generated.
/// @retval NVMEDIA_STATUS_BAD_PARAMETER One of parameters is invalid.
///                                      This could be either:
///                                      - @a Kin is NULL
///                                      - @a X is NULL
///                                      - @a Kout is NULL
///                                      - @a distModel is NULL
///                                      - @a dstRect is NULL
///                                      - @a regionParams is NULL
///                                      - @a numControlPoints is 0
///                                      - @a controlPoints is NULL
///                                      - @a regionParams has values that
///                                           out of acceptable domain.
///                                      - @a number of control points
///                                           is inconsistent with
///                                           regions config.
///                                      - @a X is invalid, determinant is 0.
///
/// @usage
/// - Allowed context for the API call
///   - Interrupt handler: No
///   - Signal handler: No
///   - Thread-safe: Yes, with the following conditions:
///     - Each thread uses different controlPoints object
///   - Re-entrant: No
///   - Async/Sync: Sync
/// - Required privileges: None
/// - API group
///   - Init: Yes
///   - Runtime: Yes
///   - De-Init: Yes
///
NvMediaStatus
NvMediaLdcGenWarpMap(
    NvMediaLdcCameraIntrinsic const *const Kin,
    NvMediaLdcCameraExtrinsic const *const X,
    NvMediaLdcCameraIntrinsic const *const Kout,
    NvMediaLdcLensDistortion const *const distModel,
    NvMediaRect const *const dstRect,
    NvMediaLdcRegionParameters const *const regionParams,
    uint32_t numControlPoints,
    NvMediaLdcControlPoint *const controlPoints);

#ifdef __cplusplus
}
#endif

#endif
