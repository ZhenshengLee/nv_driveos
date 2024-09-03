/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLISPSTRUCTS_HPP
#define NVSIPLISPSTRUCTS_HPP

#include "NvSiplControlAutoDef.hpp"

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: ISP Definitions - @ref NvSIPLClient_API </b>
 *
 */

namespace nvsipl
{
/**
 * \defgroup NvSIPLISPStructs NvSIPL ISP Structures
 *
 * @brief NvSipl ISP Defines for ISP Structures.
 *
 * @ingroup NvSIPLCamera_API
 */

/** @addtogroup NvSIPLISPStructs
 * @{
 */

/** @brief Defines the number of global tone map spline points. */
constexpr uint32_t NUM_GTM_SPLINE_POINTS = 18U;

/** @brief Defines the global tone map spline. */
struct NvSiplISPGlobalToneMapInfo
{
    /** Holds boolean to enable global tone map block */
    bool enable;
    /** Defines a spline control point */
    NvSiplISPSplineControlPoint gtmSplineControlPoint[NUM_GTM_SPLINE_POINTS];
};

/** @brief Defines the length(M) of a MxM luminance calibration matrix. */
constexpr uint32_t NVSIPL_LUMINANCE_CALIB_MATRIX_SIZE {4U};

/** @brief Defines the length(M) of a MxM color correction matrix(ccm). */
constexpr uint32_t NVSIPL_CCM_MATRIX_SIZE {3U};

/** @brief Defines the control info. */
struct NvSiplControlInfo
{
    /** Holds a flag to determine whether or not the control info is valid. If no ISP processing occurs this value is false. */
    bool valid;
    /** Holds power factor for isp statistics compression. */
    float_t alpha;
    /** Holds a flag indicating if the sensor is luminance calibrated. */
    bool isLuminanceCalibrated;
    /**
     * (note: parameter to be deprecated starting 6.0.7.0)
     * Holds a luminance calibration factor ( K / f^2 ) for luminance calibrated sensors.
     *
     * Definition:
     *
     *    N = K * (t*S / (f^2)) * L
     *    N: pixel value (e.g. RGGB channel average value)
     *    K: calibration constant
     *    t: sensor exposure time in seconds (if number of exposures > 1, use long exposure time value)
     *    S: sensor gain (if number of exposures > 1, use long exposure gain value)
     *    f: aperture number (f-stop)
     *    L: Luminance
     *
     *    Let luminanceCalibrationFactor = K / (f^2) = N / (t*S*L)
     */
    double_t luminanceCalibrationFactor;
    /**
     * Holds the luminance calibration matrix for the sensor.
     * @li Supported values [1E-12, 1E12]
     */
    float_t luminanceCalibrationMatrix[NVSIPL_LUMINANCE_CALIB_MATRIX_SIZE][NVSIPL_LUMINANCE_CALIB_MATRIX_SIZE];
    /** Holds the total white balance gains, which includes both sensor channel and ISP gains. */
    SiplControlAutoAwbGain wbGainTotal;
    /** Holds the correlated color temperature. */
    float_t cct;
    /** Holds the scene brightness key. */
    float_t brightnessKey;
    /** Holds the scene dynamic range. */
    float_t sceneDynamicRange;
    /** Holds the scene brightness level. */
    float_t sceneBrightness;
    /** Holds the midtone value of the raw image. */
    float_t rawImageMidTone;
    /** Holds the global tonemap block, containing a set of spline control points */
    NvSiplISPGlobalToneMapInfo gtmSplineInfo;
    /** Holds the color correction matrix. */
    float_t ccm[NVSIPL_CCM_MATRIX_SIZE][NVSIPL_CCM_MATRIX_SIZE];
};

/** @brief Downscale and crop configuration. */
struct NvSIPLDownscaleCropCfg
{
    /** Indicates if ISP input crop is enabled. */
    bool ispInputCropEnable {false};
    /** ISP input crop rectangle.
     * Valid range of the rectangle depends on tuning parameters
     * Users needs to ensure that crop size is greater than Stats ROI
     * Coordinates of image top-left & bottom-right points are (0, 0) &
     * (width, height) respectively.
     * Input crop only supports cropping in vertical direction, meaning
     * left & right values must be 0 & input image width, respectively.
     * Crop top should be within [0, image height-2].
     * Crop bottom should be within [2, image height].
     * Vertical cropped size (bottom - top) should be even number, and
     * not smaller than 128.
    */
    NvSiplRect ispInputCrop {};

    /** Indicates if ISP0 output crop is enabled. */
    bool isp0OutputCropEnable {false};
    /** ISP0 output crop rectangle.
     * Coordinates of image top-left & bottom-right points are (0, 0) &
     * (width, height) respectively.
     * Rectangle must be within input image or downscaled image if
     * downscaling of ISP0 output is enabled. Cropped width & height
     * must be same as output buffer width & height respectively,
     * and cropped width and height must be even, and not smaller 
     * then 128.
     * Crop left should be within [0, image width-128].
     * Crop right should be within [128, image width].
     * Crop top should be within [0, image height-128].
     * Crop bot should be within [128, image heigh].
     */
    NvSiplRect isp0OutputCrop {};

    /** Indicates if ISP1 output crop is enabled. */
    bool isp1OutputCropEnable {false};
    /** ISP1 output crop rectangle.
     * Coordinates of image top-left & bottom-right points are (0, 0) &
     * (width, height) respectively.
     * Rectangle must be within input image or downscaled image if
     * downscaling of ISP1 output is enabled. Cropped width & height
     * must be same as output buffer width & height respectively,
     * and cropped width and height must be even, and not smaller 
     * then 128.
     * Crop left should be within [0, image width-128].
     * Crop right should be within [128, image width].
     * Crop top should be within [0, image height-128].
     * Crop bot should be within [128, image heigh].
     */
    NvSiplRect isp1OutputCrop {};

    /** Indicates if ISP2 output crop is enabled. */
    bool isp2OutputCropEnable {false};
    /** ISP2 output crop rectangle.
     * Coordinates of image top-left & bottom-right points are (0, 0) &
     * (width, height) respectively.
     * Rectangle must be within input image or downscaled image if
     * downscaling of ISP2 output is enabled. Cropped width & height
     * must be same as output buffer width & height respectively,
     * and cropped width and height must be even, and not smaller 
     * then 128.
     * Crop left should be within [0, image width-128].
     * Crop right should be within [128, image width].
     * Crop top should be within [0, image height-128].
     * Crop bot should be within [128, image heigh].
     */
    NvSiplRect isp2OutputCrop {};

    /** Indicates if ISP0 downscale is enabled. */
    bool isp0DownscaleEnable {false};
    /** ISP0 downscale width.
     * Supported values: [max(128, ceil(input width / 32.0)), input width]
     */
    uint32_t isp0DownscaleWidth {0U};
    /** ISP0 downscale height.
     * Supported values: [max(128, ceil(input height / 32.0)), input height]
     */
    uint32_t isp0DownscaleHeight {0U};

    /** Indicates if ISP1 downscale is enabled.
     * For RGB-IR the flag must be enabled when ISP1 output is enabled.
    */
    bool isp1DownscaleEnable {false};
    /** ISP1 downscale width.
     * Supported values: [max(128, ceil(input width / 32.0)), input width]
     * For RGB-IR isp1DownscaleWidth must be set to half the input width
     */
    uint32_t isp1DownscaleWidth {0U};
    /** ISP1 downscale height.
     * Supported values: [max(128, ceil(input height / 32.0)), input height]
     * For RGB-IR isp1DownscaleHeight must be set to half the input height
     */
    uint32_t isp1DownscaleHeight {0U};

    /** Indicates if ISP2 downscale is enabled. */
    bool isp2DownscaleEnable {false};
    /** ISP2 downscale width.
     * Supported values: [max(128, ceil(input width / 32.0)), input width]
     */
    uint32_t isp2DownscaleWidth {0U};
    /** ISP2 downscale height.
     * Supported values: [max(128, ceil(input height / 32.0)), input height]
     */
    uint32_t isp2DownscaleHeight {0U};

};

/** @} */

} // namespace nvsipl

#endif // NVSIPLISPSTRUCTS_HPP
