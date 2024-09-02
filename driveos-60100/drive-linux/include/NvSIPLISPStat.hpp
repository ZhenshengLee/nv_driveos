/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file
 * @brief <b> NvSipl ISP stat struct </b>
 */

#ifndef NVSIPL_ISP_STAT_H
#define NVSIPL_ISP_STAT_H

#include <cstdint>
#include "NvSIPLCommon.hpp"

/**
 * @brief Number of histogram bins.
 */
#define NVSIPL_ISP_HIST_BINS                   (256U)
/**
 * @brief Maximum number of color components.
 */
#define NVSIPL_ISP_MAX_COLOR_COMPONENT         (4U)

/**
 * @brief Number of histogram knee points.
 */
#define NVSIPL_ISP_HIST_KNEE_POINTS            (8U)

/**
 * @brief Number of radial transfer function control points.
 */
#define NVSIPL_ISP_RADTF_POINTS                (6U)

/**
 * @brief Maximum number of local average and clip statistic block
 * regions of interest.
 */
#define NVSIPL_ISP_MAX_LAC_ROI                 (4U)

/**
 * @brief Maximum number of input planes.
 */
#define NVSIPL_ISP_MAX_INPUT_PLANES            (3U)

/**
 * @brief Maximum matrix dimension.
 */
#define NVSIPL_ISP_MAX_COLORMATRIX_DIM         (3U)

/**
 * @brief Maximum number of windows for local average and clip in a region of
 * interest.
 * - value is (32U * 32U)
 */
#define NVSIPL_ISP_MAX_LAC_ROI_WINDOWS         (1024U)

/**
 * @brief Maximum number of bands for flicker band statistics block.
 */
#define NVSIPL_ISP_MAX_FB_BANDS                (256U)

namespace nvsipl
{

/**
 * @defgroup NvSIPLISPStats NvSIPL ISP Stats
 *
 * @brief NvSipl ISP Defines for ISP Stat structures.
 *
 * @ingroup NvSIPLCamera_API
 */
/** @addtogroup NvSIPLISPStats
 * @{
 */

/**
 * @brief Holds bad pixel statistics (BP Stats).
 */
typedef struct {
    /**
     * Holds bad pixel count for pixels corrected upward within the window.
     * Valid Range: [0, image_width x image_height]
     */
    uint32_t highInWin;
    /**
     * Holds bad pixel count for pixels corrected downward within the window.
     * Valid Range: [0, image_width x image_height]
     */
    uint32_t lowInWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected upward within the
     * window.
     * Valid Range: [0, UINT32_MAX]
     */
    uint32_t highMagInWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected downward within
     * the window.
     * Valid Range: [0, UINT32_MAX]
     */
    uint32_t lowMagInWin;
    /**
     * Holds bad pixel count for pixels corrected upward outside the window.
     * Valid Range: [0, image_width x image_height]
     */
    uint32_t highOutWin;
    /**
     * Holds bad pixel count for pixels corrected downward outside the window.
     * Valid Range: [0, image_width x image_height]
     */
    uint32_t lowOutWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected upward outside
     * the window.
     * Valid Range: [0, UINT32_MAX]
     */
    uint32_t highMagOutWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected downward outside
     * the window.
     * Valid Range: [0, UINT32_MAX]
     */
    uint32_t lowMagOutWin;
} NvSiplISPBadPixelStatsData;

/**
 * @brief Holds histogram statistics (HIST Stats).
 */
typedef struct {
    /**
     * Holds histogram data for each color component in RGGB/RCCB/RCCC order.
     * Valid Ranges: [0, (W x H)/4].
     * W is the input width
     * H is the input height
     */
    uint32_t data[NVSIPL_ISP_HIST_BINS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds the number of pixels excluded by the elliptical mask for each
     * color component.
     * Valid Ranges: [0, (W x H)/4].
     * W is the input width
     * H is the input height
     */
    uint32_t excludedCount[NVSIPL_ISP_MAX_COLOR_COMPONENT];
} NvSiplISPHistogramStatsData;

/**
 * @brief Holds local average and clip statistics data for a region of interest.
 */
typedef struct {
    /**
     * Holds number of windows horizontally in one region of interest.
     * Valid Range: [1, 32]
     */
    uint32_t numWindowsH;
    /**
     * Holds number of windows vertically in one region of interest.
     * Valid Range: [1, 32]
     */
    uint32_t numWindowsV;
    /**
     * Holds average pixel value for each color component in each window in
     * RGGB/RCCB/RCCC order.
     * Valid Range: [0.0, 1.0]
     */
    float_t average[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds the number of pixels excluded by the elliptical mask for each
     * color component in each window
     * in RGGB/RCCB/RCCC order.
     * Valid Range: [0, M/4]
     * M is the number of pixels per color component in the window.
     */
    uint32_t maskedOffCount[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds number of clipped pixels for each color component in each window in
     * RGGB/RCCB/RCCC order.
     * Valid Range: [0, M/4]
     * M is the number of pixels per color component in the window.
     */
    uint32_t clippedCount[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
} NvSiplISPLocalAvgClipStatsROIData;

/**
 * @brief Defines an ellipse.
 */
typedef struct {
    /**
     * Holds center of the ellipse.
     * Valid Range:
     * @li X coordinate of the center: [0, input width - 1]
     * @li Y coordinate of the center: [0, input height - 1]
     */
    NvSiplPointFloat center;
    /**
     * Holds horizontal axis of the ellipse.
     * Valid Range: [17, 2 x input width]
     */
    uint32_t horizontalAxis;
    /**
     * Holds vertical axis of the ellipse.
     * Valid Range: [17, 2 x input height]
     */
    uint32_t verticalAxis;
    /**
     * Holds angle of the ellipse horizontal axis from X axis in degrees in
     * clockwise direction.
     * Valid Range: [0.0, 360.0]
     */
    float_t angle;
} NvSiplISPEllipse;

/**
 * @brief Holds controls for flicker band statistics (FB Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable flicker band statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds the offset of the first band top line.
     * @li X coordinate of start offset: [0, input width]
     * @li Y coordinate of start offset: [0, input height]
     * @li The X coordinate of the start offset must be an even number.
     */
    NvSiplPoint startOffset;
    /**
     * Holds count of flicker band samples to collect per frame.
     * Valid Range: [1, 256]
     * @li Constraints: If bandCount == 256, bottom of last band
     * must align with bottom of the image.
     */
    uint16_t bandCount;
    /**
     * Holds width of single band.
     * Valid Range: [2, input width - startOffset.x];
     *  must be an even number
     * @li Constrains: Total number of accumulated pixels must be <= 2^18
     */
    uint32_t bandWidth;
    /**
     * Holds height of single band.
     * Valid Range: [2, input height - startOffset.y]
     * @li Constrains: Total number of accumulated pixels must be <= 2^18
     * @li Constrains: If bandCount == 256, bottom of last band
     * must align with bottom of the image.
     */
    uint32_t bandHeight;
    /**
     * Holds minimum value of pixel to include for flicker band stats.
     * Valid Range: [0.0, 1.0]
     */
    float_t min;
    /**
     * Holds maximum value of pixel to include for flicker band stats.
     * Valid Range: [0.0, 1.0], max >= min
     */
    float_t max;
    /**
     * Holds a Boolean to enable an elliptical mask for excluding pixels
     * outside a specified area.
     */
    NvSiplBool ellipticalMaskEnable;
    /**
     * Holds an elliptical mask to exclude pixels outside a specified area.
     *
     * Coordinates of the image's top left and bottom right points are (0, 0)
     * and (width, height) respectively.
     */
    NvSiplISPEllipse ellipticalMask;
} NvSiplISPFlickerBandStats;

/**
 * @brief Holds flicker band statistics (FB Stats).
 */
typedef struct {
    /**
     * Holds band count.
     * Valid Range: [1, 256]
     */
    uint32_t bandCount;
    /**
     * Holds average luminance value for each band.
     * Valid Range: [0.0, 1.0]
     */
    float_t luminance[NVSIPL_ISP_MAX_FB_BANDS];
} NvSiplISPFlickerBandStatsData;

/**
 * @brief Defines the windows used in ISP LAC stats calculations.
 *
 * @code
 * ------------------------------------------------------------------------------
 * |         startOffset    horizontalInterval                                  |
 * |                    \  |--------------|                                     |
 * |                     - *******        *******        *******                |
 * |                     | *     *        *     *        *     *                |
 * |                     | *     *        *     *        *     *                |
 * |                     | *     *        *     *        *     *                |
 * |                     | *******        *******        *******                |
 * |  verticalInterval-->|                                        \             |
 * |                     |                                          numWindowsV |
 * |                     |                                        /             |
 * |                     - *******        *******        *******                |
 * |                     | *     *        *     *        *     *                |
 * |            height-->| *     *        *     *        *     *                |
 * |                     | *     *        *     *        *     *                |
 * |                     - *******        *******        *******                |
 * |                       |-----|                                              |
 * |                        width     \      |     /                            |
 * |                                    numWindowsH                             |
 * ------------------------------------------------------------------------------
 * @endcode
 */
typedef struct {
    /**
     * Holds width of the window in pixels.
     * Valid Range: [2, 256] and must be an even number
     */
    uint32_t width;
    /**
     * Holds height of the window in pixels.
     * Valid Range: [2, 256]
     */
    uint32_t height;
    /**
     * Holds number of windows horizontally.
     * Valid Range: [1, 32]
     */
    uint32_t numWindowsH;
    /**
     * Holds number of windows vertically.
     * Valid Range: [1, 32]
     */
    uint32_t numWindowsV;
    /**
     * Holds the distance between the left edge of one window and a horizontally
     * adjacent window.
     * Valid Range: [max(4, LAC window width), LAC ROI width] and must be an even number
     */
    uint32_t horizontalInterval;
    /**
     * Holds the distance between the top edge of one window and a vertically
     * adjacent window.
     * Valid Range: [max(2, LAC window height), LAC ROI height]
     */
    uint32_t verticalInterval;
    /**
     * Holds the position of the top left pixel in the top left window.
     * Valid Range:
     * @li X coordinate of start offset: [0, LAC ROI width-3] and must be an even number
     * @li Y coordinate of start offset: [0, LAC ROI height-3]
     * @li startOffset.x + horizontalInterval * (numWindowH - 1) + winWidth <= LAC ROI width
     * @li startOffset.y + veritcallInterval * (numWindowV - 1) + winHeight <= LAC ROI height
     */
    NvSiplPoint startOffset;
} NvSiplISPStatisticsWindows;


/**
 * @brief Defines a spline control point.
 */
typedef struct {
    /**
     * Holds X coordinate of the control point.
     * Valid Range: [0.0, 2.0]
     */
    float_t x;
    /**
     * Holds Y coordinate of the control point.
     * Valid Range: [0.0, 2.0]
     */
    float_t y;
    /**
     * Holds slope of the spline curve at the control point.
     * Valid Range: \f$[-2^{16}, 2^{16}]\f$
     */
    double_t slope;
} NvSiplISPSplineControlPoint;

/**
 * @brief Defines a radial transform.
 */
typedef struct {
    /**
     * Holds ellipse for radial transform.
     *
     * Coordinates of the image's top left and bottom right points are (0, 0)
     * and (width, height) respectively.
     */
    NvSiplISPEllipse radialTransform;
    /**
     * Defines spline control point for radial transfer function.
     */
    NvSiplISPSplineControlPoint controlPoints[NVSIPL_ISP_RADTF_POINTS];
} NvSiplISPRadialTF;

/**
 * @brief Holds controls for histogram statistics (HIST Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable histogram statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds offset to be applied to input data prior to bin mapping.
     * Valid Range: [-2.0, 2.0]
     */
    float_t offset;
    /**
     * Holds bin index specifying different zones in the histogram. Each zone
     * can have a different number of bins.
     * Valid Range: [1, 255]
     */
    uint8_t knees[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds \f$log_2\f$ range of the pixel values to be considered for each
     * zone. The whole pixel range is divided into NVSIPL_ISP_HIST_KNEE_POINTS
     * zones.
     * Valid Range: [0, 21]
     */
    uint8_t ranges[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds a rectangular mask for excluding pixels outside a specified area.
     *
     * The coordinates of image top left and bottom right points are (0, 0) and
     * (width, height), respectively. Set the rectangle mask to include the
     * full image (or cropped image for the case input cropping is enabled)
     * if no pixels need to be excluded.
     *
     * The rectangle settings(x0, y0, x1, y1) must follow the constraints listed below:
     * - (x0 >= 0) and (y0 >= 0)
     * - x0 and x1 should be even
     * - (x1 <= image width) and (y1 <= image height)
     * - rectangle width(x1 - x0) >= 2 and height(y1 - y0) >= 2
     */
    NvSiplRect rectangularMask;
    /**
     * Holds a Boolean to enable an elliptical mask for excluding pixels
     * outside a specified area.
     */
    NvSiplBool ellipticalMaskEnable;
    /**
     * Holds an elliptical mask for excluding pixels outside a specified area.
     *
     * Coordinates of the image top left and bottom right points are (0, 0) and
     * (width, height), respectively.
     */
    NvSiplISPEllipse ellipticalMask;
    /**
     * Holds a Boolean to enable elliptical weighting of pixels based on spatial
     * location. This can be used to compensate for lens shading when the
     * histogram is measured before lens shading correction.
     */
    NvSiplBool ellipticalWeightEnable;
    /**
     * Holds a radial transfer function for elliptical weight.
     * Valid Range: Check the declaration of @ref NvSiplISPRadialTF.
     */
    NvSiplISPRadialTF radialTF;
} NvSiplISPHistogramStats;

/**
 * @brief Holds local average and clip statistics block (LAC Stats).
 */
typedef struct {
    /**
     * Holds statistics data for each region of interest.
     */
    NvSiplISPLocalAvgClipStatsROIData data[NVSIPL_ISP_MAX_LAC_ROI];
} NvSiplISPLocalAvgClipStatsData;

/**
 * @brief Holds controls for local average and clip statistics (LAC Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable the local average and clip statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds minimum value of pixels in RGGB/RCCB/RCCC order.
     * Valid Range: [0.0, 1.0]
     */
    float_t min[NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds maximum value of pixels in RGGB/RCCB/RCCC order.
     * Valid Range: [0.0, 1.0], max >= min
     */
    float_t max[NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds a Boolean to enable an individual region of interest.
     */
    NvSiplBool roiEnable[NVSIPL_ISP_MAX_LAC_ROI];
    /**
     * Holds local average and clip windows for each region of interest.
     */
    NvSiplISPStatisticsWindows windows[NVSIPL_ISP_MAX_LAC_ROI];
    /**
     * Holds a Boolean to enable an elliptical mask for excluding pixels
     * outside a specified area for each region of interest.
     */
    NvSiplBool ellipticalMaskEnable[NVSIPL_ISP_MAX_LAC_ROI];
    /**
     * Holds an elliptical mask for excluding pixels outside specified area.
     *
     * Coordinates of the image top left and bottom right points are (0, 0) and
     * (width, height), respectively.
     */
    NvSiplISPEllipse ellipticalMask;
} NvSiplISPLocalAvgClipStats;


/**
 * @brief Holds controls for bad pixel statistics (BP Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable the bad pixel statistics block.
     * @note Bad Pixel Correction must also be enabled to get bad pixel
     *  statistics.
     */
    NvSiplBool enable;
    /**
     * Holds rectangular mask for excluding pixel outside a specified area.
     *
     * Coordinates of the image's top left and bottom right points are (0, 0)
     * and (width, height), respectively. Set the rectangle to include the
     * full image (or cropped image for the case input cropping is enabled)
     * if no pixels need to be excluded.
     *
     * Valid Range: Rectangle must be within the input image and must
     *  be a valid rectangle ((right > left) && (bottom > top)). The minimum
     *  supported rectangular mask size is 4x4.
     * Constraints: All left, top, bottom, and right coordinates must be even.
     */
    NvSiplRect rectangularMask;
} NvSiplISPBadPixelStats;


/**
 * @brief SIPL ISP Histogram Statistics Override Params
 */
typedef struct {
    /**
     * Holds a Boolean to enable histogram statistics Control block.
     */
    NvSiplBool enable;
    /**
     * Holds offset to be applied to input data prior to bin mapping.
     * Valid Range: [-2.0, 2.0]
     */
    float_t offset;
    /**
     * Holds bin index specifying different zones in the histogram. Each zone
     * can have a different number of bins.
     * Valid Range: [1, 255]
     */
    uint8_t knees[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds \f$log_2\f$ range of the pixel values to be considered for each
     * zone. The whole pixel range is divided into NVSIPL_ISP_HIST_KNEE_POINTS
     * zones.
     * Valid Range: [0, 21]
     */
    uint8_t ranges[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds a rectangular mask for excluding pixels outside a specified area.
     *
     * The coordinates of image top left and bottom right points are (0, 0) and
     * (width, height), respectively. Set the rectangle mask to include the
     * full image (or cropped image for the case input cropping is enabled)
     * if no pixels need to be excluded.
     *
     * The rectangle settings(x0, y0, x1, y1) must follow the constraints listed below:
     * - (x0 >= 0) and (y0 >= 0)
     * - x0 and x1 should be even
     * - (x1 <= image width) and (y1 <= image height)
     * - rectangle width(x1 - x0) >= 2 and height(y1 - y0) >= 2
     */
    NvSiplRect rectangularMask;
    /**
     * Holds a Boolean to enable an elliptical mask for excluding pixels
     * outside a specified area.
     */
    NvSiplBool ellipticalMaskEnable;
    /**
     * Holds an elliptical mask for excluding pixels outside a specified area.
     *
     * Coordinates of the image top left and bottom right points are (0, 0) and
     * (width, height), respectively.
     */
    NvSiplISPEllipse ellipticalMask;
    /**
     * @brief boolean flag to disable lens shading compensation for histogram statistics block
     */
    NvSiplBool disableLensShadingCorrection;
} NvSiplISPHistogramStatsOverride;

/**
 * @brief SIPL ISP Statistics Override Parameters.
 *
 * ISP Statistics settings enabled in this struct will override
 * the corresponding statistics settings provided in NITO.
 *
 * @note ISP histStats[0] and lacStats[0] statistics are consumed by internal
 * algorithms to generate new sensor and ISP settings. Incorrect usage or
 * disabling these statistics blocks would result in failure or
 * image quality degradation. Please refer to the safety manual
 * for guidance on overriding histStats[0] and lacStats[0] statistics settings.
 */
struct NvSIPLIspStatsOverrideSetting {

    /**
     * @brief boolean flag to enable histogram statistics settings override
     */
    NvSiplBool enableHistStatsOverride[2];
    /**
     * @brief Structure containing override settings for histogram statistics block
     */
    NvSiplISPHistogramStatsOverride histStats[2];
    /**
     * @brief boolean flag to enable local average clip statistics settings override
     */
    NvSiplBool enableLacStatsOverride[2];
    /**
     * @brief Structure containing override settings for local average clip statistics block
     */
    NvSiplISPLocalAvgClipStats lacStats[2];
    /**
     * @brief boolean flag to enable bad pixel statistics settings override
     */
    NvSiplBool enableBpStatsOverride[1];
    /**
     * @brief Structure containing override settings for bad pixel statistics block
     */
    NvSiplISPBadPixelStats bpStats[1];
};

/** @} */

} // namespace nvsipl

#endif /* NVSIPL_ISP_STAT_H */
