/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/**
 * \file
 * \brief <b> NvSipl ISP stat struct </b>
 */

#ifndef NVSIPL_ISP_STAT_H
#define NVSIPL_ISP_STAT_H

#include <cstdint>
#include "NvSIPLCommon.hpp"

/**
 * \brief Number of histogram bins.
 */
#define NVSIPL_ISP_HIST_BINS                   (256U)
/**
 * \brief Maximum number of color components.
 */
#define NVSIPL_ISP_MAX_COLOR_COMPONENT         (4U)

/**
 * \brief Number of histogram knee points.
 */
#define NVSIPL_ISP_HIST_KNEE_POINTS            (8U)

/**
 * \brief Number of radial transfer function control points.
 */
#define NVSIPL_ISP_RADTF_POINTS                (6U)

/**
 * \brief Maximum number of local average and clip statistic block
 * regions of interest.
 */
#define NVSIPL_ISP_MAX_LAC_ROI                 (4U)

/**
 * \brief Maximum number of input planes.
 */
#define NVSIPL_ISP_MAX_INPUT_PLANES            (3U)

/**
 * \brief Maximum matrix dimension.
 */
#define NVSIPL_ISP_MAX_COLORMATRIX_DIM         (3U)

/**
 * \brief Maximum number of windows for local average and clip in a region of
 * interest.
 */
#define NVSIPL_ISP_MAX_LAC_ROI_WINDOWS         (32U * 32U)

/**
 * \brief Maximum number of bands for flicker band statistics block.
 */
#define NVSIPL_ISP_MAX_FB_BANDS                (256U)

namespace nvsipl
{

/**
 * \defgroup NvSIPLISPStats NvSIPL ISP Stats
 *
 * @brief NvSipl ISP Defines for ISP Stat structures.
 *
 * @ingroup NvSIPLCamera_API
 */
/** @addtogroup NvSIPLISPStats
 * @{
 */

/**
 * \brief Holds bad pixel statistics (BP Stats).
 */
typedef struct {
    /**
     * Holds bad pixel count for pixels corrected upward within the window.
     */
    uint32_t highInWin;
    /**
     * Holds bad pixel count for pixels corrected downward within the window.
     */
    uint32_t lowInWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected upward within the
     * window.
     */
    uint32_t highMagInWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected downward within
     * the window.
     */
    uint32_t lowMagInWin;
    /**
     * Holds bad pixel count for pixels corrected upward outside the window.
     */
    uint32_t highOutWin;
    /**
     * Holds bad pixel count for pixels corrected downward outside the window.
     */
    uint32_t lowOutWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected upward outside
     * the window. */
    uint32_t highMagOutWin;
    /**
     * Holds accumulated pixel adjustment for pixels corrected downward outside
     * the window.
     */
    uint32_t lowMagOutWin;
} NvSiplISPBadPixelStatsData;

/**
 * \brief Holds histogram statistics (HIST Stats).
 */
typedef struct {
    /**
     * Holds histogram data for each color component in RGGB/RCCB/RCCC order.
     */
    uint32_t data[NVSIPL_ISP_HIST_BINS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds the number of pixels excluded by the elliptical mask for each
     * color component.
     */
    uint32_t excludedCount[NVSIPL_ISP_MAX_COLOR_COMPONENT];
} NvSiplISPHistogramStatsData;

/**
 * \brief Holds local average and clip statistics data for a region of interest.
 */
typedef struct {
    /**
     * Holds number of windows horizontally in one region of interest.
     */
    uint32_t numWindowsH;
    /**
     * Holds number of windows vertically in one region of interest.
     */
    uint32_t numWindowsV;
    /**
     * Holds average pixel value for each color component in each window in
     * RGGB/RCCB/RCCC order.
     */
    float_t average[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds the number of pixels excluded by the elliptical mask for each
     * color component in each window
     * in RGGB/RCCB/RCCC order.
     */
    uint32_t maskedOffCount[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds number of clipped pixels for each color component in each window in
     * RGGB/RCCB/RCCC order.
     */
    uint32_t clippedCount[NVSIPL_ISP_MAX_LAC_ROI_WINDOWS][NVSIPL_ISP_MAX_COLOR_COMPONENT];
} NvSiplISPLocalAvgClipStatsROIData;

/**
 * \brief Defines an ellipse.
 */
typedef struct {
    /**
     * Holds center of the ellipse.
     */
    NvSiplPointFloat center;
    /**
     * Holds horizontal axis of the ellipse.
     */
    uint32_t horizontalAxis;
    /**
     * Holds vertical axis of the ellipse.
     */
    uint32_t verticalAxis;
    /**
     * Holds angle of the ellipse horizontal axis from X axis in degrees in
     * clockwise direction.
     */
    float_t angle;
} NvSiplISPEllipse;

/**
 * \brief Holds controls for flicker band statistics (FB Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable flicker band statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds the offset of the first band top line.
     * @li Supported values for X coordinate of start offset: [0, input width]
     * @li Supported values for Y coordinate of start offset: [0, input height]
     * @li The X coordinate of the start offset must be an even number.
     */
    NvSiplPoint startOffset;
    /**
     * Holds count of flicker band samples to collect per frame.
     * @li Supported values: [1, 256]
     * @li Constraints: If bandCount == 256, bottom of last band
     * must align with bottom of the image.
     */
    uint16_t bandCount;
    /**
     * Holds width of single band.
     * @li Supported values: [2, input width - startOffset.x];
     *  must be an even number
     * @li Constrains: Total number of accumulated pixels must be <= 2^18
     */
    uint32_t bandWidth;
    /**
     * Holds height of single band.
     * @li Supported values: [2, input height - startOffset.y]
     * @li Constrains: Total number of accumulated pixels must be <= 2^18
     * @li Constrains: If bandCount == 256, bottom of last band
     * must align with bottom of the image.
     */
    uint32_t bandHeight;
    /**
     * Holds minimum value of pixel to include for flicker band stats.
     * @li Supported values: [0.0, 1.0]
     */
    float_t min;
    /**
     * Holds maximum value of pixel to include for flicker band stats.
     * @li Supported values: [0.0, 1.0], max >= min
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
     * and (width, height), respectively.
     *
     * @li Supported values for X coordinate of the center: [0, input width - 1]
     * @li Supported values for Y coordinate of the center: [0, input height - 1]
     * @li Supported values for horizontal axis: [17, 2 x input width]
     * @li Supported values for vertical axis: [17, 2 x input height]
     * @li Supported values for angle: [0.0, 360.0]
     */
    NvSiplISPEllipse ellipticalMask;
} NvSiplISPFlickerBandStats;

/**
 * \brief Holds flicker band statistics (FB Stats).
 */
typedef struct {
    /**
     * Holds band count.
     */
    uint32_t bandCount;
    /**
     * Holds average luminance value for each band.
     */
    float_t luminance[NVSIPL_ISP_MAX_FB_BANDS];
} NvSiplISPFlickerBandStatsData;

/**
 * \brief Defines the windows used in ISP stats calculations.
 *
 * \code
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
 * \endcode
 */
typedef struct {
    /**
     * Holds width of the window in pixels.
     */
    uint32_t width;
    /**
     * Holds height of the window in pixels.
     */
    uint32_t height;
    /**
     * Holds number of windows horizontally.
     */
    uint32_t numWindowsH;
    /**
     * Holds number of windows vertically.
     */
    uint32_t numWindowsV;
    /**
     * Holds the distance between the left edge of one window and a horizontally
     * adjacent window.
     */
    uint32_t horizontalInterval;
    /**
     * Holds the distance between the top edge of one window and a vertically
     * adjacent window.
     */
    uint32_t verticalInterval;
    /**
     * Holds the position of the top left pixel in the top left window.
     */
    NvSiplPoint startOffset;
} NvSiplISPStatisticsWindows;


/**
 * \brief Defines a spline control point.
 */
typedef struct {
    /**
     * Holds X coordinate of the control point.
     */
    float_t x;
    /**
     * Holds Y coordinate of the control point.
     */
    float_t y;
    /**
     * Holds slope of the spline curve at the control point.
     */
    double_t slope;
} NvSiplISPSplineControlPoint;

/**
 * \brief Defines a radial transform.
 */
typedef struct {
    /**
     * Holds ellipse for radial transform.
     *
     * Coordinates of image top left and bottom right points are (0, 0) &
     * (width, height) respectively.
     *
     * @li Supported values for X coordinate of the center: [0, input width - 1]
     * @li Supported values for Y coordinate of the center: [0, input height - 1]
     * @li Supported values for horizontal axis: [17, 2 x input width]
     * @li Supported values for vertical axis: [17, 2 x input height]
     * @li Supported values for angle: [0.0, 360.0]
     */
    NvSiplISPEllipse radialTransform;
    /**
     * Defines spline control point for radial transfer function.
     * @li Supported values for X coordinate of spline control point : [0.0, 2.0]
     * @li Supported values for Y coordinate of spline control point : [0.0, 2.0]
     * @li Supported values for slope of spline control point : \f$[-2^{16}, 2^{16}]\f$
     */
    NvSiplISPSplineControlPoint controlPoints[NVSIPL_ISP_RADTF_POINTS];
} NvSiplISPRadialTF;

/**
 * \brief Holds controls for histogram statistics (HIST Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable histogram statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds offset to be applied to input data prior to bin mapping.
     * @li Supported values: [-2.0, 2.0]
     */
    float_t offset;
    /**
     * Holds bin index specifying different zones in the histogram. Each zone
     * can have a different number of bins.
     * @li Supported values: [1, 255]
     */
    uint8_t knees[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds \f$log_2\f$ range of the pixel values to be considered for each
     * zone. The whole pixel range is divided into NVSIPL_ISP_HIST_KNEE_POINTS
     * zones.
     * @li Supported values: [0, 21]
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
     *
     * @li Supported values for X coordinate of the center: [0, input width - 1]
     * @li Supported values for Y coordinate of the center: [0, input height - 1]
     * @li Supported values for horizontal axis: [17, 2 x input width]
     * @li Supported values for vertical axis: [17, 2 x input height]
     * @li Supported values for angle: [0.0, 360.0]
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
     * @li Supported values: Check the declaration of \ref NvSiplISPRadialTF.
     */
    NvSiplISPRadialTF radialTF;
} NvSiplISPHistogramStats;

/**
 * \brief Holds local average and clip statistics block (LAC Stats).
 */
typedef struct {
    /**
     * Holds statistics data for each region of interest.
     */
    NvSiplISPLocalAvgClipStatsROIData data[NVSIPL_ISP_MAX_LAC_ROI];
} NvSiplISPLocalAvgClipStatsData;

/**
 * \brief Holds controls for local average and clip statistics (LAC Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable the local average and clip statistics block.
     */
    NvSiplBool enable;
    /**
     * Holds minimum value of pixels in RGGB/RCCB/RCCC order.
     * @li Supported values: [0.0, 1.0]
     */
    float_t min[NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds maximum value of pixels in RGGB/RCCB/RCCC order.
     * @li Supported values: [0.0, 1.0], max >= min
     */
    float_t max[NVSIPL_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds a Boolean to enable an individual region of interest.
     */
    NvSiplBool roiEnable[NVSIPL_ISP_MAX_LAC_ROI];
    /**
     * Holds local average and clip windows for each region of interest.
     * @li Supported values for width of the window: [2, 256] and must be an even number
     * @li Supported values for height of the window: [2, 256]
     * @li Supported values for number of the windows horizontally: [1, 32]
     * @li Supported values for number of the windows vertically: [1, 32]
     * @li Supported values for horizontal interval between windows: [max(4, window width), image ROI width]
     *  and must be an even number
     * @li Supported values for vertical interval between windows: [max(2, window height), image ROI height]
     * @li Supported values for X coordinate of start offset: [0, image ROI width-3] and must be an even number
     * @li Supported values for Y coordinate of start offset: [0, image ROI height-3]
     * @li startOffset.x + horizontalInterval * (numWindowH - 1) + winWidth <= image ROI width
     * @li startOffset.y + veritcallInterval * (numWindowV - 1) + winHeight <= image ROI height
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
     * Coordinates of the image's top left and bottom right points are (0, 0)
     *  and (width, height), respectively.
     *
     * @li Supported values for X coordinate of the center: [0, input width - 1]
     * @li Supported values for Y coordinate of the center: [0, input height - 1]
     * @li Supported values for horizontal axis: [17, 2 x input width]
     * @li Supported values for vertical axis: [17, 2 x input height]
     * @li Supported values for angle: [0.0, 360.0]
     */
    NvSiplISPEllipse ellipticalMask;
} NvSiplISPLocalAvgClipStats;


/**
 * \brief Holds controls for bad pixel statistics (BP Stats).
 */
typedef struct {
    /**
     * Holds a Boolean to enable the bad pixel statistics block.
     * \note Bad Pixel Correction must also be enabled to get bad pixel
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
     * @li Supported values: Rectangle must be within the input image and must
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
     * @li Supported values: [-2.0, 2.0]
     */
    float_t offset;
    /**
     * Holds bin index specifying different zones in the histogram. Each zone
     * can have a different number of bins.
     * @li Supported values: [1, 255]
     */
    uint8_t knees[NVSIPL_ISP_HIST_KNEE_POINTS];
    /**
     * Holds \f$log_2\f$ range of the pixel values to be considered for each
     * zone. The whole pixel range is divided into NVSIPL_ISP_HIST_KNEE_POINTS
     * zones.
     * @li Supported values: [0, 21]
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
     *
     * @li Supported values for X coordinate of the center: [0, input width - 1]
     * @li Supported values for Y coordinate of the center: [0, input height - 1]
     * @li Supported values for horizontal axis: [17, 2 x input width]
     * @li Supported values for vertical axis: [17, 2 x input height]
     * @li Supported values for angle: [0.0, 360.0]
     */
    NvSiplISPEllipse ellipticalMask;
    /**
     * @brief boolean flag to disable lens shading compensation for histogram statistics block
     */
    NvSiplBool disableLensShadingCorrection;
} NvSiplISPHistogramStatsOverride;

/**
 * @brief SIPL ISP Statistics Override Parameters.
 * ISP Statistics settings enabled in @NvSIPLIspStatsOverrideSetting will override
 * the corresponding statistics settings provided in NITO.
 *
 * note: ISP histStats[0] and lacStats[0] statistics are consumed by internal
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
