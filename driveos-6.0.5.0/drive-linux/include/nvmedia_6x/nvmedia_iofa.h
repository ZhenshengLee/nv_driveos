/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @brief <b> NVIDIA Media Interface: NvMedia Image Optical Flow Accelerator
 *            (IOFA) APIs </b>
 */

#ifndef NVMEDIA_IOFA_H
#define NVMEDIA_IOFA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvscibuf.h"

/**
 * \defgroup 6x_image_ofa_api Image Optical Flow Accelerator (IOFA)
 *
 * The NvMediaIofa object takes an uncompressed bufObj frame pair and turns them
 * into optical flow / stereo disparity estimation data.
 *
 * @ingroup 6x_nvmedia_image_top
 * @{
 */

/** @brief Major version number. */
#define NVMEDIA_IOFA_VERSION_MAJOR            1
/** @brief Minor version number. */
#define NVMEDIA_IOFA_VERSION_MINOR            1
/** @brief Patch version number. */
#define NVMEDIA_IOFA_VERSION_PATCH            0

/** @brief Maximum number of Pyramid level supported in Pyramid OF mode. */
#define NVMEDIA_IOFA_MAX_PYD_LEVEL            5U
/** @brief Maximum number of Region of Interest supported on IOFA. */
#define NVMEDIA_IOFA_MAX_ROI_SUPPORTED        32U

/**
 * Specifies the maximum number of times NvMediaIOFAInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIOFAProcessFrame().
 */
#define NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES  (8U)

/**
 * @brief Defines mode supported by IOFA Driver
 */
typedef enum
{
    /** IOFA stereo disparity mode */
    NVMEDIA_IOFA_MODE_STEREO = 0U,
    /** IOFA pyramid optical flow mode */
    NVMEDIA_IOFA_MODE_PYDOF  = 1U,
    /** OFA epipolar optical flow mode */
    /** Epipolar OF support will be added in next release */
    NVMEDIA_IOFA_MODE_EPIOF  = 2U,
} NvMediaIofaMode;

/**
 * @brief Defines the Output Grid Size.
 * \n IOFA supports variable flow vector/disparity output density.
 * \n IOFA provides single output for each input region corrosponding to grid block size.
 * \n Grid Size controls flow vector/disparity map granularity.
 * \n Application can set any grid size from the list of Grid sizes supported by IOFA.
 */
typedef enum
{
    /** Grid Size 1x1 */
    NVMEDIA_IOFA_GRIDSIZE_1X1 = 0U,
    /** Grid Size 2x2 */
    NVMEDIA_IOFA_GRIDSIZE_2X2 = 1U,
    /** Grid Size 4x4 */
    NVMEDIA_IOFA_GRIDSIZE_4X4 = 2U,
    /** Grid Size 8x8 */
    NVMEDIA_IOFA_GRIDSIZE_8X8 = 3U,
} NvMediaIofaGridSize;

/**
 * @brief Modes for pyramid SGM
 * \n Applicable to Pyramid SGM IOFA mode only.
 */
typedef enum
{
    /** All pyramid levels of a input and reference frame will be processed in single
     * @ref NvMediaIOFAProcessFrame call.
     * \n In this mode, the @ref outSurface of previous pyramid level is directly (without any processing)
     * provided as @ref pydHintSurface to process current pyramid level by IOFA driver.
     * \n     %pydHintSurface[lvl] = %outSurface[lvl+1]
     */
    NVMEDIA_IOFA_PYD_FRAME_MODE = 0U,
    /** A single pyramid level of a input and reference frame will be processed by
     *  @ref NvMediaIOFAProcessFrame API
     *  \n In this mode, the @ref outSurface of previous pyramid level can be processed/filtered by application
     *  and then provided as @ref pydHinSurface to process current pyramid level.
     *  \n @ref NvMediaIOFAProcessFrame API accept @ref pydHintSurface only in NVMEDIA_IOFA_PYD_LEVEL_MODE.
     *  \n otherwise @ref pydHintSurface is ignored by IOFA driver.
     *  \n    %pydHintSurface[lvl] = filter(%outSurface[lvl+1],....)
     */
    NVMEDIA_IOFA_PYD_LEVEL_MODE = 1U,
} NvMediaIofaPydMode;

/**
 * @brief Defines IOFA Stereo DISPARITY RANGE.
 */
typedef enum
{
    /** Maximum Stereo Disparity Range of 128 pixels */
    NVMEDIA_IOFA_DISPARITY_RANGE_128 = 0U,
    /** Maximum Stereo Disparity Range of 256 pixels */
    NVMEDIA_IOFA_DISPARITY_RANGE_256 = 1U,
} NvMediaIofaDisparityRange;

/**
 * @brief Defines NvMedia Iofa Profile Mode.
 */
typedef enum
{
    /** profiling is disabled */
    NVMEDIA_IOFA_PROFILE_DISABLED     = 0U,
    /** Profiling enabled with Async mode */
    NVMEDIA_IOFA_PROFILE_ENABLED      = 1U,
    /** Profiling enabled with sync mode */
    NVMEDIA_IOFA_PROFILE_SYNC_ENABLED = 2U,
} NvMediaIofaProfileMode;

/**
 * @brief Nvmedia Iofa Preset
 */
typedef enum
{
    /** High Quality Preset */
    NVMEDIA_IOFA_PRESET_HQ = 0U,
    /** High Performance Preset */
    NVMEDIA_IOFA_PRESET_HP = 1U,
} NvMediaIofaPreset;

/**
 * @brief NvMedia Iofa task status error codes
 */
typedef enum
{
    /** task is finished successully */
    NvSciSyncTaskStatusOFA_Success                = 0U,
    /** task status error codes */
    NvSciSyncTaskStatusOFA_Error                  = 1U,
    NvSciSyncTaskStatusOFA_Execution_Start        = 2U,
    NvSciSyncTaskStatusOFA_Error_CRC_Mismatch     = 3U,
    NvSciSyncTaskStatusOFA_Error_Timeout          = 4U,
    NvSciSyncTaskStatusOFA_Error_HW               = 5U,
    NvSciSyncTaskStatusOFA_Error_Input_TaskStatus = 6U,
    NvSciSyncTaskStatusOFA_Error_SW               = 7U,
    /** task status support is not enable */
    NvSciSyncTaskStatusOFA_Invalid                = 0XFFFFU
} NvSciSyncTaskStatusOFA;
/**
 * @brief Structure holds Epipolar information.
 * Applicable to NVMEDIA_IOFA_MODE_EPIOF mode only.
 */
typedef struct
{
    /** 3x3 Fundamental matrix in IEEE754 floating format */
    float   F_Matrix[3][3];
    /** 3x3 Homography matrix in IEEE754 floating format */
    float   H_Matrix[3][3];
    /** Epipolar X position in S17.3 format [1 Sign bit, 17-bit integer part, 3-bit fraction part] */
    int32_t epipole_x;
    /** Epipolar Y position in S17.3 format [1 Sign bit, 17-bit integer part, 3-bit fraction part] */
    int32_t epipole_y;
    /** Direction: 0/1 - Search towards / away from epipole */
    uint8_t direction;
} NvMediaIofaEpipolarInfo;

/**
 * @brief Holds Co-ordinates for Region of Interest.
 *  \n    ROI width  = %endX - %startX + 1;
 *  \n    ROI height = %endY - %startY + 1;
 *  \n Constraints for ROI programming:
 *  \n   %startX needs to align to 32 pixels and width needs to be greater than 32 pixels.
 *  \n   %startY needs to align to 16 pixels and height needs to be greater than 16 pixels.
 *  \n If ROI coordinates do not follow the above constraints, driver will try to align ROI params.
 */
typedef struct
{
    /** ROI top-left x index (in pixel unit) */
    uint16_t startX;
    /** ROI top-left y index (in pixel unit) */
    uint16_t startY;
    /** ROI bottom-right index (in pixel unit). Only endX - 1 co-ordinate is included in ROI */
    uint16_t endX;
    /** ROI bottom-right index (in pixel unit). Only endY - 1 co-ordinate is included in ROI */
    uint16_t endY;
} NvMediaIofaROIRectParams;

/**
 * @brief Structure holds ROI information
 */
typedef struct
{
    /** Number of ROIs */
    uint32_t                numOfROIs;
    /** Array of ROI co-ordinates */
    NvMediaIofaROIRectParams rectROIParams[NVMEDIA_IOFA_MAX_ROI_SUPPORTED];
} NvMediaIofaROIParams;

/**
 * @brief Nvmedia Iofa Capability structure
 */
typedef struct
{
    /** min width supported by IOFA driver */
    uint16_t minWidth;
    /** min height supported by IOFA driver */
    uint16_t minHeight;
    /** max width supported by IOFA driver */
    uint16_t maxWidth;
    /** max height supported by IOFA driver */
    uint16_t maxHeight;
} NvMediaIofaCapability;

/**
 * @brief Holds an IOFA object created and returned by NvMediaIOFACreate().
 * \n Application should use the members of IOFA object (%width, %height, %outWidth, %outHeight,
 * %outputFormat, %costFormat ) to allocate input and reference pyramid, output and cost surface.
 * \n Application can access these members after calling API @ref NvMediaIOFAInit.
 */
typedef struct NvMediaIofa
{
    /** An Opaque pointer for internal use */
    struct NvMediaIofaPriv *ofaPriv;
} NvMediaIofa;

/**
 * @brief Holds IOFA Initialization API parameters.
 */
typedef struct
{
    /** Iofa Mode type
     *  \n One of ofa mode types provided in @ref NvMediaIofaMode
     */
    NvMediaIofaMode           ofaMode;
    /** Number of input pyramid level
     *  \n Number of input pyramid levels used for pyramid optical flow estimation
     *  \n %ofaPydLevel variable is not used for other pyramid mode.
     */
    uint8_t                   ofaPydLevel;
    /** Input width
     *  \n Valid range of %width is between 32 to 8192.
     *  \n Width of the input and reference @ref NvSciBufObj that needs to be processed
     *  should be equal to the value which is passed here.
     */
    uint16_t                 width[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** Input height
     *  \n Valid range of %height is between 32 to 8192.
     *  \n Height of the input and reference @ref NvSciBufObj that needs to be processed
     *  should be equal to the value which is passed here.
     */
    uint16_t                 height[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** OFA Grid size  per pyramid level
     * \n One of the value from @ref NvMediaIofaGridSize
     */
    NvMediaIofaGridSize       gridSize[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** Output width
     *  \n Valid range of %outWidth is between 4 to 8192.
     *  \n %outWidth is calculated based on width and gridsize for that level
     *  \n %outwidth = (%width + (1 << %gridSize) - 1)) >> %gridSize
     *  Application should set %outWidth based on above equation
     */
    uint16_t                  outWidth[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** Output height
     *  \n Valid range of %outHeight is between 4 to 8192.
     *  \n %outHeight is calculated based on height and gridsize for that level
     *  \n %outHeight = (%height + (1 << %gridSize) - 1)) >> %gridSize
     *  Application should set %outHeight based on above equation
     */
    uint16_t                  outHeight[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** Stereo Disparity Range
     * \n One of the value from @ref NvMediaIofaDisparityRange
     */
    NvMediaIofaDisparityRange dispRange;
    /** Pyramid SGM Mode
     * \n One of the value from @ref NvMediaIofaPydMode
     */
    NvMediaIofaPydMode        pydMode;
    /** Profiling Support
     * \n One of the value from @ref NvMediaIofaProfileMode
     */
    NvMediaIofaProfileMode    profiling;
    /** Input and Output Surface in VPR (Not supported)
     * \n true  use VPR memory for input and output surface
     * \n false use normal memory for input and output surface
     */
    bool                      vprMode;
    /** Ofa Preset
     * \n One of the value from @ref NvMediaIofaPreset
     */
    NvMediaIofaPreset         preset;

} NvMediaIofaInitParams;

/**
 * @brief Holds SGM parameters
 *  \n TBD: Add more details about SGM Params with input range.
 */
typedef struct
{
    /** SGM P1 penalty value */
    uint8_t     penalty1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** SGM P2 penalty value */
    uint8_t     penalty2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** SGM adaptive P2
     * \n 0: To disable SGM adaptive P2 feature
     * \n 1: To enable SGM adaptive P2 feature
     */
    bool adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** SGM alpha value for adaptive P2.
     *  \n The valid range of variable is from 0 to 3.
     */
    uint8_t     alphaLog2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** SGM diagonal directions
     * \n 0: To disable the diagonal directions
     * \n 1: To enable the diagonal directions
     */
    bool enableDiag[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** Number of SGM passes.
     *  \n The valid range of variable is from 1 to 3.
     */
    uint8_t     numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaSGMParams;

/**
 * @brief Holds pointers to NvMedia bufObjs containing input and output surfaces.
 * \n In case of Pyramid OF Mode, color format of all bufObjs from @ref inputFrame and @ref refFrame
 * pyramid except bottom layer bufObj (with index 0) must be YUV400.
 * \n Bottom layerof pyramid is input and reference frame at original resolution which can be have
 * any color format.
 * \n This is done to avoid any restriction on width and height of pyramid bufObjs.
 * \n This constraint will be removed in next release and Application will be able to allocate
 * pyramid bufObjs with any color format.
 */
typedef struct
{
    /** inputSurface
     * <b> Pyramid Optical Flow Processing: </b> Array of pointers to input @ref NvSciBufObj at time T+1
     *        for number of levels returned in parameter @ref ofaPydLevel in @ref NvMediaIofa Object
     * \n <b> Stereo Disparity Processing: </b> An input NvSciBufObj containing the rectified left view.
     * \n It must be have the same width and height as provided in @ref NvMediaIofaInitParams
     * \n In case of Pyramid Optical Flow Processing: Resolution of higher levels should also
     *    be the same as provided in @ref NvMediaIofa Object
     * \n Surface is allocated through a call to @ref NvSciBufObjAlloc().
     * \inputrange A valid NvSciBufObj
     */
    NvSciBufObj inputSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** referenceSurface
     *  <b> Pyramid Optical Flow Processing: </b> Array of pointers to input @ref NvSciBufObj at time T
     *        for number of levels returned in parameter @ref ofaPydLevel in @ref NvMediaIofa Object
     * \n <b> Stereo Disparity Processing: </b> An input NvSciBufObj containing the rectified right view.
     * \n It must be have the same width and height as provided in @ref NvMediaIofaInitParams
     * \n In case of Pyramid Optical Flow Processing: Resolution of higher levels should also
     *    be the same as provided in @ref NvMediaIofa Object
     * \n Surface is allocated through a call to NvSciBufObjAlloc().
     * \inputrange A valid NvSciBufObj
     */
    NvSciBufObj refSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** pydHintSurface
     * \n <b> Pyramid Optical Flow Processing: </b> Array of pointers to NvSciBufObj containing pyd hint surface
     * \n for number of pyramid levels equal to parameter @ref ofaPydLevel in @ref NvMediaIofa Object.
     * \n <b> Stereo Disparity Processing: </b> Not Applicable
     * \n \a pydHintSurface must be the same type of @ref outputformat and width and height should be
     *    also same as @ref outWidth and @ref outHeight or @ref outWidth/2 and @ref outHeight/2
     *    for particular level provided in @ref NvMediaIofaInitParams
     * \n Frame is allocated through a call to @ref NvSciBufObjCreate API.
     * \n Non-NULL - valid pointer address.
     */
    NvSciBufObj pydHintSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** outSurface
     * An @ref NvSciBufObj where Optical flow and stereo disparity surface will be populated after IOFA processing.
     * As the processing in not synchronous \a outSurface will not be populated as soon as @ref NvMediaIOFAProcessFrame API returns.
     * Clients can optionally wait on the returned EOF NvSciSyncObj obtained via
     * @ref NvMediaIOFAGetEOFNvSciSyncObj API to know the status of processing, once this API returns.
     * \n @ref outSurface  must be of the same type as returned by @ref NvMediaIOFAInit API in @ref NvMediaIofa object
     * \n \a outSurface is allocated through a call to NvSciBufObjAlloc().
     * \n Optical flow output is in S10.5 format. This means that the MSB represents the sign of the flow vector,
     * next 10bits represent the integer value and the last 5 bits represent the fractional value.
     * Maximum range integer values is [-1024, 1023] and for fractional part all 5 bits are valid.
     * For Optical flow there are 2 components [X, Y] per pixel.
     * \n For Stereo mode, Disparity is also in S10.5 format with single component [X].
     * Output corresponds to left view and absolute value of output will provide actual disparity.
     * \inputrange A valid NvSciBufObj
     */
    NvSciBufObj outSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    /** costSurface
     * An @ref NvSciBufObj where cost surface will be populated as a result of IOFA processing.
     * This parameter is optional and can be sent as NULL if cost is not required by the application.
     * As the processing in not synchronous \a costSurface will not be populated as soon as
     * @ref NvMediaIOFAProcessFrame API returns.
     * Clients can optionally wait on the returned EOF NvSciSyncObj obtained via
     * @ref NvMediaIOFAGetEOFNvSciSyncObj API to know the status of processing, once this API returns.
     * \n @ref costSurface must be of the same type as returned by @ref NvMediaIOFAInit API in @ref NvMediaIofa object
     * \n \a outSurface is allocated through a call to NvSciBufObjAlloc()
     * For cost surface, range of values for each pixel is [0, 255]
     * \inputrange A valid NvSciBufObj
     */
    NvSciBufObj costSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaBufArray;

/**
 * @brief Parameters related to input pyramid hint surface.
 */
typedef struct
{
    /** IOFA HW supported input hint mv magnitude scaling.
     * \n false : IOFA HW hint mv = 1 x input hint mv magnitude
     * \n true  : IOFA HW hint mv = 2 x input hint mv magnitude
     */
    bool pydHintMagnitudeScale2x;
    /** IOFA HW supported input hint upsampling in X direction
     * \n false : input hint surface width is same as output surface width
     * \n true  : input hint surface width is 1/2 of output surface width
     */
    bool pydHintWidth2x;
    /** IOFA HW supported input hint upsampling in Y direction
     * \n false : input hint surface height is same as output surface height
     * \n true  : input hint surface height is 1/2 of output surface height
     */
    bool pydHintHeight2x;
} NvMediaIofaPydHintParams;

/**
 * @brief Holds IOFA Process Frame API parameters.
 */
typedef struct
{
    /** Enable right view disparity map. Applicable only for Stereo mode
     * \n false to disable right view disparity generation
     * \n true to enable right view disparity generation
     * \n If enabled, stereo disparity is estimated for right view w.r.t. left view
     */
    bool                    rightDispMap;
    /** Current level to process in Pyd SGM LEVEL Mode
     *  \n valid range of values is between 0 to ofaPydLevel-1.
     */
    uint8_t                 currentPydLevel;
    /** noop flag
     * \n true   Avoid ofa processing and signal frame done
     * \n false  Normal ofa processing (default)
     */
    bool                     noopMode;
    /** Pyramid hints parameters */
    NvMediaIofaPydHintParams pydHintParams;
} NvMediaIofaProcessParams;

/**
 * @brief IOFA Profile Data structure
 */
typedef struct
{
    /** Indicate if profile data is valid or not */
    bool        validProfData;
    /** ofa sw driver execution time in micro sec */
    uint32_t    swTimeInUS;
    /** ofa hw processing time in micro sec */
    uint32_t    hwTimeInUS;
    /** ofa hw processing time with sync mode in micro sec  */
    uint32_t    syncWaitTimeInUS;
} NvMediaIofaProfileData;

/**
 * @brief Retrieves the version information for the NvMedia IOFA library.
 *
 * @pre  None
 * @post None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 *
 * @param[out] version
 *      A pointer to a @ref NvMediaVersion structure filled by the IOFA library.
 * @return ::NvMediaStatus, the completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if the \a version pointer is NULL.
 */
NvMediaStatus
NvMediaIOFAGetVersion (
    NvMediaVersion *version
);

/**
 * @brief Creates an @ref NvMediaIofa object that can compute optical flow or
 * stereo disparity using two bufObjs.
 *
 * @pre NvMediaIOFAGetVersion()
 * @post NvMediaIofa object is created
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @return Created NvMediaIofa estimator handle if successful, or NULL otherwise.
 */
NvMediaIofa *
NvMediaIOFACreate (
    void
);

/**
 * @brief Initializes the parameters for optical flow and stereo estimation
 *
 * @pre NvMediaIOFAGetVersion()
 * @pre NvMediaIOFACreate()
 * @post NvMediaIofa object is returned with initialized parameters
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] ofaPubl
 *      A pointer to the @ref NvMediaIofa estimator to use.
 *      Non-NULL - valid pointer address
 * @param[in] initParams
 *      A pointer to a structure that specifies initialization parameters.
 *      Non-NULL - valid address.
 *      \n Ranges specific to each member in the structure can be found in
 *      @ref NvMediaIofaInitParams.
 * @param[in] maxInputBuffering
 *      Maximum number of NvMediaIOFAProcessFrame() operations that can be
 *      queued by NvMediaIofa. \n If more than @a maxInputBuffering operations
 *      are queued, %NvMediaIOFAProcessFrame() returns an error to indicate
 *      insufficient buffering.
 *      \n The values between 1 to 8, in increments of 1
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAInit (
    NvMediaIofa                 *ofaPubl,
    const NvMediaIofaInitParams *initParams,
    const uint8_t              maxInputBuffering
);

/**
 * @brief Performs IOFA estimation on a specified frame pair.
 *
 * Estimation is based on the difference between @a refFrame and @a inputframe.
 * The output of Optical Flow processing is motion vectors [X, Y Components],
 * and that of Stereo Disparity processing is a disparity surface [X component].
 *
 * @pre NvMediaIOFAInit()
 * @pre NvMediaIOFARegisterNvSciBufObj()
 * @pre NvMediaIOFARegisterNvSciSyncObj()
 * @pre NvMediaIOFASetNvSciSyncObjforEOF()
 * @pre NvMediaIOFAInsertPreNvSciSyncFence()
 * @post Optical Flow Accelerator estimation task is submitted
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] ofaPubl
 *      A pointer to the @ref NvMediaIOFA estimator to use.
 *      \n Non-NULL - valid pointer address
 * @param[in] surfArray
 *      A pointer to a structure that specifies input and output surface parameters.
 *      \n Non-NULL - valid address.
 *      \n Ranges specific to each member in the structure can be found in
 *      @ref NvMediaIofaBufArray.
 * @param[in] processParams
 *      A pointer to a structure that specifies process frame parameters.
 *      Non-NULL - valid address.
 *      \n Ranges specific to each member in the structure can be found in
 *      @ref NvMediaIofaProcessParams.
 * @param[in] pROIParams
 *      A pointer to a structure that specifies ROI parameters.
 *      \n pROIParams are optional argument and can be NULL if ROI is not set by App.
 *      \n Ranges specific to each member in the structure can be found in
 *      @ref NvMediaIofaROIParams.
 * @param[in] pEpipolarInfo
 *      A pointer to a structure that specifies Epipolar info parameters.
 *      \n pEpipolarInfo is required argument only when /ref ofaMode in /ref NvMediaIofaInitParams
 *      is set to /ref NVMEDIA_IOFA_MODE_EPIOF.
 *      \n Ranges specific to each member in the structure can be found in
 *      @ref NvMediaIofaEpipolarInfo.
 *      \n Epipolar OF support will be added in next release. Ignored right now in driver.
 *      \n Set it to NULL till EpiPolar OF support is enabled.
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAProcessFrame (
    const NvMediaIofa              *ofaPubl,
    const NvMediaIofaBufArray      *pSurfArray,
    const NvMediaIofaProcessParams *processParams,
    const NvMediaIofaEpipolarInfo  *pEpiInfo,
    const NvMediaIofaROIParams     *pROIParams
);

/**
 * @brief Destroys the created @ref NvMediaIofa object and frees associated resources.
 *
 * @pre NvMediaIOFAUnregisterNvSciSyncObj()
 * @pre NvMediaIOFAUnregisterNvSciBufObj()
 * @post NvMediaIofa object is destroyed
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] ofaPubl
 *      Pointer to the @ref NvMediaIofa object to destroy, returned by NvMediaIOFACreate().
 *      Non-NULL - valid pointer address
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFADestroy (
    const NvMediaIofa *ofaPubl
);

/**
 * @brief Registers an @ref NvSciBufObj for use with an @ref NvMediaIofa handle.
 * The NvMediaIofa handle maintains a record of all the bufObjs registered using
 * this API.
 *
 * This is a mandatory API which needs to be called before NvMediaIOFAProcessFrame()
 * \n All %NvMediaIOFANvSciBufRegister() API calls must be made before first
 * %NvMediaIOFAProcessFrame() API call.
 * Registration of the buffer is done with the same access permission as
 * that of the NvSciBufObj being registered. NvSciBufObj that need to be
 * registered with a reduced permission (Eg: Input buffer accesses being set to
 * read-only) can be done so by first duplicating the NvSciBufObj using
 * NvSciBufObjDupWithReducePerm() followed by a call the register the duplicated
 * NvSciBufObj.
 *
 *
 * Maximum of 192 NvSciBufObj handles can be registered using %NvMediaIOFARegisterNvSciSyncObj() API.
 *
 * @pre NvMediaIOFAInit()
 * @pre NvMediaIOFARegisterNvSciSyncObj()
 * @post NvSciBufObj is registered with NvMediaIofa object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[in] ofaPubl
 *      @ref NvMediaIofa handle.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj
 *      An NvSciBufObj object.
 *      \inputrange A valid NvSciBufObj

 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if ofa, bufObj or accessMode is invalid.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR in following cases:
 *          - User registers more than 192 bufObjs.
 *          - User registers same bufObj with more than one accessModes.
 *          - User registers same bufObj multiple times.
 *
 **/
NvMediaStatus
NvMediaIOFARegisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj        bufObj
);

/**
 * @brief Un-registers @ref NvSciBufObj which was previously registered with
 * @ref NvMediaIofa using NvMediaIOFARegisterNvSciBufObj().
 *
 * For all NvSciBufObj handles registered with NvMediaIofa using
 * %NvMediaIOFARegisterNvSciBufObj() API, %NvMediaIOFAUnregisterNvSciBufObj()
 * must be called before calling NvMediaIOFADestroy() API.
 * For unregistration to succeed, it should be ensured that none of the
 * submitted tasks on the bufObj are pending prior to calling
 * %NvMediaIOFAUnregisterNvSciBufObj(). In order to ensure this,
 * %NvMediaIOFAUnregisterNvSciSyncObj() should be called prior to this API on
 * all registered NvSciSyncObj. Post this NvMediaIOFAUnregisterNvSciBufObj() can
 * be successfully called on a valid NvSciBufObj.
 *
 * For deterministic execution of %NvMediaIOFAProcessFrame() API,
 * %NvMediaIOFAUnregisterNvSciBufObj() must be called only after last
 * %NvMediaIOFAProcessFrame() call.
 *
 * @pre NvMediaIOFAUnregisterNvSciSyncObj() [verify that processing is complete]
 * @post NvSciBufObj is un-registered from NvMediaIofa object
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @param[in] ofaPubl
 *      @ref NvMediaIofa handle.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] bufObj
 *      An NvSciBufObj object.
 *      \inputrange A valid NvSciBufObj
 * @return ::NvMediaStatus, the completion status of operation:
 * - ::NVMEDIA_STATUS_OK if successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if ofa or bufObj is invalid
 *           %NvMediaIOFARegisterNvSciBufObj() API.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR in following cases:
 *          - User unregisters an NvSciBufObj which is not previously
 *            registered using %NvMediaIOFARegisterNvSciBufObj() API.
 *          - User unregisters an NvSciBufObj multiple times.
 **/
NvMediaStatus
NvMediaIOFAUnregisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj       bufObj
);

/**
 * @brief Get the SGM configuration parameters being used
 *
 * @pre NvMediaIOFAInit()
 * @post SGM params are returned in structure NvMediaIofaSGMParams
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] ofaPubl
 *      Pointer to the @ref NvMediaIofa object to use.
 *      Non-NULL - valid pointer address
 * @param[out] pSGMParams
 *      A pointer to a structure that specifies SGM parameters.
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the function call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAGetSGMConfigParams (
    const NvMediaIofa    *ofaPubl,
    NvMediaIofaSGMParams *pSGMParams
);

/**
 * @brief Set the SGM configuration parameters to be used
 *
 * @pre NvMediaIOFAInit()
 * @post SGM params are set in IOFA driver and will be used for next frame processing.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] ofaPubl
 *      Pointer to the @ref NvMediaIofa object to use.
 *      Non-NULL - valid pointer address
 * @param[in] pSGMParams
 *      A pointer to a structure that specifies SGM parameters.
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the function call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFASetSGMConfigParams (
    const NvMediaIofa          *ofaPubl,
    const NvMediaIofaSGMParams *pSGMParams
);

/**
 * @brief Get IOFA Profile Data
 *
 * IOFA driver stores profiling data of last 16 frames.
 * This function returns profile data oldest frame stored in profile queue.
 *
 *
 * @pre NvMediaIOFAProcessFrame()
 * @pre NvSciBufObjGetStatus() [verify that processing is complete]
 * @post profile data is returned in structure %NvMediaIofaProfileData
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * <b>Considerations for Safety</b>:
 * - Operation Mode: Runtime
 *
 * @param[in] ofaPubl
 *      Pointer to the @ref NvMediaIofa object to use.
 *      Non-NULL - valid pointer address
 * @param[out] pProfData
 *      A pointer to a structure that contains profile data
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAGetProfileData (
    const NvMediaIofa      *ofaPubl,
    NvMediaIofaProfileData *pProfData
);

/**
 * @brief Get IOFA Capability
 *
 * This function returns ofa hw capabilities.
 *
 * @pre  NvMediaIOFACreate
 * @post hw capabilities are returned in structure %NvMediaIofaCapability
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * <b>Considerations for Safety</b>:
 * - Operation Mode: Init
 *
 * @param[in] ofaPubl
 *      Pointer to the @ref NvMediaIofa object to use.
 *      Non-NULL - valid pointer address
 * @param[in] mode
 *      one of the value from @ref NvMediaIofaMode.
 * @param[out] pCapability
 *      A pointer to a structure that contains capability data
 *
 * @return The completion status of the operation:
 * - ::NVMEDIA_STATUS_OK if the call is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAGetCapability (
    const NvMediaIofa     *ofaPubl,
    const NvMediaIofaMode mode,
    NvMediaIofaCapability *pCapability
);

/**
 * @brief Fills the NvMediaIofa specific NvSciBuf attributes which than then be
 * used to allocate an @ref NvSciBufObj that NvMediaIofa can consume.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciBufAttrList
 * created by the caller by a call to @ref NvSciBufAttrListCreate.
 *
 * @pre NvMediaIOFAGetVersion()
 * @post NvSciBufAttrList populated with NvMediaIofa specific NvSciBuf
 *      attributes. The caller can then set attributes specific to the type of
 *      surface, reconcile attribute lists and allocate an NvSciBufObj.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * @param[out] attrlist A pointer to an %NvSciBufAttrList structure where
 *                NvMediaIofa places the NvSciBuf attributes.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL
 */
NvMediaStatus
NvMediaIOFAFillNvSciBufAttrList (
    NvSciBufAttrList attrlist
);

/**
 * @brief Fills the NvMediaIofa specific NvSciSync attributes.
 *
 * This function assumes that @a attrlist is a valid @ref NvSciSyncAttrList.
 *
 * This function sets the public attribute:
 * - ::NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * @pre NvMediaIOFACreate()
 * @post NvSciSyncAttrList populated with NvMediaIofa specific NvSciSync
 *        attributes
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Can be NULL or Non-NULL valid pointer address
 * @param[out] attrlist A pointer to an %NvSciSyncAttrList structure where
 *                NvMedia places NvSciSync attributes.
 * @param[in] clienttype Indicates whether the NvSciSyncAttrList requested for
 *                an %NvMediaIofa signaler or an %NvMediaIofa waiter.
 *      \inputrange Entries in @ref NvMediaNvSciSyncClientType enumeration
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a attrlist is NULL,
 *         or any of the public attributes listed above are already set.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough
 *         memory for the requested operation.
 */
NvMediaStatus
NvMediaIOFAFillNvSciSyncAttrList (
    const NvMediaIofa          *ofaPubl,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * @brief Registers an @ref NvSciSyncObj with NvMediaIofa.
 *
 * Every NvSciSyncObj (even duplicate objects) used by %NvMediaIofa
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIOFASetNvSciSyncObjforEOF(), NvMediaIOFAGetEOFNvSciSyncFence(), or
 * NvMediaIOFAUnregisterNvSciSyncObj().
 *
 * For a given %NvMediaIofa handle,
 * one NvSciSyncObj can be registered as one @ref NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * @pre NvMediaIOFAFillNvSciSyncAttrList()
 * @post NvSciSyncObj registered with NvMediaIofa
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] syncobjtype Determines how @a nvscisync is used by @a ofaPubl.
 *      \inputrange Entries in @ref NvMediaNvSciSyncObjType enumeration
 * @param[in] nvscisync The NvSciSyncObj to be registered with @a ofaPubl.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is NULL or
 *         @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 *          only NVMEDIA_EOFSYNCOBJ and NVMEDIA_PRESYNCOBJ supported.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *         compatible NvSciSiyncObj which %NvMediaIofa can support.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a ofaPubl
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIOFARegisterNvSciSyncObj (
    const NvMediaIofa       *ofaPubl,
    NvMediaNvSciSyncObjType syncobjtype,
    NvSciSyncObj            nvscisync
);

/**
 * @brief Unregisters an @ref NvSciSyncObj with NvMediaIofa.
 *
 * Every %NvSciSyncObj registered with %NvMediaIofa by
 * NvMediaIOFARegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIOFAUnregisterNvSciBufObj to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * @ref NvMediaIOFAProcessFrame() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still
 * in use by any %NvMediaIOFAProcessFrame() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIOFAProcessFrame() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIOFAGetEOFNvSciSyncFence().
 *
 * @pre NvMediaIOFAProcessFrame()
 * @pre NvSciSyncFenceWait() [verify that processing is complete]
 * @post NvSciSyncObj un-registered with NvMediaIofa
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisync An NvSciSyncObj to be unregistered with @a ofaPubl.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if ofaPubl is NULL, or
 *        @a nvscisync is not registered with @a ofaPubl.
 * - ::NVMEDIA_STATUS_PENDING if the @ref NvSciSyncObj is still in use, i.e.,
 *        the submitted task is still in progress. In this case, the application
 *        can choose to wait for operations to complete on the output surface
 *        using NvSciSyncFenceWait() or re-try the
 *        %NvMediaIOFAUnregisterNvSciBufObj() API call, until the status
 *        returned is inot @ref NVMEDIA_STATUS_PENDING.
 * - ::NVMEDIA_STATUS_INVALID_STATE  The function was called in incorrect system state.
 * - ::NVMEDIA_STATUS_ERROR if @a ofaPubl was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIOFAUnregisterNvSciSyncObj (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      nvscisync
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an EOF
 * @ref NvSciSyncFence.
 *
 * To use NvMediaIOFAGetEOFNvSciSyncFence(), the application must call
 * %NvMediaIOFASetNvSciSyncObjforEOF() before it calls NvMediaIOFAProcessFrame().
 *
 * %NvMediaIOFASetNvSciSyncObjforEOF() currently may be called only once before
 * each call to %NvMediaIOFAProcessFrame(). The application may choose to call
 * this function only once before the first call to %NvMediaIOFAProcessFrame().
 *
 * @pre NvMediaIOFARegisterNvSciSyncObj()
 * @post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                           associated with EOF @ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is NULL, or if @a nvscisyncEOF
 *         is not registered with @a ofaPubl as either type
 *         @ref NVMEDIA_EOFSYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIOFASetNvSciSyncObjforEOF (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      nvscisyncEOF
);

/**
 * @brief Sets an @ref NvSciSyncFence as a prefence for an
 * NvMediaIOFAProcessFrame() %NvSciSyncFence operation.
 *
 * You must call %NvMediaIOFAInsertPreNvSciSyncFence() before you call
 * %NvMediaIOFAProcessFrame(). The %NvMediaIOFAProcessFrame() operation is
 * started only after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIOFAInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIOFAProcessFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * \endcode
 * the %NvMediaIOFAProcessFrame() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of @ref NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES prefences
 * by calling %NvMediaIOFAInsertPreNvSciSyncFence() before %NvMediaIOFAProcessFrame().
 * After the call to %NvMediaIOFAProcessFrame(), all NvSciSyncFences previously
 * inserted by %NvMediaIOFAInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent %NvMediaIOFAProcessFrame() calls.
 *
 * @pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * @post Pre-NvSciSync fence is set
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] prenvscisyncfence A pointer to %NvSciSyncFence.
 *      \inputrange Non-NULL - valid pointer address
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is not a valid %NvMediaIofa
 *     handle, or @a prenvscisyncfence is NULL, or if @a prenvscisyncfence was not
 *     generated with an @ref NvSciSyncObj that was registered with @a ofaPubl as
 *     either @ref NVMEDIA_PRESYNCOBJ or @ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - ::NVMEDIA_STATUS_NOT_SUPPORTED if %NvMediaIOFAInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES times
 *     with the same @a ofaPubl handle before an %NvMediaIOFAProcessFrame() call.
 */
NvMediaStatus
NvMediaIOFAInsertPreNvSciSyncFence (
    const NvMediaIofa    *ofaPubl,
    const NvSciSyncFence *prenvscisyncfence
);

/**
 * @brief Gets EOF @ref NvSciSyncFence for an NvMediaIOFAProcessFrame() operation.
 *
 * The EOF %NvSciSyncFence associated with an %NvMediaIOFAProcessFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIOFAProcessFrame() operation has finished.
 *
 * This function returns the EOF %NvSciSyncFence associated
 * with the last %NvMediaIOFAProcessFrame() call. %NvMediaIOFAGetEOFNvSciSyncFence()
 * must be called after an %NvMediaIOFAProcessFrame() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIOFAProcessFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * nvmstatus = NvMediaIOFAGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * %NvMediaIOFAProcessFrame() operation has finished.
 *
 * @pre NvMediaIOFASetNvSciSyncObjforEOF()
 * @pre NvMediaIOFAProcessFrame()
 * @post EOF NvSciSync fence for a submitted task is obtained
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order to
 * achieve synchronization between the engines
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is not a valid %NvMediaIofa
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a ofaPubl as type @ref NVMEDIA_EOFSYNCOBJ or
 *         @ref NVMEDIA_EOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIOFAProcessFrame() was called.
 */
NvMediaStatus
NvMediaIOFAGetEOFNvSciSyncFence (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/**
 * @brief Specifies the @ref NvSciSyncObj to be used for an SOF
 * @ref NvSciSyncFence.
 *
 * This function is not supported.
 *
 * To use NvMediaIOFAGetSOFNvSciSyncFence(), the application must call
 * %NvMediaIOFASetNvSciSyncObjforSOF() before it calls NvMediaIOFAProcessFrame().
 *
 * %NvMediaIOFASetNvSciSyncObjforSOF() currently may be called only once before
 * each call to %NvMediaIOFAProcessFrame(). The application may choose to call this
 * function only once before the first call to %NvMediaIOFAProcessFrame().
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] nvscisyncSOF A registered NvSciSyncObj which is to be
 *                           associated with SOF @ref NvSciSyncFence.
 *      \inputrange A valid NvSciSyncObj
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is NULL, or if @a nvscisyncSOF
 *         is not registered with @a ofaPubl as either type
 *         @ref NVMEDIA_SOFSYNCOBJ or @ref NVMEDIA_SOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIOFASetNvSciSyncObjforSOF (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      nvscisyncSOF
);

/**
 * @brief Gets SOF @ref NvSciSyncFence for an NvMediaIOFAProcessFrame() operation.
 *
 * This function is not supported.
 *
 * The SOF %NvSciSyncFence associated with an %NvMediaIOFAProcessFrame() operation
 * is an %NvSciSyncFence. Its expiry indicates that the corresponding
 * %NvMediaIOFAProcessFrame() operation has started.
 *
 * This function returns the SOF %NvSciSyncFence associated
 * with the last %NvMediaIOFAProcessFrame() call. %NvMediaIOFAGetSOFNvSciSyncFence()
 * must be called after an %NvMediaIOFAProcessFrame() call.
 *
 * @pre N/A
 * @post N/A
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * @param[in] ofaPubl A pointer to the NvMediaIofa object.
 *      \inputrange Non-NULL - valid pointer address
 * @param[in] sofnvscisyncobj    An SOF NvSciSyncObj associated with the
 *                                 NvSciSyncFence which is being requested.
 *      \inputrange A valid NvSciSyncObj
 * @param[out] sofnvscisyncfence A pointer to the SOF NvSciSyncFence.
 *
 * @return ::NvMediaStatus The status of the operation.
 * Possible values are:
 * - ::NVMEDIA_STATUS_OK if the function is successful.
 * - ::NVMEDIA_STATUS_BAD_PARAMETER if @a ofaPubl is not a valid %NvMediaIofa
 *         handle, @a sofnvscisyncfence is NULL, or @a sofnvscisyncobj is not
 *         registered with @a ofaPubl as type @ref NVMEDIA_SOFSYNCOBJ or
 *         @ref NVMEDIA_SOF_PRESYNCOBJ.
 * - ::NVMEDIA_STATUS_ERROR if the function was called before
 *         %NvMediaIOFAProcessFrame() was called.
 */
NvMediaStatus
NvMediaIOFAGetSOFNvSciSyncFence (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      sofnvscisyncobj,
    NvSciSyncFence    *sofnvscisyncfence
);

/*
 * \defgroup 6x_history_nvmedia_iofa History
 * Provides change history for the NvMedia IOFA API.
 *
 * \section 6x_history_nvmedia_iofa Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 *
 */

/** @} */
#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif // NVMEDIA_IOFA_H

