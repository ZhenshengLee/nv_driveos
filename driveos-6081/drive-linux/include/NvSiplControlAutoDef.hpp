/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/*  NVIDIA SIPL Control Auto Definitions */

#ifndef NVSIPLCONTROLAUTODEF_HPP
#define NVSIPLCONTROLAUTODEF_HPP

#include "NvSIPLISPStat.hpp"
#include "NvSIPLCDICommon.h"

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Auto Control Settings - @ref NvSIPLAutoControl </b>
 *
 */

namespace nvsipl{

/** @addtogroup NvSIPLAutoControl
 * @{
 */

/**
  *  Defines types of SIPL Control Auto plug-ins.
  */
enum PluginType {
   NV_PLUGIN = 0,  /**< NVIDIA plug-in */
   CUSTOM_PLUGIN0, /**< Custom plug-in 0 */
   MAX_NUM_PLUGINS /**< Maximum number of plug-ins supported. */
};

/**
 * \brief Sensor settings
 */
struct SiplControlAutoSensorSetting {
    /**
     * Holds the number of sensor contexts to activate. Multiple sensor contexts mode is
     * supported by some sensors, in which multiple set of settings(contexts) are programmed
     * and the sensor toggles between them at runtime. For sensors not supporting this mode
     * of operation, it shall be set to ‘1’.
     * @li Supported values: [1, DEVBLK_CDI_MAX_SENSOR_CONTEXTS]
     */
    uint8_t                           numSensorContexts;
    /**
     * Holds the sensor exposure settings to set for each context, supports up to DEVBLK_CDI_MAX_SENSOR_CONTEXTS settings.
     * DevBlkCDIExposure is defined in NvSIPLCDICommon.h and has four member variables:
     * @li expTimeValid, holds a Boolean flag which enables or disables the exposure block.
     * @li exposureTime[DEVBLK_CDI_MAX_EXPOSURES], within the range supported by the sensor with the maximum limit [0.0, 100.0].
     * @li gainValid, holds a Boolean flag which enables or disables the sensor gain block.
     * @li sensorGain[DEVBLK_CDI_MAX_EXPOSURES], within the range supported by the sensor with the maximum limit [ 0.0, 1000.0].
     */
    DevBlkCDIExposure                exposureControl[DEVBLK_CDI_MAX_SENSOR_CONTEXTS];
    /**
     * Holds the sensor white balance settings to set for each context, supports up to DEVBLK_CDI_MAX_SENSOR_CONTEXTS settings.
     * DevBlkCDIWhiteBalance is defined in NvSIPLCDICommon.h and has two member variables:
     * @li wbValid, holds a Boolean flag which enables or disables the white balance gain block.
     * @li wbGain[DEVBLK_CDI_MAX_EXPOSURES], within the range supported by the sensor with the maximum limit [1.0, 1000.0].
     */
    DevBlkCDIWhiteBalance            wbControl[DEVBLK_CDI_MAX_SENSOR_CONTEXTS];
    /**
     * Holds the setting for enabling the IR emitter and turning it ON and OFF for RGB-IR sensors.
     */
    DevBlkCDIIllumination   illuminationControl;
};

/**
 * \brief Parsed frame embedded information.
 */
struct SiplControlEmbedInfo {
    /**
     * Holds the parsed embedded data frame number of exposures info for the captured frame.
     * @li Supported values: [1, DEVBLK_CDI_MAX_EXPOSURES]
     */
    uint32_t                          numExposures;
    /**
     * Holds the parsed embedded data sensor exposure info for the captured frame.
     * DevBlkCDIExposure is defined in NvSIPLCDICommon.h and has four member variables:
     * @li expTimeValid, holds a Boolean flag which enables or disables the exposure block.
     * @li exposureTime[DEVBLK_CDI_MAX_EXPOSURES], supported values:  [0.0, 100.0]
     * @li gainValid, holds a Boolean flag which enables or disables the sensor gain block.
     * @li sensorGain[DEVBLK_CDI_MAX_EXPOSURES], supported values: [ 0.0, 1000.0]
     */
    DevBlkCDIExposure                sensorExpInfo;
    /**
     * Holds the parsed embedded data sensor white balance info for the captured frame.
     * DevBlkCDIWhiteBalance is defined in NvSIPLCDICommon.h and has two member variables:
     * @li wbValid, holds a Boolean flag which enables or disables the white balance gain block.
     * @li wbGain[DEVBLK_CDI_MAX_EXPOSURES], supported values: [1.0, 1000.0]
     */
    DevBlkCDIWhiteBalance            sensorWBInfo;
    /**
     * Holds the parsed embedded data sensor temperature info for the captured frame, this variable
     * is not supported in SIPL Control Auto.
     * DevBlkCDITemperature is defined in NvSIPLCDICommon.h and has three member variables:
     * @li tempValid, holds a Boolean flag which enables or disables the sensor temperature block.
     * @li numTemperatures, holds the number of active temperatures, supported values: [1, DEVBLK_CDI_MAX_NUM_TEMPERATURES]
     * @li sensorTempCelsius[DEVBLK_CDI_MAX_NUM_TEMPERATURES], holds the values of active sensor temperatures in degrees Celsius.
     */
    DevBlkCDITemperature             sensorTempInfo;
    /**
     * Holds the parsed embedded data for IR emitter status (ON or OFF) for RGB-IR sensors.
     */
    DevBlkCDIIllumination   illuminationInfo;
};

/**
 * \brief Embedded data and parsed information.
 */
struct SiplControlEmbedData {
    /**
     * Holds the parsed embedded info for the captured frame.
     */
    SiplControlEmbedInfo              embedInfo;
    /**
     * Holds frame sequence number for the captured frame, this variable is not supported in SIPL Control Auto.
     * DevBlkCDIFrameSeqNum is defined in NvSIPLCDICommon.h and has two member variables:
     * @li frameSeqNumValid, holds a Boolean flag which enables or disables the frame sequence number block.
     * @li frameSequenceNumber, holds the sensor frame sequence number value.
     */
    DevBlkCDIFrameSeqNum             frameSeqNum;
    /**
     * Holds information of the embedded data buffer attached to the beginning of the frame, this variable is
     * not supported in SIPL Control Auto.
     * DevBlkCDIEmbeddedDataChunk is defined in NvSIPLCDICommon.h and has two member variables:
     * @li lineLength, holds the line length of an embedded chunk, in bytes.
     * @li lineData, holds a pointer to the data chunk.
     */
    DevBlkCDIEmbeddedDataChunk       topEmbeddedData;
    /**
     * Holds information of the embedded data buffer attached to the end of the frame, this variable is not
     * supported in SIPL Control Auto.
     * DevBlkCDIEmbeddedDataChunk is defined in NvSIPLCDICommon.h and has two member variables:
     * @li lineLength, holds the line length of an embedded chunk, in bytes.
     * @li lineData, holds a pointer to the data chunk.
     */
    DevBlkCDIEmbeddedDataChunk       bottomEmbeddedData;
};

/**
 * \brief Color Gains assuming order RGGB, RCCB, RCCC.
 */
struct SiplControlAutoAwbGain {
    /**
     * A Boolean flag to control whether white balance gains are valid or not.
     */
    bool                              valid;
    /**
     * Gains that applies to individual color channels
     * @li Supported values [0, 8.0]
     */
    float_t                           gain[NVSIPL_ISP_MAX_COLOR_COMPONENT];
};

/**
 * \brief Automatic white balance settings.
 */
struct SiplControlAutoAwbSetting {
    /**
     * Total white balance gains, including both sensor channel gains and ISP gains
     * @li Supported values: [0, 8.0]
     */
    SiplControlAutoAwbGain  wbGainTotal[NVSIPL_ISP_MAX_INPUT_PLANES];
    /**
     * Correlated color temperature.
     * @li Supported values: [2000, 20000]
     */
    float_t                 cct;
    /** Color correction matrix
     * @li Supported values: [-8.0, 8.0]
     */
    float_t                 ccmMatrix[NVSIPL_ISP_MAX_COLORMATRIX_DIM][NVSIPL_ISP_MAX_COLORMATRIX_DIM];
};

/** @brief Structure containing ISP Stats information.
  */
struct SiplControlIspStatsInfo {
    /**
     * Holds pointers to 2 LAC stats data, defined in NvSIPLISPStat.hpp.
     * @li Supported values for numWindowsH: [1, 32]
     * @li Supported values for numWindowsV: [1, 32]
     * @li Supported values for average: [0.0, 1.0]
     * @li Supported values for maskedOffCount: [0, M]
     * @li Supported values for clippedCount: [0, M]
     * M is the number of pixels per color component in the window.
     */
    const NvSiplISPLocalAvgClipStatsData* lacData[2];

    /**
     * Holds pointers to 2 LAC stats settings, defined in NvSIPLISPStat.hpp.
     */
    const NvSiplISPLocalAvgClipStats* lacSettings[2];

    /**
     * Holds pointers to 2 Histogram stats data, defined in NvSIPLISPStat.hpp.
     * @li Supported values for data: [0, number of pixels per color component in the ROI defined in histSettings variable].
     * @li Supported values for excludedCount: [0, number of pixels per color component in the ROI defined in histSettings variable].
     */
    const NvSiplISPHistogramStatsData* histData[2];
    /**
     * Holds pointers to 2 Histogram stats settings, defined in NvSIPLISPStat.hpp.
     */
    const NvSiplISPHistogramStats* histSettings[2];
    /**
     * Holds pointer to Flicker Band stats data, defined in NvSIPLISPStat.hpp.
     * This variable is not supported in SIPL Control Auto.
     * @li Supported values for bandCount: [1, 256]
     * @li Supported values for luminance: [0.0, 1.0]
     */
    const NvSiplISPFlickerBandStatsData* fbStatsData;
    /**
     * Holds pointer to Flicker Band stats settings, defined in NvSIPLISPStat.hpp.
     * This variable is not supported in SIPL Control Auto.
     */
    const NvSiplISPFlickerBandStats*  fbStatsSettings;
};

/**
 * \brief Structure containing metadata info for
 * processing AE/AWB algorithm.
 */
struct SiplControlAutoMetadata {
    /**
     * @brief power factor for statistics compression
     * @li Supported Values [0.5, 1.0]
     */
    float_t    alpha;

    /**
     * @brief A Boolean flag for notifying if it is first frame
     * for processing AE/AWB algorithm without statistics.
     * @li Supported Values [true, false]
     */
    bool    isFirstFrame;

};

/**
 * \brief Input parameters for processing AE/AWB
 */
struct SiplControlAutoInputParam {
    /**
     * Embedded settings
     */
    SiplControlEmbedData        embedData;
    /**
     * Sensor attributes, DevBlkCDISensorAttributes is defined in NvSIPLCDICommon.h.
     *
     * DevBlkCDISensorAttributes has 9 member variables:
     * @li sensorName, a string holds the name attribute. If not supported, set to '\0'.
     * @li sensorCFA, holds the CFA attribute. If not supported, set to zero.
     * @li sensorFuseId, holds the fuse ID attribute. If not supported, set to '\0'.
     * @li numActiveExposures, supported value [1, DEVBLK_CDI_MAX_EXPOSURES]
     * @li sensorExpRange {float min,  float max} [DEVBLK_CDI_MAX_EXPOSURES],  supported values: min [0.0, 100.0] max [min, 100.0]
     * @li sensorGainRange {float min,  float max} [DEVBLK_CDI_MAX_EXPOSURES], supported values: min [0.0, 1000.0] max [min, 1000.0]
     * @li sensorWhiteBalanceRange {float min, float max}[DEVBLK_CDI_MAX_EXPOSURES], supported values: min [0.0, 1000.0] max [min, 1000.0]
     * @li sensorGainFactor, supported values : [0.0, 1000.0]
     * @li numFrameReportBytes, supported values: [1, DEVBLK_CDI_MAX_FRAME_REPORT_BYTES]
     */
    DevBlkCDISensorAttributes  sensorAttr;
    /**
     * Stats buffers and settings
     */
    SiplControlIspStatsInfo     statsInfo;
    /**
     * Metadata info for algorithm
     */
    SiplControlAutoMetadata   autoMetadata;
};

/**
 * \brief AE/AWB Output parameters
 */
struct SiplControlAutoOutputParam {
    /**
     * Sensor exposure and gain settings
     */
    SiplControlAutoSensorSetting    sensorSetting;
    /**
     * AWB settings
     */
    SiplControlAutoAwbSetting       awbSetting;
    /**
     * Digital gain to be applied in ISP
     * @li Supported values: [0.0, 8.0]
     */
    float_t                         ispDigitalGain;
};


/** @} */

}  // namespace nvsipl

#endif /* NVSIPLCONTROLAUTODEF_HPP */
