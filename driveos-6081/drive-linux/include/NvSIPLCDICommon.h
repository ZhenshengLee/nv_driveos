/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * All rights reserved.  All information contained herein is proprietary
 * and confidential to NVIDIA Corporation.
 * Any use, reproduction, or disclosure without the written permission of
 * NVIDIA Corporation is prohibited.
 */

#ifndef NVSIPL_CDI_COMMON_STRUCTS_H
#define NVSIPL_CDI_COMMON_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup CDI common types.
 * @ingroup cdi_api_grp
 *
 * @{
 */

/** \hideinitializer \brief A true \ref NvSiplBool value. */
#define NVSIPL_TRUE  (0 == 0)
/** \hideinitializer \brief A false \ref NvSiplBool value. */
#define NVSIPL_FALSE (0 == 1)

typedef uint32_t NvSiplBool;
/**
 * \brief Defines the double-precision location of a point on a two-dimensional
 *  object.
 * \note This structure is deprecated and will be removed in the next version.
 */
typedef struct {
    /*! Holds the horizontal location of the point. */
    double_t x;
    /*! Holds the vertical location of the point. */
    double_t y;
} NvSiplPointDouble;

/** \brief Maximum number of exposures. */
#define DEVBLK_CDI_MAX_EXPOSURES           (8U)

/** \brief Maximum possible length of sensor name. */
#define DEVBLK_CDI_MAX_SENSOR_NAME_LENGTH  (32U)

/** \brief Maximum possible length of sensor fuse id. */
#define DEVBLK_CDI_MAX_FUSE_ID_LENGTH      (32U)

/** \brief Maximum number of color components. */
#define DEVBLK_CDI_MAX_COLOR_COMPONENT         (4U)

/** \brief Maximum number of sensor temperature values. */
#define DEVBLK_CDI_MAX_NUM_TEMPERATURES    (4U)

/** \brief Maximum number of sensor contexts. */
#define DEVBLK_CDI_MAX_SENSOR_CONTEXTS     (4U)

/**
 * \brief Maximum number of sensor companding
 *  piecewise linear (PWL) curve knee points.
 */
#define DEVBLK_CDI_MAX_PWL_KNEEPOINTS      (64U)

/** \brief Maximum number of frame report bytes.  */
#define DEVBLK_CDI_MAX_FRAME_REPORT_BYTES  (4U)

/**
 * \brief  Holds the range of a sensor attribute.
 */
typedef struct {
    /**
     * Holds the sensor attribute's minimum value.
     */
    float_t  min;

    /**
     * Holds the sensor attribute's maximum value.
     */
    float_t  max;

} DevBlkCDIAttrRange;

/**
 * \brief  Holds the parameters regarding step size for non-HDR sensors.
 */
typedef struct {
    /**
     * Holds the a flag to determine whether the exposure time and sensor gain
     * quantization step sizes are required for non-HDR sensors. Set to true for
     * non-HDR sensors and false for HDR sensors.
     */
    NvSiplBool isQuantizationStepSizeValid;

    /**
     * Holds the sensor gain quantization step size.
     */
    float_t quantizationStepSizeSG;

    /**
     * Holds the exposure time quantization step size.
     */
    float_t quantizationStepSizeET;

} DevBlkCDIAttrQuantizationStepSize;

/**
 * \brief  Holds the sensor attributes.
 */
typedef struct DevBlkCDISensorAttributes {
    /**
     * Holds the name attribute. If not supported, set to NULL.
     */
    char  sensorName[DEVBLK_CDI_MAX_SENSOR_NAME_LENGTH];

    /**
     * Holds the CFA attribute. If not supported, set to zero.
     */
    uint32_t  sensorCFA;

    /**
     * Holds the fuse ID attribute. If not supported, set to NULL.
     */
    uint8_t sensorFuseId[DEVBLK_CDI_MAX_FUSE_ID_LENGTH];

    /**
     * Holds the number of active exposures attribute. Range is
     * [1, \ref DEVBLK_CDI_MAX_EXPOSURES]
     */
    uint8_t  numActiveExposures;

    /**
     * \brief Holds the sensor exposure ranges for active exposures.
     *
     * Only the first `numActiveExposures` elements are valid.
     */
    DevBlkCDIAttrRange sensorExpRange[DEVBLK_CDI_MAX_EXPOSURES];

    /**
     * Holds the sensor gain ranges for active exposures.
     *
     * Only the first `numActiveExposures` elements are valid.
     */
    DevBlkCDIAttrRange sensorGainRange[DEVBLK_CDI_MAX_EXPOSURES];

    /**
     * Holds the sensor white balance ranges for active exposures.
     *
     * Only the first `numActiveExposures` elements are valid.
     */
    DevBlkCDIAttrRange sensorWhiteBalanceRange[DEVBLK_CDI_MAX_EXPOSURES];

    /**
     * Holds the additional sensor gain factor between active exposures.
     *
     * Only the first `numActiveExposures` elements are valid.
     *
     * Element @a holds the sensitivity ratio between capture 0 and capture
     * @a a. It is usually set to 1.
     *
     * The HDR ratio between exposure @a a and exposure @a b can be computed as:
     *
     *  \f$
     *  hdrRatio_{ab} = ( Et[a] * Gain[a] * sensorGainFactor[a] ) / (Et[b] * Gain[b] * sensorGainFactor[b])
     *  \f$
     *
     * Where:
     * - Et[a] = exposure time value for HDR exposure @a a.
     * - Gain[a] = sensor gain value for HDR exposure @a a.
     * - sensorGainFactor[a] = sensor gain factor for HDR exposure @a a.
     * - Et[b] = exposure time value for HDR exposure @a b.
     * - Gain[b] = sensor gain value for HDR exposure @a b.
     * - sensorGainFactor[b] = sensor gain factor for HDR exposure @a b.
     */
    float_t  sensorGainFactor[DEVBLK_CDI_MAX_EXPOSURES];

    /**
     * Holds the number of frame report bytes supported by the sensor. If not
     * supported, set to zero.
     * Supported values : [1 , DEVBLK_CDI_MAX_FRAME_REPORT_BYTES]
     */
    uint32_t  numFrameReportBytes;

    /*
     * Holds the quantization step size parameters for non-HDR sensors.
     */
    DevBlkCDIAttrQuantizationStepSize sensorQuantizationStepSize;

} DevBlkCDISensorAttributes;

/**
 * \brief  Holds the sensor frame sequence number structure.
 */
typedef struct {
    /**
     * Holds a flag which enables OR DISABLES the frame sequence number block.
     */
    NvSiplBool  frameSeqNumValid;

    /**
     * Holds the sensor frame sequence number value.
     */
    uint64_t frameSequenceNumber;

} DevBlkCDIFrameSeqNum;

/*
 * \brief  Holds the sensor embedded data chunk structure.
 */
typedef struct DevBlkCDIEmbeddedDataChunk {
    /**
     * Holds the line length of an embedded chunk, in bytes.
     */
    uint32_t lineLength;

    /**
     * Holds a pointer to the data chunk.
     */
    uint8_t *lineData;
} DevBlkCDIEmbeddedDataChunk;

/**
 * \brief  Holds sensor exposure information.
 *
 * To use this structure correctly, the number of valid exposures must be
 * tracked separately. When `DevBlkCDIExposure` is embedded in another
 * structure, it is known according to the following members:
 *
 *   - `DevBlkCDIEmbeddedDataInfo::numExposures`
 *   - `DevBlkCDISensorControl`: The number of exposures supported by the
 *     sensor being programmed must be valid.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables the exposure block.
     */
    NvSiplBool expTimeValid;

    /**
     * Holds exposure time for each active exposure, in seconds. The array has
     * \ref DevBlkCDIEmbeddedDataInfo.numExposures elements.
     */
    float_t exposureTime[DEVBLK_CDI_MAX_EXPOSURES];

    /**
     * Holds a flag which enables or disables the sensor gain block.
     */
    NvSiplBool gainValid;

    /**
     * Holds sensor a gain value for each active gain. The array has
     * \ref DevBlkCDIEmbeddedDataInfo.numExposures elements.
     */
    float_t sensorGain[DEVBLK_CDI_MAX_EXPOSURES];

} DevBlkCDIExposure;

/**
 * \brief Per-channel gains, intended for use with DevBlkCDIWhiteBalance.
 */
typedef struct {
    /**
     * \brief Sensor white balance gain values for active
     * exposures, in R Gr Gb B order.
     */
    float_t value[DEVBLK_CDI_MAX_COLOR_COMPONENT];
} DevBlkCDIWhiteBalanceGain;

/**
 * \brief  Holds the sensor white balance gain structure.
 *
 * To use this structure correctly, the number of valid exposures must be
 * tracked separately. When accessed via a DevBlkCDIEmbeddedDataInfo structure,
 * this is known by virtue of DevBlkCDIEmbeddedDataInfo.numExposures.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables the white balance gain block.
     */
    NvSiplBool wbValid;

    /**
     * \brief Per-channel white balance gains.
     *
     * @a wbGain has \ref DevBlkCDIEmbeddedDataInfo.numExposures
     * elements.
     */
     DevBlkCDIWhiteBalanceGain wbGain[DEVBLK_CDI_MAX_EXPOSURES];

} DevBlkCDIWhiteBalance;

/**
 * \brief  Holds the sensor illumination control structure.
 */
typedef struct {
    /**
     * Holds a flag which shows the illumination is valid.
     */
    NvSiplBool bValid;

    /**
     * Holds a flag which enables or disables the illumination.
     */
    NvSiplBool bEnable;

} DevBlkCDIIllumination;

/**
 * \brief  Holds the sensor temperature structure.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables the sensor temperature block.
     */
    NvSiplBool  tempValid;

    /**
     * Holds the number of active temperatures. Must be in the range
     * [1, \ref DEVBLK_CDI_MAX_NUM_TEMPERATURES].
     */
    uint8_t numTemperatures;

    /**
     * Holds the values of active sensor temperatures in degrees Celsius. Array
     * indexes must be in the range [0, (@a numTemperatures-1)].
     */
    float_t sensorTempCelsius[DEVBLK_CDI_MAX_NUM_TEMPERATURES];

} DevBlkCDITemperature;

/**
 * \brief  Holds the sensor report frame report structure.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables frame report block.
     */
    NvSiplBool frameReportValid;

    /**
     * Holds the number of active frame report bytes. Range is
     * [1, DEVBLK_CDI_MAX_FRAME_REPORT_BYTES].
     */
    uint8_t numBytes;

    /**
     * Holds the values of active frame report bytes. Array indexes must be
     * in the range [0, (@a numBytes-1)].
     */
    uint8_t sensorframeReport[DEVBLK_CDI_MAX_FRAME_REPORT_BYTES];

} DevBlkCDIFrameReport;

/**
 * \brief  Holds the sensor companding piecewise linear (PWL) structure.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables the sensor PWL block.
     */
    NvSiplBool  pwlValid;

    /**
     * Holds the number of active PWL knee points. Must be in the range
     * [1, \ref DEVBLK_CDI_MAX_PWL_KNEEPOINTS].
     */
    uint8_t numKneePoints;

    /**
     * Holds the values of active PWL knee points. Array indexes must be in the
     * range [0, (@a numKneePoints-1)].
     */
    NvSiplPointDouble kneePoints[DEVBLK_CDI_MAX_PWL_KNEEPOINTS];

} DevBlkCDIPWL;

/**
 * \brief  Holds the sensor CRC structure.
 */
typedef struct {
    /**
     * Holds a flag which enables or disables the CRC block.
     */
    NvSiplBool  crcValid;

    /**
     * Holds the frame CRC value computed from embedded data.
     */
    uint32_t computedCRC;

    /**
     * Holds the frame CRC value parsed from embedded data.
     */
    uint32_t embeddedCRC;

} DevBlkCDICRC;

/**
 * \brief  Holds the sensor frame timestamp structure.
 */
typedef struct DevBlkCDIFrameTimestamp {
    /**
      * Holds a flag which indicates if the frame timestamp from the sensor is valid. */
    NvSiplBool  frameTimestampValid;
    /**
      * Holds the sensor frame timestamp value. */
    uint64_t frameTimestamp;
} DevBlkCDIFrameTimestamp;

/**
 * \brief  Holds the sensor embedded data parsed info structure.
 *
 * The sensor driver can selectively activate or deactivate any of the parsed
 * info blocks, depending on whether the sensor supports it.
 *
 * To activate a
 * sensor info block, the sensor driver must set the info block's @a valid flag to
 * TRUE and populate the parsed information corresponding to the block.
 * To disable a sensor info block, it must set the valid flag to FALSE.
 *
 * For example, if a sensor supports only exposure, white balance and CRC info,
 * the sensor driver activates only the @a sensorExpInfo, @a sensorWBInfo, and
 * @a sensorCRCInfo blocks (it sets their @a valid flags to TRUE) and disables
 * all of the others (it sets their @a valid flags to FALSE).
 */
typedef struct DevBlkCDIEmbeddedDataInfo {
    /**
     * Holds the parsed embedded data frame number of exposures for the
     * captured frame.
     */
    uint32_t                    numExposures;

    /**
     * Holds the parsed embedded data sensor exposure info for the
     * captured frame.
     */
    DevBlkCDIExposure           sensorExpInfo;

    /**
     * Holds the parsed embedded data sensor white balance info for the
     * captured frame.
     */
    DevBlkCDIWhiteBalance       sensorWBInfo;

    /**
     * Holds the parsed embedded data sensor PWL info for the captured frame.
     */
    DevBlkCDIPWL                sensorPWLInfo;

    /**
     * Holds the parsed embedded data sensor CRC info for the captured frame.
     */
    DevBlkCDICRC                sensorCRCInfo;

    /**
     * Holds the parsed embedded data frame report info for the captured frame.
     */
    DevBlkCDIFrameReport        sensorReportInfo;

    /**
     * Holds the parsed embedded data illumination info for the captured frame.
     */
    DevBlkCDIIllumination       illuminationInfo;

    /**
     * Holds the parsed embedded data sensor temperature info for the
     * captured frame.
     */
    DevBlkCDITemperature        sensorTempInfo;

    /**
     * Holds parsed embedded data frame sequence number info for the
     * captured frame.
     */
    DevBlkCDIFrameSeqNum        frameSeqNumInfo;

    /**
     * Holds parsed embedded data frame timestamp info for the
     * captured frame.
     */
    DevBlkCDIFrameTimestamp     frameTimestampInfo;

    /**
     * Holds information on errors present in the embedded data.
     * The meaning of these values is determined by the driver.
     */
    int8_t                      errorFlag;
} DevBlkCDIEmbeddedDataInfo;

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif /* __cplusplus */

#endif /* NVSIPL_CDI_COMMON_STRUCTS_H */
