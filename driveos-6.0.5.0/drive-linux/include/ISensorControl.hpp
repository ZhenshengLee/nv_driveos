/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef ISENSORCONTROL_HPP
#define ISENSORCONTROL_HPP

#include "devblk_cdi.h"

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Sensor Control </b>
 *
 */

namespace nvsipl
{

/** @defgroup ddi_sensor_control Sensor Control API
 *
 * @brief Provides interfaces for the Sensor Control.
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * Interface defining basic Sensor Control functionality
 */
class ISensorControl
{
public:
    /**
     * Destructor
     */
    virtual ~ISensorControl() = default;

    /**
     * @brief Interface to parse top sensor embedded data to get sensor settings applied to
     * the captured frame.
     *
     * This function can be used to retrieve sensor settings like exposure, gain, and white balance
     * settings applied to the frame.
     *
     * @note Exact API behavior is implementation defined.
     *
     * @param[in]  embeddedTopDataChunk         A pointer (of type \ref DevBlkCDIEmbeddedDataChunk)
     *                                          to the top sensor embedded data chunk structure.
     *                                          Must be non-null.
     * @param[in]  embeddedDataChunkStructSize  Size (of type \ref size_t) of the
     *                                          @a embeddedTopDataChunk and
     *                                          @a embeddedBottomDataChunk structures, in bytes.
     *                                          Valid size is device specific.
     * @param[out] embeddedDataInfo             A pointer (of type \ref DevBlkCDIEmbeddedDataInfo)
     *                                          to the embedded data parsed info structure.
     *                                          Valid size is device specific.
     * @param[in]  dataInfoStructSize           Size (of type \ref size_t) of the
     *                                          @a embeddedDataInfo structure, in bytes.
     *                                          Valid size is device specific.
     *
     * @retval          NVMEDIA_STATUS_OK      On successful operation (Success)
     * @retval          (NvMediaStatus)        On other sub-routine propagated error (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Thread-safe: No
     */
    virtual NvMediaStatus SIPLParseTopEmbDataInfo(
                            DevBlkCDIEmbeddedDataChunk const* const embeddedTopDataChunk,
                            size_t const embeddedDataChunkStructSize,
                            DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
                            size_t const dataInfoStructSize) = 0;

    /**
     * @brief Interface to parse bottom sensor embedded data to get statistics information
     * for the captured frame.
     *
     * If image statistics generation is by the sensor and enabled, this function can be used
     * to retrieve statistics like histogram of the frame.
     *
     * @note Exact API behavior is implementation defined.
     *
     * @param[in]  embeddedBotDataChunk         A pointer (of type \ref DevBlkCDIEmbeddedDataChunk)
     *                                          to the bottom sensor embedded data chunk structure.
     *                                          Must be non-null.
     * @param[in]  embeddedDataChunkStructSize  Size (of type \ref size_t) of the
     *                                          @a embeddedTopDataChunk and
     *                                          @a embeddedBottomDataChunk structures, in bytes.
     *                                          Valid size is device specific.
     * @param[out] embeddedDataInfo             A pointer (of type \ref DevBlkCDIEmbeddedDataInfo)
     *                                          to the embedded data parsed info structure.
     *                                          Valid size is device specific.
     * @param[in]  dataInfoStructSize           Size (of type \ref size_t) of the
     *                                          @a embeddedDataInfo structure, in bytes.
     *                                          Valid size is device specific.
     *
     * @retval          NVMEDIA_STATUS_OK      On successful operation (Success)
     * @retval          (NvMediaStatus)        On other sub-routine propagated error (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
     virtual NvMediaStatus SIPLParseBotEmbDataInfo(
                            DevBlkCDIEmbeddedDataChunk const* const embeddedBotDataChunk,
                            size_t const embeddedDataChunkStructSize,
                            DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
                            size_t const dataInfoStructSize) = 0;


    /**
     * @brief Interface to set sensor control parameters.
     *
     * This function enables caller to control sensor image settings
     * like exposure time, sensor gain, and white balance gain.
     * All parameters provided to this function are applied together at a frame boundary
     * through "group hold" functionality, if supported by the sensor.
     *
     * This function invokes the device driver function specified by the
     * call to SetSensorControls().
     *
     * @note Exact API behavior is implementation defined.
     *
     * @param[in]  sensorControl       A pointer (of type \ref DevBlkCDISensorControl) to a
     *                                 sensor control structure for @a device.
     * @param[in]  sensrCtrlStructSize Size (of type \ref size_t) of the @a sensorControl
     *                                 structure.
     *
     * @retval     NVMEDIA_STATUS_OK   On successful operation (Success).
     * @retval     (NvMediaStatus)     On other sub-routine propagated error (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual NvMediaStatus SIPLSetSensorControls(DevBlkCDISensorControl const* const sensorControl,
                                                size_t const sensrCtrlStructSize) = 0;
    /**
     * Interface to get the sensor attributes.
     *
     * Sensor attributes are static properties like sensor name,
     * exposure-gain ranges supported, and number of active exposures.
     *
     * @note Exact API behavior is implementation defined.
     *
     * @param[out]  sensorAttr           A pointer (of type \ref DevBlkCDISensorAttributes) a
     *                                   sensor attributes structure.
     * @param[in]   sensorAttrStructSize Size (of type \ref size_t) of the @a sensorAttr structure.
     *                                   Must be exact size of \ref DevBlkCDISensorAttributes.
     *
     * @retval      NVMEDIA_STATUS_OK    On successful operation (Success)
     * @retval      (NvMediaStatus)      On other sub-routine propagated error (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual NvMediaStatus SIPLGetSensorAttributes(DevBlkCDISensorAttributes *const sensorAttr,
                                                  size_t const sensorAttrStructSize) = 0;
#if !NV_IS_SAFETY
    /**
     * Interface to set sensor in characterization mode.
     *
     * SIPLSetSensorCharMode API calls into DevBlk CDI functionality
     * to provide the ability for the user to configure the sensor for characterization.
     * Sensor characterization provides optimal parameters,
     * corresponding to sensor physical and functional characteristics,
     * for image processing.
     *
     * Sensor characterization for High Dynamic Range (HDR) sensors
     * with multiple exposures (T1, T2, ... , Tn )
     * involves characterizing individual exposures separately, if required by the sensor.
     * This API provides the ability to configure sensor to capture each exposure separately,
     * if required by sensor characterization.
     *
     * This function re-configures the sensor
     * i.e. changes the sensor static attributes
     * like numActiveExposures, sensorExpRange, sensorGainRange
     * and hence, should be called during sensor initialization time.
     * In order to characterize the sensor exposure number 'n',
     * where n = {1,2,3, ... , N} for N-exposure HDR sensor,
     * the input parameter 'expNo' should be set to 'n'.
     * For a non-HDR sensor, the input parameter 'expNo' should always be set to '1'.
     *
     * @note Exact API behavior is implementation defined.
     *
     * @param[in]  expNo  Sensor exposure number (of type \ref uint8_t) to be used for
     *                    characterization.
     *                    Valid range for expNo : [0, \ref DEVBLK_ISC_MAX_EXPOSURES)
     *                    For Non-HDR sensor, this should be set to '1'
     *
     * @retval     NVMEDIA_STATUS_OK  On successful operation (Success)
     * @retval     (NvMediaStatus)    On other sub-routine propagated error (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual NvMediaStatus SIPLSetSensorCharMode(uint8_t expNo) = 0;
#endif // !NV_IS_SAFETY
};

/** @} */

} // end of namespace nvsipl
#endif //ISENSORCONTROL_HPP

