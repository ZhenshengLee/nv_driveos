/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
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
     * @brief Constructor.
     */
    ISensorControl() =  default;

    /**
     * @brief Prevent ISensorControl from being copy constructed.
     */
    ISensorControl(ISensorControl const &) = delete;

    /**
     * @brief Prevent ISensorControl from being move constructed.
     */
    ISensorControl(ISensorControl &&) = delete;

    /**
     * @brief Prevent ISensorControl from being copy assigned.
     */
    ISensorControl& operator=(ISensorControl const &) & = delete;

    /**
     * @brief Prevent ISensorControl from being move assigned.
     */
    ISensorControl& operator=(ISensorControl&&) & = delete;

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
     * @pre
     *   - A valid sensor device driver created with DriverCreate().
     *   - Sensor device driver implements ParseTopEmbDataInfo() function.
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
     * @pre
     *   - A valid sensor device driver created with DriverCreate().
     *   - Sensor device driver implements ParseBotEmbDataInfo() function.
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
     * @pre
     *   - A valid sensor device driver created with DriverCreate().
     *   - Sensor device driver implements SetSensorControls() function.
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
     * @pre
     *   - A valid sensor device driver created with DriverCreate().
     *   - Sensor device driver implements GetSensorAttributes() function.
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
     * @pre
     *   - A valid sensor device driver created with DriverCreate().
     *   - Sensor device driver implements SetSensorCharMode() function.
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

    /**
     * Interface to request image data authentication.
     *
     * SIPLAuthenticateImage API calls into DevBlk CDI functionality
     * to request authentication of an image data. Pixel and embedded data can be
     * authenticated.
     * Authentication session needs to be established by underlying driver prior to
     * calling the API or authentication mismatch will be returned.
     *
     * Pixel data is a mandatory input parameter, while embedded data is optional
     * and can be NULL if embedded data is not enabled on a sensor or embedded data
     * authentication is not required.
     *
     * The API may be called for different sensor device objects from multiple threads
     * simultaneously. The API will never be called from multiple threads for the same
     * sensor. Implementation shall ensure thread safety of authentication calls for
     * different sensor objects.
     *
     * @pre
     *   - @a imageDesc must be a valid sensor description.
     *   - Sensor device driver implements AuthenticateImage() function.
     *
     * @param[in]  imageDesc A description of a RAW image to authenticate
     *                        Valid value: [non-NULL].
     *
     * @retval NVMEDIA_STATUS_OK Authentication match.
     * @retval NVMEDIA_STATUS_INCOMPATIBLE_VERSION Image verification failure (authentication
     *                                             mismatch). (Failure)
     * @retval NVMEDIA_STATUS_UNDEFINED_STATE Out-of-order image is detected. (Failure)
     * @retval NVMEDIA_STATUS_ERROR Internal failure in crypto operation. (Failure)
     * @retval NVMEDIA_STATUS_INVALID_SIZE Input buffer sizes are invalid. (Failure)
     * @retval NVMEDIA_STATUS_NOT_SUPPORTED authentication is not supported by
     *                                      the device.
     * @retval NVMEDIA_STATUS_BAD_PARAMETER Invalid input parameters are passed.
     * @retval NVMEDIA_STATUS_NOT_INITIALIZED Crypto state was not initialized before
     *                                        requesting image verification. (Failure)
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
    virtual NvMediaStatus
    SIPLAuthenticateImage(DevBlkImageDesc const * const imageDesc) const = 0;
};

/** @} */

} // end of namespace nvsipl
#endif //ISENSORCONTROL_HPP

