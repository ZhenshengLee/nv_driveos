/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMSENSOR_HPP
#define CNVMSENSOR_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"

#include "CNvMDevice.hpp"
#include "utils/utils.hpp"
#include "ISensorControl.hpp"
#include "INvSIPLDeviceInterfaceProvider.hpp"

namespace nvsipl
{

/**
 * The CNvMSensor class encapsulates Sensor control and configuration information.
 */
class CNvMSensor : public CNvMDevice,  public ISensorControl
{
public:
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * Handle passed from Sensor to EEPROM
     */
    typedef struct {
        void *handle;
        uint32_t size;
    } EEPROMRequestHandle;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    /**
     * Constructor
     *
     * - set m_pipelineIndex = 0U
     * - set m_embLinesTop = 0U
     * - set m_embLinesBot = 0U
     * - set m_bEmbDataType = false
     * - set m_width = 0U
     * - set m_height = 0U
     * - set m_ePixelOrder = 0U
     * - set m_frameRate = static_cast<float_t>(0.0)
     * - set m_bEnableExtSync = false
     * - set m_eInputFormat.inputFormatType = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422
     * - set m_eInputFormat.bitsPerPixel = NVSIPL_BITS_PER_PIXEL_8
     *
     */
    CNvMSensor();

    /**
     * Default Destructor
     */
    virtual ~CNvMSensor() = default;


    /**
     * @brief Set the Sensor object's device parameters.
     *
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save the serializer's I2C address and register it with the address manager.
     * Should check @a sensorInfo for consistency where possible.
     *
     * - verify sensorInformation and params are NOT nullptr
     * - verify m_eState != CREATED
     * - set m_embLinesTop = sensorInformation->vcInfo.embeddedTopLines
     * - set m_embLinesBot = sensorInformation->vcInfo.embeddedBottomLines
     * - set m_bEmbDataType = sensorInformation->vcInfo.isEmbeddedDataTypeEnabled
     * - set m_width = sensorInformation->vcInfo.resolution.width
     * - set m_height = sensorInformation->vcInfo.resolution.height
     * - set m_eInputFormat.inputFormatType = sensorInformation->vcInfo.inputFormat
     * - verify (m_embLinesTop != 0U) OR (NOT m_bEmbDataType)
     * - call status = SetInputFormatProperty()
     * - set m_ePixelOrder = sensorInformation->vcInfo.cfa
     * - set m_frameRate = sensorInformation->vcInfo.fps
     * - set m_bEnableExtSync = sensorInformation->isTriggerModeEnabled
     * - set m_oDeviceParams = *params
     * - if (NOT m_oDeviceParams.bEnableSimulator) and (NOT m_oDeviceParams.bPassive)
     *  - set m_nativeI2CAddr =  sensorInformation->i2cAddress
     *  - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr)
     *  .
     * - set m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE
     * - set m_eState = CDI_DEVICE_CONFIG_SET
     *
     * @param[in] sensorInformation Sensor info struct containing sensor information to configure
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      NVSIPL_STATUS_INVALID_STATE if device state not Created
     * @retval                      NVSIPL_STATUS_NOT_SUPPORTED if embedded data lines do not match config
     * @retval                      (SIPLStatus) other error propagated
     */
    virtual SIPLStatus SetConfig(SensorInfo const* const sensorInformation, DeviceParams const* const params);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * @brief Set sensor's pipeline index.
     *
     * - set m_pipelineIndex = index
     *
     * @param[in]                   index
     */
    virtual void SetPipelineIndex(uint32_t const index);

    /**
     * Get sensor's pipeline index
     *
     * - returns m_pipelineIndex
     *
     * @retval                      pipeline index
     */
    virtual uint32_t GetPipelineIndex() const;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    /**
     * @brief Get width of a frame from this sensor, in pixels.
     *
     * - returns m_width
     * @retval                      width in pixels.
     */
    virtual uint32_t GetWidth() const;

    /**
     * @brief Get height of a frame from this sensor, in pixels.
     *
     * - returns m_height
     * @retval                      height in pixels.
     */
    virtual uint32_t GetHeight() const;

    /**
     * @brief Get number of top embedded lines.
     *
     * - returns m_embLinesTop
     *
     * @retval                      number of top embedded lines
     */
    virtual uint32_t GetEmbLinesTop() const;

    /**
     * @brief Get number of bottom embedded lines.
     *
     * - returns m_embLinesBot
     *
     * @retval                      number of bottom embedded lines
     */
    virtual uint32_t GetEmbLinesBot() const;

    /**
     * Get embedded data type
     *
     * - returns m_bEmbDataType
     *
     * @retval                      true if embedded data type enabled, false otherwise
     */
    virtual bool GetEmbDataType() const;

    /**
     * Get input format type and bits per pixel configured for object
     *
     * - returns m_eInputFormat
     *
     * @retval                      NvSiplCapInputFormat
     */
    virtual NvSiplCapInputFormat GetInputFormat() const;

    /**
     * Get pixel order
     *
     * - returns m_ePixelOrder
     * @retval                      uint32_t
     */
    virtual uint32_t GetPixelOrder() const;

    /**
     * Get frame rate
     *
     * - returns m_frameRate
     *
     * @retval                      frame rate in fps
     */
    virtual float_t GetFrameRate() const;

    /**
     * Get Enable external sync
     *
     * - returns m_bEnableExtSync
     *
     * @retval                      true if sensor trigger mode is enabled.
     * @retval                      false if sensor trigger mode is disabled.
     */
    virtual bool GetEnableExtSync() const;

    /**
     * Get sensor authentication setting
     *
     * @retval                      true if sensor authentication is enabled.
     * @retval                      false if sensor authentication is disabled.
     */
    virtual bool IsAuthenticationEnabled() const;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * Get Enable TPG
     *
     * @retval                      true if TPG enabled
     * @retval                      false if TPG not enabled
     */
    virtual bool GetEnableTPG() const;

    /**
     * Get Pattern Mode
     *
     * @retval                      pattern mode [0, UINT32_MAX)
     */
    virtual uint32_t GetPatternMode() const;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    /**
     * Get Sensor Identifier
     *
     * - returns m_sensorDescription
     *
     * @retval                      string description
     */
    virtual std::string GetSensorDescription() const;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * @brief Get a @ref EEPROMRequestHandle based on the last call to @ref
     * SetEEPROMRequestHandle().
     */
    virtual EEPROMRequestHandle GetEEPROMRequestHandle() const final;

    /**
     * @brief If the handle is not NULL, set the EEPROM object's
     * EEPROMRequestHandle information.
     *
     * @param[in] handle            Handle to set
     * @param[in] size              Size to set
     */
    virtual void SetEEPROMRequestHandle(void *const handle, const uint32_t size) final;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    /**
     * @brief Set the CDI driver handle.
     *
     * This sets @a m_pCDIDriver so @ref nvsipl::CNvMDevice::CreateCDIDevice() works.
     *
     * - verify driver is NOT nullptr
     * - set m_pCDIDriver = driver
     *
     * @param[in] driver            CDI driver to set
     * @retval                      NVSIPL_STATUS_OK on successful completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on Invalid argument
     */
    SIPLStatus SetDriverHandle(DevBlkCDIDeviceDriver *const driver) {
        SIPLStatus status = NVSIPL_STATUS_OK;
        if (nullptr == driver) {
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        } else {
            m_pCDIDriver = driver;
        }
        return status;
    }

    /**
     * Move the provided driver context to the sensor object
     *
     * This sets @ref nvsipl::CNvMDevice::m_upDrvContext so @ref nvsipl::CNvMDevice::CreateCDIDevice() works.
     *
     * - verify  is NOT nullptr
     * - set m_upDrvContext = std::move(context)
     *
     * @param[in] context           Context to move to object
     * @retval                      NVSIPL_STATUS_OK on successful completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on Invalid argument
     */
    SIPLStatus SetDriverContext(std::unique_ptr<DriverContext> context) {
        SIPLStatus status = NVSIPL_STATUS_OK;
        if (!context) {
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        } else {
            m_upDrvContext = std::move(context);
        }
        return status;
    }

    /**
     * ISensorControl methods
     */

    /**
     * Parse top embedded data
     *
     * - verify m_upCDIDevice is NOT nullptr
     * - call
     *  - status = DevBlkCDIParseTopEmbDataInfo(
     *   - m_upCDIDevice.get(),
     *   - embeddedTopDataChunk,
     *   - embeddedDataChunkStructSize,
     *   - embeddedDataInfo,
     *   - dataInfoStructSize)
     *   .
     *  .
     *
     * For details refer ISensorControl interface class.
     */
    virtual NvMediaStatus SIPLParseTopEmbDataInfo(
                            DevBlkCDIEmbeddedDataChunk const* const embeddedTopDataChunk,
                            size_t const embeddedDataChunkStructSize,
                            DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
                            size_t const dataInfoStructSize) override final;

    /**
     * Parse bottom embedded data
     *
     * - verify m_upCDIDevice is NOT nullptr
     * - call
     *  - status = DevBlkCDIParseBotEmbDataInfo(
     *   - m_upCDIDevice.get(),
     *   - embeddedBotDataChunk,
     *   - embeddedDataChunkStructSize,
     *   - embeddedDataInfo,
     *   - dataInfoStructSize)
     *   .
     *  .
     *
     * For details refer ISensorControl interface class.
     */
     virtual NvMediaStatus SIPLParseBotEmbDataInfo(
                            DevBlkCDIEmbeddedDataChunk const* const embeddedBotDataChunk,
                            size_t const embeddedDataChunkStructSize,
                            DevBlkCDIEmbeddedDataInfo *const embeddedDataInfo,
                            size_t const dataInfoStructSize) override final;

    /**
     * Set sensor controls
     *
     * - verify m_upCDIDevice is NOT nullptr
     * - call
     *  -  status = DevBlkCDISetSensorControls(
     *   - m_upCDIDevice.get(),
     *   - sensorControl,
     *   - sensrCtrlStructSize)
     *   .
     *  .
     *
     * For details refer ISensorControl interface class.
     */
    virtual NvMediaStatus SIPLSetSensorControls(
                            DevBlkCDISensorControl const* const sensorControl,
                            size_t const sensrCtrlStructSize) override final;

    /**
     * Set sensor attributes
     *
     * - verify m_upCDIDevice is NOT nullptr
     * - call
     *  - status = DevBlkCDIGetSensorAttributes(
     *   - m_upCDIDevice.get(),
     *   - sensorAttr,
     *   - sensorAttrStructSize)
     *   .
     *  .
     *
     * For details refer ISensorControl interface class.
     */
    virtual NvMediaStatus SIPLGetSensorAttributes(
                            DevBlkCDISensorAttributes *const sensorAttr,
                            size_t const sensorAttrStructSize) override final;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * Set sensor char mode
     *
     * For details refer ISensorControl interface class.
     */
    virtual NvMediaStatus SIPLSetSensorCharMode(uint8_t expNo) final;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

    /**
     * Authenticate a single image on a sensor device
     *
     * For details refer to ISensorControl interface class.
     */
    NvMediaStatus
    SIPLAuthenticateImage(DevBlkImageDesc const * const imageDesc) const final;

protected:

    /**
     * Set ICP input format type and bits per pixel based on object's input format
     * Supported values for input format are: 422p, rgb, raw10, raw12, raw16
     *
     * @retval                      NVSIPL_STATUS_OK on completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on invalid input format
     */
    SIPLStatus SetInputFormatProperty();

private:
    /**
     * @brief Device specific implementation for device initialization.
     *
     * By default this method is no-op, if device drivers wish to override this
     * behavior can do so by overriding this method \ref DoInit().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval (SIPLStatus)                 Subclasses can override this
     *                                      function to return other status
     *                                      values. (Failure)
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
     virtual SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * @brief Device specific implementation for device start operation.
     *
     * By default this method is no-op, if device drivers wish to override this
     * behavior can do so by overriding this method \ref DoStart().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval (SIPLStatus)                 Subclasses can override this
     *                                      function to return other status
     *                                      values. (Failure)
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
    virtual SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * @brief Device specific implementation for device stop operation.
     *
     * By default this method is no-op, if Device drivers wish to override this
     * behavior can do so by overriding this method \ref DoStop().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval (SIPLStatus)                 Subclasses can override this
     *                                      function to return other status
     *                                      values. (Failure)
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
     *   - Runtime: No
     *   - De-Init: Yes
     */
    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };


    /**
     * Holds pipeline index for the sensor
     */
    uint32_t m_pipelineIndex;

    /**
     * Holds number of top embedded lines
     */
    uint32_t m_embLinesTop;

    /**
     * Holds number of bottom embedded lines
     */
    uint32_t m_embLinesBot;

    /**
     * Holds embedded data type
     */
    bool m_bEmbDataType;

    /**
     * Holds width
     */
    uint32_t m_width;

    /**
     * Holds height
     */
    uint32_t m_height;

    /**
     * Holds input format
     */
    std::string m_inputFormat;

    /**
     * Holds ICP input format
     */
    NvSiplCapInputFormat m_eInputFormat;

    /**
     * Holds pixel order
     */
    uint32_t m_ePixelOrder;

    /**
     * Holds frame rate (in fps)
     */
    float_t m_frameRate;

    /**
     * Flag indicating whether the sensor is in trigger mode
     */
    bool m_bEnableExtSync;

    /**
     * Holds sensor description
     */
    std::string m_sensorDescription;

    /**
     * Specifies whether device and all data exchange with the device
     * needs to be cryptographically authenticated
     */
    bool m_bIsAuthEnabled;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /**
     * Flag indicating whether TPG is enabled
     */
    bool m_bEnabletpg;

    /**
     * Holds EEPROM request handle
     */
    EEPROMRequestHandle m_EEPROMRequestHandle;

    /**
     * Holds Test Pattern Mode
     */
    uint32_t m_patternMode;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
};

} // end of namespace nvsipl
#endif /* CNVMSENSOR_HPP */

