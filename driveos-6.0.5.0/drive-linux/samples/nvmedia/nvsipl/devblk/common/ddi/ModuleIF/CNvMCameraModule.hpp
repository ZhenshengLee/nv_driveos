/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMCAMERAMODULE_HPP
#define CNVMCAMERAMODULE_HPP

#include "NvSIPLPlatformCfg.hpp"
/*VCAST_DONT_INSTRUMENT_START*/
#include "utils/utils.hpp"
#include "DeserializerIF/CNvMDeserializer.hpp"
/*VCAST_DONT_INSTRUMENT_END*/
#include "ISensorControl.hpp"
#include "IInterruptStatus.hpp"

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Camera Module </b>
 *
 */

namespace nvsipl
{

/** @defgroup ddi_cam_module Camera Module API
 *
 * @brief Provides interfaces for Camera Module.
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * Class which encapsulates Camera Module information
 */
class CNvMCameraModule: public IInterfaceProvider, public IInterruptStatus

{
public:
        /**
         * struct contains necessary information for Camera Module configuration.
         */
    struct CameraModuleConfig {
        /**
         * Settings specific to this camera module
         */
        const CameraModuleInfo *cameraModuleInfo;

        /**
         * Device parameters (simulator, passive, etc)
         */
        CNvMDevice::DeviceParams *params;

        /**
         * Pointer to the deserializer
         */
        CNvMDeserializer *deserializer;

        /**
         * The CSI port of SoC to which camera module is connected (via deserializer)
         */
        NvSiplCapInterfaceType eInterface;

        /**
         * The Power port of the camera module
         */
        uint32_t pwrPort;

        /**
         * A bitmask of which links this camera module object is responsible for initializing
         * Only applicable if group Init is enabled
         */
        uint8_t initLinkMask;

        /**
         * Whether group Init is enabled
         */
        bool groupInitProg;
    };

    /**
     * All connected device properties
     */
    typedef struct {
        /**
         * All sensor properties
         */
        typedef struct {
            /**
             * Sensor index
             */
            uint32_t id;
            /**
             * Virtual channel index from CSI, range [0,15]
             */
            uint32_t virtualChannelID;
            /**
             * Capture input format
             */
            NvSiplCapInputFormat inputFormat;
            /**
             * Capture Pixel Order
             */
            uint32_t pixelOrder;
            /**
             * Capture width in pixels
             */
            uint32_t width;
            /**
             * Capture height in pixels
             */
            uint32_t height;
            /**
             * Horizontal start position
             */
            uint32_t startX;
            /**
             * Vertical start position
             */
            uint32_t startY;
            /**
             * Number of top embedded lines
             */
            uint32_t embeddedTop;
            /**
             * Number of bottom embedded lines
             */
            uint32_t embeddedBot;
            /**
             * Frame rate for the sensor in FPS
             */
            float_t frameRate;
            /**
             * Whether embedded data type is enabled
             */
            bool embeddedDataType;
            /**
             * Handle for sensor control
             */
            ISensorControl *pSensorControlHandle;

            /**
             * Custom device interface for the sensor
             */
            IInterfaceProvider *pSensorInterfaceProvider;
        } SensorProperty;

        /**
         * All EEPROM properties
         */
        typedef struct {
            bool isEEPROMSupported = false;
        } EEPROMProperty;

        /**
         * Property information for each sensor
         */
        SensorProperty sensorProperty;

        /**
         * Property information for each EEPROM
         */
        EEPROMProperty eepromProperty;

        /**
         * Custom device interface for the serializer
        */
        IInterfaceProvider *pSerializerInterfaceProvider;
    } Property;

    /** Camera module API Major Revison */
    static constexpr uint32_t MAJOR_VER  = 1U;
    /** Camera module API Minor Revision */
    static constexpr uint32_t MINOR_VER  = 0U;
    /** Camera module API Patch Revision */
    static constexpr uint32_t PATCH_VER  = 0U;

    /**
     * Camera Module API version
     */
    struct Version
    {
        uint32_t uMajor = MAJOR_VER;
        uint32_t uMinor = MINOR_VER;
        uint32_t uPatch = PATCH_VER;
    };

    /**
     * @brief Default destructor of class CNvMCameraModule.
     */
    virtual ~CNvMCameraModule() = default;

    /**
     * @brief Prevent CNvMCameraModule class from being copy constructed.
     */
    CNvMCameraModule(const CNvMCameraModule &) = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being copy assigned.
     */
    CNvMCameraModule& operator=(const CNvMCameraModule &) = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being move constructed.
     */
    CNvMCameraModule(CNvMCameraModule &&) = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being move assigned.
     */
    CNvMCameraModule& operator=(CNvMCameraModule &&) = delete;

    /**
     * @brief Gets Camera module properties.
     *
     * It must be implemented by device specific camera module driver.
     * The returned pointer is valid for the lifetime of the camera module
     * object from which it is derived.
     *
     * @retval Property*   Pointer to this camera module's properties.
     * @retval nullptr     On internal failure in device specific implementation.
     */
    virtual Property* GetCameraModuleProperty() = 0;

    /**
     * @brief Stores configuration information and allocates VCID.
     *
     * The @ref SetConfig() must be called prior to initial powerup via @ref
     * SetPower().
     *
     * - This API does following:
     *   -# Validates input arguments.
     *   -# Invokes @ref DoSetConfig(). Device drivers must implement
     *      @ref DoSetConfig() for any device-specific configuration.
     *   -# Sets internal members of camera module instance, based on
     *      provided configuration in @ref cameraModuleCfg.
     *   -# Sets m_setConfigDone to true, on successful exit.
     *
     * @param[in] cameraModuleCfg       A pointer of type \ref CameraModuleConfig which holds
     *                                  CameraModuleConfig Module info used to configure object.
     *                                  It cannot be nullptr.
     *
     * @param[in] linkIndex             Link index of type \ref uint8_t to configure virtual
     *                                  channel ID.
     *                                  Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval    NVSIPL_STATUS_OK      On successful completion
     * @retval    (SIPLStatus)          Other propagated error status.
     *
     * @note This method does not validate internal fields of @a cameraModuleCfg
     * structure argument, it is caller's responsibility to provide valid values
     * for them, as specified by @ref CameraModuleConfig.
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
    virtual SIPLStatus SetConfig(const CameraModuleConfig *const cameraModuleCfg,
                                 const uint8_t linkIndex) final;

    /**
     * @brief Performs any initialization that needs to be done before the capture
     * pipeline is started.
     *
     * The camera module will be powered on when this is called, and \ref
     * SetConfig() will have been used to provide any required configuration
     * parameters. However, the link will not have been enabled. Initialization
     * that requires communication over the link should be deferred to \ref
     * PostInit().
     *
     * This API generally involves initializing associated serializers, sensors,
     * transport links, device detection and initialization. However exact behavior
     * is implementation defined.
     *
     * Must be implemented for a particular camera module.
     *
     * @retval  NVSIPL_STATUS_OK    On successful completion.
     * @retval  (SIPLStatus)        Other implementation-defined or propagated
     *                              error.
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
    virtual SIPLStatus Init() = 0;

    /**
     * @brief Performs any post initialization that needs to be done before the capture
     * pipeline is started.
     *
     * For any camera module, certain init operations can exist, which can only be
     * performed once deserializer links are enabled, so Device Block defines this
     * API to allow drivers to perform such post initialization sequence.
     * It is called after \ref Init() and after upstream deserializer links have
     * been enabled.
     *
     * Must be implemented for a particular camera module.
     *
     * @retval  NVSIPL_STATUS_OK    On successful completion.
     * @retval  (SIPLStatus)        Other implementation-defined or propagated
     *                              error. (Failure)
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
    virtual SIPLStatus PostInit() = 0;

    /**
     * Reconfigures the given camera module
     *
     * This API is expected to perform required reset and then initialize sequences
     * for this camera module. Exact behavior is implementation defined, however a
     * camera module should be in same state as after \ref PostInit() is called
     * during initial initialization.
     *
     * Must be implemented for a particular device specific camera module.
     *
     * @retval  NVSIPL_STATUS_OK    On successful completion.
     * @retval  (SIPLStatus)        Other implementation-defined or propagated error.
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
    virtual SIPLStatus Reconfigure() = 0;

    /**
     * @brief Reads from an EEPROM in the camera module.
     *
     * This is not used by Device Block itself, but Device Block clients who
     * have specific knowledge that a given camera module is being controlled
     * by a known driver can use it to read from the camera module's EEPROM
     * storage.
     * Must be implemented by device specific camera module. if driver
     * wishes, can return NVSIPL_STATUS_NOT_SUPPORTED in the device specific
     * implementation.
     *
     * @param[in]   address The EEPROM address (of type \ref uint16_t) to read from.
     *                      The valid ranges for this parameter are device-specific.
     * @param[in]   length  The length (of type \ref uint32_t) of the buffer, in bytes.
     *                      Valid ranges are dependent on `address` and are device
     *                      specific.
     * @param[out]  buffer  A buffer pointer (of type \ref uitn8_t) of at least `length`
     *                      bytes. It cannot be nullptr.
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval NVSIPL_STATUS_NOT_SUPPORTED  If the underlying driver does not
     *                                      support EEPROM readback. (Failure)
     * @retval (SIPLStatus)                 Other error codes returned by driver
     *                                      implementations. (Failure)
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
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
    virtual SIPLStatus ReadEEPROMData(const std::uint16_t address,
                                      const std::uint32_t length,
                                      std::uint8_t * const buffer) = 0;

#if !NV_IS_SAFETY
    //! Write to an EEPROM in the camera module
    virtual SIPLStatus WriteEEPROMData(const std::uint16_t address,
                                       const std::uint32_t length,
                                       std::uint8_t * const buffer) = 0;
#endif // !NV_IS_SAFETY

    /**
     * @brief Toggles LED on the given camera module.
     *
     * Must be implemented for a particular camera module
     *
     * @param[in]  enable        A flag (of type \ref bool), when true it indicates
     *                           LED is to be enabled, when false otherwise. Must be
     *                           either true or false.
     *
     * @retval     NVSIPL_STATUS_OK             On completion.
     * @retval     NVSIPL_STATUS_NOT_SUPPORTED  If not supported for the module.
     * @retval     (SIPLStatus)                 Other implementation-defined or propagated
     *                                          error codes.
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
    virtual SIPLStatus ToggleLED(bool const enable) = 0;

    /**
     * @brief Steps to be done after the capture pipeline has started
     *
     * Start transport links followed by sensors.
     *
     * Must be implemented for a particular camera module
     *
     * This will be called after the \ref Init() and \ref PostInit() operations
     * have completed.
     *
     * @retval  NVSIPL_STATUS_OK    On completion
     * @retval  (SIPLStatus)        Other implementation-defined or propagated error codes.
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
    virtual SIPLStatus Start() = 0;

    /**
     * @brief Steps to be done when quitting, before shutting down the capture pipeline.
     *
     * Stop sensors followed by transport links.
     *
     * Must be implemented for a particular camera module.
     * This will be called after \ref Start().
     *
     * @retval  NVSIPL_STATUS_OK    On completion
     * @retval  (SIPLStatus)        Other implementation-defined or propagated error codes.
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
    virtual SIPLStatus Stop() = 0;

    /**
     * @brief Deinitializes the camera module.
     *
     * Should deinitialize the camera sensors followed by the transport links
     * meant to be called after Stop().
     *
     * Must be implemented for a particular camera module.
     *
     * @retval  NVSIPL_STATUS_OK    On completion
     * @retval  (SIPLStatus)        Other implementation-defined or propagated error codes.
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
    virtual SIPLStatus Deinit() = 0;

    /**
     * @brief Returns a string to identify the supported deserializer.
     *
     * This API is expected to return the name of associated deserializer
     * (eg "MAX96712") with this camera module.
     *
     * Must be implemented for a particular camera module.
     *
     * @retval  (string)    Contains associated deserializer device name.
     * @retval  nullptr     On other implementation specific failures.
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
    virtual std::string GetSupportedDeserailizer() = 0;

    /**
     * @brief Creates the CDI devices needed by the camera module (serializer,
     * sensors, EEPROMs).
     *
     * This API validates inputs, creates all the CDI devices specific to this
     * camera module, and wire them all together via a \ref nvsipl::CNvMTransportLink.
     * The provided root device will be valid until after the camera module is
     * deleted, so @a cdiRoot can be stored by derived classes if required.
     *
     * This function updates the internal state machine, but all work is
     * delegated to \ref DoCreateCDIDevice(), which can be overridden by device
     * drivers, if they wish to provide device specific implementation.
     *
     * @param[in] cdiRoot               A pointer of type \ref DevBlkCDIRootDevice which holds
     *                                  CDI root device handle. Must be non-null.
     * @param[in] linkIndex             Index of the deserializer link to which this device is
     *                                  connected.
     *                                  Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval    NVSIPL_STATUS_OK      On successful completion.
     * @retval    (SIPLStatus)          Other implementation-defined or propagated error codes.
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
    virtual SIPLStatus CreateCDIDevice(DevBlkCDIRootDevice* const cdiRoot,
                                       const uint8_t linkIndex) final;

    /**
     * @brief After the device is powered on, wait for this many milliseconds
     * before programming.
     *
     * Must be implemented for a particular camera module
     *
     * @retval  (uint16_t)  Delay in milliseconds.
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
    virtual uint16_t GetPowerOnDelayMs() = 0;

    /**
     * @brief After the device is powered off, wait for this many milliseconds
     * before restarting the device.
     *
     * Must be implemented for a particular camera module to reset
     *
     * @retval  (uint16_t)  Delay in milliseconds.
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
     *   - De-Init: Yes
     */
    virtual uint16_t GetPowerOffDelayMs() = 0;

    /**
     * @brief Gets the type of link between the serializer and deserializer
     *
     * Must be implemented for a particular camera module
     *
     * @retval (LinkMode)  Link mode - either GMSL1 or GMSL2
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
    virtual CNvMDeserializer::LinkMode GetLinkMode() = 0;

    /**
     * @brief Powers on or off camera module devices.
     *
     * If the device is already powered on, this does nothing.
     * Drivers should implement \ref DoSetPower() if special behavior on power
     * up is required. Note that this *does not* perform device configuration
     * or teardown, see \ref Init() and \ref Deinit() for that.
     *
     * @retval  NVSIPL_STATUS_OK    On successful completion.
     * @retval  (SIPLStatus)        An implementation-defined propagated error code.
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
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus SetPower(bool const powerOn) final;

     /**
     * @brief Gets serializer error size.
     *
     * Gets size of serializer errors to be used by the client for allocating buffers.
     * Must be implemented for a particular camera module.
     *
     * @param[out] serializerErrorSize  Size (of type \ref size_t) of serializer error information
     *                                  (0 if no valid size found).
     *
     * @retval     NVSIPL_STATUS_OK            On successful completion
     * @retval     NVSIPL_STATUS_NOT_SUPPORTED If not implemented for a particular driver
     * @retval     (SIPLStatus)                Error status propagated from driver imeplementation.
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
    virtual SIPLStatus GetSerializerErrorSize(size_t & serializerErrorSize) = 0;

     /**
     * @brief Gets sensor error size.
     *
     * Gets size of sensor errors to be used by the client for allocating buffers.
     * Must be implemented for a particular camera module.
     *
     * @param[out] sensorErrorSize     Size (of type \ref size_t) of sensor error information
     *                                 (0 if no valid size found).
     *
     * @retval     NVSIPL_STATUS_OK            On successful completion
     * @retval     NVSIPL_STATUS_NOT_SUPPORTED If not implemented for a particular driver
     * @retval     (SIPLStatus)                Error status propagated from driver imeplementation.
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
    virtual SIPLStatus GetSensorErrorSize(size_t & sensorErrorSize) = 0;

     /**
     * @brief Gets detailed serializer error information and populates a provided buffer.
     *
     * This is expected to be called after the client is notified of errors.
     * Must be implemented for a particular camera module.
     *
     * @param[out] buffer       Buffer pointer (of type \ref uint8_t) to populate with error
     *                          information. It must be non-null.
     * @param[in]  bufferSize   Size (of type \ref size_t) of buffer to read to. Error buffer
     *                          size is device driver implementation specific.
     * @param[out] size         Size (of type \ref size_t) of data actually written to the buffer
     *
     * @retval     NVSIPL_STATUS_OK            On successful completion.
     * @retval     NVSIPL_STATUS_NOT_SUPPORTED If not implemented for a particular driver.
     * @retval     (SIPLStatus)                Error status propagated from driver imeplementation.
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
    virtual SIPLStatus GetSerializerErrorInfo(std::uint8_t * const buffer,
                                              const std::size_t bufferSize,
                                              std::size_t &size) = 0;

    /**
     * @brief Gets detailed sensor error information and populates a provided buffer.
     *
     * This is expected to be called after the client is notified of errors.
     * Must be implemented for a particular camera module.
     *
     * The client can choose to poll on this.
     *
     * @param[out] buffer       Buffer pointer (of type \ref uint8_t) to populate with error
     *                          information. It must be non-null.
     * @param[in]  bufferSize   Size (of type \ref size_t) of buffer to read to. Sensor error
     *                          buffer size is device driver implementation specific.
     * @param[out] size         Size (of type \ref size_t) of data actually written to the buffer.
     *
     * @retval      NVSIPL_STATUS_OK            On successful completion
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED If not implemented for a particular driver
     * @retval      (SIPLStatus)                Error status propagated from driver implementation.
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
    virtual SIPLStatus GetSensorErrorInfo(std::uint8_t * const buffer,
                                          const std::size_t bufferSize,
                                          std::size_t &size) = 0;

    /** Link states notified from SIPL Device Block */
    enum class NotifyLinkStates {
        /* The link will be disabled upon return from notification. */
        PENDING_DISABLE,
        /* The link has been enabled upon start of notification. */
        ENABLED
    };

    /**
     * @brief Invoked by SIPL Device Block to inform a camera module of a change in
     * link state.
     *
     * Must be implemented for a particular camera module.
     *
     * The behavior of this API is implementation defined.
     *
     * @param[in]   linkState           New state of the link (of type \ref NotifyLinkStates), to
     *                                  be updated to the camera module driver.
     *
     * @retval      NVSIPL_STATUS_OK    The notification was successfully received
     *                                  and acknowledged.
     * @retval      (SIPLStatus)        An implementation-specific error
     *                                  occured while switching to the new link
     *                                  state (Failure).
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
    virtual SIPLStatus NotifyLinkState(const NotifyLinkStates linkState) = 0;

protected:
    /**
     * Default constructor to instantiate CNvMCameraModule objects.
     */
    CNvMCameraModule() : IInterfaceProvider() {};

    /**
     * Holds power port number
     */
    std::uint32_t m_pwrPort{UINT32_MAX};

    /**
     * Holds device parameters
     */
    CNvMDevice::DeviceParams m_params{};

    /**
     * Holds camera module's link index
     */
    uint8_t m_linkIndex{0U};

private:
    /**
     * @brief Device specific implementation to create CDI Devices for a
     * camera module.
     *
     * If required, camera module drivers can override this method to provide
     * device specific implementation.
     *
     * @param[in] cdiRoot            A pointer of type \ref DevBlkCDIRootDevice which holds
     *                               CDI root device handle. Must be non-null.
     * @param[in] linkIndex          Link index of a camera module.
     *                               Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval    NVSIPL_STATUS_OK   On completion
     * @retval    (SIPLStatus)       Other propagated error codes.
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
    virtual SIPLStatus DoCreateCDIDevice(DevBlkCDIRootDevice* const cdiRoot, const uint8_t linkIndex);

    /**
     * @brief Device specific implementation to set configuration params for each
     * camera module.
     *
     * This should only copy the camera module configuration to internal
     * storage. The link to the module will not yet be powered up, so the
     * actual work of configuring the device must be deferred until @ref
     * Init().
     *
     * It must be implemented by device specific camera module driver.
     *
     * @param[in] cameraModuleCfg       Module info (of type \ref CameraModuleConfig) used to
     *                                  configure object. The input pointer is valid for the
     *                                  duration of @ref SetConfig(), but should not be stored.
     * @param[in] linkIndex             Link index to configure virtual channel ID.
     *                                  Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval    NVSIPL_STATUS_OK      On completion
     * @retval    (SIPLStatus)          Other propagated error codes.
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
    virtual SIPLStatus DoSetConfig(const CameraModuleConfig *const cameraModuleCfg,
                                   const uint8_t linkIndex) = 0;

    /**
     * @brief Device specific implementation to Power on or off camera module devices
     *
     * The camera module subclasses must implement it to hook desired
     * power control backend.
     *
     * When @a powerOn is true, device-specific implementations use this to
     * perform any operations required to prepare the camera module (and its
     * component devices) for I2C communication (enable power gates, request
     * ownership from a board management controller, etc.). However, the camera
     * module will not be initialized when this is invoked, so implementations
     * should not assume Init() has been performed.
     *
     * When @a powerOn is false, the inverse of the operations for powerup
     * should be performed. This will be called after Deinit().
     *
     * @param[in] powerOn               Flag (of type \ref bool) to indicate, whether to power on
     *                                  the device (true) or power it off (false).
     *
     * @retval    NVSIPL_STATUS_OK      On successful completion.
     * @retval    (SIPLStatus)          Other propagated error codes.
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
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus DoSetPower(bool const powerOn) = 0;

    /**
     * @brief Tracks whether the device is powered on.
     *
     * This should be updated by @ref SetPower().
     */
    bool m_isPoweredOn{false};

    /**
     * @brief Tracks whether we have been provided with configuration
     * information.
     *
     * This should be set by @ref SetConfig
     */
    bool m_setConfigDone{false};
};

/** @} */

} // end of namespace

#endif //CNVMCAMERAMODULE_HPP
