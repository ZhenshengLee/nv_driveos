/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
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
    /** @brief Describes a single globally-visible crypto key object.
     */
    class ModuleCryptoKeyInfo {
        /** @brief Holds a purpose of the key. Supported purposes are specific to
         sensor model. The purpose is communicated to sensor driver which must know how
         to create and use a key of the specified "purpose". */
        std::string keyPurpose;
        /** @brief Holds a name of a key. Name is unique and global within a process.
         If the same name is specified for multiple modules in the same config - the
         same key object will be shared. */
        std::string keyName;
        /** @brief Unique ID identifying a channel. */
        uint32_t ivcNumber;
        /** @brief An ID of a crypto hardware engine which services the associated key */
        uint32_t cryptoHwEngine;
        /** @brief Size of a buffer used to service requests on that channel. */
        uint32_t mapBufSize;
        /** @brief A group ID of a crypto channel, used for access permission. */
        uint32_t groupID;
        /** @brief ID if an SMMU instance which services all memory access requests
         for the channel */
        uint32_t smmuHwID;
        /** @brief SMMU streamID used for all memory accesses associated with the channel. */
        uint32_t streamID;

    public:
        /** @brief Construct an entry describing single cryptographic key config  */
        ModuleCryptoKeyInfo(std::string const &purpose,
                            std::string const &name,
                            uint32_t const ivc,
                            uint32_t const engine,
                            uint32_t const bufSize,
                            uint32_t const gid,
                            uint32_t const smmuID,
                            uint32_t const sid):
            keyPurpose(purpose), keyName(name), ivcNumber(ivc), cryptoHwEngine(engine),
            mapBufSize(bufSize), groupID(gid),
            smmuHwID(smmuID), streamID(sid) { }

        /** @brief Return a string signifying purpose of the key */
        std::string const &getPurpose() const noexcept {
            return keyPurpose;
        };
        /** @brief Return name of the key */
        std::string const &getName() const noexcept {
            return keyName;
        };
        /** @brief Return crypto ivcNumber the key is serviced with */
        uint32_t getIVC() const noexcept {
            return ivcNumber;
        };
        /** @brief Return crypto channel ID the key is serviced with */
        uint32_t getGroupID() const noexcept {
            return groupID;
        };
        /** @brief Return HW ID of cryptographic engine which services all
         * operations with that key */
        uint32_t getCryptoEngineID() const noexcept {
            return cryptoHwEngine;
        };
        /** @brief Returns true if direct memory access by crypto engine is
         * supported for that key. */
        bool getIsZeroCopy() const noexcept {
            return mapBufSize == 0U;
        };
        /** @brief Return HW ID of SMMU instance servicing memory accesses for
         * operations on the key */
        uint32_t getSmmuHWID() const noexcept {
            return smmuHwID;
        };
        /** @brief Return SMMU streamID used for servicing memory accesses for
         * operations on the key */
        uint32_t getSmmuSID() const noexcept {
            return streamID;
        };
    };

    /**
     * struct contains necessary information for Camera Module configuration.
     */
    struct CameraModuleConfig {
        /**
         * Settings specific to this camera module
         */
        const CameraModuleInfo *cameraModuleInfos;

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

        /**
         * Pointer to a list of crypto keys specified for the platform
         */
        std::vector<ModuleCryptoKeyInfo> *cryptoKeysListP;
    };

    /**
     * All connected device properties
     */
    struct Property {
        /**
         * All sensor properties
         */
        struct SensorProperty {
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
             * Whether sensor authentication must be performed
             */
            bool isAuthEnabled;
            /**
             * Indicates which "thread ID" to use for image authentication work.
             *
             * threadID is an abstract identification number for camera authentication
             * thread which is global for the camera app process. It is not related to
             * an Operating System concept of a thread ID.
             *
             * All Camera Module instances which specified the same threadID will be
             * handled by the same authentication thread sequentially on
             * first-come-first-serve basis.
             * Specifying different threadID allows driver implementation to parallelize
             * authentication work to multiple threads.
             *
             * One Camera Module object can be handled only by a single thread, as
             * indicated by a threadID. Only that Thread will invoke the implementation's
             * DevBlkCDIDeviceDriver::AuthenticateImage() method.
             *
             * If not specified - authentication work will be handled by a single default
             * thread with ID 0.
             */
            uint8_t imgAuthThreadID;
            /**
             * Bitmap of CPU cores an image authentication thread can execute on.
             * The setting only makes sense when image authentication is implemented by
             * the module driver. If driver defines an image authentication method - the
             * method will be invoked by a thread running only on CPU cores specified
             * by imgAuthAffinity mask.
             *
             * Depending on implementation of authentication procedure (which is driver
             * specific) there can be a performance benefit from doing authentication work
             * on a limited number of CPU cores.
             * For example handling image authentication work for a single camera on a
             * specific CPU core can result in more efficient Dcache usage.
             * As another example, using hardware acceleration interface can be more
             * efficient if crypto work is submitted by the same CPU core which will be
             * communicating with HW engine as this avoids extra context switches.
             *
             */
            uint32_t imgAuthAffinity;
            /**
             * Handle for sensor control
             */
            ISensorControl *pSensorControlHandle;

            /**
             * Custom device interface for the sensor
             */
            IInterfaceProvider *pSensorInterfaceProvider;
        };

        /**
         * All EEPROM properties
         */
        struct EEPROMProperty{
            bool isEEPROMSupported = false;
        };

        /**
         * Property information for each sensor
         */
        SensorProperty sensorProperties;

        /**
         * Property information for each EEPROM
         */
        EEPROMProperty eepromProperties;

        /**
         * Custom device interface for the serializer
        */
        IInterfaceProvider *pSerializerInterfaceProvider;
    };

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
    ~CNvMCameraModule() override = default;

    /**
     * @brief Prevent CNvMCameraModule class from being copy constructed.
     */
    CNvMCameraModule(const CNvMCameraModule &) = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being copy assigned.
     */
    CNvMCameraModule& operator=(const CNvMCameraModule &) & = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being move constructed.
     */
    CNvMCameraModule(CNvMCameraModule &&) = delete;

    /**
     * @brief Prevent CNvMCameraModule class from being move assigned.
     */
    CNvMCameraModule& operator=(CNvMCameraModule &&) & = delete;

    /**
     * @brief Gets Camera module properties.
     *
     * It must be implemented by device specific camera module driver.
     * The returned pointer is valid for the lifetime of the camera module
     * object from which it is derived.
     *
     * @pre A valid camera module property must be configured.
     *
     * @retval Property*   Pointer to this camera module's properties.
     * @retval nullptr     On internal failure in device specific implementation.
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
     * @pre A valid camera module configuration must be provided.
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
                                 uint8_t const linkIndex);

    /**
     * @brief Enables physical link and detects a Module
     *
     * Enables Transport layer link connection to a camera module and tries to
     * detect the module on that transport link.
     * The camera module and any hardware components of a Transport link will be
     * powered-ON before this interface is called.
     *
     * This API generally involves initializing associated serializer, enabling
     * corresponding link on the platform deserializer, executing camera device
     * detection sequence. Exact behavior is implementation defined and depends
     * on how physical connection with a camera Module is organized.
     *
     * Must be implemented by device specific camera module. Implementation is
     * expected to NOT perform any initialization after module was detected in
     * this method.
     *
     * @pre A valid transportation link must be configured.
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
    virtual SIPLStatus EnableLinkAndDetect(void) = 0;

    /**
     * @brief Authenticate a module
     *
     * The camera module must be powered-on and be able to reply to control
     * commands in order to be authenticated.
     *
     * @pre A valid camera module object created with CNvMCameraModule_Create().
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
    virtual SIPLStatus Authenticate(void) noexcept
    {
        return NVSIPL_STATUS_OK;
    }

    /**
     * @brief Create a broadcast sensor object
     *
     * Creates a sensor object on the module which can be used for a broadcast
     * initialization.
     * Broadcasting allows to initialize multiple camera modules with a single
     * initialization sequence. All camera modules initialized in such a way must
     * be identical and configured with the same settings.
     *
     * If a broadcast sensor was created on a module with a call to this API -
     * it is expected that a consequent call to ::Init() will initialize not
     * only the module specified but all modules in a group.
     *
     * @pre A valid camera module object created with CNvMCameraModule_Create().
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
    virtual CNvMDevice *CreateBroadcastSensor() = 0;

    /**
     * @brief Register a listener for control bus transactions
     *
     * Register control bus listener on a specified srcSensor sensor object.
     * After registration a listening module will be notified of any control bus
     * transactions (i2c reads and writes) performed by srcSensor device.
     *
     * @pre A valid camera module object created with CNvMCameraModule_Create().
     *
     * @retval NVSIPL_STATUS_OK            On successful completion.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT  Invalid input parameter (e.g., srcSensor points
     *                                     to an object of a type different from self).
     * @retval NVSIPL_STATUS_NOT_SUPPORTED In case CameraModule driver does not support
     *                                     control bus listeners
     * @retval (SIPLStatus)        Other implementation-defined or propagated
     *                             error.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
    virtual SIPLStatus
    RegisterControlListener(const CNvMDevice * const srcSensor) const noexcept
    {
        return NVSIPL_STATUS_OK;
    }

    /**
     * @brief Performs any initialization that needs to be done before the capture
     * pipeline is started.
     *
     * Applies camera module configuration corresponding to settings previously set
     * with SetConfig. After a successful return from that call the camera module shall
     * be properly configured and ready to start streaming.
     *
     * Configuration passed with SetConfig is converted to device-specific settings
     * by a driver. Often this means that any configuration is converted to a sequence
     * of i2c registers writes from a driver to a camera module.
     * The interface shall also take care of any unspecified device-specific
     * configuration, setting those to "sane defaults" in order to get resonable default
     * image quality. This can also involve applying any fixed "calibration" data.
     *
     * Must be implemented for a particular camera module.
     *
     * @pre A valid camera module object created with CNvMCameraModule_Create().
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
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
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
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
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
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
     *
     * @param[in]   address The EEPROM address (of type \ref uint16_t) to read from.
     *                      The valid ranges for this parameter are device-specific.
     * @param[in]   length  The length (of type \ref uint32_t) of the buffer, in bytes.
     *                      Valid ranges are dependent on 'address' and are device
     *                      specific.
     * @param[out]  buffer  A buffer pointer (of type \ref uitn8_t) of at least 'length'
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
    virtual SIPLStatus ReadEEPROMData(std::uint16_t const address,
                                      std::uint32_t const length,
                                      std::uint8_t * const buffer) = 0;

#if !NV_IS_SAFETY
    //! Write to an EEPROM in the camera module
    virtual SIPLStatus WriteEEPROMData(const std::uint16_t address,
                                       const std::uint32_t length,
                                       std::uint8_t * const buffer) = 0;

    /**
     * @brief Toggles LED on the given camera module.
     *
     * Must be implemented for a particular camera module
     *
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
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
#endif // !NV_IS_SAFETY

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
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
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
     * @brief Steps to be done to finalize camera streaming start
     *
     * Additional steps required to start sensor streaming after Start has finished.
     * The difference with Start is that PostStart will always be called for individual
     * camera object, while Start can be called for a broadcst camera object controlling
     * multiple cameras at the same time. PostStart gives a chance to perform
     * instance-specific initialization steps.
     *
     * Optional for a particular camera module.
     *
     * This will be called after the \ref Start() operation has successfully completed.
     *
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Start() and before @ref Deinit().
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
    virtual SIPLStatus PostStart() noexcept
    {
        return NVSIPL_STATUS_OK;
    }

    /**
     * @brief Steps to be done when quitting, before shutting down the capture pipeline.
     *
     * Stop sensors followed by transport links.
     *
     * Must be implemented for a particular camera module.
     * This will be called after \ref Start().
     *
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Start().
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
     * @pre
     *   - A valid camera module object created with CNvMCameraModule_Create().
     *   - This function must be called only after @ref Init().
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
     * @pre None.
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
     * @pre @a cdiRoot must be valid DevBlkCDIRootDevice handle created with DevBlkCDIRootDeviceCreate().
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
                                       const uint8_t linkIndex);

    /**
     * @brief After the device is powered on, wait for this many milliseconds
     * before programming.
     *
     * Must be implemented for a particular camera module
     *
     * @pre A valid delay value must be set.
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
     * @pre A valid delay value must be set.
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
     * @pre A valid transportation link must be configured.
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
     * @pre A valid camera module power control method is provided.
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
    virtual SIPLStatus SetPower(bool const powerOn);

     /**
     * @brief Gets serializer error size.
     *
     * Gets size of serializer errors to be used by the client for allocating buffers.
     * Must be implemented for a particular camera module.
     *
     * @pre A valid serializer device object must be created.
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
     * @pre A valid sensor device object must be created.
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
     * @pre A valid serializer device object must be created.
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
                                              std::size_t const bufferSize,
                                              std::size_t &size) = 0;

    /**
     * @brief Gets detailed sensor error information and populates a provided buffer.
     *
     * This is expected to be called after the client is notified of errors.
     * Must be implemented for a particular camera module.
     *
     * The client can choose to poll on this.
     *
     * @pre A valid sensor device object must be created.
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
                                          std::size_t const bufferSize,
                                          std::size_t &size) = 0;

    /** Link states notified from SIPL Device Block */
    enum class NotifyLinkStates : std::uint8_t {
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
     * @pre
     *   - A valid transpotation link must be created.
     *   - This function must be called only after @ref Init() and before @ref Deinit().
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

    /**
     * @brief Returns a pointer to a key used for image authentication
     *
     * Returns a pointer to an information entry describing a crypto key which
     * was indicated as "used for image authentication" by low level device driver.
     *
     * @retval (pointer)  A pointer to image auth key. Can be nullptr if not set.
     */
    ModuleCryptoKeyInfo const *GetImgAuthKey() const noexcept
    {
        return keyForImgAuth;
    }

protected:
    /**
     * Default constructor to instantiate CNvMCameraModule objects.
     */
    CNvMCameraModule() : IInterfaceProvider(), IInterruptStatus() {};

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

    /**
     * @brief Get a list of crypto keys for the sensor
     *
     * Returns a constant reference to a list of entries describing
     * configuration of crypto keys associated with a sensor.
     *
     * @retval Constant reference to a list of crypto keys info entries.
     */
    const std::vector<ModuleCryptoKeyInfo> &GetCryptoKeys() const noexcept
    {
        return m_cryptoKeysList;
    }

    /**
     * @brief Indicate which key is used for image authentication
     *
     * The API can be used by low level camera driver to indicate which crypto
     * key is used for image authentication. Full list of keys for the module
     * is available through GetCryptoKeys(), but only low level driver can
     * interpret purpose of each key.
     * Configuration of image auth key might be important for creating image
     * buffers mappings.
     *
     * @param[in] key A pointer to a key info entry describing image
     *                authentication key.
     */
    void SetImgAuthKey(ModuleCryptoKeyInfo const * const key) noexcept
    {
        keyForImgAuth = key;
    }

private:
    /**
     * @brief Device specific implementation to create CDI Devices for a
     * camera module.
     *
     * If required, camera module drivers can override this method to provide
     * device specific implementation.
     *
     * @pre @a cdiRoot must be valid DevBlkCDIRootDevice handle created with DevBlkCDIRootDeviceCreate().
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
    virtual SIPLStatus DoCreateCDIDevice(DevBlkCDIRootDevice* const cdiRoot, uint8_t const linkIndex);

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
     * @pre A valid camera module configuration must be provided.
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
                                  uint8_t const linkIndex) = 0;

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
     * @pre A valid camera module power control method is provided.
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

    /**
     * A list of information about cryptographic keys used for module operation.
     * Keys usage is defined by an associated module driver. For example, keys can be used
     * for image authentication.
     */
    std::vector<ModuleCryptoKeyInfo> m_cryptoKeysList;

    /**
     * A pointer to an information entry for a key used for image authentication.
     * Device specific driver can identify which key to be used for image data access
     * to let camera stack know if DMA mapping shall be created for image buffers for
     * direct crypto hardware access.
     */
    ModuleCryptoKeyInfo const *keyForImgAuth{nullptr};
};

/** @} */

} // end of namespace

#endif //CNVMCAMERAMODULE_HPP
