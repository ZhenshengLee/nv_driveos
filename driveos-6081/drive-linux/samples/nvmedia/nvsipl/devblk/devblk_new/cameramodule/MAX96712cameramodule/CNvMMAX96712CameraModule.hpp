/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMMAX96712CAMERAMODULE_HPP
#define CNVMMAX96712CAMERAMODULE_HPP

#include "ModuleIF/CNvMCameraModule.hpp"
#include "SensorIF/CNvMSensor.hpp"
#include "EEPROMIF/CNvMEEPROM.hpp"
#include "PMICIF/CNvMPMIC.hpp"
#include "VCSELIF/CNvMVCSEL.hpp"
#include "SerializerIF/CNvMSerializer.hpp"
#include "TransportLinkIF/CNvMTransportLink.hpp"
#include "NvSIPLPlatformCfg.hpp"
#include "CNvMCameraModuleCommon.hpp"
#include "CampwrIF/CNvMCampwr.hpp"
#include "cdd_nv_error.h"

namespace nvsipl
{

class CNvMMAX96712CameraModule : public CNvMCameraModule
{
public:

    ~CNvMMAX96712CameraModule() override = default;
    CNvMMAX96712CameraModule() = default;
    CNvMMAX96712CameraModule(CNvMMAX96712CameraModule const &) = delete;
    CNvMMAX96712CameraModule(CNvMMAX96712CameraModule &&) = delete;
    CNvMMAX96712CameraModule &operator = (CNvMMAX96712CameraModule const &) & = delete;
    CNvMMAX96712CameraModule &operator = (CNvMMAX96712CameraModule &&) & = delete;

    /** @brief Set the Camera moduleobject's configuration
     * - Save device params and serializer info
     *  - m_oDeviceParams = *cameraModuleCfg->params
     *  - m_oSerInfo = cameraModuleCfg->cameraModuleInfo->serInfo
     *  - m_oSensorInfo = cameraModuleCfg->cameraModuleInfo->sensorInfo
     *  - m_pDeserializer = cameraModuleCfg->deserializer
     *  - m_sModuleName = cameraModuleCfg->cameraModuleInfo->name
     *  .
     * - Set CameraModuleInfo const* const moduleInfo =
     *  - cameraModuleCfg->cameraModuleInfo
     *  .
     * - Config serializer
     *  - m_upSerializer = std::move(CreateNewSerializer())
     *  .
     * - set m_oDeviceParams.bUseNativeI2C = false
     * - set serializer configuration
     *  - status = m_upSerializer->SetConfig(
     *   - &moduleInfo->serInfo,
     *   - &m_oDeviceParams)
     *   .
     *  .
     * - Configure Sensor
     *  - m_upSensor.reset(new CNvMSensor())
     *  - m_upSensor->SetConfig(
     *   - &moduleInfo->sensorInfo,
     *   - cameraModuleCfg->params)
     *   .
     *  .
     * - configure module
     *  - SetConfigModule(
     *   - &moduleInfo->sensorInfo,
     *   - cameraModuleCfg->params)
     *   .
     *  .
     * - set driver handle
     *  - m_upSensor->SetDriverHandle(GetCDIDeviceDriver())
     *  .
     * - ser driver context
     *  - m_upSensor->SetDriverContext(GetCDIDeviceContext())
     *  .
     * - verify moduleInfo->linkIndex <= 7
     * - Set Link Index
     *  - m_linkIndex = static_cast<uint8_t>(
     *   - moduleInfo->linkIndex)
     *   .
     *  .
     * - set m_initLinkMask = cameraModuleCfg->initLinkMask
     * - set m_groupInitProg = cameraModuleCfg->groupInitProg
     * - if moduleInfo->isEEPROMSupported
     *  - initiate an eeprom object
     *   - m_upEeprom.reset(new CNvMEEPROM())
     *   .
     *  - set eeprom configuration
     *   - m_upEeprom->SetConfig(
     *    - &moduleInfo->eepromInfo,
     *    - cameraModuleCfg->params)
     *    .
     *   .
     *  - get driver handle for eeprom
     *   - m_upEeprom->SetDriverHandle(
     *    - GetCDIDeviceDriver(
     *     - MODULE_COMPONENT_EEPROM))
     *     .
     *    .
     *   .
     *  .
     * - SetupConnectionProperty(cameraModuleCfg, linkIndex)
     *
     * @param[in] cameraModuleCfg specific to camera module
     * @param[in] linkIndex link Index
     * @return NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus DoSetConfig(CameraModuleConfig const* const cameraModuleCfg,
                           uint8_t const linkIndex) final;

    /** @brief Run backends power sequence, default is NVCCP
     *
     * - if (!m_camPwrControlInfo.method
     *  - Note: Default is NvCCP, other power backends can be used
     *   - here based on platform/usecase.
     *   .
     *  - PowerControlSetUnitPower(
     *   - m_pwrPort,
     *   - m_linkIndex, powerOn)
     *   .
     *  .
     * - elif (m_camPwrControlInfo.method == UINT8_MAX)
     *  - exit ok
     *  .
     * - elif (m_upCampwr->isSupported() == NVSIPL_STATUS_OK)
     *  - m_upCampwr->PowerControlSetUnitPower(
     *   - m_upCampwr->GetDeviceHandle(),
     *   - m_linkIndex,
     *   - powerOn)
     *   .
     *  .
     *
     * @param[in]  powerOn true is power on, false is power off [true, false]
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
    */
    SIPLStatus DoSetPower(bool const powerOn) override;

    /** @brief Enables physical link and detects a Module
     *
     * Enables Transport layer link to a camera module and tries to detect that
     * module on the transport link.
     * The camera module and any hardware components of a Transport link between
     * the module and the Platform must be powered-ON before this interface is
     * called.
     *
     * This API involves initializing associated serializers,
     * transport links, device detection.
     *
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus EnableLinkAndDetect() final;

    /** @brief Camera module initialization
     *
     * Initializes camera module according to provided settings.
     * Initialization usually configures all internal module components like
     * image sensor, EEPROM, etc.
     *
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus Init() final;

    CNvMDevice *CreateBroadcastSensor() final;

    /** @brief This is a NO-OP for the scope of CDD.
     * The base class exposes this function for camera modules
     * which need to do post initialization steps before starting streaming.
     * - m_upTransport->MiscInit()
     * - PostInitModule()
     * - set m_groupInitProg = false
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus PostInit() final;

    /** @brief start streaming
     * - Start the individual transport links
     *  - m_upTransport->Start()
     *  .
     * - StartModule()
     *
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus Start() final;

    /** @brief Stop streaming
     * - Stop the sensors
     *  - StopModule()
     *  .
     * - Stop the transport links
     *  - m_upTransport->Stop()
     *  .
     *
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus Stop() final;

    /** @brief Camera module re-configured after reset Sensor / Serializer / EEPROM / Transport
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus Reconfigure() final;

    /** @brief Read EEPROM data
     * - if (!m_upEeprom)
     *  - m_upEeprom->ReadData(address, length, buffer)
     *  .
     * - else
     *  - return error NVSIPL_STATUS_NOT_SUPPORTED
     *  .
     * .
     *
     * @param[in] address The start address to read from. Valid ranges dependon
     * the driver backing this device.
     * @param[in] length  The length of data to read, in bytes.May not be zero.
     * @param[out] buffer Data buffer. Must be at least @a lengthbytes long.Must not be null.
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return NVSIPL_STATUS_NOT_SUPPORTED not available EEPROM handle
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus ReadEEPROMData(std::uint16_t const address,
                              std::uint32_t const length,
                              std::uint8_t * const buffer) noexcept override;

/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if !(NV_IS_SAFETY)
    SIPLStatus WriteEEPROMData(std::uint16_t const address,
                               std::uint32_t const length,
                               std::uint8_t * const buffer);
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */

    /** @brief Toggle LED - function not supported */
    SIPLStatus ToggleLED(bool const enable) noexcept override;
#endif // !(NV_IS_SAFETY)
    /** @brief Deinitialize the module.
     * - Deinit the sensors
     *  - DeinitModule()
     *  .
     * - Deinit hte transport links
     *  - m_upTransport->Deinit()
     *  .
     * .
     *
     * @return  NVSIPL_STATUS_OK  on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus Deinit() final;

    /** @brief return camera module property
     * - return m_upCameraModuleProperty.get()
     * .
     * @return  Property *Camera module property
     */
    Property* GetCameraModuleProperty() noexcept final;

    /** @brief Gets the module power on delay (in milliseconds)
     * @return  uint16_t Power on delay in milliseconds
     */
    uint16_t GetPowerOnDelayMs() override;

    /** @brief Gets the module power offdelay (in milliseconds)
     * @return  uint16_t Power offdelay in milliseconds
     */
    uint16_t GetPowerOffDelayMs() noexcept override;

    /** @brief Gets the supported Deserializer device name
     * - return "MAX96712"
     * .
     *
     * @return  std::string Supported Deserializer device name
     */
    std::string GetSupportedDeserailizer() noexcept override;

    /** @brief Get the serializer error payload size
     * - m_upSerializer->GetErrorSize(serializerErrorSize)
     * .
     * @param[out] serializerErrorSize Error payload size
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus GetSerializerErrorSize(size_t & serializerErrorSize) final;

    /** @brief Get the sensor error payload size
     * - GetErrorSize(sensorErrorSize)
     * .
     *
     * @param[out] sensorErrorSize pointer to return payload size
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus GetSensorErrorSize(size_t & sensorErrorSize) final;

    /** @brief Get the Error information from the serializer
     * - m_upSerializer->GetErrorInfo(buffer, bufferSize, size)
     * .
     *
     * @param[in]  buffer Error buffer to be filled.
     * @param[in]  bufferSize Error buffer payload size.
     * @param[out]  size Filled size.
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus GetSerializerErrorInfo(std::uint8_t * const buffer,
                                      std::size_t const bufferSize,
                                      std::size_t &size) final;

    /** @brief Get the Error information from the serializer
     * - GetErrorInfo(buffer, bufferSize, size)
     *
     * @param[in]  buffer Error buffer to be filled.
     * @param[in]  bufferSize Error buffer payload size.
     * @param[out]  size Filled size.
     * @return  NVSIPL_STATUS_OK on successful completion
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus GetSensorErrorInfo(std::uint8_t * const buffer,
                                  std::size_t const bufferSize,
                                  std::size_t &size) final;

    /**  Notify the link state
     * - if link state == NotifyLinkStates::ENABLED
     *  - Start()
     *  .
     * - elif link state == NotifyLinkStates::PENDING_DISABLE
     *  - Stop()
     *  .
     * - else
     *  - error return NVSIPL_STATUS_BAD_ARGUMENT
     *  .
     * .
     *
     * @param[in]  linkState link state
     * @return  NVSIPL_STATUS_OK on successful completion [ENABLED, PENDING_DISABLE]
     * @return NVSIPL_STATUS_BAD_ARGUMENT out of argument range
     * @return (SIPLStatus) other error propagated
     */
    SIPLStatus NotifyLinkState(NotifyLinkStates const linkState) final;

    /** @brief Gets custom interface handle
     * NOT IMPLEMENTED
     * @param[in]  interfaceId custom interface UUID value
     * @return Interface* null pointer if not implemented
     */
    Interface* GetInterface(UUID const &interfaceId) override;

    using IInterruptStatus::GetInterruptStatus;

    SIPLStatus GetInterruptStatus(
        uint32_t const gpioIdx,
        IInterruptNotify &intrNotifier) noexcept override;

protected:
    /** enumeration of components IDs */
    enum class ModuleComponent : std::uint8_t {
        MODULE_COMPONENT_SENSOR = 0,
        MODULE_COMPONENT_EEPROM,
        MODULE_COMPONENT_PMIC,
        MODULE_COMPONENT_VCSEL
    };

    /** @brief Set thesensorConnectionProperty to initialize
     * @param[in] sensorConnectionProperty  connection properites
     * @return NVSIPL_STATUS_OK : on successful completion
     * @return NVSIPL_STATUS_BAD_ARGUMENT : if input argument is invalid
     * @return (SIPLStatus) : other implementation-determined or propagated error
    */
    virtual SIPLStatus SetSensorConnectionProperty(
        CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty
            * const sensorConnectionProperty) const noexcept = 0 ;

    /** @brief Configuration specific module characteristic
     * Must be implemented for a particular module
     * @param[in] sensorInformation :image sensor information
     * @param[in] params contains iformation for communication link with the device
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus SetConfigModule(
        SensorInfo const* const sensorInformation,
        CNvMDevice::DeviceParams const* const params) = 0;

    /** @brief Detects if the sensor module is present.
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus DetectModule() = 0;

    /** @brief Initialize the module.
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus InitModule() const = 0;

    /** @brief PostInit the module
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus PostInitModule() = 0;

    /** @brief Start the module streaming.
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus StartModule() = 0;

    /** @brief Stop the module streaming.
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus StopModule() = 0;

    /** @brief Deinitialize themodule.
     * Must be implemented for a particular module
     * @return NVSIPL_STATUS_OK : on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    virtual SIPLStatus DeinitModule() = 0;


    /** @brief Get the CDI device driver handle for thesensor.
     * Must be implemented for a particular module
     * @return DevBlkCDIDeviceDriver*: CDI device driver handle.
     */
    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver() const = 0;

    /** @brief Get the CDI device driver handle of a particular component.
     * Must be implemented for a particular module
     * @param[in]  component :Module component type. (Sensor/EEPROM/PMIC)
     * @return DevBlkCDIDeviceDriver*: CDI device driver handle.
     */
    virtual DevBlkCDIDeviceDriver *GetCDIDeviceDriver(ModuleComponent const component) const = 0;

    /** @brief Gets the CDI device context handle.
     * Must be implemented for a particular module
     * @return std::unique_ptr<CNvMDevice::DriverContext>:CDI device context handle.
     */
    virtual std::unique_ptr<CNvMDevice::DriverContext> GetCDIDeviceContext() const = 0;

    /** @brief Create a New Serializer object.
     * Must be implemented for a particular module
     * @return std::unique_ptr<CNvMSerializer>:New serializer object.
     */
    virtual std::unique_ptr<CNvMSerializer> CreateNewSerializer() const = 0;

    /** @brief Create a New Transport Link object.
     * Must be implemented for a particular module
     * @return std::unique_ptr<CNvMTransportLink>:New transport link object.
     */
    virtual std::unique_ptr<CNvMTransportLink> CreateNewTransportLink() const = 0;

    /** @brief Create a New Serializer object
     * - extract CNvMDevice::DeviceParams params = m_oDeviceParams
     * - set params.bUseNativeI2C = true
     * - create up serializer
     *  - std::unique_ptr<CNvMSerializer> up_var = std::move(CreateNewSerializer())
     *  .
     * - up_var->SetConfig(&m_oSerInfo, &params)
     * - up_var->CreateCDIDevice(m_pCDIRoot, linkIndex)
     * - return serializer object
     *  - serializer = std::move(up_var)
     *  .
     * .
     * @param[in] linkIndex : Link index [0, 3]
     * @param[out] serializer : Created broadcast serializer handle
     * @return NVSIPL_STATUS_OK: on completion
     * @return (SIPLStatus)    : other implementation-determined or propagated error
     */
    SIPLStatus CreateBroadcastSerializer(uint8_t const linkIndex,
                                         std::unique_ptr<CNvMSerializer>& serializer);

    /** @brief Get the error payload size.
     *
     * @param[out] sensorErrorSize : Error payload size.
     * @return NVSIPL_STATUS_OK:Get was successful.
     */
    virtual SIPLStatus GetErrorSize(size_t & sensorErrorSize) const = 0 ;

    /** @brief Get the Error information from the module.
     *
     * @param[in] buffer :Error buffer to be filled.
     * @param[in] bufferSize :Error buffer payload size.
     * @param[out] size :Filled size.
     * @return NVSIPL_STATUS_BAD_ARGUMENT:Input argument is invalid.
     * @return NVSIPL_STATUS_OK:Get error information was successful.
     * @return NVSIPL_STATUS_ERROR:Get error information failed.
     */
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                                    std::size_t const bufferSize,
                                    std::size_t &size) const = 0 ;

    /** @brief returns module's name
     * @return std::string module name
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    std::string GetModuleName() const noexcept
    {
        return m_sModuleName;
    }

    uint8_t GetLinkIndex() const noexcept
    {
        return m_linkIndex;
    }

    /** @brief returns Up Sensor pointer
     * @return CNvMSensor *
     */
     CNvMSensor *GetUpSensor() const noexcept
    {
        return m_upSensor.get();
    }

    /** @brief Sets Up Sensor pointer
     * @param[in] sensor CNvMSensor *
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    void SetUpSensor(CNvMSensor *const sensor) noexcept
    {
        m_upSensor.reset(sensor);
    }

    /** @brief returns Up Serializer pointer
     * @return CNvMSerializer *
     */
    CNvMSerializer *GetUpSerializer() const noexcept
    {
        return m_upSerializer.get();
    }

    /** @brief returns Up EEPROM pointer
     * @return CNvMEEPROM *
     */
    CNvMEEPROM *GetUpEeprom() const noexcept
    {
        return m_upEeprom.get();
    }

    /** @brief returns Up Camera Power pointer
     * @return CNvMCampwr *
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    CNvMCampwr *GetUpCampwr() const noexcept
    {
        return m_upCampwr.get();
    }

    /** @brief returns Up PMIC pointer
     * @return CNvMPMIC *
     */
    CNvMPMIC *GetUpPmic() const noexcept
    {
        return m_upPmic.get();
    }

    /** @brief returns Up Vcsel pointer
     * @return CNvMVCSEL *
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    CNvMVCSEL *GetUpVcsel() const noexcept
    {
        return m_upVcsel.get();
    }

    /** @brief returns Up Transport pointer
     * @return CNvMTransportLink *
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    CNvMTransportLink *GetUpTransport() const noexcept
    {
        return m_upTransport.get();
    }

    /** @brief returns Up Broadcast Sensor pointer
     * @return CNvMSensor *
     */
    CNvMSensor *GetBroadcastSensor() const noexcept
    {
        return m_broadcastSensor.get();
    }

    /** @brief returns Device Parameter reference
     * @return CNvMDevice::DeviceParams &
     */
    /* coverity[autosar_cpp14_m0_1_10_violation] : intentional TID-2013 */
    CNvMDevice::DeviceParams const &GetDeviceParams() const noexcept
    {
        return m_oDeviceParams;
    }

    /** @brief Check whether Broadcast sensor is active on that module
     * @return bool
     */
    bool GetGroupInitProg() const noexcept
    {
        return GetBroadcastSensor() != nullptr;
    }

    /** @brief Checks whether the module is part of broadcast initialization
     * group (may or may not have BroadcastSensor created on that module)
     * @return bool
     */
    bool IsBroadcastInitUsed() const noexcept
    {
        return m_groupInitProg;
    }

    /** @brief sets Group Init prog boolean
     * @@param[in] value boolean true/false
     */
    void SetGroupInitProg(bool const value) noexcept
    {
        m_groupInitProg = value;
    }

/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if !(NV_IS_SAFETY)
    DevBlkCDIRootDevice* m_pCDIRoot; /*!< Pointer to CDI Root device */
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#endif // !(NV_IS_SAFETY)

private:
    /** stores Up Sensor meta data */
    std::unique_ptr<CNvMSensor> m_upSensor;

    /** stores Up Eeprom meta data */
    std::unique_ptr<CNvMEEPROM> m_upEeprom;

    /** stores Up Camera power meta data */
    std::unique_ptr<CNvMCampwr> m_upCampwr;

    /** stores Up PMIC meta data */
    std::unique_ptr<CNvMPMIC> m_upPmic;

    /** stores Up Vcsel meta data */
    std::unique_ptr<CNvMVCSEL> m_upVcsel;

    /** stores Up Serializer meta data */
    std::unique_ptr<CNvMSerializer> m_upSerializer;

    /** stores Up Transport meta data */
    std::unique_ptr<CNvMTransportLink> m_upTransport;

    /** stores Up Broadcast Sensor meta data */
    std::unique_ptr<CNvMSensor> m_broadcastSensor;

    /** Device params needs to be cached to support reinitialization */
    CNvMDevice::DeviceParams m_oDeviceParams;

    /** Indicate if the homogeneous camera support enabled or not */
    bool m_groupInitProg;

    /** stores Up Camera Control Information meta data */
    DevBlkCDIPowerControlInfo m_camPwrControlInfo = {0xFFU, 0U};

    uint8_t m_linkIndex; /*!< Link Index */

    uint8_t m_initLinkMask; /*!< initialize Link Mask */

/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#if (NV_IS_SAFETY)
    DevBlkCDIRootDevice* m_pCDIRoot; /*!< Pointer to CDI Root device */
/* coverity[autosar_cpp14_a16_0_1_violation] : intentional TID-2039 */
#endif // !(NV_IS_SAFETY)

    SerInfo m_oSerInfo; /*!< Serializer Communication information */

    SensorInfo m_oSensorInfo; /*!< Sensor Communication information */

    CNvMDeserializer *m_pDeserializer; /*!< Pointer to De-serializer device */

    std::string m_sModuleName; /*!< Description of camera module */

    std::unique_ptr<Property> m_upCameraModuleProperty; /*!< Camera module Property */

    /** Camera module connection property */
    std::unique_ptr<CNvMCameraModuleCommon::ConnectionProperty> m_upCameraModuleConnectionProperty;

    /** stores interfaceType for CSI Port Information */
    NvSiplCapInterfaceType m_interfaceType;

    /** @brief Create CDI instance
     * - status = CreateSerializer(linkIndex)
     * - status = CreateSensor( status, linkIndex )
     * - status = CreateEEPROM( status, linkIndex )
     * - status = CreatePMIC( status, linkIndex)
     * - status = CreateVCSEL( status, linkIndex)
     * - status = CreateCameraPowerLoadSwitch( status, linkIndex)
     * - status = CreateTransport( status )
     * .
     *
     * @param[in] cdiRoot : Root device Handle
     * @param[in] linkIndex : link index
     * @return NVSIPL_STATUS_BAD_ARGUMENT:Input argument is invalid.
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus DoCreateCDIDevice(DevBlkCDIRootDevice *const  cdiRoot,
                                 uint8_t const linkIndex) final;

    /** @brief Specify camera module connection property
     * - Create camera module property and connection property
     *  - CameraModuleInfo const* const moduleInfo = cameraModuleConfig->cameraModuleInfo
     *  - m_upCameraModuleProperty = std::move(std::unique_ptr<Property>(new Property))
     *  - m_upCameraModuleConnectionProperty =  std::move(
     *   - std::unique_ptr<CNvMCameraModuleCommon::ConnectionProperty>(
     *    - new CNvMCameraModuleCommon::ConnectionProperty))
     *    .
     *   .
     *  .
     * - acquire CNvMSensor *const sensor = m_upSensor.get()
     * - collect and setup sensor properties
     *  - CNvMCameraModule::Property::SensorProperty const oSensorProperty = {
     *   - .id = moduleInfo->sensorInfo.id,
     *   - .virtualChannelID = linkIndex,
     *   - .inputFormat = sensor->GetInputFormat(),
     *   - .pixelOrder = sensor->GetPixelOrder(),
     *   - .width = sensor->GetWidth(),
     *   - .height = sensor->GetHeight(),
     *   - .startX = 0U,
     *   - .startY = 0U,
     *   - .embeddedTop = sensor->GetEmbLinesTop(),
     *   - .embeddedBot = sensor->GetEmbLinesBot(),
     *   - .frameRate = sensor->GetFrameRate(),
     *   - .embeddedDataType = sensor->GetEmbDataType(),
     *   - .pSensorControlHandle = sensor,
     *   - .pSensorInterfaceProvider = NULL,
     *   - }
     *   .
     *  .
     * - save sensor properties
     *  - m_upCameraModuleProperty->sensorProperty = oSensorProperty
     *  .
     * - collect and setup sensor connections properties
     *  - CNvMCameraModuleCommon::ConnectionProperty::SensorConnectionProperty
     *    sensorConnectionProperty
     *   - SetSensorConnectionProperty(&sensorConnectionProperty);
     *   - sensorConnectionProperty.uBrdcstSensorAddrs  = sensor->GetNativeI2CAddr()
     *   - sensorConnectionProperty.uVCID = static_cast<uint8_t>(oSensorProperty.virtualChannelID)
     *   - sensorConnectionProperty.inputFormat =  sensor->GetInputFormat()
     *   - sensorConnectionProperty.bEmbeddedDataType =  sensor->GetEmbDataType()
     *   - sensorConnectionProperty.bEnableTriggerModeSync =  sensor->GetEnableExtSync()
     *   - sensorConnectionProperty.fFrameRate =  sensor->GetFrameRate()
     *   - sensorConnectionProperty.height = sensor->GetHeight()
     *   - sensorConnectionProperty.width = sensor->GetWidth()
     *   - sensorConnectionProperty.embeddedTop = sensor->GetEmbLinesTop()
     *   - sensorConnectionProperty.embeddedBot = sensor->GetEmbLinesBot()
     *   - sensorConnectionProperty.sensorDescription = sensor->GetSensorDescription()
     *   - sensorConnectionProperty.vGpioMap.resize(moduleInfo->
     *     serInfo.serdesGPIOPinMappings.size())
     *   - for (uint8_t i = 0U; i < moduleInfo->serInfo.serdesGPIOPinMappings.size(); i++)
     *    - sensorConnectionProperty.vGpioMap[i].sourceGpio =
     *     - moduleInfo->serInfo.serdesGPIOPinMappings[i].sourceGpio
     *     .
     *    - sensorConnectionProperty.vGpioMap[i].destGpio =
     *     - moduleInfo->serInfo.serdesGPIOPinMappings[i].destGpio
     *     .
     *    .
     *   .
     * - save sensor connection properties
     *  - m_upCameraModuleConnectionProperty->sensorConnectionProperty = sensorConnectionProperty
     * - set and save eeprom properties
     *  - CNvMCameraModule::Property::EEPROMProperty const oEepromProperty = {
     *   - .isEEPROMSupported = moduleInfo->isEEPROMSupported,
     *   .
     *  - }
     *  - m_upCameraModuleProperty->eepromProperty = oEepromProperty
     *  .
     * .
     *
     * @param[in] cameraModuleConfig
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus SetupConnectionProperty(CameraModuleConfig const* const cameraModuleConfig,
                                       uint8_t const linkIndex);

    /** @brief Create serializer
     * - m_upSerializer->CreateCDIDevice(m_pCDIRoot, linkIndex)
     * .
     * .
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreateSerializer(uint8_t const linkIndex);

    /** @brief Create transport
     * - if no previous error
     *  - set m_upTransport = std::move(CreateNewTransportLink())
     *  - set link params structure
     *   - CNvMTransportLink::LinkParams params;
     *   - params.pSerCDIDevice = m_upSerializer->GetCDIDeviceHandle()
     *   - params.pDeserCDIDevice = m_pDeserializer->GetCDIDeviceHandle()
     *   - params.ulinkIndex = m_linkIndex
     *   - params.uBrdcstSerAddr = m_upSerializer->GetNativeI2CAddr()
     *   - params.uSerAddr = m_upSerializer->GetI2CAddr()
     *   - params.moduleConnectionProperty = *m_upCameraModuleConnectionProperty
     *   - params.bEnableSimulator = m_oDeviceParams.bEnableSimulator
     *   - params.bPassive = m_oDeviceParams.bPassive
     *   - params.m_groupInitProg = m_groupInitProg
     *   .
     *  - m_upTransport->SetConfig(params)
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreateTransport(SIPLStatus const instatus);

    /** @brief Create pmic
     * - if no previous error
     *  - if m_upCameraModuleConnectionProperty->sensorConnectionProperty.pmicProperty.isSupported)
     *   - if (m_upCameraModuleConnectionProperty->sensorConnectionProperty.pmicProperty.i2cAddress
     *         != 0U)
     *    - m_upPmic.reset(new CNvMPMIC())
     *    - m_upPmic->SetConfig(
     *     - m_upCameraModuleConnectionProperty->sensorConnectionProperty.pmicProperty.i2cAddress,
     *     - &m_oDeviceParams)
     *     .
     *    - m_upPmic->SetDriverHandle(GetCDIDeviceDriver(MODULE_COMPONENT_PMIC))
     *    - m_upPmic->CreateCDIDevice(m_pCDIRoot, linkIndex)
     *    .
     *   - else
     *    - error exit NVSIPL_STATUS_BAD_ARGUMENT
     *    .
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreatePMIC(SIPLStatus const instatus, uint8_t const linkIndex);

    /** @brief Create VCSEL
     * - if no previous error
     *  - if m_upCameraModuleConnectionProperty->sensorConnectionProperty.vcselProperty.isSupported)
     *   - if (m_upCameraModuleConnectionProperty->sensorConnectionProperty.vcselProperty.i2cAddress
     *         != 0U)
     *    - m_upVcsel.reset(new CNvMVCSEL())
     *    - m_upVcsel->SetConfig(
     *     - m_upCameraModuleConnectionProperty->sensorConnectionProperty.vcselProperty.i2cAddress,
     *     - &m_oDeviceParams)
     *     .
     *    - m_upVcsel->SetDriverHandle(GetCDIDeviceDriver(MODULE_COMPONENT_VCSEL))
     *    - m_upVcsel->CreateCDIDevice(m_pCDIRoot, linkIndex)
     *    .
     *   - else
     *    - error exit NVSIPL_STATUS_BAD_ARGUMENT
     *    .
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    SIPLStatus CreateVCSEL(SIPLStatus const instatus, uint8_t const linkIndex);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif /* !NV_IS_SAFETY */

    /** @brief Create eeprom
     * - if no previous error
     *  - set m_upCameraModuleConnectionProperty->sensorConnectionProperty.uSensorAddrs =
     *   - m_upSensor->GetI2CAddr()
     *   .
     *  - if (m_upCameraModuleProperty->eepromProperty.isEEPROMSupported)
     *   - m_upEeprom->CreateCDIDevice(m_pCDIRoot, linkIndex)
     *   - set m_upCameraModuleConnectionProperty->eepromAddr =
     *    - m_upEeprom->GetI2CAddr()
     *    .
     *   - set m_upCameraModuleConnectionProperty->brdcstEepromAddr =
     *    - m_upEeprom->GetNativeI2CAddr()
     *    .
     *   .
     *  - else
     *   - set m_upCameraModuleConnectionProperty->eepromAddr = UINT8_MAX
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreateEEPROM(SIPLStatus const instatus, uint8_t const linkIndex);

    /** @brief Create sensor
     * - if no previous error
     *  - m_upSensor->CreateCDIDevice(m_pCDIRoot, linkIndex)
     *  .
     *
     * @param[in] instatus  previous error code
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreateSensor(SIPLStatus const instatus, uint8_t const linkIndex);

    /** @brief Create camera power load swith
     * - if no previous error
     *  - if ((!m_oDeviceParams.bPassive) && (!m_oDeviceParams.bEnableSimulator))
     *   - Get power control information
     *    - NvMediaStatus nvmStatus = DevBlkCDIGetCamPowerControlInfo(
     *     - m_pDeserializer->GetCDIDeviceHandle(),
     *     - &m_camPwrControlInfo)
     *     .
     *    .
     *   - if error in nvmStatus
     *    - return ConvertNvMediaStatus(nvmStatus)
     *    .
     *   - if ((m_camPwrControlInfo.method > 0) && (m_camPwrControlInfo.method != UINT8_MAX))
     *    - set CNvMDevice::DeviceParams params = m_oDeviceParams
     *    - set params.bUseNativeI2C = true
     *    - acquire up camera power
     *     - m_upCampwr =
     *      - CNvMMax20087DriverFactory::RequestPowerDriver(
     *       - m_pCDIRoot, linkIndex)
     *       .
     *      .
     *     .
     *    - verify m_upCampwr is NOT nullptr
     *    - m_upCampwr->SetConfig(m_camPwrControlInfo.i2cAddr, &params)
     *    - m_upCampwr->CreatePowerDevice(m_pCDIRoot, linkIndex)
     *    - m_upCampwr->InitPowerDevice(
     *     - m_upCampwr->GetDeviceHandle(), linkIndex)
     *     .
     *    .
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] linkIndex: link index
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus CreateCameraPowerLoadSwitch(SIPLStatus const instatus, uint8_t const linkIndex);

    /** @brief reset all components
     * - m_upSensor->Reset()
     * - m_upSerializer->Reset()
     * - m_upTransport->Reset()
     *
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus Reset(SIPLStatus const instatus);

    /** @brief Detect the sensor
     * - if no previous error
     *  - DetectModule()
     *  .
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitDetectModule(SIPLStatus const instatus);

    /** @brief Use the broadcast serializer to initialize all transport links
     * - if no previous error
     *  - m_upTransport->Init(
     *   - broadcastSerializer->GetCDIDeviceHandle(),
     *   - m_initLinkMask,
     *   - m_groupInitProg)
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] broadcastSerializer  handle
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitTransportLinks(SIPLStatus const instatus,
                                  CNvMSerializer const &broadcastSerializer);

    /** @brief create broadcast serializer
     * - if no previous error
     *  - CreateBroadcastSerializer(
     *   - m_linkIndex, broadcastSerializer)
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[out] broadcastSerializer
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitCreateBroadcastSerializer(
        SIPLStatus const instatus,
        std::unique_ptr<CNvMSerializer>& broadcastSerializer );

    /** @brief initialize transport links
     * - if no previous error
     *  - Add additional delays to get module stable as link rebuild can hit over 100ms
     *   - std::this_thread::sleep_for<>(std::chrono::milliseconds(100))
     *   .
     *  - set uint32_t const tmpLinkIndex = leftBitsShift(
     *   - 1U, m_linkIndex).
     *   .
     *  - verify index is within range
     *   - (UINT8_MAX > tmpLinkIndex)
     *   .
     *  - status = m_upTransport->Init(
     *   - broadcastSerializer->GetCDIDeviceHandle(),
     *   - tmpLinkIndex,
     *   - m_groupInitProg)
     *   .
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] broadcastSerializer
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitializeTransportLinks(
        SIPLStatus const instatus,
        CNvMSerializer const &broadcastSerializer);

    /** @brief detect sensor
     * - if no previous error
     *  - status = DetectModule()
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus DetectSensor(SIPLStatus const instatus);

    /** @brief authenticate sensor
     * - if no previous error
     *  - status = Authenticate()
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus AuthenticateSensor(SIPLStatus const instatus);

    /** @brief initialize sensor
     * - if no previous error
     *  - status = InitModule()
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @param[in] item link action
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitSensors(SIPLStatus const instatus);

    /** @brief post initialize transport link
     * - if no previous error
     *  - verify m_linkIndex <= 7
     *  - set uint32_t const tmpLinkIndex = leftBitsShift(
     *   - 1U,
     *   - m_linkIndex)
     *   .
     *  - verify index is within range
     *   - (UINT8_MAX > tmpLinkIndex)
     *   .
     *  - status = m_upTransport->PostSensorInit(
     *   - static_cast<uint8_t>(tmpLinkIndex),
     *   - m_groupInitProg)
     *   .
     *  .
     * .
     *
     * @param[in] instatus prev ious error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
     SIPLStatus TransportLinkPostInit(
        SIPLStatus const instatus);

    /** @brief misc. initialize transport link
     * - if no previous error
     *  - status = m_upTransport->MiscInit()
     *  .
     * .
     *
     * @param[in] instatus  previous error code
     * @return NVSIPL_STATUS_OK:on completion
     * @return (SIPLStatus) : other implementation-determined or propagated error
     */
    SIPLStatus InitTransportLinkMisc(
        SIPLStatus const instatus);

    /** @brief disable transport link
     * - create std::vector<CNvMDeserializer::LinkAction> linkAction
     * - set item.linkIdx = m_linkIndex
     * - set item.eAction = CNvMDeserializer::LinkAction::LINK_DISABLE
     * - put item into vector
     *  - linkAction.push_back(item)
     *  .
     * - m_pDeserializer->ControlLinks(linkAction)
     *
     * @param[inout] item link action
     */
    void LinkDisable(CNvMDeserializer::LinkAction& item);

    /** @brief Create a New Sensor object
     * @param[in] linkIndex : Link index [0, 3]
     * @param[out] sensor : Created broadcast sensor handle
     * @return NVSIPL_STATUS_OK: on completion
     */
    SIPLStatus DoCreateBroadcastSensor(uint8_t const linkIndex,
                                       std::unique_ptr<CNvMSensor>& sensor);

    /** Mask or restore mask of power switch interrupt */
    SIPLStatus MaskRestorePowerSwitchInterrupt(const bool enable);

#if !NV_IS_SAFETY
    /**
     * @brief Function to map SIPL log verbosity level to
     * C log verbosity level.
     *
     * @param[in] level SIPL Log trace level.
     * @return C log level.
     */
    C_LogLevel ConvertLogLevel(INvSIPLDeviceBlockTrace::TraceLevel const level);
#endif
};


} // end of namespace

#endif /* CNVMMAX96712CAMERAMODULE_HPP */
