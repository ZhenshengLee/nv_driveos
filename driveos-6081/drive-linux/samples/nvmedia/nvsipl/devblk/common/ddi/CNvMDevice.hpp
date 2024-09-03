/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMDEVICE_HPP
#define CNVMDEVICE_HPP

#include <string>
#include <memory>

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"
/*VCAST_DONT_INSTRUMENT_START*/
#include "utils/utils.hpp"
#include "utils/CNvMI2CAddrMgr.hpp"
/*VCAST_DONT_INSTRUMENT_END*/

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Device </b>
 *
 */

namespace nvsipl
{
/**
 * @brief A function object which invokes DevBlkCDIRootDeviceDestroy on a given device
 */
struct CloseDevBlkCDIRoot {
    /**
     * Calls DevBlkCDIRootDeviceDestroy() on the given device.
     *
     * See DevBlkCDIRootDeviceDestroy() for details.
     *
     * @param [in] device The device to destroy.
     */
    void operator ()(DevBlkCDIRootDevice *const device) const {
        DevBlkCDIRootDeviceDestroy(device);
    }
};

/**
 * @brief A function object which invokes DevBlkCDIDeviceDestroy on a given device
 */
struct CloseDevBlkCDI {
    /**
     * Calls DevBlkCDIDeviceDestroy() on the given device.
     *
     * See DevBlkCDIDeviceDestroy() for details.
     *
     * @param [in] device The device to destroy.
     */
    void operator ()(DevBlkCDIDevice *const device) const {
        DevBlkCDIDeviceDestroy(device);
    }
};

/** @defgroup ddi_device Device APIs
 *
 * @brief Provides interfaces for the device.
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * @brief Base class representing a generic device.
 *
 * All specific device types (deserializers, serializers, cameras, etc.) should
 * inherit from this class.
 */
class CNvMDevice
{
public:
    /**
     * @brief Container for parameters common to all devices.
     *
     * This contains most of the information required to configure a
     * communication link with the device.
     */
    struct DeviceParams {
        /**
         * @brief True when Device Block is running in simulation mode, and no
         * communication with the device should be performed.
         */
        bool bEnableSimulator;
        /**
         * @brief True when no communication should be performed with any
         * device other than the deserializer.
         */
        bool bPassive;
        /**
         * @brief True when the Camera Boot Accelerator is enabled.
         */
        bool bCBAEnabled;
        /**
         * @brief True when the device needs to use its "native" I2C address.
         *
         * This should be set to true for devices on the local side of a serdes
         * link, where address cannot be remapped. Remote devices can set this
         * to false, when the transport link supports I2C address remapping.
         */
        bool bUseNativeI2C;
        /**
         * @brief True when the V2 CDI API should be used (CDAC).
         *
         * This is only applicable on QNX-like operating systems where CDAC is
         * supported.
         */
        bool bUseCDIv2API;
        /**
         * @brief True when the device is a deserializer.
         *
         * In some cases deserializer type devices need to be handled
         * differently. For example, upstream camera modules must be
         * disabled prior to disabling the deserializer itself.
         */
        bool bIsDeserializer;
        /**
         * @brief Pointer to the I2C address manager for the I2C bus this
         * device lives on.
         *
         * It cannot be null. The lifecycle of this pointer is managed by
         * DeviceBlock core, and thus it should not be deleted or overwritten
         * by DDI or any of the drivers it loads.
         */
        CNvMI2CAddrMgr *pI2CAddrMgr;
    };

    /**
     * @brief Type-erasing interface through which device-specific drivers can
     * hold additional data.
     *
     * The value returned by 'GetPtr' here will later be passed to the
     * 'DriverCreate' function exposed via CDI.
     */
    class DriverContext {
        public :

        /**
         * @brief Constructor.
         */
        DriverContext() = default;
        /**
         * @brief Prevent DriverContext from being copy constructed.
         */
        DriverContext(DriverContext const &) = delete;
        /**
         * @brief Prevent DriverContext from being move constructed.
         */
        DriverContext(DriverContext &&) = delete;
        /**
         * @brief Prevent DriverContext from being copy assigned.
         */
        DriverContext& operator=( DriverContext const &) & = delete;
        /**
         * @brief Prevent DriverContext from being move assigned.
         */
        DriverContext& operator=(DriverContext&&) & = delete;

        /**
         * @brief Get a pointer to the driver context.
         *
         * The pointer returned from this function should have a lifetime that
         * closely matches that of the DriverContext structure itself. That is,
         * it should probably be a pointer to a member of the implementation of
         * this interface.
         *
         * @pre A valid device context is created.
         *
         * @retval (void*)  A pointer to the driver context.
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
        virtual void *GetPtr() = 0;

        /**
         * @brief Default destructor of struct DriverContext.
         */
        virtual ~DriverContext() = default;
    };

    /**
     * @brief Implements the most common case for @ref DriverContext, where the
     * context structure stores only a single context type.
     *
     * This should be sufficient for the vast majority of use cases.
     */
    template <typename ContextType>
    class DriverContextImpl : public DriverContext {
        public :

        /**
         * @brief Constructor.
         */
        DriverContextImpl() = default;
        /**
         * @brief Prevent DriverContextImpl from being copy constructed.
         */
        DriverContextImpl(DriverContextImpl const &) = delete;
        /**
         * @brief Prevent DriverContextImpl from being move constructed.
         */
        DriverContextImpl(DriverContextImpl &&) = delete;
        /**
         * @brief Prevent DriverContextImpl from being copy assigned.
         */
        DriverContextImpl& operator=(DriverContextImpl const &) & = delete;
        /**
         * @brief Prevent DriverContextImpl from being move assigned.
         */
        DriverContextImpl& operator=(DriverContextImpl&&) & = delete;

        /**
         * @brief Storage for the value which will be returned by @ref GetPtr().
         */
        ContextType m_Context;

        /**
         * @brief Implementation of @ref DriverContext::GetPtr
         *
         * @pre A valid device context is created.
         *
         * @retval (void*)  A pointer to the driver context.
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
        void *GetPtr() override {return &m_Context;}

        /**
         * @brief Default destructor of struct DriverContextImpl.
         */
        ~DriverContextImpl() override = default;
    };

    /**
     * @brief Gets CDI handle.
     *
     * This API is used to retrieve CDI Device handle associated with this CNvMDevice
     * instance.
     *
     * @pre A valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *
     * @retval (DevBlkCDIDevice*)    A pointer to CDI handle.
     * @retval NULL                  If CNvMDevice instance is neither
     *                               in CREATED nor in CDI_DEVICE_CONFIG_SET
     *                               state.
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
     *   - De-Init: Yes
     */
    virtual DevBlkCDIDevice* GetCDIDeviceHandle() const;

    /**
     * @brief Gets I2C address.
     *
     * This API is used to retrieve I2C address of the device represented by this CNvMDevice
     * instance.
     *
     * @pre A valid device object created with CreateCDIDevice().
     *
     * @retval (uint8_t)  I2C address of the device represented by this CNvMDevice instance.
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
    virtual uint8_t GetI2CAddr() const;

    /**
     * @brief Gets Native I2C address.
     *
     * This API is used to retrieve native I2C address of the device represented by this
     * CNvMDevice instance.
     *
     * @pre A valid device object created with CreateCDIDevice().
     *
     * @retval (uint8_t)  Native I2C address of the device represented by CNvMDevice instance.
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
    virtual uint8_t GetNativeI2CAddr() const;

    /**
     * @brief Creates a CDI device based on previously set device configuration.
     *
     * The provided CDI root device and link index must be valid, and this
     * device must be in the @ref CDI_DEVICE_CONFIG_SET state. On successful
     * execution of this function, @ref m_upCDIDevice will be populated with a
     * pointer to a valid CDI device structure. This also updates @ref
     * m_i2CAddr to the actual address which should be used to communicate with
     * the device. In safety builds, this will always be the I2C address
     * derived by CDAC via CDI from @ref m_nativeI2CAddr.
     *
     * Device drivers which wish to override this behavior can do so by
     * overriding @ref DoCreateCDIDevice. However, any alternative
     * implementation must still set @ref m_upCDIDevice and @ref m_i2CAddr.
     *
     * @pre @a cdiRoot must be valid DevBlkCDIRootDevice handle created with DevBlkCDIRootDeviceCreate().
     *
     * @param[in] cdiRoot   The root device handle (of type \ref DevBlkCDIRootDevice)
     *                      from which a child device will be created. Must be non-null.
     * @param[in] linkIndex Link index off of the deserializer to which this
     *                      device is connected.
     *                      Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval NVSIPL_STATUS_OK             On successful execution.
     * @retval NVSIPL_STATUS_INVALID_STATE  If the device is not in
     *                                      CDI_DEVICE_CONFIG_SET. (Failure)
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   If argument validation fails.
     *                                      (Failure)
     * @retval NVSIPL_STATUS_ERROR          If CDI device creation fails for
     *                                      any reason. (Failure)
     * @retval (SIPLStatus)                 Subclasses can override
     *                                      @ref DoCreateCDIDevice to return
     *                                      other status values. (Failure)
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
    virtual SIPLStatus CreateCDIDevice(DevBlkCDIRootDevice* const cdiRoot, uint8_t const linkIndex);

    /**
     * @brief Performs device initialization work.
     *
     * This device should be in the @ref CDI_DEVICE_CREATED state prior to
     * invoking this function. This API, after successful state validation
     * calls @ref DoInit() method.
     * Driver subclasses can override the @ref DoInit() method to perform device
     * specific init sequence, such as completing any required configuration
     * of the device over I2C.
     * Finally it transitions this device to the @ref INIT_DONE state.
     *
     * @note It is caller's responsibility to ensure that the link is powered on,
     * before calling this API.
     *
     * @pre A valid device object created with CreateCDIDevice().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval NVSIPL_STATUS_INVALID_STATE  If the device is not in @ref
     *                                      CDI_DEVICE_CONFIG_SET. (Failure)
     * @retval (SIPLStatus)                 Subclasses can override @ref DoInit()
     *                                      to return other status values. (Failure)
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
    virtual SIPLStatus Init();

    /**
     * @brief Transitions the device from a quiescent state to actively
     * sending/receiving data.
     *
     * This device should be in the @ref INIT_DONE state prior to invoking
     * this function, unless this requirement is relaxed by the specific device
     * implementation.
     *
     * This API checks that the device is in @ref INIT_DONE and calls @ref DoStart()
     * method, finally transitions this device to the @ref STARTED state.
     * If device drivers wish they can override @ref DoStart() to perform device
     * specific start sequence.
     *
     * @pre
     *   - A valid device object created with CreateCDIDevice().
     *   - This function must be called only after @ref Init() and before @ref Deinit().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval NVSIPL_STATUS_INVALID_STATE  If the device is not in a state
     *                                      where it can be started. (Failure)
     * @retval (SIPLStatus)                 Subclasses can override @ref DoStart()
     *                                      to return other status values. (Failure)
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
    virtual SIPLStatus Start();

    /**
     * @brief Returns the device to a quiescent state from a running state.
     *
     * This device should be in the @ref STARTED state when this function is
     * invoked. This API, after successful inputs and state validation, calls
     * @ref DoStop().
     * The driver subclasses can override the @ref DoStop() to perform device
     * specific stop sequence, such as issueing whatever I2C commands are required
     * to disable active data streaming.
     *
     * Following successful execution of this function, the device should be in
     * the @ref STOPPED state, and running @ref Start() again should be safe.
     *
     * @pre
     *   - A valid device object created with CreateCDIDevice().
     *   - This function must be called only after @ref Start().
     *
     * @retval NVSIPL_STATUS_OK             On success.
     * @retval NVSIPL_STATUS_INVALID_STATE  If the device is not in a state
     *                                      where it can be started. (Failure)
     * @retval (SIPLStatus)                 Subclasses can override @ref DoStop()
     *                                      to return other status values. (Failure)
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
    virtual SIPLStatus Stop();

    /**
     * @brief Resets the device back to a state from which it can be initialized.
     *
     * The device can be in any of the following states:
     *  - @ref CDI_DEVICE_CREATED
     *  - @ref INIT_DONE
     *  - @ref STARTED
     *  - @ref STOPPED
     *
     * Following the successful execution of this function, this device will be
     * in the @ref CDI_DEVICE_CREATED state.
     *
     * @pre
     *   - A valid device object created with CreateCDIDevice().
     *   - This function must be called only after @ref Init() or @ref Start() and before @ref Deinit().
     *
     * @retval NVSIPL_STATUS_OK If the device has been successfully reset.
     * @retval (SIPLStatus)     Subclasses
     *                          other status values. (Failure)
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
    virtual SIPLStatus Reset();

    /**
     * @brief Performs deinitialization, reseting hardware state and tearing
     * down the CDI device.
     *
     * The device must be in one of the following states:
     *  - @ref INIT_DONE
     *  - @ref STARTED
     *  - @ref STOPPED
     *
     * Following the successful execution of this function, this device will be
     * in the @ref DEINITED state.
     *
     * @pre
     *   - A valid device object created with CreateCDIDevice().
     *   - This function must be called only after @ref Init().
     *
     * @retval NVSIPL_STATUS_OK If the device has been successfully deinitialized.
     * @retval (SIPLStatus)     Subclasses
     *                          other status values. (Failure)
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
     SIPLStatus Deinit();

    //! Destructor
    virtual ~CNvMDevice() {
        static_cast<void>(Deinit());
    }


    /**
     * @brief Prevent CNvMDevice class from being copy constructed.
     */
    CNvMDevice(const CNvMDevice &) = delete;

    /**
     * @brief Prevent CNvMDevice class from being copy assigned.
     */
    CNvMDevice& operator=(const CNvMDevice &) & = delete;

    /**
     * @brief Prevent CNvMDevice class from being move constructed.
     */
    CNvMDevice(CNvMDevice &&) = delete;

    /**
     * @brief Prevent CNvMDevice class from being move assigned.
     */
    CNvMDevice& operator=(CNvMDevice &&) & = delete;

protected:
    /**
     * @brief Initializes common device fields.
     *
     * This is protected because CNvMDevice is allowed to be
     * constructed only by its subclasses.
     */
    CNvMDevice();

    /**
     * @brief DDI Device state machine states
     *
     * This enumeration is used to track what state a device is in. See (@todo
     * STATE TRANSITION DIAGRAM) for how this maps to device states and
     * function calls.
     */
    enum class DeviceState {
        /**
         * @brief The device is freshly created.
         *
         * The initial state for all devices.
         */
        CREATED = 0,
        /**
         * @brief The device driver has been given configuration information,
         * and forwarded it to the CDI driver for this device.
         */
        CDI_DEVICE_CONFIG_SET,
        /**
         * @brief The device has initialized a CDI device.
         */
        CDI_DEVICE_CREATED,
        /**
         * @brief The device has completed initialization.
         */
        INIT_DONE,
        /**
         * @brief The device is currently actively facilitating data transfer.
         */
        STARTED,
        /**
         * @brief The device is initialized, was previously actively
         * transferring data, but is no longer doing so.
         */
        STOPPED,
        /**
         * @brief The device has been torn down, including releasing the
         * corresponding CDI device.
         */
        DEINITED
    };

    /**
     * \brief Pointer to the CDI device driver for this device.
     *
     * Subclasses should provide a way to initialize this variable, or set it
     * themselves. Must be valid at the time @ref CreateCDIDevice() is called.
     */
    DevBlkCDIDeviceDriver *m_pCDIDriver {};

    /**
     * \brief Pointer to advanced driver context for initialization.
     *
     * Subclasses should provide a way to initialize this variable, or set it
     * themselves. Must be valid at the time @ref CreateCDIDevice() is called.
     */
    std::unique_ptr<DriverContext> m_upDrvContext;

    /**
     * \brief Advanced configuration for CDI itself
     *
     * Subclasses should provide a way to initialize this variable, or set it
     * themselves if required. Must be valid at the time @ref CreateCDIDevice()
     * is called.
     */
    DevBlkCDIAdvancedConfig m_oAdvConfig{};

    /**
     * \brief Pointer to this device's corresponding CDI device.
     *
     * This is initialized by @ref CreateCDIDevice() and torn down by @ref
     * Deinit().
     */
    std::unique_ptr<DevBlkCDIDevice, CloseDevBlkCDI> m_upCDIDevice;

    /**
     * \brief Device communication parameters.
     *
     * This holds all information required to establish a communication link
     * with the device, with the exception of the I2C address (which is managed
     * by @ref m_i2CAddr).
     *
     * Subclasses should provide a way to initialize this variable, or set it
     * themselves if required. Must be valid at the time @ref CreateCDIDevice()
     * is called.
     */
    DeviceParams m_oDeviceParams {};

    /**
     * \brief Device I2C communication address.
     *
     * This address will be valid after @ref CreateCDIDevice() is invoked.
     */
    uint8_t m_i2CAddr {};

    /**
     * \brief Device native I2C address.
     *
     * Should hold the native I2C address for this device. Note that this should
     * not be assumed to be the same as the I2C address required for communication
     * with the device, as CDAC or the serdes can remap them, if required.
     *
     * Subclasses should provide a way to initialize this variable, or set it
     * themselves if required. Must be valid at the time @ref CreateCDIDevice()
     * is called.
     */
    uint8_t m_nativeI2CAddr {};

    /**
     * \brief Holds the internal state machine state.
     *
     * This should be kept up-to-date by subclasses by either calling the
     * appropriate base class functions in their overrides, or by manually
     * modifying this variable.
     *
     */
    DeviceState m_eState {};

private:
    /**
     * @brief Generates I2C Address of device.
     *
     * This API is used to generate I2C address of the device represented by this CNvMDevice
     * instance.
     *
     * @pre None.
     *
     * @param[out] addressList       A reference (of type \ref uint32_t) to
     *                               1-element list, containing an I2C address
     *                               of device. Must be valid device specifc
     *                               7-bit I2C address.
     *
     * @retval    NVSIPL_STATUS_OK     On successful completion.
     * @retval    NVSIPL_STATUS_ERROR  Failed to get I2C address from device driver.
     * @retval    (SIPLStatus)         Other propagated error codes.
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
    SIPLStatus GenerateI2CAddress(std::uint32_t (&addressList)[1]);

     /**
     * @brief Device specific method to create CDI device.
     *
     * The provided CDI root device and link index must be valid. On successful
     * execution of this function, @ref m_upCDIDevice will be populated with a
     * pointer to a valid CDI device structure. This also updates @ref
     * m_i2CAddr to the actual address which should be used to communicate with
     * the device. In safety builds, this will always be the I2C address
     * derived by CDAC via CDI from @ref m_nativeI2CAddr.
     *
     * Device drivers which wish to override this behavior can do so by
     * overriding this method \ref DoCreateCDIDevice. However, any alternative
     * implementation must still set @ref m_upCDIDevice and @ref m_i2CAddr.
     *
     * @pre @a cdiRoot must be valid DevBlkCDIRootDevice handle created with DevBlkCDIRootDeviceCreate().
     *
     * @param[in] cdiRoot   The root device handle (of type \ref DevBlkCDIRootDevice)
     *                      from which a child device will be created. Must be non-null.
     * @param[in] linkIndex Link index off of the deserializer to which this
     *                      device is connected.
     *                      Must be in [0, \ref MAX_CAMERAMODULES_PER_BLOCK).
     *
     * @retval NVSIPL_STATUS_OK             On successful execution.
     * @retval NVSIPL_STATUS_INVALID_STATE  If the device is not in
     *                                      CDI_DEVICE_CONFIG_SET. (Failure)
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   If argument validation fails.
     *                                      (Failure)
     * @retval NVSIPL_STATUS_ERROR          If CDI device creation fails for
     *                                      any reason. (Failure)
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
    virtual SIPLStatus DoCreateCDIDevice(DevBlkCDIRootDevice* const cdiRoot, uint8_t const linkIndex);

    /**
     * @brief Device specific implementation for device initialization.
     *
     * By default this method is no-op, if device drivers wish to override this
     * behavior can do so by overriding this method \ref DoInit().
     *
     * @pre A valid device object created with CreateCDIDevice().
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
    virtual SIPLStatus DoInit() = 0;

    /**
     * @brief Device specific implementation for device start operation.
     *
     * By default this method is no-op, if device drivers wish to override this
     * behavior can do so by overriding this method \ref DoStart().
     *
     * @pre A valid device object created with CreateCDIDevice().
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
    virtual SIPLStatus DoStart() = 0;


    /**
     * @brief Device specific implementation for device stop operation.
     *
     * By default this method is no-op, if Device drivers wish to override this
     * behavior can do so by overriding this method \ref DoStop().
     *
     * @pre A valid device object created with CreateCDIDevice().
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
    virtual SIPLStatus DoStop() = 0;

};

/** @} */

} // end of namespace nvsipl
#endif //CNVMDEVICE_HPP
