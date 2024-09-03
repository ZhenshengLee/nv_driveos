/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMDESERIALIZER_HPP
#define CNVMDESERIALIZER_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"

#include "CNvMDevice.hpp"
/*VCAST_DONT_INSTRUMENT_START*/
#include "utils/utils.hpp"
/*VCAST_DONT_INSTRUMENT_END*/

#include "INvSIPLDeviceInterfaceProvider.hpp"
#include "IInterruptStatus.hpp"

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Deserializer </b>
 *
 */

namespace nvsipl
{

/** @defgroup ddi_deserializer Deserializer API
 *
 * @brief Provides interfaces for the Deserializer.
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * The CNvMDeserializer class encapsulates deserializer control and configuration information.
 */
class CNvMDeserializer
    : public CNvMDevice
    , public IInterfaceProvider
    , public IInterruptStatus
{
public:
    /**
     * @brief destructor of class CNvMDeserializer.
     */
    ~CNvMDeserializer() override = default;

    /**
     * @brief Prevent CNvMDeserializer class from being copy constructed.
     */
    CNvMDeserializer(const CNvMDeserializer &) = delete;

    /**
     * @brief Prevent CNvMDeserializer class from being copy assigned.
     */
    CNvMDeserializer& operator=(const CNvMDeserializer &) & = delete;

    /**
     * @brief Prevent CNvMDeserializer class from being move constructed.
     */
    CNvMDeserializer(CNvMDeserializer &&) = delete;

    /**
     * @brief Prevent CNvMDeserializer class from being move assigned.
     */
    CNvMDeserializer& operator=(CNvMDeserializer &&) & = delete;

    /**
     * Defines Deserializer link modes
     */
    enum class LinkMode : std::uint32_t {
        /**
         * No link mode required, especially for the deserializer TPG mode
         */
        LINK_MODE_NONE,
        /**
         * GMSL (Gigabit Multimedia Serial Link) version 1
         */
        LINK_MODE_GMSL1,
        /**
         * GMSL (Gigabit Multimedia Serial Link) version 2, 6GBPS mode
         */
        LINK_MODE_GMSL2_6GBPS,
        /**
         * GMSL (Gigabit Multimedia Serial Link) version 2, 3GBPS mode
         */
        LINK_MODE_GMSL2_3GBPS,
#if !NV_IS_SAFETY
        /**
         * FPD3-Link, SYNC mode
         */
        LINK_MODE_FPDLINK3_SYNC,
        /**
         * FPD4-Link, SYNC mode
         */
        LINK_MODE_FPDLINK4_SYNC,
#endif
    };

    /**
     * Encapsulates deserializer link information - index and mode
     */
    struct DeserLinkModes{
        std::uint8_t linkIndex;
        LinkMode elinkMode;
    };

    /**
     * Defines the Deserializer Parameter structure
     * This includes all parameters needed to configure the Deserializer
     */
    struct DeserializerParams {
        /**
         * Generic device parameters
         */
        CNvMDevice::DeviceParams oDeviceParams;
        /**
         * InterfaceType that specifies the CSI port of the SoC to which the deserializer is connected
         */
        NvSiplCapInterfaceType eInterface;

        /**
         * The Power port of the deserializer
         */
        uint32_t pwrPort;

        /**
         * PHY mode for CSI, either DPHY or CPHY
         */
        NvSiplCapCsiPhyMode ePhyMode;
        /**
         * Link modes for each deserializer link
         */
        std::vector<DeserLinkModes> ovLinkModes;
        /**
         * @brief Bit masks for camera links.
         *
         * Must be constructed from members of @ref LinkMask.
         */
        uint8_t linkMask;
        /**
         * I2C port number
         */
        uint32_t I2CPortNum;
        /**
         * @brief Tx port number.
         *
         * Valid values for this are deserializer specific.
         */
        uint32_t TxPortNum;
        /**
         * @brief DPHY data rate.
         *
         * Valid values for this are deserializer specific.
         */
        uint32_t dphyRate[MAX_CSI_LANE_CONFIGURATION];
        /**
         * @brief CPHY data rate.
         *
         * Valid values for this are deserializer specific.
         */
        uint32_t cphyRate[MAX_CSI_LANE_CONFIGURATION];
        /**
         * do resetall by default during deser init sequence
         */
        bool defaultResetAll;
        /**
         * @brief long cables values for all camera modules (each is either true or false).
         *
         * A value of true indicates that high-gain mode should be enabled for
         * the corresponding link index.
         */
        bool longCables[MAX_CAMERAMODULES_PER_BLOCK] = {false, false, false, false};
    };

    /**
     * LinkAction for a particular link (given by linkIdx)
     * Action default value is LINK_NO_ACTION
     */
    struct LinkAction {
        /** link action code */
        enum class Action : std::uint32_t {
            LINK_NO_ACTION = 0, /*!< noop */
            LINK_ENABLE,        /*!< Enable link */
            LINK_DISABLE,       /*!< Disable link */
        };

        /**
         * @brief Index to operate on.
         *
         * Should be in the range 0 <= linkIdx <= MAX_CAMERAMODULES_PER_BLOCK.
         */
        uint8_t linkIdx;
        /**
         * @brief The action to perform on the given link.
         */
        Action eAction = Action::LINK_NO_ACTION;
    };

    /**
     * @brief Sets the configuration for the Deserializer instance.
     *
     * This performs several operations:
     *  - Set common information from the provided information structure:
     *      - Interface ID (m_eInterface)
     *      - Power port (m_pwrPort)
     *      - PHY mode (m_ePhyMode)
     *      - Link modes (m_ovLinkModes)
     *      - MIPI speed (m_uMipiSpeed)
     *      - Device-specific parameters (m_oDeviceParams)
     *      - Link mask (m_linkMask)
     *      - I2C Port (m_I2CPort)
     *      - TX port number (m_TxPort)
     *      - CPHY and DPHY data rates (m_cphyRate/m_dphyRate), for all camera modules.
     *      - Long Cable state (m_longCables), for all camera modules.
     *      - Native I2C address (m_nativeI2CAddr)
     *  - Default some common components of m_oDeviceParams
     *      - Whether or not to use CDAC (m_oDeviceParams.useCDIv2API)
     *      - Indicates that the device is a deserializer
     *        (m_oDeviceParams.bIsDeserializer)
     *  - Delegates to DoSetConfig() for device-specific configuration.
     *  - Updates Deserializer's state to CDI_DEVICE_CONFIG_SET.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[in] siplDeserInfo     A pointer of type \ref DeserInfo which holds Deserializer
     *                              device information, needed to configure Deserializer instance.
     *                              It cannot be NULL.
     * @param[in] params            A pointer of type \ref DeserializerParams which holds
     *                              Deserializer configuration parameters. It cannot be NULL.
     *
     * @retval    NVSIPL_STATUS_OK              Success.
     * @retval    NVSIPL_STATUS_INVALID_STATE   Deserializer is not in CREATED state.
     * @retval    NVSIPL_STATUS_BAD_ARGUMENT    When @a siplDeserInfo or @a params is NULL.
     * @retval    (SIPLStatus)                  Error propagated (Failure) from sub-routine calls.
     *
     * @note This method does not validate internal fields of @a siplDeserInfo
     * and @a params structures, it is caller's responsibility to provide valid
     * values for them, as specified by @ref DeserInfo and @ref DeserializerParams
     * respectively.
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
    virtual SIPLStatus SetConfig(const DeserInfo *const siplDeserInfo, DeserializerParams *const params);

    /**
     * @brief Enables specified Deserializer links.
     *
     * Device Block calls this API to enable Deserializer links specified
     * by @a linkMask, in order to enable camera modules connected on the
     * given links of this Deserializer.
     *
     * It must be implemented by device specific Deserializer driver.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[in] linkMask         Mask of type \ref uint8_t defining which
     *                             links to be enabled. Must be in [0x0U, 0xFU].
     *
     * @retval    NVSIPL_STATUS_OK  On successful completion.
     * @retval    (SIPLStatus)      Error propagated (Failure).
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
    virtual SIPLStatus EnableLinks(uint8_t const linkMask) = 0;

    /**
     * @brief Performs specified actions on given Deserializer links.
     *
     * Supported actions which can be performed on any Deserializer link
     * are defined by @ref LinkAction.
     *
     * It must be implemented by device specific Deserializer driver.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[in] linkActions       Vector of type \ref LinkAction, which
     *                              specifies the action to be performed for
     *                              each deserializer link. If zero size vector
     *                              is supplied, it acts as no-op.
     *
     * @retval    NVSIPL_STATUS_OK  On successful completion.
     * @retval    (SIPLStatus)      Error propagated (Failure).
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
    virtual SIPLStatus ControlLinks(const std::vector<LinkAction>& linkActions) = 0;


    /**
     * @brief Check link lock for specified Deserializer links.
     *
     * It must be implemented by device specific Deserializer driver.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[in] linkMask         Mask of type \ref uint8_t defining which
     *                              links to be checked for lock.
     *                              Must be in [0x0U, 0xFU].
     *
     * @retval    NVSIPL_STATUS_OK  On successful completion.
     * @retval    (SIPLStatus)      Error propagated (Failure).
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
    virtual SIPLStatus CheckLinkLock(uint8_t const linkMask) = 0;

    /**
     * @brief Gets size of deserializer error information.
     *
     * Used to allocate a buffer for requesting detailed errors.
     * By default this feature is not supported. If needed, the device specific
     * Deserializer driver can override/implement this method (together with
     * GetErrorInfo method) to publish the buffer size required for holding the
     * detailed deserializer error information.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[out] errorSize    Size (of type \ref size_t) of deserializer error
     *                          information (0 if no valid size found).
     *
     * @retval     NVSIPL_STATUS_OK             On successful completion.
     * @retval     NVSIPL_STATUS_NOT_SUPPORTED  Not supported/implemented by this
     *                                          Deserializer instance.
     * @retval      (SIPLStatus)                Error status propagated (Failure).
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
    virtual SIPLStatus GetErrorSize(size_t& errorSize);

     /**
     * @brief Gets detailed error information and populates a provided buffer.
     *
     * This is expected to be called after the client is notified of errors.
     * By default this feature is not supported. If needed, the device specific
     * Deserializer driver can override/implement this method (together with
     * GetErrorSize method) to export detailed deserializer error information.
     *
     * If no error info is expected (max error size is 0), this can be called with
     * null buffer to retrieve only the remote and link error information.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[out] buffer           A byte pointer Buffer (of type \ref uint8_t) to populate
     *                              with error information. It cannot be NULL.
     * @param[in]  bufferSize       Size (of type \ref size_t) of buffer to read to. Should be in
     *                              range of any value from 0 to the maximum size of an
     *                              allocation (0 if no valid size found). Error buffer size is
     *                              device driver implementation specific.
     * @param[out] size             Size (of type \ref size_t) of data read to the buffer
     *                              (0 if no valid size found).
     * @param[out] isRemoteError    A flag (of type \ref bool) set to true if remote serializer
     *                              error detected.
     * @param[out] linkErrorMask    Mask (of type \ref uint8_t) for link error state
     *                              (1 in index position indicates error).
     *                              Expected range is [0x0U, 0xFU].
     *
     * @retval      NVSIPL_STATUS_OK            On successful completion.
     * @retval      NVSIPL_STATUS_NOT_SUPPORTED Not supported/implemented by this
     *                                          Deserializer instance.
     * @retval      (SIPLStatus)                Error status propagated (Failure).
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
    virtual SIPLStatus GetErrorInfo(std::uint8_t * const buffer, std::size_t const bufferSize,
                                    std::size_t &size, bool & isRemoteError,
                                    std::uint8_t& linkErrorMask);

    /**
     * @brief Power on or off the Deserializer device.
     *
     * This operation short-circuits if the device is already in the requested
     * power state (as determined by m_isPoweredOn). If there is a mismatch,
     * DoSetPower() is invoked with the requested power state.
     *
     * Device drivers must implement DoSetPower to provide power control.  Note
     * that this function does not fully initialize or deinitialize the
     * deserializer, that must be done via Init() and Deinit(). Power control
     * most only occur when the device is deinitialized - either prior to
     * calling Init(), or after calling Deinit().
     *
     * @pre
     *   - A valid deserializer object created with CNvMDeserializer_Create().
     *   - A valid deserializer power control method is provided.
     *
     * @param[in] powerOn   A flag (of type \ref bool) denotes the power operation
     *                      to be performed. It can be either true or false.
     *                      When true, power on is requested.
     *                      When false, power off is requested.
     *
     * @retval    NVSIPL_STATUS_OK            On successful completion.
     * @retval    NVSIPL_STATUS_INVALID_STATE If Deserializer instance is not in
     *                                        desired CDI_DEVICE_CONFIG_SET state,
     *                                        when power on requested.
     * @retval    (SIPLStatus)                Error status propagated (Failure).
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
    virtual SIPLStatus SetPower(const bool powerOn);

    /**
     * @brief Get the MIPI speed, in Kbps (DPHY mode) or Ksps (CPHY mode).
     *
     * @pre
     *   - A valid deserializer object created with CNvMDeserializer_Create().
     *   - This function must be called only after @ref SetConfig().
     *
     * @retval (uint32_t)                  MIPI speed in Kbps (DPHY mode) or Ksps (CPHY mode).
     * @retval NVSIPL_STATUS_INVALID_STATE If Deserializer instance is not in CDI_DEVICE_CONFIG_SET
     *                                     state.
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
    std::uint32_t GetMipiSpeed();

protected:
    /**
     * Constructor for deserializer objects
     */
    CNvMDeserializer();

    /**
     * Holds ICP interface type corresponding to one of the CSI ports
     */
    NvSiplCapInterfaceType m_eInterface {};

    /**
     * Holds ICP CSI phy mode (either DPHY or CPHY)
     */
    NvSiplCapCsiPhyMode m_ePhyMode {};

    /**
     * Holds link modes for all links
     */
    std::vector<DeserLinkModes> m_ovLinkModes;

    /**
     * MIPI data rate. Units: Kbps in DPHY and Ksps in CPHY
     */
    std::uint32_t m_uMipiSpeed {};

    /**
     * Bit mask for camera links
     */
    std::uint8_t m_linkMask {};

    /**
     * Holds I2C port number
     */
    std::uint32_t m_I2CPort {};

    /**
     * Holds power port number
     */
    std::uint32_t m_pwrPort {};

    /**
     * Holds Tx port number
     */
    std::uint32_t m_TxPort {};

    /**
     * Holds DPHY data rate
     */
    std::uint32_t m_dphyRate[MAX_CSI_LANE_CONFIGURATION] {};

    /**
     * Holds CPHY data rate
     */
    std::uint32_t m_cphyRate[MAX_CSI_LANE_CONFIGURATION] {};

    /*
     * reset all sequence during deser init.
     */
    bool m_resetAll = false;

    /**
     * long cables values for all camera modules (each is either true or false)
     */
    bool m_longCables[MAX_CAMERAMODULES_PER_BLOCK] = {false, false, false, false};

private:
    /**
     * @brief Device specific implementation to power control Deserializer
     *
     * When @a powerOn is true, device-specific implementations use this to
     * perform any operations required to prepare the deserializer for I2C
     * communication (enable power gates, request ownership from a board
     * management controller, etc.). However, the deserializer will not be
     * initialized when this is invoked, so implementations should not assume
     * Init() has been performed.
     *
     * @pre
     *   - A valid deserializer object created with CNvMDeserializer_Create().
     *   - A valid deserializer power control method is provided.
     *
     * When @a powerOn is false, the inverse of the operations for powerup
     * should be performed. This will be called after Deinit(), and after
     * deinit and poweroff for all device connected to this deserializer.
     *
     * @param[in] powerOn   A flag, when true indicates the deserializer should
     *                      be powered on, and when false indicates that the
     *                      deserializer should be powered off.
     *
     * @retval NVSIPL_STATUS_OK When link power has been modified successfully
     * @retval (SIPLStatus)     Error propagated (Failure).
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
     * @brief Device specific implementation to set the configuration of Deserializer.
     *
     * It must be implemented by device specific Deserializer driver.
     *
     * @pre A valid deserializer object created with CNvMDeserializer_Create().
     *
     * @param[in] deserInfoObj      A pointer of type \ref DeserInfo which holds Deserializer
     *                              device information, needed to configure Deserializer instance.
     *                              It cannot be NULL.
     * @param[in] params            A pointer of type \ref DeserializerParams which holds
     *                              Deserializer configuration parameters. It cannot be NULL.
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
    virtual SIPLStatus DoSetConfig(DeserInfo const* const deserInfoObj,
                                   DeserializerParams *const params) = 0;

    SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };


    /**
     * Holds the power state of Deserializer device
     */
    bool m_isPoweredOn{false};
};

/** @} */

} // end of namespace nvsipl
#endif //CNVMDESERIALIZER_HPP
