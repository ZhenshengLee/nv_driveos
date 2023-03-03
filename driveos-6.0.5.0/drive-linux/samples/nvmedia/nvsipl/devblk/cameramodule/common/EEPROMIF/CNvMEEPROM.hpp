/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMEEPROM_HPP
#define CNVMEEPROM_HPP

#include "NvSIPLPlatformCfg.hpp"
#include "devblk_cdi.h"

#include "CNvMDevice.hpp"
#include "utils/utils.hpp"

namespace nvsipl
{

/**
 * The CNvMEEPROM class encapsulates EEPROM control and configuration information.
 */
class CNvMEEPROM : public CNvMDevice
{
public:
    /**
    * Default constructor
    */
    CNvMEEPROM() = default;

    /**
    * Default Destructor
    */
    virtual ~CNvMEEPROM() = default;

    /**
     * Sets the configuration for the EEPROM object based on the params
     * Updates the object's state information to indicate that configuration has been set
     *
     * If simulator mode is not enabled and passive mode is not enabled,
     * this will save the EEPROM's I2C address and register it with the address manager.
     *
     * - verifiy eepromInfo and params are not nulptr
     * - if (NOT m_oDeviceParams.bEnableSimulator) and ( NOT m_oDeviceParams.bPassive)
     *  - set m_nativeI2CAddr =  eepromInfo->i2cAddress
     *  - register I2C address by
     *   - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr)
     *   .
     *  - Note: This is WAR to reseve the I2C address for 2nd page and the
     *   - identification page only for M24C04
     *   .
     *  - if (eepromInfo->name == "M24C04")
     *   - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 1U)
     *   - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 8U)
     *   - m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 9U)
     *   .
     *  - set m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE
     *  .
     *
     * @param[in] eepromInfo        EEPROM info struct containing I2C address
     * @param[in] params            Device information used to register I2C address
     * @retval                      NVSIPL_STATUS_OK on completion
     */
    virtual SIPLStatus SetConfig(const EEPROMInfo *const info, const DeviceParams *const params);

    /**
     * Sets the object's driver handle to the provided driver
     *
     * - verify driver is NOT nullptr
     * - set m_pCDIDriver = driver
     *
     * @param[in] driver            DevBlkCDIDeviceDriver ptr to store
     * @retval                      NVSIPL_STATUS_OK on successful completion
     * @retval                      NVSIPL_STATUS_BAD_ARGUMENT on error
     */
    SIPLStatus SetDriverHandle(DevBlkCDIDeviceDriver * const driver) {
        SIPLStatus status = NVSIPL_STATUS_BAD_ARGUMENT;
        if (driver != nullptr) {
            m_pCDIDriver = driver;
            status = NVSIPL_STATUS_OK;
        }

        return status;
    }

    /**
     * @brief Sets the camera module streaming state
     *
     * - set m_eState = streaming ? STARTED : STOPPED
     *
     * @param[in] streaming         Camera module STARTED (true) or STOPPED (false)
     * @retval                      None
     */
    void SetStreamingState(bool const streaming) {
        if (streaming == true) {
            m_eState = STARTED;
        } else {
            m_eState = STOPPED;
        }
    }

    /** @brief stub for unsupported function
     * @return NVSIPL_STATUS_OK */
    virtual SIPLStatus DoInit() override
    {
        return NVSIPL_STATUS_OK;
    };

    /** @brief stub for unsupported function
     * @return NVSIPL_STATUS_OK */
    virtual SIPLStatus DoStart() override
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop() override
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * @brief Reads data from this EEPROM device.
     *
     * The default implementation validates the arguments and then uses the
     * underlying CDI device driver to read back the specified address range
     * over I2C.
     *
     * Device drivers should override this to return
     * `NVSIPL_STATUS_NOT_SUPPORTED` when reading is not supported by the
     * device.
     *
     * - verify length > 0, buffer is NOT nullptr
     * - verify  m_pCDIDriver is NOT nullptr, m_eState == STARTED
     * - extract dev  = GetCDIDeviceHandle() , verify it is NOT nullptr
     * - read register data by:
     *  - NvMediaStatus const nvmerr = m_pCDIDriver->ReadRegister(
     *   - dev, 0U, static_cast<uint32_t>(address), length, buffer)
     *   .
     *  .
     *
     * @param[in]   address The start address to read from. Valid ranges depend
     *                      on the driver backing this device.
     * @param[in]   length  The length of data to read, in bytes.
     *                      May not be zero.
     * @param[out]  buffer  Output data buffer. Must be at least @a length
     *                      bytes long.
     *                      Must not be null.
     *
     * @retval NVSIPL_STATUS_OK             if the operation completes successfully.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT   on Invalid argument
     * @retval NVSIPL_STATUS_NOT_SUPPORTED  if data readback is not supported
     *                                      for this particular device.
     *                                      (Failure)
     * @retval (SIPLStatus)                 device implementations may return
     *                                      other error codes. (Failure)
     */
    virtual SIPLStatus ReadData(const std::uint16_t address,
                                const std::uint32_t length,
                                std::uint8_t * const buffer);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    virtual SIPLStatus WriteData(const std::uint16_t address,
                                 const std::uint32_t length,
                                 std::uint8_t * const buffer);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY
};

} // end of namespace nvsipl
#endif /* CNVMEEPROM_HPP */

