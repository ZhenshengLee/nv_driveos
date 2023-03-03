/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMEEPROM.hpp"
#include "sipl_error.h"

namespace nvsipl
{

SIPLStatus CNvMEEPROM::SetConfig(const EEPROMInfo *const info, const DeviceParams *const params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (( params == nullptr ) || (info == nullptr)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (status == NVSIPL_STATUS_OK) {
        m_oDeviceParams = *params;

        if ((!m_oDeviceParams.bEnableSimulator) and (!m_oDeviceParams.bPassive)) {
            m_nativeI2CAddr =  info->i2cAddress;
            m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr);
            /* This is WAR to reseve the I2C address for 2nd page and the identification page
             * only for M24C04
             */
            if (info->name == "M24C04") {
                m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 1U);
                m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 8U);
                m_oDeviceParams.pI2CAddrMgr->RegisterNativeI2CAddr(m_nativeI2CAddr + 9U);
            }
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
#if (USE_CDAC == 0)
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_FALSE;
#else
        m_oDeviceParams.bUseCDIv2API = info->useCDIv2API;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#else
        m_oDeviceParams.bUseCDIv2API = NVMEDIA_TRUE;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        m_eState = CDI_DEVICE_CONFIG_SET;
    }

    return status;
}

SIPLStatus CNvMEEPROM::ReadData(const std::uint16_t address,
                                const std::uint32_t length,
                                std::uint8_t * const buffer)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (length == 0UL) {
        LOG_INFO("Invalid read length %u\n", length);
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else if (buffer == nullptr) {
        LOG_INFO("Invalid buffer\n");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("EEPROM device is not initialize");
        status = NVSIPL_STATUS_INVALID_STATE;
    } else if (m_eState == STARTED) {
        SIPL_LOG_ERR_STR("EEPROM device is not accessible during video streaming");
        status = NVSIPL_STATUS_INVALID_STATE;
    } else {
        DevBlkCDIDevice * const dev  = GetCDIDeviceHandle();

        if (dev == nullptr) {
            SIPL_LOG_ERR_STR("Failed to retrieve CDI device handle");
            status = NVSIPL_STATUS_ERROR;
        } else {
            NvMediaStatus const nvmerr = m_pCDIDriver->ReadRegister(dev, 0U, static_cast<uint32_t>(address), length, buffer);
            if (nvmerr != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("Failed to perform device read");
                status = NVSIPL_STATUS_ERROR;
            }
        }
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
SIPLStatus CNvMEEPROM::WriteData(const std::uint16_t address,
                                 const std::uint32_t length,
                                 std::uint8_t * const buffer)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if (length == 0UL) {
        LOG_INFO("Invalid write length %u\n", length);
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else if (buffer == nullptr) {
        LOG_INFO("Invalid buffer\n");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else if ((m_pCDIDriver == nullptr) || (m_pCDIDriver->WriteRegister == nullptr)) {
        SIPL_LOG_ERR_STR("EEPROM device is not initialize or not writable");
        status = NVSIPL_STATUS_INVALID_STATE;
    } else {
        DevBlkCDIDevice *dev = nullptr;
        dev = GetCDIDeviceHandle();
        if (dev == nullptr) {
            SIPL_LOG_ERR_STR("Failed to retrieve CDI device handle");
            status = NVSIPL_STATUS_ERROR;
        } else {
            NvMediaStatus nvmerr = NVMEDIA_STATUS_OK;
            nvmerr = m_pCDIDriver->WriteRegister(dev, 0, address, length, buffer);
            if (nvmerr != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR("Failed to perform device write");
                status = NVSIPL_STATUS_ERROR;
            }
        }
    }
    return status;
}
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif // !NV_IS_SAFETY

} // end of nvsipl
