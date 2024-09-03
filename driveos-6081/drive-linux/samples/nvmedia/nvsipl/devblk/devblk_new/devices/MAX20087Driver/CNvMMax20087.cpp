/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax20087.hpp"
#include "NvSIPLDeviceBlockTrace.hpp"
#include "cdi_max20087.h"
#include "sipl_error.h"
#include "sipl_util.h"
#include "os_common.h"
#include "cdd_nv_error.h"

#define GET_4_LSB(a)     ((a) & (LSB_4_MASK))
#define GET_4_MSB(a)     (((a) & (MSB_4_MASK)) >> 4U)

namespace nvsipl
{

/**
 * Verifying I2C transactions for mask, config and ID registers with read after
 * read. Skipping I2C readback for Stat registers (due to clear on read
 * property) and ADC registers (as it may report false positive)
*/
static SIPLStatus
ReadRegisterVerify(CNvMCampwr & campwr, uint8_t const linkIndex,
                    uint32_t const registerNum, uint32_t const dataLength,
                    uint8_t * const dataBuff)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    uint8_t dataVerify[dataLength];

    status = campwr.ReadRegister(campwr.GetDeviceHandle(),
                linkIndex, registerNum, dataLength, dataBuff);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    SIPLStatus statusVerify {NVSIPL_STATUS_OK};
    statusVerify = campwr.ReadRegister(campwr.GetDeviceHandle(),
                        linkIndex, registerNum, dataLength, dataVerify);
    if (statusVerify != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087: Readback verify on config type register "
                            "failed");
        return statusVerify;
    }
    uint32_t i {0U};
    for (; i < dataLength; i++) {
        if (dataVerify[i] != dataBuff[i]) {
            SIPL_LOG_ERR_STR("MAX20087: Readback verify on config type register"
                                " failed");
            return NVSIPL_STATUS_ERROR;
        }
    }

    return status;
}

/**
 * Verifies write transactions to a PowerSwitch register with a read after
 * write.
*/
static SIPLStatus
WriteRegisterVerify(CNvMCampwr & campwr, uint8_t const linkIndex,
                        uint32_t const registerNum, uint32_t const dataLength,
                        uint8_t const * const dataBuff)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    uint8_t dataVerify[dataLength];

    status = campwr.WriteRegister(campwr.GetDeviceHandle(),
                    linkIndex, registerNum, dataLength, dataBuff);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    SIPLStatus statusVerify {NVSIPL_STATUS_OK};
    statusVerify = campwr.ReadRegister(campwr.GetDeviceHandle(),
                        linkIndex, registerNum, dataLength, dataVerify);
    if (statusVerify != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087: Readback verify on config type register "
                            "failed");
        return statusVerify;
    }
    uint32_t i {0U};
    for (; i < dataLength; i++) {
        if (dataVerify[i] != dataBuff[i]) {
            SIPL_LOG_ERR_STR("MAX20087: Readback verify on config type register"
                                " failed");
            return NVSIPL_STATUS_ERROR;
        }
    }

    return status;
}

/**
 * MISRA static analysis reports violation as "const" keyword is not used in
 * function parameter. False Positive maybe?
 **/
/* coverity[misra_c_2012_rule_8_13_violation] : intentional */
C_LogLevel CNvMMax20087::ConvertLogLevel(
    INvSIPLDeviceBlockTrace::TraceLevel const level)
{
    return static_cast<C_LogLevel>(level);
}

/*
 * Step 1: Get the Output Voltage
 * Read REG_CONFIG, without altering any other bit, write 01 to MUX[1:0],
 * we have a 0.5 ms delay and then read register ADC1-4 for link 0 to 3
 * respectively. The respective ADC register will store the output Voltage
 * per channel for that link_index.
 *
 * Step 2: Get the Input Voltage
 * Read REG_CONFIG, without altering any other bit, write 10 to MUX[1:0],
 * we have a 0.5 ms delay and then read register ADC1-3, ADC1 will store
 * VIN, ADC2 will store VDD and ADC3 will store VISET.
 *
 * Step 3: Get the Output Channel Current
 * Read REG_CONFIG, without altering any other bit, write 00 to MUX[1:0],
 * we have a 0.5 ms delay and then read register ADC1-4 for link 0 to 3
 * respectively. The respective ADC register will store the Output channel
 * Current for that link_index.
 */

SIPLStatus CNvMMax20087::Max20087ReadVoltageAndCurrentValues(
                         CNvMCampwr * campwr,
                         uint8_t const linkIndex,
                         uint16_t data_values[],
                         uint8_t const data_values_size)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    /** to store register data for read and write operations */
    uint8_t data;

    if ((NULL == campwr) ||
        (NULL == data_values) ||
        (data_values_size != DATA_VALUES_ARR_LENGTH))
    {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
        return status;
    }

    status = ReadRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                                REGISTER_DATA_LENGTH, &data);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Setting MUX[1:0] to 01 for reading Output Voltage */
    data &= ~(1U << MUX_BIT1);
    data |=   1U << MUX_BIT0;

    status = WriteRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                                 REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Delay given after write to give time for the values to get set */
    nvsleep(500); // 0.5 ms

    /** Reading Output Voltage for specified link/channel */
    status = campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
                ADDR_REG_ADC1 + linkIndex, REGISTER_DATA_LENGTH, &data);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    data_values[DATA_INDEX_VOUT] = data * VOLTAGE_MULTIPLIER;

    status = ReadRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                                REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Setting MUX[1:0] to 10 for reading Input Voltages */
    data &= ~(1U << MUX_BIT0);
    data |=   1U << MUX_BIT1;

    status = WriteRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                                 REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Delay given after write to give time for the values to get set */
    nvsleep(500); // 0.5 ms

    /** Reading Vin value from Register ADC1 */
    status = campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
                ADDR_REG_ADC1, REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    data_values[DATA_INDEX_VIN] = data * VIN_MULTIPLIER;

    /** Reading Vdd value from Register ADC2 */
    status = campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
                ADDR_REG_ADC2, REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    data_values[DATA_INDEX_VDD] = data * VDD_MULTIPLIER;

    /** Reading Viset value from Register ADC3 */
    status = campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
                ADDR_REG_ADC3, REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    data_values[DATA_INDEX_VISET] = data * VISET_MULTIPLIER;

    status = ReadRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Setting MUX[1:0] to 00 for Reading Current Values */
    data &= ~(1U << MUX_BIT0);
    data &= ~(1U << MUX_BIT1);

    status = WriteRegisterVerify(*campwr, linkIndex, ADDR_REG_CONFIG,
                                 REGISTER_DATA_LENGTH, &data);

    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    /** Delay given after write to give time for the values to get set */
    nvsleep(500); // 0.5 ms

    /** Reading Current for specified link/channel */
    status = campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
             ADDR_REG_ADC1 + linkIndex, REGISTER_DATA_LENGTH, &data);
    if (status != NVSIPL_STATUS_OK) {
        return status;
    }

    data_values[DATA_INDEX_CURRENT] = data * CURRENT_MULTIPLIER;

    LOG_INFO("MAX20087: SM9, SM10, SM18 Max20087ReadVoltageAndCurrentValues"
                " is successfully executed\n");

    return status;
}

/*
 * Max20087ReadErrorBits checks for all possible errors linked to
 * a particular link group (contains 2 links cause STAT2 register stores
 * 2 different link error bits in one Byte). As STAT2 register is a clear
 * on read register hence we need to read error bits for both links at
 * once and pass them to the caller via errorCode, so that the caller can
 * notify the user in any error case.
 *
 * It fills in 16bit data to errorCode in below format
 * Link Group 0 format (Link 0 and 1)
 * [OVIN|UVIN|OVDD|UVDD|TS2|OC2|OV2|UV2][OVIN|UVIN|OVDD|UVDD|TS1|OC1|OV1|UV1]
 * Link Group 1 format(Link 2 and 3)
 * [OVIN|UVIN|OVDD|UVDD|TS4|OC4|OV4|UV4][OVIN|UVIN|OVDD|UVDD|TS3|OC3|OV3|UV3]
 */
static SIPLStatus Max20087ReadErrorBits(CNvMCampwr * campwr,
    uint8_t const linkIndex, uint16_t *errorCode)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    uint8_t enabledLinkData {0U};
    uint8_t stat1Data {0U};
    uint8_t stat2Data {0U};
    bool read_stat2_1 {true};
    constexpr uint8_t MAX_LINK_NUM     {0x04U};

    if ((campwr == NULL) || (linkIndex >= MAX_LINK_NUM) ||
        (errorCode == NULL)) {
        SIPL_LOG_ERR_STR("Max20087GetInterruptStatus : Bad Arguments");
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    /* Read config Reg */
    status = ReadRegisterVerify(
                *campwr,
                linkIndex,
                ADDR_REG_CONFIG,
                REGISTER_DATA_LENGTH,
                &enabledLinkData);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "Max20087GetInterruptStatus : Failed to read ADDR_REG_CONFIG");
        return status;
    }
    /* Clean up the data to only store EN[1:4] */
    enabledLinkData &= ALL_OUT_LINKS_ENABLED;

    /* Read STAT1 Reg */
    status = campwr->ReadRegister(
                campwr->GetDeviceHandle(),
                linkIndex,
                ADDR_REG_STAT1,
                REGISTER_DATA_LENGTH,
                &stat1Data);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "Max20087GetInterruptStatus : Failed to read ADDR_REG_STAT1");
        return status;
    }
    /* Clean up the data to only store Input Error Bits(0-3) */
    stat1Data &= REG_STAT1_MASK_INP_ERR;

    /* Read STAT2 Reg */
    read_stat2_1 = (linkIndex / NUM_LINKS_PER_STAT2) == 0U;
    status = campwr->ReadRegister(
                campwr->GetDeviceHandle(),
                linkIndex,
                read_stat2_1 ? ADDR_REG_STAT2_1 : ADDR_REG_STAT2_2,
                REGISTER_DATA_LENGTH,
                &stat2Data);
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "Max20087GetInterruptStatus : Failed to read ADDR_REG_STAT2_X");
        return status;
    }

    /* Write Input and Output Error Bits per Link in one Stat2 Byte. */
    *errorCode = 0U;
    /*
     * Copy input error bits from STAT1DATA to inputErrorCode in below format
     * [OVIN|UVIN|OVDD|UVDD|0|0|0|0]
     */
    uint16_t inputErrorCode = 0U;
    inputErrorCode |= stat1Data;
    inputErrorCode = lshift16(inputErrorCode, 4U);
    inputErrorCode &= 0x00F0U;

    /*
     * Append the error bits from STAT2DATA specific to the link group to which
     * linkIndex belongs in below format.
     * [OVIN|UVIN|OVDD|UVDD|TS|OC|OV|UV]
     * as we need to pass error bits for a link group, we club 2 8bit
     * data (such as above) to fill 16bit errorCode.
     */
    if (read_stat2_1) {
        if (enabledLinkData & REG_CONFIG_MASK_EN2) {
            *errorCode |= inputErrorCode;
            *errorCode |= static_cast<uint16_t>(GET_4_MSB(stat2Data));
        }
        *errorCode = lshift16(*errorCode, 8U);
        if (enabledLinkData & REG_CONFIG_MASK_EN1) {
            *errorCode |= inputErrorCode;
            *errorCode |= static_cast<uint16_t>(GET_4_LSB(stat2Data));
        }
    } else {
        if (enabledLinkData & REG_CONFIG_MASK_EN4) {
            *errorCode |= inputErrorCode;
            *errorCode |= static_cast<uint16_t>(GET_4_MSB(stat2Data));
        }
        *errorCode = lshift16(*errorCode, 8U);
        if (enabledLinkData & REG_CONFIG_MASK_EN3) {
            *errorCode |= inputErrorCode;
            *errorCode |= static_cast<uint16_t>(GET_4_LSB(stat2Data));
        }
    }

    LOG_INFO("MAX20087: SM1, SM2, SM3, SM6, SM14 Max20087GetInterruptStatus"
                " is successfully executed\n");

    return status;
}

SIPLStatus CNvMMax20087::Max20087GetInterruptStatus(CNvMCampwr* const campwr,
    uint8_t const linkIndex, uint32_t const gpioIdx,
    IInterruptNotify &intrNotifier)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    uint16_t errorCode {0U};

    status = Max20087ReadErrorBits(campwr, linkIndex, &errorCode);
    if (status == NVSIPL_STATUS_OK) {
        /*
        * Max20087ReadErrorBits fills in error data for 2 links
        * at a time as STAT2 Register store data for 2 different links
        * in one Byte.
        * baseLinkIdxMultiple is the variable used to get the link group,
        * i.e. either 0(for channels 0 and 1) or 1(for channels(2 and 3)
        * for PowerSwitch, to which the interrupt has to be notified
        * STAT2_DATA_MASK_0 and STAT2_DATA_MASK_1 are used to get to the
        * correct link Index, in a given link group, to which we need to
        * notify the interrupt.
        */
        constexpr uint16_t PWR_STAT2_LS_BYTE  {0x00FFU};
        constexpr uint16_t PWR_STAT2_MS_BYTE  {0xFF00U};
        constexpr uint8_t PWR_STAT2_PER_BYTE  {0x02U};
        uint8_t const baseLinkIdxMultiple {
            static_cast<uint8_t>(linkIndex / PWR_STAT2_PER_BYTE)};
        if ((static_cast<uint32_t>(errorCode) &
            static_cast<uint32_t>(PWR_STAT2_LS_BYTE)) > 0U) {
            constexpr uint8_t PWR_STAT2_DATA_MASK_0 {0x00U};
            status = intrNotifier.Notify(
                InterruptCode::INTR_STATUS_PWR_FAILURE,
                (static_cast<uint64_t>(errorCode) &
                    static_cast<uint64_t>(PWR_STAT2_LS_BYTE)),
                gpioIdx,
                (static_cast<uint32_t>(baseLinkIdxMultiple)*PWR_STAT2_PER_BYTE)
                    + PWR_STAT2_DATA_MASK_0);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("IInterruptNotify::Notify failed",
                    static_cast<int32_t>(status));
                return status;
            }
        }
        if ((static_cast<uint32_t>(errorCode) &
            static_cast<uint32_t>(PWR_STAT2_MS_BYTE)) > 0U) {
            constexpr uint8_t PWR_STAT2_DATA_MASK_1 {0x01U};
            status = intrNotifier.Notify(
                InterruptCode::INTR_STATUS_PWR_FAILURE,
                ((static_cast<uint64_t>(errorCode) &
                    static_cast<uint64_t>(PWR_STAT2_MS_BYTE)) >> 8U),
                gpioIdx,
                (static_cast<uint32_t>(baseLinkIdxMultiple)*PWR_STAT2_PER_BYTE)
                    + PWR_STAT2_DATA_MASK_1);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("IInterruptNotify::Notify failed",
                    static_cast<int32_t>(status));
                return status;
            }
        }
        static_cast<void>(baseLinkIdxMultiple);
    } else {
        SIPL_LOG_ERR_STR("Call to Max20087ReadErrorBits failed");
    }
    return status;
}

SIPLStatus CNvMMax20087::ReadIsetComparator(CNvMCampwr* campwr,
                            uint8_t const linkIndex, uint8_t *data)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    if ((nullptr == campwr) || (campwr->isSupported() != NVSIPL_STATUS_OK) ||
        (linkIndex > MAX_VALID_LINK_INDEX) || (nullptr == data)) {
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    } else {
        status =  campwr->ReadRegister(campwr->GetDeviceHandle(), linkIndex,
                          ADDR_REG_STAT1, REGISTER_DATA_LENGTH, data);

        *data = (((*data) & MASK_STAT1_ISET) == MASK_STAT1_ISET) ?
                (uint8_t){1U} : (uint8_t){0U};
    }
    if (status == NVSIPL_STATUS_OK) {
        LOG_INFO("MAX20087: SM11: ReadIsetComparator is successfully "
                    "executed\n");
    }
    return status;
}

SIPLStatus CNvMMax20087::SetConfig(uint8_t i2cAddress, const DeviceParams *const params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    status = CNvMCampwr::SetConfig(i2cAddress, params);
    if (status == NVSIPL_STATUS_OK) {
        /*! Get CDI Driver */
        m_pCDIDriver = GetMAX20087Driver();
        if (m_pCDIDriver == nullptr) {
            SIPL_LOG_ERR_STR("GetMAX20087Driver() failed!");
            status = NVSIPL_STATUS_ERROR;
        }
    } else {
        SIPL_LOG_ERR_STR_INT("CNvMCampwr::SetConfig failed with SIPL error",
                             static_cast<int32_t>(status));
    }

    return status;
}

SIPLStatus CNvMMax20087::GetErrorSize(size_t & errorSize)
{
    errorSize = 0;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax20087::GetErrorInfo(std::uint8_t * const buffer,
                                     const std::size_t bufferSize,
                                     std::size_t &size)
{
    static_cast<void>(buffer);
    static_cast<void>(bufferSize);
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax20087::isSupported()
{
    if (m_eState == DeviceState::CDI_DEVICE_CREATED) {
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }
}

SIPLStatus CNvMMax20087::PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};

    status = MAX20087SetLinkPower(cdiDev, linkIndex, enable,
                                  static_cast<uint8_t>(m_interrupMasktState),
                                  &m_SavedInterruptMask);
    if (status == NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_OK;
    } else {
        return NVSIPL_STATUS_ERROR;
    }
}

SIPLStatus CNvMMax20087::CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev, const uint8_t linkIndex)
{
    return CNvMDevice::CreateCDIDevice(cdiRootDev, linkIndex);
}

DevBlkCDIDevice* CNvMMax20087::GetDeviceHandle()
{
    return CNvMDevice::GetCDIDeviceHandle();
}

SIPLStatus CNvMMax20087::CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                                       DevBlkCDIDevice* const cdiDev)
{
    static_cast<void>(cdiRootDev);
    NvMediaStatus status {NVMEDIA_STATUS_OK};
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        INvSIPLDeviceBlockTrace::TraceLevel level = instance-> GetLevel();
        C_LogLevel c_level = ConvertLogLevel(level);
        SetCLogLevel(c_level);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    status = MAX20087CheckPresence(cdiDev);
    if (status != NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_ERROR;
    } else {
        return NVSIPL_STATUS_OK;
    }
}

SIPLStatus CNvMMax20087::InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                            DevBlkCDIDevice* const cdiDev,
                            uint8_t const linkIndex,
                            int32_t const csiPort)
{
    static_cast<void>(linkIndex);
    NvMediaStatus status {NVMEDIA_STATUS_OK};

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        INvSIPLDeviceBlockTrace::TraceLevel level = instance-> GetLevel();
        C_LogLevel c_level = ConvertLogLevel(level);
        SetCLogLevel(c_level);
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
    if (csiPort < 0) {
        SIPL_LOG_ERR_STR_INT("Incorrect csiPort passed to InitPowerDevice: ", csiPort);
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }
    status = MAX20087Init(cdiRootDev, cdiDev, csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        return NVSIPL_STATUS_ERROR;
    } else {
        return NVSIPL_STATUS_OK;
    }
}

/**
 * Read PowerSwitch Register
*/
SIPLStatus
CNvMMax20087::ReadRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t * const dataBuff)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};

    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        constexpr uint32_t const deviceIndex {0U};
        static_cast<void>(linkIndex);
        status = MAX20087ReadRegister(handle, deviceIndex, registerNum,
                    dataLength, dataBuff);
    }

    return ConvertNvMediaStatus(status);
}

/**
 * Write PowerSwitch Register
*/
SIPLStatus
CNvMMax20087::WriteRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t const * const dataBuff)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};

    if ((handle == nullptr) || (dataBuff == nullptr)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        constexpr uint32_t const deviceIndex {0U};
        static_cast<void>(linkIndex);
        status = MAX20087WriteRegister(handle, deviceIndex, registerNum,
                    dataLength, dataBuff);
    }

    return ConvertNvMediaStatus(status);
}

/* Mask or restore mask of power switch interrupts */
SIPLStatus
CNvMMax20087::MaskRestoreInterrupt(const bool enableGlobalMask)
{
    NvMediaStatus status {NVMEDIA_STATUS_OK};
    DevBlkCDIDevice const * const handle {GetDeviceHandle()};

    // Verify that the enum values match the header file defines.
    static_assert(static_cast<uint32_t>(
        interruptMaskState::INTERRUPT_GLOBAL_MASKED_STATE) == INTERRUPT_MASKED_STATE);
    static_assert(static_cast<uint32_t>(
        interruptMaskState::INTERRUPT_MASK_RESTORED_STATE) == INTERRUPT_RESTORED_STATE);

    /* Check if interrupt mask is not already in restored state */
    if (m_interrupMasktState != interruptMaskState::INTERRUPT_MASK_RESTORED_STATE) {
        /* Check if enableMask is true to mask interrupt */
        if (enableGlobalMask) {
            /* Check if interrupt mask is in initialized state */
            if (m_interrupMasktState == interruptMaskState::INTERRUPT_MASK_INITED_STATE) {

                /* Save the current interrupt mask */
                status = MAX20087MaskRestoreGlobalInterrupt(handle, &m_SavedInterruptMask,
                                                            enableGlobalMask);
                m_interrupMasktState = interruptMaskState::INTERRUPT_GLOBAL_MASKED_STATE;
            }
        } else {
            /* Restore masks only if interrupts were masked previously
             * Means m_interruptMaskState is in INTERRUPT_GLOBAL_MASKED_STATE */
            if (m_interrupMasktState == interruptMaskState::INTERRUPT_GLOBAL_MASKED_STATE) {

                /* Restore mask for interrupts */
                status = MAX20087MaskRestoreGlobalInterrupt(handle, &m_SavedInterruptMask,
                                                            enableGlobalMask);
            }
        }
    }

    return ConvertNvMediaStatus(status);
}

} // end of namespace nvsipl
