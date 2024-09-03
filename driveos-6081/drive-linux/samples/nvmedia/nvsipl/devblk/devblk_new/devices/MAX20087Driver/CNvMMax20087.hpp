/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CNVMMAX20087_HPP
#define CNVMMAX20087_HPP

#include "CampwrIF/CNvMCampwr.hpp"
#include "IInterruptNotify.hpp"
#include "utils/utils.hpp"
#include "NvSIPLDeviceBlockTrace.hpp"
#include <devblk_cdi.h>
#include "cdd_nv_error.h"

#define MAX_VALID_LINK_INDEX          (3U)

namespace nvsipl
{

/*! Serializer */
class CNvMMax20087 final : public CNvMCampwr
{
public:

    /**
     * @brief Function to map SIPL log verbosity level to
     * C log verbosity level.
     *
     * @param[in] level SIPL Log trace level.
     * @return C log level.
     */
    C_LogLevel ConvertLogLevel(INvSIPLDeviceBlockTrace::TraceLevel const level);

    /*
     * Max20087GetInterruptStatus checks for all possible errors linked to
     * a particular link group and notifies the caller in case of error.
     */
    static SIPLStatus Max20087GetInterruptStatus(CNvMCampwr* const campwr,
        uint8_t const linkIndex, uint32_t const gpioIdx,
        IInterruptNotify &intrNotifier);

    static SIPLStatus ReadIsetComparator(CNvMCampwr* campwr,
                            uint8_t const linkIndex, uint8_t *data);

    static SIPLStatus Max20087ReadVoltageAndCurrentValues(CNvMCampwr * campwr,
                            uint8_t const linkIndex, uint16_t *data_values,
                            uint8_t const data_values_size);

    SIPLStatus SetConfig(uint8_t i2cAddress, const DeviceParams *const params) override;

    SIPLStatus GetErrorSize(size_t & errorSize) override;

    SIPLStatus GetErrorInfo(std::uint8_t * const buffer,
                            std::size_t const bufferSize,
                            std::size_t &size) override;


    SIPLStatus isSupported() override;

    SIPLStatus PowerControlSetUnitPower(DevBlkCDIDevice* cdiDev, uint8_t const linkIndex, bool const enable) override;

    SIPLStatus CreatePowerDevice(DevBlkCDIRootDevice* const cdiRootDev, const uint8_t linkIndex) override;

    DevBlkCDIDevice* GetDeviceHandle() override;

    SIPLStatus CheckPresence(DevBlkCDIRootDevice* const cdiRootDev,
                             DevBlkCDIDevice* const cdiDev) override;

    SIPLStatus InitPowerDevice(DevBlkCDIRootDevice* const cdiRootDev,
                DevBlkCDIDevice* const cdiDev, uint8_t const linkIndex,
                int32_t const csiPort) override;

    /**
     * Read from a PowerSwitch Register.
    */
    SIPLStatus ReadRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength, uint8_t * const dataBuff) override;

    /**
     * Write to a PowerSwitch Register.
    */
    SIPLStatus WriteRegister(DevBlkCDIDevice const * const handle,
                uint8_t const linkIndex, uint32_t const registerNum,
                uint32_t const dataLength,
                uint8_t const * const dataBuff) override;

    virtual SIPLStatus DoInit()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStart()
    {
        return NVSIPL_STATUS_OK;
    };

    virtual SIPLStatus DoStop()
    {
        return NVSIPL_STATUS_OK;
    };

    /**
     * Mask or restore mask of power switch interrupts
     */
    SIPLStatus MaskRestoreInterrupt(const bool enableGlobalMask) final;

private:

    /* To hold the interrupt mask set during power switch initialization */
    uint8_t m_SavedInterruptMask = 0xFFU;

    /* Interrupt mask state */
    enum class interruptMaskState : std::uint8_t
    {
        /* Interrupt mask initialized state */
        INTERRUPT_MASK_INITED_STATE = 0,
        /* Interrupt gloablly masked state */
        INTERRUPT_GLOBAL_MASKED_STATE = 1,
        /* Original interrupt mask restored state */
        INTERRUPT_MASK_RESTORED_STATE = 2
    };

    /* Hold the interrupt mask state for MaskRestoreInterrupt API */
    interruptMaskState m_interrupMasktState = interruptMaskState::INTERRUPT_MASK_INITED_STATE;
};

} // end of namespace nvsipl
#endif // CNVMMAX20087_HPP
