/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMDS90UB9724.hpp"
#include "DeserializerIF/CNvMDeserializerExport.hpp"
#include "sipl_error.h"

extern "C" {
    #include "pwr_utils.h"
}

namespace nvsipl
{

CNvMDS90UB9724::~CNvMDS90UB9724() { }

SIPLStatus CNvMDS90UB9724::DoSetConfig(const DeserInfo *deserInfo, DeserializerParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    /*! Get CDI Driver */
    m_pCDIDriver = GetDS90UB9724NewDriver();
    if (m_pCDIDriver == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: GetDS90UB9724NewDriver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    /* Configure DS90UB9724 settings */
    status = SetDS90UB9724Ctx();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: SetDS90UB9724Ctx() failed with SIPL error", (int32_t)status);
        return status;
    }

    return NVSIPL_STATUS_OK;
}

// Set DS90UB9724 context
SIPLStatus CNvMDS90UB9724::SetDS90UB9724Ctx()
{
    ContextDS90UB9724 *ctx = NULL;

    DriverContextImpl<ContextDS90UB9724> *driverContext = new DriverContextImpl<ContextDS90UB9724>();
    if (!driverContext) {
        SIPL_LOG_ERR_STR("DS90UB9724: Failed to create CDI driver context");
        return NVSIPL_STATUS_ERROR;
    }

    m_upDrvContext.reset(driverContext);
    ctx = &driverContext->m_Context;

    // Select I2C port
    ctx->i2cPort = (m_I2CPort == 0u) ? CDI_DS90UB9724_I2CPORT_0 : CDI_DS90UB9724_I2CPORT_1;

    /* Set CSI output frequency */
    if (m_ePhyMode == NVSIPL_CAP_CSI_DPHY_MODE) {
        m_uMipiSpeed = 2500000u; /* 400MHz, 800, 1200, 1600, 2500MHz available */
    } else if (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) {
        m_uMipiSpeed = 2500000u; /* (Supported c-phy speeds?) */
    } else {
        SIPL_LOG_ERR_STR("DS90UB9724: mode not supported");
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    /* Set link mask to be enabled */
    ctx->linkMask = m_linkMask;

    /* Set passive mode */
    ctx->passiveEnabled = m_oDeviceParams.bPassive;

    /* Set py mode */
    ctx->phyMode = (m_ePhyMode==NVSIPL_CAP_CSI_DPHY_MODE)? CDI_DS90UB9724_PHY_MODE_DPHY : CDI_DS90UB9724_PHY_MODE_CPHY;

    if ((m_uMipiSpeed == 400000) || (m_uMipiSpeed == 800000) || (m_uMipiSpeed == 1200000) ||
        (m_uMipiSpeed == 1600000) || (m_uMipiSpeed == 2500000)) {
        /* Set CSI speed */
        ctx->mipiSpeed = m_uMipiSpeed;
    } else {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Unsupported CSI MIPI data rate", (int32_t)m_uMipiSpeed);
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::DoInit()
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    /*! Check deserializer is present */
    nvmStatus = DS90UB9724CheckPresence(m_upCDIDevice.get());
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: DS90UB9724CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    nvmStatus = DS90UB9724SetDefaults(m_upCDIDevice.get());
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: DS90UB9724SetDefaults failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::EnableLinks(uint8_t linkMask)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::ControlLinks(const std::vector<LinkAction>& linkActions)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::CheckLinkLock(uint8_t linkMask)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::DoStart()
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (m_oDeviceParams.bEnableSimulator) {
        return NVSIPL_STATUS_OK;
    }

    nvmStatus = DS90UB9724SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_DS90UB9724_SET_MIPI);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_CONFIG_DS90UB9724_SET_MIPI failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Trigger the initial deskew */
    if (m_ePhyMode == NVSIPL_CAP_CSI_DPHY_MODE and m_uMipiSpeed >= 1500000u) {
        nvmStatus = DS90UB9724SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UB9724: CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    return NVSIPL_STATUS_OK;
}

CNvMDeserializer* CNvMDeserializer_Create() {
    return new CNvMDS90UB9724();
}

const char** CNvMDeserializer_GetName() {
    static const char* names[] = {
        "DS90UB9724",
        NULL
    };

    return names;
}

SIPLStatus CNvMDS90UB9724::GetErrorSize(size_t & errorSize)
{
    errorSize = 0;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::GetErrorInfo(std::uint8_t * const buffer,
                                        const std::size_t bufferSize,
                                        std::size_t &size,
                                        bool & isRemoteError,
                                        std::uint8_t& linkErrorMask)
{
    size = 0U;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMDS90UB9724::DoSetPower(bool powerOn)
{
    DevBlkCDIPowerControlInfo m_pwrControlInfo;
    SIPLStatus status = NVSIPL_STATUS_OK;
    const NvMediaStatus nvmStatus = DevBlkCDIGetDesPowerControlInfo(m_upCDIDevice.get(),
                                                                    &m_pwrControlInfo);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Deser power control method with NVMEDIA error",
                              static_cast<int32_t>(nvmStatus));
        return ConvertNvMediaStatus(nvmStatus);
    }

    if (!m_pwrControlInfo.method) {
        // Default is NvCCP, other power backends can be used here based on platform/usecase.
        status = PowerControlSetAggregatorPower(m_pwrPort, m_oDeviceParams.bPassive, powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DS90UB9724: CNvMDS90UB9724::DoSetPower failed with SIPL error", (int32_t)status);
            return status;
        }
    } else {
        //For Drive Orin and firespray
        status = (SIPLStatus) DevBlkCDISetDeserPower(m_upCDIDevice.get(),powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMDS90UB9724::DoSetPower failed with SIPL error",
                                static_cast<int32_t>(status));
        }
    }

    return NVSIPL_STATUS_OK;
}


Interface* CNvMDS90UB9724::GetInterface(const UUID &interfaceId)
{
    return nullptr;
}

SIPLStatus CNvMDS90UB9724::GetInterruptStatus(const uint32_t gpioIdx,
                                            IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of nvsipl
