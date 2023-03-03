/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "NvSIPLDeviceBlockInfo.hpp"
#include "CNvMMax96712.hpp"
#include "DeserializerIF/CNvMDeserializerExport.hpp"
#include "sipl_error.h"

#include <chrono>
#include <thread>

extern "C" {
    #include "pwr_utils.h"
    #include "devblk_cdi.h"
}

namespace nvsipl
{

CNvMMax96712::~CNvMMax96712()
{
}

SIPLStatus CNvMMax96712::DoSetConfig(const DeserInfo *deserInfo, DeserializerParams *params)
{
    SIPLStatus status = NVSIPL_STATUS_OK;

    /*! Get CDI Driver */
    m_pCDIDriver = GetMAX96712NewDriver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("MAX96712: GetMAX96712NewDriver() failed");
        return NVSIPL_STATUS_ERROR;
    }

    /* Configure MAX96712 settings */
    status = SetMAX96712Ctx();
    if (status != NVSIPL_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CNvMMax96712::SetMAX96712Ctx failed with SIPL error", (int32_t)status);
        return status;
    }

    return status;
}

// Set Max96712 context
SIPLStatus CNvMMax96712::SetMAX96712Ctx()
{
    ContextMAX96712 *ctx = NULL;
    DriverContextImpl<ContextMAX96712> *driverContext = new DriverContextImpl<ContextMAX96712>();
    if (driverContext == nullptr) {
        SIPL_LOG_ERR_STR("MAX96712: Failed to create CDI driver context in CNvMMax96712::SetMax96712Ctx");
        return NVSIPL_STATUS_ERROR;
    }

    m_upDrvContext.reset(driverContext);
    ctx = &driverContext->m_Context;

    for (std::uint8_t i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        ctx->gmslMode[i] = CDI_MAX96712_GMSL_MODE_UNUSED;
        ctx->longCables[i] = m_longCables[i];
    }

    ctx->defaultResetAll = m_resetAll;

    ctx->camRecCfg = m_camRecCfg;

    for (const DeserLinkModes & item : m_ovLinkModes) {
        if (item.elinkMode == LinkMode::LINK_MODE_GMSL1) {
            ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL1_MODE;
        } else if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_6GBPS) {
            ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_6GBPS;
        } else if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_3GBPS) {
            ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_3GBPS;
        } else if (item.elinkMode == LinkMode::LINK_MODE_NONE) {
            ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL_MODE_UNUSED;
        } else {
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }
    }

    if (m_oDeviceParams.bEnableSimulator) {
        m_I2CPort = 0U;
        m_eInterface = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A;
        m_ePhyMode = NVSIPL_CAP_CSI_DPHY_MODE;
        m_pwrPort = 0U;
    }

    // Select I2C port
    ctx->i2cPort = (m_I2CPort == 0u) ? CDI_MAX96712_I2CPORT_0 :
                   ((m_I2CPort == 1u) ? CDI_MAX96712_I2CPORT_1 : CDI_MAX96712_I2CPORT_2);

    switch (m_eInterface) {
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G:
            ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_C :
                                                                     CDI_MAX96712_TXPORT_PHY_E;
            ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
            for (uint8_t i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                ctx->lanes[i] = 2U;
            }
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H:
            ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_D :
                                                                     CDI_MAX96712_TXPORT_PHY_F;
            ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
            for (uint8_t i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                ctx->lanes[i] = 2U;
            }
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH:
            ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_D :
                                                                     CDI_MAX96712_TXPORT_PHY_E;
#if !NV_IS_SAFETY
            ctx->mipiOutMode = (m_camRecCfg > CAMREC_VER1) ? CDI_MAX96712_MIPI_OUT_4a_2x2 : CDI_MAX96712_MIPI_OUT_2x4;
#else
            ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_2x4;
#endif
            for (uint8_t i = 0U; i < 2U; i++) {
                ctx->lanes[i] = 4U;
            }
            for (uint8_t i = 2U; i < MAX96712_MAX_NUM_PHY; i++) {
#if !NV_IS_SAFETY
                ctx->lanes[i] = (m_camRecCfg > CAMREC_VER1) ? 2U : 4U;
#else
                ctx->lanes[i] = 4U;
#endif
            }
            break;
#if !NV_IS_SAFETY
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G1:
            ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_C :
                                                                     CDI_MAX96712_TXPORT_PHY_E;
            ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
            for (uint8_t i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                ctx->lanes[i] = (i == ctx->txPort) ? 1U : 2U;
            }
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F1:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H1:
            ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_D :
                                                                     CDI_MAX96712_TXPORT_PHY_F;
            ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
            for (uint8_t i = 0U; i < MAX96712_MAX_NUM_PHY; i++) {
                ctx->lanes[i] = (i == ctx->txPort) ? 1U : 2U;
            }
            break;
#endif
        default:
            return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    /* Overwrite txPort if tx_port is defined in DT */
    if (m_TxPort < 4) {
        switch (m_TxPort) {
            case 0:
                    ctx->txPort = CDI_MAX96712_TXPORT_PHY_C;
                    break;
            case 1:
                    ctx->txPort = CDI_MAX96712_TXPORT_PHY_D;
                    break;
            case 2:
                    ctx->txPort = CDI_MAX96712_TXPORT_PHY_E;
                    break;
            case 3:
                    ctx->txPort = CDI_MAX96712_TXPORT_PHY_F;
                    break;
            default:
                    SIPL_LOG_ERR_STR("MAX96712: Bad state, txport must be in range from 0 to 3");
                    return NVSIPL_STATUS_NOT_SUPPORTED;
        }
    }

    /* Set CSI output frequency */
    if (m_oDeviceParams.bEnableSimulator) {
        m_uMipiSpeed = 2500000u;
    } else {
        if (m_ePhyMode == NVSIPL_CAP_CSI_DPHY_MODE) {
            if (ctx->mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
                m_uMipiSpeed = m_dphyRate[X4_CSI_LANE_CONFIGURATION];
            } else {
                m_uMipiSpeed = m_dphyRate[X2_CSI_LANE_CONFIGURATION];
            }
        } else if (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) {
            if (ctx->mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
                m_uMipiSpeed = m_cphyRate[X4_CSI_LANE_CONFIGURATION];
            } else {
                m_uMipiSpeed = m_cphyRate[X2_CSI_LANE_CONFIGURATION];
            }
        }
    }

    /* Set link mask to be enabled */
    ctx->linkMask = m_linkMask;

    /* Set passive mode */
    ctx->passiveEnabled = m_oDeviceParams.bPassive;

    /* Set py mode */
    ctx->phyMode = (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) ? CDI_MAX96712_PHY_MODE_CPHY : CDI_MAX96712_PHY_MODE_DPHY;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96712::DoInit()
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    WriteParametersParamMAX96712 writeParamsMAX96712 = {};
    ReadParametersParamMAX96712 readParamsMAX96712 = {};

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    /* I2C wake time typically takes 2.25ms
     * TODO: add the link lock time up to 60ms
     */
    std::this_thread::sleep_for(std::chrono::milliseconds(3));

    /*! Check deserializer is present */
    LOG_INFO("Check deserializer is present\n");
    nvmStatus = MAX96712CheckPresence(m_upCDIDevice.get());
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712CheckPresence failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    LOG_INFO("Set deserializer defaults\n");
    nvmStatus = MAX96712SetDefaults(m_upCDIDevice.get());
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712SetDefaults failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Get deserializer revision */
    LOG_INFO("Get deserializer revision\n");
    nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(readParamsMAX96712.revision),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_REV_ID) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    m_eRevision = readParamsMAX96712.revision;

    /* Set MIPI output mode */
    LOG_INFO("Set MIPI output mode\n");
    writeParamsMAX96712.MipiSettings.mipiSpeed = m_uMipiSpeed / 100000u;

    writeParamsMAX96712.MipiSettings.phyMode = (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) ? CDI_MAX96712_PHY_MODE_CPHY : CDI_MAX96712_PHY_MODE_DPHY;
    nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI,
                                        sizeof(writeParamsMAX96712.MipiSettings),
                                        &writeParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Disable CSI out */
    nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_DISABLE_CSI_OUT);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_DISABLE_CSI_OUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}


SIPLStatus CNvMMax96712::EnableLinks(uint8_t linkMask)
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    WriteParametersParamMAX96712 writeParamsMAX96712 = {};
    uint8_t link = 0u;

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    /* Get the links currently enabled */
    nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                       CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
                                       sizeof(readParamsMAX96712.link),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }
    link = readParamsMAX96712.link;

    if (link != linkMask) {
        LOG_INFO("Enabling links\n");
        writeParamsMAX96712.link = (LinkMAX96712) linkMask;
        nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                            CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
                                            sizeof(writeParamsMAX96712.link),
                                            &writeParamsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK) failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96712::ControlLinks(const std::vector<LinkAction>& linkActions)
{
    SIPLStatus status = NVSIPL_STATUS_OK;
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    NvMediaStatus nvmStatus;

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    /* Get the links currently enabled */
    nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                       CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
                                       sizeof(int32_t),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS) failed with NvMedia error",
                              static_cast<int32_t>(nvmStatus));
        return ConvertNvMediaStatus(nvmStatus);
    } else {
        uint8_t link = readParamsMAX96712.link;
        bool link_enabled = false;
        uint8_t linkMask = 0U;
        uint8_t linkIdx = 0U;
        uint8_t powerReset = 0U;

        for (std::uint32_t i = 0U; i < linkActions.size(); i++) {
            WriteParametersParamMAX96712 writeParamsMAX96712 = {};
            const LinkAction& item = linkActions[i];
            if ((item.linkIdx & 0x80U) == 0x80U) {
                /* Bit 7 is the flag of power cycling.
                 * Without power reset, remote ERRB needs to be re-enabled after link locked
                 */
                powerReset = 1U;
            }
            linkIdx = item.linkIdx & 0x7FU;
            if (linkIdx >= MAX96712_MAX_NUM_LINK) {
                return NVSIPL_STATUS_ERROR;
            }

            switch (item.eAction) {
            case LinkAction::LINK_ENABLE:
                link |= (static_cast<uint8_t>(1U<< linkIdx));
                linkMask |= (static_cast<uint8_t>(1U<< linkIdx));
                link_enabled = true;
                break;
            case LinkAction::LINK_DISABLE:
                link &= ((static_cast<uint8_t> (~(static_cast<uint8_t>(1U << linkIdx))))& 0x0FU);
                writeParamsMAX96712.linkIndex = linkIdx;
                nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK,
                                                    sizeof(uint8_t),
                                                    &writeParamsMAX96712);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK) failed with NvMedia error",
                                         static_cast<int32_t>(nvmStatus));
                    return ConvertNvMediaStatus(nvmStatus);
                }
                break;
            case LinkAction::LINK_NO_ACTION:
                break;
            default:
                SIPL_LOG_ERR_STR("Invalid Link Action");
                return NVSIPL_STATUS_BAD_ARGUMENT;
            }
        }

        if (linkActions.size() > 0U) {
            WriteParametersParamMAX96712 writeParamsMAX96712 = {};
            writeParamsMAX96712.link = static_cast<LinkMAX96712>(link);
            nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
                                                sizeof(int32_t),
                                                &writeParamsMAX96712);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK) failed with NvMedia error",
                                     static_cast<int32_t>(nvmStatus));
                return ConvertNvMediaStatus(nvmStatus);
            }

            if (link_enabled) {
                status = CheckLinkLock(linkMask);
                if ((status == NVSIPL_STATUS_OK) && (powerReset == 0U)) {
                    /* Restore ERRB Rx Eanble and Video Pipeline Enable bits */
                    for (std::uint32_t i = 0U; i < linkActions.size(); i++) {
                        const LinkAction& item = linkActions[i];
                        if (item.eAction == LinkAction::LINK_ENABLE) {
                            WriteParametersParamMAX96712 writeParamsMAX96712 = {};
                            writeParamsMAX96712.linkIndex = (item.linkIdx & 0x7FU);
                            nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                        CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK,
                                                        sizeof(uint8_t),
                                                        &writeParamsMAX96712);
                            if (nvmStatus != NVMEDIA_STATUS_OK) {
                                SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK) failed with NvMedia error",
                                                     static_cast<int32_t>(nvmStatus));
                                return ConvertNvMediaStatus(nvmStatus);
                            }
                        }
                    }
                }
            }
        }
    }

    return status;
}

SIPLStatus CNvMMax96712::CheckLinkLock(uint8_t linkMask) {
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    for(uint8_t linkIndex = 0; linkIndex < m_ovLinkModes.size(); linkIndex++) {
        const DeserLinkModes& item = m_ovLinkModes[linkIndex];

        if ((linkMask & (1 << linkIndex)) == 0 ) {
            continue;
        }

        if ((item.elinkMode == LinkMode::LINK_MODE_GMSL2_6GBPS) ||
            (item.elinkMode == LinkMode::LINK_MODE_GMSL2_3GBPS)) {
            LOG_INFO("Check config link lock\n");
            nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                            GetMAX96712Link(linkIndex),
                                            CDI_MAX96712_LINK_LOCK_GMSL2,
                                            false);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), (LinkMAX96712)linkMask);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712OneShotReset failed with NvMedia error", (int32_t)nvmStatus);
                    return ConvertNvMediaStatus(nvmStatus);
                }
                nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                              GetMAX96712Link(linkIndex),
                                              CDI_MAX96712_LINK_LOCK_GMSL2,
                                              true);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712CheckLink(CDI_MAX96712_GMSL2_LINK_LOCK) failed with NvMedia error", (int32_t)nvmStatus);
                    return ConvertNvMediaStatus(nvmStatus);
                }
            }
        } else if (item.elinkMode == LinkMode::LINK_MODE_GMSL1) {
            LOG_INFO("Check config link lock \n");
            nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                          GetMAX96712Link(linkIndex),
                                          CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                          false);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), (LinkMAX96712)linkMask);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712OneShotReset failed with NvMedia error", (int32_t)nvmStatus);
                    return ConvertNvMediaStatus(nvmStatus);
                }
                nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                              GetMAX96712Link(linkIndex),
                                              CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                              true);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712CheckLink(CDI_MAX96712_GMSL1_CONFIG_LINK_LOCK) failed with NvMedia error", (int32_t)nvmStatus);
                    return ConvertNvMediaStatus(nvmStatus);
                }
            }
        } else if (item.elinkMode == LinkMode::LINK_MODE_NONE) {
            LOG_INFO("No Need to check the link lock \n");
        } else {
            return NVSIPL_STATUS_NOT_SUPPORTED;
        }
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96712::DoStart()
{
    NvMediaStatus nvmStatus = NVMEDIA_STATUS_OK;
    SIPLStatus status = NVSIPL_STATUS_OK;
    ReadParametersParamMAX96712 readParamsMAX96712 = {};

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        return NVSIPL_STATUS_OK;
    }

    /* Check CSIPLL lock */
    nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK failed with NvMedia error", (int32_t)nvmStatus);
    }

    /* Trigger the initial deskew */
    if ((m_ePhyMode == NVSIPL_CAP_CSI_DPHY_MODE) and (m_uMipiSpeed >= 1500000u)) {
        nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_TRIGGER_DESKEW);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_TRIGGER_DESKEW failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    status = CheckLinkLock(m_linkMask);
    if (status != NVSIPL_STATUS_OK) {
        nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), (LinkMAX96712)m_linkMask);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712OneShotReset failed with NvMedia error", (int32_t)nvmStatus);
            return ConvertNvMediaStatus(nvmStatus);
        }
    }

    /* Enable CSI out */
    nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_ENABLE_CSI_OUT);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: CDI_CONFIG_MAX96712_ENABLE_CSI_OUT failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    /* Check & Clear if ERRB set */
    nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                       CDI_READ_PARAM_CMD_MAX96712_ERRB,
                                       sizeof(readParamsMAX96712.ErrorStatus),
                                       &readParamsMAX96712);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: DevBlkCDIReadParameters(CDI_READ_PARAM_CMD_MAX96712_ERRB) failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    return NVSIPL_STATUS_OK;
}

CNvMDeserializer *CNvMDeserializer_Create() {
    return new CNvMMax96712();
}

const char** CNvMDeserializer_GetName() {
    static const char* names[] = {
        "MAX96712",
        "MAX96722",
        NULL
    };

    return names;
}

SIPLStatus CNvMMax96712::GetErrorSize(size_t & errorSize)
{
#if USE_MOCK_ERRORS
    errorSize = 1;
#else
    errorSize  = sizeof(ErrorStatusMAX96712::globalRegVal) + sizeof(ErrorStatusMAX96712::globalFailureType);
    errorSize += sizeof(ErrorStatusMAX96712::pipelineRegVal) + sizeof(ErrorStatusMAX96712::pipelineFailureType);
    errorSize += sizeof(ErrorStatusMAX96712::linkRegVal) + sizeof(ErrorStatusMAX96712::linkFailureType);
#endif
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96712::GetErrorInfo(std::uint8_t * const buffer,
                                      const std::size_t bufferSize,
                                      std::size_t &size,
                                      bool & isRemoteError,
                                      std::uint8_t& linkErrorMask)
{
#if USE_MOCK_ERRORS
    size = 1U;
    linkErrorMask = 1U;
#else
    ErrorStatusMAX96712 errorStatus;
    NvMediaStatus nvmStatus;

    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    nvmStatus = MAX96712GetErrorStatus(m_upCDIDevice.get(),
                                       sizeof(errorStatus),
                                       &errorStatus);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712GetErrorStatus failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    // Get link error mask
    linkErrorMask = errorStatus.link;

    // Check whether serializer error is detected as well
    nvmStatus = MAX96712GetSerializerErrorStatus(m_upCDIDevice.get(),
                                                 &isRemoteError);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712GetSerializerErrorStatus failed with NvMedia error", (int32_t)nvmStatus);
        return ConvertNvMediaStatus(nvmStatus);
    }

    if (buffer != nullptr) {
        // Get other error details in buffer.
        // Doing this for all regVals and failureTypes
        // but it's up to driver + client to work out how this will be laid out.
        std::size_t sizeMax96712ErrorStatusGlobal = 0U;
        std::size_t sizeMax96712ErrorStatusPipeline = 0U;
        std::size_t sizeMax96712ErrorStatusLink = 0U;
        std::size_t sizeMax96712ErrorStatus = 0U;

        sizeMax96712ErrorStatusGlobal   = (sizeof(errorStatus.globalRegVal))   + (sizeof(errorStatus.globalFailureType));
        sizeMax96712ErrorStatusPipeline = (sizeof(errorStatus.pipelineRegVal)) + (sizeof(errorStatus.pipelineFailureType));
        sizeMax96712ErrorStatusLink     = (sizeof(errorStatus.linkRegVal))     + (sizeof(errorStatus.linkFailureType));
        sizeMax96712ErrorStatus = sizeMax96712ErrorStatusGlobal + sizeMax96712ErrorStatusPipeline + sizeMax96712ErrorStatusLink;

        if (bufferSize < sizeMax96712ErrorStatus)
        {
            SIPL_LOG_ERR_STR("MAX96712: Buffer provided to GetErrorInfo is too small");
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        std::size_t sizeIncr = 0U;

        memcpy(buffer, &errorStatus.globalFailureType, sizeof(errorStatus.globalFailureType));
        sizeIncr = sizeof(errorStatus.globalFailureType);
        memcpy(buffer + sizeIncr, &errorStatus.globalRegVal, sizeof(errorStatus.globalRegVal));
        sizeIncr += sizeof(errorStatus.globalRegVal);

        memcpy(buffer + sizeIncr, &(errorStatus.pipelineFailureType), sizeof(errorStatus.pipelineFailureType));
        sizeIncr += sizeof(errorStatus.pipelineFailureType);
        memcpy(buffer + sizeIncr, &(errorStatus.pipelineRegVal), sizeof(errorStatus.pipelineRegVal));
        sizeIncr += sizeof(errorStatus.pipelineRegVal);

        memcpy(buffer + sizeIncr, &(errorStatus.linkFailureType), sizeof(errorStatus.linkFailureType));
        sizeIncr += sizeof(errorStatus.linkFailureType);
        memcpy(buffer + sizeIncr, &(errorStatus.linkRegVal), sizeof(errorStatus.linkRegVal));

        size = sizeMax96712ErrorStatus;
    }
#endif /* not USE_MOCK_ERRORS */
    return NVSIPL_STATUS_OK;
}

SIPLStatus CNvMMax96712::DoSetPower(bool powerOn)
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
            SIPL_LOG_ERR_STR_INT("MAX96712: CNvMMax96712::DoSetPower failed with SIPL error", (int32_t)status);
            return status;
        }
    } else {
        //For Drive Orin and firespray
        status = (SIPLStatus) DevBlkCDISetDeserPower(m_upCDIDevice.get(),powerOn);
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMax96712_Fusa::DoSetPower failed with SIPL error",
                                static_cast<int32_t>(status));
        }
    }

    return NVSIPL_STATUS_OK;
}

Interface* CNvMMax96712::GetInterface(const UUID &interfaceId)
{
    return nullptr;
}

SIPLStatus CNvMMax96712::GetInterruptStatus(const uint32_t gpioIdx,
                                            IInterruptNotify &intrNotifier)
{
    return NVSIPL_STATUS_NOT_SUPPORTED;
}

} // end of nvsipl
