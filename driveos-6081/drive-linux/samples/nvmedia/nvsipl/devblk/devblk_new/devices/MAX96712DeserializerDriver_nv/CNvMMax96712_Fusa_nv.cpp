/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "CNvMMax96712_Fusa_nv.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "DeserializerIF/CNvMDeserializerExport.hpp"
#include "MAX96712_Fusa_nv_CustomInterface.hpp"
#include "cdi_max96712_nv.h"
#include "pwr_utils.h"
#include "devblk_cdi.h"
#include "sipl_error.h"
#include "sipl_util.h"

#include <chrono>
#include <thread>

/* TODO : Clean this up
 * SoC's GPIO Index 0, to which DES ERRB and LOCK pins are connected
 * Even for multiple Deserializer instances on Orin SoC, the GPIO
 * index still remains "0" for all of them.
 */
#define SOC_GPIO_IDX_0      0U

namespace nvsipl
{
/**
 * False report for coverity violation M0-1-10.
 * Following APIs are to be called by external application and hence no call
 * is found in the current module.
 */
CNvMMax96712_Fusa::CNvMMax96712_Fusa () : CNvMDeserializer()
{
    m_eRevision = CDI_MAX96712_REV_INVALID;
    // Required API: squelch M0-1-10
    static_cast<void>(&nvsipl::CNvMMax96712_Fusa::GetVidSeqErrStatus \
                      != nullptr);
}

CNvMMax96712_Fusa::~CNvMMax96712_Fusa()
{
}

C_LogLevel CNvMMax96712_Fusa::ConvertLogLevel(
    INvSIPLDeviceBlockTrace::TraceLevel const level
)
{
    return static_cast<C_LogLevel>(level);
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::DoSetConfig(
    DeserInfo const* const deserInfoObj,
    DeserializerParams *const params)
{
    (void) params;
    SIPLStatus status {NVSIPL_STATUS_OK};
#if !NV_IS_SAFETY || SAFETY_DBG_OV
    /*! Get the deserializer name */
    m_sDeserializerName = deserInfoObj->name;
#endif // !NV_IS_SAFETY || SAFETY_DBG_OV
    /*! Get CDI Driver */
    m_pCDIDriver = GetMAX96712NewDriver();
    if (m_pCDIDriver == nullptr) {
        SIPL_LOG_ERR_STR("GetMAX96712NewDriver() failed!");
        status = NVSIPL_STATUS_ERROR;
    } else {
        /* Configure MAX96712 settings */
        status = SetMAX96712Ctx();
        if (status != NVSIPL_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CNvMMax96712_Fusa::SetMAX96712Ctx failed with SIPL error",
                                 static_cast<int32_t>(status));
        }
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::UpdateMipiSpeed_CNvMMax96712Fusa(ContextMAX96712 const *const ctx)
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    /* Set CSI output frequency */
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
    } else {
        status = NVSIPL_STATUS_NOT_SUPPORTED;
    }
    return status;
}

SIPLStatus CNvMMax96712_Fusa::UpdateTxPort_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    /* Overwrite txPort if tx_port is defined in DT */
    if (m_TxPort != UINT32_MAX) {
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
            SIPL_LOG_ERR_STR("txport has a range from 0 to 3");
            status = NVSIPL_STATUS_NOT_SUPPORTED;
            break;
        }
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::UpdateCtxConfiguration_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    // Select I2C port
    ctx->i2cPort = (m_I2CPort == 0U) ? CDI_MAX96712_I2CPORT_0 :
                   ((m_I2CPort == 1U) ? CDI_MAX96712_I2CPORT_1 : CDI_MAX96712_I2CPORT_2);

    switch (m_eInterface) {
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G:
        ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_C :
                                                                 CDI_MAX96712_TXPORT_PHY_E;
        ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
        break;
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H:
        ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_D :
                                                                   CDI_MAX96712_TXPORT_PHY_F;
        ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_4x2;
        break;
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF:
    case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH:
        ctx->txPort = (ctx->i2cPort == CDI_MAX96712_I2CPORT_0) ? CDI_MAX96712_TXPORT_PHY_D :
                                                                 CDI_MAX96712_TXPORT_PHY_E;
#if !NV_IS_SAFETY || SAFETY_DBG_OV
        ctx->mipiOutMode = (ctx->cfgPipeCopy == CDI_MAX96712_CFG_PIPE_COPY_MODE_2) ?
            CDI_MAX96712_MIPI_OUT_4a_2x2 : CDI_MAX96712_MIPI_OUT_2x4;
#else
        ctx->mipiOutMode = CDI_MAX96712_MIPI_OUT_2x4;
#endif
        break;
    default:
        SIPL_LOG_ERR_STR("txport has a range from 0 to 3");
        status = NVSIPL_STATUS_NOT_SUPPORTED;
        break;
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::UpdateGmslMode_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    for (DeserLinkModes const& item : m_ovLinkModes) {
        if (item.linkIndex < MAX96712_MAX_NUM_LINK) {
        /* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            if (item.elinkMode == LinkMode::LINK_MODE_GMSL1) {
                ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL1_MODE;
            } else
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
            if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_6GBPS) {
                ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_6GBPS;
            } else if (item.elinkMode == LinkMode::LINK_MODE_GMSL2_3GBPS) {
                ctx->gmslMode[item.linkIndex] = CDI_MAX96712_GMSL2_MODE_3GBPS;
            } else {
                status = NVSIPL_STATUS_NOT_SUPPORTED;
                break;
            }
        } else {
            status = NVSIPL_STATUS_NOT_SUPPORTED;
            break;
        }
    }

    return status;
}

void CNvMMax96712_Fusa::SetCtxInitValue_CNvMMax96712Fusa(ContextMAX96712 * const ctx) const
{
    for (std::uint8_t i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        ctx->gmslMode[i] = CDI_MAX96712_GMSL_MODE_UNUSED;
        ctx->longCables[i] = m_longCables[i];
    }
}

#if !NV_IS_SAFETY || SAFETY_DBG_OV
void CNvMMax96712_Fusa::SetMAX96712CfgPipelineCpy(ContextMAX96712 *ctx) noexcept
{
    if (m_sDeserializerName == "MAX96712_Fusa_nv_camRecCfg_V1" ||
        m_sDeserializerName == "MAX96722_Fusa_nv_camRecCfg_V1") {
        ctx->cfgPipeCopy = CDI_MAX96712_CFG_PIPE_COPY_MODE_1;
    } else if (m_sDeserializerName == "MAX96712_Fusa_nv_camRecCfg_V2" ||
        m_sDeserializerName == "MAX96722_Fusa_nv_camRecCfg_V2") {
        ctx->cfgPipeCopy = CDI_MAX96712_CFG_PIPE_COPY_MODE_2;
    }
}
#endif // !NV_IS_SAFETY || SAFETY_DBG_OV

SIPLStatus CNvMMax96712_Fusa::SetMAX96712Ctx()
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    ContextMAX96712 *ctx = NULL;
    /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
    DriverContextImpl<ContextMAX96712> *const driverContext = new DriverContextImpl<ContextMAX96712>();

    m_upDrvContext.reset(driverContext);
    ctx = &driverContext->m_Context;

    SetCtxInitValue_CNvMMax96712Fusa(ctx);

#if !NV_IS_SAFETY
    if (m_oDeviceParams.bEnableSimulator) {
        m_I2CPort = 0U;
        m_eInterface = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A;
        m_ePhyMode = NVSIPL_CAP_CSI_DPHY_MODE;
        m_pwrPort = 0U;
    }
#endif

#if !NV_IS_SAFETY || SAFETY_DBG_OV
    SetMAX96712CfgPipelineCpy(ctx);
#endif // !NV_IS_SAFETY || SAFETY_DBG_OV

    status = UpdateGmslMode_CNvMMax96712Fusa(ctx);
    if (status == NVSIPL_STATUS_OK) {
       status = UpdateCtxConfiguration_CNvMMax96712Fusa(ctx);
    }

    if (status == NVSIPL_STATUS_OK) {
        status = UpdateTxPort_CNvMMax96712Fusa(ctx);
    }

    if (status == NVSIPL_STATUS_OK) {
        status = UpdateMipiSpeed_CNvMMax96712Fusa(ctx);
    }

    if (status == NVSIPL_STATUS_OK) {
        /* Set link mask to be enabled */
        ctx->linkMask = m_linkMask;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
        /* Set slave mode */
        ctx->passiveEnabled = m_oDeviceParams.bPassive;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

        /* Set phy mode */
        ctx->phyMode = (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) ? CDI_MAX96712_PHY_MODE_CPHY : CDI_MAX96712_PHY_MODE_DPHY;

        ctx->defaultResetAll = m_resetAll;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::DoInit()
{
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    INvSIPLDeviceBlockTrace *instance = INvSIPLDeviceBlockTrace::GetInstance();
    if (instance != nullptr) {
        INvSIPLDeviceBlockTrace::TraceLevel level = instance-> GetLevel();
        C_LogLevel c_level = ConvertLogLevel(level);
        SetCLogLevel(c_level);
    }

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        status = NVSIPL_STATUS_OK;
        exitFlag = true;
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    if (!exitFlag) {
        /*! Check deserializer is present */
        LOG_INFO("Check deserializer is present\n");
        nvmStatus = MAX96712CheckPresence(m_upCDIDevice.get());
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712CheckPresence failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }
    if (!exitFlag && !m_oDeviceParams.bCBAEnabled) {
        LOG_INFO("Set deserializer defaults\n");
        nvmStatus = MAX96712SetDefaults(m_upCDIDevice.get());
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712SetDefaults failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag) {
        /* Get deserializer revision */
        LOG_INFO("Get deserializer revision\n");
        nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                       CDI_READ_PARAM_CMD_MAX96712_REV_ID,
                                       sizeof(int32_t),
                                       &readParamsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_REV_ID) failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag && !m_oDeviceParams.bCBAEnabled) {
        /* Set MIPI output mode */
        m_eRevision = readParamsMAX96712.revision;
        WriteParametersParamMAX96712 writeParamsMAX96712 = {};
        LOG_INFO("Set MIPI output mode\n");
        writeParamsMAX96712.MipiSettings.mipiSpeed = static_cast<uint8_t>(m_uMipiSpeed / 100000U);

        writeParamsMAX96712.MipiSettings.phyMode = (m_ePhyMode == NVSIPL_CAP_CSI_CPHY_MODE) ? CDI_MAX96712_PHY_MODE_CPHY : CDI_MAX96712_PHY_MODE_DPHY;
        nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                        CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI,
                                        sizeof(writeParamsMAX96712.MipiSettings),
                                        &writeParamsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI) failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag && !m_oDeviceParams.bCBAEnabled) {
        /* Disable CSI out */
        nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_DISABLE_CSI_OUT);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CDI_CONFIG_MAX96712_DISABLE_CSI_OUT failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }
    return status;
}

SIPLStatus CNvMMax96712_Fusa::EnableLinks(uint8_t const linkMask)
{
    ReadParametersParamMAX96712 readParamsMAX96712 = {};
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive or m_oDeviceParams.bCBAEnabled) {
        status = NVSIPL_STATUS_OK;
        exitFlag = true;
    }

    NvMediaStatus nvmStatus;

    if (!exitFlag) {
        /* Get the links currently enabled */
        nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                           CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
                                           sizeof(int32_t),
                                           &readParamsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS) failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag) {

        if (readParamsMAX96712.link != linkMask) {
            LOG_INFO("Enabling links\n");
            WriteParametersParamMAX96712 writeParamsMAX96712 = {};
            writeParamsMAX96712.link = (static_cast<LinkMAX96712>(linkMask));
            nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
                                                sizeof(int32_t),
                                                &writeParamsMAX96712);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK) failed with NvMedia error",
                                     static_cast<int32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
            } else {
                std::this_thread::sleep_for<>(std::chrono::milliseconds(15));
            }
        }
    }
    return status;
}

SIPLStatus CNvMMax96712_Fusa::ControlLinks(std::vector<LinkAction> const& linkActions)
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive or m_oDeviceParams.bCBAEnabled) {
        status = NVSIPL_STATUS_OK;
        exitFlag = true ;
    }

    if (!exitFlag) {

        ReadParametersParamMAX96712 readParamsMAX96712 = {};
        NvMediaStatus nvmStatus;

        /* Get the links currently enabled */
        nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                           CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
                                           sizeof(int32_t),
                                           &readParamsMAX96712);

        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712ReadParameters(CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS) failed with NvMedia error",
                                  static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        } else {
            uint8_t link {readParamsMAX96712.link};
            bool link_enabled {false};
            uint8_t linkMask {0U};
            uint8_t linkIdx {0U};
            uint8_t linkBit {0U};
            uint8_t powerReset {0U};

            bool prevLink = link;

            for (std::uint32_t i = 0U; i < linkActions.size(); i++) {
                WriteParametersParamMAX96712 writeParamsMAX96712 = {};
                LinkAction const& item = linkActions[i];
                if ((item.linkIdx & 0x80U) == 0x80U) {
                    /* Bit 7 is the flag of power cycling.
                     * Without power reset, remote ERRB needs to be re-enabled after link locked
                     */
                    powerReset = 1U;
                }
                linkIdx = item.linkIdx & 0x7FU;
                if (linkIdx >= MAX96712_MAX_NUM_LINK) {
                    status = NVSIPL_STATUS_ERROR;
                    break;
                } else {
                    linkBit = (static_cast<uint8_t>(1U) << linkIdx);
                }

                switch (item.eAction) {
                case LinkAction::Action::LINK_ENABLE:
                    link |= linkBit;
                    linkMask |= linkBit;
                    link_enabled = true;
                    break;
                case LinkAction::Action::LINK_DISABLE:
                    link &= (static_cast<uint8_t>(~linkBit) & 0x0FU);
                    writeParamsMAX96712.linkIndex = linkIdx;
                    nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                        CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK,
                                                        sizeof(uint8_t),
                                                        &writeParamsMAX96712);
                    if (nvmStatus != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK) failed with NvMedia error",
                                             static_cast<int32_t>(nvmStatus));
                        status = ConvertNvMediaStatus(nvmStatus);
                    }
                    break;
                case LinkAction::Action::LINK_NO_ACTION:
                    break;
                default:
                    SIPL_LOG_ERR_STR("Invalid Link Action");
                    status = NVSIPL_STATUS_BAD_ARGUMENT;
                    break;
                }
            }

            if (linkActions.size() > 0U) {
                if (status == NVSIPL_STATUS_OK) {
                    WriteParametersParamMAX96712 writeParamsMAX96712 = {};
                    writeParamsMAX96712.link = static_cast<LinkMAX96712>(link);
                    WriteParametersCmdMAX96712 param = CDI_WRITE_PARAM_CMD_MAX96712_INVALID;
                    if (link_enabled) {
                        param = CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS;
                    } else {
                        /* No need to check link lock when links are being disabled */
                        param = CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK;
                    }
                    nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                        param,
                                                        sizeof(writeParamsMAX96712.link),
                                                        &writeParamsMAX96712);
                    if (nvmStatus != NVMEDIA_STATUS_OK) {

                        /* Enable new links failed. Need to restore to original links */
                        writeParamsMAX96712.link = static_cast<LinkMAX96712>(prevLink);
                        NvMediaStatus temp = MAX96712WriteParameters(m_upCDIDevice.get(),
                            CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK,
                            sizeof(writeParamsMAX96712.link),
                            &writeParamsMAX96712);
                        if (temp != NVMEDIA_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK) failed to restore link settings with NvMedia error",
                            static_cast<int32_t>(temp));
                        }
                        SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK) failed with NvMedia error",
                                             static_cast<int32_t>(nvmStatus));

                        /* Failed to enable the specific links */
                        status = ConvertNvMediaStatus(nvmStatus);
                    }
                }

                if ((status == NVSIPL_STATUS_OK) && link_enabled) {
                    status = CheckLinkLock(linkMask);
                    if ((status == NVSIPL_STATUS_OK) && (powerReset == 0U)) {
                        /* Restore ERRB Rx Eanble and Video Pipeline Enable bits */
                        for (std::uint32_t i = 0U; i < linkActions.size(); i++) {
                            const LinkAction& item = linkActions[i];
                            if (item.eAction == LinkAction::Action::LINK_ENABLE) {
                                WriteParametersParamMAX96712 writeParamsMAX96712 = {};
                                writeParamsMAX96712.linkIndex = (item.linkIdx & 0x7FU);
                                nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                                            CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK,
                                                            sizeof(uint8_t),
                                                            &writeParamsMAX96712);
                                if (nvmStatus != NVMEDIA_STATUS_OK) {
                                    SIPL_LOG_ERR_STR_INT("MAX96712WriteParameters(CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK) failed with NvMedia error",
                                                         static_cast<int32_t>(nvmStatus));
                                    status = ConvertNvMediaStatus(nvmStatus);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            /* NOP: just to make sure misra_cpp_2008_rule_0_1_6_violation is suppressed */
            (void) link;
            (void) link_enabled;
            (void) linkMask;
            (void) powerReset;
            (void) linkIdx;
            (void) linkBit;

        }
    }
    return status;
}

SIPLStatus CNvMMax96712_Fusa::CheckLinkLock(uint8_t const linkMask)
{

    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};

    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive or m_oDeviceParams.bCBAEnabled) {
        status = NVSIPL_STATUS_OK;
        exitFlag = true;
    }

    if (!exitFlag) {
        (void)exitFlag;
        for(uint8_t linkIndex {0U}; linkIndex < m_ovLinkModes.size(); linkIndex++) {
            DeserLinkModes const& item = m_ovLinkModes[linkIndex];
            if (item.linkIndex < MAX96712_MAX_NUM_LINK) {
                uint8_t const tmpMask[] = {1U, 2U, 4U, 8U};
                if ((linkMask & tmpMask[item.linkIndex])== 0U ) {
                    continue;
                }

                if ((item.elinkMode == LinkMode::LINK_MODE_GMSL2_6GBPS) ||
                    (item.elinkMode == LinkMode::LINK_MODE_GMSL2_3GBPS)) {
                    NvMediaStatus nvmStatus;
                    LOG_INFO("Check config link lock\n");
                    nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                                    GetMAX96712Link(item.linkIndex),
                                                    CDI_MAX96712_LINK_LOCK_GMSL2,
                                                    false);
                    if (nvmStatus == NVMEDIA_STATUS_TIMED_OUT) {
                        nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), GetMAX96712Link(item.linkIndex));
                        if (nvmStatus != NVMEDIA_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("MAX96712OneShotReset failed with NvMedia error",
                                                 static_cast<int32_t>(nvmStatus));
                            status = ConvertNvMediaStatus(nvmStatus);
                            exitFlag = true;
                        } else {
                            nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                                      GetMAX96712Link(item.linkIndex),
                                                      CDI_MAX96712_LINK_LOCK_GMSL2,
                                                      true);
                            if (nvmStatus != NVMEDIA_STATUS_OK) {
                               SIPL_LOG_ERR_STR_INT("MAX96712CheckLink(CDI_MAX96712_GMSL2_LINK_LOCK) failed with NvMedia error",
                                                 static_cast<int32_t>(nvmStatus));
                               status = ConvertNvMediaStatus(nvmStatus);
                               exitFlag = true;
                            }
                        }
                    } else {
                        status = ConvertNvMediaStatus(nvmStatus);
                    }
                }
            }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            else if (item.elinkMode == LinkMode::LINK_MODE_GMSL1) {
                NvMediaStatus nvmStatus;
                LOG_INFO("Check config link lock \n");
                nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                              GetMAX96712Link(item.linkIndex),
                                              CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                              false);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), GetMAX96712Link(item.linkIndex));
                    if (nvmStatus != NVMEDIA_STATUS_OK) {
                        SIPL_LOG_ERR_STR_INT("MAX96712OneShotReset failed with NvMedia error",
                                             static_cast<int32_t>(nvmStatus));
                        status = ConvertNvMediaStatus(nvmStatus);
                        exitFlag = true;
                    } else {
                        nvmStatus = MAX96712CheckLink(m_upCDIDevice.get(),
                                                      GetMAX96712Link(item.linkIndex),
                                                      CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
                                                      true);
                        if (nvmStatus != NVMEDIA_STATUS_OK) {
                            SIPL_LOG_ERR_STR_INT("MAX96712CheckLink(CDI_MAX96712_GMSL1_CONFIG_LINK_LOCK) failed with NvMedia error",
                                                 static_cast<int32_t>(nvmStatus));
                            status = ConvertNvMediaStatus(nvmStatus);
                            exitFlag = true;
                        }
                    }
                }
            }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
            else {
                status = NVSIPL_STATUS_NOT_SUPPORTED;
                exitFlag = true;
            }
            if (exitFlag) {
                break;
            }
        }
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::DoStart()
{
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};
    NvMediaStatus nvmStatus;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    if (m_oDeviceParams.bEnableSimulator or m_oDeviceParams.bPassive) {
        status = NVSIPL_STATUS_OK;
        exitFlag = true;
    }

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    if (!exitFlag) {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif
        /* Check CSIPLL lock */
        nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
        }
        /* Trigger the initial deskew */
        if ((m_ePhyMode == NVSIPL_CAP_CSI_DPHY_MODE) and (m_uMipiSpeed >= 1500000U)) {
            nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_TRIGGER_DESKEW);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("CDI_CONFIG_MAX96712_TRIGGER_DESKEW failed with NvMedia error",
                                     static_cast<int32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
                exitFlag = true;
            }
        }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    }
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    if (!exitFlag) {
        status = CheckLinkLock(m_linkMask);
        if (status != NVSIPL_STATUS_OK) {
            nvmStatus = MAX96712OneShotReset(m_upCDIDevice.get(), (static_cast<LinkMAX96712>(m_linkMask)));
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("MAX96712OneShotReset failed with NvMedia error",
                                     static_cast<int32_t>(nvmStatus));
                status = ConvertNvMediaStatus(nvmStatus);
                exitFlag = true;
            } else {
                status = CheckLinkLock(m_linkMask);
                if (status != NVSIPL_STATUS_OK) {
                    exitFlag = true;
                }
            }
        }
    }

    if (!exitFlag) {
        /* Enable CSI out */
        nvmStatus = MAX96712SetDeviceConfig(m_upCDIDevice.get(), CDI_CONFIG_MAX96712_ENABLE_CSI_OUT);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("CDI_CONFIG_MAX96712_ENABLE_CSI_OUT failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag) {
        /* Check & Clear if ERRB set */
        ReadParametersParamMAX96712 readParamsMAX96712 = {};
        nvmStatus = MAX96712ReadParameters(m_upCDIDevice.get(),
                                           CDI_READ_PARAM_CMD_MAX96712_ERRB,
                                           sizeof(readParamsMAX96712.ErrorStatus),
                                           &readParamsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("DevBlkCDIReadParameters(CDI_READ_PARAM_CMD_MAX96712_ERRB) failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    return status;
}

/**
 * @brief Get a newly allocated deserializer object, as implemented by the
 * driver library.
 *
 * be freed using `delete`.
 *
 * @retval pointer to new CNvMMax96712_Fusa() .
 * @retval nullptr On bad allocation or other failure.
 */
/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
CNvMDeserializer *CNvMDeserializer_Create()
{
    /* coverity[misra_cpp_2008_rule_18_4_1_violation] : intentional TID-1968 */
    return new CNvMMax96712_Fusa();
}

/**
 * @brief Gets a null terminated C-style string containing the name of the
 * deserializer.
 *
 * The returned string is expected to be valid at any time, so it should
 * be a string constant and not dynamically allocated.
 *
 * @retval (char_t*)      A null-terminated C-style string containing the name of
 *                        deserializer device.
 * @retval nullptr        On bad allocation or other failure.
 */
/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
const char_t** CNvMDeserializer_GetName()
{
    static char const *names_fusa[] {
        "MAX96712_Fusa_nv",
        "MAX96722_Fusa_nv",
#if !NV_IS_SAFETY || SAFETY_DBG_OV
        "MAX96712_Fusa_nv_camRecCfg_V1",
        "MAX96712_Fusa_nv_camRecCfg_V2",
        "MAX96722_Fusa_nv_camRecCfg_V1",
        "MAX96722_Fusa_nv_camRecCfg_V2",
#endif
        NULL
    };

    return names_fusa;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::GetErrorSize(size_t & errorSize)
{
    errorSize = MAX_DESER_ERROR_SIZE;
    return NVSIPL_STATUS_OK;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::GetErrorInfo(std::uint8_t * const buffer,
                                      std::size_t const bufferSize,
                                      std::size_t &size,
                                      bool & isRemoteError,
                                      std::uint8_t& linkErrorMask)
{
    ErrorStatusMAX96712 errorStatus;
    NvMediaStatus nvmStatus;
    SIPLStatus status {NVSIPL_STATUS_OK};
    bool exitFlag {false};

    std::this_thread::sleep_for<>(std::chrono::milliseconds(30));

    nvmStatus = MAX96712GetErrorStatus(m_upCDIDevice.get(),
                                       static_cast<uint32_t>(sizeof(errorStatus)),
                                       &errorStatus);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712GetErrorStatus failed with NvMedia error",
                             static_cast<int32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
        exitFlag = true;
    }

    if (!exitFlag) {
        // Get link error mask
        linkErrorMask = errorStatus.link;

        // Check whether serializer error is detected as well
        nvmStatus = MAX96712GetSerializerErrorStatus(m_upCDIDevice.get(),
                                                 &isRemoteError);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712GetSerializerErrorStatus failed with NvMedia error",
                                 static_cast<int32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
            exitFlag = true;
        }
    }

    if (!exitFlag) {
        size = 0U;
        if ((buffer != nullptr) && (errorStatus.count != 0U)) {
            // Get other error details in buffer.
            // Doing this for all regVals and failureTypes
            std::size_t sizeMax96712ErrorStatusGlobal = 0U;
            std::size_t sizeMax96712ErrorStatusPipeline = 0U;
            std::size_t sizeMax96712ErrorStatusLink = 0U;
            std::size_t sizeMax96712ErrorStatus = 0U;

            sizeMax96712ErrorStatusGlobal   = (sizeof(errorStatus.globalRegVal))   + (sizeof(errorStatus.globalFailureType));
            sizeMax96712ErrorStatusPipeline = (sizeof(errorStatus.pipelineRegVal)) + (sizeof(errorStatus.pipelineFailureType));
            sizeMax96712ErrorStatusLink     = (sizeof(errorStatus.linkRegVal))     + (sizeof(errorStatus.linkFailureType));
            sizeMax96712ErrorStatus = sizeMax96712ErrorStatusGlobal + sizeMax96712ErrorStatusPipeline + sizeMax96712ErrorStatusLink;

            if (sizeMax96712ErrorStatus != MAX_DESER_ERROR_SIZE) {
                SIPL_LOG_ERR_STR("MAX96712GetSerializerErrorStatus something is wrong in finding max deser error size");
                status = NVSIPL_STATUS_ERROR;
            } else {
                if (bufferSize >= sizeMax96712ErrorStatus) {
                    std::uint8_t tmpBuf[MAX_DESER_ERROR_SIZE]={0U};
                    std::size_t sizeIncr = 0U;
                    fusa_memcpy(&tmpBuf[sizeIncr], &errorStatus.globalFailureType, sizeof(errorStatus.globalFailureType));
                    sizeIncr += sizeof(errorStatus.globalFailureType);
                    fusa_memcpy(&tmpBuf[sizeIncr], &errorStatus.globalRegVal, sizeof(errorStatus.globalRegVal));
                    sizeIncr += sizeof(errorStatus.globalRegVal);
                    fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.pipelineFailureType), sizeof(errorStatus.pipelineFailureType));
                    sizeIncr += sizeof(errorStatus.pipelineFailureType);
                    fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.pipelineRegVal), sizeof(errorStatus.pipelineRegVal));
                    sizeIncr += sizeof(errorStatus.pipelineRegVal);
                    fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.linkFailureType), sizeof(errorStatus.linkFailureType));
                    sizeIncr += sizeof(errorStatus.linkFailureType);
                    fusa_memcpy(&tmpBuf[sizeIncr], &(errorStatus.linkRegVal), sizeof(errorStatus.linkRegVal));
                    size = sizeMax96712ErrorStatus;
                    fusa_memcpy(buffer, static_cast<void *>(&tmpBuf[0]), sizeof(tmpBuf));
                } else {
                status = NVSIPL_STATUS_BAD_ARGUMENT;
                }
            }
        }
    }
    return status;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
SIPLStatus CNvMMax96712_Fusa::DoSetPower(bool const powerOn)
{
    DevBlkCDIPowerControlInfo m_pwrControlInfo;
    bool exitFlag {false};
    SIPLStatus status {NVSIPL_STATUS_OK};
    NvMediaStatus nvmStatus = DevBlkCDIGetDesPowerControlInfo(m_upCDIDevice.get(),
                                                              &m_pwrControlInfo);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("Deser power control method with NVMEDIA error",
                              static_cast<int32_t>(nvmStatus));
        status = ConvertNvMediaStatus(nvmStatus);
        exitFlag = true;
    }

    if (!exitFlag) {
        if (m_pwrControlInfo.method == 0U) {
            // Default is NvCCP, other power backends can be used here based on platform/usecase.
            status = PowerControlSetAggregatorPower(m_pwrPort, m_oDeviceParams.bPassive, powerOn);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("CNvMMax96712_Fusa::DoSetPower failed with SIPL error",
                                 static_cast<int32_t>(status));
            }
        } else {
            //For Drive Orin and firespray
            status = (SIPLStatus) DevBlkCDISetDeserPower(m_upCDIDevice.get(),powerOn);
            if (status != NVSIPL_STATUS_OK) {
                SIPL_LOG_ERR_STR_INT("CNvMMax96712_Fusa::DoSetPower failed with SIPL error",
                                 static_cast<int32_t>(status));
            }
        }
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::GetOverflowError(
    DeserializerOverflowErrInfo *const customErrInfo
) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(customErrInfo != nullptr) {
        NvMediaStatus nvmstatus {NVMEDIA_STATUS_OK};
        uint8_t reg_backtop11_lower {0U};
        nvmstatus = MAX96712ReadRegisterVerify(m_upCDIDevice.get(), MAX96712_REG_BACKTOP11_LOWER,
                                         &reg_backtop11_lower);
        if(nvmstatus == NVMEDIA_STATUS_OK) {
            customErrInfo->cmd_buffer_overflow_info =
                            static_cast<uint8_t>((reg_backtop11_lower & 0xF0U) >> SHIFT_NIBBLE);
            customErrInfo->lm_overflow_info = static_cast<uint8_t>(reg_backtop11_lower & 0x0FU);

            uint8_t reg_backtop11_higher {0U};
            nvmstatus = MAX96712ReadRegisterVerify(m_upCDIDevice.get(),
                                                   MAX96712_REG_BACKTOP11_HIGHER,
                                                   &reg_backtop11_higher);
            if(nvmstatus == NVMEDIA_STATUS_OK) {
                customErrInfo->cmd_buffer_overflow_info |=
                            static_cast<uint8_t>(reg_backtop11_higher & 0xF0U);
                customErrInfo->lm_overflow_info |=
                            static_cast<uint8_t>((reg_backtop11_higher & 0x0FU) << SHIFT_NIBBLE);
            } else {
                SIPL_LOG_ERR_STR("MAX96712ReadRegisterVerify failed for BACKTOP11(higher)"
                                 " register.");
            }
        } else {
            SIPL_LOG_ERR_STR("MAX96712ReadRegisterVerify failed for BACKTOP11(lower) register.");
        }
        status = ConvertNvMediaStatus(nvmstatus);
    } else {
        SIPL_LOG_ERR_STR("MAX96712: GetDeserOverflowError: Bad argument.");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::GetCSIPLLLockStatus(
    DeserializerCSIPLLLockInfo *const customErrInfo
) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(customErrInfo != nullptr) {
        NvMediaStatus nvmstatus {NVMEDIA_STATUS_OK};
        uint8_t reg_backtop1 {0U};
        nvmstatus = MAX96712ReadRegisterVerify(m_upCDIDevice.get(), MAX96712_REG_BACKTOP1,
                                         &reg_backtop1);
        if(nvmstatus == NVMEDIA_STATUS_OK) {
            customErrInfo->csi_pll_lock_status =
                            static_cast<uint8_t>((reg_backtop1 & 0xF0U) >> SHIFT_NIBBLE);
        } else {
            SIPL_LOG_ERR_STR("MAX96712ReadRegisterVerify failed for BACKTOP1 register.");
        }
        status = ConvertNvMediaStatus(nvmstatus);
    } else {
        SIPL_LOG_ERR_STR("MAX96712: GetCSIPLLLockStatus: Bad argument.");
        status =  NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::GetRTFlagsStatus(
    DeserializerRTFlagsInfo *const customErrInfo
) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(customErrInfo != nullptr) {
        NvMediaStatus nvmstatus {NVMEDIA_STATUS_OK};
        uint8_t reg_intr11 {0U};
        nvmstatus = MAX96712ReadRegisterVerify(m_upCDIDevice.get(), MAX96712_REG_INTR11,
                                         &reg_intr11);
        if(nvmstatus == NVMEDIA_STATUS_OK) {
            customErrInfo->rt_cnt_flag_info =
                            static_cast<uint8_t>((reg_intr11 & 0xF0U)>> SHIFT_NIBBLE);
            customErrInfo->max_rt_flag_info = static_cast<uint8_t>(reg_intr11 & 0x0F);
        } else {
            SIPL_LOG_ERR_STR("MAX96712ReadRegisterVerify failed for INTR11 register.");
        }
        status = ConvertNvMediaStatus(nvmstatus);
    } else {
        SIPL_LOG_ERR_STR("MAX96712: GetRTFlagsStatus: Bad argument.");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return status;
}

/**
 * @brief Get Deserializer VID_SEQ_ERR bits status.
 *
 */
SIPLStatus CNvMMax96712_Fusa::GetVidSeqErrStatus(
    DeserializerVidSeqErrInfo *const customErrInfo
) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    if(customErrInfo != nullptr) {
        NvMediaStatus nvmstatus {NVMEDIA_STATUS_OK};
        nvmstatus = MAX96712GetVidSeqError(m_upCDIDevice.get(),
                                        &(customErrInfo->vid_seq_err_info));
        status = ConvertNvMediaStatus(nvmstatus);
    } else {
        SIPL_LOG_ERR_STR("MAX96712: GetVidSeqErrStatus: Bad argument");
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return status;
}

/* coverity[misra_cpp_2008_rule_0_1_10_violation] : intentional TID-1966 */
Interface* CNvMMax96712_Fusa::GetInterface(UUID const &interfaceId)
{
    if (interfaceId == (CNvMMax96712_Fusa::getClassInterfaceID())) {
        return static_cast<MAX96712FusaNvCustomInterface*>(this);
    } else {
        return nullptr;
    }
}

SIPLStatus CNvMMax96712_Fusa::GetInterruptStatus(
    uint32_t const gpioIdx,
    IInterruptNotify &intrNotifier)
{
    SIPLStatus ret = NVSIPL_STATUS_ERROR;

    if (gpioIdx == SOC_GPIO_IDX_0) {
        /* check if this is link lock error */
        bool bLocked {false};
        bool bSet {false};
        uint8_t i = 0U;

        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (m_linkMask & (1U << i)) {
                NvMediaStatus nvmStatus = IsLinkLock(m_upCDIDevice.get(),
                                                     i, &bLocked);
                if (nvmStatus != NVMEDIA_STATUS_OK) {
                    SIPL_LOG_ERR_STR_INT("Deser link lock status read failed",
                            static_cast<int32_t>(nvmStatus));
                    ret = ConvertNvMediaStatus(nvmStatus);
                } else {
                    /* For case where lock bit is not set, bLocked will be false.
                     * Need to notify application of link lock error found for
                     * link 'i'
                     */
                    if (!bLocked) {
                        intrNotifier.Notify(
                            InterruptCode::INTR_STATUS_DES_LOCK_ERR, 0U,
                            gpioIdx, i);
                        ret = NVSIPL_STATUS_OK;
                    }
                }
             }
        }
        /* check if this local deserializer error */
        if((IsErrbSet(m_upCDIDevice.get(), &bSet) == NVMEDIA_STATUS_OK)
            && bSet) {
            intrNotifier.Notify(InterruptCode::INTR_STATUS_DES_ERRB_ERR,
                0U, gpioIdx, 0U);
            ret = NVSIPL_STATUS_OK;
        }
    } else {
        /* no-op */
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    return ret;
}

SIPLStatus CNvMMax96712_Fusa::GetGroupID(uint32_t &grpId) const
{
    SIPLStatus status {NVSIPL_STATUS_OK};

    switch (m_eInterface) {
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB:
            grpId = 0U;
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD:
            grpId = 1U;
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF:
            grpId = 2U;
            break;
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H:
        case NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH:
            grpId = 3U;
            break;
        default:
            SIPL_LOG_ERR_STR("The wrong CSI interface type was used");
            grpId = UINT32_MAX;
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            break;
    }

    return status;
}

SIPLStatus CNvMMax96712_Fusa::SetHeteroFrameSync(uint8_t const muxSel, uint32_t const gpioNum[MAX_CAMERAMODULES_PER_BLOCK]) const
{
    static_cast<void>(muxSel);
    SIPLStatus status {NVSIPL_STATUS_OK};
    NvMediaStatus nvmStatus {NVMEDIA_STATUS_OK};
    uint32_t camGrpIdx {0U};
    bool exitFlag {false};

    if (gpioNum == nullptr) {
        exitFlag = true;
        status = NVSIPL_STATUS_BAD_ARGUMENT;
    }

    if (exitFlag == false) {
        /* Update the GPIO number to receive the frame sync signal per each link */
        WriteParametersParamMAX96712 paramsMAX96712 = {};
        memcpy(paramsMAX96712.HeteroFSyncSettings.gpioNum,
               gpioNum,
               MAX_CAMERAMODULES_PER_BLOCK * sizeof(uint32_t));

        LOG_INFO("Set Heterogeneous frame sync mode\n");
        nvmStatus = MAX96712WriteParameters(m_upCDIDevice.get(),
                                            CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC,
                                            sizeof(paramsMAX96712.FSyncSettings),
                                            &paramsMAX96712);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_HEX_UINT("CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC failed with nvmediaErr",
                                       static_cast<uint32_t>(nvmStatus));
            exitFlag = true;
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }

    /* Update the MUX selection signals */
    if (exitFlag == false) {
        status = GetGroupID(camGrpIdx);
        if (status != NVSIPL_STATUS_OK) {
            exitFlag = true;
            status = NVSIPL_STATUS_BAD_ARGUMENT;
        }
    }

#if !NV_IS_SAFETY
    if (exitFlag == false) {
        nvmStatus = DevBlkCDISetFsyncMux(m_upCDIDevice.get(), muxSel, camGrpIdx);
        if (nvmStatus != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("Failed to set the fsync muxtiplexer with NVMEDIA error",
                                       static_cast<uint32_t>(nvmStatus));
            status = ConvertNvMediaStatus(nvmStatus);
        }
    }
#endif

    return status;
}

} // end of nvsipl
