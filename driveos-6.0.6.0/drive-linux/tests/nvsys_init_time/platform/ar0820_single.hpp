/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef AR0820_S_HPP
#define AR0820_S_HPP

#include "common.hpp"

static PlatformCfg platformCfgAr0820_s = {
    .platform = "F008A120RM0AV2_CPHY_x4",
    .platformConfig = "F008A120RM0AV2_CPHY_x4_s",
    .description = "F008A120RM0AV2 24BITS RGGB module in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = 2,
            .deserInfo = {
                .name = "MAX96712",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x29,
                .errGpios = {},
#if !NV_IS_SAFETY
                .camRecCfg = CAMREC_NONE,
#endif // !NV_IS_SAFETY
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "F008A120RM0AV2",
#if !NV_IS_SAFETY
                    .description = "Entron F008A120RM0AV2 module - 120-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x62,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX,
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "N24C64",
#if !NV_IS_SAFETY
                        .description = "N24C64 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x54,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 0U,
                            .name = "AR0820",
#if !NV_IS_SAFETY
                            .description = "OnSemi AR0820 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x10,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_GRBG,
                                    .embeddedTopLines = 2U,
                                    .embeddedBottomLines = 4U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12,
                                    .resolution = {
                                        .width = 3848U,
                                        .height = 2168U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    }
                }
            },
            .desI2CPort = DESER_TO_SOC_I2C_PORT_NUMBER,
            .desTxPort = DESER_TO_SOC_TX_PORT_NUMBER,
            .pwrPort = 2U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 2000000U},
#if !NV_IS_SAFETY
            .isSimulatorModeEnabled = false,
            .isPassiveModeEnabled = false,
#endif // !NV_IS_SAFETY
            .isGroupInitProg = true,
            .gpios = {},
#if !NV_IS_SAFETY
            .isPwrCtrlDisabled = false,
            .longCables = {false, false, false, false},
#endif // !NV_IS_SAFETY
            .resetAll = false
        }
    }
};

#endif // AR0820_S_HPP