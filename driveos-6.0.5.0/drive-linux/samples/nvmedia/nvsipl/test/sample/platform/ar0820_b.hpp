/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef AR0820_B_HPP
#define AR0820_B_HPP

#include "common.hpp"

static PlatformCfg platformCfgAr0820B = {
    .platform = "AR0820C120FOV_24BIT_RGGB_CPHY_x2",
    .platformConfig = "AR0820C120FOV_24BIT_RGGB_CPHY_x2",
    .description = "AR0820C120FOV_24BIT RGGB module in 2 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER,
            .deserInfo = {
                .name = "MAX96712_Fusa",
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
                    .name = "AR0820C120FOV_24BIT_RGGB",
#if !NV_IS_SAFETY
                    .description = "AR0820C120FOV 24BIT RGGB module - 120-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
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
                        .name = "M24C32",
#if !NV_IS_SAFETY
                        .description = "M24C32 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x50,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = SIPL_PIPELINE_ID,
                            .name = "AR0820",
#if !NV_IS_SAFETY
                            .description = "OnSemi AR0820 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x10,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_GRBG,
                                    .embeddedTopLines = 6U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12,
                                    .resolution = {
                                        .width = 3848U,
                                        .height = 2168U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
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
            .pwrPort = 0U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 1700000U}
        }
    }
};

#endif // AR0820_B_HPP
