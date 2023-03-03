/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef OV2311_HPP
#define OV2311_HPP

#include "common.hpp"

static PlatformCfg platformCfgOv2311 = {
    .platform = "LI-OV2311-VCSEL-GMSL2-60H_DPHY_x4",
    .platformConfig = "LI-OV2311-VCSEL-GMSL2-60H_DPHY_x4",
    .description = "LI-OV2311-VCSEL-GMSL2-60H module in 4 lane DPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
            .phyMode = NVSIPL_CAP_CSI_DPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER,
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
                    .name = "LI-OV2311-VCSEL-GMSL2-60H",
#if !NV_IS_SAFETY
                    .description = "OV2311, MAX9295, 60FPS",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295A Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x62
                    },
                    .isEEPROMSupported = false,
                    .sensorInfo = {
                            .id = SIPL_PIPELINE_ID,
                            .name = "OV2311",
#if !NV_IS_SAFETY
                            .description = "OmniVision OV2311 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x60,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_BGGR,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10,
                                    .resolution = {
                                        .width = 1600U,
                                        .height = 1300U
                                    },
                                    .fps = 60.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true
                    }
                }
            },
            .desI2CPort = DESER_TO_SOC_I2C_PORT_NUMBER,
            .desTxPort = DESER_TO_SOC_TX_PORT_NUMBER,
            .pwrPort = 0U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 2000000U}
        }
    }
};

#endif // OV2311_HPP
