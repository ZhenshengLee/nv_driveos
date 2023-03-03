/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef OV2311_C_HPP
#define OV2311_C_HPP

#include "common.hpp"

static PlatformCfg platformCfgOv2311C = {
    .platform = "OV2311_C_3461_CPHY_x2",
    .platformConfig = "OV2311_C_3461_CPHY_x2",
    .description = "OV2311+Max96717f module in 2 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A,
            .phyMode = NVSIPL_CAP_CSI_DPHY_MODE,
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
                    .name = "OV2311_C_3461",
#if !NV_IS_SAFETY
                    .description = "OV2311 sensor with Max96717f serializer",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x42
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C02",
#if !NV_IS_SAFETY
                        .description = "M24C02 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x50
                    },
                    .sensorInfo = {
                            .id = SIPL_PIPELINE_ID,
                            .name = "OV2311C",
#if !NV_IS_SAFETY
                            .description = "OV2311C",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x60,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_BGGR,
                                    .embeddedTopLines = 2U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8,
                                    .resolution = {
                                        .width = 1600U,
                                        .height = 1300U
                                    },
                                    .fps = 30.0,
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
            .cphyRate = {2000000U, 1700000U}
        }
    }
};

#endif // OV2311_C_HPP
