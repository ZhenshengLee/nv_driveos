/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef IMX728VB2_HPP
#define IMX728VB2_HPP

#include "common.hpp"

static PlatformCfg platformCfgIMX728VB2 = {
    .platform = "V1SIM728S1RU3120NB20_CPHY_x4",
    .platformConfig = "V1SIM728S1RU3120NB20_CPHY_x4",
    .description = "IMX728+Max96717 module in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER,
            .deserInfo = {
                .name = "MAX96712_Fusa_nv",
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
                    .name = "V1SIM728S1RU3120NB20",
#if !NV_IS_SAFETY
                    .description = "Valeo IMX728 RGGB module - 120-deg FOV, MIPI-IMX728, MAX96717",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C04",
#if !NV_IS_SAFETY
                        .description = "M24C04 EEPROM",
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
                            .name = "IMX728",
#if !NV_IS_SAFETY
                            .description = "SONY IMX728 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1C,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_RGGB,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 20U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ,
                                    .resolution = {
                                        .width = 3840U,
                                        .height = 2160U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
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
            .cphyRate = {2000000U, 2000000U}
        }
    }
};

static PlatformCfg platformCfgIMX728VB3_1C = {
    .platform = "V1SIM728S2RU3120HB30_CPHY_x4",
    .platformConfig = "V1SIM728S2RU3120HB30_CPHY_x4",
    .description = "IMX728+Max96717 module in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER,
            .deserInfo = {
                .name = "MAX96712_Fusa_nv",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x29,
                .errGpios = {},
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "V1SIM728S2RU3120HB30",
#if !NV_IS_SAFETY
                    .description = "Valeo IMX728 RGGB module - 120-deg FOV, MIPI-IMX728, MAX96717",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
    #ifdef NVMEDIA_QNX
                                .useCDIv2API = true
    #else // Linux
                                .useCDIv2API = false
    #endif //NVMEDIA_QNX
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C04",
#if !NV_IS_SAFETY
                        .description = "M24C04 EEPROM",
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
                            .name = "IMX728",
#if !NV_IS_SAFETY
                            .description = "SONY IMX728 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1C,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_RGGB,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 20U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ,
                                    .resolution = {
                                        .width = 3840U,
                                        .height = 2160U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
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
            .cphyRate = {2000000U, 2000000U}
        }
    }
};

static PlatformCfg platformCfgIMX728VB2_2 = {
    .platform = "V1SIM728S2RU4120NB20_CPHY_x4",
    .platformConfig = "V1SIM728S2RU4120NB20_CPHY_x4",
    .description = "IMX728+Max96717 module in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER,
            .deserInfo = {
                .name = "MAX96712_Fusa_nv",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x29,
                .errGpios = {},
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "V1SIM728S2RU4120NB20",
#if !NV_IS_SAFETY
                    .description = "Valeo IMX728 RGGB module - 120-deg FOV, MIPI-IMX728, MAX96717",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C04",
#if !NV_IS_SAFETY
                        .description = "M24C04 EEPROM",
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
                            .name = "IMX728",
#if !NV_IS_SAFETY
                            .description = "SONY IMX728 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1C,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_RGGB,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 20U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ,
                                    .resolution = {
                                        .width = 3840U,
                                        .height = 2160U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
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
            .cphyRate = {2000000U, 2000000U}
        }
    }
};

#endif // IMX728VB2_HPP
