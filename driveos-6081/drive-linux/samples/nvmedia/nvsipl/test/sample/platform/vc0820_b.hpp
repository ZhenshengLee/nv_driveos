/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef VC0820_B_HPP
#define VC0820_B_HPP

#include "common.hpp"

static PlatformCfg platformCfgVC0820C120_A = {
    .platform = "VC0820C120R24_CPHY_x2",
    .platformConfig = "VC0820C120R24_CPHY_x2",
    .description = "AR0820C120FOV_24BIT RGGB module in 2 lane CSI A CPHY mode",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C120R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C120FOV 24BIT RGGB module - 120-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
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
                                    .embeddedTopLines = 2U,
                                    .embeddedBottomLines = 4U,
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

static PlatformCfg platformCfgVC0820C030_A = {
    .platform = "VC0820C030R24_CPHY_x2",
    .platformConfig = "VC0820C030R24_CPHY_x2",
    .description = "AR0820C030FOV_24BIT RGGB module in 2 lane CSI A CPHY mode",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C030R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C030FOV 24BIT RGGB module - 30-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
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
                                    .embeddedTopLines = 2U,
                                    .embeddedBottomLines = 4U,
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

static PlatformCfg platformCfgVC0820C070_A = {
    .platform = "VC0820C070R24_CPHY_x2",
    .platformConfig = "VC0820C070R24_CPHY_x2",
    .description = "AR0820C070FOV_24BIT RGGB module in 2 lane CSI A CPHY mode",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C070FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
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
                                    .embeddedTopLines = 2U,
                                    .embeddedBottomLines = 4U,
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

static PlatformCfg platformCfgVC0820C030_A_CUST1 = {
    .platform = "VC0820C030R24_CPHY_x1_A_CUST1",
    .platformConfig = "VC0820C030R24_CPHY_x1_A_CUST1",
    .description = "AR0820 30FOV 24BIT RGGB module in A-lane CPHY",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C030R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C030FOV 24BIT RGGB module - 30-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
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
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else
                            .useCDIv2API = false
#endif
                    }
                },
            },
            .desI2CPort = DESER_TO_SOC_I2C_PORT_NUMBER,
            .desTxPort = DESER_TO_SOC_TX_PORT_NUMBER,
            .pwrPort = 0U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 1700000U}
        }
    }
};

static PlatformCfg platformCfgVC0820C030_B_CUST1 = {
    .platform = "VC0820C030R24_CPHY_x1_B_CUST1",
    .platformConfig = "VC0820C030R24_CPHY_x1_B_CUST1",
    .description = "AR0820 30FOV 24BIT RGGB module in B-lane CPHY",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else
                .useCDIv2API = false
#endif
            },
            .numCameraModules = 1U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C030R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C030FOV 24BIT RGGB module - 30-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                            .id = 1U,
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

static PlatformCfg platformCfgVC0820C030_C_CUST1 = {
    .platform = "VC0820C030R24_CPHY_x1_C_CUST1",
    .platformConfig = "VC0820C030R24_CPHY_x1_C_CUST1",
    .description = "AR0820 30FOV 24BIT RGGB module in C-lane CPHY",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD,
            .deserInfo = {
                .name = "MAX96712_Fusa",
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
                    .name = "VC0820C030R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C030FOV 24BIT RGGB module - 30-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C32",
#if !NV_IS_SAFETY
                        .description = "M24C32 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x50,
                        .useCDIv2API = true
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
                },
            },
            .desI2CPort = DESER_TO_SOC_I2C_PORT_NUMBER,
            .desTxPort = DESER_TO_SOC_TX_PORT_NUMBER,
            .pwrPort = 0U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 1700000U}
        }
    }
};

static PlatformCfg platformCfgVC0820C030_D_CUST1 = {
    .platform = "VC0820C030R24_CPHY_x1_D_CUST1",
    .platformConfig = "VC0820C030R24_CPHY_x1_D_CUST1",
    .description = "AR0820 30FOV 24BIT RGGB module in D-lane CPHY",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD,
            .deserInfo = {
                .name = "MAX96712_Fusa",
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
                    .name = "VC0820C030R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C030FOV 24BIT RGGB module - 30-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                            .id = 1U,
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
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
                            .errGpios = {},
                            .useCDIv2API = true
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

static PlatformCfg platformCfgVC0820C070_AB_CUST1 = {
    .platform = "VC0820C070R24_CPHY_x2_AB_CUST1",
    .platformConfig = "VC0820C070R24_CPHY_x2_AB_CUST1",
    .description = "AR0820 70FOV 24BIT RGGB module in 2 lane CPHY mode",
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
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 2U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C070FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
                            .errGpios = {},
                            .useCDIv2API = true
                    }
                },
                {
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C070FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                            .id = 1U,
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
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
                            .errGpios = {},
                            .useCDIv2API = true
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

static PlatformCfg platformCfgVC0820C070_CD_CUST1 = {
    .platform = "VC0820C070R24_CPHY_x2_CD_CUST1",
    .platformConfig = "VC0820C070R24_CPHY_x2_CD_CUST1",
    .description = "AR0820 70FOV 24BIT RGGB module in 2 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD,
            .deserInfo = {
                .name = "MAX96712_Fusa",
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
            .numCameraModules = 2U,
            .cameraModuleInfoList = {
                {
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C070FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                },
		{
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820C070FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
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
                            .id = 1U,
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

static PlatformCfg platformCfgVC0820C070_VCU = {
    .platform = "VCC_GEN3_VC0820C070_CPHY_x2_VCU",
    .platformConfig = "VCC_GEN3_VC0820C070_CPHY_x2_VCU",
#if !NV_IS_SAFETY
    .description = "1 x FLC (AR0820) module in 2 lane CPHY mode",
#endif // !NV_IS_SAFETY
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD,
            .deserInfo = {
                .name = "MAX96712_Fusa",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator Fusa",
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
                { /////////////////////// ID : 3 ///////////////////////////////
                    .name = "VC0820C070R24",
#if !NV_IS_SAFETY
                    .description = "AR0820 70FOV 24BIT RGGB module - 70-deg FOV, 24-bit capture, MIPI-AR0820, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 3U,
                    .isSimulatorModeEnabled = false,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim MAX96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX
                        .serdesGPIOPinMappings = {
                            {
                                .sourceGpio = 5,
                                .destGpio = 10
                            }
                        }
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C02",
#if !NV_IS_SAFETY
                        .description = "M24C02 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x50,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 0U,
                            .name = "AR0820C",
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
                                    .isEmbeddedDataTypeEnabled = true
                            },
                            .isTriggerModeEnabled = false,
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
            .pwrPort = 1U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2500000U, 2500000U},
            .gpios = {7}
        }
    }
};

#endif // VC0820_B_HPP
