/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_MAX9295_H_
#define _CDI_MAX9295_H_

#include "devblk_cdi.h"
#include <stdbool.h>

typedef enum {
    /* This type must be contiguous and start from 0 */
    CDI_WRITE_PARAM_CMD_MAX9295_INVALID = 0u,
    CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_A,
    CDI_WRITE_PARAM_CMD_MAX9295_SET_TRANSLATOR_B,
    CDI_WRITE_PARAM_CMD_MAX9295_SET_DEVICE_ADDRESS,
    CDI_WRITE_PARAM_CMD_MAX9295_SET_GPIO_OUTPUT,
    CDI_WRITE_PARAM_CMD_MAX9295_SET_FSYNC_GPIO,
    CDI_WRITE_PARAM_CMD_MAX9295_ENABLE_REF_CLOCK,
    CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_VIDEO_PIPELINES,
    CDI_WRITE_PARAM_CMD_MAX9295_CONFIG_PHY,
    CDI_WRITE_PARAM_CMD_MAX9295_GPIO_FORWARD,
    CDI_WRITE_PARAM_CMD_MAX9295_NUM,
} WriteParametersCmdMAX9295;

typedef enum {
    CDI_MAX9295_DATA_TYPE_INVALID = 0u,
    CDI_MAX9295_DATA_TYPE_RAW10,
    CDI_MAX9295_DATA_TYPE_RAW12,
    CDI_MAX9295_DATA_TYPE_RAW16,
} DataTypeMAX9295;

typedef enum {
    CDI_MAX9295_GPIO_TYPE_INVALID = 0u,
    CDI_MAX9295_GPIO_TYPE_MFP0,
    CDI_MAX9295_GPIO_TYPE_MFP1,
    CDI_MAX9295_GPIO_TYPE_MFP2,
    CDI_MAX9295_GPIO_TYPE_MFP3,
    CDI_MAX9295_GPIO_TYPE_MFP4,
    CDI_MAX9295_GPIO_TYPE_MFP5,
    CDI_MAX9295_GPIO_TYPE_MFP6,
    CDI_MAX9295_GPIO_TYPE_MFP7,
    CDI_MAX9295_GPIO_TYPE_MFP8,
    CDI_MAX9295_GPIO_TYPE_NUM,
} GPIOTypeMAX9295;

typedef enum {
    CDI_MAX9295_INVALID_REV = 0u,
    CDI_MAX9295_REV_5,
    CDI_MAX9295_REV_7,
    CDI_MAX9295_REV_8,
} RevisionMAX9295;

typedef struct {
    uint8_t phy0_d0;
    uint8_t phy0_d1;
    uint8_t phy1_d0;              /* data lane0 connected between sensor and serializer */
    uint8_t phy1_d1;              /* data lane1 connected between sensor and serializer */
    uint8_t phy2_d0;              /* data lane2 connected between sensor and serializer */
    uint8_t phy2_d1;              /* data lane3 connected between sensor and serializer */
    uint8_t phy3_d0;
    uint8_t phy3_d1;
    bool enableMapping;
} phyMapMAX9295;

typedef struct {
    uint8_t phy1_d0;               /* lane0pol */
    uint8_t phy1_d1;               /* lane1pol */
    uint8_t phy1_clk;              /* clk1pol */
    uint8_t phy2_d0;               /* lane2pol */
    uint8_t phy2_d1;               /* lane3pol */
    uint8_t phy2_clk;              /* clk2pol */
    bool setPolarity;
} phyPolarityMAX9295;

typedef union {
    struct {
        uint8_t source;             /* 7 bit I2C address */
        uint8_t destination;        /* 7 bit I2C address */
    } Translator;

    struct {
        uint8_t address;            /* 7 bit I2C address */
    } DeviceAddress;

    struct {
        GPIOTypeMAX9295 gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool level;                 /* level = true to set logic high */
    } GPIOOutp;

    struct {
        GPIOTypeMAX9295 gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        uint8_t rxID;               /* GPIO Rx ID. Must match with deserialiser val */
    } FSyncGPIO;

    struct {
        GPIOTypeMAX9295 gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool enableRClk;            /* Enable RCLK output on PCLKOUT pin */
    } RefClkGPIO;

    struct {
        uint8_t srcGpio;            /* Serializer GPIO number as the input */
        uint8_t dstGpio;            /* Destination GPIO number as the output */
    } GPIOForward;

    struct {
        DataTypeMAX9295 dataType;   /* Sensor data type for pixel data */
        bool embDataType;           /* Set to true if emb data has emb data type */
    } ConfigVideoPipeline;

    struct {
        phyMapMAX9295 mapping;
        phyPolarityMAX9295 polarity;
        uint8_t numDataLanes;
    } ConfigPhy;
} ReadWriteParamsMAX9295;

DevBlkCDIDeviceDriver *GetMAX9295Driver(void);

NvMediaStatus
MAX9295CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX9295SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX9295WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX9295ReadErrorStatus(
    DevBlkCDIDevice *handle,
    uint32_t dataLength,
    uint8_t *dataBuff);

#if !NV_IS_SAFETY
NvMediaStatus
MAX9295DumpRegisters(
    DevBlkCDIDevice *handle);
#endif
#endif /* _CDI_MAX9295_H_ */
