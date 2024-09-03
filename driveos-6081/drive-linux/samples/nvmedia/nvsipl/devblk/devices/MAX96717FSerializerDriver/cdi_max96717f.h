/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef _CDI_MAX96717F_H_
#define _CDI_MAX96717F_H_
#include "devblk_cdi.h"
#include <stdbool.h>

typedef enum {
    /* This type must be contiguous and start from 0 */
    CDI_WRITE_PARAM_CMD_MAX96717F_INVALID = 0u,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK,
    CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES,
    CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY,
    CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK,
    CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS,
    CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW,
    CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD,
    CDI_WRITE_PARAM_CMD_MAX96717F_NUM,
} WriteParametersCmdMAX96717F;

typedef enum {
    CDI_MAX96717F_DATA_TYPE_INVALID = 0u,
    CDI_MAX96717F_DATA_TYPE_RAW10,
    CDI_MAX96717F_DATA_TYPE_RAW12
} DataTypeMAX96717F;

typedef enum {
    CDI_MAX96717F_GPIO_TYPE_INVALID = 0u,
    CDI_MAX96717F_GPIO_TYPE_MFP0,
    CDI_MAX96717F_GPIO_TYPE_MFP1,
    CDI_MAX96717F_GPIO_TYPE_MFP2,
    CDI_MAX96717F_GPIO_TYPE_MFP3,
    CDI_MAX96717F_GPIO_TYPE_MFP4,
    CDI_MAX96717F_GPIO_TYPE_MFP5,
    CDI_MAX96717F_GPIO_TYPE_MFP6,
    CDI_MAX96717F_GPIO_TYPE_MFP7,
    CDI_MAX96717F_GPIO_TYPE_MFP8,
    CDI_MAX96717F_GPIO_TYPE_NUM,
} GPIOTypeMAX96717F;


typedef enum {
    CDI_MAX96717F_INVALID_DEV = 0u,
    CDI_MAX96717F_MAX96717F,
    CDI_MAX96717F_MAX96717,
} DeviceIDMAX96717F;

typedef enum {
    CDI_MAX96717F_INVALID_REV = 0u,
    CDI_MAX96717F_REV_2,
    CDI_MAX96717F_REV_4,
} RevisionMAX96717F;

typedef struct {
    uint8_t phy0_d0;
    uint8_t phy0_d1;
    uint8_t phy1_d0;
    uint8_t phy1_d1;
    uint8_t phy2_d0;
    uint8_t phy2_d1;
    uint8_t phy3_d0;
    uint8_t phy3_d1;
    bool enableMapping;
} phyMapMAX96717F;

typedef union {
    struct {
        uint8_t source;               /* 7 bit I2C address */
        uint8_t destination;          /* 7 bit I2C address */
    } Translator;
    struct {
        uint8_t address;              /* 7 bit I2C address */
    } DeviceAddress;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool level;                   /* level = true to set logic high */
    } GPIOOutp;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        uint8_t rxID;                 /* GPIO Rx ID. Must match with deserializer val */
    } FSyncGPIO;
    struct {
        GPIOTypeMAX96717F gpioInd;    /* Must be 0-8 for MFP0-MFP8 pins */
        bool enableRClk;              /* Enable RCLK output on PCLKOUT pin */
    } RefClkGPIO;
    struct {
        DataTypeMAX96717F dataType;   /* Sensor data type for pixel data */
        bool embDataType;             /* Set to true if emb data has emb data type */
    } ConfigVideoPipeline;
    struct {
        phyMapMAX96717F mapping;
        uint8_t numDataLanes;
    } ConfigPhy;
    struct {
        uint8_t freq;                 /* Generate Clock Rate in Mhz */
    } ClockRate;
    struct {
        uint8_t srcGpio;              /* Serializer GPIO number as the input */
        uint8_t dstGpio;              /* Destination GPIO number as the output */
    } GPIOForward;
} ReadWriteParamsMAX96717F;

DevBlkCDIDeviceDriver *GetMAX96717FDriver(void);

NvMediaStatus
MAX96717FCheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96717FSetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96717FReadRegister(
    DevBlkCDIDevice *handle,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX96717FWriteRegister(
    DevBlkCDIDevice *handle,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX96717FWriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96717FDumpRegisters(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96717FGetTemperature(
    DevBlkCDIDevice *handle,
    float_t *temperature);

#endif /* _CDI_MAX96717F_H_ */
