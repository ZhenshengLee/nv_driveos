/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_DS90UB971_H_
#define _CDI_DS90UB971_H_

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include <stdbool.h>

typedef enum {
    CDI_CONFIG_DS90UB971_INVALID = 0u,
    CDI_CONFIG_DS90UB971_LINE_ENHANCEMENT,
    CDI_CONFIG_DS90UB971_DUMP_REGS,
    CDI_CONFIG_DS90UB971_NUM,
} ConfigSetsDS90UB971;

typedef enum {
    /* This type must be contiguous and start from 0 */
    CDI_WRITE_PARAM_CMD_DS90UB971_INVALID = 0u,
    CDI_WRITE_PARAM_CMD_DS90UB971_SET_TPG,
    CDI_WRITE_PARAM_CMD_DS90UB971_SET_CLKOUT,
    CDI_WRITE_PARAM_CMD_DS90UB971_SET_FSYNC_GPIO,
    CDI_WRITE_PARAM_CMD_DS90UB971_SENSOR_RESET,
    CDI_WRITE_PARAM_CMD_DS90UB971_NUM,
} WriteParametersCmdDS90UB971;

typedef enum {
    CDI_DS90UB971_INVALID_REV = 0u,
    CDI_DS90UB971_REV_1 = 1u,
    CDI_DS90UB971_REV_2 = 2u,
} RevisionDS90UB971;

typedef enum {
    CDI_DS90UB971_GPIO_0 = 0u,
    CDI_DS90UB971_GPIO_1,
    CDI_DS90UB971_GPIO_2,
    CDI_DS90UB971_GPIO_3,
    CDI_DS90UB971_GPIO_NUM,
} GPIODS90UB971;

typedef struct {
    DevBlkCDII2CPgmr i2cProgrammer;
} _DriverHandle;

typedef union {
    struct {
        GPIODS90UB971 gpioInd;
        uint8_t rxGpioID;
    } fsyncGpio;

    struct {
        GPIODS90UB971 gpio;
        bool level;
    } localGpio;

    struct {
        uint32_t height;
        uint32_t width;
        uint32_t frameRate;
    } TPG;

    struct {
        uint8_t N;
        uint8_t M;
    } ClkOut;

} ReadWriteParamsDS90UB971;

DevBlkCDIDeviceDriver *GetDS90UB971Driver(void);

NvMediaStatus
DS90UB971CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS90UB971SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS90UB9715WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
DS90UB971ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
DS90UB971WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
DS90UB971WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
DS90UB971DumpRegisters(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS971DumpRegs(
    DevBlkCDIDevice *handle);

#endif /* _CDI_DS90UB971_H_ */
