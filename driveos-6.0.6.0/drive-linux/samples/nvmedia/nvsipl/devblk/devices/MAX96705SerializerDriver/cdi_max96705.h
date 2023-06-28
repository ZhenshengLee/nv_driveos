/*
 * Copyright (c) 2015-2020, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_MAX96705_H_
#define _CDI_MAX96705_H_

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"

#define MAX96705_DEVICE_ID    0x41 //could also be 0x43/0x51/0x53
#define CHECK_96705ID(id)     (((id & 0xED) == MAX96705_DEVICE_ID)? 1:0)
#define MAX96705_MAX_REG_ADDRESS        255
#define MAX96705_NUM_ADDR_BYTES         1u
#define MAX96705_NUM_DATA_BYTES         1u

typedef enum {
    CDI_CONFIG_MAX96705_DEFAULT                   = 0,
    CDI_CONFIG_MAX96705_ENABLE_SERIAL_LINK,
    CDI_CONFIG_MAX96705_PCLKIN,
    CDI_CONFIG_MAX96705_ENABLE_REVERSE_CHANNEL,
    CDI_CONFIG_MAX96705_SET_AUTO_CONFIG_LINK,
    CDI_CONFIG_MAX96705_DOUBLE_INPUT_MODE,
    CDI_CONFIG_MAX96705_ENABLE_HIM_MODE,
    CDI_CONFIG_MAX96705_DISABLE_HIM_MODE,
    CDI_CONFIG_MAX96705_SET_XBAR,
    CDI_CONFIG_MAX96705_SET_MAX_REMOTE_I2C_MASTER_TIMEOUT,
    CDI_CONFIG_MAX96705_PROGRAM_CUTOFF_FREQ,
} ConfigSetsMAX96705;

typedef enum {
    CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A = 0,
    CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_B,
    CDI_WRITE_PARAM_CMD_MAX96705_SET_DEVICE_ADDRESS,
    CDI_WRITE_PARAM_CMD_MAX96705_CONFIG_INPUT_MODE,
    CDI_WRITE_PARAM_CMD_MAX96705_SET_PREEMP,
    CDI_WRITE_PARAM_CMD_MAX96705_REGEN_VSYNC,
} WriteParametersCmdMAX96705;


typedef enum {
    CDI_SET_PREEMP_MAX96705_PREEMP_OFF = 0,
    CDI_SET_PREEMP_MAX96705_NEG_1_2DB,
    CDI_SET_PREEMP_MAX96705_NEG_2_5DB,
    CDI_SET_PREEMP_MAX96705_NEG_4_1DB,
    CDI_SET_PREEMP_MAX96705_NEG_6_0DB,
    CDI_SET_PREEMP_MAX96705_PLU_1_1DB = 0x8,
    CDI_SET_PREEMP_MAX96705_PLU_2_2DB,
    CDI_SET_PREEMP_MAX96705_PLU_3_3DB,
    CDI_SET_PREEMP_MAX96705_PLU_4_4DB,
    CDI_SET_PREEMP_MAX96705_PLU_6_0DB,
    CDI_SET_PREEMP_MAX96705_PLU_8_0DB,
    CDI_SET_PREEMP_MAX96705_PLU_10_5DB,
    CDI_SET_PREEMP_MAX96705_PLU_14_0DB,
} SetPREEMPMAX96705;

typedef enum {
    CDI_READ_PARAM_CMD_MAX96705_GET_DEVICE_ADDRESS = 0,
} ReadParametersCmdMAX96705;

#define CDI_INPUT_MODE_MAX96705_DOUBLE_INPUT_MODE      1
#define CDI_INPUT_MODE_MAX96705_SINGLE_INPUT_MODE      0
#define CDI_INPUT_MODE_MAX96705_HIGH_BANDWIDTH_MODE    1
#define CDI_INPUT_MODE_MAX96705_LOW_BANDWIDTH_MODE     0
#define CDI_INPUT_MODE_MAX96705_BWS_22_BIT_MODE        0
#define CDI_INPUT_MODE_MAX96705_BWS_30_BIT_MODE        1
#define CDI_INPUT_MODE_MAX96705_PCLKIN_RISING_EDGE     0
#define CDI_INPUT_MODE_MAX96705_PCLKIN_FALLING_EDGE    1
#define CDI_INPUT_MODE_MAX96705_HVEN_ENCODING_ENABLE   1
#define CDI_INPUT_MODE_MAX96705_HVEN_ENCODING_DISABLE  0
#define CDI_INPUT_MODE_MAX96705_EDC_1_BIT_PARITY       0
#define CDI_INPUT_MODE_MAX96705_EDC_6_BIT_CRC          1
#define CDI_INPUT_MODE_MAX96705_EDC_6_BIT_HAMMING_CODE 2
#define CDI_INPUT_MODE_MAX96705_EDC_NOT_USE            3

typedef union {
    struct {
        unsigned edc : 2;
        unsigned hven : 1;
        unsigned reserved : 1;
        unsigned es : 1;
        unsigned bws : 1;
        unsigned hibw : 1;
        unsigned dbl : 1;
    } bits;
    unsigned char byte;
} ConfigureInputModeMAX96705;

typedef struct {
    union {
        struct {
            unsigned char source;
            unsigned char destination;
        } Translator;
        struct {
            unsigned char address;
        } DeviceAddress;
        struct {
            unsigned int vsync_high;  /* usec */
            unsigned int vsync_low;   /* usec */
            unsigned int vsync_delay; /* usec */
            unsigned char vsync_trig; /* VS trigger edge, 1 - falling edge, 2 - rising edge */
            unsigned int pclk;        /* Hz */
        } vsyncRegen;
        ConfigureInputModeMAX96705 *inputmode;
        unsigned char preemp;
    };
} WriteReadParametersParamMAX96705;

typedef struct {
    DevBlkCDII2CPgmr i2cProgrammer;
} _DriverHandle;

DevBlkCDIDeviceDriver *GetMAX96705Driver(void);

NvMediaStatus
MAX96705CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96705SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96705SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig);

NvMediaStatus
MAX96705ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    unsigned char *dataBuff);

NvMediaStatus
MAX96705WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    unsigned char *dataBuff);

NvMediaStatus
MAX96705WriteParameters(
        DevBlkCDIDevice *handle,
        uint32_t parameterType,
        uint32_t parameterSize,
        void *parameter);

NvMediaStatus
MAX96705ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96705DumpRegisters(
    DevBlkCDIDevice *handle);

#endif /* _CDI_MAX96705_H_ */
