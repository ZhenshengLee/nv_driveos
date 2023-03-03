/*
 * Copyright (c) 2015-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "devblk_cdi.h"
#include "cdi_max96705.h"
#include "os_common.h"

#define REGISTER_ADDRESS_BYTES  1
#define REG_WRITE_BUFFER        32

#define GET_BLOCK_LENGTH(x) x[0]
#define GET_BLOCK_DATA(x)   &x[1]
#define SET_NEXT_BLOCK(x)   x += (x[0] + 1)

DevBlkCDII2CReg max96705_enable_reverse_channel_reg = {
    0x04, 0x43,  // Enable config link, wait 5ms
};

DevBlkCDII2CReg max96705_enable_serial_link_reg = {
    0x04, 0x83,   // Enable serial link, disable config link, wait 5ms
};

DevBlkCDII2CReg max96705_defaults_regs[] = {
    {0x08, 0x01},
    {0x97, 0x5F}, //enable bit only on 96705
};
DevBlkCDII2CRegList max96705_defaults = {
    .regs = max96705_defaults_regs,
    .numRegs = I2C_ARRAY_SIZE(max96705_defaults_regs),
};

DevBlkCDII2CReg max96705_program_cutoff_freq_regs[] = {
    {0x08, 0x03},
    {0x97, 0x3F},
};
DevBlkCDII2CRegList max96705_program_cutoff_freq = {
    .regs = max96705_program_cutoff_freq_regs,
    .numRegs = I2C_ARRAY_SIZE(max96705_program_cutoff_freq_regs),
};

DevBlkCDII2CReg max96705_config_input_mode_reg = {
    0x07, 0xC4,  // PCLKIN setting DBL=1, HIBW=1, BWS=0, ES=0, HVEN=1
};

DevBlkCDII2CReg max96705_set_translator_a_regs[] = {
    {0x09, 0x00},
    {0x0a, 0x00},
};
DevBlkCDII2CRegList max96705_set_translator_a = {
    .regs = max96705_set_translator_a_regs,
    .numRegs = I2C_ARRAY_SIZE(max96705_set_translator_a_regs),
};

DevBlkCDII2CReg max96705_set_translator_b_regs[] = {
    {0x0b, 0x00},
    {0x0c, 0x00},
};
DevBlkCDII2CRegList max96705_set_translator_b = {
    .regs = max96705_set_translator_b_regs,
    .numRegs = I2C_ARRAY_SIZE(max96705_set_translator_b_regs),
};

DevBlkCDII2CReg max96705_set_xbar_regs[] = {
    {0x20, 0x04},
    {0x21, 0x03},
    {0x22, 0x02},
    {0x23, 0x01},
    {0x24, 0x00},
    {0x25, 0x40},
    {0x26, 0x40},
    {0x27, 0x0E},
    {0x28, 0x2F},
    {0x29, 0x0E},
    {0x2A, 0x40},
    {0x2B, 0x40},
    {0x2C, 0x40},
    {0x2D, 0x40},
    {0x2E, 0x40},
    {0x2F, 0x40},
    {0x30, 0x17},
    {0x31, 0x16},
    {0x32, 0x15},
    {0x33, 0x14},
    {0x34, 0x13},
    {0x35, 0x12},
    {0x36, 0x11},
    {0x37, 0x10},
    {0x38, 0x07},
    {0x39, 0x06},
    {0x3A, 0x05},
    {0x3B, 0x40},
    {0x3C, 0x40},
    {0x3D, 0x40},
    {0x3E, 0x40},
    {0x3F, 0x0E},
    {0x40, 0x2F},
    {0x41, 0x0E},
};
DevBlkCDII2CRegList max96705_set_xbar = {
    .regs = max96705_set_xbar_regs,
    .numRegs = I2C_ARRAY_SIZE(max96705_set_xbar_regs),
};

DevBlkCDII2CReg max96705_auto_config_link_reg = {
    0x67, 0xE4,
};

DevBlkCDII2CReg max96705_double_input_mode_reg = {
    0x07, 0x80,  // PCLKIN setting DBL=1
};

DevBlkCDII2CReg max96705_disable_reverse_channel_HIM_mode_reg = {
    0x4D, 0x40,
};

DevBlkCDII2CReg max96705_enable_reverse_channel_HIM_mode_reg = {
    0x4D, 0xC0,
};

DevBlkCDII2CReg max96705_i2c_remote_master_timeout_reg = {
    0x99, 0x0F, // set remote master timeout to never (Bug 1802338, 200419005)
};

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *driverHandle = NULL;

    if(!handle)
        return NVMEDIA_STATUS_BAD_PARAMETER;

    driverHandle = calloc(1, sizeof(_DriverHandle));
    if (driverHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96705: Memory allocation for context failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }
    handle->deviceDriverHandle = (void *)driverHandle;

    // Create the I2C programmer for register read/write
    driverHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                         MAX96705_NUM_ADDR_BYTES,
                                                         MAX96705_NUM_DATA_BYTES);
    if(driverHandle->i2cProgrammer == NULL) {
        printf("Failed to initialize the I2C programmer\n");
        free(driverHandle);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *driverHandle = NULL;

    if((handle == NULL) || ((driverHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(driverHandle->i2cProgrammer);

    if (handle->deviceDriverHandle != NULL) {
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96705CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       0x1e,
                                       &data);
    if(status != NVMEDIA_STATUS_OK)
        return status;

    if (CHECK_96705ID(data) == 0) {
        return NVMEDIA_STATUS_ERROR;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       0x1f,
                                       &data);
    if(status != NVMEDIA_STATUS_OK) {
        return status;
    }

    LOG_MSG("MAX96705: Revision %u detected!\n", data & 0xF);
    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
SetPreEmp(
    DevBlkCDIDevice *handle,
    unsigned char preemp)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DevBlkCDII2CReg i2cReg = {0x06, 0xA0};
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    i2cReg.data |= (preemp & 0xF);

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        i2cReg.address,
                                        i2cReg.data);
    LOG_MSG("MAX96705: Pre-emphasis set to 0x%02x\n", i2cReg.data);

    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    nvsleep(5000); /* Delay to wait since I2C unavailable while GMSL locks from programming guide */

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
SetRegenVsync(
    DevBlkCDIDevice *handle,
    unsigned int vsync_high,
    unsigned int vsync_low,
    unsigned int vsync_delay,
    unsigned char vsync_trig,
    unsigned int pclk)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DevBlkCDII2CReg max96705_regen_vsync_delay_regs[] = {
        {0x44, 0x00},
        {0x45, 0x9C},
        {0x46, 0x80},
    };
    DevBlkCDII2CRegList max96705_regen_vsync_delay = {
        .regs = max96705_regen_vsync_delay_regs,
        .numRegs = I2C_ARRAY_SIZE(max96705_regen_vsync_delay_regs),
    };
    DevBlkCDII2CReg max96705_regen_vsync_high_regs[] = {
        {0x47, 0x00},
        {0x48, 0xb0},
        {0x49, 0x00},
    };
    DevBlkCDII2CRegList max96705_regen_vsync_high = {
        .regs = max96705_regen_vsync_high_regs,
        .numRegs = I2C_ARRAY_SIZE(max96705_regen_vsync_high_regs),
    };
    DevBlkCDII2CReg max96705_regen_vsync_low_regs[] = {
        {0x4A, 0x00},
        {0x4B, 0xb0},
        {0x4C, 0x00},
    };
    DevBlkCDII2CRegList max96705_regen_vsync_low = {
        .regs = max96705_regen_vsync_low_regs,
        .numRegs = I2C_ARRAY_SIZE(max96705_regen_vsync_low_regs),
    };
    DevBlkCDII2CReg max96705_regen_vsync_trig_reg = {
        0x43, 0x21
    };
#if 0 // TODO : need to check with MAXIM
    DevBlkCDII2CReg max96705_vsync_align_reg = {
        0x67, 0xc4,
    };
#endif
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (vsync_delay != 0) {
        max96705_regen_vsync_delay_regs[0].data = vsync_delay * (pclk / 1000000) / 256 / 256;
        max96705_regen_vsync_delay_regs[1].data = ((vsync_delay * (pclk / 1000000)) % (256 * 256)) / 256;
        max96705_regen_vsync_delay_regs[2].data = ((vsync_delay * (pclk / 1000000)) % 256) / 256;

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_regen_vsync_delay);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    if (vsync_high != 0) {
        max96705_regen_vsync_high_regs[0].data = vsync_high * (pclk / 1000000) / 256 / 256;
        max96705_regen_vsync_high_regs[1].data = ((vsync_high * (pclk / 1000000)) % (256 * 256)) / 256;
        max96705_regen_vsync_high_regs[2].data = ((vsync_high * (pclk / 1000000)) % 256) / 256;

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_regen_vsync_high);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    if (vsync_low != 0) {
        max96705_regen_vsync_low_regs[0].data = vsync_low * (pclk / 1000000) / 256 / 256;
        max96705_regen_vsync_low_regs[1].data = ((vsync_low * (pclk / 1000000)) % (256 * 256)) / 256;
        max96705_regen_vsync_low_regs[2].data = ((vsync_low * (pclk / 1000000)) % 256) / 256;

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_regen_vsync_low);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    if ((vsync_trig == 1) || (vsync_trig == 2)) {
        if (vsync_trig == 2) {
           max96705_regen_vsync_trig_reg.data |= (1 << 2);
        }

        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            max96705_regen_vsync_trig_reg.address,
                                            max96705_regen_vsync_trig_reg.data);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96705SetDefaults(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_defaults);
    return status;
}

NvMediaStatus
MAX96705SetDeviceConfig(
        DevBlkCDIDevice *handle,
        uint32_t enumeratedDeviceConfig)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch(enumeratedDeviceConfig) {
        case CDI_CONFIG_MAX96705_DEFAULT:
            status = MAX96705SetDefaults(
                handle);
           break;
        case CDI_CONFIG_MAX96705_ENABLE_SERIAL_LINK:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_enable_serial_link_reg.address,
                                                max96705_enable_serial_link_reg.data);
            break;
        case CDI_CONFIG_MAX96705_PCLKIN:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_config_input_mode_reg.address,
                                                max96705_config_input_mode_reg.data);
            nvsleep(10000);  /* wait 10ms */
            break;
        case CDI_CONFIG_MAX96705_ENABLE_REVERSE_CHANNEL:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_enable_reverse_channel_reg.address,
                                                max96705_enable_reverse_channel_reg.data);
            nvsleep(5000);  /* wait 5ms */
            break;
        case CDI_CONFIG_MAX96705_SET_AUTO_CONFIG_LINK:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_auto_config_link_reg.address,
                                                max96705_auto_config_link_reg.data);
            break;
        case CDI_CONFIG_MAX96705_DOUBLE_INPUT_MODE:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_double_input_mode_reg.address,
                                                max96705_double_input_mode_reg.data);
            break;
        case CDI_CONFIG_MAX96705_DISABLE_HIM_MODE:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_disable_reverse_channel_HIM_mode_reg.address,
                                                max96705_disable_reverse_channel_HIM_mode_reg.data);
            nvsleep(10000);  /* wait 10ms */
            break;
        case CDI_CONFIG_MAX96705_ENABLE_HIM_MODE:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_enable_reverse_channel_HIM_mode_reg.address,
                                                max96705_enable_reverse_channel_HIM_mode_reg.data);
            nvsleep(10000);  /* wait 10ms */
            break;
        case CDI_CONFIG_MAX96705_SET_XBAR:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_set_xbar);
            break;
         case CDI_CONFIG_MAX96705_SET_MAX_REMOTE_I2C_MASTER_TIMEOUT:
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                max96705_i2c_remote_master_timeout_reg.address,
                                                max96705_i2c_remote_master_timeout_reg.data);
            break;
        case CDI_CONFIG_MAX96705_PROGRAM_CUTOFF_FREQ:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_program_cutoff_freq);
           break;
       default:
            status =  NVMEDIA_STATUS_NOT_SUPPORTED;
            break;
    }

    return status;
}

NvMediaStatus
MAX96705ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    unsigned char *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t i;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || (dataBuff == NULL) ||
       ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                        (uint16_t) registerNum,
                                        &dataBuff[i]);
    }

    return status;
}

NvMediaStatus
MAX96705WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    unsigned char *dataBuff)
{
    NvMediaStatus status;
    DevBlkCDII2CReg *dataRegs = NULL;
    DevBlkCDII2CRegList data;
    uint32_t i, length;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || (dataBuff == NULL) ||
        ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    length = (REG_WRITE_BUFFER < dataLength) ? REG_WRITE_BUFFER: dataLength;
    dataRegs = (DevBlkCDII2CReg *) calloc(1, sizeof(DevBlkCDII2CReg) * length);
    if (dataRegs == NULL) {
        SIPL_LOG_ERR_STR("MAX96705: Memory allocation for resiter write failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    for (i = 0; i < length; i++) {
        dataRegs[i].address = (uint16_t) (registerNum + i);
        dataRegs[i].data = dataBuff[i];
    }
    data.regs = dataRegs;
    data.numRegs = length;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &data);

    return status;
}

static NvMediaStatus
SetTranslator(
        DevBlkCDIDevice *handle,
        uint32_t parameterType,
        WriteReadParametersParamMAX96705 *param)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;
    DevBlkCDII2CReg *max96705_set_translator_regs;
    DevBlkCDII2CRegList max96705_set_translator;
    uint32_t numRegs;


    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if(parameterType == CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A) {
        max96705_set_translator_regs = max96705_set_translator_a_regs;
        numRegs = I2C_ARRAY_SIZE(max96705_set_translator_a_regs);
    } else {
        max96705_set_translator_regs = max96705_set_translator_b_regs;
        numRegs = I2C_ARRAY_SIZE(max96705_set_translator_b_regs);
    }

    max96705_set_translator_regs[0].data = param->Translator.source << 1;
    max96705_set_translator_regs[1].data = param->Translator.destination << 1;

    max96705_set_translator.regs = max96705_set_translator_regs;
    max96705_set_translator.numRegs = numRegs;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &max96705_set_translator);
    return status;
}

static NvMediaStatus
SetDeviceAddress(
        DevBlkCDIDevice *handle,
        unsigned char address)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if(address > 0x80) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, 0x0, address << 1);

    return status;
}

static NvMediaStatus
GetDeviceAddress(
        DevBlkCDIDevice *handle,
        unsigned char *address)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0, address);

    return status;
}

static NvMediaStatus
SetInputMode(
        DevBlkCDIDevice *handle,
        ConfigureInputModeMAX96705 *inputmode)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    max96705_config_input_mode_reg.data =  (unsigned char)inputmode->byte;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        max96705_config_input_mode_reg.address,
                                        max96705_config_input_mode_reg.data);

    return status;
}

NvMediaStatus
MAX96705WriteParameters(
        DevBlkCDIDevice *handle,
        uint32_t parameterType,
        uint32_t parameterSize,
        void *parameter)
{
    NvMediaStatus status;
    WriteReadParametersParamMAX96705 *param;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || (parameter == NULL) ||
       ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    param = (WriteReadParametersParamMAX96705 *)parameter;

    switch(parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_A:
        case CDI_WRITE_PARAM_CMD_MAX96705_SET_TRANSLATOR_B:
            if(parameterSize != sizeof(param->Translator))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            status = SetTranslator(
                handle,
                parameterType,
                param);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96705_SET_DEVICE_ADDRESS:
            if(parameterSize != sizeof(param->DeviceAddress))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            status = SetDeviceAddress(
                handle,
                param->DeviceAddress.address);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96705_CONFIG_INPUT_MODE:
            if(parameterSize != sizeof(param->inputmode))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            status = SetInputMode(
                handle,
                param->inputmode);
           break;
        case CDI_WRITE_PARAM_CMD_MAX96705_SET_PREEMP:
            if(parameterSize != sizeof(param->preemp))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            status = SetPreEmp(
                handle,
                param->preemp);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96705_REGEN_VSYNC:
            if (parameterSize != sizeof(param->vsyncRegen)) {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            status = SetRegenVsync(handle,
                                   param->vsyncRegen.vsync_high,
                                   param->vsyncRegen.vsync_low,
                                   param->vsyncRegen.vsync_delay,
                                   param->vsyncRegen.vsync_trig,
                                   param->vsyncRegen.pclk);
            break;
        default:
            status = NVMEDIA_STATUS_NOT_SUPPORTED;
            break;
    }

    return status;
}

NvMediaStatus
MAX96705ReadParameters(
        DevBlkCDIDevice *handle,
        uint32_t parameterType,
        uint32_t parameterSize,
        void *parameter)
{
    NvMediaStatus status;
    WriteReadParametersParamMAX96705 *param;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || (parameter == NULL) ||
       ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    param = (WriteReadParametersParamMAX96705 *)parameter;

    switch(parameterType) {
        case CDI_READ_PARAM_CMD_MAX96705_GET_DEVICE_ADDRESS:
            if(parameterSize != sizeof(param->DeviceAddress))
                return NVMEDIA_STATUS_BAD_PARAMETER;
            status = GetDeviceAddress(
                handle,
                (unsigned char*)param);
            break;
        default:
            status = NVMEDIA_STATUS_NOT_SUPPORTED;
            break;
    }

    return status;
}

NvMediaStatus
MAX96705DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data;
    uint32_t i = 0u, addr = 0u;
    _DriverHandle *drvHandle = NULL;

    if((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0; i <= MAX96705_MAX_REG_ADDRESS; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, i, &data);
        if (status != NVMEDIA_STATUS_OK)
            return status;

        if (i == 0)
            addr = data;

        printf(" Max96705(0x%x) : 0x%02x - 0x%02x\n", addr, i, data);
    }

    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 96705 Serializer",
    .regLength = 1,
    .dataLength = 1,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetMAX96705Driver(void)
{
    return &deviceDriver;
}

