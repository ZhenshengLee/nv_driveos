/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max96717f.h"
#include "os_common.h"

#define INCK_24MHZ                    (24U)
#define INCK_25MHZ                    (25U)
#define UINT8_MAX_SHIFT                7U
#define DPLL_LOCK_CHECK_RETRY_CNT      5U
#define DPLL_LOCK_UNLOCKED_BIT         7U
#define GET_BLOCK_LENGTH(x)            x[0]
#define GET_SIZE(x)                    sizeof(x)
#define GET_BLOCK_DATA(x)              &x[1]
#define SET_NEXT_BLOCK(x)              x += (x[0] + 1)
#define MAX96717F_NUM_ADDR_BYTES       2u
#define MAX96717F_NUM_DATA_BYTES       1u
#define REG_WRITE_BUFFER_BYTES         MAX96717F_NUM_DATA_BYTES
#define MAX96717F_CDI_DEVICE_INDEX     0u
#define KELVIN_TO_CELSIUS(x)           ((x) - 273.15F)

#define REG_DEV_ID_ADDR                0x0D
#define MAX96717F_DEV_ID               0xC8 // C8 received from query MAX96717 (T32) --> 0xBF
#define MAX96717_DEV_ID                0xBF
#define REG_DEV_REV_ADDR               0x0E
#define REG_REF_VTG0                   0x03F0U
#define REG_ADCBIST13                  0x1D3B
#define REG_ADCBIST14                  0x1D3C
#define MAX96717F_REG_MAX_ADDRESS      0x1D3D

#define DPLL_LOCK_MAX_CNT              0x777U

typedef struct {
    RevisionMAX96717F revId;
    uint32_t revVal;
} Revision;

typedef struct {
    DevBlkCDII2CPgmr    i2cProgrammer;
} _DriverHandle;

// These values must include all of values in the RevisionMAX96717F enum
static Revision supportedRevisions[] = {
    {CDI_MAX96717F_REV_2, 2u},
    {CDI_MAX96717F_REV_4, 4u},
};

static _DriverHandle * getHandlePriv(DevBlkCDIDevice *handle)
{
    if (handle != NULL) {
        return (_DriverHandle *) handle->deviceDriverHandle;
    }
    return NULL;
}

static inline uint8_t GetBit(uint8_t var, uint8_t shift) {
    uint8_t retVal = 0U;
    if (UINT8_MAX_SHIFT >= shift) {
        retVal = var & (1U << shift);
    }

    return retVal;
}

static NvMediaStatus
SetDeviceAddress(
    DevBlkCDIDevice *handle,
    uint8_t address)
{
    DevBlkCDII2CReg setAddrRegs[] = {
        {0x0000, 0x00},
    };

    _DriverHandle *drvHandle = getHandlePriv(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to SetDeviceAddress");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver passed to SetDeviceAddress");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    // Check for 7 bit I2C address
    if (address >= 0x80) {
        SIPL_LOG_ERR_STR_UINT("MAX96717F: Bad parameter: Address is greater than 0x80", (uint32_t)address);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    setAddrRegs[0].data = address << 1;

    DevBlkCDII2CRegList max96717_setAddr = {
        .regs = setAddrRegs,
        .numRegs = I2C_ARRAY_SIZE(setAddrRegs),
    };

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &max96717_setAddr);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: max96717f_setAddr is failed", (int32_t)status);
    }

done:
    return status;
}

static NvMediaStatus
SetTranslator(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint8_t source,
    uint8_t destination)
{
    DevBlkCDII2CReg setTranslatorRegs[] = {
        {0x0042, 0x00},
        {0x0043, 0x00},
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle pased to SetTranslator");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle pased to SetTranslator");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    // Check for 7 bit I2C address
    if (source >= 0x80) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: Source address will not fit in 7 bits", (uint32_t)source);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (destination >= 0x80) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: Destination address will not fit in 7 bits", (uint32_t)destination);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (parameterType == CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B) {
        setTranslatorRegs[0].address += 2;
        setTranslatorRegs[1].address += 2;
    }

    setTranslatorRegs[0].data = source << 1;
    setTranslatorRegs[1].data = destination << 1;

    DevBlkCDII2CRegList max96717_setTranslator = {
        .regs = setTranslatorRegs,
        .numRegs = I2C_ARRAY_SIZE(setTranslatorRegs),
    };

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &max96717_setTranslator);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: DevBlkCDII2CPgmrWriteArray failed", (int32_t)status);
    }

done:
    return status;
}

static NvMediaStatus
ConfigPipelines(
    DevBlkCDIDevice *handle,
    DataTypeMAX96717F dataType,
    bool embDataType)
{
    DevBlkCDII2CReg mappingPixel12EMBRegs[] = {
        {0x0308, 0x64},
        {0x0311, 0x40},
        {0x0312, 0x04},
        {0x0110, 0x60},
        {0x031E, 0x2C},
        {0x0111, 0x50},
        {0x0318, 0x6C},
        {0x0319, 0x52},
        {0x02D5, 0x07},
        {0x02D8, 0x08},
    };

    DevBlkCDII2CRegList mappingPixel12EMB = {
        .regs = mappingPixel12EMBRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel12EMBRegs),
    };
    DevBlkCDII2CReg mappingPixel12Regs[] = {
        {0x0383, 0x00}, // Disable tunneling mode
        {0x0318, 0x6C}, // RAW12 to pipe Z
        {0x0313, 0x40}, // Double 12-bit data on pipe Z
        {0x031E, 0x38}, // Pipe Z BPP = 24
    };
    DevBlkCDII2CRegList mappingPixel12 = {
        .regs = mappingPixel12Regs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel12Regs),
    };
    DevBlkCDII2CReg mappingPixel10Regs[] = {
        {0x0308, 0x64},
        {0x0311, 0x40},
        {0x0312, 0x04},
        {0x0313, 0x04},
        {0x0110, 0x60},
        {0x031E, 0x34},
        {0x0318, 0x6B}
    };
    DevBlkCDII2CRegList mappingPixel10 = {
        .regs = mappingPixel10Regs,
        .numRegs = I2C_ARRAY_SIZE(mappingPixel10Regs),
    };

    DevBlkCDII2CReg disableHeartbeatRegs[] = {
        {0x0112, 0x0C},
    };
    DevBlkCDII2CRegList disableHeartbeat = {
        .regs = disableHeartbeatRegs,
        .numRegs = I2C_ARRAY_SIZE(disableHeartbeatRegs),
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to ConfigPipelines");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (dataType == CDI_MAX96717F_DATA_TYPE_RAW10) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                            &mappingPixel10);
    } else if (dataType == CDI_MAX96717F_DATA_TYPE_RAW12) {
        if (embDataType) {
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &mappingPixel12EMB);
        } else {
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &mappingPixel12);
        }
    } else {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: ConfigPipelines failed", (int32_t)status);
        goto done;
    }

    if (embDataType == true) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                            &disableHeartbeat);
    }

done:
    return status;
}

static NvMediaStatus
ConfigPhyMap(
    DevBlkCDIDevice *handle,
    phyMapMAX96717F *mapping,
    uint8_t numDataLanes)
{
    DevBlkCDII2CReg phyMapRegs[] = {
        {0x0330, 0x00},
        {0x0331, 0x00},
        {0x0332, 0xEE},
        {0x0333, 0xE4},
    };
    DevBlkCDII2CRegList phyMap = {
        .regs = phyMapRegs,
        .numRegs = I2C_ARRAY_SIZE(phyMapRegs),
    };

    uint8_t regVal = 0U;
    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to ConfigPhyMap");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to ConfigPhyMap");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (mapping == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null mapping passed to ConfigPhyMap");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((numDataLanes != 2U) && (numDataLanes != 4U)) {
        SIPL_LOG_ERR_STR_UINT("MAX96717F: Invalid number of data lines, must be 2 or 4", numDataLanes);
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }
    if ((numDataLanes == 2U) && (mapping->enableMapping)) {
        SIPL_LOG_ERR_STR("MAX96717F: Lane swapping is supported only in 4 lane mode");
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
        goto done;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       phyMapRegs[1].address,      // regData
                                       &regVal);         // data
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("MAX96717F: Failed to read phy map register", (int32_t)phyMapRegs[1].address, (int32_t)status);
        goto done;
    }
    // data lanes indexing starts at 0 (0 = 1 lane, 1 = 2 lanes, etc)
    numDataLanes -= 1;
    // clear the data lanes settings for Port B
    regVal &= ~0x30;
    // Set num data lanes for Port B
    regVal |= (numDataLanes << 4);
    phyMapRegs[1].data = regVal;

    if (mapping->enableMapping) {
        regVal = (mapping->phy1_d1 << 6) |
                (mapping->phy1_d0 << 4) |
                (mapping->phy0_d1 << 2) |
                (mapping->phy0_d0);
        phyMapRegs[2].data = regVal;

        regVal = (mapping->phy3_d1 << 6) |
                 (mapping->phy3_d0 << 4) |
                 (mapping->phy2_d1 << 2) |
                 (mapping->phy2_d0);
        phyMapRegs[3].data = regVal;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &phyMap);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Failed to write phy map array", (int32_t)status);
    }

done:
    return status;
}

static bool
IsDPLLLock(
    DevBlkCDII2CPgmr *i2cProgrammer,
    uint16_t retryCnt)
{
    bool lockStatus = false;
    int16_t cnt = 0;

    if (retryCnt < DPLL_LOCK_MAX_CNT) {
        cnt = (int16_t)retryCnt;
    }

    /* Check DPLL Lock status here */
    do {
        uint8_t regVal = 0U;
        const NvMediaStatus status =
            DevBlkCDII2CPgmrReadUint8(i2cProgrammer,
                                      REG_REF_VTG0,
                                      &regVal);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_HEX_UINT("DevBlkCDII2CPgmrReadUint8 failed",
                                      (uint32_t)status);
            break;
        } else {
            if (GetBit(regVal, DPLL_LOCK_UNLOCKED_BIT) > 0U) {
                lockStatus = true;
                break;
            } else {
                if (usleep((useconds_t)1000U) == 0) {
                    cnt -= 1;
                } else {
                    break;
                }
            }
        }
    } while (cnt > 0);

    return lockStatus;
}

static NvMediaStatus
GenerateClock(
    DevBlkCDIDevice *handle,
    uint8_t freq)
{
    DevBlkCDII2CReg const genClockRegs_25MHz[] = {
        {0x03F1, 0x05},
        {0x03F0, 0x12},
        {0x03F4, 0x0A},
        {0x03F5, 0x07},
        {0x03F0, 0x10},
        {0x1A03, 0x12},
        {0x1A07, 0x04},
        {0x1A08, 0x3D},
        {0x1A09, 0x40},
        {0x1A0A, 0xC0},
        {0x1A0B, 0x7F},
        {0x03F0, 0x11},
    };
    DevBlkCDII2CRegList genClock_25MHz = {
        .regs = genClockRegs_25MHz,
        .numRegs = (uint32_t)(sizeof(genClockRegs_25MHz) / sizeof(genClockRegs_25MHz[0])),
    };

    DevBlkCDII2CReg genClockRegs_24MHz[] = {
        {0x0003, 0x03},
        {0x0006, 0xB0},
        {0x03F0, 0x59},
        {0x0570, 0x0C},
    };
    DevBlkCDII2CRegList genClock_24MHz = {
        .regs = genClockRegs_24MHz,
        .numRegs = (uint32_t)(sizeof(genClockRegs_24MHz) / sizeof(genClockRegs_24MHz[0])),
    };

    _DriverHandle *drvHandle = getHandlePriv(handle);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    LOG_DBG("Max96717F: Generate Clock\n");
    if ((INCK_24MHZ != freq) && (INCK_25MHZ != freq)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Not supported clock rate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        if (INCK_24MHZ == freq) {
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &genClock_24MHz);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GenerateClock: GetClock 24MHz failed :", (uint32_t)status);
            }
        } else {
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &genClock_25MHz);
            if (NVMEDIA_STATUS_OK != status) {
                SIPL_LOG_ERR_STR_HEX_UINT("GenerateClock: GetClock 25MHz failed :", (uint32_t)status);
            }
        }

        if (status == NVMEDIA_STATUS_OK) {
            /* Check DPLL Lock status here */
            if (IsDPLLLock(drvHandle->i2cProgrammer, DPLL_LOCK_CHECK_RETRY_CNT)) {
                status = NVMEDIA_STATUS_OK;
            } else {
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

static NvMediaStatus
EnablePClkPIOSlew(
    DevBlkCDIDevice *handle)
{
    DevBlkCDII2CReg const setLinkMapsRegs[] = {
        {0x0010, 0x01},
    };
    DevBlkCDII2CRegList setLinkMaps = {
        .regs = setLinkMapsRegs,
        .numRegs = (uint32_t)(sizeof(setLinkMapsRegs) /
        sizeof(setLinkMapsRegs[0])),
    };

    DevBlkCDII2CReg pioPClkSlewRegs[] = {
        {0x0570, 0x00},
        {0x03F1, 0x89},
        {0x056F, 0x00},
    };
    DevBlkCDII2CRegList pioPClkSlew = {
        .regs = pioPClkSlewRegs,
        .numRegs = I2C_ARRAY_SIZE(pioPClkSlewRegs),
    };

    DevBlkCDII2CReg pipeZRaw12Regs[] = {
        {0x0318, 0x6C},
    };
    DevBlkCDII2CRegList pipeZRaw12 = {
        .regs = pipeZRaw12Regs,
        .numRegs = I2C_ARRAY_SIZE(pipeZRaw12Regs),
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to EnablePClkPIOSlew");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to EnablePClkPIOSlew");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setLinkMaps);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_HEX_UINT("Error configuring PHY Links failed", (uint32_t)status);
        goto done;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &pioPClkSlew);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Error configuration pioPClkSle", (int32_t)status);
        goto done;
    }

    LOG_DBG("Max96717F: Sleep 128ms\n");
    nvsleep(128000);
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &pipeZRaw12);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Error pipZRaw12 configuration", (int32_t)status);
    }

done:
    return status;
}

static NvMediaStatus
SetGPIOOutput(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX96717F gpio,
    bool level)
{
    DevBlkCDII2CReg setGPIOAModeRegs[] = {
        {0x02BE, 0x80},
    };
    DevBlkCDII2CRegList setGPIOAMode = {
        .regs = setGPIOAModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOAModeRegs),
    };

    DevBlkCDII2CReg setGPIOBModeRegs[] = {
        {0x02BF, 0x63},
    };
    DevBlkCDII2CRegList setGPIOBMode = {
        .regs = setGPIOBModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOBModeRegs),
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to SetGPIOOutput");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to SetGPIOOutput");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) || (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    setGPIOAModeRegs[0].address += ((uint8_t) gpio - 1U) * 3U;
    setGPIOBModeRegs[0].address += ((uint8_t) gpio - 1U) * 3U;

    LOG_DBG("Max96717F: Release Reset\n");
    LOG_DBG("Max96717F: 0x%04X - 0x%02X\n", setGPIOAModeRegs[0].address, setGPIOAModeRegs[0].data);
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setGPIOBMode);

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Set GPIOBMode failed", (int32_t)status);
        goto done;
    }

    if (level) {
        setGPIOAModeRegs[0].data |= (1 << 4);
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setGPIOAMode);

done:
    return status;
}

static NvMediaStatus
SetFsyncGPIO(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX96717F gpio,
    uint8_t rxID)
{
    DevBlkCDII2CReg setGPIOModeRegs[] = {
        {0x02BE, 0x84},
        {0x02BF, 0xA0},
        {0x02C0, 0x42},
    };
    DevBlkCDII2CRegList setGPIOMode = {
        .regs = setGPIOModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOModeRegs),
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to SetFsyncGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to SetFsyncGPIO");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) ||
        (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    setGPIOModeRegs[0].address += ((uint8_t) gpio - 1U) * 3U;
    setGPIOModeRegs[1].address += ((uint8_t) gpio - 1U) * 3U;
    setGPIOModeRegs[2].address += ((uint8_t) gpio - 1U) * 3U;
    setGPIOModeRegs[1].data |= rxID;
    LOG_DBG("WriteArray 0x%04x 0x%x\n", setGPIOModeRegs[0].address, setGPIOModeRegs[0].data);
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setGPIOMode);

    // sleep 10ms
    nvsleep(10000);
done:
    return status;
}
static NvMediaStatus
EnableRefClock(
    DevBlkCDIDevice *handle,
    GPIOTypeMAX96717F gpio,
    bool enableRClk)
{
    DevBlkCDII2CReg enablePCLKOutRegs[] = {
        {0x03F1, 0x00},
    };
    DevBlkCDII2CRegList enablePCLKOut = {
        .regs = enablePCLKOutRegs,
        .numRegs = I2C_ARRAY_SIZE(enablePCLKOutRegs),
    };

    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to EnableRefClock");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to EnableRefClock");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if ((gpio == CDI_MAX96717F_GPIO_TYPE_INVALID) ||
        (gpio >= CDI_MAX96717F_GPIO_TYPE_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid GPIO pin");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (enableRClk) {
        enablePCLKOutRegs[0].data |= 0x80;
    }

    enablePCLKOutRegs[0].data |= ((((uint8_t) gpio - 1U) << 1) | 0x1);
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &enablePCLKOut);

done:
    return status;
}

static NvMediaStatus
ForwardGPIO(
    DevBlkCDIDevice *handle,
    uint8_t srcGpio,
    uint8_t dstGpio)
{
    DevBlkCDII2CReg setGPIOModeRegs[] = {
        {0x02BE, 0x1C},
    };
    DevBlkCDII2CRegList setGPIOMode = {
        .regs = setGPIOModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setGPIOModeRegs),
    };

    uint8_t regVal = 0U;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to ForwardGPIO");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to ForwardGPIO");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (srcGpio > 10U ) {
        SIPL_LOG_ERR_STR("MAX96717F: srcGpio too high");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dstGpio > 31U) {
        SIPL_LOG_ERR_STR("MAX96717F: dstGpio too high");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    setGPIOModeRegs[0].address += (srcGpio * 3u);

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       setGPIOModeRegs[0].address,
                                       &regVal);

    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    regVal |= 0x3; /* Set GPIO_TX_EN, GPIO_OUT_DIS */
    regVal &= ~(1 << 2); /* Unset GPIO_RX_EN */
    setGPIOModeRegs[0].data = regVal;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Update the offset from GPIO A to GPIO B */
    setGPIOModeRegs[0].address += 1;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       setGPIOModeRegs[0].address,
                                       &regVal);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    regVal &= 0xE0U;
    regVal |= (dstGpio & 0x1FU);
    setGPIOModeRegs[0].data = regVal;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &setGPIOMode);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Failed to write to GPIO mode register", (int32_t)status);
    }

    return status;
}


static NvMediaStatus
GetRevId(
    DevBlkCDIDevice  *handle,
    uint8_t devID,
    RevisionMAX96717F *rev)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint32_t numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);
    uint8_t revision = 0U;
    uint32_t i = 0U;

    status = MAX96717FReadRegister(handle,
                                   REG_DEV_REV_ADDR,
                                   1,
                                   &revision);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    revision &= 0x0F;

    for (i = 0U; i < numRev; i++) {
        if (revision == supportedRevisions[i].revVal) {
            *rev = supportedRevisions[i].revId;
            LOG_MSG("%s: Revision %u detected!\n",
                (devID == MAX96717F_DEV_ID) ? "MAX96717F" : "MAX96717", revision);
            return NVMEDIA_STATUS_OK;
        }
    }

    SIPL_LOG_ERR_STR_HEX_UINT("MAX96717F: Unsupported MAX96717F revision detected", (uint32_t)revision);
    LOG_MSG("MAX96717F: Unsupported MAX96717F revision %u detected!\n"
            "Supported revisions are:", revision);
    for (i = 0u; i < numRev; i++) {
        LOG_MSG("MAX96717F: Revision %u\n", supportedRevisions[i].revVal);
    }

    return NVMEDIA_STATUS_NOT_SUPPORTED;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to DriverCreate");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (clientContext != NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Context must not be supplied");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    // allocation MAX96717F drvHandle
    drvHandle = calloc(1, sizeof(*drvHandle));
    if (NULL == drvHandle) {
        status =  NVMEDIA_STATUS_OUT_OF_MEMORY;
        SIPL_LOG_ERR_STR("MAX96717F: Failed to allocate storage for driver handle");
        goto done;
    }

    handle->deviceDriverHandle = (void *)drvHandle;

    /* Create the I2C programmer for register read/write */
    drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                      MAX96717F_NUM_ADDR_BYTES,
                                                      MAX96717F_NUM_DATA_BYTES);

    if (drvHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Failed to initialize the I2C Programmer");
        status = NVMEDIA_STATUS_ERROR;
    }

done:
    return status;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C Programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    free(handle->deviceDriverHandle);
    handle->deviceDriverHandle = NULL;

    return NVMEDIA_STATUS_OK;
}
NvMediaStatus
MAX96717FSetDefaults(
    DevBlkCDIDevice *handle)
{
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FSetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96717FReadRegister(
    DevBlkCDIDevice *handle,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FReadRegister");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to MAX96717FReadRegister");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           (registerNum+i),
                                           &dataBuff[i]);
        if (NVMEDIA_STATUS_OK != status) {
            SIPL_LOG_ERR_STR_2UINT("MAX96717F: MAX96717F read register failed", (uint32_t)(registerNum+i), (uint32_t)status);
        }
    }

done:
    return status;
}

NvMediaStatus
MAX96717FWriteRegister(
    DevBlkCDIDevice *handle,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FWriteRegister");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to MAX96717FWriteRegister");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (dataLength > REG_WRITE_BUFFER_BYTES) {
        SIPL_LOG_ERR_STR("MAX96717F: Buffer length passed to MAX96717FWriteRegister exceeds buffer size");
        goto done;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            (registerNum + i),
                                            dataBuff[i]);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX96717F: Failed to write to register with status", (int32_t)registerNum, (int32_t)status);
        }
    }

done:
    return status;
}

NvMediaStatus
MAX96717FWriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    bool isValidSize = false;
    ReadWriteParamsMAX96717F *param = (ReadWriteParamsMAX96717F *) parameter;
    static const char *cmdString[] = {
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A",
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B",
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS",
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO",
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT",
        [CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK",
        [CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES",
        [CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY",
        [CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK",
        [CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS",
        [CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW",
        [CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD] =
            "CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FWriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null paramter storage passed to MAX96717FWriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_WRITE_PARAM_CMD_MAX96717F_INVALID) ||
        (parameterType >= CDI_WRITE_PARAM_CMD_MAX96717F_NUM)) {
        SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    LOG_DBG("MAX96717F: %s", cmdString[parameterType]);
    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_A:
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_TRANSLATOR_B:
            if (parameterSize == sizeof(param->Translator)) {
                isValidSize = true;
                status = SetTranslator(handle,
                                       parameterType,
                                       param->Translator.source,
                                       param->Translator.destination);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEVICE_ADDRESS:
            if (parameterSize == sizeof(param->DeviceAddress)) {
                isValidSize = true;
                status = SetDeviceAddress(handle,
                                          param->DeviceAddress.address);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_GPIO_OUTPUT:
            if (parameterSize == sizeof(param->GPIOOutp)) {
                isValidSize = true;
                status = SetGPIOOutput(handle,
                                       param->GPIOOutp.gpioInd,
                                       param->GPIOOutp.level);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_FSYNC_GPIO:
            if (parameterSize == sizeof(param->FSyncGPIO)) {
                isValidSize = true;
                status = SetFsyncGPIO(handle,
                                       param->FSyncGPIO.gpioInd,
                                       param->FSyncGPIO.rxID);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_SET_DEBUG_REGS:
            isValidSize = true;
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_PCLK_PIO_SLEW:
            isValidSize = true;
            status = EnablePClkPIOSlew(handle);
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_ENABLE_REF_CLOCK:
            if (parameterSize == sizeof(param->RefClkGPIO)) {
                isValidSize = true;
                status = EnableRefClock(handle,
                                        param->RefClkGPIO.gpioInd,
                                        param->RefClkGPIO.enableRClk);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_GPIO_FORWARD:
            if (parameterSize == sizeof(param->GPIOForward)) {
                isValidSize = true;
                status = ForwardGPIO(handle,
                                     param->GPIOForward.srcGpio,
                                     param->GPIOForward.dstGpio);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_VIDEO_PIPELINES:
            if (parameterSize == sizeof(param->ConfigVideoPipeline)) {
                isValidSize = true;
                status = ConfigPipelines(handle,
                                         param->ConfigVideoPipeline.dataType,
                                         param->ConfigVideoPipeline.embDataType);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_CONFIG_PHY:
            if (parameterSize == sizeof(param->ConfigPhy)) {
                isValidSize = true;
                status = ConfigPhyMap(handle,
                                      &param->ConfigPhy.mapping,
                                      param->ConfigPhy.numDataLanes);
            };
            break;
        case CDI_WRITE_PARAM_CMD_MAX96717F_GENERATE_CLOCK:
            if (parameterSize == sizeof(param->ClockRate)) {
                isValidSize = true;
                status = GenerateClock(handle,
                                       param->ClockRate.freq);
            };
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96717F: Bad parameter: Unrecognized command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX96717F: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("MAX96717F: Bad parameter: Invalid param size", cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

NvMediaStatus
MAX96717FDumpRegisters(
    DevBlkCDIDevice *handle)
{
    uint16_t address = 0U;
    uint8_t regVal = 0U;
    _DriverHandle *drvHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FDumpRegisters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to MAX96717FDumpRegisters");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

     for (uint32_t i = 0U; i <= MAX96717F_REG_MAX_ADDRESS; i++) {
         address = (i / 256U) << 8 | (i%256U);
         status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                            address,
                                            &regVal);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX96717F: Register I2C read failed with status", (int32_t)i, (int32_t)status);
            return status;
        }
        LOG_MSG("Max96717F: address: 0x%04x - data: 0x%02X\n", address, regVal);
    }

done:
    return status;
}

NvMediaStatus
MAX96717FCheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    RevisionMAX96717F rev = CDI_MAX96717F_INVALID_REV;
    uint8_t devID = 0U;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FCheckPresence");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    status = MAX96717FReadRegister(handle,
                                   REG_DEV_ID_ADDR,
                                   1,
                                   &devID);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    if ((devID != MAX96717F_DEV_ID) &&
        (devID != MAX96717_DEV_ID)) {
        SIPL_LOG_ERR_STR_2UINT("MAX96717F: Device ID mismatch", (uint32_t)MAX96717F_DEV_ID, (uint32_t)devID);
        return NVMEDIA_STATUS_ERROR;
    }

    status = GetRevId(handle,
                      devID,
                      &rev);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96717FGetTemperature(
    DevBlkCDIDevice *handle,
    float_t *temperature)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t temperatureEst[2] = {0U};
    uint16_t tmp = 0U;
    DevBlkCDII2CReg temperatureWrRegs[] = {
        {0x0535, 0x80},
        {0x0502, 0x00},
        {0x0500, 0x00},
        {0x0501, 0xB8},
        {0x050C, 0x83},
        {0x0500, 0x1E},
        {0x050C, 0x81},
        {0x0502, 0x00},
        {0x050C, 0x81},
        {0x1D28, 0x01},
    };
    DevBlkCDII2CRegList temperatureWrite = {
        .regs = temperatureWrRegs,
        .numRegs = I2C_ARRAY_SIZE(temperatureWrRegs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null handle passed to MAX96717FGetTemperature");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    if (temperature == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null temperature passed to MAX96717FGetTemperature");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = getHandlePriv(handle);
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96717F: Null driver handle passed to MAX96717FGetTemperature");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Write registers before reading */
    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &temperatureWrite);
    if (NVMEDIA_STATUS_OK != status) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Failed to write to temperatur control registers", (int32_t)status);
        goto done;
    }

    /* sleep for 0.5ms before reading registers */
    nvsleep(500);

    /* Read ADC registers */
    status = MAX96717FReadRegister(handle,
                                   REG_ADCBIST13,
                                   2,
                                   temperatureEst);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96717F: Read register REG_ADCBIST13 & REG_ADCBIST14 failed", (int32_t)status);
        goto done;
    }

    /* Combine the values from 2 registers to 10-bit ADC value*/
    tmp = (temperatureEst[1] & 0xC0);
    tmp = tmp << 2;
    tmp |= temperatureEst[0];

    /* Convert temperature from Kelvin to Celsius
     *   - Dividing 10-bit ADC value by 2 to get Kelvin value
     *   - Subtracting 273 to convert it to Celsius
     */
    tmp = (tmp >> 1);
    *temperature = KELVIN_TO_CELSIUS(tmp);

done:
    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 96717F Serializer",
    .regLength = MAX96717F_NUM_ADDR_BYTES,
    .dataLength = MAX96717F_NUM_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetMAX96717FDriver(
    void)
{
    return &deviceDriver;
}
