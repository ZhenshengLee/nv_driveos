/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <unistd.h>
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "cdi_max96712.h"
#include "cdi_max96712_priv.h"
#include "cdi_max96712_pg_setting.h"
#include "os_common.h"

static NvMediaStatus
EnableReplication(
    DevBlkCDIDevice *handle,
    bool enable);

/*
 * The following pointers may be used in the functions local to this file but they are checked for
 * NULL in the entry points for CDI functions.
 * DevBlkCDIDevice *handle
 */

static bool
IsGMSL2Mode(const GMSLModeMAX96712 mode)
{
    if ((mode == CDI_MAX96712_GMSL2_MODE_6GBPS) ||
        (mode == CDI_MAX96712_GMSL2_MODE_3GBPS)) {
        return true;
    } else {
        return false;
    }
}

static NvMediaStatus
AddToRegFieldQ(
    DevBlkCDIDevice *handle,
    RegBitField name,
    uint8_t val)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t index = drvHandle->regBitFieldQ.numRegBitFieldArgs;

    if (index == MAX96712_REG_MAX_FIELDS_PER_REG) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: RegFieldQ full. Failed to add", (uint32_t)name);
        return NVMEDIA_STATUS_ERROR;
    }

    if (name >= REG_FIELD_MAX) {
        SIPL_LOG_ERR_STR_INT("MAX96712: RegFieldQ name over max. Failed to add", (uint32_t)name);
        return NVMEDIA_STATUS_ERROR;
    }

    LOG_DBG("MAX96712: Adding regField = %u, val = %u to index %u in RegFieldQ\n",
            name,
            val,
            index);

    drvHandle->regBitFieldQ.name[index] = name;
    drvHandle->regBitFieldQ.val[index] = val;
    drvHandle->regBitFieldQ.numRegBitFieldArgs = index + 1u;
    return NVMEDIA_STATUS_OK;
}

static void
ClearRegFieldQ(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    LOG_DBG("MAX96712: Clearing RegFieldQ");
    drvHandle->regBitFieldQ.numRegBitFieldArgs = 0u;
}

static uint8_t
ReadFromRegFieldQ(
    DevBlkCDIDevice *handle,
    uint8_t index)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t val = 0u;

    if (index >= drvHandle->regBitFieldQ.numRegBitFieldArgs) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Bad parameter. Invalid index", (uint32_t)index);
        return 0u;
    }

    val = drvHandle->regBitFieldQ.val[index];

    LOG_DBG("MAX96712: Read index %u from RegFieldQ. Val = %u", index, val);
    return val;
}

/* Access register fields belong to a single register.
 * REG_READ_MODE: Register is read and specified field vals are unpacked into regBitFieldArg array.
 * REG_WRITE_MODE: Specified field vals from regBitFieldArg array are packed and written to register.
 * REG_READ_MOD_WRITE_MODE: Register is read, specified field vals in regBitFieldArg are modified
 *                          and written to register */
static NvMediaStatus
sSingleRegAccessRegFieldQ(
    DevBlkCDIDevice *handle,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    RegBitFieldQ *regBit = &(drvHandle->regBitFieldQ);
    uint8_t numFields = regBit->numRegBitFieldArgs;
    uint16_t regAddr = 0u;
    const RegBitFieldProp *regBitProp = NULL;
    uint8_t fieldMask = 0u;
    uint8_t regData = 0u;
    uint8_t i = 0u;
    uint8_t loop = 0u;

    if (numFields == 0u) {
        LOG_DBG("MAX96712: Skipping sAccessRegFieldQ");
        goto done;
    }

    regBitProp = &regBitFieldProps[regBit->name[0]];
    regAddr = regBitProp->regAddr;

    /* Check if msbPos and lsbPos are valid. */
    for (i = 0u; i < numFields; i++) {
        regBitProp = &regBitFieldProps[regBit->name[i]];
        if (regBitProp->lsbPos > regBitProp->msbPos) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            goto done;
        }
    }

    if (mode == REG_READ_MOD_WRITE_MODE) {
        for (loop = 0u; loop < 10u; loop++) {
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               regAddr,
                                               &regData);
            if (status == NVMEDIA_STATUS_OK) {
                break;
            }
            nvsleep(10);
        }

        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX96712: Register I2C read failed with status", regAddr, status);
            goto done;
        }
     }

    for (i = 0u; i < numFields; i++) {
        regBitProp = &regBitFieldProps[regBit->name[i]];
        fieldMask = (1u << (regBitProp->msbPos + 1u)) - (1u << regBitProp->lsbPos);
        /* Pack fieldVals for write*/
        regData &= ~fieldMask;
        regData |= ((regBit->val[i] << regBitProp->lsbPos) & fieldMask);
    }

    for (loop = 0u; loop < 10u; loop++) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            regAddr,
                                            regData);
        if (status == NVMEDIA_STATUS_OK) {
            break;
        }
        nvsleep(10);
    }

    if (regBitProp->delayNS != 0) {
        nvsleep(regBitProp->delayNS);
    } else {
        nvsleep(20);
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("MAX96712: Register I2C write failed with status", regAddr, status);
    }

done:
    return status;
}


/*
 * Check whether all register fields belong to the same register.
 */
static bool
IsSingleRegister(
    RegBitFieldQ *regBit,
    uint8_t numFields)
{
    bool status = true;
    const RegBitFieldProp *regBitProp = NULL;
    uint16_t regAddr = 0u;
    uint8_t i;

    regBitProp = &regBitFieldProps[regBit->name[0]];
    regAddr = regBitProp->regAddr;

    for (i = 0u; i < numFields; i++) {
        regBitProp = &regBitFieldProps[regBit->name[i]];
        if (regBitProp->regAddr != regAddr) {
            status = false;
            goto done;
        }
    }

done:
    return status;
}

/* Access register fields.
 * REG_READ_MODE: Register is read and specified field vals are unpacked into regBitFieldArg array.
 * REG_WRITE_MODE: Specified field vals from regBitFieldArg array are packed and written to register.
 * REG_READ_MOD_WRITE_MODE: Register is read, specified field vals in regBitFieldArg are modified
 *                          and written to register */
static NvMediaStatus
sAccessRegFieldQ(
    DevBlkCDIDevice *handle,
    RegBitFieldAccessMode mode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    RegBitFieldQ *regBit = &(drvHandle->regBitFieldQ);
    uint8_t numFields = regBit->numRegBitFieldArgs;
    uint16_t regAddr = regBitFieldProps[regBit->name[0]].regAddr;
    const RegBitFieldProp *regBitProp = NULL;
    uint8_t fieldMask = 0u;
    uint8_t regData = 0;
    uint8_t i = 0u;
    uint8_t loop = 0u;

    if (numFields == 0u) {
        LOG_DBG("MAX96712: Skipping sAccessRegFieldQ");
        return status;
    }

    /*
     * use sSingleRegAccessRegFieldQ() if all register fields belong to
     * a single register
     */
    if (IsSingleRegister(regBit, numFields) &&
        ((mode == REG_WRITE_MODE) ||
        (mode == REG_READ_MOD_WRITE_MODE))) {
        return sSingleRegAccessRegFieldQ(handle, mode);
    }

    /* Check if all the supplied fields belongs to same register addr.
     * Check if msbPos and lsbPos are valid. */
    for (i = 0u; i < numFields; i++) {
        regAddr = regBitFieldProps[regBit->name[i]].regAddr;
        regBitProp = &regBitFieldProps[regBit->name[i]];
        if ((regAddr != regBitProp->regAddr) ||
            (regBitProp->lsbPos > regBitProp->msbPos)) {
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: invalid register bit specification");
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }

        if ((mode == REG_READ_MODE) || (mode == REG_READ_MOD_WRITE_MODE)) {
            for (loop = 0u; loop < 10u; loop++) {
                status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                                   regAddr,
                                                   &regData);
                if (status == NVMEDIA_STATUS_OK) {
                    break;
                }
                nvsleep(10);
            }

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_2INT("MAX96712: I2C register read failed with status", regAddr, status);
                return status;
            }

            nvsleep(20);
        }

        regBitProp = &regBitFieldProps[regBit->name[i]];
        fieldMask = (1u << (regBitProp->msbPos + 1u)) - (1u << regBitProp->lsbPos);
        if (mode == REG_READ_MODE) {
            /* Unpack fieldVals */
            regBit->val[i] = ((regData & fieldMask) >> (regBitProp->lsbPos));
        } else {
            /* Pack fieldVals for write*/
            regData &= ~fieldMask;
            regData |= ((regBit->val[i] << regBitProp->lsbPos) & fieldMask);
        }

        if (mode != REG_READ_MODE) {
            for (loop = 0u; loop < 10u; loop++) {
                status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                    regAddr,
                                                    regData);
                if (status == NVMEDIA_STATUS_OK) {
                    break;
                }
                nvsleep(10);
            }

            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_2INT("MAX96712: I2C register write failed with status", regAddr, status);
                return status;
            }
            if (regBitProp->delayNS != 0) {
                nvsleep(regBitProp->delayNS);
            } else {
                nvsleep(20);
            }
        }
    }

    return status;
}

NvMediaStatus
MAX96712OneShotReset(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t i = 0U;

    for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
        if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL_MODE_UNUSED) {
            link &= ~(1 << i);
        }
    }

    if (link != CDI_MAX96712_LINK_NONE) {
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_RESET_ONESHOT, (uint8_t)link, REG_WRITE_MODE);

        nvsleep(100000);
    }

    return status;
}

static NvMediaStatus
EnableSpecificLinks(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t i = 0u;

    drvHandle->ctx.linkMask = link;

    /* Disable the link lock error report to avoid the false alarm */
    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_ENABLE_LOCK,
                                 0u,
                                 REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL_MODE_UNUSED) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_LINK_EN_A + i, 0u);
        } else {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_LINK_EN_A + i,
                                      MAX96712_IS_GMSL_LINK_SET(link, i) ? 1u : 0u);
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    /* Make sure the link is locked properly before enabling the link lock signal */
    for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) {
            if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                /* HIM mode is not enabled yet so the link lock will not be set
                 * Instead use sleep function */
                nvsleep(100000);
            } else {
                status = MAX96712CheckLink(handle, drvHandle->ctx.linkMask, CDI_MAX96712_LINK_LOCK_GMSL2, true);
            }
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    /* Enable the link lock error report */
    if (drvHandle->ctx.linkMask != CDI_MAX96712_LINK_NONE) {
        ClearRegFieldQ(handle);
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_ENABLE_LOCK,
                                     1u,
                                     REG_READ_MOD_WRITE_MODE);
    }

    return status;
}

static NvMediaStatus
SetLinkMode(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t i = 0u;

    /* Set GMSL mode */
    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            LOG_DBG("SetLinkMode: Setting Link for %d", i);
            if (drvHandle->ctx.gmslMode[i] != CDI_MAX96712_GMSL_MODE_UNUSED) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_LINK_GMSL2_A + i,
                                          (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) ? 0u : 1u);
            }
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    /* Set Link speed */
    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            LOG_DBG("SetLinkMode: Link is set and now setting the speed for %i", i);
            /*CDI_MAX96712_GMSL1_MODE     : 1
            CDI_MAX96712_GMSL2_MODE_6GBPS : 2
            CDI_MAX96712_GMSL2_MODE_3GBPS : 1*/
            if (drvHandle->ctx.gmslMode[i] != CDI_MAX96712_GMSL_MODE_UNUSED) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_RX_RATE_PHY_A + i,
                                          (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL2_MODE_6GBPS) ? 2u : 1u);
            }
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    return status;
}

static NvMediaStatus
EnablePeriodicAEQ(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ClearRegFieldQ(handle);
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_AEQ_PHY_A + i,
                                      1u);
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_PERIODIC_AEQ_PHY_A + i,
                                      1u);
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_EOM_PER_THR_PHY_A + i,
                                      0x10);
            ACCESS_REG_FIELD_RET_ERR(REG_WRITE_MODE);

            LOG_MSG("MAX96712: Enable periodic AEQ on Link %d\n", i);
            nvsleep(10000);
        }
    }

    return status;
}

static NvMediaStatus
SetDefaultGMSL1HIMEnabled(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    uint8_t step)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg max96712_defaults_HIM_step0_regs[] = {
        /* GMSL1 - Turn on HIM */
        {0x0B06, 0xEF},
        /* GMSL1 - Enable reverse channel cfg and turn on local I2C ack */
        {0x0B0D, 0x81},
    };
    DevBlkCDII2CRegList max96712_defaults_HIM_step0 = {
        .regs = max96712_defaults_HIM_step0_regs,
        .numRegs = I2C_ARRAY_SIZE(max96712_defaults_HIM_step0_regs),
    };
    DevBlkCDII2CReg max96712_defaults_HIM_step1_regs[] = {
        /* GMSL1 - Turn off HIM */
        {0x0B06, 0x6F},
        /* GMSL1 - Enable manual override of reverse channel pulse length */
        {0x14C5, 0xAA},
        /* GMSL1 - Enable manual override of reverse channel rise fall time setting */
        {0x14C4, 0x80},
        /* GMSL1 - Tx amplitude manual override */
        {0x1495, 0xC8},
    };
    DevBlkCDII2CRegList max96712_defaults_HIM_step1 = {
        .regs = max96712_defaults_HIM_step1_regs,
        .numRegs = I2C_ARRAY_SIZE(max96712_defaults_HIM_step1_regs),
    };
    DevBlkCDII2CReg max96712_defaults_HIM_step2_regs[] = {
        /* Enable HIM */
        {0x0B06, 0xEF},
        /* Manual override of reverse channel pulse length */
        {0x14C5, 0x40},
        /* Manual override of reverse channel rise fall time setting */
        {0x14C4, 0x40},
        /* TxAmp manual override */
        {0x1495, 0x69},
    };
    DevBlkCDII2CRegList max96712_defaults_HIM_step2 = {
        .regs = max96712_defaults_HIM_step2_regs,
        .numRegs = I2C_ARRAY_SIZE(max96712_defaults_HIM_step2_regs),
    };
    DevBlkCDII2CRegList *stepHIM = NULL;
    uint8_t i = 0u;

    if (step > 2u) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter. Step must be either 0, 1 or 2.");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            /* Update register offset */
            if (step == 0) {
                max96712_defaults_HIM_step0_regs[0].address += (i << 8);
                max96712_defaults_HIM_step0_regs[1].address += (i << 8);
            } else if (step == 1) {
                max96712_defaults_HIM_step1_regs[0].address += (i << 8);
                max96712_defaults_HIM_step1_regs[1].address += (i << 8);
                max96712_defaults_HIM_step1_regs[2].address += (i << 8);
                max96712_defaults_HIM_step1_regs[3].address += (i << 8);
            } else {
                max96712_defaults_HIM_step2_regs[0].address += (i << 8);
                max96712_defaults_HIM_step2_regs[1].address += (i << 8);
                max96712_defaults_HIM_step2_regs[2].address += (i << 8);
                max96712_defaults_HIM_step2_regs[3].address += (i << 8);
            }

            stepHIM = (step == 0) ? &max96712_defaults_HIM_step0 :
                      ((step == 1) ? &max96712_defaults_HIM_step1 : &max96712_defaults_HIM_step2);

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, stepHIM);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    return status;
}

static NvMediaStatus
EnablePacketBasedControlChannel(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg ctrlChannelReg = {0x0B08, 0x25};
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ctrlChannelReg.address += (i << 8);

            if (!enable) {
                ctrlChannelReg.data = 0x21;
            }

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                ctrlChannelReg.address,
                                                ctrlChannelReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
            nvsleep(10000);
        }
    }

    return status;
}

static NvMediaStatus
EnableDoublePixelModeDoublePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    DataTypeMAX96712 dataType,
    const bool embDataType)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    TxPortMAX96712 txPort;
    DevBlkCDII2CReg disPktDetectorReg = {0x0100, 0x13};
    DevBlkCDII2CReg altModeArrReg = {0x0933, 0x01}; /* ALT_MEM_MAP12 = 1 on Ctrl 0 */
    uint8_t i = 0u;

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    txPort = drvHandle->ctx.txPort;

    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_BPP8DBL_4 + i,
                                      1u);
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_BPP8DBL_MODE_4 + i,
                                      1u);
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    altModeArrReg.address += (txPort - CDI_MAX96712_TXPORT_PHY_C) * 0x40;

    ClearRegFieldQ(handle);

    if (dataType == CDI_MAX96712_DATA_TYPE_RAW12) {
        ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP12_PHY0 + txPort, 1u);
    } else if (dataType == CDI_MAX96712_DATA_TYPE_RAW10) {
        ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP10_PHY0 + txPort, 1u);
    }

    if (embDataType == true) {
        ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP8_PHY0 + txPort, 1u);
    }

    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            disPktDetectorReg.address =
                ((disPktDetectorReg.address & 0xFF00U) + (0x12U * i));
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                disPktDetectorReg.address,
                                                disPktDetectorReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            disPktDetectorReg.address =
                (disPktDetectorReg.address & 0xFF00U) +
                (0x12U * (i + 4U)) +
                ((i != 0U) ? 0x6U : 0U);
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                disPktDetectorReg.address,
                                                disPktDetectorReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    return status;
}

static NvMediaStatus
EnableDoublePixelModeSinglePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    DataTypeMAX96712 dataType,
    const bool embDataType)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    DevBlkCDII2CReg disPktDetectorRegs[] = {
        {0x0100U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0112U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0124U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0136U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0148U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0160U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0172U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
        {0x0184U, 0x22U},    /* VIDEO_RX0 SEQ_MISS_EN = 1, DIS_PKT_DET = 0 */
    };

    DevBlkCDII2CReg disLIMHeartRegs[] = {
        {0x0106U, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x0118U, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x012AU, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x013CU, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x014EU, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x0166U, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x0178U, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
        {0x018AU, 0x0AU},    /* VIDEO_RX6 LIM_HEART = 1 : Disable */
    };
    uint8_t i = 0u;

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (dataType == CDI_MAX96712_DATA_TYPE_RAW12) {
            if (embDataType) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT2_MEM_MAP8_PHY0 + i, 1u);
            } else {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP12_PHY0 + i, 1u);
            }
        } else if (dataType == CDI_MAX96712_DATA_TYPE_RAW10) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP10_PHY0 + i, 1u);

            if (embDataType == true) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ALT_MEM_MAP8_PHY0 + i, 1u);
            }
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            if (embDataType == true) {
                disPktDetectorRegs[i].data = 0x23U;
                disPktDetectorRegs[i + 4U].data = 0x23U;
            }
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                disPktDetectorRegs[i].address,
                                                (uint8_t)disPktDetectorRegs[i].data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            if (drvHandle->ctx.camRecCfg > 0U) {
                status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                    disPktDetectorRegs[i + 4U].address,
                                                    (uint8_t)disPktDetectorRegs[i + 4U].data);
                if (status != NVMEDIA_STATUS_OK) {
                    return status;
                }
            }

            if (embDataType == false) {
                status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                    disLIMHeartRegs[i].address,
                                                    (uint8_t)disLIMHeartRegs[i].data);
                if (status != NVMEDIA_STATUS_OK) {
                    return status;
                }

                if (drvHandle->ctx.camRecCfg > 0U) {
                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        disLIMHeartRegs[i + 4U].address,
                                                        (uint8_t)disLIMHeartRegs[i + 4U].data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
EnableDoublePixelMode(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    DataTypeMAX96712 dataType,
    const bool embDataType,
    bool isSharedPipeline)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (isSharedPipeline) {
        status = EnableDoublePixelModeSinglePipe(handle, link, dataType, embDataType);
    } else {
        status = EnableDoublePixelModeDoublePipe(handle, link, dataType, embDataType);
    }

    return status;
}

/*
 * Bug 2182451: The below errors were observed intermittently in GMSL2 6Gbps link speed.
 *              To resolve it, adjust the Tx amplitude and timing parameters
 * CSI error(short or long line) is seen
 * Decoding error is seen on the deserializer
 * Link margin becomes bad
 */
static NvMediaStatus
ConfigTxAmpTiming(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    RevisionMAX96712 rev = drvHandle->ctx.revision;
    DevBlkCDII2CReg adjTxAmpAndTimingArrRegs[] = {
        {0x1458, 0x28, 0x2701}, /* vth1 : Error channel power-down then delay 1ms*/
        {0x1459, 0x68, 0x2701},/* vth0 : + 104 * 4.7mV = 488.8 mV  then delay 1ms*/
        {0x143E, 0xB3, 0x2701},/* Error channel phase secondary timing adjustment  then delay 1ms*/
        {0x143F, 0x72, 0x2701}, /* Error channel phase primary timing adjustment  then delay 1ms*/
        {0x1495, 0xD2, 0x2701}, /* Reverse channel Tx amplitude to 180 mV  then delay 1ms*/
    };
    DevBlkCDII2CRegList adjTxAmpAndTimingArr = {
        .regs = adjTxAmpAndTimingArrRegs,
        .numRegs = I2C_ARRAY_SIZE(adjTxAmpAndTimingArrRegs),
    };
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            gmslMode = drvHandle->ctx.gmslMode[i];

            if (!IsGMSL2Mode(gmslMode)) {
                LOG_INFO("MAX96712: Link %d: Tx amplitude is only required in GMSL2 mode\n", i);
                continue;
            }

            adjTxAmpAndTimingArrRegs[0].address += (i << 8);
            adjTxAmpAndTimingArrRegs[1].address += (i << 8);
            adjTxAmpAndTimingArrRegs[2].address += (i << 8);
            adjTxAmpAndTimingArrRegs[3].address += (i << 8);
            adjTxAmpAndTimingArrRegs[4].address += (i << 8);
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &adjTxAmpAndTimingArr);
            if (status != NVMEDIA_STATUS_OK) {
                LOG_INFO("MAX96712: Link %d: Failed to updte Tx amplitude\n", i);
                return status;
            }
            (void)rev;
            LOG_MSG("MAX96712 Rev %d: Link %d: ", rev, i);
            LOG_MSG("Tx amplitude 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n", adjTxAmpAndTimingArrRegs[0].data,
                                                                             adjTxAmpAndTimingArrRegs[1].data,
                                                                             adjTxAmpAndTimingArrRegs[2].data,
                                                                             adjTxAmpAndTimingArrRegs[3].data,
                                                                             adjTxAmpAndTimingArrRegs[4].data);
        }
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
UpdateVGAHighGain(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg VGAHighGainReg = {0x1418, 0x03};
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            gmslMode = drvHandle->ctx.gmslMode[i];

            if (!IsGMSL2Mode(gmslMode)) {
                LOG_INFO("MAX96712: Link %d: VGAHighGain is valid in ONLY GMSL2 mode\n", i);
                continue;
            }
            VGAHighGainReg.address += (i << 8);
            VGAHighGainReg.data = (enable) ? 0x07 : 0x03;

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                VGAHighGainReg.address,
                                                VGAHighGainReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                LOG_INFO("MAX96712: Link %d: Failed to set VGAHighGain\n", i);
                return status;
            }
        }
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
OverrideDataType(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t bpp = 0u;
    uint8_t dataFormat = 0u;
    uint8_t i = 0u;

    /* Override is enabled only for pipes 0-3 */
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i) &&
            linkPipelineMap[i].isDTOverride) {
            switch (linkPipelineMap[i].dataType) {
                case CDI_MAX96712_DATA_TYPE_RAW10:
                    bpp = 0xA;         /* 10 bits per pixel */
                    dataFormat = 0x2B; /* raw10 */
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW12:
                    bpp = 0xC;         /* 12 bits per pixel */
                    dataFormat = 0x2C; /* raw12 */
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW16:
                    bpp = 0x10;        /* 16 bits per pixel */
                    dataFormat = 0x2E; /* raw16 */
                    break;
                case CDI_MAX96712_DATA_TYPE_RGB:
                    bpp = 0x18;        /* 24 bits per pixel */
                    dataFormat = 0x24; /* RGB */
                    break;
                case CDI_MAX96712_DATA_TYPE_YUV_8:
                    bpp = 0x10;        /* 16 bits per pixel */
                    dataFormat = 0x1E; /* YUV */
                    break;
                case CDI_MAX96712_DATA_TYPE_YUV_10:
                    bpp = 0x14;        /* 20 bits per pixel */
                    dataFormat = 0x1F; /* YUV */
                    break;
                default:
                    SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
                    return NVMEDIA_STATUS_BAD_PARAMETER;
            }

            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_0 + i,
                                         bpp,
                                         REG_READ_MOD_WRITE_MODE);

            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_0 + i,
                                         dataFormat,
                                         REG_READ_MOD_WRITE_MODE);


            if (i == 1u) {
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_1_H,
                                             (dataFormat >> 4u),
                                             REG_READ_MOD_WRITE_MODE);
            } else if (i == 2u) {
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_2_H,
                                             (bpp >> 2u),
                                             REG_READ_MOD_WRITE_MODE);

                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_2_H,
                                             (dataFormat >> 2u),
                                             REG_READ_MOD_WRITE_MODE);
            }

            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_OVR_0_EN + i,
                                         1u,
                                         REG_READ_MOD_WRITE_MODE);

            if (drvHandle->ctx.tpgEnabled &&
                drvHandle->ctx.pipelineEnabled & (0x10 << i)) {
                /* Override BPP, DT for the pipeline 4 ~ 7 */
                if (i == 0U) {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_4,
                                                 bpp,
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_4,
                                                 dataFormat,
                                                 REG_READ_MOD_WRITE_MODE);
                } else if (i == 1U) {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_5,
                                                 bpp,
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_5_H,
                                                 (dataFormat >> 4U),
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_5_L,
                                                 (dataFormat & 0xF),
                                                 REG_READ_MOD_WRITE_MODE);
                } else if (i == 2U) {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_6_H,
                                                 (bpp >> 2U),
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_6_L,
                                                 (bpp & 0x3),
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_6_H,
                                                 (dataFormat >> 2U),
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_6_L,
                                                 (dataFormat & 0x3),
                                                 REG_READ_MOD_WRITE_MODE);
                } else {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_BPP_7,
                                                 bpp,
                                                 REG_READ_MOD_WRITE_MODE);

                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_DT_7,
                                                 dataFormat,
                                                 REG_READ_MOD_WRITE_MODE);
                }

                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_OVR_0_EN + i,
                                             0u,
                                             REG_READ_MOD_WRITE_MODE);
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SOFT_OVR_4_EN + i,
                                             1u,
                                             REG_READ_MOD_WRITE_MODE);
            }
        }
    }

    return status;
}

static NvMediaStatus
VideoPipelineSelDoublePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t pipelineEnabled = 0U;
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            if (IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                if (linkPipelineMap[i].isSinglePipeline) {
                    /* in case of single pipe Z from ser, select that for pipe in deser */
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_SEL_0 + i,
                                                 (4u * i) + 2u,
                                                 REG_READ_MOD_WRITE_MODE);
                } else {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_SEL_0 + i,
                                                 4u * i,
                                                 REG_READ_MOD_WRITE_MODE);

                    if (linkPipelineMap[i].isEmbDataType) {
                        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_SEL_4 + i,
                                                     (4u * i) + 1u,
                                                     REG_READ_MOD_WRITE_MODE);
                    }
                }
            }
        }
    }

    /* Enable Pipelines*/
    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_0 + i, 1u);
            pipelineEnabled |= (1U << i);
            if (linkPipelineMap[i].isEmbDataType && !linkPipelineMap[i].isSinglePipeline) {
                pipelineEnabled |= (0x10U << i);
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_4 + i, 1u);
            }
        }
    }
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);
    drvHandle->ctx.pipelineEnabled |= pipelineEnabled;

    return status;
}

static NvMediaStatus
VideoPipelineSelSinglePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t pipelineEnabled = 0U;
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            if (IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                /* pipe Z from ser is connected to the pipe in deser
                 * Two different pipelines receives the data from the same camera
                 */
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_SEL_0 + i,
                                             (4u * i) + 2u,
                                             REG_READ_MOD_WRITE_MODE);
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_SEL_4 + i,
                                             (4u * i) + 2u,
                                             REG_READ_MOD_WRITE_MODE);
            }
        }
    }

    ClearRegFieldQ(handle);
    /* Enable Pipelines from 0 to 3 */
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_0 + i, 1u);
            pipelineEnabled |= (1U << i);
        }
    }
    /* Enable Pipelines from 4 to 7 */
    if ((drvHandle->ctx.camRecCfg > 0U) && (drvHandle->ctx.camRecCfg < 3U)) {
        for (i = 0U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_4 + i, 1u);
                pipelineEnabled |= (0x10U << i);
            }
        }
    } else if (drvHandle->ctx.camRecCfg == 3U) {
        for (i = 0U; i < 2U; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_4 + i, 1u);
                pipelineEnabled |= (0x10U << i);
            }
        }
    } else if (drvHandle->ctx.camRecCfg == 4U) {
        for (i = 2U; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_4 + i, 1u);
                pipelineEnabled |= (0x10U << i);
            }
        }
    }

    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);
    drvHandle->ctx.pipelineEnabled |= pipelineEnabled;

    return status;
}

static NvMediaStatus
SetPipelineMapSinglePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    TxPortMAX96712 txPort = drvHandle->ctx.txPort;
    /* Each camera's data is processed per link in the separate pipeline individually.
     * Camera on the link 0 --> Pipeline 0
     * Camera on the link 1 --> Pipeline 1
     * Camera on the link 2 --> Pipeline 2
     * Camera on the link 3 --> Pipeline 3
     * The same data is also processed in the different pipeline
     * Camera on the link 0 --> Pipeline 4
     * Camera on the link 1 --> Pipeline 5
     * Camera on the link 2 --> Pipeline 6
     * Camera on the link 3 --> Pipeline 7
     */
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send Pixel data, FS and FE to Controller 0 */
        {0x090B, 0x07},
        {0x092D, 0x00},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090D, 0x2C},
        {0x090E, 0x2C},
        {0x090F, 0x00},
        {0x0910, 0x00},
        {0x0911, 0x01},
        {0x0912, 0x01},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingRawRegs),
    };
    DevBlkCDII2CReg mappingEmbRegs[] = {
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x0913U, 0x12U},
        {0x0914U, 0x12U},
    };
    DevBlkCDII2CRegList mappingEmb = {
        .regs = mappingEmbRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbRegs),
    };
    DevBlkCDII2CReg mappingRawRegsB[] = {
        /* Send Pixel data, FS and FE to Controller 0 */
        {0x0A0B, 0x07},
        {0x0A2D, 0x00},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x0A0D, 0x2C},
        {0x0A0E, 0x2C},
        {0x0A0F, 0x00},
        {0x0A10, 0x00},
        {0x0A11, 0x01},
        {0x0A12, 0x01},
    };
    DevBlkCDII2CRegList mappingRawB = {
        .regs = mappingRawRegsB,
        .numRegs = I2C_ARRAY_SIZE(mappingRawRegsB),
    };
    DevBlkCDII2CReg mappingEmbRegsB[] = {
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x0A13U, 0x12U},
        {0x0A14U, 0x12U},
    };
    DevBlkCDII2CRegList mappingEmbB = {
        .regs = mappingEmbRegsB,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbRegsB),
    };

    bool isEmbDataType = false;
    uint8_t vcID = 0u;
    uint8_t dataTypeVal = 0u;
    uint8_t i = 0u;

    if (linkPipelineMap == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null link pipeline map");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Pipeline 0 ~ pipeline 3 */
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            isEmbDataType = linkPipelineMap[i].isEmbDataType;
            vcID = linkPipelineMap[i].vcID;

            if (isEmbDataType && !IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                SIPL_LOG_ERR_STR("MAX96712: Emb data type is valid for GMSL2 only");
                return NVMEDIA_STATUS_ERROR;
            }

            if (isEmbDataType && linkPipelineMap[i].isDTOverride) {
                SIPL_LOG_ERR_STR("MAX96712: Emb data type is not supported with dt override enabled");
                return NVMEDIA_STATUS_ERROR;
            }

            /* Update the reg addr for the next link */
            mappingRawRegs[0].address = (0x090BU + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[1].address = (0x092DU + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[2].address = (0x090DU + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[3].address = (0x090EU + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[4].address = (0x090FU + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[5].address = (0x0910U + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[6].address = (0x0911U + i * 0x40U) & 0xFFFFU;
            mappingRawRegs[7].address = (0x0912U + i * 0x40U) & 0xFFFFU;

            if (isEmbDataType) {
                mappingEmbRegs[0].address = (0x0913U + i * 0x40U) & 0xFFFFU;
                mappingEmbRegs[1].address = (0x0914U + i * 0x40U) & 0xFFFFU;
            }

            /* Update Tx Port */
            /*Enable 3 mappings for FS, FE, PIX */
            mappingRawRegs[0].data = 0x07U;
            /* Map all 3 to controller specified by txPort */
            mappingRawRegs[1].data = (txPort << 4u) | (txPort << 2u) | txPort;
            if (isEmbDataType) {
                /*Enable a mappings for EMB */
                mappingRawRegs[0].data |= (1U << 3U);
                /* Map EMB to controller specified by txPort */
                mappingRawRegs[1].data |= (txPort << 6u);
            }

            switch (linkPipelineMap[i].dataType) {
                case CDI_MAX96712_DATA_TYPE_RAW10:
                    dataTypeVal = 0x2BU;
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW12:
                    dataTypeVal = 0x2CU;
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW16:
                    dataTypeVal = 0x2EU;
                    break;
                case CDI_MAX96712_DATA_TYPE_RGB:
                    dataTypeVal = 0x24U;
                    break;
                case CDI_MAX96712_DATA_TYPE_YUV_8:
                    dataTypeVal = 0x1EU;
                    break;
                case CDI_MAX96712_DATA_TYPE_YUV_10:
                    dataTypeVal = 0x1F;
                    break;
                default:
                    SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
                    return NVMEDIA_STATUS_BAD_PARAMETER;
            }

            /* update offset + VC */
            /* Pixel data */
            mappingRawRegs[2].data = dataTypeVal;
            mappingRawRegs[3].data = (vcID << 6u) | dataTypeVal;
            mappingRawRegs[5].data = ((vcID << 6u) | 0x0U);
            mappingRawRegs[7].data = ((vcID << 6u) | 0x1U);
            if (isEmbDataType) {
                /* EMB data */
                mappingEmbRegs[1].data = (((vcID << 6U) | 0x12U) & 0xFFU);
            }

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRaw);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            if (isEmbDataType) {
                status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmb);
                if (status != NVMEDIA_STATUS_OK) {
                    return status;
                }
            }
        }
    }

    /* Pipeline 4 ~ pipeline 7 */
    if (drvHandle->ctx.camRecCfg > 0U) {
        for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                /* Update the reg addr for the next link */
                mappingRawRegsB[0].address = (0x0A0BU + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[1].address = (0x0A2DU + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[2].address = (0x0A0DU + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[3].address = (0x0A0EU + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[4].address = (0x0A0FU + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[5].address = (0x0A10U + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[6].address = (0x0A11U + i * 0x40U) & 0xFFFFU;
                mappingRawRegsB[7].address = (0x0A12U + i * 0x40U) & 0xFFFFU;

                if (isEmbDataType) {
                    mappingEmbRegsB[0].address = (0x0A13U + i * 0x40U) & 0xFFFFU;
                    mappingEmbRegsB[1].address = (0x0A14U + i * 0x40U) & 0xFFFFU;
                }
                isEmbDataType = linkPipelineMap[i].isEmbDataType;
                vcID = linkPipelineMap[i].vcID;

                if (isEmbDataType && !IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                    SIPL_LOG_ERR_STR("MAX96712: Emb data type is valid for GMSL2 only");
                    return NVMEDIA_STATUS_ERROR;
                }

                if (isEmbDataType && linkPipelineMap[i].isDTOverride) {
                    SIPL_LOG_ERR_STR("MAX96712: Emb data type is not supported with dt override enabled");
                    return NVMEDIA_STATUS_ERROR;
                }

                if (drvHandle->ctx.camRecCfg == 1U) { /* Ver 1 */
                    txPort = 2U; /* for x2 and x4 */
                } else if ((drvHandle->ctx.camRecCfg > 1U) && (drvHandle->ctx.camRecCfg < 5U)) { /* Ver 2 */
                    switch (i + 4U) {
                        case 4:
                        case 5:
                            txPort = 2U;
                            break;
                        case 6:
                        case 7:
                            txPort = 3U;
                            break;
                        default:
                            txPort = 2U;
                            break;
                    }
                }

                /* Update Tx Port */
                /*Enable 3 mappings for FS, FE, PIX */
                mappingRawRegsB[0].data = 0x07;
                /* Map all 3 to controller specified by txPort */
                mappingRawRegsB[1].data = (txPort << 4u) | (txPort << 2u) | txPort;
                if (isEmbDataType) {
                    /*Enable a mappings for EMB */
                    mappingRawRegsB[0].data |= (1U <<  3U);
                    /* Map EMB to controller specified by txPort */
                    mappingRawRegsB[1].data |= (txPort << 6u);
                }

                switch (linkPipelineMap[i].dataType) {
                    case CDI_MAX96712_DATA_TYPE_RAW10:
                        dataTypeVal = 0x2BU;
                        break;
                    case CDI_MAX96712_DATA_TYPE_RAW12:
                        dataTypeVal = 0x2CU;
                        break;
                    case CDI_MAX96712_DATA_TYPE_RAW16:
                        dataTypeVal = 0x2EU;
                        break;
                    case CDI_MAX96712_DATA_TYPE_RGB:
                        dataTypeVal = 0x24U;
                        break;
                    case CDI_MAX96712_DATA_TYPE_YUV_8:
                        dataTypeVal = 0x1EU;
                        break;
                    default:
                        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
                        return NVMEDIA_STATUS_BAD_PARAMETER;
                }

                /* update offset + VC */
                /* Pixel data */
                mappingRawRegsB[2].data = dataTypeVal;
                mappingRawRegsB[3].data = ((vcID << 6u) | dataTypeVal);
                mappingRawRegsB[5].data = ((vcID << 6u) | 0x0U);
                mappingRawRegsB[7].data = ((vcID << 6u) | 0x1U);
                /* EMB data */
                mappingEmbRegsB[1].data = ((vcID << 6u) | 0x12U);

                status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRawB);
                if (status != NVMEDIA_STATUS_OK) {
                    return status;
                }

                if (isEmbDataType) {
                    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmbB);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            }
        }
    }

    return VideoPipelineSelSinglePipe(handle,
                                      link,
                                      linkPipelineMap);
}

static NvMediaStatus
SetPipelineMapDoublePipe(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    TxPortMAX96712 txPort = drvHandle->ctx.txPort;
    /* Two pipelines are one set to process raw12 and emb */
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send RAW12 FS and FE from X to Controller 1 */
        {0x090B, 0x07},
        {0x092D, 0x00},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090D, 0x2C},
        {0x090E, 0x2C},
        {0x090F, 0x00},
        {0x0910, 0x00},
        {0x0911, 0x01},
        {0x0912, 0x01},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingRawRegs),
    };
    DevBlkCDII2CReg mappingEmbRegs[] = {
        /* Send EMB8 from Y to Controller 1 with VC unchanged */
        {0x0A0B, 0x07},
        {0x0A2D, 0x00},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x0A0D, 0x12},
        {0x0A0E, 0x12},
        {0x0A0F, 0x00},
        {0x0A10, 0x00},
        {0x0A11, 0x01},
        {0x0A12, 0x01},
    };
    DevBlkCDII2CRegList mappingEmb = {
        .regs = mappingEmbRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbRegs),
    };
    DevBlkCDII2CReg mappingEmbPipeZRegs[] = {
        /* Send EMB data from pipe Z to controller 1 */
        {0x0913, 0x12},
        {0x0914, 0x12},
    };
    DevBlkCDII2CRegList mappingEmbPipeZ = {
        .regs = mappingEmbPipeZRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingEmbPipeZRegs),
    };

    bool isEmbDataType = false;
    uint8_t vcID = 0u;
    uint8_t dataTypeVal = 0u;
    uint8_t i = 0u;
    uint8_t j = 0u;

    if (linkPipelineMap == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null link pipeline map");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            isEmbDataType = linkPipelineMap[i].isEmbDataType;
            vcID = linkPipelineMap[i].vcID;

            /* Update Tx Port */
            if (linkPipelineMap[i].isSinglePipeline) {
                if (isEmbDataType) {
                    /*Enable 4 mappings FS, FE, PIX, EMB */
                    mappingRawRegs[0].data = 0x0F;
                    /* Map all 4 to controller specified by txPort */
                    mappingRawRegs[1].data = (txPort << 6u) | (txPort << 4u) | (txPort << 2u) | txPort;
                } else {
                    /*Enable 3 mappings FS, FE, PIX, EMB */
                    mappingRawRegs[0].data = 0x07;
                    /* Map all 3 to controller specified by txPort */
                    mappingRawRegs[1].data = (txPort << 4u) | (txPort << 2u) | txPort;
                }
            } else {
                mappingRawRegs[1].data = (txPort << 4u) | (txPort << 2u) | txPort;
                mappingEmbRegs[1].data = (txPort << 4u) | (txPort << 2u) | txPort;
            }

            if (isEmbDataType && !IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                SIPL_LOG_ERR_STR("MAX96712: Emb data type is valid for GMSL2 only");
                return NVMEDIA_STATUS_ERROR;
            }

            if (isEmbDataType && linkPipelineMap[i].isDTOverride) {
                SIPL_LOG_ERR_STR("MAX96712: Emb data type is not supported with dt override enabled");
                return NVMEDIA_STATUS_ERROR;
            }

            switch (linkPipelineMap[i].dataType) {
                case CDI_MAX96712_DATA_TYPE_RAW10:
                    dataTypeVal = 0x2B;
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW12:
                    dataTypeVal = 0x2C;
                    break;
                case CDI_MAX96712_DATA_TYPE_RAW16:
                    dataTypeVal = 0x2E;
                    break;
                case CDI_MAX96712_DATA_TYPE_RGB:
                    dataTypeVal = 0x24;
                    break;
                case CDI_MAX96712_DATA_TYPE_YUV_8:
                    dataTypeVal = 0x1E;
                    break;
                default:
                    SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
                    return NVMEDIA_STATUS_BAD_PARAMETER;
            }

            /* update offset */
            /* 4 mapping for data and 4 mapping for emb */
            mappingRawRegs[2].data = dataTypeVal;
            mappingRawRegs[3].data = (vcID << 6u) | dataTypeVal;

            mappingEmbPipeZRegs[1].data = (vcID << 6u) | 0x12;

            if (linkPipelineMap[i].isSinglePipeline) {
                /* If this is a single pipeline only map raw, no need to map emb data */
                mappingRawRegs[5].data = (vcID << 6u) | 0x0;
                mappingRawRegs[7].data = (vcID << 6u) | 0x1;
            } else {
                /* Change FS packet's DT to reserved for RAW pipeline if emb data is used */
                mappingRawRegs[5].data = (vcID << 6u) | (isEmbDataType ? 2u : 0u);
                /* Change FE packet's DT to reserved for RAW pipeline if emb data is used */
                mappingRawRegs[7].data = (vcID << 6u) | (isEmbDataType ? 3u : 1u);
                mappingEmbRegs[3].data = (vcID << 6u) | 0x12;
                mappingEmbRegs[5].data = (vcID << 6u) | 0x0;
                mappingEmbRegs[7].data = (vcID << 6u) | 0x1;
            }

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRaw);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            if (!linkPipelineMap[i].isSinglePipeline) {
                if (isEmbDataType) {
                    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmb);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            } else {
                if (isEmbDataType) {
                    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingEmbPipeZ);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            }
        }

        /* Update the reg addr for the next link */
        for (j = 0u; j < 8u; j++) {
            mappingRawRegs[j].address += 0x40;
            mappingEmbRegs[j].address += 0x40;
        }

        /* Update the reg addr for the next link */
        for (j = 0u; j < 2u; j++) {
            mappingEmbPipeZRegs[j].address += 0x40;
        }
    }

    status = VideoPipelineSelDoublePipe(handle,
                                        link,
                                        linkPipelineMap);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return EnableReplication(handle, true);
}

static NvMediaStatus
SetPipelineMap(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    LinkPipelineMapMAX96712 *linkPipelineMap,
    bool isSharedPipeline)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (isSharedPipeline) {
        status = SetPipelineMapSinglePipe(handle, link, linkPipelineMap);
    } else {
        status = SetPipelineMapDoublePipe(handle, link, linkPipelineMap);
    }

    return status;
}

static NvMediaStatus
SetPipelineMapTPG(
    DevBlkCDIDevice *handle,
    uint8_t linkIndex,
    LinkPipelineMapMAX96712 *linkPipelineMap)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    TxPortMAX96712 txPort = drvHandle->ctx.txPort;
    bool PGGen0 = true;
    uint8_t vcID = 0u, dataTypeVal = 0u;
    uint8_t i = 0u;
    /* Two pipelines are one set to process raw12 and emb */
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send RAW12 FS and FE from X to Controller 1 */
        {0x090B, 0x07},
        {0x092D, 0x00},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090D, 0x2C},
        {0x090E, 0x2C},
        {0x090F, 0x00},
        {0x0910, 0x00},
        {0x0911, 0x01},
        {0x0912, 0x01},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingRawRegs),
    };

    if (linkPipelineMap == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null link pipeline map");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    vcID = linkPipelineMap[linkIndex].vcID;

    /* Update Tx Port */
    mappingRawRegs[1].data = (txPort << 4u) | (txPort << 2u) | txPort;

    switch (linkPipelineMap[linkIndex].dataType) {
        case CDI_MAX96712_DATA_TYPE_RAW10:
            dataTypeVal = 0x2B;
            break;
        case CDI_MAX96712_DATA_TYPE_RAW12:
            dataTypeVal = 0x2C;
            break;
        case CDI_MAX96712_DATA_TYPE_RAW16:
            dataTypeVal = 0x2E;
            break;
        case CDI_MAX96712_DATA_TYPE_RGB:
            dataTypeVal = 0x24;
            break;
        case CDI_MAX96712_DATA_TYPE_YUV_8:
            dataTypeVal = 0x1E;
            break;
        case CDI_MAX96712_DATA_TYPE_YUV_10:
            dataTypeVal = 0x1F;
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid data type");
            return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (drvHandle->ctx.pipelineEnabled & (0x1 << linkIndex)) {
        PGGen0 = true;
    } else if (drvHandle->ctx.pipelineEnabled & (0x10 << linkIndex)) {
        PGGen0 = false;
    } else {
        SIPL_LOG_ERR_STR_UINT("MAX96712: No pipeline enabled for the link", (uint32_t)linkIndex);
        LOG_MSG("No pipeline enabled for the link %d\n", linkIndex);
        LOG_MSG("          Please make sure if CDI_WRITE_PARAM_CMD_MAX96712_SET_PG calling\n");
        LOG_MSG("          before CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG\n");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* update offset */
    for (i = 0u; i < 8u; i++) {
        mappingRawRegs[i].address += (linkIndex * 0x40) + ((PGGen0 == false) ? 0x100 : 0x0);
    }

    /* 4 mapping for the pixel data */
    mappingRawRegs[2].data = dataTypeVal;
    mappingRawRegs[3].data = (vcID << 6u) | dataTypeVal;
    /* Change FS packet's DT to reserved for RAW pipeline */
    mappingRawRegs[5].data = (vcID << 6u) | 0u;
    /* Change FE packet's DT to reserved for RAW pipeline */
    mappingRawRegs[7].data = (vcID << 6u) | 1u;

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRaw);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return status;
}

static NvMediaStatus
ConfigPGSettings(
    DevBlkCDIDevice *handle,
    uint32_t width,
    uint32_t height,
    float frameRate,
    uint8_t linkIndex)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CRegList *PGarray = NULL;
    DevBlkCDII2CReg *regsPG = NULL;
    uint8_t i = 0U, j = 0U, pclk = 0U;
    PGModeMAX96712 pgMode = CDI_MAX96712_PG_NUM;

    drvHandle->ctx.tpgEnabled = true;

    if (i >= MAX96712_MAX_NUM_PG) {
        return NVMEDIA_STATUS_ERROR;
    }

    if ((width == 1920U) && (height == 1236U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_1920_1236_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr1920x1236_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_150MHX;
    } else if ((width == 1920U) && (height == 1236U) && (frameRate == 60.0f)) {
        pgMode = CDI_MAX96712_PG_1920_1236_60FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr1920x1236_60FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_375MHX;
    } else if ((width == 3848U) && (height == 2168U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_3848_2168_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr3848x2168_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_375MHX;
    } else if ((width == 3848U) && (height == 2174U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_3848_2174_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr3848x2174_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_375MHX;
    } else if ((width == 2880U) && (height == 1860U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_2880_1860_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr2880x1860_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_375MHX;
    } else if ((width == 1920U) && (height == 1559U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_1920_1559_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr1920x1559_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_150MHX;
    } else if ((width == 3840U) && (height == 2181U) && (frameRate == 30.0f)) {
        pgMode = CDI_MAX96712_PG_3840_2181_30FPS;
        PGarray = &configPGArrCmd[pgMode];
        regsPG = PGArr3840x2181_30FPS_PATGEN0;
        pclk = CDI_MAX96712_PG_PCLK_375MHX;
    } else {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0U; i < MAX96712_MAX_NUM_PG; i++) {
        if (drvHandle->ctx.pgMode[i] == pgMode) {
            break;
        }
        if (drvHandle->ctx.pgMode[i] == CDI_MAX96712_PG_NUM) {
            drvHandle->ctx.pgMode[i] = pgMode;
            break;
        }
    }

    drvHandle->ctx.pipelineEnabled |= ((1 << linkIndex) << (i * 4));

    if (i == 1U) { /* For 2nd PG, need to update the register offset */
        /* PG setting */
        for (j = 0U; j < 38U; j++) {
            regsPG[j].address += 0x30;
        }
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        PGarray);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    if (pclk == CDI_MAX96712_PG_PCLK_150MHX) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                            &configPGPCLK150MHZ[i]);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_FORCE_CSI_OUT_EN,
                                 1u,
                                 REG_READ_MOD_WRITE_MODE);

    return status;
}

static NvMediaStatus
MapUnusedPipe(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    drvHandle->ctx.tpgEnabled = true;
    uint8_t i = 0U, j = 0U;
    DevBlkCDII2CReg mappingRawRegs[] = {
        /* Send RAW12 FS and FE from X to Controller 1 */
        {0x090B, 0x07},
        {0x092D, 0x3F},
        /* For the following MSB 2 bits = VC, LSB 6 bits = DT */
        {0x090D, 0x24},
        {0x090E, 0x3F},
        {0x090F, 0x00},
        {0x0910, 0x02},
        {0x0911, 0x01},
        {0x0912, 0x03},
    };
    DevBlkCDII2CRegList mappingRaw = {
        .regs = mappingRawRegs,
        .numRegs = I2C_ARRAY_SIZE(mappingRawRegs),
    };

    /* When enabling TPG on Max96712, 1st TPG output is going to pipeline 0 ~ 3,
     * 2nd TPG output is going to pipeline 4 ~ 7.
     * And pipeline 0/4 is going to controller 0, pipeline 1/5 is going to controller 1
     * pipeline 2/6 is going to controller 2, pipeline 3/7 is going to controller 3 by default.
     * Since there is no way to disable TPG and TPG is behind the pipeline,
     * undesired pipeline output has to be mapped to unused controller.
     */
    for (i = 0U; i < MAX96712_NUM_VIDEO_PIPELINES; i++) {
        if (!(drvHandle->ctx.pipelineEnabled & (0x1 << i))) {
            if (drvHandle->ctx.mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
                mappingRawRegs[1].data = 0x3F; /* controller 1 */
            } else if (drvHandle->ctx.mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
                /* 2x4 mode*/
                mappingRawRegs[1].data = 0x3F; /* controller 0 */
            } else {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &mappingRaw);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }

        for (j = 0U; j < 8U; j++) {
            mappingRawRegs[j].address += 0x40;
        }
    }

    return status;
}

static NvMediaStatus
EnablePG(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg enablePGArrCmdReg = {0x1050, 0xF3};
    drvHandle->ctx.tpgEnabled = true;

    if (drvHandle->ctx.pipelineEnabled & 0xF0) {
        enablePGArrCmdReg.address += 0x30;
    }

    return DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                      enablePGArrCmdReg.address,
                                      enablePGArrCmdReg.data);
}

static NvMediaStatus
SetTxSRCId(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg txSrcIdReg = {0x0503, 0x00};
    uint8_t i = 0u;

    if (MAX96712_IS_MULTIPLE_GMSL_LINK_SET(link)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad param: Multiple links specified");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            txSrcIdReg.address += (i << 4); /* update the address */
            txSrcIdReg.data = i; /* 0 - link 0, 1 - link 1, so on */

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                txSrcIdReg.address,
                                                txSrcIdReg.data);
            break;
        }
    }

    return status;
}

static NvMediaStatus
DisableAutoAck(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg autoAckReg = {0x0B0D, 0x00};
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            autoAckReg.address += (i << 8); /* update the address */

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                autoAckReg.address,
                                                autoAckReg.data);
            nvsleep(25000);
        }
    }

    return status;
}

static NvMediaStatus
EnableERRB(
    DevBlkCDIDevice *handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_ENABLE_ERRB,
                                 (enable ? 1u : 0u),
                                 REG_READ_MOD_WRITE_MODE);

    return status;
}

static NvMediaStatus
EnableCSIOut(
    DevBlkCDIDevice *handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t phys = 0U;

    if (drvHandle->ctx.revision == CDI_MAX96712_REV_2) {
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_BACKTOP_EN,
                                     (enable ? 1u : 0u),
                                     REG_READ_MOD_WRITE_MODE);
    } else {
        if (drvHandle->ctx.camRecCfg == 0U) { /* No recorder support */
            phys = 0x3U;
        } else {
            phys = 0xFU;
        }
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_PHY_STANDBY,
                                     (enable ? phys : 0u),
                                     REG_READ_MOD_WRITE_MODE);
    }

    return status;
}

static NvMediaStatus
TriggerDeskew(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t i = 0u;
    uint8_t phy_num = 0u;
    uint8_t temp;
    DevBlkCDII2CReg deskewReg = {0x0903, 0x00};

    /* Trigger the initial deskew patterns two times
     * to make sure Rx device recevies the patterns */
    for (i = 0u; i < 2u; i++) {
        for (phy_num = 0u; phy_num < MAX96712_MAX_NUM_PHY; phy_num++) {
            /* Update the register offset */
            deskewReg.address = (deskewReg.address & 0xFF00U) +
                                (0x40U * phy_num) + 0x03U;
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               deskewReg.address,
                                               &temp);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
            deskewReg.data = temp;

            deskewReg.data ^= (1 << 5);
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                deskewReg.address,
                                                deskewReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
        nvsleep((i == 0u) ? 10000 : 0);
    }

    return status;
}

static NvMediaStatus
EnableExtraSMs(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR4,
                                 0xF3u,
                                 REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR6,
                                 0xFDu, /* TODO : Enable the remote error */
                                 REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR10,
                                 0xFFu,
                                 REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR12_MEM_ERR_OEN,
                                 0x1u,
                                 REG_READ_MOD_WRITE_MODE);

    ClearRegFieldQ(handle);
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_MASKED_OEN,
                                 0x3Fu,
                                 REG_READ_MOD_WRITE_MODE);

    return status;
}

static NvMediaStatus
SetI2CPort(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort;
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) {
            if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_I2C_PORT_GMSL1_PHY_A + i,
                                             (i2cPort == CDI_MAX96712_I2CPORT_0) ? 0u : 1u,
                                             REG_READ_MOD_WRITE_MODE);
            } else if (IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) {
                /* Disable connection from both port 0/1 */
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_DIS_REM_CC_A + i,
                                             0x3u,
                                             REG_READ_MOD_WRITE_MODE);

                /* Select port 0 or 1 over the link */
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SEC_XOVER_SEL_PHY_A + i,
                                             (i2cPort == CDI_MAX96712_I2CPORT_0) ? 0u : 1u,
                                             REG_READ_MOD_WRITE_MODE);

                /* Enable connection from port 0 or 1 */
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_DIS_REM_CC_A + i,
                                             (i2cPort == CDI_MAX96712_I2CPORT_0) ? 2u : 1u,
                                             REG_READ_MOD_WRITE_MODE);
            }
        }

        /* Update I2C slave timeout */
        if (i2cPort == CDI_MAX96712_I2CPORT_0) {
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SLV_TO_P0_A + i,
                                         0x5, /* 16 ms timeout. This value is less than I2C_INTREG_SLV_0_TO */
                                         REG_READ_MOD_WRITE_MODE);
        } else if (i2cPort == CDI_MAX96712_I2CPORT_1)  {
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_SLV_TO_P1_A + i,
                                         0x5, /* 16 ms timeout. This value is less than I2C_INTREG_SLV_1_TO */
                                         REG_READ_MOD_WRITE_MODE);
        } else {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    return status;
}

static NvMediaStatus
SetFSYNCMode(
    DevBlkCDIDevice *handle,
    FSyncModeMAX96712 FSyncMode,
    uint32_t pclk,
    uint32_t fps,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    DevBlkCDII2CReg fsyncPeriodRegs[] = {
        {0x04A7, 0x00}, /* Calculate FRSYNC period H. don't move position */
        {0x04A6, 0x00}, /* Calculate FRSYNC period M. don't move position */
        {0x04A5, 0x00}, /* Calculate FRSYNC period L. don't move position */
    };
    DevBlkCDII2CRegList fsyncPeriod = {
        .regs = fsyncPeriodRegs,
        .numRegs = I2C_ARRAY_SIZE(fsyncPeriodRegs),
    };
    DevBlkCDII2CReg setManualFsyncModeRegs[] = {
        {0x04A2, 0xE0}, /* video link for fsync generation */
        {0x04AA, 0x00}, /* Disable overlap window */
        {0x04AB, 0x00}, /* Disable overlap window */
        {0x04A8, 0x00}, /* Disable error threshold */
        {0x04A9, 0x00}, /* Disable error threshold */
        {0x04AF, 0x1F}, /* Set FSYNC to GMSL1 type */
        {0x04A0, 0x10}, /* Set FSYNC to manual mode */
    };
    DevBlkCDII2CRegList setManualFsyncMode = {
        .regs = setManualFsyncModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setManualFsyncModeRegs),
    };
    DevBlkCDII2CReg setAutoFsyncModeRegs[] = {
        {0x04A2, 0xE1}, /* video link for fsync generation */
        {0x04AA, 0x00}, /* Disable overlap window */
        {0x04AB, 0x00}, /* Disable overlap window */
        {0x04A8, 0x00}, /* Disable error threshold */
        {0x04A9, 0x00}, /* Disable error threshold */
        {0x04B1, 0x78}, /* GPIO ID setup to output FSYNC. For Auto mode, select ID=0xF */
        {0x04A0, 0x12}, /* Set FSYNC to auto mode */
    };
    DevBlkCDII2CRegList setAutoFsyncMode = {
        .regs = setAutoFsyncModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setAutoFsyncModeRegs),
    };
    DevBlkCDII2CReg setOSCManualFsyncModeRegs[] = {
        {0x04AF, 0x40, 0x2710}, /* Set FSYNC to GMSL1 type then delay 10ms*/
        {0x04A0, 0x00, 0x2710}, /* Set FSYNC to manual mode then delay 10ms*/
        {0x04A2, 0x00, 0x2710}, /* Turn off auto master link selection then delay 10ms*/
        {0x04AA, 0x00, 0x2710}, /* Disable overlap window then delay 10ms*/
        {0x04AB, 0x00, 0x2710}, /* Disable overlap window then delay 10ms*/
        {0x04A8, 0x00, 0x2710}, /* Disable error threshold then delay 10ms*/
        {0x04A9, 0x00, 0x2710}, /* Disable error threshold then delay 10ms*/
    };
    DevBlkCDII2CRegList setOSCManualFsyncMode = {
        .regs = setOSCManualFsyncModeRegs,
        .numRegs = I2C_ARRAY_SIZE(setOSCManualFsyncModeRegs),
    };
    DevBlkCDII2CReg setExtFsyncModeReg = {0x04A0, 0x08};
    DevBlkCDII2CReg setTxIDIntReg = {0x04B1, CDI_MAX96712_GPIO_20 << 3}; /* GPIO ID 20 */
    DevBlkCDII2CReg setGpio2Mode = {0x0306, 0x83};
    DevBlkCDII2CReg setGMSL2PerLinkExtFsyncModeRegs[4] = {
        {0x0307, 0xA0 | CDI_MAX96712_GPIO_2},
        {0x033D, 0x20 | CDI_MAX96712_GPIO_2},
        {0x0374, 0x20 | CDI_MAX96712_GPIO_2},
        {0x03AA, 0x20 | CDI_MAX96712_GPIO_2},
    };
    DevBlkCDII2CReg enableGpiGpoReg = {0x0B08, 0x00};
    uint8_t i = 0u;

    /* TODO: Handle GMSL1 + GMSL2 case */
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            gmslMode = drvHandle->ctx.gmslMode[i];
            break;
        }
    }

    if (FSyncMode == CDI_MAX96712_FSYNC_MANUAL) {
        /* Calculate FRSYNC period in manual mode based on PCLK */
        if (drvHandle->ctx.manualFSyncFPS != 0u) {
            if (drvHandle->ctx.manualFSyncFPS != fps) {
                /* Requested a new manual fsync frequency*/
                SIPL_LOG_ERR_STR("MAX96712: 2 different manual fsync frequencies requested");
                return NVMEDIA_STATUS_NOT_SUPPORTED;
            }
        } else {
            /* calculate frsync high period */
            fsyncPeriodRegs[0].data = (uint8_t)((gmslMode ==
                                      (CDI_MAX96712_GMSL1_MODE)) ?
                                      (((pclk / fps) >> 16U) &
                                        0xFFU) : 0x25U);
            /* calculate frsync middle period */
            fsyncPeriodRegs[1].data = (uint8_t)((gmslMode ==
                                      (CDI_MAX96712_GMSL1_MODE)) ?
                                      (((pclk / fps) >> 8U) &
                                        0xFFU) : 0x4CU);
            /* calculate frsync low period */
            fsyncPeriodRegs[2].data = (uint8_t)((gmslMode ==
                                      (CDI_MAX96712_GMSL1_MODE)) ?
                                      ((pclk / fps) & 0xFFU) : 0x9CU);

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &fsyncPeriod);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            if (IsGMSL2Mode(gmslMode)) {
                setManualFsyncModeRegs[6].data = 0x90; /* Set FSYNC to GMSL2 type */
            }

            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setManualFsyncMode);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            nvsleep(10000);

            drvHandle->ctx.manualFSyncFPS = fps;
        }

        if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
                enableGpiGpoReg.data = 0x35;

                for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
                    if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                        enableGpiGpoReg.address += (i << 8);

                        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                            enableGpiGpoReg.address,
                                                            enableGpiGpoReg.data);
                        if (status != NVMEDIA_STATUS_OK) {
                            return status;
                        }
                        nvsleep(10000);
                    }
                }
        }
    } else if (FSyncMode == CDI_MAX96712_FSYNC_OSC_MANUAL) {
        /* Calculate FRSYNC period in manual with OSC mode */
        if (drvHandle->ctx.manualFSyncFPS != 0u) {
            if (drvHandle->ctx.manualFSyncFPS != fps) {
                /* Requested a new manual fsync frequency*/
                SIPL_LOG_ERR_STR("MAX96712: 2 different manual osc fsync frequencies requested");
                return NVMEDIA_STATUS_NOT_SUPPORTED;
            }
        }

        /* MAXIM doesn't recommend to use auto or semi-auto mode for the safety concern.
         * If the master link is lost, the frame sync will be lost for other links in both modes.
         * Instead the manual mode with OSC in MAX96712 is recommended.
         */
        if (IsGMSL2Mode(gmslMode)) {
            setOSCManualFsyncModeRegs[0].data |= (1 << 7); /* Set FSYNC to GMSL2 type */
        }

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setOSCManualFsyncMode);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }

        if (IsGMSL2Mode(gmslMode)) {
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                setTxIDIntReg.address,
                                                setTxIDIntReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }

        /* calculate frsync high period
         */
        fsyncPeriodRegs[0].data =
           (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U / fps) >> 16U) &
                        0xFFU);
        /* calculate frsync middle period */
        fsyncPeriodRegs[1].data =
           (uint16_t)(((MAX96712_OSC_MHZ * 1000U * 1000U / fps) >> 8U) &
                        0xFFU);
        /* calculate frsync low period */
        fsyncPeriodRegs[2].data =
           (uint16_t)((MAX96712_OSC_MHZ * 1000U * 1000U / fps) & 0xFFU);

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &fsyncPeriod);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }

        drvHandle->ctx.manualFSyncFPS = fps;

        if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
            enableGpiGpoReg.data = 0x35;

            for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
                if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                    enableGpiGpoReg.address += (i << 8);

                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        enableGpiGpoReg.address,
                                                        enableGpiGpoReg.data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                    nvsleep(10000);
                }
            }
        }
    } else if (FSyncMode == CDI_MAX96712_FSYNC_EXTERNAL) {
        for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
            if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
                if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        setExtFsyncModeReg.address,
                                                        setExtFsyncModeReg.data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                    enableGpiGpoReg.data = 0x65;
                    enableGpiGpoReg.address += (i << 8);

                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        enableGpiGpoReg.address,
                                                        enableGpiGpoReg.data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                    nvsleep(10000);
                } else {
                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        setGpio2Mode.address,
                                                        setGpio2Mode.data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }

                    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                        setGMSL2PerLinkExtFsyncModeRegs[i].address,
                                                        setGMSL2PerLinkExtFsyncModeRegs[i].data);
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                    nvsleep(10000);
                }
            }
        }
    } else if (FSyncMode == CDI_MAX96712_FSYNC_AUTO) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &setAutoFsyncMode);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    } else {
        SIPL_LOG_ERR_STR("MAX96712: Invalid param: FSyncMode");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

static NvMediaStatus
ReadCtrlChnlCRCErr(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    uint8_t *errVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0u;

    if (errVal == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null error value");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_CC_CRC_ERRCNT_A + i,
                                         0u,
                                         REG_READ_MODE);
            *errVal = ReadFromRegFieldQ(handle, 0u);
        }
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
GetEnabledLinks(
    DevBlkCDIDevice *handle,
    LinkMAX96712 *link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0u;

    *link = 0u;
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        ClearRegFieldQ(handle);
        ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_LINK_EN_A + i,
                                  0u);
        ACCESS_REG_FIELD_RET_ERR(REG_READ_MODE);
        *link |= (ReadFromRegFieldQ(handle, 0u) << i);
    }

    return status;
}

static NvMediaStatus
ClearErrb(
    DevBlkCDIDevice *handle,
    LinkMAX96712 *link,
    uint8_t *errVal)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    ErrorStatusMAX96712 errorStatus;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    if (drvHandle->ctx.tpgEnabled == true) {
        return NVMEDIA_STATUS_OK;
    }

    ClearRegFieldQ(handle);
    ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_ERRB,
                              0u);
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MODE);
    if (ReadFromRegFieldQ(handle, 0u) == 1u) {
        SIPL_LOG_ERR_STR("MAX96712: MAX96712 ERRB was Set");
        status = MAX96712GetErrorStatus(handle,
                                        sizeof(errorStatus),
                                        &errorStatus);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    return status;
}

static NvMediaStatus
EnableReplication(
    DevBlkCDIDevice *handle,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    TxPortMAX96712 port = CDI_MAX96712_TXPORT_PHY_C;
    MipiOutModeMAX96712 mipiOutMode = CDI_MAX96712_MIPI_OUT_INVALID;
    PHYModeMAX96712 phyMode = CDI_MAX96712_PHY_MODE_INVALID;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    _DriverHandle *drvHandle = NULL;
    DevBlkCDII2CReg dataRegs[] = {
        {0x08A9, 0},  /* For the replication from Tegra A to Tegra B */
        {0x08AA, 0},  /* For the replication from Tegra A to Tegra C */
    };
    DevBlkCDII2CRegList data = {
        .regs = dataRegs,
        .numRegs = I2C_ARRAY_SIZE(dataRegs),
    };
    if ((handle == NULL) || ((drvHandle = (_DriverHandle *)handle->deviceDriverHandle) == NULL)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    port = drvHandle->ctx.txPort;
    mipiOutMode = drvHandle->ctx.mipiOutMode;
    revision = drvHandle->ctx.revision;
    phyMode = drvHandle->ctx.phyMode;

    /* Replication is not supported on revision 1 in CPHY mode */
    if ((revision == CDI_MAX96712_REV_1) &&
        (phyMode == CDI_MAX96712_PHY_MODE_CPHY)) {
        SIPL_LOG_ERR_STR("MAX96712: Replication in CPHY mode is supported only "
                         "on platforms with MAX96712 revision 2 or higher.");
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    switch (port) {
        case CDI_MAX96712_TXPORT_PHY_C :
            if (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
                dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_C, CDI_MAX96712_TXPORT_PHY_E);
            } else if (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
                dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_D, CDI_MAX96712_TXPORT_PHY_E);
            } else {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_MAX96712_TXPORT_PHY_D :
            if ((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) || (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2)) {
                dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_D, CDI_MAX96712_TXPORT_PHY_E);
            } else {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        case CDI_MAX96712_TXPORT_PHY_E :
            if (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) {
                dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_E, CDI_MAX96712_TXPORT_PHY_D);
            } else if (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) {
                dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_E, CDI_MAX96712_TXPORT_PHY_C);
            } else {
                return NVMEDIA_STATUS_BAD_PARAMETER;
            }
            break;
        default :
            dataRegs[0].data = MAX96712_REPLICATION(CDI_MAX96712_TXPORT_PHY_C, CDI_MAX96712_TXPORT_PHY_E);
            break;
    }

    /* Enable the replication mode */
    dataRegs[0].data |= (1U << 7U);

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return status;
}

static NvMediaStatus
ConfigureMIPIOutput(
    DevBlkCDIDevice *handle,
    uint8_t mipiSpeed,
    PHYModeMAX96712 phyMode)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    MipiOutModeMAX96712 mipiOutMode = drvHandle->ctx.mipiOutMode;
    uint8_t mipiOutModeVal = (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ? (1u << 0u) :
                             ((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) ? (1U << 2U) : (1U << 3U));
    DevBlkCDII2CReg mipiOutputReg = {0x08A2, 0x00};
    uint8_t i = 0u;
    uint8_t temp;
    uint8_t prebegin = 0U, post = 0U;

    if ((phyMode != CDI_MAX96712_PHY_MODE_DPHY) &&
        (phyMode != CDI_MAX96712_PHY_MODE_CPHY)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid MIPI output port");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((mipiSpeed < 1u) || (mipiSpeed > 25u)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Force DPHY0 clk enabled not needed for Rev 1 */
    if ((drvHandle->ctx.revision != CDI_MAX96712_REV_1) &&
        (phyMode == CDI_MAX96712_PHY_MODE_DPHY) &&
        (mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4)) {
        mipiOutModeVal = mipiOutModeVal | (1u << 5u);
    }

    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_MIPI_OUT_CFG,
                                 mipiOutModeVal,
                                 REG_READ_MOD_WRITE_MODE);

    /* Set prebegin phase, post length and prepare for CPHY mode
     * This is a requirement for CPHY periodic calibration */
    if (phyMode == CDI_MAX96712_PHY_MODE_CPHY) {
        if (mipiSpeed == 17) {
            /* TODO : This is a temporal solution to support the previous platform
             * This will be updated once CPHY calibration logic in RCE updated
             */
            /* t3_prebegin = (63 + 1) * 7 = 448 UI
             * Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
             * Bit[1:0] = t3_prepare = 86.7ns
             */
            prebegin = 0x3F;
            post = 0x7F;
        } else {
            /* t3_prebegin = (19 + 1) * 7 = 140 UI
             * Bit[6:2] = t3_post = (31 + 1) * 7 = 224 UI
             * Bit[1:0] = t3_prepare = 40ns
             */
            prebegin = 0x13;
            post = 0x7c;
        }
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_T_T3_PREBEGIN,
                                     prebegin,
                                     REG_READ_MOD_WRITE_MODE);

        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_T_T3_POST_PREP,
                                     post,
                                     REG_READ_MOD_WRITE_MODE);
    }

    /* Put all Phys in standby mode */
    mipiOutputReg.address = 0x08A2;
    mipiOutputReg.data = 0xF4; /* Bug 200383247 : t_lpx 106.7 ns */
    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        mipiOutputReg.address,
                                        mipiOutputReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Mapping data lanes Port A */
    mipiOutputReg.address = 0x08A3;
    mipiOutputReg.data = (mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) ? 0x44 : 0xE4;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        mipiOutputReg.address,
                                        mipiOutputReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Mapping data lanes Port B */
    mipiOutputReg.address = 0x08A4;
    mipiOutputReg.data = (drvHandle->ctx.camRecCfg > 1U) ? 0x44 : 0xE4;
    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        mipiOutputReg.address,
                                        mipiOutputReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Set CSI2 lane count per Phy */
    for (i = 0u; i < MAX96712_MAX_NUM_PHY; i++) {
        mipiOutputReg.data = (drvHandle->ctx.lanes[i] - 1U) << 6U;
        mipiOutputReg.data |= (phyMode == CDI_MAX96712_PHY_MODE_CPHY) ? (1u << 5u) : 0u;
        mipiOutputReg.address = 0x090A + (i * 0x40U);
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            mipiOutputReg.address,
                                            mipiOutputReg.data);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    if (drvHandle->ctx.revision == CDI_MAX96712_REV_2) {
        /* deactive DPLL */
        mipiOutputReg.address = 0x1C00;
        mipiOutputReg.data = 0xF4;

        for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
            mipiOutputReg.address = (0x1C + i) << 8;

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                mipiOutputReg.address,
                                                mipiOutputReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    /* Set MIPI speed */
    mipiOutputReg.address = 0x0415;
    for (i = 0u; i < MAX96712_MAX_NUM_PHY; i++) {
        mipiOutputReg.address =
           (mipiOutputReg.address & 0xFF00U) +
           0x15U +
           (i * 0x3U);
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           mipiOutputReg.address,
                                           &temp);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
        mipiOutputReg.data = temp;

        mipiOutputReg.data &= ~0x3F;
        mipiOutputReg.data |= ((1u << 5u) | mipiSpeed);
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            mipiOutputReg.address,
                                            mipiOutputReg.data);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    if (drvHandle->ctx.revision == CDI_MAX96712_REV_2) {
        /* active DPLL */
        mipiOutputReg.address = 0x1C00;
        mipiOutputReg.data = 0xF5;

        for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
            mipiOutputReg.address = (0x1C + i) << 8;

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                mipiOutputReg.address,
                                                mipiOutputReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    /* Deskew is enabled if MIPI speed is faster than or equal to 1.5GHz */
    if ((phyMode == CDI_MAX96712_PHY_MODE_DPHY) && (mipiSpeed >= 15)) {
        mipiOutputReg.address = 0x0903;
        mipiOutputReg.data = 0x97; /* enable the initial deskew with 8 * 32K UI */
        for (i = 0; i < MAX96712_MAX_NUM_PHY; i++) {
            mipiOutputReg.address = (mipiOutputReg.address & 0xff00) + ((mipiOutputReg.address + 0x40) & 0xff) ;
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                mipiOutputReg.address,
                                                mipiOutputReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    return status;
}

static NvMediaStatus
DisableDE(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t i = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            ClearRegFieldQ(handle);

            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_DE_EN_PHY_A + i,
                                      0u);
            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_DE_PRBS_TYPE_PHY_A + i,
                                      1u);

            ACCESS_REG_FIELD_RET_ERR(REG_WRITE_MODE);
        }
    }

    return status;
}

static NvMediaStatus
SetDBL(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg dblReg = {0x0B07, 0x8C};
    uint8_t i = 0u;

    if (enable == false) {
        dblReg.data = 0x0;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            dblReg.address += (i << 8);
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, dblReg.address, dblReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
            nvsleep(5000);
        }
    }

    return status;
}

static NvMediaStatus
ControlForwardChannels(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    bool enable)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    I2CPortMAX96712 i2cPort = drvHandle->ctx.i2cPort;
    uint8_t i = 0u;
    uint8_t data = 0;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            /* WAR Bug 2411206 and 200653773:
               Sometimes when reading the 0x0B04/0x0C04/0x0D04/0x0E04 registers,
               0x00 is returned, regardless of what was written to the register.
               To get around this issue, program the register with i2c write command
               directly, instead of using read-modify-write method with access field
               queue.
             */

            /* Set forward control channel bit if enabled */
            if (enable) {
                data |= 0x1;
            }

            /* Always set reverse control channel bit to 1 */
                data |= 0x2;

            /* Set I2C/UART port bit for Port 1 */
            if (i2cPort == CDI_MAX96712_I2CPORT_1) {
                data |= 0x8;
            }

            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                regBitFieldProps[REG_FIELD_I2C_FWDCCEN_PHY_A + i].regAddr,
                                                data);
            nvsleep(10000);
        }
    }

    return status;
}

/*
 *  The functions defined below are the entry points when CDI functions are called.
 */

NvMediaStatus
MAX96712CheckLink(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link,
    uint32_t linkType,
    bool display)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    uint8_t i = 0u, linkIndex = 0u, success = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null handle passed to MAX96712CheckLink");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null driver handle passed to MAX96712CheckLink");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (drvHandle->ctx.tpgEnabled == true) {
        return NVMEDIA_STATUS_OK;
    }

    for (linkIndex = 0u; linkIndex < MAX96712_MAX_NUM_LINK; linkIndex++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, linkIndex)) {
            gmslMode = drvHandle->ctx.gmslMode[linkIndex];
            /* Check lock for each link */
            switch (linkType) {
                case CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG:
                    if (gmslMode != CDI_MAX96712_GMSL1_MODE) {
                        SIPL_LOG_ERR_STR_INT("MAX96712: Config link lock is only valid in GMSL1 mode on link", (int32_t)linkIndex);
                        return NVMEDIA_STATUS_ERROR;
                    }

                    /* Check for GMSL1 Link Lock.*/
                    ClearRegFieldQ(handle);
                    ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_LOCK_A + linkIndex,
                                              0u);
                    ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_CONFIG_LOCK_A + linkIndex,
                                              0u);

                    /* From Max96712 programming guide V1.1, typical link rebuilding time is 25 ~ 100ms
                     * check the link lock in 100ms periodically every 10ms */
                    for (i = 0u; i < 50u; i++) {
                        ACCESS_REG_FIELD_RET_ERR(REG_READ_MODE);
                        if ((ReadFromRegFieldQ(handle, 0u) == 1u) ||
                            (ReadFromRegFieldQ(handle, 1u) == 1u))  {
                            LOG_DBG("MAX96712: Link %u: GMSL1 config link lock after %u ms\n", linkIndex, (i * 10u));
                            success = 1;
                            break;
                        }
                        nvsleep(10000);
                    }
                    if (success == 1) {
                        success = 0;
                        break;
                    } else {
                        if (display) {
                            SIPL_LOG_ERR_STR_2INT("MAX96712: GMSL1 config link lock not detected", (int32_t)linkIndex, (int32_t)i);
                        }
                        return NVMEDIA_STATUS_ERROR;
                    }
                case CDI_MAX96712_LINK_LOCK_GMSL2:
                    if (!IsGMSL2Mode(gmslMode)) {
                        SIPL_LOG_ERR_STR_2INT("MAX96712: GMSL2 link lock is only valid in GMSL2 mode", (int32_t)linkIndex, (int32_t)gmslMode);
                        return NVMEDIA_STATUS_ERROR;
                    }

                    /* Only register 0x001A is available on MAX96712 Rev 1 to check
                     * link lock in GMSL2 mode*/
                    if ((drvHandle->ctx.revision == CDI_MAX96712_REV_1) &&
                                                    (linkIndex > 0U)) {
                        LOG_DBG("%s: GMSL2 link lock for link %u is not available on MAX96712 Rev 1\n",
                                 linkIndex);
                        return NVMEDIA_STATUS_OK;
                    }

                    /* From Max96712 programming guide V1.1, typical link rebuilding time is 25 ~ 100ms
                     * check the link lock in 100ms periodically
                     * TODO : Intermittently the link lock takes more than 100ms. Check it with MAXIM */
                    for (i = 0u; i < 50u; i++) {
                        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL2_LOCK_A + linkIndex,
                                                     0u,
                                                     REG_READ_MODE);

                        if (ReadFromRegFieldQ(handle, 0) == 1u)  {
                            LOG_DBG("MAX96712: Link %u: GMSL2 link lock after %u ms", linkIndex, (i * 10u));
                            success = 1;
                            break;
                        }
                        nvsleep(10000);
                    }
                    if (success == 1) {
                        if (i > 10) {
                            LOG_INFO("MAX96712: GMSL2 Link time %d\n", i * 10);
                        }
                        success = 0;
                        break;
                    } else {
                        if (display) {
                            SIPL_LOG_ERR_STR_INT("MAX96712: GMSL2 link lock not detected on link", (int32_t)linkIndex);
                        }
                        return NVMEDIA_STATUS_ERROR;
                    }
                case CDI_MAX96712_LINK_LOCK_VIDEO:
                    if (gmslMode == CDI_MAX96712_GMSL1_MODE) {
                        for (i = 0u; i < 10u; i++) {
                            ClearRegFieldQ(handle);
                            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_LOCK_A + linkIndex,
                                                      0u);
                            ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_VIDEO_LOCK_A + linkIndex,
                                                      0u);
                            ACCESS_REG_FIELD_RET_ERR(REG_READ_MODE);

                            if ((ReadFromRegFieldQ(handle, 0u) == 1u) &&
                                (ReadFromRegFieldQ(handle, 1u) == 1u))  {
                                LOG_DBG("MAX96712: Link %u: GMSL1 video lock after %u ms", linkIndex, (i * 10u));
                                success = 1;
                                break;
                            }
                            nvsleep(10000);
                        }
                        if (success == 1) {
                            success = 0;
                            break;
                        } else {
                            if (display) {
                                SIPL_LOG_ERR_STR_INT("MAX96712: GMSL1 video lock not detected on link", (int32_t)linkIndex);
                            }
                            return NVMEDIA_STATUS_ERROR;
                        }
                    } else if (IsGMSL2Mode(gmslMode)){
                        /* TODO: Check emb pipes if enabled */
                        for (i = 0u; i < 10u; i++) {
                            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_LOCK_PIPE_0 + linkIndex,
                                                         0u,
                                                         REG_READ_MODE);

                            if (ReadFromRegFieldQ(handle, 0u) == 1u)  {
                                LOG_DBG("MAX96712: Link %u: GMSL2 video lock after %u ms", linkIndex, (i * 10u));
                                success = 1;
                                break;
                            }
                            nvsleep(10000);
                        }

                        if (success == 1) {
                            success = 0;
                            break;
                        } else {
                            if (display) {
                                SIPL_LOG_ERR_STR_INT("MAX96712: GMSL2 video lock not detected on link", (int32_t)linkIndex);
                            }
                            return NVMEDIA_STATUS_ERROR;
                        }
                    }
                    break;
                default:
                    SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid link type");
                    return NVMEDIA_STATUS_BAD_PARAMETER;
            }
        }
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96712CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_ERROR;
    _DriverHandle *drvHandle = NULL;
    RevisionMAX96712 revision = CDI_MAX96712_REV_INVALID;
    uint8_t revisionVal = 0u;
    uint32_t numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);
    uint8_t devID = 0u;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712CheckPresence");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto done;
    }

    /* Check device ID */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_DEV_ID,
                                 0u,
                                 REG_READ_MODE);
    devID = ReadFromRegFieldQ(handle, 0u);
    if ((devID != MAX96712_DEV_ID) && (devID != MAX96722_DEV_ID)) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Device ID mismatch", (uint32_t)devID);
        SIPL_LOG_ERR_STR_UINT("MAX96712: Expected/Readval", MAX96712_DEV_ID);
        SIPL_LOG_ERR_STR_UINT("MAX96722: Expected/Readval", MAX96722_DEV_ID);
        status = NVMEDIA_STATUS_ERROR;
        goto done;
    }

    /* Check revision ID */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_DEV_REV,
                                 0u,
                                 REG_READ_MODE);
    revisionVal = ReadFromRegFieldQ(handle, 0u);
    for (i = 0u; i < numRev; i++) {
        if (revisionVal == supportedRevisions[i].revVal) {
            revision = supportedRevisions[i].revId;
            LOG_MSG("%s: Revision %u detected\n", ((devID == MAX96712_DEV_ID) ? "MAX96712" : "MAX96722"), revision);

            if (revision == CDI_MAX96712_REV_1) {
                LOG_MSG("MAX96712: Warning: MAX96712 revision 1 detected. All features may not be supported\n"
                        "Please use a platform with MAX96712 revision 2 or higher for full support\n");
                LOG_MSG("And the below error can be observed"
                        "  - FE_FRAME_ID_FAULT on CSIMUX_FRAME : Frame IDs are mismatched between FS and FE packets\n");
            }
            drvHandle->ctx.revision = revision;
            status = NVMEDIA_STATUS_OK;
            break;
        }
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_UINT("MAX96712: Unsupported MAX96712 revision detected!", (uint32_t)revisionVal);
        LOG_MSG("MAX96712: Unsupported MAX96712 revision %u detected! Supported revisions are:", revisionVal);
        for (i = 0u; i < numRev; i++) {
            LOG_MSG("MAX96712: Revision %u\n", supportedRevisions[i].revVal);
        }
        status = NVMEDIA_STATUS_NOT_SUPPORTED;
    }

done:
    return status;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *drvHandle = NULL;
    ContextMAX96712 *ctx = (ContextMAX96712 *) clientContext;
    uint8_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null handle");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (clientContext == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null client context");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    /* Check supplied context */
    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if ((ctx->gmslMode[i] != CDI_MAX96712_GMSL1_MODE) &&
            !IsGMSL2Mode(ctx->gmslMode[i]) &&
            (ctx->gmslMode[i] != CDI_MAX96712_GMSL_MODE_UNUSED)) {
            SIPL_LOG_ERR_STR("MAX96712: Invalid GMSL mode");
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    if ((ctx->i2cPort != CDI_MAX96712_I2CPORT_0) &&
        (ctx->i2cPort != CDI_MAX96712_I2CPORT_1) &&
        (ctx->i2cPort != CDI_MAX96712_I2CPORT_2)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid I2C port");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((ctx->txPort != CDI_MAX96712_TXPORT_PHY_C) &&
        (ctx->txPort != CDI_MAX96712_TXPORT_PHY_D) &&
        (ctx->txPort != CDI_MAX96712_TXPORT_PHY_E) &&
        (ctx->txPort != CDI_MAX96712_TXPORT_PHY_F)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid Tx port");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((ctx->mipiOutMode != CDI_MAX96712_MIPI_OUT_4x2) &&
        (ctx->mipiOutMode != CDI_MAX96712_MIPI_OUT_2x4) &&
        (ctx->mipiOutMode != CDI_MAX96712_MIPI_OUT_4a_2x2)) {
        SIPL_LOG_ERR_STR("MAX96712: Invalid MIPI output port");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = calloc(1, sizeof(_DriverHandle));
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Memory allocation for context failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    memcpy(&drvHandle->ctx, ctx, sizeof(ContextMAX96712));
    drvHandle->ctx.revision = CDI_MAX96712_REV_INVALID;
    drvHandle->ctx.manualFSyncFPS = 0u;
    handle->deviceDriverHandle = (void *)drvHandle;

    // Create the I2C programmer for register read/write
    drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                      MAX96712_NUM_ADDR_BYTES,
                                                      MAX96712_NUM_DATA_BYTES);
    if(drvHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Failed to initialize the I2C programmer");
        free(drvHandle);
        return NVMEDIA_STATUS_ERROR;
    }

    for (i = 0U; i < MAX96712_MAX_NUM_PG; i++) {
        drvHandle->ctx.pgMode[i] = CDI_MAX96712_PG_NUM;
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712DumpRegisters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    if (handle->deviceDriverHandle != NULL) {
        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }

    return NVMEDIA_STATUS_OK;
}
#if !NV_IS_SAFETY
NvMediaStatus
MAX96712DumpRegisters(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t data = 0;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0u; i <= MAX96712_REG_MAX_ADDRESS; i++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           ((i / 256u) << 8) | (i % 256u),
                                           &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("MAX96712: I2C read failed from register/with status", (int32_t)i, (int32_t)status);
            return status;
        }

        LOG_MSG("Max96712: 0x%04X%02X - 0x%02X\n", (i / 256u), (i % 256u), data);
    }

    return status;
}
#endif

NvMediaStatus
MAX96712GetErrorStatus(
    DevBlkCDIDevice *handle,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    ErrorStatusMAX96712 *errorStatus = (ErrorStatusMAX96712 *) parameter;
    uint8_t globalErrorCount = 0U;
    uint8_t linkErrorCount = 0U;
    uint8_t pipelineErrorCount = 0U;
    uint8_t linkNum = 0u;
    uint8_t pipelineNum = 0u;
    bool pipelineErrAppears = false;
    uint8_t regFieldVal = 0u;
    uint8_t i = 0u;

    if (MAX96712_MAX_LINK_BASED_ERROR_NUM < CDI_MAX96712_MAX_LINK_BASED_FAILURE_TYPES) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: max link based error found smaller than failure types");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null handle in MAX96712GetErrorStatus");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null driver handle in MAX96712GetErrorStatus");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null parameter storage in MAX96712GetErrorStatus");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameterSize != sizeof(ErrorStatusMAX96712)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Incorrect param size in MAX96712GetErrorStatus");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    memset(errorStatus, 0u, sizeof(ErrorStatusMAX96712));

    /* MAX96712_REG_GMSL1_LINK_A read back as 0 without this delay when any link is powered down */
    nvsleep(5000);

    /* ctrl3 (R0x1A)
     * intr5[3]: LOCKED
     * intr5[2]: ERROR
     * intr5[1]: CMU_LOCKED
     * rest bits are reserved. */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_CTRL3,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x4) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global ERRB status", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_ERR, false);
    } else {
        /* error disappears, or false alarm, or some other cases not yet handled. Show message and return */
        LOG_INFO("MAX96712: not supported case found: global ERRB not asserted while GetErrorStatus is called\n");
        return NVMEDIA_STATUS_OK;
    }

    if ((regFieldVal & 0x8) == 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global locked status", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_UNLOCK_ERR, false);
    }
    if ((regFieldVal & 0x2) == 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global CMU locked status", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_CMU_UNLOCK_ERR, false);
    }

    /* intr5 (R0x28)
     * intr5[2]: LFLT_INT
     * (rest bits are for RTTN_CRC_INT, WM, EOM link based errors) */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR5,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x4) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global line fault error in bit 2", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_LINE_FAULT, false);
    }

    /* intr7 (R0x2A)
     * intr7[3]: LCRC_ERR_FLAG
     * intr7[2]: VPRBS_ERR_FLAG
     * intr7[1]: REM_ERR_FLAG
     * intr7[0]: FSYNC_ERR_FLAG
     * (rerst bits are for G1 link based errors, note we use R0xBCB than R0xB etc in later code) */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR7,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x8) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global video line crc error in bit 3", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VID_LINE_CRC, false);
    }
    if ((regFieldVal & 0x4)!= 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global video PRBS error in bit 2", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VID_PRBS, false);
    }
    if ((regFieldVal & 0x2)!= 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global remote side error in bit 1", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_REMOTE_SIDE, false);
    }
    if ((regFieldVal & 0x1)!= 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global frame sync error in bit 0", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_FRAME_SYNC, false);
    }

    /* vid_pxl_crc_err_int (R0x45)
     * vid_pxl_crc_err_int[7]: mem ecc 2
     * vid_pxl_crc_err_int[6]: mem ecc 1
     * (rest bits are for video pixel crc link based errors) */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VID_PXL_CRC_ERR_INT,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x80) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global mem ecc 2 error in bit 7", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_MEM_ECC2, false);
    }
    if ((regFieldVal & 0x40) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global mem ecc error in bit 6", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_MEM_ECC1, false);
    }

    /* fsync_22 (R0x4B6)
     * fsync_22[7]: FSYNC_LOSS_OF_LOCK
     * fsync_22[6]: FSYNC_LOCKED
     * (rest 6 bits are for FRM_DIFF_H, currently not to report) */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_FSYNC_22,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x80) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global fsync sync loss error in bit 7", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_FSYNC_SYNC_LOSS, false);
    }
    if ((regFieldVal & 0x40) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: global fsync status in bit 6", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_FSYNC_STATUS, false);
    }

    /* VIDEO_MASKED_FLAG (R0x04A)
     * VIDEO_MASKED_FLAG[5]: CMP_VTERM_STATUS
     * VIDEO_MASKED_FLAG[4]: VDD_OV_FLAG */
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_MASKED_FLAG,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x20) == 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vterm is latched low and less than 1v, in video mask reg bit 5", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_CMP_VTERM_STATUS, false);
    }
    if ((regFieldVal & 0x10) != 0u) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vdd_sw overvoltage condition detected, in video masked reg bit 4", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VDD_OV_FLAG, false);
    }

    /* PWR0 (R0x012)
     * PWR0[7:5]: VDDBAD_STATUS with bits 5 and bit 6 are effectively used.
     * PWR0[4:0]: CMP_STATUS, with bit 0,1,2 are for Vdd18/Vddio/Vdd_sw undervoltage latch low indicator */
    bool readAgainPwr0 = false;
    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_PWR0,
                                 0u,
                                 REG_READ_MODE);
    regFieldVal = ReadFromRegFieldQ(handle, 0u);
    if ((regFieldVal & 0x60U) == 0x60U) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vdd_sw less than 0.82v is observed since last read", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VDDBAD_STATUS, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x4U) == 0U) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vdd_sw (1.0v) is latched low (undervoltage), reg value", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VDDSW_UV, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x2U) == 0U) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vddio (1.8v) is latched low (undervoltage), reg value:", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VDDIO_UV, false);
        readAgainPwr0 = true;
    }
    if ((regFieldVal & 0x1U) == 0U) {
        SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: Vdd 1.8v is latched low (undervoltage), reg value:", (uint32_t)regFieldVal);
        UPDATE_GLOBAL_ERROR(CDI_MAX96712_GLOBAL_VDD18_UV, true);
        readAgainPwr0 = true;
    }
    if (readAgainPwr0) {
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_PWR0,
                                     0u,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0u);
        if (regFieldVal == 0x0U) {
            LOG_INFO("MAX96712: all undervoltage latches in PWR0 are cleared");

            // further read clear PWR_STATUS_FLAG (VDDBAD_INT_FLAG and VDDCMP_INT_FLAG)
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_PWR_STATUS_FLAG,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            LOG_INFO("MAX96712: read clear PWR_STATUS_FLAG (%u)", regFieldVal);
        } else {
            // PWR0 are not cleared, and PWR_STATUS_FLAG will still flag ERRB
            SIPL_LOG_ERR_STR("MAX96712: not all undervoltage latches are cleared!");
        }
    }

    for (pipelineNum = 0u; pipelineNum < MAX96712_NUM_VIDEO_PIPELINES; pipelineNum++) {
        pipelineErrorCount = 0U;

        // overflow
        errorStatus->pipelineFailureType[pipelineNum][pipelineErrorCount] = CDI_MAX96712_PIPELINE_ERROR_INVALID;
        if (pipelineNum <= 3U) {
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OVERFLOW_FIRST4,
                                         0u,
                                         REG_READ_MODE);
        } else { /* pipelineNum >= 4U && pipelineNum <= 7U) */
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OVERFLOW_LAST4,
                                         0U,
                                         REG_READ_MODE);
        }

        regFieldVal = ReadFromRegFieldQ(handle, 0u);
        if (((regFieldVal) & (uint8_t)(1U << pipelineNum)) ||
            ((uint8_t)(regFieldVal >> 4U) & (uint8_t)(1U << pipelineNum))) {
            SIPL_LOG_ERR_STR_INT("MAX96712: pipeline overflow", (int32_t)pipelineNum);
            pipelineErrAppears = true;

            /* Check overflow status every 1ms periodically */
            for (i = 0u; i < 100u; i++) {
                if (pipelineNum <= 3U) {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OVERFLOW_FIRST4,
                                                 0u,
                                                 REG_READ_MODE);
                } else {
                    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OVERFLOW_LAST4,
                                                 0u,
                                                 REG_READ_MODE);
                }

                regFieldVal = ReadFromRegFieldQ(handle, 0u);
                if (((regFieldVal &
                    (uint8_t)(1U << pipelineNum)) == 0U) &&
                    (((uint8_t)(regFieldVal >> 4U) &
                    (uint8_t)(1U << pipelineNum)) == 0U)) {
                    SIPL_LOG_ERR_STR_2INT("MAX96712: overflow disappear after x ms on link y", (int32_t)i, (int32_t)pipelineNum);
                    pipelineErrAppears = false;
                    break;
                }
                nvsleep(1000);
            }

            if (pipelineErrAppears) {
                // line memory overflow bits are at BACKTOP11 register's bit[3:0]
                if (regFieldVal & (uint8_t)(1U << pipelineNum)) {
                    SIPL_LOG_ERR_STR_UINT("MAX96712: lmo overflow error for pipeline:", pipelineNum);
                    UPDATE_PIPELINE_ERROR(CDI_MAX96712_PIPELINE_LMO_OVERFLOW_ERR, false);
                }
                // cmd overflow bits are at BACKTOP11 register's bit[7:4]
                if ((uint8_t)(regFieldVal >> 4) & (uint8_t)(1U << pipelineNum)) {
                    SIPL_LOG_ERR_STR_UINT("MAX96712: cmd overflow error for pipeline:", pipelineNum);
                    UPDATE_PIPELINE_ERROR(CDI_MAX96712_PIPELINE_CMD_OVERFLOW_ERR, false);
                }
            }
        }

        // pipe pattern generator video lock status, register 0x1DC etc's bit 0, defined by 8 contiguous enums.
        errorStatus->pipelineFailureType[pipelineNum][pipelineErrorCount] = CDI_MAX96712_PIPELINE_ERROR_INVALID;
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_PATGEN_CLK_SRC_PIPE_0 + pipelineNum,
                                     0u,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0u);
        if ((regFieldVal & 0x1) == 0u) {
            SIPL_LOG_ERR_STR_INT("MAX96712: video unlock on pipeline", (int32_t)pipelineNum);
            UPDATE_PIPELINE_ERROR(CDI_MAX96712_PIPELINE_PGEN_VID_UNLOCK_ERR, false);
        }

        // mem_err
        errorStatus->pipelineFailureType[pipelineNum][pipelineErrorCount] = CDI_MAX96712_PIPELINE_ERROR_INVALID;
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_BACKTOP25,
                                     0u,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0u);
        if ((regFieldVal & (0x1 << pipelineNum)) != 0u) {
            SIPL_LOG_ERR_STR_INT("MAX96712: line memory error on pipeline", (int32_t)pipelineNum);
            UPDATE_PIPELINE_ERROR(CDI_MAX96712_PIPELINE_MEM_ERR, false);
        }

        // video sequence error status, register 0x108 etc's bit 4, defined by 8 contiguous enums.
        errorStatus->pipelineFailureType[pipelineNum][pipelineErrorCount] = CDI_MAX96712_PIPELINE_ERROR_INVALID;
        ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_RX8_PIPE_0 + pipelineNum,
                                     0u,
                                     REG_READ_MODE);
        regFieldVal = ReadFromRegFieldQ(handle, 0u);
        if ((regFieldVal & 0x10) != 0u) {
            SIPL_LOG_ERR_STR_INT("MAX96712: video sequence error on pipeline", (int32_t)pipelineNum);
            UPDATE_PIPELINE_ERROR(CDI_MAX96712_PIPELINE_VID_SEQ_ERR, true);
        }
    }

    for (linkNum = 0u; linkNum < MAX96712_MAX_NUM_LINK; linkNum++) {
        linkErrorCount = 0U;
        errorStatus->linkFailureType[linkNum][linkErrorCount] = CDI_MAX96712_GMSL_LINK_ERROR_INVALID;

        // GMSL1/GMSL2 link based errors to be reported
        if (drvHandle->ctx.gmslMode[linkNum] == CDI_MAX96712_GMSL1_MODE) {
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_LOCK_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if (regFieldVal != 1u) {
                SIPL_LOG_ERR_STR_INT("MAX96712: GMSL1 link unlocked on link", (int32_t)linkNum);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL1_LINK_UNLOCK_ERR, false);
            }

            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL1_DET_ERR_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                SIPL_LOG_ERR_STR_2UINT("MAX96712: GMSL1 decoding error on pipeline", (uint32_t)linkNum, (uint32_t)regFieldVal);
                SIPL_LOG_ERR_STR_2UINT("MAX96712: Link &  GMSL1 decoding error :", linkNum, regFieldVal);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL1_LINK_DET_ERR, false);
            }

            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_CC_CRC_ERRCNT_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                SIPL_LOG_ERR_STR_2UINT("MAX96712: GMSL1 PKTCC CRC failure on link", (uint32_t)linkNum, (uint32_t)regFieldVal);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL1_LINK_PKTCC_CRC_ERR, true);
            }
        } else if (IsGMSL2Mode(drvHandle->ctx.gmslMode[linkNum])) {
            // link lock err
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL2_LOCK_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if (regFieldVal == 0u) {
                SIPL_LOG_ERR_STR_UINT("MAX96712: GMSL2 link unlocked", (uint32_t)linkNum);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_UNLOCK_ERR, false);
            }

            // dec err
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL2_DEC_ERR_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                SIPL_LOG_ERR_STR_2UINT("MAX96712: GMSL2 decoding error", (uint32_t)linkNum, (uint32_t)regFieldVal);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_DEC_ERR, false);
            }

            // idle err
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GMSL2_IDLE_ERR_A + linkNum,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                SIPL_LOG_ERR_STR_2UINT("MAX96712: GMSL2 idle error", (uint32_t)linkNum,  (uint32_t)regFieldVal);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_IDLE_ERR, false);
            }

            // EOM error (intr5, bit[7:4])
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR5,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if (((regFieldVal & (0x1 << (linkNum + 4u))) != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                SIPL_LOG_ERR_STR_2UINT("MAX96712: Link eye open monitor error", (uint32_t)linkNum, (uint32_t)regFieldVal);
                UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_EOM_ERR, false);
            }

            // ARQ errors (intr11)
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_INTR11,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal != 0u) &&
                (errorStatus->linkFailureType[linkNum][linkErrorCount] == CDI_MAX96712_GMSL_LINK_ERROR_INVALID)) {
                if ((regFieldVal & (1 << (linkNum + 4u))) != 0u) {
                    SIPL_LOG_ERR_STR_2UINT("MAX96712: Combined ARQ transmission error", (uint32_t)linkNum, (uint32_t)regFieldVal);
                    UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_ARQ_RETRANS_ERR, false);
                }
                if ((regFieldVal & (1U << linkNum)) != 0u) {
                    SIPL_LOG_ERR_STR_2UINT("MAX96712: Combined ARQ max transmission error", (uint32_t)linkNum, (uint32_t)regFieldVal);
                    UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_MAX_RETRANS_ERR, false);
                }
            }

            // vid_pxl_crc_err_int (R0x45[[3:0])
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_VID_PXL_CRC_ERR_INT,
                                         0u,
                                         REG_READ_MODE);
            regFieldVal = ReadFromRegFieldQ(handle, 0u);
            if ((regFieldVal & 0x0f) != 0u) {
                if ((regFieldVal & (uint8_t)(1U << linkNum)) != 0u) {
                    SIPL_LOG_ERR_STR_2UINT("MAX96712: Video pixel crc count", (uint32_t)linkNum, (uint32_t)regFieldVal);
                    UPDATE_LINK_ERROR(CDI_MAX96712_GMSL2_LINK_VIDEO_PXL_CRC_ERR, true);
                }
            }
        }
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96712GetSerializerErrorStatus(DevBlkCDIDevice *handle,
                                 bool * isSerError)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Null handle passed to MAX96712GetSerializerErrorStatus");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_REM_ERR_FLAG,
                                 0u,
                                 REG_READ_MODE);
    if (ReadFromRegFieldQ(handle, 0u) == 1u) {
        *isSerError = true;
    }

    return status;
}

NvMediaStatus
MAX96712ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    bool isValidSize = false;
    ReadParametersParamMAX96712 *param = (ReadParametersParamMAX96712 *) parameter;
    static const char *cmdString[] = {
        [CDI_READ_PARAM_CMD_MAX96712_REV_ID] =
            "CDI_READ_PARAM_CMD_MAX96712_REV_ID",
        [CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR] =
            "CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR",
        [CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS] =
            "CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS",
        [CDI_READ_PARAM_CMD_MAX96712_ERRB] =
            "CDI_READ_PARAM_CMD_MAX96712_ERRB",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712ReadParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle pased to MAX96712ReadParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Bad driver parameter: Null ptr");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_READ_PARAM_CMD_MAX96712_INVALID) ||
        (parameterType >= CDI_READ_PARAM_CMD_MAX96712_NUM)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    LOG_DBG("MAX96712: %s", cmdString[parameterType]);
    switch (parameterType) {
        case CDI_READ_PARAM_CMD_MAX96712_REV_ID:
            if (parameterSize == sizeof(param->revision)) {
                isValidSize = true;
                param->revision = drvHandle->ctx.revision;
                status = NVMEDIA_STATUS_OK;
            }
            break;
        case CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR:
            if (parameterSize == sizeof(param->ErrorStatus)) {
                isValidSize = true;
                status = ReadCtrlChnlCRCErr(handle,
                                            param->ErrorStatus.link,
                                            &param->ErrorStatus.errVal);
            }
            break;
        case CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = GetEnabledLinks(handle,
                                         &param->link);
            }
            break;
        case CDI_READ_PARAM_CMD_MAX96712_ERRB:
            if (parameterSize == sizeof(param->ErrorStatus)) {
                isValidSize = true;
                status = ClearErrb(handle,
                                   &param->ErrorStatus.link,
                                   &param->ErrorStatus.errVal);
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Unhandled command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX96712: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("MAX96712: Bad parameter: Invalid param size", cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

NvMediaStatus
MAX96712ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712ReadRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712ReadRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dataBuff == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null data buffer passed to MAX96712ReadRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (i = 0; i < dataLength; i ++) {
        status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                           registerNum,
                                           &dataBuff[i]);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2UINT("MAX96712: Register read failed with status", registerNum, (uint32_t)status);
        }
    }

    return status;
}

static NvMediaStatus
GMSL2LinkAdaptation(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    GMSLModeMAX96712 gmslMode = CDI_MAX96712_GMSL_MODE_INVALID;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    RevisionMAX96712 rev = drvHandle->ctx.revision;
    uint8_t regVal = 0u, i = 0u, loop = 0u;

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(link, i)) {
            gmslMode = drvHandle->ctx.gmslMode[i];

            if (!IsGMSL2Mode(gmslMode)) {
                LOG_INFO("MAX96712: Link %d: adaptation is required only in GMSL2 mode\n", i);
                continue;
            }

            /* Disable OSN */
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_ENABLE_OSN_0 + i,
                                         0u,
                                         REG_READ_MOD_WRITE_MODE);

            /* Reseed and set to default value 31 */
            ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OSN_COEFF_MANUAL_SEED_0 + i,
                                         1u,
                                         REG_READ_MOD_WRITE_MODE);

            nvsleep(10000);

            for (loop = 0; loop < 100; loop++) {
                /* Read back OSN value */
                ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_OSN_COEFFICIENT_0 + i, 0u, REG_READ_MODE);
                regVal = ReadFromRegFieldQ(handle, 0u);
                if (regVal == 31) {
                    break;
                }
                nvsleep(1000);
            }
            (void)rev;
            LOG_MSG("MAX96712 Rev %d manual adaptation on the link %d (%d)\n", rev,
                                                                               i,
                                                                               regVal);
        }
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
EnableMemoryECC(
    DevBlkCDIDevice *handle,
    bool enable2bitReport,
    bool enable1bitReport)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg memECCReg = {0x0044, 0x0F};

    if (drvHandle->ctx.revision < CDI_MAX96712_REV_3) {
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    if (enable2bitReport) {
        memECCReg.data |= (uint8_t)(1U << 7);
    }
    if (enable1bitReport) {
        memECCReg.data |= (uint8_t)(1U << 6);
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        memECCReg.address,
                                        memECCReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return status;
}

static NvMediaStatus
SetCRUSSCModes(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg CRUSSCMode0 = {0x1445U, 0x0U};
    DevBlkCDII2CReg CRUSSCMode1 = {0x1545U, 0x0U};
    DevBlkCDII2CReg CRUSSCMode2 = {0x1645U, 0x0U};
    DevBlkCDII2CReg CRUSSCMode3 = {0x1745U, 0x0U};

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        CRUSSCMode0.address,
                                        (uint8_t)CRUSSCMode0.data);

    if (status != NVMEDIA_STATUS_OK)
        return status;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        CRUSSCMode1.address,
                                        (uint8_t)CRUSSCMode1.data);

    if (status != NVMEDIA_STATUS_OK)
        return status;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        CRUSSCMode2.address,
                                        (uint8_t)CRUSSCMode2.data);

    if (status != NVMEDIA_STATUS_OK)
        return status;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        CRUSSCMode3.address,
                                        (uint8_t)CRUSSCMode3.data);

    if (status != NVMEDIA_STATUS_OK)
        return status;

    return status;
}

static NvMediaStatus
CheckCSIPLLLock(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaBool passiveEnabled = drvHandle->ctx.passiveEnabled;

    uint8_t i = 0u;
    MipiOutModeMAX96712 mipiOutMode;
    DevBlkCDII2CReg CSIPllLockReg = {0x0400, 0x00};
    uint8_t data = 0;

    mipiOutMode = drvHandle->ctx.mipiOutMode;

    if (!passiveEnabled) {
        for (i = 0u; i < 20u; i++) {
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               CSIPllLockReg.address,
                                               &data);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }

            if (((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) && ((data & 0xF0) == 0x60)) ||
                ((mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) && ((data & 0xF0) == 0xF0))) {
                break;
            }
            nvsleep(10000);
        }

        if (((mipiOutMode == CDI_MAX96712_MIPI_OUT_2x4) && ((data & 0xF0) != 0x60)) ||
            ((mipiOutMode == CDI_MAX96712_MIPI_OUT_4x2) && ((data & 0xF0) != 0xF0))) {
            SIPL_LOG_ERR_STR_HEX_UINT("MAX96712: CSI PLL unlock", (uint32_t)(data & 0xF0));
            return NVMEDIA_STATUS_ERROR;
        }
    }

    return status;
}

static NvMediaStatus
GMSL2PHYOptimizationRevE(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t i = 0u;
    DevBlkCDII2CReg increaseCMUOutVoltageReg = {0x06C2, 0x10}; /* Increase CMU regulator output voltage (bit 4) */
    DevBlkCDII2CReg vgaHiGain_InitReg = {0x14D1, 0x03}; /* Set VgaHiGain_Init_6G (bit 1) and VgaHiGain_Init_3G (bit 0) */

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        increaseCMUOutVoltageReg.address,
                                        increaseCMUOutVoltageReg.data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: Failed to increase CMU output voltage", (int32_t)status);
        return status;
    }

    for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
        if ((MAX96712_IS_GMSL_LINK_SET(link, i)) &&
            (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))) {
            vgaHiGain_InitReg.address = 0x14D1 + (i * 0x100);
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                vgaHiGain_InitReg.address,
                                                vgaHiGain_InitReg.data);
            if (status != NVMEDIA_STATUS_OK) {
                SIPL_LOG_ERR_STR_2INT("MAX96712: Failed to set VgaHighGain_Init on link", (int32_t)i, (int32_t)status);
                return status;
            }

            LOG_MSG("MAX96712 Link %d: PHY optimization was enabled\n", i);
        }
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
GMSL2PHYOptimization(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t i = 0u;

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to GMSL2PHYOptimization");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to GMSL2PHYOptimization");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    // If any link is configured in GMSL2 mode, execute the link adaptation
    if ((drvHandle->ctx.revision == CDI_MAX96712_REV_2) ||
        (drvHandle->ctx.revision == CDI_MAX96712_REV_3)){
        for (i = 0u; i < MAX96712_MAX_NUM_LINK; i++) {
            if (drvHandle->ctx.gmslMode[i] != CDI_MAX96712_GMSL_MODE_UNUSED) {
                if (MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) {
                    status = ConfigTxAmpTiming(handle, (LinkMAX96712)(1 << i));
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }

                    status = GMSL2LinkAdaptation(handle, (LinkMAX96712)(1 << i));
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            }
        }
    } else if (drvHandle->ctx.revision == CDI_MAX96712_REV_5) {
        for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
            if (drvHandle->ctx.gmslMode[i] != CDI_MAX96712_GMSL_MODE_UNUSED) {
                if ((MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) &&
                    (IsGMSL2Mode(drvHandle->ctx.gmslMode[i]))) {
                    status = GMSL2PHYOptimizationRevE(handle, (LinkMAX96712)(1 << i));
                    if (status != NVMEDIA_STATUS_OK) {
                        return status;
                    }
                }
            }
        }
    }

    return status;
}

static NvMediaStatus
EnableGPIORx(
    DevBlkCDIDevice *handle,
    uint8_t gpioIndex)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg setGPIOMode = {0x0300, 0x1C}; /* pull-up 1M ohm, GPIO source en for GMSL2 */
    uint8_t data = 0;

    setGPIOMode.address += (gpioIndex * 3u);
    /* 0x30F, 0x31F, 0x32F are not used */
    setGPIOMode.address += ((setGPIOMode.address & 0xFF) > 0x2E) ? 3 :
                      (((setGPIOMode.address & 0xFF) > 0x1E) ? 2 :
                      (((setGPIOMode.address & 0xFF) > 0xE) ? 1 : 0));
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       setGPIOMode.address,
                                       &data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    data |= 0x4; /* Set GPIO_RX_EN */
    data &= ~0x3; /* Unset GPIO_TX_EN, GPIO_OUT_DIS */

    return DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                      setGPIOMode.address,
                                      data);
}

static NvMediaStatus
MAX96712DisableLink(
    DevBlkCDIDevice const* handle,
    uint8_t link,
    size_t paramSize,
    size_t pipelink_sz)
{
    _DriverHandle *drvHandle = NULL;
    NvMediaStatus status;

    uint8_t regVal = 0U;
    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    /* Disable Video Pipeline */
    status = DevBlkCDII2CPgmrReadUint8(
            drvHandle->i2cProgrammer,
            (uint16_t)(0xF4U),
            &regVal);
    if (status == NVMEDIA_STATUS_OK ) {
        regVal &= (uint8_t)(~(0x11U << link));
        status = DevBlkCDII2CPgmrWriteUint8(
                drvHandle->i2cProgrammer,
                (uint16_t)(0xF4U),
                regVal);
        if (status == NVMEDIA_STATUS_OK ) {
            /* Disable ERRB Rx */
            regVal = 0U;
            status = DevBlkCDII2CPgmrReadUint8(
                drvHandle->i2cProgrammer,
                (uint16_t)(0x30U + link),
                &regVal);
            if (status == NVMEDIA_STATUS_OK ) {
                regVal &= 0x7FU;
                status = DevBlkCDII2CPgmrWriteUint8(
                    drvHandle->i2cProgrammer,
                    (uint16_t)(0x30U + link),
                    regVal);
            }
        }
    }

    return status;
}

static NvMediaStatus
MAX96712RestoreLink(
    DevBlkCDIDevice const* handle,
    uint8_t link,
    size_t paramSize,
    size_t pipelink_sz)
{
    _DriverHandle *drvHandle = NULL;
    NvMediaStatus status;

    uint8_t regVal = 0U;
    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    /* Restore Video Pipeline Enable bits*/
    status = DevBlkCDII2CPgmrReadUint8(
            drvHandle->i2cProgrammer,
            (uint16_t)(0xF4U),
            &regVal);
    if (status == NVMEDIA_STATUS_OK ) {
        regVal |= (drvHandle->ctx.pipelineEnabled & (0x11U << link));
        status = DevBlkCDII2CPgmrWriteUint8(
            drvHandle->i2cProgrammer,
            (uint16_t)(0xF4U),
            regVal);
        if (status == NVMEDIA_STATUS_OK ) {
            /* Restore ERRB Rx Enable bit*/
            regVal = 0U;
            status = DevBlkCDII2CPgmrReadUint8(
                drvHandle->i2cProgrammer,
                (uint16_t)(0x30U + link),
                &regVal);
            if (status == NVMEDIA_STATUS_OK ) {
                status = DevBlkCDII2CPgmrWriteUint8(
                    drvHandle->i2cProgrammer,
                    (uint16_t)(0x30U + link),
                    regVal);
            }
        }
    }

    return status;
}

static NvMediaStatus
EnableIndividualReset(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    DevBlkCDII2CReg individualReset = {0x06DFU, 0x7FU};


    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (drvHandle->ctx.revision >= CDI_MAX96712_REV_3) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            individualReset.address,
                                            (uint8_t)individualReset.data);
    }

    return status;
}

NvMediaStatus
MAX96712SetDefaults(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint8_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (drvHandle->ctx.revision == CDI_MAX96712_REV_3) {
        /* Bug 2446492: Disable 2-bit ECC error reporting as spurious ECC errors are
         * intermittently observed on Rev C of MAX96712
         * Disable reporting 2-bit ECC errors to ERRB
         */
        status = EnableMemoryECC(handle, false, false);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    } else if (drvHandle->ctx.revision >= CDI_MAX96712_REV_4) {
        status = EnableMemoryECC(handle, true, true);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    status = EnableIndividualReset(handle);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712SetDefaults: SetCRUSSCModes failed", (int32_t)status);
        return status;
    }

    if (drvHandle->ctx.revision == CDI_MAX96712_REV_5) {
        status = SetCRUSSCModes(handle);

        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712SetDefaults: SetCRUSSCModes failed", (int32_t)status);
            return status;
        }
    }

    status = SetLinkMode(handle, (LinkMAX96712)(drvHandle->ctx.linkMask));
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX96712: MAX96712SetDefaults: SetLinkMode failed", (int32_t)status);
        return status;
    }

    status = GMSL2PHYOptimization(handle);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Default mode is GMSL2, 6Gbps
     * one shot reset is required for GMSL1 mode & GMSL2
     */
    status = MAX96712OneShotReset(handle, (LinkMAX96712)(drvHandle->ctx.linkMask));
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
        if (MAX96712_IS_GMSL_LINK_SET(drvHandle->ctx.linkMask, i)) {
            if (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL1_MODE) {
                /* HIM mode is not enabled yet so the link lock will not be set
                 * Instead use sleep function */
                nvsleep(100000);
            } else if ((drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL2_MODE_3GBPS) ||
                       (drvHandle->ctx.gmslMode[i] == CDI_MAX96712_GMSL2_MODE_6GBPS)) {
                status = MAX96712CheckLink(handle, drvHandle->ctx.linkMask,CDI_MAX96712_LINK_LOCK_GMSL2, true);
                if (status != NVMEDIA_STATUS_OK) {
                    return status;
                }
            }
        }
    }

    for (i = 0; i < MAX96712_MAX_NUM_LINK; i++) {
        if ((IsGMSL2Mode(drvHandle->ctx.gmslMode[i])) &&
            (drvHandle->ctx.longCables[i] == true)) {
            status = UpdateVGAHighGain(handle, (LinkMAX96712)(1 << i), drvHandle->ctx.longCables[i]);
            if (status != NVMEDIA_STATUS_OK) {
                return status;
            }
        }
    }

    status = SetI2CPort(handle);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    /* Disable all pipelines*/
    ClearRegFieldQ(handle);
    for (i = 0u; i < MAX96712_NUM_VIDEO_PIPELINES; i++) {
        ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_VIDEO_PIPE_EN_0 + i,
                                  0u);
    }
    ACCESS_REG_FIELD_RET_ERR(REG_WRITE_MODE);

    // Enable extra SMs
    if (drvHandle->ctx.revision >= CDI_MAX96712_REV_4) {
        status = EnableExtraSMs(handle);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
MAX96712SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    static const char *cmdString[] = {
        [CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE] =
            "CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE",
        [CDI_CONFIG_MAX96712_ENABLE_PG] =
            "CDI_CONFIG_MAX96712_ENABLE_PG",
        [CDI_CONFIG_MAX96712_ENABLE_CSI_OUT] =
            "CDI_CONFIG_MAX96712_ENABLE_CSI_OUT",
        [CDI_CONFIG_MAX96712_DISABLE_CSI_OUT] =
            "CDI_CONFIG_MAX96712_DISABLE_CSI_OUT",
        [CDI_CONFIG_MAX96712_TRIGGER_DESKEW] =
            "CDI_CONFIG_MAX96712_TRIGGER_DESKEW",
        [CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK] =
            "CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK",
        [CDI_CONFIG_MAX96712_ENABLE_REPLICATION] =
            "CDI_CONFIG_MAX96712_ENABLE_REPLICATION",
        [CDI_CONFIG_MAX96712_DISABLE_REPLICATION] =
            "CDI_CONFIG_MAX96712_DISABLE_REPLICATION",
    };

    if ((handle == NULL)) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((enumeratedDeviceConfig == CDI_CONFIG_MAX96712_INVALID) ||
        (enumeratedDeviceConfig >= CDI_CONFIG_MAX96712_NUM)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    LOG_DBG("MAX96712: %s", cmdString[enumeratedDeviceConfig]);
    switch (enumeratedDeviceConfig) {
        case CDI_CONFIG_MAX96712_ENABLE_PG:
            status = EnablePG(handle);
            break;
        case CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE:
            status = MapUnusedPipe(handle);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_CSI_OUT:
            status = EnableCSIOut(handle,
                                  true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_CSI_OUT:
            status = EnableCSIOut(handle,
                                  false);
            break;
        case CDI_CONFIG_MAX96712_TRIGGER_DESKEW:
            status = TriggerDeskew(handle);
            break;
        case CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK:
            status = CheckCSIPLLLock(handle);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_REPLICATION:
            status = EnableReplication(handle, true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_REPLICATION:
            status = EnableReplication(handle, false);
            break;
        case CDI_CONFIG_MAX96712_ENABLE_ERRB:
            status = EnableERRB(handle, true);
            break;
        case CDI_CONFIG_MAX96712_DISABLE_ERRB:
            status = EnableERRB(handle, false);
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Unrecognized command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX96712: command failed:", cmdString[enumeratedDeviceConfig]);
    }

    return status;
}

NvMediaStatus
MAX96712WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712WriteRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712WriteRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dataBuff == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null data buffer passed to MAX96712WriteRegister");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (dataLength > REG_WRITE_BUFFER_BYTES) {
        SIPL_LOG_ERR_STR("MAX96712: Insufficient buffer size in MAX96712WriteRegister");
        return NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
    }

    for (i = 0; i < dataLength; i++) {
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            registerNum,
                                            dataBuff[i]);
    }
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2UINT("MAX96712: Register I2C write failed with status", registerNum, (uint32_t)status);
    }

    return status;
}

NvMediaStatus
MAX96712WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    size_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    WriteParametersParamMAX96712 *param = (WriteParametersParamMAX96712 *) parameter;
    bool isValidSize = false;
    static const char *cmdString[] = {
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS",
        [CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS] =
            "CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS",
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS",
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL",
        [CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE] =
            "CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC",
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE",
        [CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING] =
            "CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING",
        [CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG] =
            "CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG",
        [CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE] =
            "CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID",
        [CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL] =
            "CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL",
        [CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL] =
            "CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL",
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ",
        [CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK] =
            "CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK",
        [CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX] =
            "CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX",
        [CDI_WRITE_PARAM_CMD_MAX96712_SET_PG] =
            "CDI_WRITE_PARAM_CMD_MAX96712_SET_PG",
        [CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK] =
            "CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK",
        [CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK] =
            "CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null handle passed to MAX96712WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null driver handle passed to MAX96712WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("MAX96712: Null parameter storage passed to MAX96712WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_WRITE_PARAM_CMD_MAX96712_INVALID) ||
        (parameterType >= CDI_WRITE_PARAM_CMD_MAX96712_NUM)) {
        SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    LOG_DBG("MAX96712: %s", cmdString[parameterType]);
    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING:
            if (parameterSize == sizeof(param->PipelineMapping)) {
                isValidSize = true;
                status = SetPipelineMap(handle,
                                        param->PipelineMapping.link,
                                        param->PipelineMapping.linkPipelineMap,
                                        param->PipelineMapping.isSinglePipeline);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG:
            if (parameterSize == sizeof(param->PipelineMappingTPG)) {
                isValidSize = true;
                status = SetPipelineMapTPG(handle,
                                           param->PipelineMappingTPG.linkIndex,
                                           param->PipelineMappingTPG.linkPipelineMap);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE:
            if (parameterSize == sizeof(param->PipelineMapping)) {
                isValidSize = true;
                status = OverrideDataType(handle,
                                          param->PipelineMapping.link,
                                          param->PipelineMapping.linkPipelineMap);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC:
            if (parameterSize == sizeof(param->FSyncSettings)) {
                isValidSize = true;
                status = SetFSYNCMode(handle,
                                      param->FSyncSettings.FSyncMode,
                                      param->FSyncSettings.pclk,
                                      param->FSyncSettings.fps,
                                      param->FSyncSettings.link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = EnableSpecificLinks(handle,
                                             param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = ControlForwardChannels(handle,
                                                   param->link,
                                                   false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = ControlForwardChannels(handle,
                                                   param->link,
                                                   true);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = EnablePacketBasedControlChannel(handle,
                                                         param->link,
                                                         true);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = DisableDE(handle,
                                   param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED:
            if (parameterSize == sizeof(param->GMSL1HIMEnabled)) {
                isValidSize = true;
                status = SetDefaultGMSL1HIMEnabled(handle,
                                                   param->GMSL1HIMEnabled.link,
                                                   param->GMSL1HIMEnabled.step);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = SetDBL(handle,
                                param->link,
                                true);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE:
            if (parameterSize == sizeof(param->DoublePixelMode)) {
                isValidSize = true;
                status = EnableDoublePixelMode(handle,
                                               param->DoublePixelMode.link,
                                               param->DoublePixelMode.dataType,
                                               param->DoublePixelMode.embDataType,
                                               param->DoublePixelMode.isSharedPipeline);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI:
            if (parameterSize == sizeof(param->MipiSettings)) {
                isValidSize = true;
                status = ConfigureMIPIOutput(handle,
                                             param->MipiSettings.mipiSpeed,
                                             param->MipiSettings.phyMode);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = SetTxSRCId(handle,
                                    param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = SetDBL(handle,
                                param->link,
                                false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = EnablePacketBasedControlChannel(handle,
                                                         param->link,
                                                         false);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = EnablePeriodicAEQ(handle,
                                           param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = DisableAutoAck(handle,
                                        param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX:
            if (parameterSize == sizeof(param->gpioIndex)) {
                isValidSize = true;
                status = EnableGPIORx(handle,
                                      param->gpioIndex);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_SET_PG:
            if (parameterSize == sizeof(param->SetPGSetting)) {
                isValidSize = true;
                status = ConfigPGSettings(handle,
                                          param->SetPGSetting.width,
                                          param->SetPGSetting.height,
                                          param->SetPGSetting.frameRate,
                                          param->SetPGSetting.linkIndex);
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK:
            if (parameterSize == sizeof(param->linkIndex)) {
                isValidSize = true;
                status = MAX96712DisableLink(handle,
                                             param->linkIndex,
                                             parameterSize,
                                             sizeof(uint8_t));
            }
            break;
        case CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK:
            if (parameterSize == sizeof(param->linkIndex)) {
                isValidSize = true;
                status = MAX96712RestoreLink(handle,
                                             param->linkIndex,
                                             parameterSize,
                                             sizeof(uint8_t));
            }
            break;
        default:
            SIPL_LOG_ERR_STR("MAX96712: Bad parameter: Unrecognized command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX96712: Command failed:", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("MAX96712: Bad parameter: Invalid param size", cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

LinkMAX96712
GetMAX96712Link(
    uint8_t linkNum)
{
    switch (linkNum) {
        case 0u:
            return CDI_MAX96712_LINK_0;
        case 1u:
            return CDI_MAX96712_LINK_1;
        case 2u:
            return CDI_MAX96712_LINK_2;
        case 3u:
            return CDI_MAX96712_LINK_3;
        default:
            return CDI_MAX96712_LINK_NONE;
    }
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "Maxim 96712 Deserializer",
    .regLength = MAX96712_NUM_ADDR_BYTES,
    .dataLength = MAX96712_NUM_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetMAX96712NewDriver(
    void)
{
    return &deviceDriver;
}
