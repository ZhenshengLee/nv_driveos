/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <unistd.h>
#if !NV_IS_SAFETY
#include "log_utils.h"
#endif
#include "sipl_error.h"
#include "cdi_ds90ub9724.h"
#include "cdi_ds90ub9724_priv.h"
#include "os_common.h"

/*
 * The following pointers may be used in the functions local to this file but they are checked for
 * NULL in the entry points for CDI functions.
 * DevBlkCDIDevice *handle
 */

static NvMediaStatus
AddToRegFieldQ(
    DevBlkCDIDevice *handle,
    RegBitField name,
    uint8_t val)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    uint8_t index = drvHandle->regBitFieldQ.numRegBitFieldArgs;

    if (index == DS90UB9724_REG_MAX_FIELDS_PER_REG) {
        SIPL_LOG_ERR_STR_UINT("DS90UB9724: RegFieldQ full. Failed to add",
            (uint32_t)name);
        return NVMEDIA_STATUS_ERROR;
    }

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

    drvHandle->regBitFieldQ.numRegBitFieldArgs = 0u;
}

/* Access register fields belonging to a same register.
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
        return status;
    }

    /* Check if all the supplied fields belongs to same register addr.
     * Check if msbPos and lsbPos are valid. */
    for (i = 0u; i < numFields; i++) {
        regAddr = regBitFieldProps[regBit->name[i]].regAddr;
        regBitProp = &regBitFieldProps[regBit->name[i]];
        if ((regAddr != regBitProp->regAddr) ||
            (regBitProp->lsbPos > regBitProp->msbPos)) {
            SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter");
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
                SIPL_LOG_ERR_STR_2INT("DS90UB9724: Register I2C read failed with status",
                    (int32_t)regAddr, (int32_t)status);
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
                SIPL_LOG_ERR_STR_2INT("DS90UB9724: Register I2C write failed with status",
                    (int32_t)regAddr, (int32_t)status);
                return status;
            }
            nvsleep(20);
        }
    }

    return status;
}

static NvMediaStatus
ConfigPGSettings(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg configPGArrCmd_regs[] = {
        {0xB0, 0x02}, /* PATGEN bank + IA_AUTO_INC=1 */
        {0xB1, 0x01}, /* PGEN_CTL */
        {0xB2, 0x01}, /* PGEN_ENABLE=1 */
        {0xB2, 0x33}, /* PGEN_CFG, 4 bars,  */
        {0xB2, 0x2C}, /* PGEN_CSI_DI, 0x24 for RGB888, 0x2C raw12 */
        {0xB2, 0x0F}, /* PGEN_LINE_SIZE1 */
        {0xB2, 0x00}, /* PGEN_LINE_SIZE0 --> 5760 bytes */
        {0xB2, 0x01}, /* PGEN_BAR_SIZE1 */
        {0xB2, 0xE0}, /* PGEN_BAR_SIZE0 --> 1440 bytes */
        {0xB2, 0x02}, /* PGEN_ACT_LPF1 */
        {0xB2, 0xD0}, /* PGEN_ACT_LPF0 --> 1928 active lines */
        {0xB2, 0x04}, /* PGEN_TOT_LPF1 */
        {0xB2, 0x1A}, /* PGEN_TOT_LPF0 --> 1944 total lines */
        {0xB2, 0x0C}, /* PGEN_LINE_PD1 */
        {0xB2, 0x67}, /* PGEN_LINE_PD0 --> 3236 line period */
        {0xB2, 0x21}, /* PGEN_VBP backporch */
        {0xB2, 0x0A}, /* PGEN_VFP frontporch */
    };
    DevBlkCDII2CRegList configPGArrCmd = {
        .regs = configPGArrCmd_regs,
        .numRegs = I2C_ARRAY_SIZE(configPGArrCmd_regs),
    };

    /* Deser patgen only supported in D-PHY mode*/
    if (drvHandle->ctx.phyMode == CDI_DS90UB9724_PHY_MODE_CPHY) {
        SIPL_LOG_ERR_STR("patgen deser only supported in D-PHY mode");
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &configPGArrCmd);

    return status;
}

static NvMediaStatus
SetMIPIOutput(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    DevBlkCDII2CReg configMIPI_DPHY_regs[] = {
        {0x02, 0x1E}, /* D-phy */
        {0x20, 0xF0}, /* FWD_CTL1 disable all */

        {0x1F, 0x10}, /* CSI_PLL_CTL */
        {0xC9, 0x19}, /* CSI_PLL_DIV */
        {0xB0, 0x1C}, /* Select CSI-2 Analog indirect */
        {0xB1, 0x92}, /* CSIPLL_REG_1 */
        {0xB2, 0x80}, /* PLL_OUT_DIV <- 0, PLL_FB_DIV <- 1 */

        {0x32, 0x03},
        {0x33, 0x03}, /* Enable CSI port: CSI_CTL: 4 lane, continuous clock */
        {0x20, 0x00} // port 0 and 1 ma
    };
    DevBlkCDII2CRegList configMIPI_DPHY = {
        .regs = configMIPI_DPHY_regs,
        .numRegs = I2C_ARRAY_SIZE(configMIPI_DPHY_regs),
    };

    DevBlkCDII2CReg configMIPI_CPHY_regs[] = {
        {0x3F, 0x02},  //C-PHY_PIN_MAP Just in case
        {0x02, 0x9E}, /* C-phy */

        {0xB0, 0x1C}, /* Select CSI indirect page */
        {0xB1, 0x02},
        {0xB2, 0x00}, /* remove qudrature phase shift for clock lane */
        {0xB1, 0x12},
        {0xB2, 0x00}, /* remove qudrature phase shift for clock lane */
        {0xB1, 0x22},
        {0xB2, 0x00}, /* remove qudrature phase shift for clock lane */

        {0x32, 0x03}, /* Read CSI port 0, write csi port 0 and 1 */

        {0x1F, 0x10}, /* Set to 2.5Gsps */
        {0xC9, 0x32},

        {0xB0, 0x1C}, /* Select CSI-2 Analog indirect */
        {0xB1, 0x92}, /* CSIPLL_REG_1 */
        {0xB2, 0x40}, /* PLL_OUT_DIV <- 0, PLL_FB_DIV <- 1 */

        {0xB0, 0x00}, /* Write to PATGEN and CSI-2 Trim Register */
        {0xB1, 0x52}, /* Adjust preamble and post settings */
        {0xB2, 0x1A}, /* CSI0 Prebegin = 182UI */
        {0xB1, 0x53},
        {0xB2, 0x7E}, /* CSI0 Post = 210UI */
        {0xB1, 0x78},
        {0xB2, 0x1A}, /* CSI1 Prebegin = 182UI */
        {0xB1, 0x79},
        {0xB2, 0x7E}, /* CSI1 Post = 210UI */

        /* This is  for speed over 1200Msps/lane */
        {0xB0, 0x00}, /* Write to PATGEN and CSI-2 Trim Register */
        {0xB1, 0x44}, /* Override THS_PREP for CSI port 0 */
        {0xB2, 0x82},
        {0xB1, 0x6A}, /* Override THS_PREP for CSI port 1 */
        {0xB2, 0x82},

        {0x20, 0xF0}, /* Disable RX port forwarding */
        {0x3C, 0xAA}, /* Send all RX port to CSI0 and CSI1 */
        {0x20, 0x00}, /* Enable RX port forwarding */
        {0x33, 0x03}, /* Enable CSI port: CSI_CTL: 2 lane, continuous clock */
        {0x01, 0x21},  /* soft reset */
    };
    DevBlkCDII2CRegList configMIPI_CPHY = {
        .regs = configMIPI_CPHY_regs,
        .numRegs = I2C_ARRAY_SIZE(configMIPI_CPHY_regs),
    };

    if (drvHandle->ctx.phyMode == CDI_DS90UB9724_PHY_MODE_DPHY) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &configMIPI_DPHY);
    } else if (drvHandle->ctx.phyMode == CDI_DS90UB9724_PHY_MODE_CPHY) {
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &configMIPI_CPHY);
    } else {
        return NVMEDIA_STATUS_NOT_SUPPORTED;
    }

    // todo: is this still needed?
    nvsleep(100000);

    return status;
}

static NvMediaStatus
TriggerDeskew(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg csi_port_sel = {0x32, 0x03};
    DevBlkCDII2CReg triggerDeskew = {0x34, 0x72};

    if (drvHandle->ctx.i2cPort == CDI_DS90UB9724_I2CPORT_1) {
        csi_port_sel.data |= (1 << 4);
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                       csi_port_sel.address,
                                       csi_port_sel.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                       triggerDeskew.address,
                                       triggerDeskew.data);
    if (status == NVMEDIA_STATUS_OK) {
        nvsleep(10000);
    }

    return status;
}

static NvMediaStatus
selectRxPort(
    DevBlkCDIDevice *handle,
    uint8_t link)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    DevBlkCDII2CReg linkSelect_regs[] = {
        {0x4C, (link << 4) | (1 << link)}, // Set FPDlink4 Sync mode
        {0xB0, (link+1)<<2}, // Select the link
    };
    DevBlkCDII2CRegList linkSelect = {
        .regs = linkSelect_regs,
        .numRegs = I2C_ARRAY_SIZE(linkSelect_regs),
    };

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                        &linkSelect);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return status;
}

static NvMediaStatus
AEQRestartFPD4(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0xB1,
                                        0x25);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0xB2,
                                        0xC1);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0xB2,
                                        0x41);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    nvsleep(20000);

    return status;
}

static NvMediaStatus
ClearAllErrors(
    DevBlkCDIDevice *handle,
    uint8_t link)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    DevBlkCDII2CReg clear_All_Errors_regs[] = {
        { 0x4D, 0x00 },
        { 0x4E, 0x00 },
        { 0x55, 0x00 },
        { 0x56, 0x00 },
        { 0x7A, 0x00 },
    };
    DevBlkCDII2CRegListWritable clear_All_Errors = {
        .regs = clear_All_Errors_regs,
        .numRegs = I2C_ARRAY_SIZE(clear_All_Errors_regs)
    };

    status = selectRxPort(handle, link);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    status = DevBlkCDII2CPgmrReadArray(drvHandle->i2cProgrammer, &clear_All_Errors);

    return status;
}

static NvMediaStatus
DisableAutoAck(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    return DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, 0x58, 0x5E);
}

static NvMediaStatus
SetI2CPort(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    DevBlkCDII2CReg rx_port_ctrl = {0x0C, 0x00}; //use I2C target port 0

    uint8_t i = 0u;
    for (i = 0u; i < DS90UB9724_MAX_NUM_LINK; i++) {
        if ((drvHandle->ctx.i2cPort == CDI_DS90UB9724_I2CPORT_1)) {
            rx_port_ctrl.data |= (1 << (i + 4));
        }
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        rx_port_ctrl.address,
                                        rx_port_ctrl.data);
    if (status == NVMEDIA_STATUS_OK) {
        nvsleep(1000);
    }

    return status;
}

static NvMediaStatus
EnableFsyncGPIO(
    DevBlkCDIDevice *handle,
    uint8_t link,
    GPIOIndexDS90UB9724 extFsyncGpio,
    uint8_t bc_gpio)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DevBlkCDII2CReg rxSelect = {0x4C, 0x0};

    rxSelect.data = (link << 4) | (1 << link);
    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        rxSelect.address,
                                        rxSelect.data);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    nvsleep(10000);

    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_GPIO_INPUT_CTL_GPIO0 + extFsyncGpio,
                                 1u, REG_READ_MOD_WRITE_MODE);

    ACCESS_ONE_REG_FIELD_RET_ERR(REG_FIELD_BC_GPIO_CTL_GPIO0 + bc_gpio,
                                 extFsyncGpio,
                                 REG_READ_MOD_WRITE_MODE);

    return status;
}

NvMediaStatus
DS90UB9724CheckPresence(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    RevisionDS90UB9724 revision = CDI_DS90UB9724_REV_INVALID;
    uint8_t revisionVal = 0u;
    uint32_t numRev = sizeof(supportedRevisions) / sizeof(supportedRevisions[0]);
    uint32_t i = 0u;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724CheckPresence");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DS90UB9724CheckPresence");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Wait for power-up sequence to complete
    nvsleep(10000);

    /* Check revision ID */
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                       0x03,
                                       &revisionVal);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("DS90UB9724: Failed to read device revision\n");
        return status;
    }

    for (i = 0u; i < numRev; i++) {
        if (revisionVal == supportedRevisions[i].revVal) {
            revision = supportedRevisions[i].revId;
            LOG_MSG("DS90UB9724: Revision %x detected\n", revision);
            drvHandle->ctx.revision = revision;
            return NVMEDIA_STATUS_OK;
        }
    }

    SIPL_LOG_ERR_STR_INT("DS90UB9724: Unsupported DS90UB9724 revision detected",
        (int32_t)revisionVal);

    return NVMEDIA_STATUS_NOT_SUPPORTED;
}

static NvMediaStatus
DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    _DriverHandle *driverHandle = NULL;
    ContextDS90UB9724 *ctx = (ContextDS90UB9724 *) clientContext;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DriverCreate");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (clientContext == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Client context passed to DriverCreate");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (ctx->i2cPort != CDI_DS90UB9724_I2CPORT_0 &&
        ctx->i2cPort != CDI_DS90UB9724_I2CPORT_1) {
        SIPL_LOG_ERR_STR("DS90UB9724: Invalid I2C port");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    driverHandle = calloc(1, sizeof(_DriverHandle));
    if (driverHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Memory allocation for context failed");
        return NVMEDIA_STATUS_OUT_OF_MEMORY;
    }

    memcpy(&driverHandle->ctx, ctx, sizeof(ContextDS90UB9724));
    driverHandle->ctx.revision = CDI_DS90UB9724_REV_INVALID;
    driverHandle->ctx.manualFSyncFPS = 0u;
    handle->deviceDriverHandle = (void *)driverHandle;

    // Create the I2C programmer for register read/write
    driverHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                         DS90UB9724_NUM_ADDR_BYTES,
                                                         DS90UB9724_NUM_DATA_BYTES);
    if (driverHandle->i2cProgrammer == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Failed to initialize the I2C programmer");
        free(driverHandle);
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

static NvMediaStatus
DriverDestroy(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // Destroy the I2C programmer
    DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

    free(drvHandle);
    drvHandle = NULL;

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
DS90UB9724DumpRegisters(
    DevBlkCDIDevice *handle)
{
    _DriverHandle *drvHandle = NULL;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DriverDestroy");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;

    typedef struct indirectReg {
        char*a;
        int regStart;
        int regEnd;
        int IA_SEL;
    } indirectReg;

    indirectReg indReg[10] = {
        { "Pattern Generator Registers", 0x01, 0x8B, 0x0 },
        { "FPD RX Port 0 Registers", 0x00, 0xF0, 0x01 },
        { "FPD RX Port 1 Registers", 0x00, 0xF0, 0x02 },
        { "FPD RX Port 2 Registers", 0x00, 0xF0, 0x03 },
        { "FPD RX Port 3 Registers", 0x00, 0xF0, 0x04 },
        { "PLL Control Registers", 0x80, 0xC6, 0x05 },
        { "CSI-2 Analog Registers", 0x01, 0x92, 0x07 },
        { "Read of Configuration Data (loaded from eFuse ROM)", 0x01, 0x8B, 0x07 },
        { "Read of DIE ID (loaded from eFuse ROM)", 0x00, 0x15, 0x09 },
        { "SAR ADC Registers", 0x04, 0xE7, 0x0A },
    };

    for (int i = 0; i < 10; i++) {
        // Select the indirect page (Read+Autoincrement)
        status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                            0xB0,
                                            (0x03 | (indReg[i].IA_SEL<<2)));
    }
    return status;
}

NvMediaStatus
DS90UB9724ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = NULL;
    bool isValidSize = false;
    ReadParametersParamDS90UB9724 *param = (ReadParametersParamDS90UB9724 *) parameter;
    static const char *cmdString[] = {
        [CDI_READ_PARAM_CMD_DS90UB9724_REV_ID] =
            "CDI_READ_PARAM_CMD_DS90UB9724_REV_ID",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724ReadParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DS90UB9724ReadParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null parameter passed to DS90UB9724ReadParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_READ_PARAM_CMD_DS90UB9724_INVALID) ||
        (parameterType >= CDI_READ_PARAM_CMD_DS90UB9724_NUM)) {
        SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch (parameterType) {
        case CDI_READ_PARAM_CMD_DS90UB9724_REV_ID:
            if (parameterSize == sizeof(param->revision)) {
                isValidSize = true;
                param->revision = drvHandle->ctx.revision;
                status = NVMEDIA_STATUS_OK;
            }
            break;
        default:
            SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Unrecognized command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("DS90UB9724: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("DS90UB9724: Bad parameter: Invalid param size",
            cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

NvMediaStatus
DS90UB9724SetDefaults(
    DevBlkCDIDevice *handle)
{
    /* Should reset everything to nominal */
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    _DriverHandle *drvHandle = NULL;

    DevBlkCDII2CReg configRxPortArrCmd_regs[] = {
        {0x00, 0x02}
    };
    DevBlkCDII2CRegList configRxPortArrCmd = {
        .regs = configRxPortArrCmd_regs,
        .numRegs = I2C_ARRAY_SIZE(configRxPortArrCmd_regs),
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DS90UB9724SetDefaults");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &configRxPortArrCmd);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return SetI2CPort(handle);
}

NvMediaStatus
DS90UB9724SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    static const char *cmdString[] = {
        [CDI_CONFIG_DS90UB9724_SET_PG_3840x1928] =
            "CDI_CONFIG_DS90UB9724_SET_PG_3840x1928",
        [CDI_CONFIG_DS90UB9724_SET_MIPI] =
            "CDI_CONFIG_DS90UB9724_SET_MIPI",
        [CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW] =
            "CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW",
        [CDI_CONFIG_DS90UB9724_DISABLE_BC_AUTOACK] =
            "CDI_CONFIG_DS90UB9724_DISABLE_BC_AUTOACK",
        [CDI_CONFIG_DS90UB9724_AEQ_RESTART_FPD4] =
            "CDI_CONFIG_DS90UB9724_AEQ_RESTART_FPD4",
    };

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724SetDeviceConfig");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DS90UB9724SetDeviceConfig");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((enumeratedDeviceConfig == CDI_CONFIG_DS90UB9724_INVALID) ||
        (enumeratedDeviceConfig >= CDI_CONFIG_DS90UB9724_NUM)) {
        SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch (enumeratedDeviceConfig) {
        case CDI_CONFIG_DS90UB9724_SET_PG_3840x1928:
            status = ConfigPGSettings(handle);
            break;
        case CDI_CONFIG_DS90UB9724_SET_MIPI:
            status = SetMIPIOutput(handle);
            break;
        case CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW:
            status = TriggerDeskew(handle);
            break;
        case CDI_CONFIG_DS90UB9724_DISABLE_BC_AUTOACK:
            status = DisableAutoAck(handle);
            break;
        case CDI_CONFIG_DS90UB9724_AEQ_RESTART_FPD4:
            status = AEQRestartFPD4(handle);
            break;
        default:
            SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Unrecognized command");
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("DS90UB9724: Command failed", cmdString[enumeratedDeviceConfig]);
    }

    return status;
}

static NvMediaStatus
EnableI2CPass(
    DevBlkCDIDevice *handle,
    uint8_t link)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;


    uint8_t reg_58;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                                     0x58U,
                                                     &reg_58);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("DS90UB9724: Failed to read 0x58 register", link, status);
        return status;
    }
    reg_58 |= 0x40; // Enable I2C Passthrough

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0x58U,
                                        reg_58);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    return status;
}

NvMediaStatus
DS90UB9724CheckLinkStatus(
    DevBlkCDIDevice *handle,
    uint8_t link)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    uint8_t reg_4D;
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                                     0x4DU,
                                                     &reg_4D);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("DS90UB9724: Failed to read 0x58 register", link, status);
        return status;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                                     0x4DU,
                                                     &reg_4D);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("DS90UB9724: Failed to read 0x58 register", link, status);
        return status;
    }

    if (reg_4D == ((link << 6) + 3U)) {
        LOG_MSG("link %d is good\n", link);
    } else {
        LOG_MSG("link %d is bad\n", link);
    }

    return status;
}

static NvMediaStatus
SetBCCConfig(
    DevBlkCDIDevice *handle,
    bool I2CPassThrough,
    bool autoAckAll)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t val;

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0x58U, &val);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to read BCC_CONFIG register", status);
        return status;
    }

    if (I2CPassThrough) {
        val |= (1U << 6U);
    } else {
        val &= ~(1U << 6U);
    }

    if (autoAckAll) {
        val |= (1U << 5U);
    } else {
        val &= ~(1U << 5U);
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0x58U,
                                        val);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to write BCC_CONFIG register", status);
        return status;
    }

    return status;
}

static NvMediaStatus
SetI2CTranslation(
    DevBlkCDIDevice *handle,
    uint8_t link,
    uint8_t slaveID,
    uint8_t slaveAlias,
    uint8_t lock,
    I2CTranslateDS90UB9724 i2cTransID)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (i2cTransID == CDI_DS90UB9724_I2C_TRANSLATE_SER) {
        DevBlkCDII2CReg serI2C_regs[] = {
            {0x5B, 0x00},
            {0x5C, 0x00},
        };
        DevBlkCDII2CRegList serI2C = {
            .regs = serI2C_regs,
            .numRegs = I2C_ARRAY_SIZE(serI2C_regs),
        };

        serI2C_regs[0].data = (slaveID << 1) | (lock ? 1 : 0); // lock the ser address?
        serI2C_regs[1].data = (slaveAlias << 1) | (lock ? 1 : 0); // Enable auto ack for now?

        status = selectRxPort(handle, link);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }

        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                    &serI2C);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    } else if (i2cTransID < CDI_DS90UB9724_I2C_TRANSLATE_MAX) {
        DevBlkCDII2CReg slaveI2C_regs[] = {
            {0x4C, (link << 4) | (1 << link)},
            {0x5D, 0x00},
            {0x65, 0x00},
        };
        DevBlkCDII2CRegList slaveI2C = {
            .regs = slaveI2C_regs,
            .numRegs = I2C_ARRAY_SIZE(slaveI2C_regs),
        };

        slaveI2C_regs[1].data = slaveID << 1;
        slaveI2C_regs[2].data = slaveAlias << 1;
        slaveI2C_regs[1].address = 0x5D + (i2cTransID - CDI_DS90UB9724_I2C_TRANSLATE_SLAVE0);
        slaveI2C_regs[2].address = 0x65 + (i2cTransID - CDI_DS90UB9724_I2C_TRANSLATE_SLAVE0);
        status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                    &slaveI2C);
        if (status != NVMEDIA_STATUS_OK) {
            return status;
        }
    } else {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

static NvMediaStatus
setVCMap(
    DevBlkCDIDevice *handle,
    uint8_t link,
    uint8_t inVCID,
    uint8_t outVCID)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if ((inVCID > 15) || (outVCID > 15)) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                        0x4C,
                                        (link << 4) | (1 << link));
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    ClearRegFieldQ(handle);
    ADD_ONE_REG_FIELD_RET_ERR(REG_FIELD_CSI_VC_MAP_VC0 + inVCID,
                                      outVCID);
    ACCESS_REG_FIELD_RET_ERR(REG_READ_MOD_WRITE_MODE);

    return status;
}

static NvMediaStatus
waitLinkLock(
    DevBlkCDIDevice *handle,
    const uint8_t link,
    const uint32_t max_iters)
{
    uint16_t rx_port = 0U;
    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to en_AEQ_LMS");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to en_AEQ_LMS");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    for (uint32_t i = 0U; i < max_iters; i++) {
        uint8_t val;
        NvMediaStatus status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0x4DU, &val);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR_2INT("DS90UB9724: Failed to poll link lock status", link, status);
            return status;
        }

        rx_port = ((val >> 6U) & 0x3U);
        if (link != rx_port) {
            SIPL_LOG_ERR_STR_2INT("DS90UB9724: Link is not selected at the device", link, rx_port);
            return NVMEDIA_STATUS_INVALID_STATE;
        } else if ((val & 0x1U) == 0x1U) {
            return NVMEDIA_STATUS_OK;
        } else {
            nvsleep(10000);
        }
    }

    SIPL_LOG_ERR_STR_2INT("DS90UB9724: Link was not locked", link, rx_port);

    return NVMEDIA_STATUS_TIMED_OUT;
}

static NvMediaStatus
en_AEQ_LMS (
       DevBlkCDIDevice *handle,
       uint8_t link)
{
    uint8_t read_aeq_init;
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to en_AEQ_LMS");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to en_AEQ_LMS");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    // first_time_power_up only
    status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer, 0xB1, 0x2C);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to write to the deserializer device", status);
        return status;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0xB2, &read_aeq_init);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to write to the deserializer device", status);
        return status;
    }

    DevBlkCDII2CReg AEQ_LMS_regs[] = {
        //First run
        {0xB1, 0x27}, // read/write indirect registers
        {0xB2, read_aeq_init},  // read/write indirect registers
        {0xB1, 0x28},  // ADDED THIS HERE
        {0xB2, read_aeq_init + 1U},  // ADDED THIS HERE
        {0xB1, 0x2B}, // ADDED THIS HERE
        {0xB2, 0x00}, // set EQ offset to 0 # ADDED THIS HERE
        // Normal
        {0xB1, 0x9E}, // read/write indirect registers
        {0xB2, 0x00}, // enable sumbuf tap2
        {0xB1, 0x2E}, // read/write indirect registers
        {0xB2, 0x40}, // enable VGA sweep/adapt
        {0xB1, 0xF0}, // read/write indirect registers
        {0xB2, 0x00}, // disable over-write of VGA sweep/adapt
        {0xB1, 0x71}, // read/write indirect registers
        {0xB2, 0x00}, // disable over-write of VGA sweep/adapt
        {0xB1, 0x21}, // read/write indirect registers
        {0xB2, 0xFF}, // Increase parity error threshold

        {0x01, 0x21, 20*1000}, // Soft reset, wait 20ms only for FPDLink4, 100ms delay required for FPDLink3

        // en_DFE_LMS
        {0xB1, 0x90},
        {0xB2, 0x40},
    };
    DevBlkCDII2CRegList AEQ_LMSArrCmd = {
        .regs = AEQ_LMS_regs,
        .numRegs = I2C_ARRAY_SIZE(AEQ_LMS_regs),
    };

    status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer, &AEQ_LMSArrCmd);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to write to the deserializer device", status);
        return status;
    }

    status = waitLinkLock(handle, link, 50U);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_2INT("DS90UB9724: Link failed to come up", link, status);
    }

    return status;
}

static NvMediaStatus
SetLinkSpeedFPDLink(
    DevBlkCDIDevice *handle,
    uint8_t link,
    FPDLinkModeDS90UB9724 linkMode)
{
    _DriverHandle *drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t enabledLinks = 0U;
    uint8_t reg_0x58 = 0U;

    /* Check the currently enabled links. */
    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0x0C, &enabledLinks);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer, 0x58, &reg_0x58);
    if (status != NVMEDIA_STATUS_OK) {
        return status;
    }

    // disable I2C pass-through & disable AUTO_ACK_ALL
    reg_0x58 &= ~(0x3U << 5U);

    DevBlkCDII2CReg const linkSpeedFPDLink4_regs[] = {
        {0xE4, 0x00}, // Set FPDlink4 Sync mode

        // Configure backchannel
        {0x58, reg_0x58 | 0x06}, // Backchannel: forward decoded aliases, 50mbps (953 backward comp)
        {0xB1, 0x04},
        {0xB2, 0x00}, // set FPD PBC drv into FPD IV mode
        {0xB1, 0x1B},
        {0xB2, 0x00}, //set FPD PBC drv into FPD IV mode

        {0xB1, 0x21},
        {0xB2, 0x2F}, // set 960 AEQ timer to 400us/step
        {0xB1, 0x25},
        {0xB2, 0xC1}, // set 960 AEQ in reset mode
        {0x3C, 0x0F}, // disable lock lost feature

        {0x0C, enabledLinks | (1U << link)}, // Enable RX port based on link
        {0xB2, 0x41, 10000}, // unreset 960 AEQ + 10ms delay

        {0x3C, 0x1F}, // enable lock lost feature
    };
    DevBlkCDII2CRegList linkSpeedFPDLink4 = {
        .regs = linkSpeedFPDLink4_regs,
        .numRegs = I2C_ARRAY_SIZE(linkSpeedFPDLink4_regs),
    };

    DevBlkCDII2CReg const linkSpeedFPDLink3_regs[] = {
        {0xE4, 0x02}, // Set FPDlink3 Sync mode

        // Set Backchannel
        {0x58, 0x7E}, // BC_FREQ_SELECT=(PLL_FREQ/3200) Mbps
        {0xB1, 0xA8},
        {0xB2, 0x80}, // set aeq_lock_mode = 1
        // bc_drv_config

        {0xB1, 0x04},
        {0xB2, 0x40}, //  remove HiZ of NMOS drv of spare driver

        {0xB1, 0x1B},
        {0xB2, 0x08}, // remove HiZ of PMOS drv of spare driver & disable one 1 CMR ladder

        {0xB1, 0x0D},
        {0xB2, 0x7F,100000}, // enable the FPD3 spare driver, wait 100ms

        {0x0C, enabledLinks | (1U << link)}, // Enable RX port based on link
    };
    DevBlkCDII2CRegList linkSpeedFPDLink3 = {
        .regs = linkSpeedFPDLink3_regs,
        .numRegs = I2C_ARRAY_SIZE(linkSpeedFPDLink3_regs),
    };

    // Select Rx port
    status = selectRxPort(handle, link);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("DS90UB9724: Failed to select Rx Port", status);
        return status;
    }

    switch (linkMode) {
        case CDI_DS90UB9724_LINK_MODE_FPD4SYNC:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &linkSpeedFPDLink4);
            break;
        case CDI_DS90UB9724_LINK_MODE_FPD3SYNC:
            status = DevBlkCDII2CPgmrWriteArray(drvHandle->i2cProgrammer,
                                                &linkSpeedFPDLink3);
            break;
        default:
            SIPL_LOG_ERR_STR("DS90UB9724: Invalid link mode");
            break;
    }

    return status;
}

NvMediaStatus
DS90UB9724WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    _DriverHandle *drvHandle = NULL;
    WriteParametersDS90UB9724 *param = (WriteParametersDS90UB9724 *) parameter;
    bool isValidSize = false;
    static const char *cmdString[] = {
        [CDI_WRITE_PARAM_CMD_DS90UB9724_SET_BCC_CONFIG] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_SET_BCC_CONFIG",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_PIPELINE_MAPPING] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_PIPELINE_MAPPING",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_SELECT_RX_PORT] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_SELECT_RX_PORT",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_FSYNC_GIPO] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_FSYNC_GIPO",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_I2C_PASSTHROUGH] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_I2C_PASSTHROUGH",
        [CDI_WRITE_PARAM_CMD_DS90UB9724_CLEAR_ALL_ERRORS] =
            "CDI_WRITE_PARAM_CMD_DS90UB9724_CLEAR_ALL_ERRORS"
    };


    if (handle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    drvHandle = (_DriverHandle *)handle->deviceDriverHandle;
    if (drvHandle == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null driver handle passed to DS90UB9724WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if (parameter == NULL) {
        SIPL_LOG_ERR_STR("DS90UB9724: Null handle passed to DS90UB9724WriteParameters");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    if ((parameterType == CDI_WRITE_PARAM_CMD_DS90UB9724_INVALID) ||
        (parameterType >= CDI_WRITE_PARAM_CMD_DS90UB9724_NUM)) {
        SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Invalid command");
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    switch (parameterType) {
        case CDI_WRITE_PARAM_CMD_DS90UB9724_SET_BCC_CONFIG:
            if (parameterSize == sizeof(param->BCCCfg)) {
                isValidSize = true;
                status = SetBCCConfig(handle,
                                      param->BCCCfg.I2CPassThrough,
                                      param->BCCCfg.autoAckAll);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION:
            if (parameterSize == sizeof(param->i2cTranslation)) {
                isValidSize = true;
                status = SetI2CTranslation(handle,
                                           param->i2cTranslation.link,
                                           param->i2cTranslation.slaveID,
                                           param->i2cTranslation.slaveAlias,
                                           param->i2cTranslation.lock,
                                           param->i2cTranslation.i2cTransID);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_SELECT_RX_PORT:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = selectRxPort(handle,
                                        param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_I2C_PASSTHROUGH:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = EnableI2CPass(handle,
                                       param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_SET_VC_MAP:
            if (parameterSize == sizeof(param->VCMap)) {
                isValidSize = true;
                status = setVCMap(handle,
                                  param->VCMap.link,
                                  param->VCMap.inVCID,
                                  param->VCMap.outVCID);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = en_AEQ_LMS(handle, param->link);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK:
            if (parameterSize == sizeof(param->linkMode)) {
                isValidSize = true;
                status = SetLinkSpeedFPDLink(handle,
                                             param->linkMode.link,
                                             param->linkMode.mode);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_FSYNC_GIPO:
            if (parameterSize == sizeof(param->fsyncGpio)) {
                isValidSize = true;
                status = EnableFsyncGPIO(handle,
                                         param->fsyncGpio.link,
                                         param->fsyncGpio.extFsyncGpio,
                                         param->fsyncGpio.bc_gpio);
            }
            break;
        case CDI_WRITE_PARAM_CMD_DS90UB9724_CLEAR_ALL_ERRORS:
            if (parameterSize == sizeof(param->link)) {
                isValidSize = true;
                status = ClearAllErrors(handle,
                                       param->link);
            }
            break;
        default:
            SIPL_LOG_ERR_STR("DS90UB9724: Bad parameter: Unrecognized command");
            isValidSize = true;
            status = NVMEDIA_STATUS_BAD_PARAMETER;
            break;
    }

    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("DS90UB9724: Command failed", cmdString[parameterType]);
    }

    if (!isValidSize) {
        SIPL_LOG_ERR_2STR("DS90UB9724: Bad parameter: Invalid param size", cmdString[parameterType]);
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "TI DS90UB9724 Deserializer",
    .regLength = DS90UB9724_NUM_ADDR_BYTES,
    .dataLength = DS90UB9724_NUM_DATA_BYTES,
    .DriverCreate = DriverCreate,
    .DriverDestroy = DriverDestroy,
};

DevBlkCDIDeviceDriver *
GetDS90UB9724NewDriver(void)
{
    return &deviceDriver;
}
