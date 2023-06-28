/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_DS90UB9724_PRIV_H_
#define _CDI_DS90UB9724_PRIV_H_

#define GET_SIZE(x)                                 sizeof(x)
#define GET_BLOCK_LENGTH(x)                         x[0]
#define GET_BLOCK_DATA(x)                           &x[1]
#define SET_NEXT_BLOCK(x)                           (x += (x[0] + 1u))

#define DS90UB9724_IS_MULTIPLE_GMSL_LINK_SET(link)    (((uint8_t)link & ((uint8_t)link - 1u)) != 0u)
#define DS90UB9724_REPLICATION(SRC, DST)              ((DST << 5u) | (SRC << 3u))

/*
 * Utility macro used to call sAccessRegFieldArray() and return if status is not OK.
 * The macro expects the following variables to be available: handle, status
 */
#define ACCESS_REG_FIELD_RET_ERR(mode)                                          \
                                {                                               \
                                    status = sAccessRegFieldQ(handle,           \
                                                              mode);            \
                                    if (status != NVMEDIA_STATUS_OK) {          \
                                        return status;                          \
                                    }                                           \
                                }

/*
 * Utility macro used when add one reg field to queue and return if status is not OK.
 * The macro expects the following variables to be available: handle, status
 */
#define ADD_ONE_REG_FIELD_RET_ERR(name, val)                                    \
                                {                                               \
                                    status = AddToRegFieldQ(handle,             \
                                                            name,               \
                                                            val);               \
                                    if (status != NVMEDIA_STATUS_OK) {          \
                                        return status;                          \
                                    }                                           \
                                }

/*
 * Utility macro used when access to only one reg field is needed.
 * This will clear the RegFieldQ, add name, val to the queue, access the register in specified mode
 * and return if status is not OK.
 * The macro expects the following variables to be available: handle, status
 */
#define ACCESS_ONE_REG_FIELD_RET_ERR(name, val, mode)                           \
                                {                                               \
                                    ClearRegFieldQ(handle);                     \
                                    ADD_ONE_REG_FIELD_RET_ERR(name, val);       \
                                    ACCESS_REG_FIELD_RET_ERR(mode)              \
                                }

#define DS90UB9724_NUM_ADDR_BYTES           (1U)
#define DS90UB9724_NUM_DATA_BYTES           (1U)
#define DS90UB9724_REG_MAX_ADDRESS          (0x1F03U)
#define DS90UB9724_REG_MAX_FIELDS_PER_REG   (8U)
#define REG_WRITE_BUFFER_BYTES              (DS90UB9724_NUM_DATA_BYTES)
#define DS90UB9724_CDI_DEVICE_INDEX         (0U)
#define DS90UB9724_DEV_ID                   (0xA0U)

typedef enum {
    /* Used for array indexes. Must start from 0
     * Do not change the order.
     * New fields must be added to the bottom of the list */
    REG_FIELD_DEV_REV,
    REG_FIELD_DEV_ID,

    REG_FIELD_GPIO_INPUT_CTL_GPIO0,
    REG_FIELD_GPIO_INPUT_CTL_GPIO1,
    REG_FIELD_GPIO_INPUT_CTL_GPIO2,
    REG_FIELD_GPIO_INPUT_CTL_GPIO3,
    REG_FIELD_GPIO_INPUT_CTL_GPIO4,
    REG_FIELD_GPIO_INPUT_CTL_GPIO5,
    REG_FIELD_GPIO_INPUT_CTL_GPIO6,
    REG_FIELD_GPIO_INPUT_CTL_GPIO7,

    REG_FIELD_BC_GPIO_CTL_GPIO0,
    REG_FIELD_BC_GPIO_CTL_GPIO1,
    REG_FIELD_BC_GPIO_CTL_GPIO2,
    REG_FIELD_BC_GPIO_CTL_GPIO3,

    REG_FIELD_CSI_VC_MAP_VC0,
    REG_FIELD_CSI_VC_MAP_VC1,
    REG_FIELD_CSI_VC_MAP_VC2,
    REG_FIELD_CSI_VC_MAP_VC3,
    REG_FIELD_CSI_VC_MAP_VC4,
    REG_FIELD_CSI_VC_MAP_VC5,
    REG_FIELD_CSI_VC_MAP_VC6,
    REG_FIELD_CSI_VC_MAP_VC7,
    REG_FIELD_CSI_VC_MAP_VC8,
    REG_FIELD_CSI_VC_MAP_VC9,
    REG_FIELD_CSI_VC_MAP_VC10,
    REG_FIELD_CSI_VC_MAP_VC11,
    REG_FIELD_CSI_VC_MAP_VC12,
    REG_FIELD_CSI_VC_MAP_VC13,
    REG_FIELD_CSI_VC_MAP_VC14,
    REG_FIELD_CSI_VC_MAP_VC15,

    REG_FIELD_MAX,
} RegBitField;

typedef enum {
    REG_READ_MODE,
    REG_WRITE_MODE,
    REG_READ_MOD_WRITE_MODE,
} RegBitFieldAccessMode;

typedef struct {
    uint8_t regAddr;
    uint8_t msbPos;
    uint8_t lsbPos;
} RegBitFieldProp;

typedef struct {
    RegBitField name[DS90UB9724_REG_MAX_FIELDS_PER_REG];
    uint8_t val[DS90UB9724_REG_MAX_FIELDS_PER_REG];
    uint8_t numRegBitFieldArgs;
} RegBitFieldQ;

typedef struct {
    ContextDS90UB9724 ctx;
    RegBitFieldQ regBitFieldQ;
    DevBlkCDII2CPgmr i2cProgrammer;
} _DriverHandle;

typedef struct {
    RevisionDS90UB9724 revId;
    uint32_t revVal;
} SupportedRevisions;

/* These values must include all of values in the RevisionDS90UB9724 enum */
static const SupportedRevisions supportedRevisions[] = {
    {CDI_DS90UB9724_REV_1, 0x40},
};

static const RegBitFieldProp regBitFieldProps[REG_FIELD_MAX] = {
    [REG_FIELD_DEV_REV]                  = {.regAddr = 0x03, .msbPos = 7, .lsbPos = 0},
    [REG_FIELD_DEV_ID]                   = {.regAddr = 0x03, .msbPos = 7, .lsbPos = 0},

    [REG_FIELD_GPIO_INPUT_CTL_GPIO0]     = {.regAddr = 0x0F, .msbPos = 0, .lsbPos = 0},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO1]     = {.regAddr = 0x0F, .msbPos = 1, .lsbPos = 1},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO2]     = {.regAddr = 0x0F, .msbPos = 2, .lsbPos = 2},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO3]     = {.regAddr = 0x0F, .msbPos = 3, .lsbPos = 3},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO4]     = {.regAddr = 0x0F, .msbPos = 4, .lsbPos = 4},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO5]     = {.regAddr = 0x0F, .msbPos = 5, .lsbPos = 5},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO6]     = {.regAddr = 0x0F, .msbPos = 6, .lsbPos = 6},
    [REG_FIELD_GPIO_INPUT_CTL_GPIO7]     = {.regAddr = 0x0F, .msbPos = 7, .lsbPos = 7},

    [REG_FIELD_BC_GPIO_CTL_GPIO0]        = {.regAddr = 0x6E, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_BC_GPIO_CTL_GPIO1]        = {.regAddr = 0x6E, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_BC_GPIO_CTL_GPIO2]        = {.regAddr = 0x6F, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_BC_GPIO_CTL_GPIO3]        = {.regAddr = 0x6F, .msbPos = 7, .lsbPos = 4},

    [REG_FIELD_CSI_VC_MAP_VC0]           = {.regAddr = 0xA0, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC1]           = {.regAddr = 0xA0, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC2]           = {.regAddr = 0xA1, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC3]           = {.regAddr = 0xA1, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC4]           = {.regAddr = 0xA2, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC5]           = {.regAddr = 0xA2, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC6]           = {.regAddr = 0xA3, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC7]           = {.regAddr = 0xA3, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC8]           = {.regAddr = 0xA4, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC9]           = {.regAddr = 0xA4, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC10]          = {.regAddr = 0xA5, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC11]          = {.regAddr = 0xA5, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC12]          = {.regAddr = 0xA6, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC13]          = {.regAddr = 0xA6, .msbPos = 7, .lsbPos = 4},
    [REG_FIELD_CSI_VC_MAP_VC14]          = {.regAddr = 0xA7, .msbPos = 3, .lsbPos = 0},
    [REG_FIELD_CSI_VC_MAP_VC15]          = {.regAddr = 0xA7, .msbPos = 7, .lsbPos = 4},
};

#endif /* _CDI_DS90UB9724_PRIV_H_ */
