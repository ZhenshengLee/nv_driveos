/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_DS90UB971_SETTING_H_
#define _CDI_DS90UB971_SETTING_H_

#include <stdint.h>

static const DevBlkCDII2CReg ds90ub971_pg_setting_1920x1236_regs[] = {
    {0xB0, 0x02}, /* PATGEN bank + IA_AUTO_INC=1 */
    {0xB1, 0x01}, /* PGEN_CTL */
    {0xB2, 0x01}, /* PGEN_ENABLE=1 */
    {0xB2, 0x23}, /* PGEN_CFG, 4 bars,  */
    {0xB2, 0x2C}, /* PGEN_CSI_DI, 0x24 for RGB888, 0x2C raw12 */
    {0xB2, 0x16}, /* PGEN_LINE_SIZE1 */
    {0xB2, 0x80}, /* PGEN_LINE_SIZE0 --> 5760 bytes */
    {0xB2, 0x05}, /* PGEN_BAR_SIZE1 */
    {0xB2, 0xA0}, /* PGEN_BAR_SIZE0 --> 1440 bytes */
    {0xB2, 0x07}, /* PGEN_ACT_LPF1 */
    {0xB2, 0x88}, /* PGEN_ACT_LPF0 --> 1928 active lines */
    {0xB2, 0x07}, /* PGEN_TOT_LPF1 */
    {0xB2, 0x98}, /* PGEN_TOT_LPF0 --> 1944 total lines */
    {0xB2, 0x0C}, /* PGEN_LINE_PD1 */
    {0xB2, 0xA4}, /* PGEN_LINE_PD0 --> 3236 line period */
    {0xB2, 0x07}, /* PGEN_VBP backporch */
    {0xB2, 0x08}, /* PGEN_VFP frontporch */
};

static const DevBlkCDII2CRegList ds90ub971_pg_setting_1920x1236 = {
    .regs = ds90ub971_pg_setting_1920x1236_regs,
    .numRegs = (uint32_t)I2C_ARRAY_SIZE(ds90ub971_pg_setting_1920x1236_regs),
};

#endif /* _CDI_DS90UB971_SETTING_H_ */
