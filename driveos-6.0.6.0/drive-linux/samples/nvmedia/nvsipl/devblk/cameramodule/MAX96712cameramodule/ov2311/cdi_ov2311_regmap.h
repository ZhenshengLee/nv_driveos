/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * Copyright (c) 2021-2022, OmniVision Technologies.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __CDI_OV2311_REGMAP_H__
#define __CDI_OV2311_REGMAP_H__

#include <stdbool.h>

#define REG_CHIP_ID         0x300A
#define REG_CHIP_REV        0x302A
#define REG_FUSE_ID         0x7000

#define REG_GROUP_HOLD      0x3208

#define REG_VTS             0x380E
#define REG_HTS             0x380C

#define REG_EXPOSURE        0x3501
#define REG_AGAIN           0x3508
#define REG_DGAIN           0x350a

#define REG_WBGAIN          0x5280

#define REG_STROBE_WIDTH    0x3927
#define REG_STROBE_ST       0x3929

/* REGIDX */
#define REGIDX_ENTRY(regname)     REG_## regname ##_OV2311

/* REGMAP */
#define REGMAP_ENTRY(regname, regaddr)          \
    [REG_## regname ##_OV2311] = {       \
        .name = #regname,                       \
        .addr = regaddr,                        \
        .data = 0,                              \
    }

#define SEL_EMB_LENGTH_OV2311 (27)

typedef struct {
    const char *name;
    const uint16_t addr;
    uint8_t data;
} CDIRegSetOV2311;

/*--- enum for indexing regset ---*/
enum reg_index_ov2311 {
    REGIDX_ENTRY(FRAME_CNT_B3) = 0, //0
    REGIDX_ENTRY(FRAME_CNT_B2),
    REGIDX_ENTRY(FRAME_CNT_B1),
    REGIDX_ENTRY(FRAME_CNT_B0),

    REGIDX_ENTRY(X_OUTPUT_SIZE_B1),
    REGIDX_ENTRY(X_OUTPUT_SIZE_B0),
    REGIDX_ENTRY(Y_OUTPUT_SIZE_B1),
    REGIDX_ENTRY(Y_OUTPUT_SIZE_B0),

    REGIDX_ENTRY(VTS_B1), //8
    REGIDX_ENTRY(VTS_B0),

    REGIDX_ENTRY(HTS_B1),
    REGIDX_ENTRY(HTS_B0),

    REGIDX_ENTRY(EXPOSURE_B1), //16
    REGIDX_ENTRY(EXPOSURE_B0),

    REGIDX_ENTRY(AGAIN_B1),
    REGIDX_ENTRY(AGAIN_B0),
    REGIDX_ENTRY(DGAIN_B2), //24
    REGIDX_ENTRY(DGAIN_B1),
    REGIDX_ENTRY(DGAIN_B0),

    REGIDX_ENTRY(AWB_GAIN_0),
    REGIDX_ENTRY(AWB_GAIN_1),
    REGIDX_ENTRY(AWB_GAIN_2),
    REGIDX_ENTRY(AWB_GAIN_3),
    REGIDX_ENTRY(AWB_GAIN_4),
    REGIDX_ENTRY(AWB_GAIN_5),


    REGIDX_ENTRY(TPM_INT_RDOUT),//80
    REGIDX_ENTRY(TPM_DEC_RDOUT),

    OV2311_EMB_DATA_NUM_REGISTERS
};

/*--- register set ---*/
const CDIRegSetOV2311 regsel_ov2311[] = {
    REGMAP_ENTRY(FRAME_CNT_B3,          0x4448),
    REGMAP_ENTRY(FRAME_CNT_B2,          0x4449),
    REGMAP_ENTRY(FRAME_CNT_B1,          0x444a),
    REGMAP_ENTRY(FRAME_CNT_B0,          0x444b),

    REGMAP_ENTRY(X_OUTPUT_SIZE_B1,      0x3808),
    REGMAP_ENTRY(X_OUTPUT_SIZE_B0,      0x3809),
    REGMAP_ENTRY(Y_OUTPUT_SIZE_B1,      0x380a),
    REGMAP_ENTRY(Y_OUTPUT_SIZE_B0,      0x380b),

    REGMAP_ENTRY(VTS_B1,                0x380e),
    REGMAP_ENTRY(VTS_B0,                0x380f),

    REGMAP_ENTRY(HTS_B1,                0x380c),
    REGMAP_ENTRY(HTS_B0,                0x380d),

    REGMAP_ENTRY(EXPOSURE_B1,           0x3501),
    REGMAP_ENTRY(EXPOSURE_B0,           0x3502),

    REGMAP_ENTRY(AGAIN_B1,              0x3508),
    REGMAP_ENTRY(AGAIN_B0,              0x3509),

    REGMAP_ENTRY(DGAIN_B2,              0x350a),
    REGMAP_ENTRY(DGAIN_B1,              0x350b),
    REGMAP_ENTRY(DGAIN_B0,              0x350c),

    REGMAP_ENTRY(AWB_GAIN_0,            0x5d00),
    REGMAP_ENTRY(AWB_GAIN_1,            0x5d01),
    REGMAP_ENTRY(AWB_GAIN_2,            0x5d02),
    REGMAP_ENTRY(AWB_GAIN_3,            0x5d03),
    REGMAP_ENTRY(AWB_GAIN_4,            0x5d04),
    REGMAP_ENTRY(AWB_GAIN_5,            0x5d05),

    REGMAP_ENTRY(TPM_INT_RDOUT,         0x4417),
    REGMAP_ENTRY(TPM_DEC_RDOUT,         0x4418),
};

#endif //__CDI_OV2311_REGMAP_H__