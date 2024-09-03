/*
 * Copyright (c) 2012-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _TESTUTIL_BOARD_H_
#define _TESTUTIL_BOARD_H_

#define MAX_ID_LENGTH 100
#define MAX_MODULE_CONFIG_COUNT 20
#define MAX_WORKAROUND_PER_MODULE 3

typedef enum
{
    BOARD_TYPE_NONE,
    BOARD_TYPE_E1611,
    BOARD_TYPE_E1688,
    BOARD_TYPE_E1853,
    BOARD_TYPE_E1861,
    BOARD_TYPE_PM358,
    BOARD_TYPE_P2360,
    BOARD_TYPE_P1859,
    // Add more here
    BOARD_TYPE_MAX_COUNT
} BoardType;

typedef enum
{
    BOARD_VERSION_NONE,
    BOARD_VERSION_A00,
    BOARD_VERSION_A01,
    BOARD_VERSION_A02,
    BOARD_VERSION_A03,
    BOARD_VERSION_A04,
    BOARD_VERSION_B00,
    // Add more here
    BOARD_VERSION_MAX_COUNT
} BoardVersion;

typedef enum
{
    MODULE_TYPE_NONE,
    MODULE_TYPE_CAPTURE_VIP,
    MODULE_TYPE_CAPTURE_CSI_H2C,
    MODULE_TYPE_CAPTURE_CSI_C2C,
    MODULE_TYPE_CAPTURE_CSI_F2C,
    // Add more here
    MODULE_TYPE_MAX_COUNT
} ModuleType;

typedef struct
{
    unsigned int csi_port:3;    /*  000 - Default,
                                    001 - A,
                                    010 - B,
                                    011 - AB,
                                    100 - CD,
                                    101 - E,
                                    110 - Reserved,
                                    111 - Reserved  */
    unsigned int csi_lanes:2;   /*   00 - Default,
                                     01 - 1,
                                     10 - 2,
                                     11 - 4  */
    unsigned int standard:2;    /*   00 - Default,
                                     01 - NTSC,
                                     10 - PAL
                                     11 - Reserved  */
    unsigned int format:2;      /*   00 - Default,
                                     01 - RGB,
                                     10 - YUV422,
                                     11 - YUV422x2  */
    unsigned int resolution:3;  /*  000 - Default,
                                    001 - 640x480,
                                    010 - 720x480,
                                    011 - 720x576,
                                    100 - 1280x720,
                                    101 - 1920x1080,
                                    110 - Reserved,
                                    111 - Reserved  */
    unsigned int mode:2;        /*   00 - Default,
                                     01 - Live,
                                     10 - Test (Color bars),
                                     11 - Reserved  */
    unsigned int reserved:2;
} CaptureConfig;

typedef int (*Workaround)(void);

typedef struct
{
    ModuleType type;
    char id[MAX_ID_LENGTH];
    int i2c;
    union
    {
        CaptureConfig capture[MAX_MODULE_CONFIG_COUNT];
        // Add more here
    } configs;
    Workaround wars[MAX_WORKAROUND_PER_MODULE];
//    unsigned char wars_run[MAX_WORKAROUND_PER_MODULE];
} ModuleChars;

typedef struct
{
    BoardType type;
    BoardVersion version;
    char id[MAX_ID_LENGTH];
    ModuleChars modules[MODULE_TYPE_MAX_COUNT - 1];
} BoardChars;

// Utility function prototypes
int testutil_board_detect(BoardType board_type, BoardVersion board_version);
int testutil_board_module_query(BoardType board_type, BoardVersion board_version, ModuleType module_type, int display_on);
int testutil_board_module_get_i2c(BoardType board_type, BoardVersion board_version, ModuleType module_type, int *i2c);
int testutil_board_module_workaround(BoardType board_type, BoardVersion board_version, ModuleType module_type);

#endif /* _TESTUTIL_BOARD_H_ */
