/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_MAX96759_H_
#define _CDI_MAX96759_H_

#include "devblk_cdi.h"

#define MAX96759_MAX_REG_ADDRESS        0x2EFF
#define MAX96759_NUM_ADDR_BYTES         2U
#define MAX96759_NUM_DATA_BYTES         1U
#define REG_WRITE_BUFFER_BYTES          MAX96759_NUM_DATA_BYTES
#define MAX96759_MAX_BPP                24

typedef enum {
    CDI_CONFIG_MAX96759_SET_ONESHOT_RESET = 0,
    CDI_CONFIG_MAX96759_SET_RESET,
    CDI_CONFIG_MAX96759_SETUP_DUAL_VIEW,
    CDI_CONFIG_MAX96759_ENABLE_EXT_FRAME_SYNC,
} ConfigSetsMAX96759;

typedef enum {
    CDI_WRITE_PARAM_CMD_MAX96759_SET_DEVICE_ADDRESS = 0,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_A,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_TRANSLATOR_B,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_EDID,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_LINK_MODE,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_BPP,
    CDI_WRITE_PARAM_CMD_MAX96759_SET_TPG,
} WriteParametersCmdMAX96759;

typedef enum {
    CDI_READ_PARAM_CMD_MAX96759_REV_ID = 0,
} ReadParametersCmdMAX96759;

typedef enum {
    LINK_MODE_MAX96759_AUTO = 0,
    LINK_MODE_MAX96759_SPLITTER,
    LINK_MODE_MAX96759_DUAL,
    LINK_MODE_MAX96759_LINK_A,
    LINK_MODE_MAX96759_LINK_B,
} LinkModeMax96759;

typedef enum {
    CDI_MAX96777_REV_3 = 0,
    CDI_MAX96777_REV_4,
    CDI_MAX96759_REV_5,
    CDI_MAX96759_REV_7,
    CDI_MAX96777_MAX96759_INVALID_REV,
} RevisionMax96777Max96759;

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
            uint32_t height;
            uint32_t width;
            uint32_t frameRate;
            const char *identifier;
        } EDID;

        struct {
            uint32_t height;
            uint32_t width;
            uint32_t frameRate;
        } TPG;

        struct {
            LinkModeMax96759 mode;
        } LinkMode;

        uint8_t BitsPerPixel;

        RevisionMax96777Max96759 Revision;
    };
} ReadWriteParams96759;

typedef struct {
    RevisionMax96777Max96759 revision;         /* chip revision information */
} ContextMAX96759;

DevBlkCDIDeviceDriver *GetMAX96759Driver(void);

NvMediaStatus
MAX96759CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96759SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96759SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig);

NvMediaStatus
MAX96759WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96759ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96759DumpRegisters(
    DevBlkCDIDevice *handle);

#endif /* _CDI_MAX96759_H_ */
