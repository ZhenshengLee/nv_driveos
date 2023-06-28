/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_DS90UB9724_H_
#define _CDI_DS90UB9724_H_

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include <stdbool.h>

#define DS90UB9724_MAX_NUM_LINK                 4u
#define DS90UB9724_MAX_NUM_PHY                  4u
#define DS90UB9724_NUM_VIDEO_PIPELINES          8u
#define DS90UB9724_OSC_MHZ                      25u

#define DS90UB9724_IS_GMSL_LINK_SET(linkVar, linkNum) (((1u << linkNum) & (uint8_t) linkVar) != 0u)
#define DS90UB9724_IS_FPD_LINK_SET(linkVar, linkNum) (((1u << linkNum) & (uint8_t) linkVar) != 0u)
#define DS90UB9724_ADD_LINK(linkVar, linkVal)   (linkVar = (uint8_t)((uint8_t) linkVar | \
                                                                        (uint8_t) linkVal))

typedef enum {
    CDI_CONFIG_DS90UB9724_INVALID = 0u,
    CDI_CONFIG_DS90UB9724_SET_PG_1920x1236,
    CDI_CONFIG_DS90UB9724_TRIGGER_DESKEW,
    CDI_CONFIG_DS90UB9724_SET_PG_3840x1928,
    CDI_CONFIG_DS90UB9724_SET_MIPI,
    CDI_CONFIG_DS90UB9724_DISABLE_BC_AUTOACK,
    CDI_CONFIG_DS90UB9724_NUM,
} ConfigSetsDS90UB9724;

typedef enum {
    CDI_READ_PARAM_CMD_DS90UB9724_INVALID = 0u,
    CDI_READ_PARAM_CMD_DS90UB9724_REV_ID,
    CDI_READ_PARAM_CMD_DS90UB9724_NUM,
} ReadParametersCmdDS90UB9724;

typedef enum {
    CDI_WRITE_PARAM_CMD_DS90UB9724_INVALID = 0u,
    CDI_WRITE_PARAM_CMD_DS90UB9724_SET_I2C_TRANSLATION,
    CDI_WRITE_PARAM_CMD_DS90UB9724_SET_VC_MAP,
    CDI_WRITE_PARAM_CMD_DS90UB9724_PIPELINE_MAPPING,
    CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK3,
    CDI_WRITE_PARAM_CMD_DS90UB9724_SET_LINK_SPEED_FPDLINK4,
    CDI_WRITE_PARAM_CMD_DS90UB9724_CHECK_LOCK,
    CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_FSYNC_GIPO,
    CDI_WRITE_PARAM_CMD_DS90UB9724_ENABLE_AEQ_LMS,
    CDI_WRITE_PARAM_CMD_DS90UB9724_NUM,
} WriteParametersCmdDS90UB9724;

typedef enum {
    CDI_DS90UB9724_I2CPORT_INVALID = 0u,
    CDI_DS90UB9724_I2CPORT_0,
    CDI_DS90UB9724_I2CPORT_1,
} I2CPortDS90UB9724;

typedef enum {
    CDI_DS90UB9724_PHY_MODE_INVALID = 0u,
    CDI_DS90UB9724_PHY_MODE_DPHY,
    CDI_DS90UB9724_PHY_MODE_CPHY,
} PHYModeDS90UB9724;


typedef enum {
    CDI_DS90UB9724_REV_INVALID = 0u,
    CDI_DS90UB9724_REV_1,
} RevisionDS90UB9724;

typedef enum {
    CDI_DS90UB9724_DATA_TYPE_INVALID = 0u,
    CDI_DS90UB9724_DATA_TYPE_RAW12,
} DataTypeDS90UB9724;

typedef enum {
    CDI_DS90UB9724_GPIO_0 = 0u,
    CDI_DS90UB9724_GPIO_1,
    CDI_DS90UB9724_GPIO_2,
    CDI_DS90UB9724_GPIO_3,
    CDI_DS90UB9724_GPIO_4,
    CDI_DS90UB9724_GPIO_5,
    CDI_DS90UB9724_GPIO_6,
    CDI_DS90UB9724_GPIO_7,
} GPIOIndexDS90UB9724;

typedef enum {
    CDI_DS90UB9724_I2C_TRANSLATE_SER = 0u,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE0,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE1,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE2,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE3,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE4,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE5,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE6,
    CDI_DS90UB9724_I2C_TRANSLATE_SLAVE7,
    CDI_DS90UB9724_I2C_TRANSLATE_MAX,
} I2CTranslateDS90UB9724;

typedef union {
    /* Used with CDI_READ_PARAM_CMD_DS90UB9724_REV_ID */
    RevisionDS90UB9724 revision;

    /* Used with CDI_READ_PARAM_CMD_DS90UB9724_ENABLED_LINKS */
    uint8_t link;
} ReadParametersParamDS90UB9724;

typedef union {
    struct {
        uint8_t link;
        uint8_t slaveID;           /* 7 bit I2C address */
        uint8_t slaveAlias;        /* 7 bit I2C address */
        uint8_t lock;              /* Lock the alias configuration to avoid auto update */
        I2CTranslateDS90UB9724 i2cTransID; /* I2C translation set ID */
    } i2cTranslation;

    struct {
        uint8_t link;
        uint8_t inVCID;
        uint8_t outVCID;
    } VCMap;

    uint8_t link;

    struct {
        uint8_t link;
        GPIOIndexDS90UB9724 extFsyncGpio;
        uint8_t bc_gpio;
    } fsyncGpio;
} WriteParametersDS90UB9724;

typedef struct {
    I2CPortDS90UB9724 i2cPort;                          /* I2C port 1 or 2 */
    PHYModeDS90UB9724 phyMode;                          /* CPHY or DPHY */
    bool passiveEnabled;                                /* Doesn't need to control sensor/serializer
                                                       * through aggregator */
    bool tpgEnabled;                                  /* TPG mode */
    /* These will be overwritten during creation */
    RevisionDS90UB9724 revision;                        /* Chip revision information */
    uint16_t manualFSyncFPS;                          /* Used to store manual fsync frequency */
    uint8_t linkMask;                                 /* Indicate what links to be enabled */

    uint32_t mipiSpeed;
} ContextDS90UB9724;

DevBlkCDIDeviceDriver *GetDS90UB9724NewDriver(void);

#if 0
uint8_t GetDS90UB9724Link(uint8_t linkNum);
#endif

NvMediaStatus
DS90UB9724CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS90UB9724SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS90UB9724SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig);

NvMediaStatus
DS90UB9724ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
DS90UB9724WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
DS90UB9724WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
DS90UB9724ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
DS90UB9724DumpRegisters(
    DevBlkCDIDevice *handle);

NvMediaStatus
DS90UB9724GetErrorStatus(
    DevBlkCDIDevice *handle,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
DS90UB9724CheckLink(
    DevBlkCDIDevice *handle,
    uint32_t linkIndex,
    uint32_t linkType,
    bool display);

NvMediaStatus
DS90UB9724OneShotReset(
    DevBlkCDIDevice *handle,
    uint8_t link);

#endif /* _CDI_DS90UB9724_H_ */
