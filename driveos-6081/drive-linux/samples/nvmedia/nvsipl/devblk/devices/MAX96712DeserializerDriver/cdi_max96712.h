/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX96712_H
#define CDI_MAX96712_H

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include <stdbool.h>

#define MAX96712_MAX_NUM_LINK                 4U
#define MAX96712_MAX_NUM_PHY                  4U
#define MAX96712_MAX_NUM_PG                   2U
#define MAX96712_NUM_VIDEO_PIPELINES          8U
#define MAX96712_OSC_MHZ                      25U

#define MAX96712_MAX_GLOBAL_ERROR_NUM         20U
#define MAX96712_MAX_PIPELINE_ERROR_NUM       16U
#define MAX96712_MAX_LINK_BASED_ERROR_NUM     16U

// Currently driver implements about 20+16*8+16*4 = 212 error status
#define MAX96712_MAX_ERROR_STATUS_COUNT       255U

#define MAX96712_IS_GMSL_LINK_SET(linkVar, linkNum) (((1U << (linkNum)) & (uint8_t) (linkVar)) != 0U)
#define MAX96712_ADD_LINK(linkVar, linkVal)   ((linkVar) = (LinkMAX96712)((uint8_t) (linkVar) | \
                                                                        (uint8_t) (linkVal)))

typedef enum {
    CDI_CONFIG_MAX96712_INVALID,
    CDI_CONFIG_MAX96712_ENABLE_PG,
    CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE,
    CDI_CONFIG_MAX96712_ENABLE_CSI_OUT,
    CDI_CONFIG_MAX96712_DISABLE_CSI_OUT,
    CDI_CONFIG_MAX96712_TRIGGER_DESKEW,
    CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK,
    CDI_CONFIG_MAX96712_ENABLE_REPLICATION,
    CDI_CONFIG_MAX96712_DISABLE_REPLICATION,
    CDI_CONFIG_MAX96712_ENABLE_ERRB,
    CDI_CONFIG_MAX96712_DISABLE_ERRB,
    CDI_CONFIG_MAX96712_NUM,
} ConfigSetsMAX96712;

typedef enum {
    CDI_READ_PARAM_CMD_MAX96712_INVALID,
    CDI_READ_PARAM_CMD_MAX96712_REV_ID,
    CDI_READ_PARAM_CMD_MAX96712_ERRB,
    CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR,
    CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,
    CDI_READ_PARAM_CMD_MAX96712_GET_PWR_METHOD,
    CDI_READ_PARAM_CMD_MAX96712_NUM,
} ReadParametersCmdMAX96712;

typedef enum {
    CDI_WRITE_PARAM_CMD_MAX96712_INVALID,

    /* GMSL1 related APIs */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS,
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL,
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL,
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK,
    CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK,
    CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,
    CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG,
    CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID,
    CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ,
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK,
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_PG,
    CDI_WRITE_PARAM_CMD_MAX96712_NUM,
} WriteParametersCmdMAX96712;

/** enumerations of pipeline copy configuration */
typedef enum {
     CDI_MAX96712_CFG_PIPE_COPY_NONE,       /*!< No pipeline copy support */
     CDI_MAX96712_CFG_PIPE_COPY_MODE_1,     /*!< Pipeline copy support mode 1. Phy mode 2x4. Link 0/1/2/3 to Phy B */
     CDI_MAX96712_CFG_PIPE_COPY_MODE_2,     /*!< Pipeline copy support mode 2. Phy mode 4+2x2. Link 0/1 to Phy E and Link 2/3 to Phy F */
     CDI_MAX96712_CFG_PIPE_COPY_INVALID,    /*!< Invalid pipeline copy configuration */
} CfgPipelineCopyMAX96712;

typedef enum {
     CDI_MAX96712_GMSL_MODE_INVALID,
     CDI_MAX96712_GMSL_MODE_UNUSED,
     CDI_MAX96712_GMSL1_MODE,
     CDI_MAX96712_GMSL2_MODE_6GBPS,
     CDI_MAX96712_GMSL2_MODE_3GBPS,
} GMSLModeMAX96712;

typedef enum {
    CDI_MAX96712_I2CPORT_INVALID,
    CDI_MAX96712_I2CPORT_0,
    CDI_MAX96712_I2CPORT_1,
    CDI_MAX96712_I2CPORT_2,
} I2CPortMAX96712;

typedef enum {
    CDI_MAX96712_MIPI_OUT_INVALID,
    CDI_MAX96712_MIPI_OUT_4x2,
    CDI_MAX96712_MIPI_OUT_2x4,
    CDI_MAX96712_MIPI_OUT_4a_2x2,
} MipiOutModeMAX96712;

typedef enum {
    CDI_MAX96712_PHY_MODE_INVALID,
    CDI_MAX96712_PHY_MODE_DPHY,
    CDI_MAX96712_PHY_MODE_CPHY,
} PHYModeMAX96712;

typedef enum {
    CDI_MAX96712_TXPORT_PHY_C,
    CDI_MAX96712_TXPORT_PHY_D,
    CDI_MAX96712_TXPORT_PHY_E,
    CDI_MAX96712_TXPORT_PHY_F,
} TxPortMAX96712;

typedef enum {
    CDI_MAX96712_REV_INVALID,
    CDI_MAX96712_REV_1,
    CDI_MAX96712_REV_2,
    CDI_MAX96712_REV_3,
    CDI_MAX96712_REV_4,
    CDI_MAX96712_REV_5,
} RevisionMAX96712;

typedef enum {
    CDI_MAX96712_LINK_NONE = 0U,
    CDI_MAX96712_LINK_0 = (1U << 0U),
    CDI_MAX96712_LINK_1 = (1U << 1U),
    CDI_MAX96712_LINK_2 = (1U << 2U),
    CDI_MAX96712_LINK_3 = (1U << 3U),
    CDI_MAX96712_LINK_ALL = 0xFU,
} LinkMAX96712;

typedef enum {
    CDI_MAX96712_FSYNC_INVALID,
    CDI_MAX96712_FSYNC_MANUAL,
    CDI_MAX96712_FSYNC_AUTO,
    CDI_MAX96712_FSYNC_OSC_MANUAL,
    CDI_MAX96712_FSYNC_EXTERNAL,
} FSyncModeMAX96712;

typedef enum {
    CDI_MAX96712_DATA_TYPE_INVALID, /*!<  */
    CDI_MAX96712_DATA_TYPE_RAW10,   /*!< RAW10 */
    CDI_MAX96712_DATA_TYPE_RAW12,   /*!< RAW12 */
    CDI_MAX96712_DATA_TYPE_RAW16,   /*!< RAW16 */
    CDI_MAX96712_DATA_TYPE_RGB,     /*!< RGB */
    CDI_MAX96712_DATA_TYPE_YUV_8,   /*!< YUV8 */
    CDI_MAX96712_DATA_TYPE_YUV_10,  /*!< YUV10 */
} DataTypeMAX96712;

/* Used as param for CheckLink() */
typedef enum {
    CDI_MAX96712_LINK_LOCK_INVALID,
    CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG,
    CDI_MAX96712_LINK_LOCK_GMSL2,
    CDI_MAX96712_LINK_LOCK_VIDEO,
} LinkLockTypeMAX96712;

typedef enum {
    CDI_MAX96712_PIPELINE_ERROR_INVALID,

    CDI_MAX96712_PIPELINE_LMO_OVERFLOW_ERR,
    CDI_MAX96712_PIPELINE_CMD_OVERFLOW_ERR,

    CDI_MAX96712_PIPELINE_PGEN_VID_UNLOCK_ERR,

    CDI_MAX96712_PIPELINE_MEM_ERR,

    CDI_MAX96712_PIPELINE_VID_SEQ_ERR,

    CDI_MAX96712_MAX_PIPELINE_FAILURE_TYPES
} PipelineFailureTypeMAX96712;

typedef enum {
    CDI_MAX96712_GLOBAL_ERROR_INVALID,

    /* GMSL non-link based global errors (cnt:19) */
    CDI_MAX96712_GLOBAL_UNLOCK_ERR,
    CDI_MAX96712_GLOBAL_ERR,
    CDI_MAX96712_GLOBAL_CMU_UNLOCK_ERR,

    CDI_MAX96712_GLOBAL_WM,
    CDI_MAX96712_GLOBAL_WM2,
    CDI_MAX96712_GLOBAL_LINE_FAULT,
    CDI_MAX96712_GLOBAL_MEM_STORE_CRC,

    CDI_MAX96712_GLOBAL_FRAME_SYNC,
    CDI_MAX96712_GLOBAL_REMOTE_SIDE,
    CDI_MAX96712_GLOBAL_VID_PRBS,
    CDI_MAX96712_GLOBAL_VID_LINE_CRC,

    CDI_MAX96712_GLOBAL_MEM_ECC1,
    CDI_MAX96712_GLOBAL_MEM_ECC2,

    CDI_MAX96712_GLOBAL_FSYNC_SYNC_LOSS,
    CDI_MAX96712_GLOBAL_FSYNC_STATUS,

    CDI_MAX96712_GLOBAL_CMP_VTERM_STATUS,
    CDI_MAX96712_GLOBAL_VDD_OV_FLAG,

    CDI_MAX96712_GLOBAL_VDDBAD_STATUS,
    CDI_MAX96712_GLOBAL_CMP_STATUS,

    CDI_MAX96712_GLOBAL_VDDSW_UV,
    CDI_MAX96712_GLOBAL_VDDIO_UV,
    CDI_MAX96712_GLOBAL_VDD18_UV,

    CDI_MAX96712_MAX_GLOBAL_FAILURE_TYPES
} GlobalFailureTypeMAX96712;

typedef enum {
    CDI_MAX96712_GMSL_LINK_ERROR_INVALID,

    /* GMSL2 link based errors (cnt:7) */
    CDI_MAX96712_GMSL2_LINK_UNLOCK_ERR,
    CDI_MAX96712_GMSL2_LINK_DEC_ERR,
    CDI_MAX96712_GMSL2_LINK_IDLE_ERR,
    CDI_MAX96712_GMSL2_LINK_EOM_ERR,
    CDI_MAX96712_GMSL2_LINK_ARQ_RETRANS_ERR,
    CDI_MAX96712_GMSL2_LINK_MAX_RETRANS_ERR,
    CDI_MAX96712_GMSL2_LINK_VIDEO_PXL_CRC_ERR,

    /* GMSL1 link based errors (cnt:3) */
    CDI_MAX96712_GMSL1_LINK_UNLOCK_ERR,
    CDI_MAX96712_GMSL1_LINK_DET_ERR,
    CDI_MAX96712_GMSL1_LINK_PKTCC_CRC_ERR,

    CDI_MAX96712_MAX_LINK_BASED_FAILURE_TYPES
} LinkFailureTypeMAX96712;

typedef enum {
    CDI_MAX96712_GPIO_0  = 0U,  /* MFP0 */
    CDI_MAX96712_GPIO_1  = 1U,  /* MFP1 */
    CDI_MAX96712_GPIO_2  = 2U,  /* MFP2 */
    CDI_MAX96712_GPIO_3  = 3U,  /* MFP3 */
    CDI_MAX96712_GPIO_4  = 4U,  /* MFP4 */
    CDI_MAX96712_GPIO_5  = 5U,  /* MFP5 */
    CDI_MAX96712_GPIO_6  = 6U,  /* MFP6 */
    CDI_MAX96712_GPIO_7  = 7U,  /* MFP7 */
    CDI_MAX96712_GPIO_8  = 8U,  /* MFP8 */
    CDI_MAX96712_GPIO_20 = 20U  /* This GPIO is not existed physically. Used only for the GPIO ID */
} GPIOIndexMAX96712;

typedef enum {
    CDI_MAX96712_PG_1920_1236_30FPS,
    CDI_MAX96712_PG_1920_1236_60FPS,
    CDI_MAX96712_PG_3848_2168_30FPS,
    CDI_MAX96712_PG_3848_2174_30FPS,
    CDI_MAX96712_PG_2880_1860_30FPS,
    CDI_MAX96712_PG_1920_1559_30FPS,
    CDI_MAX96712_PG_3840_2181_30FPS,
    CDI_MAX96712_PG_NUM,
} PGModeMAX96712;

typedef enum {
    CDI_MAX96712_PG_GEN_0,
    CDI_MAX96712_PG_GEN_1,
    CDI_MAX96712_PG_GEN_NUM,
} PGGENMAX96712;

typedef enum {
    CDI_MAX96712_PG_PCLK_150MHX,
    CDI_MAX96712_PG_PCLK_375MHX,
    CDI_MAX96712_PG_PCLK_NUM,
} PGPCLKMAX96712;

typedef struct {
    uint8_t vcID;
    /* flag to indicate if emb data lines have emb data type */
    bool isEmbDataType;
    DataTypeMAX96712 dataType;
    /* flag to indicate if data type override is enabled */
    bool isDTOverride;
    /* flag to indicate if pipeline output is mapped to unused CSI controller.
     * Used only for Max96712 TPG modes */
    bool isMapToUnusedCtrl;
    /* flag to indicate if all DTs come via a single pipeline */
    bool isSinglePipeline;
} LinkPipelineMapMAX96712;

typedef union {
    /* Used with CDI_READ_PARAM_CMD_MAX96712_REV_ID */
    RevisionMAX96712 revision;

    /* Used with CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR */
    struct {
        LinkMAX96712 link;
        uint8_t errVal;
    } ErrorStatus;

    /* Used with CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS */
    LinkMAX96712 link;

    /* Used with CDI_READ_PARAM_CMD_MAX96712_GET_PWR_METHOD */
    uint8_t pwrMethod;
} ReadParametersParamMAX96712;

typedef union {
    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
    struct {
        LinkPipelineMapMAX96712 linkPipelineMap[MAX96712_MAX_NUM_LINK];
        LinkMAX96712 link;
        bool isSinglePipeline;      /* flag to indicate if all DTs come via a single pipeline */
    } PipelineMapping;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG */
    struct {
        LinkPipelineMapMAX96712 linkPipelineMap[MAX96712_MAX_NUM_LINK];
        uint8_t linkIndex;
    } PipelineMappingTPG;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED */
    struct {
        LinkMAX96712 link;
        uint8_t step;
    } GMSL1HIMEnabled;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC */
    struct {
        LinkMAX96712 link;
        FSyncModeMAX96712 FSyncMode;
        uint32_t pclk;
        uint32_t fps;
    } FSyncSettings;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI */
    struct {
        uint8_t mipiSpeed;                       /* MIPI speed in multiples of 100MHz */
        PHYModeMAX96712 phyMode;                 /* CPHY or DPHY */
    } MipiSettings;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
    struct {
        DataTypeMAX96712 dataType;
        LinkMAX96712 link;
        bool embDataType;           /* Set to true if emb data has emb data type */
    } DoublePixelMode;

    /* Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_PG */
    struct {
        uint32_t width;
        uint32_t height;
        float frameRate;
        uint8_t linkIndex;
    } SetPGSetting;

    /* Used with
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_I2C_PORT
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNE
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID
     */
    LinkMAX96712 link;

    uint8_t gpioIndex;

    /* Used with
     * CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_ERRB_RX
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PIPELINE
     */
    uint8_t linkIndex;
} WriteParametersParamMAX96712;

/* Parameter type used for GetErrorStatus() */
typedef struct {
    GlobalFailureTypeMAX96712 globalFailureType[MAX96712_MAX_GLOBAL_ERROR_NUM]; /* Outp param */
    uint8_t globalRegVal[MAX96712_MAX_GLOBAL_ERROR_NUM];  /* Outp param */

    uint8_t pipeline;                       /* Inp param. A pipeline whose status needs to be checked */
    PipelineFailureTypeMAX96712 pipelineFailureType[MAX96712_NUM_VIDEO_PIPELINES][MAX96712_MAX_PIPELINE_ERROR_NUM]; /* Outp param */
    uint8_t pipelineRegVal[MAX96712_NUM_VIDEO_PIPELINES][MAX96712_MAX_PIPELINE_ERROR_NUM];  /* Outp param */

    uint8_t link;                           /* Inp param. A single link whose status needs to be checked */
    LinkFailureTypeMAX96712 linkFailureType[MAX96712_MAX_NUM_LINK][MAX96712_MAX_LINK_BASED_ERROR_NUM]; /* Outp param */
    uint8_t linkRegVal[MAX96712_MAX_NUM_LINK][MAX96712_MAX_LINK_BASED_ERROR_NUM];  /* Outp param */

    uint8_t count; /* Outp param, total max96712 error count in current state */
} ErrorStatusMAX96712;

typedef struct {
    /* These must be set in supplied client ctx during driver creation */
    GMSLModeMAX96712 gmslMode[MAX96712_MAX_NUM_LINK]; /* GMSL1 or GMSL2. Unused links must be set to
                                                         CDI_MAX96712_GMSL_MODE_UNUSED */
    I2CPortMAX96712 i2cPort;                          /* I2C port 1 or 2 */
    TxPortMAX96712 txPort;                            /* MIPI output port */
    MipiOutModeMAX96712 mipiOutMode;                  /* MIPI configuration */
    uint8_t lanes[MAX96712_MAX_NUM_PHY];              /* The number of lanes */
    PHYModeMAX96712 phyMode;                          /* CPHY or DPHY */
    bool passiveEnabled;                              /* Doesn't need to control sensor/serializer
                                                       * through aggregator */

    /* These will be overwritten during creation */
    RevisionMAX96712 revision;                        /* Chip revision information */
    uint16_t manualFSyncFPS;                          /* Used to store manual fsync frequency */
    uint8_t linkMask;                                 /* Indicate what links to be enabled */

    /* Long cable support */
    bool longCables[MAX96712_MAX_NUM_LINK];

    /* reset all sequence needed at init */
    bool defaultResetAll;

    /* PG setting */
    bool tpgEnabled;                                  /* TPG mode */
    PGModeMAX96712 pgMode[MAX96712_MAX_NUM_PG];       /* PG0/1 modes */
    uint8_t pipelineEnabled;                          /* Pipeline status. 0 - disabled, 1 - enabled */

    /** Recorder configuration
     * CDI_MAX96712_CFG_PIPE_COPY_NONE
     * CDI_MAX96712_CFG_PIPE_COPY_MODE_1
     * CDI_MAX96712_CFG_PIPE_COPY_MODE_2
     */
    CfgPipelineCopyMAX96712 cfgPipeCopy;
} ContextMAX96712;

DevBlkCDIDeviceDriver *GetMAX96712NewDriver(void);

LinkMAX96712 GetMAX96712Link(uint8_t linkNum);

NvMediaStatus
MAX96712CheckPresence(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96712SetDefaults(
    DevBlkCDIDevice *handle);

NvMediaStatus
MAX96712SetDeviceConfig(
    DevBlkCDIDevice *handle,
    uint32_t enumeratedDeviceConfig);

NvMediaStatus
MAX96712ReadRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX96712WriteRegister(
    DevBlkCDIDevice *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff);

NvMediaStatus
MAX96712WriteParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    size_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96712ReadParameters(
    DevBlkCDIDevice *handle,
    uint32_t parameterType,
    uint32_t parameterSize,
    void *parameter);
#if !NV_IS_SAFETY
NvMediaStatus
MAX96712DumpRegisters(
    DevBlkCDIDevice *handle);
#endif
NvMediaStatus
MAX96712GetErrorStatus(
    DevBlkCDIDevice *handle,
    uint32_t parameterSize,
    void *parameter);

NvMediaStatus
MAX96712GetSerializerErrorStatus(
    DevBlkCDIDevice *handle,
    bool * isSerError);

NvMediaStatus
MAX96712CheckLink(
    DevBlkCDIDevice *handle,
    uint32_t linkIndex,
    uint32_t linkType,
    bool display);

NvMediaStatus
MAX96712OneShotReset(
    DevBlkCDIDevice *handle,
    LinkMAX96712 link);

#endif /* CDI_MAX96712_H */
