/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef CDI_MAX96712_NV_H
#define CDI_MAX96712_NV_H

#include "devblk_cdi_i2c.h"
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
#include <cstdbool>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#else
#include <stdbool.h>
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/** Maximum number of supported links */
#define MAX96712_MAX_NUM_LINK                 4U

#ifdef __cplusplus
static_assert(MAX96712_MAX_NUM_LINK <= 4U ,
               "avoid CERT INT31-C defect");
#else
_Static_assert(MAX96712_MAX_NUM_LINK <= 4U ,
               "avoid CERT INT31-C defect");
#endif

/** Maximum number of supported pattern generators */
#define MAX96712_MAX_NUM_PG                   2U
/** Maximum number of supported video pipelines */
#define MAX96712_NUM_VIDEO_PIPELINES          8U
/** Maximum number of supported global errors */
#define MAX96712_MAX_GLOBAL_ERROR_NUM         24U
/** Maximum number of supported pipeline errors */
#define MAX96712_MAX_PIPELINE_ERROR_NUM       16U
/** Maximum number of supported link errors */
#define MAX96712_MAX_LINK_BASED_ERROR_NUM     16U

#define MAX96712_REG_BACKTOP11_LOWER          0X040AU
#define MAX96712_REG_BACKTOP11_HIGHER         0X042AU
#define MAX96712_REG_BACKTOP1                 0x0400U
#define MAX96712_REG_INTR11                   0x002EU
#define MAX96712_REG_MEM_ECC0                 0x1250U
#define SHIFT_NIBBLE                          4U

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
namespace nvsipl {

extern "C" {
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif

/**
 * @brief Contains the masks used to perform read-verify and write-verify.
 * Masks are needed to avoid verification of bits which don't have a persistent state,
 * eg. Clear on Read bits will change state as soon as first read is done over them,
 * thus we can't perform read-verify over these bits as it would fail as the bits have
 * changed states.
 *
 * maskRead : this is used to perform read-verify
 * maskWrite : this is used to perform write-verify
 * If a register doesn't have any bit which can change states then the default mask i.e 0xFF can
 * be applied to verify all bits.
 */
typedef struct {
    /** Mask for read-verify */
    uint8_t      maskRead;
    /** Mask for write-verify */
    uint8_t      maskWrite;
} MAX96712RegMask;

/** enumerations of deserializer configuration items */
typedef enum {
    /* DO NOT re-order w/o checking */
    CDI_CONFIG_MAX96712_INVALID,                /*!< Invalid configuration */
    CDI_CONFIG_MAX96712_ENABLE_CSI_OUT,         /*!< Enable CSI output */
    CDI_CONFIG_MAX96712_DISABLE_CSI_OUT,        /*!< Disable CSI output */
    CDI_CONFIG_MAX96712_TRIGGER_DESKEW,         /*!< Start Deserializer Deskew pattern */
    CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK,      /*!< Check CSI PLL lock */
    CDI_CONFIG_MAX96712_ENABLE_ERRB,            /*!< Enable ERRB */
    CDI_CONFIG_MAX96712_DISABLE_ERRB,           /*!< Disable ERRB */
    CDI_CONFIG_MAX96712_ENABLE_LOCK,            /*!< Enable LOCK output to GPIO */
    CDI_CONFIG_MAX96712_DISABLE_LOCK,           /*!< Disable LOCK output to GPIO */
#if !NV_IS_SAFETY
    CDI_CONFIG_MAX96712_ENABLE_PG,
    CDI_CONFIG_MAX96712_MAP_UNUSED_PIPE,
    CDI_CONFIG_MAX96712_ENABLE_REPLICATION,
    CDI_CONFIG_MAX96712_DISABLE_REPLICATION,
#endif
    CDI_CONFIG_MAX96712_NUM,                    /*!< Number of configuration enums */
} ConfigSetsMAX96712;

/** enumerations of deserializer status items */
typedef enum {
    CDI_READ_PARAM_CMD_MAX96712_REV_ID = 1U,                /*!< Read Deserializer Revision */
    CDI_READ_PARAM_CMD_MAX96712_ERRB,                       /*!< Read ERRB status (if set) and clear */
    CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR,  /*!< Read Link CRC error count */
    CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS,              /*!< Read Link Enabled Mask */
} ReadParametersCmdMAX96712;

/** enumerations of deserializer writable parameters */
typedef enum {
    /* DO NOT re-order w/o checking */
    CDI_WRITE_PARAM_CMD_MAX96712_INVALID,                   /*!< Invalid param command */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS,     /*!< Enable Specific Links */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINKS_NO_CHECK,
    CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC,                 /*!< Set FSYNC mode */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC,     /*!< Set Heterogenous frame sync */
    CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING,          /*!< Set Pipeline mapping */
    CDI_WRITE_PARAM_CMD_MAX96712_OVERRIDE_DATATYPE,         /*!< Set Override Datatype */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI,                  /*!< Configure MIPI output */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_DOUBLE_PIXEL_MODE,  /*!< Enable Double Pixel mode */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID,             /*!< Set Source identifier used in packets transmitted from port */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PERIODIC_AEQ,       /*!< Enable periodic AEQ on Link */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX,            /*!< GPIO source enabled for GMSL2 reception */
    CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX,             /*!< Unset ERRB RX */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX,               /*!< Set ERRB RX */
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_LINK,              /*!< Disable link */
    CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_LINK,              /*!< Restore link */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID,            /*!< Set ERRB Receive ID */
#if !NV_IS_SAFETY
    CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG,      /*!< Set Pipeline mapping TPG */
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNEL, /*!< Disable Packet Based Control Channel */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL, /*!< Enable Packet Based Control Channel */
    CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS,   /*!< Enable Forward Channel */
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS,  /*!< Disable Forward Channel */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL,                   /*!< Enable Double Rate Output */
    CDI_WRITE_PARAM_CMD_MAX96712_UNSET_DBL,                 /*!< Disable Double Rate Output */
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE,                /*!< Disable Double Rate Output */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED, /*!< Enable GMSL1 High Immunity Mode */
    CDI_WRITE_PARAM_CMD_MAX96712_SET_PG,                    /*!< Set PG */
    CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_AUTO_ACK,          /*!< Disable I2c auto ACK */
#endif
    CDI_WRITE_PARAM_CMD_MAX96712_NUM,                       /*!< Number of write parameter commands */
} WriteParametersCmdMAX96712;
#if !NV_IS_SAFETY || SAFETY_DBG_OV
/** enumerations of pipeline copy configuration */
typedef enum {
     CDI_MAX96712_CFG_PIPE_COPY_NONE,       /*!< No pipeline copy support */
     CDI_MAX96712_CFG_PIPE_COPY_MODE_1,     /*!< Pipeline copy support mode 1. Phy mode 2x4. Link 0/1/2/3 to Phy B */
     CDI_MAX96712_CFG_PIPE_COPY_MODE_2,     /*!< Pipeline copy support mode 2. Phy mode 4+2x2. Link 0/1 to Phy E and Link 2/3 to Phy F */
     CDI_MAX96712_CFG_PIPE_COPY_INVALID,    /*!< Invalid pipeline copy configuration */
} CfgPipelineCopyMAX96712;
#endif
/** enumerations of GMSL modes */
typedef enum {
     CDI_MAX96712_GMSL_MODE_INVALID,    /*!< Invalid GMSL mode */
     CDI_MAX96712_GMSL_MODE_UNUSED,     /*!< Link unused */
     CDI_MAX96712_GMSL1_MODE,           /*!< Set Link to GMSL1 mode */
     CDI_MAX96712_GMSL2_MODE_6GBPS,     /*!< Set Link to GMSL2 mode (6GBPS) */
     CDI_MAX96712_GMSL2_MODE_3GBPS,     /*!< Set Link to GMSL2 mode (3GBPS) */
} GMSLModeMAX96712;

/** enumerations of deserializer I2C ports */
typedef enum {
    CDI_MAX96712_I2CPORT_INVALID,   /*!< Invalid I2C port */
    CDI_MAX96712_I2CPORT_0, /*!< Port 0 */
    CDI_MAX96712_I2CPORT_1, /*!< Port 1 */
    CDI_MAX96712_I2CPORT_2, /*!< Port 2 */
} I2CPortMAX96712;

/** enumerations of deserializer MIPI output modes */
typedef enum {
    CDI_MAX96712_MIPI_OUT_INVALID,
    CDI_MAX96712_MIPI_OUT_4x2,    /*!< Four 2 Lane */
    CDI_MAX96712_MIPI_OUT_2x4,    /*!< Two  4 Lane */
    CDI_MAX96712_MIPI_OUT_4a_2x2, /*!< One 4 Lane and Two 2 Lane */
} MipiOutModeMAX96712;

/** enumerations of deserializer PHY modes */
typedef enum {
    CDI_MAX96712_PHY_MODE_INVALID,  /*!< Invalid phy mode */
    CDI_MAX96712_PHY_MODE_DPHY,     /*!< CPHY mode */
    CDI_MAX96712_PHY_MODE_CPHY,     /*!< DPHY mode */
} PHYModeMAX96712;

/** enumerations of deserializer PHY ports */
typedef enum {
    CDI_MAX96712_TXPORT_PHY_C,  /*!< TxPort PHY C */
    CDI_MAX96712_TXPORT_PHY_D,  /*!< TxPort PHY D */
    CDI_MAX96712_TXPORT_PHY_E,  /*!< TxPort PHY E */
    CDI_MAX96712_TXPORT_PHY_F,  /*!< TxPort PHY F */
} TxPortMAX96712;

/** enumerations of deserializer revisions */
typedef enum {
    CDI_MAX96712_REV_INVALID, /*!< Invalid revision */
    CDI_MAX96712_REV_1, /*!< Revision 1 */
    CDI_MAX96712_REV_2, /*!< Revision 2 */
    CDI_MAX96712_REV_3, /*!< Revision 3 */
    CDI_MAX96712_REV_4, /*!< Revision 4 */
    CDI_MAX96712_REV_5, /*!< Revision 5 */
} RevisionMAX96712;

/** enumerations of deserializer links bitmap */
typedef enum {
    CDI_MAX96712_LINK_NONE = 0U,
    CDI_MAX96712_LINK_0 = (1U << 0U),   /*!< Link 0 */
    CDI_MAX96712_LINK_1 = (1U << 1U),   /*!< Link 1 */
    CDI_MAX96712_LINK_2 = (1U << 2U),   /*!< Link 2 */
    CDI_MAX96712_LINK_3 = (1U << 3U),   /*!< Link 3 */
    /** links 0, 1, 3 mask */
    CDI_MAX96712_LINK_0_1_3 =
        (CDI_MAX96712_LINK_0 |
        CDI_MAX96712_LINK_1 |
        CDI_MAX96712_LINK_3),
    /** all links mask */
    CDI_MAX96712_LINK_ALL =
        (CDI_MAX96712_LINK_0_1_3 |
        CDI_MAX96712_LINK_2) ,
} LinkMAX96712;

/** enumerations of deserializer FSYNC modes */
typedef enum {
    CDI_MAX96712_FSYNC_INVALID,     /*!< Invalid FYNC mode */
    CDI_MAX96712_FSYNC_MANUAL,      /*!< Manual FSYNC */
    CDI_MAX96712_FSYNC_AUTO,        /*!< Auto FSYNC */
    CDI_MAX96712_FSYNC_OSC_MANUAL,  /*!< Oscillator Manual */
    CDI_MAX96712_FSYNC_EXTERNAL,    /*!< External */
} FSyncModeMAX96712;

/** enumerations of deserializer Data types */
typedef enum {
    CDI_MAX96712_DATA_TYPE_INVALID, /*!<  */
    CDI_MAX96712_DATA_TYPE_RAW8,    /*!< RAW8 */
    CDI_MAX96712_DATA_TYPE_RAW10,   /*!< RAW10 */
    CDI_MAX96712_DATA_TYPE_RAW12,   /*!< RAW12 */
    CDI_MAX96712_DATA_TYPE_RAW16,   /*!< RAW16 */
    CDI_MAX96712_DATA_TYPE_RGB,     /*!< RGB */
    CDI_MAX96712_DATA_TYPE_YUV_8,   /*!< YUV8 */
    CDI_MAX96712_DATA_TYPE_YUV_10,  /*!< YUV10 */
} DataTypeMAX96712;

/** enumerations of deserializer LOCK types */
typedef enum {
/** Used as param for CheckLink() */
    CDI_MAX96712_LINK_LOCK_GMSL2 = 1U,  /*!< GMSL2 link lock */
    CDI_MAX96712_LINK_LOCK_VIDEO,       /*!< GMSL1 video link lock */
#if !NV_IS_SAFETY
    CDI_MAX96712_LINK_LOCK_GMSL1_CONFIG, /*!< GMSL1 config link lock */
#endif
} LinkLockTypeMAX96712;

/** enumerations of deserializer Line Failure types */
typedef enum {
    CDI_MAX96712_PIPELINE_ERROR_INVALID,        /*!< Invalid pipeline error */

    /* GMSL pipeline based errors (cnt:6) */
    CDI_MAX96712_PIPELINE_LMO_OVERFLOW_ERR,     /*!< Line Memory Overflow Error */
    CDI_MAX96712_PIPELINE_CMD_OVERFLOW_ERR,     /*!< Command Overflow Error */

    CDI_MAX96712_PIPELINE_PGEN_VID_UNLOCK_ERR,  /*!< Pattern Generator Video Unlock Error */

    CDI_MAX96712_PIPELINE_MEM_ERR,              /*!< Memory Error */

    CDI_MAX96712_PIPELINE_VID_SEQ_ERR,          /*!< Video Pipeline Sequence Error */

    CDI_MAX96712_PIPELINE_LCRC_ERR,             /*!< Video line CRC Error */

    CDI_MAX96712_MAX_PIPELINE_FAILURE_TYPES     /*!< number of pipeline errors */
} PipelineFailureTypeMAX96712;

/** enumerations of deserializer Global Failure types
    GMSL non-link based global errors (cnt:21) */
typedef enum {
    CDI_MAX96712_GLOBAL_ERROR_INVALID,  /*!< Invalid global error */
    CDI_MAX96712_GLOBAL_ERR,            /*!< ERRB */
    CDI_MAX96712_GLOBAL_CMU_UNLOCK_ERR, /*!< Clock Multiplier Unlocked error */
    CDI_MAX96712_GLOBAL_WM,             /*!< Water Mark error */
    CDI_MAX96712_GLOBAL_WM2,            /*!< Water Mark 2 error */
    CDI_MAX96712_GLOBAL_LINE_FAULT,     /*!< Line Fault */
    CDI_MAX96712_GLOBAL_MEM_STORE_CRC,  /*!< Memory Store CRC error */
    CDI_MAX96712_GLOBAL_FRAME_SYNC,     /*!< Frame Sync error */
    CDI_MAX96712_GLOBAL_REMOTE_SIDE,    /*!< Remote Side error */
    CDI_MAX96712_GLOBAL_VID_PRBS,       /*!< Video PRBS error */
    CDI_MAX96712_GLOBAL_VID_LINE_CRC,   /*!< Video Line CRC error */
    CDI_MAX96712_GLOBAL_MEM_ECC1,       /*!< Memory ECC single bit error */
    CDI_MAX96712_GLOBAL_MEM_ECC2,       /*!< Memory ECC  double bit error */
    CDI_MAX96712_GLOBAL_FSYNC_SYNC_LOSS,/*!< FSYNC sync loss error */
    CDI_MAX96712_GLOBAL_FSYNC_STATUS,   /*!< FSYNC status */
    CDI_MAX96712_GLOBAL_CMP_VTERM_STATUS,/*!< VTERM latched low and less than 1v */
    CDI_MAX96712_GLOBAL_VDD_OV_FLAG,    /*!< VDD Over Voltage flag */
    CDI_MAX96712_GLOBAL_VDDBAD_STATUS,  /*!< VDD Bad */
    CDI_MAX96712_GLOBAL_CMP_STATUS,     /*!< CMP Status */
    CDI_MAX96712_GLOBAL_VDDSW_UV,       /*!< VCCSW Under Voltage */
    CDI_MAX96712_GLOBAL_VDDIO_UV,       /*!< VDDIO Under Voltage */
    CDI_MAX96712_GLOBAL_VDD18_UV,       /*!< VDD18 Under Voltage */

    CDI_MAX96712_MAX_GLOBAL_FAILURE_TYPES /*!< number of global failure errors */
} GlobalFailureTypeMAX96712;

/** enumerations of deserializer Link Failure types */
typedef enum {
    /* DO NOT re-order w/o checking */
    CDI_MAX96712_GMSL_LINK_ERROR_INVALID,

    /* GMSL2 link based errors (cnt:7) */
    CDI_MAX96712_GMSL2_LINK_UNLOCK_ERR,       /*!< GMSL2 Link Unlock error */
    CDI_MAX96712_GMSL2_LINK_DEC_ERR,          /*!< GMSL2 Decoder error */
    CDI_MAX96712_GMSL2_LINK_IDLE_ERR,         /*!< GMSL2 Link Idle error */
    CDI_MAX96712_GMSL2_LINK_EOM_ERR,          /*!< GMSL2 Link EOM error */
    CDI_MAX96712_GMSL2_LINK_ARQ_RETRANS_ERR,   /*!< GMSL2 Link ARQ Retransmission error */
    CDI_MAX96712_GMSL2_LINK_MAX_RETRANS_ERR,   /*!< GMSL2 Link Max Retransmission error */
    CDI_MAX96712_GMSL2_LINK_VIDEO_PXL_CRC_ERR, /*!< GMSL2 Video Pixel CRC error */

#if !NV_IS_SAFETY
    /* GMSL1 link based errors (cnt:3) */
    CDI_MAX96712_GMSL1_LINK_UNLOCK_ERR,       /*!< GMSL1 Link Unlock error */
    CDI_MAX96712_GMSL1_LINK_DET_ERR,          /*!< GMSL1 Detected Decode error */
    CDI_MAX96712_GMSL1_LINK_PKTCC_CRC_ERR,    /*!< GMSL1 Link Packet Based Control Channel CRC error */
#endif

    CDI_MAX96712_MAX_LINK_BASED_FAILURE_TYPES  /*!< number of link based errors */
} LinkFailureTypeMAX96712;

/** Deserializer GPIO pin MFP0 */
#define CDI_MAX96712_GPIO_0   0U
/** Deserializer GPIO pin MFP1 */
#define CDI_MAX96712_GPIO_1   (CDI_MAX96712_GPIO_0 + 1U)
/** Deserializer GPIO pin MFP1 */
#define CDI_MAX96712_GPIO_2   (CDI_MAX96712_GPIO_1 + 1U)
/** Deserializer GPIO pin MFP3 */
#define CDI_MAX96712_GPIO_3   (CDI_MAX96712_GPIO_2 + 1U)
/** Deserializer GPIO pin MFP4 */
#define CDI_MAX96712_GPIO_4   (CDI_MAX96712_GPIO_3 + 1U)
/** Deserializer GPIO pin MFP5 */
#define CDI_MAX96712_GPIO_5   (CDI_MAX96712_GPIO_4 + 1U)
/** Deserializer GPIO pin MFP6 */
#define CDI_MAX96712_GPIO_6   (CDI_MAX96712_GPIO_5 + 1U)
/** Deserializer GPIO pin MFP7 */
#define CDI_MAX96712_GPIO_7   (CDI_MAX96712_GPIO_6 + 1U)
/** Deserializer GPIO pin MFP8 */
#define CDI_MAX96712_GPIO_8   (CDI_MAX96712_GPIO_7 + 1U)
/** This GPIO does not exist physically. Used only for the GPIO ID */
#define CDI_MAX96712_GPIO_20  (CDI_MAX96712_GPIO_8 + 12U)

/** enumerations of deserializer Pattern Generators modes */
typedef enum {
    CDI_MAX96712_PG_1920_1236_30FPS,
    CDI_MAX96712_PG_1920_1236_60FPS,
    CDI_MAX96712_PG_3848_2168_30FPS,
    CDI_MAX96712_PG_2880_1860_30FPS,
    CDI_MAX96712_PG_NUM,
} PGModeMAX96712;

/** contains deserializer Link Pipeline mapping information */
typedef struct {
    /** Virtual Channel ID */
    uint8_t vcID;
    /** flag to indicate if emb data lines have emb data type */
    bool isEmbDataType;
    DataTypeMAX96712 dataType;
    /** flag to indicate if data type override is enabled */
    bool isDTOverride;
    /** flag to indicate if pipeline output is mapped to unused CSI controller.
     * Used only for Max96712 TPG modes */
    bool isMapToUnusedCtrl;
    /** flag to indicate if all DTs come via a single pipeline */
    bool isSinglePipeline;
} LinkPipelineMapMAX96712;

/** contains parameters read from deserializer */
typedef struct {
    /** Used with CDI_READ_PARAM_CMD_MAX96712_REV_ID */
    RevisionMAX96712 revision;

    /** Used with CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR */
    struct {
        LinkMAX96712 link;
        uint8_t errVal;
    } ErrorStatus;

    /** Used with CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS */
    LinkMAX96712 link;
} ReadParametersParamMAX96712;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
typedef struct {
    LinkPipelineMapMAX96712 linkPipelineMap[MAX96712_MAX_NUM_LINK];
    LinkMAX96712 link;
} PipelineMappingStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG */
typedef struct {
    LinkPipelineMapMAX96712 linkPipelineMap[MAX96712_MAX_NUM_LINK];
    uint8_t linkIndex;
} PipelineMappingTPGStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC */
typedef struct {
    LinkMAX96712 link;
    FSyncModeMAX96712 FSyncMode;
    uint32_t pclk;
    uint32_t fps;
} FSyncSettingsStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI */
typedef struct  {
    /** MIPI speed in multiples of 100MHz */
    uint8_t mipiSpeed;
    /** CPHY or DPHY */
    PHYModeMAX96712 phyMode;
} MipiSettingsStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC */
typedef struct {
    /** GPIO number per link */
    uint32_t gpioNum[MAX96712_MAX_NUM_LINK];
} HeteroFSyncSettingsStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX and
 * CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID*/
typedef struct  {
    uint8_t gpioIndex;
    LinkMAX96712 link;
} GpioErrbSettingStruct;

/** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
typedef struct {
    DataTypeMAX96712 dataType;
    LinkMAX96712 link;
    /** Set to true if emb data has emb data type */
    bool embDataType;
    /** flag to indicate if all DTs come via a single pipeline */
    bool isSharedPipeline;
} DoublePixelModeStruct;

/** contains parameters to be written to deserializer */
typedef struct {
    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
    PipelineMappingStruct PipelineMapping;

    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING_TPG */
    PipelineMappingTPGStruct PipelineMappingTPG;
#if !NV_IS_SAFETY
    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_DEFAULT_GMSL1_HIM_ENABLED */
    struct {
        LinkMAX96712 link;
        uint8_t step;
    } GMSL1HIMEnabled;
#endif
    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_FSYNC */
    FSyncSettingsStruct FSyncSettings;

    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_MIPI */
    MipiSettingsStruct MipiSettings;

    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_PIPELINE_MAPPING */
    DoublePixelModeStruct DoublePixelMode;

    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_HETERO_FRAME_SYNC */
    HeteroFSyncSettingsStruct HeteroFSyncSettings;

#if !NV_IS_SAFETY
    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_SET_PG */
    struct {
        uint32_t width;
        uint32_t height;
        float_t frameRate;
        uint8_t linkIndex;
    } SetPGSetting;
#endif
    /** Used with CDI_WRITE_PARAM_CMD_MAX96712_UNSET_ERRB_RX and
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID*/
    GpioErrbSettingStruct GpioErrbSetting;

    /** Used with
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_I2C_PORT
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_SPECIFIC_LINK
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_FORWARD_CHANNELS
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_FORWARD_CHANNELS
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_PACKET_BASED_CONTROL_CHANNEL
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PACKET_BASED_CONTROL_CHANNE
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_DE
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_DBL
     * CDI_WRITE_PARAM_CMD_MAX96712_SET_TX_SRC_ID
     * CDI_WRITE_PARAM_CMD_MAX96712_ENABLE_GPIO_RX
     */
    LinkMAX96712 link;

    uint8_t gpioIndex;

    /** Used with
     * CDI_WRITE_PARAM_CMD_MAX96712_RESTORE_ERRB_RX
     * CDI_WRITE_PARAM_CMD_MAX96712_DISABLE_PIPELINE
     */
    uint8_t linkIndex;

} WriteParametersParamMAX96712;

/** Parameter type used for GetErrorStatus() */
typedef struct {
    /** Outp param */
    GlobalFailureTypeMAX96712 globalFailureType[MAX96712_MAX_GLOBAL_ERROR_NUM];

    /** Outp param */
    uint8_t globalRegVal[MAX96712_MAX_GLOBAL_ERROR_NUM];

    /** Inp param. A pipeline whose status needs to be checked */
    uint8_t pipeline;

    /** Outp param */
    PipelineFailureTypeMAX96712 pipelineFailureType[MAX96712_NUM_VIDEO_PIPELINES][MAX96712_MAX_PIPELINE_ERROR_NUM];

    /** Outp param */
    uint8_t pipelineRegVal[MAX96712_NUM_VIDEO_PIPELINES][MAX96712_MAX_PIPELINE_ERROR_NUM];

    /** Inp param. A single link whose status needs to be checked */
    uint8_t link;
    /** Outp param */
    LinkFailureTypeMAX96712 linkFailureType[MAX96712_MAX_NUM_LINK][MAX96712_MAX_LINK_BASED_ERROR_NUM];

    /** Outp param */
    uint8_t linkRegVal[MAX96712_MAX_NUM_LINK][MAX96712_MAX_LINK_BASED_ERROR_NUM];

    /** Outp param, total max96712 error count in current state */
    uint8_t count;
} ErrorStatusMAX96712;

/** contains context information for deserializer */
typedef struct {
    /** These must be set in supplied client ctx during driver creation
     *  GMSL1 or GMSL2. Unused links must be set to CDI_MAX96712_GMSL_MODE_UNUSED */
    GMSLModeMAX96712 gmslMode[MAX96712_MAX_NUM_LINK];

    /** I2C port 1 or 2 */
    I2CPortMAX96712 i2cPort;

    /** MIPI output port */
    TxPortMAX96712 txPort;

    /** MIPI configuration */
    MipiOutModeMAX96712 mipiOutMode;

    /** CPHY or DPHY */
    PHYModeMAX96712 phyMode;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
    /** Doesn't need to control sensor/serializer through aggregator */
    bool passiveEnabled;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

    /** These will be overwritten during creation */

    /** Chip revision information */
    RevisionMAX96712 revision;

    /** Used to store manual fsync frequency */
    uint16_t manualFSyncFPS;

    /** Indicate what links to be enabled */
    uint8_t linkMask;

    /** These will be updated before a link get programmed */

    /** whether link has been locked */
    bool linkHasBeenLocked[MAX96712_MAX_NUM_LINK];

    /** Used to store mask of active links */
    uint8_t activeLinkMask;

    /** Long cable support */
    bool longCables[MAX96712_MAX_NUM_LINK];

    /** reset all sequence needed at init */
    bool defaultResetAll;

    /** PG setting */

    /** TPG mode */
    bool tpgEnabled;

    /** PG0/1 modes */
    PGModeMAX96712 pgMode[MAX96712_MAX_NUM_PG];

    /** Pipeline status. 0 - disabled, 1 - enabled */
    uint8_t pipelineEnabled;

    /** Remote Errb Rx status. 0 - disabled, 1 - enabled */
    uint8_t errbRxEnabled;
#if !NV_IS_SAFETY || SAFETY_DBG_OV
    /** Recorder configuration
     * CDI_MAX96712_CFG_PIPE_COPY_NONE
     * CDI_MAX96712_CFG_PIPE_COPY_MODE_1
     * CDI_MAX96712_CFG_PIPE_COPY_MODE_2
     */
    CfgPipelineCopyMAX96712 cfgPipeCopy;
#endif
    /** Store the sync mode per link */
    FSyncModeMAX96712 FSyncMode[MAX96712_MAX_NUM_LINK];

    /** Store MFP number for the external synchronization per link */
    uint32_t FsyncExtGpioNum[MAX96712_MAX_NUM_LINK];
} ContextMAX96712;

/** @brief returns the device driver handle structure pointer
 * need to define the following static variables within function scope:
 * required for MISRA complience
 *
 *  ** true if structure is initialized, otherwise false
 *  static bool deviceDriverMAX96712_initialized = false;
 *
 *  ** device driver handler structure
 *  static DevBlkCDIDeviceDriver deviceDriverMAX96712;
 *
 * - initializes structure if not initialized yet by
 *  - static DevBlkCDIDeviceDriver deviceDriverMAX96712 = {
 *   - deviceDriverMAX96712.deviceName    = "Maxim 96712 Deserializer"
 *   - deviceDriverMAX96712.regLength     = (int32_t)MAX96712_NUM_ADDR_BYTES
 *   - deviceDriverMAX96712.dataLength    = (int32_t)MAX96712_NUM_DATA_BYTES
 *   - deviceDriverMAX96712.DriverCreate  = DriverCreateMAX96712,
 *   - deviceDriverMAX96712.DriverDestroy = DriverDestroyMAX96712,
 *   .
 *  - }
 *  .
 * @return pointer to deviceDriverMAX96712
 */
DevBlkCDIDeviceDriver *GetMAX96712NewDriver(void);

/** @brief returns Link ID for for deserializer for specified link number
 * @param[in]  linkNum   link number
 * @returns LinkMAX96712 enumeration for link number, or CDI_MAX96712_LINK_NONE if error*/
LinkMAX96712 GetMAX96712Link(uint8_t linkNum);

/** @brief checks the precence of deserializer by verifying its devId and revId
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies handle and device handle are not NULL
 * - Note: These values must include all of values in the RevisionMAX96712 enum
 *  - static const MAX96712SupportedRevisions supportedRevisionsMAX96712[5] = {
 *   - {CDI_MAX96712_REV_1, 0x1U},
 *   - {CDI_MAX96712_REV_2, 0x2U},
 *   - {CDI_MAX96712_REV_3, 0x3U},
 *   - {CDI_MAX96712_REV_4, 0x4U},
 *   - {CDI_MAX96712_REV_5, 0x5U},
 *   .
 *  - }
 *  .
 * - reads REG_FIELD_DEV_ID register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_DEV_ID, 0U, REG_READ_MODE)
 *   .
 *  .
 * - get read value from queue by
 *  - ReadFromRegFieldQ(handle, 0U)
 *  .
 * - verifies it is MAX96712_DEV_ID or  MAX96722_DEV_ID , error if not match
 * - reads REG_FIELD_DEV_REV register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_DEV_REV, 0U, REG_READ_MODE)
 *   .
 *  .
 * - get read value from queue by
 *  - revisionVal = ReadFromRegFieldQ(handle, 0U)
 *  .
 * - looks for a match in supportedRevisionsMAX96712
 * - if not found
 *  - error: Unsupported MAX96712 revision  detected
 *  .
 * - else
 *  - perform read/write verification by
 *   - status = MAX96712VerifyI2cReadWrite(handle)
 *   .
 *  .
 * @param[in] handle DEVBLK handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712CheckPresence(
    DevBlkCDIDevice const* handle);

/** @brief MAX96712 Set Defaults
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify drvHandle is not NULL
 * - initialize setGpioRegs by
 *  - DevBlkCDII2CReg const setGpioRegs[10] = {
 *   - {0x0306U, 0x80U}, ** GPIO2, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=0
 *   - {0x0306U, 0x90U}, ** GPIO2, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=1
 *
 *   - {0x0310U, 0x80U}, ** GPIO5, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=0
 *   - {0x0310U, 0x90U}, ** GPIO5, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=1
 *
 *   - {0x0313U, 0x80U}, ** GPIO6, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=0
 *   - {0x0313U, 0x90U}, ** GPIO6, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=1
 *
 *   - {0x0316U, 0x80U}, ** GPIO7, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=0
 *   - {0x0316U, 0x90U}, ** GPIO7, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=1
 *
 *   - {0x0319U, 0x80U}, ** GPIO8, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=0
 *   - {0x0319U, 0x90U}, ** GPIO8, unset GPIO_OUT_DIS, GPIO_RX_EN/GPIO_TX_EN, GPIO_OUT=1
 *   .
 *  -   }
 * - initialize wrapper for setGpioRegs by
 *   - DevBlkCDII2CRegList setGpioRegList = {
 *    - .regs = setGpioRegs,
 *    - .numRegs = (uint32_t)(sizeof(setGpioRegs) /
 *     - sizeof(setGpioRegs[0])),
 *     .
 *    .
 *   .
 *  - }
 *  .
 * - if (drvHandle->ctx.defaultResetAll == true )
 *  - status = ResetAll(handle)
 *  .
 * - if (drvHandle->ctx.revision == CDI_MAX96712_REV_3)
 *  - Note: Bug 2446492: Disable 2-bit ECC error reporting as spurious ECC errors are
 *   - intermittently observed on Rev C of MAX96712
 *   - Disable reporting 2-bit ECC errors to ERRB
 *   - status = EnableMemoryECC(status, handle, false, false)
 *   .
 *  .
 * - elif (drvHandle->ctx.revision >= CDI_MAX96712_REV_4)
 *  - status = EnableMemoryECC(status, handle, true, true)
 *  .
 * - else
 *  - nothing
 *  .
 * - status = EnableIndividualReset(status, handle)
 * - extract LinkMAX96712 linkMask = drvHandle->ctx.linkMask
 * - status = CheckAgainstActiveLink(status, handle, linkMask);
 * - status = SetLinkMode(status, handle, linkMask)
 * - if (drvHandle->ctx.revision == CDI_MAX96712_REV_5)
 *  -  status = SetCRUSSCModes(status, handle)
 *  .
 * - status = CheckAgainstActiveLink(status, handle, linkMask)
 * - status = GMSL2PHYOptimization(status, handle)
 * - Note: Default mode is GMSL2, 6Gbps one shot reset is required for GMSL1 mode & GMSL2
 * - status = CheckAgainstActiveLink(status, handle, linkMask)
 * - status = MAX96712OneShotReset(handle, linkMask)
 * - status = SetI2CPort(status, handle)
 * - status = MAX96712VerifyGPIOOutput(handle, &setGpioRegList)
 * - status = EnableExtraSMs(status, handle)
 * - status = VerifySMsEnabled(handle)
 * @param[in] handle Dev Blk handle
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712SetDefaults(
    DevBlkCDIDevice const* handle);

/** @brief Set Device Config
 * - if switchSetDeviceCfg == CDI_CONFIG_MAX96712_ENABLE_BACKTOP
 *  - call EnableBackTop(handle, true)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_DISABLE_BACKTOP
 *  - call EnableBackTop(handle, false)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_TRIGGER_DESKEW
 *  - call TriggerDeskew
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_CHECK_CSIPLL_LOCK
 *  - call CheckCSIPLLLock
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_ENABLE_ERRB
 *  - call EnableERRB(handle, true)
 *  .
 * - elif switchSetDeviceCfg == CDI_CONFIG_MAX96712_DISABLE_ERRB
 *  - call EnableERRB(handle, false)
 *  .
 * - else
 *  - Error: Bad parameter: Invalid command
 *
 * @param[in] handle Dev Blk handle
 * @param[in] enumeratedDeviceConfig device configuration to apply
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712SetDeviceConfig(
    DevBlkCDIDevice const* handle,
    ConfigSetsMAX96712 enumeratedDeviceConfig);

NvMediaStatus
MAX96712ReadRegister(
    DevBlkCDIDevice const* handle,
    uint16_t registerNum,
    uint8_t *dataBuff);


/**
 * @brief Read the register with read verify
 *
 * @param[in] handle Device Handle for DevBlkCDIDevice
 * @param[in] registerAddr register address on device to read
 * @param[out] regData read data
 *
 * @retval NVMEDIA_STATUS_OK Success
 * @retval NVMEDIA_STATUS_BAD_PARAMETER If invalid parameter
 * @retval NVMEDIA_STATUS_ERROR on any other error
 */
NvMediaStatus
MAX96712ReadRegisterVerify(
    DevBlkCDIDevice const* handle,
    uint16_t registerAddr,
    uint8_t *regData);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus
MAX96712WriteRegister(
    DevBlkCDIDevice const* handle,
    uint32_t deviceIndex,
    uint16_t registerNum,
    uint32_t dataLength,
    uint8_t const* dataBuff);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/** @brief Write all Parameters
 * - verifies that handle, device driver handle, and parameter are not NULL
 * - note: verify range supported in safety build
 * - if (parameterType > CDI_WRITE_PARAM_CMD_MAX96712_INVALID) &&
 *  - (parameterType <= CDI_WRITE_PARAM_CMD_MAX96712_SET_ERRB_RX_ID)
 *  - call MAX96712WriteParametersGroup1
 *  .
 * - else
 *  - error: Bad parameter: invalid write param cmd
 * @param[in] handle dev BLK handle
 * @param[in] parameterType parameter type to apply
 * @param[in] parameterSize user specified parameter record size
 * @param[in] parameter void pointer to user's parameter buffer
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712WriteParameters(
    DevBlkCDIDevice const* handle,
    WriteParametersCmdMAX96712 parameterType,
    size_t parameterSize,
    void const* parameter);

/** @brief  Read Parameters
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies drvHandle and parameter are not NULL
 * - if parameterType is CDI_READ_PARAM_CMD_MAX96712_REV_ID
 *  - verifie parameterSize == sizeof(param->revision)
 *  - return value of context revision by
 *   - param->revision = drvHandle->ctx.revision
 *   .
 *  .
 * - elif parameterType is CDI_READ_PARAM_CMD_MAX96712_CONTROL_CHANNEL_CRC_ERROR
 *  - verifie parameterSize == sizeof(param->ErrorStatus)
 *  - call ReadCtrlChnlCRCErr for returning value by
 *   - status = ReadCtrlChnlCRCErr(
 *    - handle, parameterSize, sizeof(param->ErrorStatus),
 *    - param->ErrorStatus.link, &param->ErrorStatus.errVal)
 *    .
 *   .
 *  -
 * - elif parameterType is CDI_READ_PARAM_CMD_MAX96712_ENABLED_LINKS
 *  - verifie parameterSize == sizeof(param->link)
 *  - call MAX96712GetEnabledLinks for returning value by
 *   - status = MAX96712GetEnabledLinks(handle, (uint8_t *)&param->link)
 *   .
 *  .
 * - elif parameterType is CDI_READ_PARAM_CMD_MAX96712_ERRB
 *  -  verifie parameterSize == sizeof(param->ErrorStatus)
 *  - call ClearErrb for returning value by
 *   - status = ClearErrb(handle, parameterSize, sizeof(param->ErrorStatus),
 *    - &param->ErrorStatus.link, &param->ErrorStatus.errVal)
 *    .
 *   .
 *  .
 * - else
 *  - exit error NVMEDIA_STATUS_BAD_PARAMETER
 *
 * @param[in] handle DEVBLK handle
 * @param[out] parameterType specifies pameter type requested
 * @param[in] parameterSize size of buffer for returned buffer
 * @param[out] parameter buffer pointer for storing status
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712ReadParameters(
    DevBlkCDIDevice const* handle,
    ReadParametersCmdMAX96712 parameterType,
    size_t parameterSize,
    void *parameter);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
NvMediaStatus
MAX96712DumpRegisters(
    DevBlkCDIDevice *handle);
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

/** @brief Get all Error Status
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verify that handle, device driver handle, and parameter are not NULL
 * - verify that parameterSize == sizeof(ErrorStatusMAX96712)
 * - get  all errors by
 *  - status = MAX96712GetPipelineErrorStatus(handle, errorStatus)
 *  - status = MAX96712GetLinkErrorStatus(handle, errorStatus)
 *  - status = MAX96712GetGlobalErrorStatus(handle, errorStatus)
 *  .
 *
 * @param[in] handle DEVBLK handle
 * @param[in] parameterSize caller specified parameter buffer size
 * @param[out] parameter pointer to return status buffer
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712GetErrorStatus(
    DevBlkCDIDevice const* handle,
    uint32_t parameterSize,
    void *parameter);

/** @brief  Get Serializer Error Status
 * - verifies handle is not NULL
 * - reads REG_FIELD_REM_ERR_FLAG register by
 *  - status = Max96712AccessOneRegField(
 *   - status, handle, REG_FIELD_REM_ERR_FLAG, 0U, REG_READ_MODE)
 *   .
 * - if (ReadFromRegFieldQ(handle, 0U) == 1U)
 *  - error: *isSerError = true
 *  .
 *
 * @param[in] handle DEVBLK handle
 * @param[out] isSerError set to true if error detected
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712GetSerializerErrorStatus(
    DevBlkCDIDevice const* handle,
    bool * isSerError);

/** @brief Check Link status
 *
 * - extracts device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle)
 *  .
 * - verifies handle and device driver handle are not NULL
 * - for each link
 *  - if (MAX96712_IS_GMSL_LINK_SET(link, linkIndex))
 *   - extract gmslMode from context' gmslMode[linkIndex]


 *   - if linkType == CDI_MAX96712_LINK_LOCK_GMSL2
 *    - if  (!IsGMSL2Mode(gmslMode))
 *     - error: Link: GMSL2 link lock is only valid in GMSL2 mode, mode
 *     - break link loop
 *     .
 *    - if ( context' revision == CDI_MAX96712_REV_1) && (linkIndex > 0U)
 *     - Error" GMSL2 link lock for link %u is not available on MAX96712 Rev 1
 *     - break link loop
 *     .
 *    - else
 *     - loop for 50 times try
 *     - read REG_FIELD_GMSL2_LOCK_A register by
 *      - status = Max96712AccessOneRegFieldOffset(
 *       - status, handle, REG_FIELD_GMSL2_LOCK_A,
 *       - linkIndex, 0U, REG_READ_MODE)
 *       .
 *      .
 *     - if  read values value (ReadFromRegFieldQ(handle, 0U)  == 0x01
 *      - lock identified - break loop with success
 *      .
 *     - else
 *      -  nvsleep(10000)
 *      .
 *     .
 *    - if not success
 *     - exit error: MAX96712: Link : GMSL2 link lock not detected
 *     -  break link loop
 *     .
 *   - else
 *    - exit Error: Bad parameter: Invalid link type
 * @param[in] handle DEVBLK handle
 * @param[in] link link mask
 * @param[in] linkType link type to check
 * @param[in] display boolean, to enable printouts if true
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code */
NvMediaStatus
MAX96712CheckLink(
    DevBlkCDIDevice const* handle,
    const LinkMAX96712 link,
    const LinkLockTypeMAX96712 linkType,
    bool display);

/** @brief Helper function for clearing deserializer oneshot for a given link mask
 * - verifies not previous error occured
 * - Verifies validity of link mask
 * - writes REG_FIELD_RESET_ONESHOT register with link value
 *
 * @param[in] handle   DEVBLK handle
 * @param[in] link link mask where oneshot is to be reset
 * @return NVMEDIA_STATUS_OK if done, otherwise an error code */
NvMediaStatus
MAX96712OneShotReset(
    DevBlkCDIDevice const* handle,
    const LinkMAX96712 link);

NvMediaStatus
MAX96712PRBSChecker(
    DevBlkCDIDevice const* handle,
    bool enable);

NvMediaStatus
MAX96712VerifyVPRBSErrs(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX96712VerifyERRGDiagnosticErrors(
    DevBlkCDIDevice const* handle);

NvMediaStatus
MAX96712EnableDECErrToERRB(
    DevBlkCDIDevice const* handle,
    bool enable);

/**
 * @brief Check if GMSL2 link locked.
 *
 * - extract device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
 *  .
 * - verify if driver handle or status is not NULL and link >= MAX96712_MAX_NUM_LINK
 * - read lock status by
 *  - DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - (uint16_t)regBitFieldProps
 *    - [REG_FIELD_GMSL2_LOCK_A + linkIndex].regAddr,
 *     - &lockStatus);
 *     .
 *    .
 *   .
 *  .
 * - if lock status & 0x8
 *  - bLocked = true, means no error.
 *  .
 * .
 * @param[in] handle DEVBLK handle
 * @param[in] link GMSL2 link number
 * @param[in] bLocked lock status
 * @return NVMEDIA_STATUS_OK if all done, else error code.
*/
NvMediaStatus
IsLinkLock(
    DevBlkCDIDevice const* handle,
    const uint8_t link,
    bool *bLocked);

/**
 * @brief Check if ERRB set
 *
 * - extract device driver handle from handle by
 *  - DriverHandleMAX96712 const* drvHandle = getHandlePrivMAX96712(handle);
 *  .
 * - verify if driver handle or status is not NULL
 * - read Errb status by
 *  - DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
 *   - (uint16_t)regBitFieldProps
 *    - [REG_FIELD_ERRB].regAddr,
 *     - &errbStatus);
 *     .
 *    .
 *   .
 *  .
 * - if errb status & 0x4
 *  - bSet = true, means error.
 *  .
 * .
 * @param[in] handle DEVBLK handle
 * @param[in] bSet errb status
 * @return NVMEDIA_STATUS_OK if all done, else error code.
*/
NvMediaStatus
IsErrbSet(
    DevBlkCDIDevice const* handle,
    bool *bSet);

/**
 * @brief Get Pipeline video sequence error for custom interface
 *
 * @param[in] handle DEVBLK handle
 * @param[out] customErrInfo pointer for storing custom information.
 *                           Every bit holds the status of VID_SEQ_ERR
 *                           per pipeline, for a total of 8 pipelines.
 * @return NVMEDIA_STATUS_OK all done, otherwise an error code
 */
NvMediaStatus
MAX96712GetVidSeqError(
    DevBlkCDIDevice const* handle,
    uint8_t *customErrInfo);

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#ifdef __cplusplus
} /* end of extern "C" */
} /* end of namespace nvsipl */
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1433 */
#endif
#endif /* CDI_MAX96712_NV_H */
