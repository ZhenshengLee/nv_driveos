/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: NvMedia Image Encode Processing Input ExtraData </b>
 *
 * This file contains the input extradata definition for "Image Encode Processing API".
 */

#ifndef NVMEDIA_IEP_INPUT_EXTRA_DATA_H
#define NVMEDIA_IEP_INPUT_EXTRA_DATA_H

#include <stdbool.h>
#include "nvmedia_core.h"
#include "nvmedia_iep.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum number of reference pictures including current frame */
#define NVMEDIA_ENCODE_MAX_RPS_SIZE 17U

/**
 * @brief Defines video frame flags. Each bit of bEncodeParamsFlag indicate parameter for specific featue.
 */
typedef enum {
    /** \hideinitializer \brief enable preprocessing enhancements buffer */
    NvMediaVideoEncFrame_PPEMetadata         = (1<<6),
    /** \hideinitializer \brief enable QP Delta Buffer */
    NvMediaVideoEncFrame_QPDeltaBuffer       = (1<<7),
    /* Add other flags using bitfields */
} NvMediaVideoEncEncFrameFlags;

/**
 * Holds an Video encoder input extradata configuration.
 */
typedef struct {
    /** Size of this extradata structure. This needs to be filled correctly by client
      * This size is used as sanity check before reading input extradata from this buffer */
    uint32_t ulExtraDataSize;
    /** bit fields defined in \ref NvMediaVideoEncEncFrameFlags to indicate valid frame parameters */
    uint32_t EncodeParamsFlag;
    /** Preprocessing enhancements metadata */
    uint8_t *PPEMetadata;
    /** Parameter to program QP Delta Buffer*/
    signed char *QPDeltaBuffer;
    /** Parameter to program QP Delta Buffer Size */
    uint32_t QPDeltaBufferSize;
} NvMediaEncodeInputExtradata;

/**
 * Defines the resolution change parameters. This is used if encoding resolution change on the fly.
 */
typedef struct
{
    /** Holds the encode Width. It will be used for DRC */
    uint16_t ulDRCWidth;
    /** Holds the encode Height. It will be used for DRC */
    uint16_t ulDRCHeight;
} NvMediaEncodeDRCConfig;

/**
 * Specifies the Video encoder set attribute type.
 * This can be extended to set other encoding parameter information.
 * \hideinitializer
 */
typedef enum {
    /** This is used to pass dynamic resolution change specific information \ref
        NvMediaEncodeDRCConfig. Driver will update encode resolution specific
        parameters for encoding with new specified resolution.
        Supported for H264 and H265 Encoding.*/
    NvMediaEncSetAttr_DRCParams,
} NvMediaEncSetAttrType;

/**
 * \brief Set the encoder extra data for current frame for encoding.
 * This should be called before before every \ref NvMediaIEPFeedFrame
 * so that parameters can be applied for current frame.
 * This should be called from same thread from where \ref NvMediaIEPFeedFrame
 * is called.
 * \param[in] encoder The encoder to use.
 * \param[in] extradata encode input extra data for current frame.
 * Input extradata ia defined as
 * \n \ref NvMediaEncodeInputExtradata
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetInputExtraData(
    const NvMediaIEP *encoder,
    const void *extradata
);

/**
 * \brief Set the encoder attribute for current encoding session.
 * This can be called before passing the first frame for encoding.
 * \param[in] encoder The encoder to use.
 * \param[in] attrType attribute type as defined in \ref NvMediaEncSetAttrType
 * \param[in] attrSize size of the data structure associated with attribute
 * Possible values are:
 * \n \ref NvMediaEncSetAttr_DRCParams if resolution of encoding changes.
 * \param[in] pointer to data structure associated with attribute
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVMEDIA_STATUS_OK
 * \n \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetAttribute(
    const NvMediaIEP *encoder,
    NvMediaEncSetAttrType attrType,
    uint32_t attrSize,
    const void *AttributeData
);

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_INPUT_EXTRA_DATA_H */
