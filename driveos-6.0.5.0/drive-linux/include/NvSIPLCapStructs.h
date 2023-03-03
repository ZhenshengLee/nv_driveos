/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef NVSIPL_CAP_STRUCTS_H
#define NVSIPL_CAP_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \defgroup NvSIPLCapStructs NvSIPL Capture definitions
 *
 * @brief NvSipl Cap Defines for image capture interface and input format types.
 *
 * @ingroup NvSIPLCamera_API
 */

/** @addtogroup NvSIPLCapStructs
 * @{
 */

typedef enum {
    /*! Specifies CSI port A. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A,
    /*! Specifies CSI port B. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B,
    /*! Specifies CSI port AB. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
    /*! Specifies CSI port C. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C,
    /*! Specifies CSI port D. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D,
    /*! Specifies CSI port CD. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD,
    /*! Specifies CSI port E. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E,
    /*! Specifies CSI port F. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F,
    /*! Specifies CSI port EF. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF,
    /*! Specifies CSI port G. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G,
    /*! Specifies CSI port H. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H,
    /*! Specifies CSI port GH. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH,
#if (NV_IS_SAFETY == 0)
    /*! Specifies CSI port A with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A1,
    /*! Specifies CSI port B with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B1,
    /*! Specifies CSI port C with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C1,
    /*! Specifies CSI port D with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D1,
    /*! Specifies CSI port E with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E1,
    /*! Specifies CSI port F with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F1,
    /*! Specifies CSI port G with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G1,
    /*! Specifies CSI port H with 1 lane. */
    NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H1,
#endif
    NVSIPL_CAP_CSI_INTERFACE_TYPE_MAX,
} NvSiplCapInterfaceType;

typedef enum {
    /*! Specifies YUV 4:2:2 8 bits. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
    /*! Specifies YUV 4:2:2 10 bits. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422_10,
    /*! Specifies YUV 4:4:4.
     * This input format is not supported */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV444,
    /*! Specifies RGB. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RGB888,
    /*! Specifies RAW 6. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW6,
    /*! Specifies RAW 7. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW7,
    /*! Specifies RAW 8. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW8,
    /*! Specifies RAW 10. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10,
    /*! Specifies RAW 12. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12,
    /*! Specifies RAW 14. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW14,
    /*! Specifies RAW 16. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW16,
    /*! Specifies RAW 20. */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW20,
    /*! Specifies User defined 1 (0x30). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_1,
    /*! Specifies User defined 2 (0x31). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_2,
    /*! Specifies User defined 3 (0x32). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_3,
    /*! Specifies User defined 4 (0x33). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_4,
    /*! Specifies User defined 5 (0x34). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_5,
    /*! Specifies User defined 6 (0x35). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_6,
    /*! Specifies User defined 7 (0x36). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_7,
    /*! Specifies User defined 8 (0x37). */
    NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_8
} NvSiplCapInputFormatType;

typedef enum {
    /*! Specifies 8 bits per pixel. */
    NVSIPL_BITS_PER_PIXEL_8 = 0,
    /*! Specifies 10 bits per pixel. */
    NVSIPL_BITS_PER_PIXEL_10,
    /*! Specifies 12 bits per pixel. */
    NVSIPL_BITS_PER_PIXEL_12,
    /*! Specifies 14 bits per pixel. */
    NVSIPL_BITS_PER_PIXEL_14,
    /*! Specifies 16 bits per pixel. */
    NVSIPL_BITS_PER_PIXEL_16,
    /*! Specifies 20 bits per pixel. This value is not supported */
    NVSIPL_BITS_PER_PIXEL_20
} NvSiplBitsPerPixel;

typedef enum {
    /*! Specifies that CSI is in DPHY mode. */
    NVSIPL_CAP_CSI_DPHY_MODE = 0,
    /*! Specifies that CSI is in CPHY mode. */
    NVSIPL_CAP_CSI_CPHY_MODE
} NvSiplCapCsiPhyMode;

/** Defines the minimum supported image width. */
#define NVSIPL_CAP_MIN_IMAGE_WIDTH       640U

/** Defines the maximum supported image width. */
#define NVSIPL_CAP_MAX_IMAGE_WIDTH       3848U

/** Defines the minimum supported image height. */
#define NVSIPL_CAP_MIN_IMAGE_HEIGHT      480U

/** Defines the maximum supported image height. */
#define NVSIPL_CAP_MAX_IMAGE_HEIGHT      2168U

/** Defines the minimum supported frame rate. */
#define NVSIPL_CAP_MIN_FRAME_RATE        10U

/** Defines the maximum supported frame rate. */
#define NVSIPL_CAP_MAX_FRAME_RATE        60U

/** \brief PIXEL_ORDER flags for YUV surface type. */
/** \brief NVSIPL_PIXEL_ORDER flags for YUV surface type. */
/** Luma component order flag. */
#define NVSIPL_PIXEL_ORDER_LUMA                (0x00000001U)
/** YUV component order flag. */
#define NVSIPL_PIXEL_ORDER_YUV                 (0x00000002U)
/** YVU component order flag. */
#define NVSIPL_PIXEL_ORDER_YVU                 (0x00000003U)
/** YUYV component order flag. */
#define NVSIPL_PIXEL_ORDER_YUYV                (0x00000004U)
/** YVYU component order flag. */
#define NVSIPL_PIXEL_ORDER_YVYU                (0x00000005U)
/** VYUY component order flag. */
#define NVSIPL_PIXEL_ORDER_VYUY                (0x00000006U)
/** UYVY component order flag. */
#define NVSIPL_PIXEL_ORDER_UYVY                (0x00000007U)
/** XUYV component order flag. */
#define NVSIPL_PIXEL_ORDER_XUYV                (0x00000008U)
/** XYUV component order flag. */
#define NVSIPL_PIXEL_ORDER_XYUV                (0x00000009U)
/** VUYX component order flag. */
#define NVSIPL_PIXEL_ORDER_VUYX                (0x0000000AU)

/** \brief NVM_SURF_ATTR_PIXEL_ORDER flags for RGBA surface type. */
/** Alpha component order flag. */
#define NVSIPL_PIXEL_ORDER_ALPHA               (0x00000011U)
/** RGBA component order flag. */
#define NVSIPL_PIXEL_ORDER_RGBA                (0x00000012U)
/** ARGB component order flag. */
#define NVSIPL_PIXEL_ORDER_ARGB                (0x00000013U)
/** BGRA component order flag. */
#define NVSIPL_PIXEL_ORDER_BGRA                (0x00000014U)
/** RG component order flag. */
#define NVSIPL_PIXEL_ORDER_RG                  (0x00000015U)

/** \brief NVM_SURF_ATTR_PIXEL_ORDER flags for RAW surface type. */
/** RGGB component order flag. */
#define NVSIPL_PIXEL_ORDER_RGGB                (0x00000021U)
/** BGGR component order flag. */
#define NVSIPL_PIXEL_ORDER_BGGR                (0x00000022U)
/** GRBG component order flag. */
#define NVSIPL_PIXEL_ORDER_GRBG                (0x00000023U)
/** GBRG component order flag. */
#define NVSIPL_PIXEL_ORDER_GBRG                (0x00000024U)

/** RCCB component order flag. */
#define NVSIPL_PIXEL_ORDER_RCCB                (0x00000025U)
/** BCCR component order flag. */
#define NVSIPL_PIXEL_ORDER_BCCR                (0x00000026U)
/** CRBC component order flag. */
#define NVSIPL_PIXEL_ORDER_CRBC                (0x00000027U)
/** CBRC component order flag. */
#define NVSIPL_PIXEL_ORDER_CBRC                (0x00000028U)

/** RCCC component order flag. */
#define NVSIPL_PIXEL_ORDER_RCCC                (0x00000029U)
/** CCCR component order flag. */
#define NVSIPL_PIXEL_ORDER_CCCR                (0x0000002AU)
/** CRCC component order flag. */
#define NVSIPL_PIXEL_ORDER_CRCC                (0x0000002BU)
/** CCRC component order flag. */
#define NVSIPL_PIXEL_ORDER_CCRC                (0x0000002CU)

/** CCCC component order flag. */
#define NVSIPL_PIXEL_ORDER_CCCC                (0x0000002DU)

/** \brief NVM_SURF_ATTR_PIXEL_ORDER flags for RAW RGB-IR surface type. */
/** BGGI_RGGI component order flag. */
#define NVSIPL_PIXEL_ORDER_BGGI_RGGI                (0x0000002EU)
/** GBIG_GRIG component order flag. */
#define NVSIPL_PIXEL_ORDER_GBIG_GRIG                (0x0000002FU)
/** GIBG_GIRG component order flag. */
#define NVSIPL_PIXEL_ORDER_GIBG_GIRG                (0x00000030U)
/** IGGB_IGGR component order flag. */
#define NVSIPL_PIXEL_ORDER_IGGB_IGGR                (0x00000031U)
/** RGGI_BGGI component order flag. */
#define NVSIPL_PIXEL_ORDER_RGGI_BGGI                (0x00000032U)
/** GRIG_GBIG component order flag. */
#define NVSIPL_PIXEL_ORDER_GRIG_GBIG                (0x00000033U)
/** GIRG_GIBG component order flag. */
#define NVSIPL_PIXEL_ORDER_GIRG_GIBG                (0x00000034U)
/** IGGR_IGGB component order flag. */
#define NVSIPL_PIXEL_ORDER_IGGR_IGGB                (0x00000035U)

/**
 * \brief Holds the capture input format.
 */
typedef struct {
    /*! Holds capture input format type. */
    NvSiplCapInputFormatType inputFormatType;
    /*! Holds number of bits per pixel for NVSIPL_CAP_INPUT_FORMAT_TYPE_USER_DEFINED_x
     input format types.*/
    NvSiplBitsPerPixel bitsPerPixel;
} NvSiplCapInputFormat;

#ifdef __cplusplus
}     /* extern "C" */
#endif /* __cplusplus */

#endif /* NVSIPL_CAP_STRUCTS_H */
