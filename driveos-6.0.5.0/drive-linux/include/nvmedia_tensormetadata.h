/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: Tensor Metadata Interface </b>
 *
 * @b Description: This file defines the Tensor metadata structure
 *                 to be used for Tensor Interop.
 *
 */

#ifndef NVM_TENSORMETADATA_H
#define NVM_TENSORMETADATA_H

#ifdef __cplusplus
extern "C" {
#endif

 /**
 * \defgroup nvmedia_tensor_metadata_api Tensor Metadata API
 *
 * The NvMedia Tensor Metadata API encompasses all NvMedia Tensor metadata processing.
 *
 * @ingroup nvmedia_tensor_top
 * @{
 *
 * Tensor metadata processing follows the producer/consumer model:
 * - The producer must first populate the \ref NvMediaTensorMetaData structure
 *   using producer metadata APIs.
 * - The producer posts the tensor.
 * - On the CUDA consumer side, the consumer acquires the metadata and the
 *   CUgraphicsResource, to form the tensor in the CUDA domain.
 */

/** \brief Defines the maximum number of tensor dimensions. */
#define NVMEDIA_TENSOR_MAX_DIMENSIONS                       (8u)

/** \brief NVM_TENSOR_ATTR_DIMENSION_ORDER flags */
/**  Specifies the NHWC dimension order for 4-D tensors. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NHWC                (0x00000001u)
/** Specifies the NCHW dimension order for 4-D tensors. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NCHW                (0x00000002u)
/** Specifies the NCxHWCx dimension order for 4-D tensors. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NCxHWx              (0x00000003u)

#if (NV_IS_SAFETY == 0)
/** Specifies the unsigned integer tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_UINT                      (0x00000001u)
#endif

/** Specifies the integer tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_INT                       (0x00000002u)
/** Specifies the float tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_FLOAT                     (0x00000003u)

#if (NV_IS_SAFETY == 0)
/** Indicates that each element is 64 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_64                 (64U)
/** Indicates that each element is 32 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_32                 (32U)
#endif /* (NV_IS_SAFETY == 0) */

/** Indicates that each element is 16 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_16                 (16U)
/** Indicates that each element is 8 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_8                  (8U)

#if (NV_IS_SAFETY == 0)
/**
 * \defgroup stride_dim Stride and Dimension Indices
 *
 * The application can use the stride and dimension indices defined below
 * to access the strides and sizes from the corresponding arrays. For example,
 * the stride indices are used to access `dimstrides[]` in the
 * NvMediaTensorMetaData structure.
 * \ingroup nvmedia_tensor_metadata_api
 * @{
 */

/**
 * \defgroup nhwc Stride Indices - NHWC
 *
 * For the NHWC tensor, assuming `p,q,r,s` correspond to `N,H,W,C` dimensions,
 * the offset from the base address of the tensor to access a tensor value
 * at coordinate (`p,q,r,s`) is calculated using the expression below:
 *
 * \code
 * offset = p * tensormetadata.dimstrides[NVM_TENSOR_NHWC_N_STRIDE_INDEX] +
 *          q * tensormetadata.dimstrides[NVM_TENSOR_NHWC_H_STRIDE_INDEX] +
 *          r * tensormetadata.dimstrides[NVM_TENSOR_NHWC_W_STRIDE_INDEX] +
 *          s * tensormetadata.dimstrides[NVM_TENSOR_NHWC_C_STRIDE_INDEX]
 * \endcode
 * \ingroup stride_dim
 * @{
 */

/**
 * Defines the stride index for NHWC element.
 * `tensormetadata.dimstrides[NVM_TENSOR_NHWC_E_STRIDE_INDEX]` is stride
 * in bytes to go from the current element to the next element for NHWC tensor.
 * \ingroup nhwc
 */
#define NVM_TENSOR_NHWC_E_STRIDE_INDEX         0U

/**
 * Defines the stride index for NHWC channel.
 * `tensormetadata.dimstrides[NVM_TENSOR_NHWC_C_STRIDE_INDEX]` is stride
 * in bytes to go from current channel(C) to the next channel(C) for NHWC tensor.
 * The index is assigned to the same index as for the element stride.
 * \ingroup nhwc
 */
#define NVM_TENSOR_NHWC_C_STRIDE_INDEX         NVM_TENSOR_NHWC_E_STRIDE_INDEX

/**
 * Defines the stride index for NHWC column along the W dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NHWC_W_STRIDE_INDEX]` is stride
 * in bytes to go from one column to the next column or along the W dimension
 * for NHWC tensor.
 * \ingroup nhwc
 */
#define NVM_TENSOR_NHWC_W_STRIDE_INDEX         1U

/**
 * Defines the stride index for NHWC line along the H dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NHWC_H_STRIDE_INDEX]` is stride
 * in bytes to go from one line to the next line or along the H dimension
 * for NHWC tensor.
 * \ingroup nhwc
 */
#define NVM_TENSOR_NHWC_H_STRIDE_INDEX         2U

/**
 * Defines the stride index for NHWC plane along the N dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NHWC_N_STRIDE_INDEX]` is stride
 * in bytes to go from one plane to the next plane along the N dimension
 * for NHWC tensor.
 * \ingroup nhwc
 */
#define NVM_TENSOR_NHWC_N_STRIDE_INDEX         3U

/** @} <!-- ends nhwc group --> */

/**
 * \defgroup nchw Stride Indices - NCHW
 *
 * For NCHW tensor, assuming `p,q,r,s` correspond to `N,C,H,W` dimensions,
 * the offset from the base address of the tensor to access a tensor value
 * at coordinate (`p,q,r,s`) is calculated using the expression below:
 *
 * \code
 * offset = p * tensormetadata.dimstrides[NVM_TENSOR_NCHW_N_STRIDE_INDEX] +
 *          q * tensormetadata.dimstrides[NVM_TENSOR_NCHW_C_STRIDE_INDEX] +
 *          r * tensormetadata.dimstrides[NVM_TENSOR_NCHW_H_STRIDE_INDEX] +
 *          s * tensormetadata.dimstrides[NVM_TENSOR_NCHW_W_STRIDE_INDEX]
 * \endcode
 * \ingroup stride_dim
 * @{
 */

/**
 * Defines the stride index for NCHW element.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCHW_E_STRIDE_INDEX]` is stride
 * in bytes to go from current element to the next element for NCHW tensor
 * \ingroup nchw
 */
#define NVM_TENSOR_NCHW_E_STRIDE_INDEX         0U

/**
 * Defines the stride index for NCHW column along the W dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCHW_W_STRIDE_INDEX]` is stride
 * in bytes to go from one column to the next column or along W dimension
 * for NCHW tensor.
 * The index is assigned to the same index as for the element stride.
 * \ingroup nchw
 */
#define NVM_TENSOR_NCHW_W_STRIDE_INDEX         NVM_TENSOR_NCHW_E_STRIDE_INDEX

/**
 * Defines the stride index for NCHW line along the H dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCHW_H_STRIDE_INDEX]` is stride
 * in bytes to go from one line to the next line or along H dimension
 * for NCHW tensor.
 * \ingroup nchw
 */
#define NVM_TENSOR_NCHW_H_STRIDE_INDEX         1U

/**
 * Defines the stride index for NCHW channel.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCHW_C_STRIDE_INDEX]` is stride
 * in bytes to go from current channel(C) to the next channel(C) for NCHW tensor.
 * \ingroup nchw
 */
#define NVM_TENSOR_NCHW_C_STRIDE_INDEX         2U

/**
 * Defines the stride index for NCHW plane along the N dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCHW_N_STRIDE_INDEX]` is stride
 * in bytes to go from one plane to the next plane along N dimension
 * for NCHW tensor.
 * \ingroup nchw
 */
#define NVM_TENSOR_NCHW_N_STRIDE_INDEX         3U

/** @} <!-- ends nchw group --> */

/**
 * \defgroup ncxhwx Stride Indices - NCxHWx
 *
 * For NCxHWx tensor, assuming `p,q,r,s` correspond to `N,C,H,W` dimensions,
 * the offset from the base address of the tensor to access a tensor value
 * at coordinate (`p,q,r,s`) is calculated using the expression below:
 *
 * \code
 * offset = p * tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_N_STRIDE_INDEX] +
 *          floor(q/x) * tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_Cx_STRIDE_INDEX] +
 *          r * tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_H_STRIDE_INDEX] +
 *          s * tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_W_STRIDE_INDEX] +
 *          (q % x) * tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_X_STRIDE_INDEX] +
 *          s * tensormetadata.dimstrides[NVM_TENSOR_NCHW_W_STRIDE_INDEX]
 * \endcode
 * \ingroup stride_dim
 */

/**
 * Defines the stride index for NCxHWx element.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_E_STRIDE_INDEX]` is stride
 * in bytes to go from current element to the next element for NCxHWx tensor
 * \ingroup ncxhwx
 * @{
 */
#define NVM_TENSOR_NCxHWx_E_STRIDE_INDEX       0U

/**
 * Defines the stride index for NCxHWx channel.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_H_STRIDE_INDEX]` is stride
 * in bytes to go from one channel(C) to the next channel(C) along X dimension
 * for the NCxHWx tensor.
 * The index is assigned to the same index as for the element stride.
 * \ingroup ncxhwx
 */
#define NVM_TENSOR_NCxHWx_X_STRIDE_INDEX       NVM_TENSOR_NCxHWx_E_STRIDE_INDEX

/**
 * Defines the stride index for NCxHWx column along the W dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_W_STRIDE_INDEX]` is stride
 * in bytes to go from one column to the next column or along W dimension
 * for NCxHWx tensor.
 * \ingroup ncxhwx
 */
#define NVM_TENSOR_NCxHWx_W_STRIDE_INDEX       1U

/**
 * Defines the stride index for NCxHWx line along the H dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_H_STRIDE_INDEX]` is stride
 * in bytes to go from one line to the next line or along the H dimension
 * for the NCxHWx tensor.
 * \ingroup ncxhwx
 */
#define NVM_TENSOR_NCxHWx_H_STRIDE_INDEX       2U

/**
 * Defines the stride index for NCxHWx channel along the Cx dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_Cx_STRIDE_INDEX]` is stride
 * in bytes to go from one channel(C) to the next channel(C) along the Cx
 * dimension for the NCxHWx tensor.
 * \ingroup ncxhwx
 */
#define NVM_TENSOR_NCxHWx_Cx_STRIDE_INDEX      3U

/**
 * Defines the stride index for NCxHWx plane along the N dimension.
 * `tensormetadata.dimstrides[NVM_TENSOR_NCxHWx_N_STRIDE_INDEX]` is stride
 * in bytes to go from one plane to the next plane along the N dimension
 * for the NCxHWx tensor.
 * \ingroup ncxhwx
 */
#define NVM_TENSOR_NCxHWx_N_STRIDE_INDEX       4U

/** @} <!-- ends ncxhwx group --> */

/**
 * \defgroup nhwc_dim Dimension Indices - NHWC
 *
 * \ingroup stride_dim
 * @{
 */

/**
 * Defines the C dimension index for NHWC.
 * `tensormetadata.dimSizes[NVM_TENSOR_NHWC_C_DIMSZ_INDEX] = C`, is the dimension
 * value along C dimension for NHWC tensor.
 * \ingroup nhwc_dim
 */
#define NVM_TENSOR_NHWC_C_DIMSZ_INDEX          0U

/**
 * Defines the W dimension index for NHWC.
 * `tensormetadata.dimSizes[NVM_TENSOR_NHWC_W_DIMSZ_INDEX] = W`, is the dimension
 * value along W dimension for NHWC tensor.
 * \ingroup nhwc_dim
 */
#define NVM_TENSOR_NHWC_W_DIMSZ_INDEX          1U

/**
 * Defines the H dimension index for NHWC.
 * `tensormetadata.dimSizes[NVM_TENSOR_NHWC_H_DIMSZ_INDEX] = H`, is the dimension
 * value along H dimension for NHWC tensor.
 * \ingroup nhwc_dim
 */
#define NVM_TENSOR_NHWC_H_DIMSZ_INDEX          2U

/**
 * Defines the N dimension index for NHWC.
 * `tensormetadata.dimSizes[NVM_TENSOR_NHWC_N_DIMSZ_INDEX] = N`, is the dimension
 * value along N dimension for NHWC tensor.
 * \ingroup nhwc_dim
 */
#define NVM_TENSOR_NHWC_N_DIMSZ_INDEX          3U

/** @} <!-- ends nhwc_dim group --> */

/**
 * \defgroup nchw_dim Dimension Indices - NCHW
 *
 * \ingroup stride_dim
 * @{
 */

/**
 * Defines the W dimension index for NCHW.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCHW_W_DIMSZ_INDEX] = W`, is the dimension
 * value along W dimension for NCHW tensor.
 * \ingroup nchw_dim
 */
#define NVM_TENSOR_NCHW_W_DIMSZ_INDEX          0U

/**
 * Defines the H dimension index for NCHW.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCHW_H_DIMSZ_INDEX] = H`, is the dimension
 * value along H dimension for NHWC tensor.
 * \ingroup nchw_dim
 */
#define NVM_TENSOR_NCHW_H_DIMSZ_INDEX          1U

/**
 * Defines the C dimension index for NCHW.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCHW_C_DIMSZ_INDEX] = C`, is the dimension
 * value along C dimension for NHWC tensor.
 * \ingroup nchw_dim
 */
#define NVM_TENSOR_NCHW_C_DIMSZ_INDEX          2U

/**
 * Defines the N dimension index for NCHW.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCHW_N_DIMSZ_INDEX] = N`, is the dimension
 * value along N dimension for NHWC tensor.
 * \ingroup nchw_dim
 */
#define NVM_TENSOR_NCHW_N_DIMSZ_INDEX          3U

/** @} <!-- ends nchw_dim group --> */

/**
 * \defgroup ncxhwx_dim Dimension Indices - NCxHWx
 *
 * \ingroup stride_dim
 * @{
 */

/**
 * Defines the x dimension index for NCxHWx.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCxHWx_x_DIMSZ_INDEX] = x`, is the dimension
 * value along the x dimension for the NCxHWx tensor.
 * \ingroup ncxhwx_dim
 */
#define NVM_TENSOR_NCxHWx_x_DIMSZ_INDEX        0U

/**
 * Defines the W dimension index for NCxHWx.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCxHWx_W_DIMSZ_INDEX] = W`, is the dimension
 * value along the W dimension for the NCxHWx tensor.
 * \ingroup ncxhwx_dim
 */
#define NVM_TENSOR_NCxHWx_W_DIMSZ_INDEX        1U

/**
 * Defines the H dimension index for NCxHWx.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCxHWx_H_DIMSZ_INDEX] = H`, is the dimension
 * value along the H dimension for the NCxHWx tensor.
 * \ingroup ncxhwx_dim
 */
#define NVM_TENSOR_NCxHWx_H_DIMSZ_INDEX        2U

/**
 * Defines the Cx dimension index for NCxHWx.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCxHWx_Cx_DIMSZ_INDEX] = Cx`, is the dimension
 * value along the Cx dimension for the NCxHWx tensor.
 * \ingroup ncxhwx_dim
 */
#define NVM_TENSOR_NCxHWx_Cx_DIMSZ_INDEX       3U

/**
 * Defines the N dimension index for NCxHWx.
 * `tensormetadata.dimSizes[NVM_TENSOR_NCxHWx_N_DIMSZ_INDEX] = N`, is the dimension
 * value along the N dimension for the NCxHWx tensor.
 * \ingroup ncxhwx_dim
 */
#define NVM_TENSOR_NCxHWx_N_DIMSZ_INDEX        4U

/** @} <!-- Ends ncxhwx_dim group --> */

/** @} <!-- Ends stride_dim --> */
#endif /* (NV_IS_SAFETY == 0) */

/**
 * Holds the tensor metadata.
 */
typedef struct {
    /*!
     *  Holds the number of valid elements in dimSizes[] and dimstrides[].
     *  For example, `dimsNum = 4` for NHWC and NCHW,
     *  and `dimsNum = 5` for NCxHWx.
     */
    uint32_t dimsNum;
    /** \brief Holds the size of each dimension.
     *
     * The dimSizes ordering is as below:
     *  - For NVM_TENSOR_ATTR_DIMENSION_ORDER_NHWC: N -> 0, H -> 1, W -> 2, C -> 3
     *  - For NVM_TENSOR_ATTR_DIMENSION_ORDER_NCHW: N -> 0, C -> 1, H -> 2, W -> 3
     *  - For NVM_TENSOR_ATTR_DIMENSION_ORDER_NCxHWx: N -> 0, C -> 1, H -> 2, W -> 3, X -> 4
     *
    */
    uint32_t dimSizes[NVMEDIA_TENSOR_MAX_DIMENSIONS];
    /** \brief Holds strides(in bytes) for each dimension present in the tensor.
     *  The dimstrides ordering is same as that of dimSizes documented above.
    */
    uint32_t dimstrides[NVMEDIA_TENSOR_MAX_DIMENSIONS];
    /** \brief Holds the order of the dimensions. */
    uint32_t dimsOrder;
    /** \brief Holds the bitsPerElement such as NVM_TENSOR_ATTR_BITS_PER_ELEMENT_8/16 */
    uint32_t bitsPerElement;
    /** \brief Holds the tensor datatype, such as NVM_TENSOR_ATTR_DATA_TYPE_INT/FLOAT */
    uint32_t dataType;
    /** \brief Holds the tensor attribute N. */
    uint32_t attrib4D_N;
    /** \brief Holds the tensor attribute C. */
    uint32_t attrib4D_C;
    /** \brief Holds the tensor attribute H. */
    uint32_t attrib4D_H;
    /** \brief Holds the tensor attribute W. */
    uint32_t attrib4D_W;
    /** \brief Holds the tensor attribute X. */
    uint32_t attrib4D_X;
} NvMediaTensorMetaData;

#ifdef __cplusplus
}     /* extern "C" */
#endif
/** @} <!-- Ends nvmedia_tensor_metadata_api --> */
#endif /* NVM_TENSORMETADATA_H */
