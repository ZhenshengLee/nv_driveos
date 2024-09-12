/*
 * Header file for NvSciBuf APIs
 *
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */
/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSciBuf </b>
 *
 * Allows applications to allocate and exchange buffers in memory.
 */
#ifndef INCLUDED_NVSCIBUF_H
#define INCLUDED_NVSCIBUF_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "nvscierror.h"
#include <nvsciipc.h>

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup nvscibuf_blanket_statements NvSciBuf blanket statements.
 * Generic statements applicable for NvSciBuf interfaces.
 * @ingroup nvsci_buf
 * @{
 */

/**
 * \page nvscibuf_page_blanket_statements NvSciBuf blanket statements
 * \section nvscibuf_in_params Input parameters
 * - NvSciBufModule passed as input parameter to an API is valid input if it is
 * returned from a successful call to NvSciBufModuleOpen() and has not yet been
 * deallocated using NvSciBufModuleClose().
 * - NvSciIpcEndpoint passed as input parameter to an API is valid if it is
 * obtained from successful call to NvSciIpcOpenEndpoint() and has not yet been
 * freed using NvSciIpcCloseEndpoint().
 * - NvSciBufObj is valid if it is obtained from successful call to
 * NvSciBufObjAlloc() or if it is obtained from successful call to
 * NvSciBufAttrListReconcileAndObjAlloc() or if it is obtained from
 * successful call to NvSciBufObjIpcImport() and has not been deallocated
 * using NvSciBufObjFree().
 * - Unreconciled NvSciBufAttrList is valid if it is obtained from successful
 * call to NvSciBufAttrListCreate() or if it is obtained from successful call to
 * NvSciBufAttrListClone() where input to NvSciBufAttrListClone() is valid
 * unreconciled NvSciBufAttrList or if it is obtained from successful call to
 * NvSciBufAttrListIpcImportUnreconciled() and has not been deallocated using
 * NvSciBufAttrListFree().
 * - Reconciled NvSciBufAttrList is valid if it is obtained from successful call
 * to NvSciBufAttrListReconcile() or if it is obtained from successful call to
 * NvSciBufAttrListClone() where input to NvSciBufAttrListClone() is valid
 * reconciled NvSciBufAttrList or if it is obtained from successful call to
 * NvSciBufAttrListIpcImportReconciled() and has not been deallocated using
 * NvSciBufAttrListFree().
 *
 * \section nvscibuf_out_params Output parameters
 * - In general, output parameters are passed by reference through pointers.
 * Also, since a null pointer cannot be used to convey an output parameter, API
 * functions typically return an error code if a null pointer is supplied for a
 * required output parameter unless otherwise stated explicitly. Output
 * parameter is valid only if error code returned by an API is
 * NvSciError_Success unless otherwise stated explicitly.
 *
 * \section nvscibuf_concurrency Concurrency
 * - Every individual function can be called concurrently with itself without
 * any side-effects unless otherwise stated explicitly in the interface
 * specifications.
 * - The conditions for combinations of functions that cannot be called
 * concurrently or calling them concurrently leads to side effects are
 * explicitly stated in the interface specifications.
 */

/**
 * @}
 */

/**
 * @defgroup nvsci_buf Buffer Allocation APIs
 *
 * The NvSciBuf library contains the APIs for applications to allocate
 * and exchange buffers in memory.
 *
 * @ingroup nvsci_group_stream
 * @{
 */
/**
 * @defgroup nvscibuf_datatype NvSciBuf Datatype Definitions
 * Contains a list of all NvSciBuf datatypes.
 * @{
 */

/**
 * @brief Enum definitions of NvSciBuf datatypes.
 *
 * @implements{17824095}
 */
typedef enum {
    /** Reserved for General keys.
     * Shouldn't be used as valid value for  NvSciBufGeneralAttrKey_Types.
     */
    NvSciBufType_General = 0U,
    NvSciBufType_RawBuffer = 1U,
    NvSciBufType_Image = 2U,
    NvSciBufType_Tensor = 3U,
#if (NV_IS_SAFETY == 0)
    NvSciBufType_Array = 4U,
    NvSciBufType_Pyramid = 5U,
#endif
    NvSciBufType_MaxValid = 6U,
    NvSciBufType_UpperBound = 6U,
} NvSciBufType;

/**
 * @}
 */

/**
 * @defgroup nvscibuf_constants NvSciBuf Global Constants
 * Definitions of all NvSciBuf Global Constants/Macros
 *
 * @{
 */
/**
 * @brief NvSciBuf API Major version number.
 *
 * @implements{18840105}
 */
static const uint32_t NvSciBufMajorVersion = 2U;

/**
 * @brief NvSciBuf API Minor version number.
 *
 * @implements{18840108}
 */
static const uint32_t NvSciBufMinorVersion = 0U;

#if defined(__cplusplus)

/**
 * @brief Maximum number of dimensions supported by tensor datatype.
 */
static const int NV_SCI_BUF_TENSOR_MAX_DIMS = 8;

/**
 * @brief Maximum number of planes supported by image datatype.
 */
static const int NV_SCI_BUF_IMAGE_MAX_PLANES = 3;

#if (NV_IS_SAFETY == 0)
/**
 * @brief Maximum number of levels supported by pyramid datatype.
 */
static const int NV_SCI_BUF_PYRAMID_MAX_LEVELS = 10;
#endif

/**
 * @brief Indicates the size of export descriptor.
 */
static const int NVSCIBUF_EXPORT_DESC_SIZE = 32;

/**
 * @brief Indicates number of bits used for defining an attribute key.
 * Note: Maximum 16K attribute Keys per datatype.
 */
static const int NV_SCI_BUF_ATTRKEY_BIT_COUNT = 16;

/**
 * @brief Indicates number of bits used for defining an datatype of a key.
 * Note: Maximum 1K datatypes.
 */
static const int NV_SCI_BUF_DATATYPE_BIT_COUNT = 10;

/**
 * @brief Indicates the attribute key is a public key type.
 */
static const int NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC = 0;

/*
 * @brief Global constant to specify the start-bit of attribute Keytype.
 */
static const int NV_SCI_BUF_KEYTYPE_BIT_START =
        (NV_SCI_BUF_DATATYPE_BIT_COUNT + NV_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates starting value of General attribute keys.
 */
static const int NV_SCI_BUF_GENERAL_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_General << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of Raw-buffer Datatype keys.
 */
static const int NV_SCI_BUF_RAW_BUF_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_RawBuffer << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of Image Datatype keys.
 */
static const int NV_SCI_BUF_IMAGE_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_Image << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

#if (NV_IS_SAFETY == 0)
/**
 * @brief Indicates the start of ImagePyramid Datatype keys.
 */
static const int NV_SCI_BUF_PYRAMID_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_Pyramid << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

/**
 * @brief Indicates the start of NvSciBuf Array Datatype keys.
 */
static const int NV_SCI_BUF_ARRAY_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_Array << NV_SCI_BUF_ATTRKEY_BIT_COUNT);
#endif

/**
 * @brief Indicates the start of Tensor Datatype keys.
 */
static const int NV_SCI_BUF_TENSOR_ATTR_KEY_START =
           (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) |
           (NvSciBufType_Tensor << NV_SCI_BUF_ATTRKEY_BIT_COUNT);

#else

/**
 * @brief Maximum number of dimensions supported by NvSciBufType_Tensor.
 *
 * @implements{18840096}
 */
#define NV_SCI_BUF_TENSOR_MAX_DIMS  8u

/**
 * @brief Maximum number of planes supported by NvSciBufType_Image.
 *
 * @implements{18840099}
 */
#define NV_SCI_BUF_IMAGE_MAX_PLANES 3u

#if (NV_IS_SAFETY == 0)
/**
 * @brief Maximum number of levels supported by NvSciBufType_Pyramid.
 */
#define NV_SCI_BUF_PYRAMID_MAX_LEVELS 10u
#endif

/**
 * @brief Indicates the size of export descriptor.
 */
#define NVSCIBUF_EXPORT_DESC_SIZE   32u

/**
 * @brief Global constant to indicate number of bits used for
 * defining an attribute key. Note: Maximum 16K attribute keys
 * per NvSciBufType.
 */
#define NV_SCI_BUF_ATTRKEY_BIT_COUNT  16u

/**
 * @brief Global constant to indicate number of bits used for
 * defining NvSciBufType of an attribute key. Note: Maximum 1K
 * NvSciBufType(s).
 */
#define NV_SCI_BUF_DATATYPE_BIT_COUNT  10u

/**
 * @brief Global constant to indicate the attribute key type is public.
 */
#define NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC 0u

/**
 * @brief Global constant to specify the start-bit of attribute key type.
 */
#define NV_SCI_BUF_KEYTYPE_BIT_START \
        (NV_SCI_BUF_DATATYPE_BIT_COUNT + NV_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates starting value of NvSciBufAttrKey for NvSciBufType_General.
 */
#define NV_SCI_BUF_GENERAL_ATTR_KEY_START \
        (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
        (NvSciBufType_General << NV_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates starting value of NvSciBufAttrKey for NvSciBufType_RawBuffer.
 */
#define NV_SCI_BUF_RAW_BUF_ATTR_KEY_START \
          (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
          (NvSciBufType_RawBuffer << NV_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of NvSciBufAttrKey for NvSciBufType_Image.
 */
#define NV_SCI_BUF_IMAGE_ATTR_KEY_START \
          (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
          (NvSciBufType_Image << NV_SCI_BUF_ATTRKEY_BIT_COUNT)

#if (NV_IS_SAFETY == 0)
/**
 * @brief Indicates the starting value of NvSciBufAttrKey for NvSciBufType_Pyramid.
 */
#define NV_SCI_BUF_PYRAMID_ATTR_KEY_START \
          (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
          (NvSciBufType_Pyramid << NV_SCI_BUF_ATTRKEY_BIT_COUNT)

/**
 * @brief Indicates the starting value of NvSciBufAttrKey for NvSciBufType_Array.
 */
#define NV_SCI_BUF_ARRAY_ATTR_KEY_START \
          (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
          (NvSciBufType_Array << NV_SCI_BUF_ATTRKEY_BIT_COUNT)
#endif

/**
 * @brief Indicates the starting value of NvSciBufAttrKey for NvSciBufType_Tensor.
 */
#define NV_SCI_BUF_TENSOR_ATTR_KEY_START \
          (NV_SCI_BUF_ATTR_KEY_TYPE_PUBLIC << NV_SCI_BUF_KEYTYPE_BIT_START) | \
          (NvSciBufType_Tensor << NV_SCI_BUF_ATTRKEY_BIT_COUNT)

#endif

/**
 * @}
 */

/**
 * @defgroup nvscibuf_attr_key NvSciBuf Enumerations for Attribute Keys
 * List of all NvSciBuf enumerations for attribute keys.
 * @{
 */

/**
 * @brief Describes the NvSciBuf public attribute keys holding corresponding
 * values specifying buffer constraints. Input attribute keys specify desired
 * buffer constraints and can be set/retrieved from unreconciled
 * NvSciBufAttrList using NvSciBufAttrListGetAttrs()/NvSciBufAttrListSetAttrs()
 * respectively. Output attribute keys specify actual buffer constraints
 * computed by NvSciBuf if reconciliation succeeds. Output attributes can be
 * retrieved from reconciled NvSciBufAttrList using NvSciBufAttrListGetAttrs().
 *
 * @implements{17824098}
 */
typedef enum {
    /**
     * Specifies the lower bound value to check for a valid NvSciBuf attribute
     * key type.
     */
    NvSciBufAttrKey_LowerBound =         NV_SCI_BUF_GENERAL_ATTR_KEY_START,

    /** An array of all types that the buffer is expected to have. For each type
     * the buffer has, the associated attributes are valid. In order to set
     * @a NvSciBufAttrKeys corresponding to the NvSciBufType, NvSciBufType must
     * be set first using this key.
     * NOTE: A single buffer may have multiple types. For example, a buffer may
     * simultaneously be a NvSciBufType_Image (for integration with NvMedia), a
     * NvSciBufType_Tensor (for integration with TensorRT or NvMedia), and a
     * NvSciBufType_RawBuffer (for integration with CUDA kernels that will
     * directly access it).
     *
     * During reconciliation, if all the NvSciBufTypes
     * specified by all the unreconciled NvSciBufAttrLists are same, this
     * key outputs the specified NvSciBufType. If all NvSciBufTypes are
     * not same, reconciliation succeeds only if the set of NvSciBufTypes
     * contains NvSciBufType_Image and NvSciBufType_Tensor only otherwise
     * reconciliation fails.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if aggregate
     * of all the values of all the input unreconciled NvSciBufAttrList(s) that
     * set this attribute are present in the reconciled NvSciBufAttrList.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufType[]
     *
     * valid input value: All values defined by NvSciBufType for safety build
     * such that NvSciBufType_General < value < NvSciBufType_MaxValid
     */
    NvSciBufGeneralAttrKey_Types,

    /** Specifies if CPU access is required for the buffer. If this attribute is
     * set to @c true, then the CPU will be able to obtain a pointer to the
     * buffer from NvSciBufObjGetConstCpuPtr() if at least read permissions are
     * granted or from NvSciBufObjGetCpuPtr() if read/write permissions are
     * granted.
     *
     * During reconciliation, reconciler sets value of this key to true in the
     * reconciled NvSciBufAttrList if any of the unreconciled
     * NvSciBufAttrList(s) involved in reconciliation that is owned by the
     * reconciler has this key set to true, otherwise it is set to false in
     * reconciled NvSciBufAttrList.
     *
     * When importing the reconciled NvSciBufAttrList, for every peer owning the
     * unreconciled NvSciBufAttrList(s) involved in reconciliation, if any of
     * the unreconciled NvSciBufAttrList(s) owned by the peer set the key to
     * true then value of this key is true in the reconciled NvSciBufAttrList
     * imported by the peer otherwise its false.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds if value of this
     * attribute in the reconciled NvSciBufAttrList is true provided any of the
     * input unreconciled NvSciBufAttrList(s) owned by the peer set it to
     * true OR if value of this attribute in the reconciled NvSciBufAttrList is
     * false provided all of the input unreconciled NvSciBufAttrList(s) owned by
     * the peer set it to false.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c bool
     */
    NvSciBufGeneralAttrKey_NeedCpuAccess,

    /** Specifies buffer access permissions.
     * If reconciliation succeeds, granted buffer permissions are reflected in
     * NvSciBufGeneralAttrKey_ActualPerm. If
     * NvSciBufGeneralAttrKey_NeedCpuAccess is true and write permission
     * are granted, then NvSciBufObjGetCpuPtr() can be used to obtain a
     * non-const pointer to the buffer.
     * NOTE: Whether this key is present in reconciled attribute lists is
     * unspecified, as is its value if it is present.
     *
     * Type: Input attribute
     *
     * Datatype: @ref NvSciBufAttrValAccessPerm
     *
     * valid input value: NvSciBufAccessPerm_Readonly or
     * NvSciBufAccessPerm_ReadWrite
     */
    NvSciBufGeneralAttrKey_RequiredPerm,

    /** Specifies whether to enable/disable CPU caching.
     * If set to @c true:
     *
     * The CPU must perform write-back caching of the buffer to the greatest
     * extent possible considering all the CPUs that are sharing the buffer.
     *  Coherency is guaranteed with:
     *         - Other CPU accessors.
     *         - All I/O-Coherent accessors that do not have CPU-invisible
     *           caches.
     *
     * If set to @c false:
     *
     * The CPU must not access the caches at all on read or write accesses
     * to the buffer from applications.
     *  Coherency is guaranteed with:
     *         - Other CPU accessors.
     *         - All I/O accessors (whether I/O-coherent or not) that do not
     *              have CPU-invisible caches.
     *
     * During reconciliation, this key is set to true in reconciled
     * NvSciBufAttrList if any of the unreconciled NvSciBufAttrList owned by any
     * peer set it to true.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds if value of this
     * attribute in the reconciled NvSciBufAttrList is true provided any of the
     * input unreconciled NvSciBufAttrList(s) set it to true OR if value of this
     * attribute in the reconciled NvSciBufAttrList is false provided all of the
     * input unreconciled NvSciBufAttrList(s) set it to false.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c bool
     */
    NvSciBufGeneralAttrKey_EnableCpuCache,

    /** GpuIDs of the GPUs in the system that will access the buffer.
     * In a multi GPU System, if multiple GPUs are supposed to access
     * the buffer, then provide the GPU IDs of all the GPUs that
     * need to access the buffer. The GPU which is not specified in the
     * list of GPUIDs may not be able to access the buffer.
     *
     * During reconciliation, the value of this key in reconciled
     * NvSciBufAttrList is equivalent to the aggregate of all the values
     * specified by all the unreconciled NvSciBufAttrLists involved in
     * reconciliation.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if aggregate
     * of all the values of all the input unreconciled NvSciBufAttrList(s) that
     * set it are present in the reconciled NvSciBufAttrList.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciRmGpuId[]
     *
     * valid input value: Valid NvSciRmGpuId of the GPU(s) present in the
     * system.
     */
    NvSciBufGeneralAttrKey_GpuId,

    /** Indicates whether the CPU is required to flush before reads and
     * after writes. This can be accomplished using
     * NvSciBufObjFlushCpuCacheRange(), or (if the application prefers) with
     * OS-specific flushing functions. It is set to true in reconciled
     * NvSciBufAttrList if both NvSciBufGeneralAttrKey_EnableCpuCache and
     * NvSciBufGeneralAttrKey_NeedCpuAccess are requested by setting them
     * to true in any of the unreconciled NvSciBufAttrList(s) from which
     * reconciled NvSciBufAttrList is obtained.
     *
     * Type: Output attribute
     *
     * Datatype: @c bool
     */
    NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency,

    /** Specifies the buffer access permissions to the NvSciBufObj.
     * This key is only valid in reconciled NvSciBufAttrList
     * and undefined in unreconciled NvSciBufAttrList.
     *
     * During reconciliation, this key is set to the maximum value of the
     * requested permission set in NvSciBufGeneralAttrKey_RequiredPerm of all
     * the unreconciled NvSciBufAttrLists that are owned by the reconciler and
     * are involved in the reconciliation.
     *
     * If NvSciBufObj is obtained by calling NvSciBufObjAlloc(),
     * NvSciBufGeneralAttrKey_ActualPerm is set to NvSciBufAccessPerm_ReadWrite
     * in the reconciled NvSciBufAttrList corresponding to it since allocated
     * NvSciBufObj gets read-write permissions by default.
     *
     * For any peer importing the reconciled NvSciBufAttrList, this key is set
     * to maximum value of the requested permission set in
     * NvSciBufGeneralAttrKey_RequiredPerm of all the unreconciled
     * NvSciBufAttrLists that were exported by the peer for reconciliation.
     * The key is set by the reconciler when exporting the reconciled
     * NvSciBufAttrList.
     *
     * For any peer importing the NvSciBufObj, this key is set in the reconciled
     * NvSciBufAttrList to the permissions associated with
     * NvSciBufObjIpcExportDescriptor.
     *
     * Type: Output attribute
     *
     * Datatype: NvSciBufAttrValAccessPerm
     */
    NvSciBufGeneralAttrKey_ActualPerm,

    /** GPU ID of dGPU from which vidmem allocation should come when multiple
     * GPUs are sharing buffer. This key should be empty if multiple GPUs
     * access shared buffer from sysmem.
     *
     * Type: Input/Output attribute
     *
     * Datatype: NvSciRmGpuId
     *
     * valid input value: Valid NvSciRmGpuId of the dGPU present in the system.
     */
    NvSciBufGeneralAttrKey_VidMem_GpuId,

    /** Specifies the size of the buffer to be allocated for
     * NvSciBufType_RawBuffer. Input size specified in unreconciled
     * NvSciBufAttrList should be greater than 0.
     *
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufRawBufferAttrKey_Size   =  NV_SCI_BUF_RAW_BUF_ATTR_KEY_START,

    /** Specifies the alignment requirement of NvSciBufType_RawBuffer. Input
     * alignment should be power of 2. If more than one unreconciled
     * NvSciBufAttrLists specify this input attribute, value in the reconciled
     * NvSciBufAttrList corresponds to maximum of the values specified in all of
     * the unreconciled NvSciBufAttrLists.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList is greater than or equal to
     * the value of the same attribute in every input unreconciled
     * NvSciBufAttrList that set it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t
     *
     * valid input value: value is power of 2.
     */
    NvSciBufRawBufferAttrKey_Align,

    /** Specifies the layout of NvSciBufType_Image: Block-linear or
     * Pitch-linear. If more than one unreconciled NvSciBufAttrLists specify
     * this input attribute, reconciliation is successful if all the input
     * attributes of this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValImageLayoutType
     *
     * valid input value: Any value defined by NvSciBufAttrValImageLayoutType
     * enum.
     */
    NvSciBufImageAttrKey_Layout   =    NV_SCI_BUF_IMAGE_ATTR_KEY_START,

    /** Specifies the top padding for the NvSciBufType_Image. If more than one
     * unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_TopPadding,

    /** Specifies the bottom padding for the NvSciBufType_Image. If more than
     * one unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: uint64_t[]
     */
    NvSciBufImageAttrKey_BottomPadding,

    /** Specifies the left padding for the NvSciBufType_Image. If more than one
     * unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_LeftPadding,

    /** Specifies the right padding for the NvSciBufType_Image. If more than one
     * unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_RightPadding,

    /** Specifies the VPR flag for the NvSciBufType_Image.
     *
     * During reconciliation, this key is set to true in reconciled
     * NvSciBufAttrList if any of the unreconciled NvSciBufAttrList owned by any
     * peer set it to true.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds if value of this
     * attribute in the reconciled NvSciBufAttrList is true provided any of the
     * input unreconciled NvSciBufAttrList(s) set it to true OR if value of this
     * attribute in the reconciled NvSciBufAttrList is false provided all of the
     * input unreconciled NvSciBufAttrList(s) set it to false.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c bool
     */
    NvSciBufImageAttrKey_VprFlag,

    /** Output size of the NvSciBufType_Image after successful reconciliation.
     * The output size for this key is computed by aggregating size of all the
     * planes in the output key NvSciBufImageAttrKey_PlaneAlignedSize.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufImageAttrKey_Size,

    /** Output alignment of the NvSciBufType_Image after successful
     * reconciliation.
     * The output value of this key is same as alignment value of the first
     * plane in the key NvSciBufImageAttrKey_PlaneBaseAddrAlign after
     * reconciliation.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufImageAttrKey_Alignment,

    /** Specifies the number of planes for NvSciBufType_Image. If more than one
     * unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match. If NvSciBufType_Image and NvSciBufType_Tensor are involved in
     * reconciliation and if this attribute is set in any of unreconciled
     * NvSciBufAttrList(s) to be reconciled, the value of this attribute should
     * be 1.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it. Additionally, if one or more input unreconciled NvSciBufAttrList(s)
     * specify NvSciBufType_Image and NvSciBufType_Tensor as NvSciBufTypes in
     * their NvSciBufAttrList(s), validation succeeds only if value of this
     * attribute in reconciled NvSciBufAttrList is 1.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t
     *
     * valid input value: 1 <= value <= NV_SCI_BUF_IMAGE_MAX_PLANES
     */
    NvSciBufImageAttrKey_PlaneCount,

    /** Specifies the NvSciBufAttrValColorFmt of the NvSciBufType_Image plane.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValColorFmt[]
     *
     * valid input value: NvSciColor_LowerBound < value < NvSciColor_UpperBound
     */
    NvSciBufImageAttrKey_PlaneColorFormat,

    /** Specifies a set of plane color standards. If more than
     * one unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValColorStd[]
     *
     * valid input value: Any value defined by NvSciBufAttrValColorStd enum.
     */
    NvSciBufImageAttrKey_PlaneColorStd,

    /** Specifies the NvSciBufType_Image plane base address alignment for every
     * plane in terms of an array. Input alignment must be power of 2.
     * If more than one unreconciled NvSciBufAttrLists specify this attribute,
     * reconciled NvSciBufAttrList has maximum alignment value per array index
     * of the values specified in unreconciled NvSciBufAttrLists for the same
     * array index. On top of that, for all the HW engines for which buffer is
     * being allocated, if the maximum start address alignment constraint of all
     * the engines taken together is greater than the reconciled alignment value
     * at any index, it is replaced with start address alignment value. In other
     * words,
     * reconciled alignment per array index =
     * MAX(MAX(alignments in unreconciled list at the same index),
     * MAX(start address alignment constraint of all the engines))
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList is greater than or equal to
     * the value of the same attribute in every input unreconciled
     * NvSciBufAttrList that set it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t[]
     *
     * valid input value: value is power of 2.
     */
    NvSciBufImageAttrKey_PlaneBaseAddrAlign,

    /** Specifies the NvSciBufType_Image plane width in pixels. If more than
     * one unreconciled NvSciBufAttrLists specify this input attribute,
     * reconciliation is successful if all the input attributes of this
     * type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t[]
     */
    NvSciBufImageAttrKey_PlaneWidth,

    /** Specifies the NvSciBufType_Image plane height in number of lines. If
     * more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t[]
     */
    NvSciBufImageAttrKey_PlaneHeight,

    /** Specifies the NvSciBufType_Image scan type: Progressive or Interlaced.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes
     * of this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValImageScanType
     *
     * valid input value: Any value defined by NvSciBufAttrValImageScanType
     * enum.
     */
    NvSciBufImageAttrKey_PlaneScanType = 0x2000e,
    NvSciBufImageAttrKey_ScanType = NvSciBufImageAttrKey_PlaneScanType,

    /** Outputs number of bits per pixel corresponding to the
     * NvSciBufAttrValColorFmt for each plane specified in
     * NvSciBufImageAttrKey_PlaneColorFormat.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint32_t[]
     */
    NvSciBufImageAttrKey_PlaneBitsPerPixel,

    /** Indicates the starting offset of the NvSciBufType_Image plane from the
     * first plane.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_PlaneOffset,

    /** Outputs the NvSciBufAttrValDataType of each plane based on the
     * NvSciBufAttrValColorFmt provided in
     * NvSciBufImageAttrKey_PlaneColorFormat for every plane.
     *
     * Type: Output attribute
     *
     * Datatype: @ref NvSciBufAttrValDataType[]
     */
    NvSciBufImageAttrKey_PlaneDatatype,

    /** Outputs number of channels per plane.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint8_t[]
     */
    NvSciBufImageAttrKey_PlaneChannelCount,

    /** Indicates the offset of the start of the second field, 0 for progressive
     * valid for interlaced.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_PlaneSecondFieldOffset,

    /** Outputs the pitch (aka width in bytes) for every plane. The pitch is
     * aligned to the maximum of the pitch alignment constraint value of all the
     * HW engines that are going to operate on the buffer.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint32_t[]
     */
    NvSciBufImageAttrKey_PlanePitch,

    /** Outputs the aligned height of evey plane in terms of number of lines.
     * This height is calculated by aligning value for every plane provided in
     * NvSciBufImageAttrKey_PlaneHeight with maximum of the height alignment
     * constraints of all the engines that are going to operate on the buffer.
     * The output value is always power of 2.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint32_t[]
     */
    NvSciBufImageAttrKey_PlaneAlignedHeight,

    /** Indicates the aligned size of every plane. The size is calculated from
     * the value of NvSciBufImageAttrKey_PlanePitch and
     * NvSciBufImageAttrKey_PlaneAlignedHeight.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufImageAttrKey_PlaneAlignedSize,

    /** Attribute to specify number of NvSciBufType_Image(s) for which buffer
     * should be allocated. If more than one unreconciled NvSciBufAttrLists
     * specify this input attribute, reconciliation is successful if all the
     * input attributes of this type match. NvSciBuf supports allocating buffer
     * for single image only and thus, this attribute should be set to 1.
     * A single buffer cannot be allocated for multiple images.
     * Allocating 'N' buffers corresponding to 'N' images is allowed
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     * Datatype: @c uint64_t
     *
     * valid input value: 1
     */
    NvSciBufImageAttrKey_ImageCount,

    /** Specifies the tensor data type.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValDataType
     *
     * valid input value: NvSciDataType_Int4 <= value <= NvSciDataType_Float32
     */
    NvSciBufTensorAttrKey_DataType  =  NV_SCI_BUF_TENSOR_ATTR_KEY_START,

    /** Specifies the number of tensor dimensions. A maximum of 8 dimensions are
     * allowed. If more than one unreconciled NvSciBufAttrLists specify this
     * input attribute, reconciliation is successful if all the input attributes
     * of this type match.
     * If NvSciBufType_Image and NvSciBufType_Tensor NvSciBufTypes are used
     * in reconciliation, reconciliation succeeds only if this key is set
     * to 4, since NvSciBuf only supports reconciliation of NvSciBufType_Tensor
     * of NHWC type with NvSciBufType_Image.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it. Additionally, if one or more input unreconciled NvSciBufAttrList(s)
     * specify NvSciBufType_Image and NvSciBufType_Tensor as NvSciBufTypes in
     * their NvSciBufAttrList(s), validation succeeds only if value of this
     * attribute in reconciled NvSciBufAttrList is 4.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t
     *
     * valid input value: 1 <= value <= NV_SCI_BUF_TENSOR_MAX_DIMS
     */
    NvSciBufTensorAttrKey_NumDims,

    /** Specifies the size of each tensor dimension.
     * This attribute takes size value in terms of an array.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match. Number of elements in value array of this attribute
     * should not be less than value specified by NvSciBufTensorAttrKey_NumDims
     * attribute.
     * @note Array indices are not tied to the semantics of the
     * dimension if NvSciBufType_Tensor is the only NvSciBufType involved
     * in reconciliation. If NvSciBufType_Tensor and NvSciBufType_Image
     * are involved in reconciliation, NvSciBuf only supports
     * reconciliation of NvSciBufType_Image with NHWC NvSciBufType_Tensor
     * where N=1 and thus reconciliation succeeds only if value of
     * dimension 0 is 1 and it matches with value of
     * NvSciBufImageAttrKey_ImageCount, value of dimension 1 matches with value
     * of NvSciBufImageAttrKey_PlaneHeight, value of dimension 2 matches with
     * value of NvSciBufImageAttrKey_PlaneWidth and dimension 3 specifies
     * channel count for NvSciBufAttrValColorFmt specified in
     * NvSciBufTensorAttrKey_PixelFormat key
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it. Additionally, if one or more input unreconciled NvSciBufAttrList(s)
     * specify NvSciBufType_Image and NvSciBufType_Tensor as NvSciBufTypes in
     * their NvSciBufAttrList(s), validation succeeds only if value of dimension
     * 0 in reconciled NvSciBufAttrList is 1 and it matches with value of
     * NvSciBufImageAttrKey_ImageCount in the same NvSciBufAttrList, value of
     * dimension 1 matches with value of NvSciBufImageAttrKey_PlaneHeight in the
     * same reconciled NvSciBufAttrList, value of dimension 2 matches with value
     * of NvSciBufImageAttrKey_PlaneWidth in the same reconciled
     * NvSciBufAttrList and dimension 3 matches with channel count for
     * NvSciBufAttrValColorFmt specified in NvSciBufTensorAttrKey_PixelFormat
     * in the same reconciled NvSciBufAttrList.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufTensorAttrKey_SizePerDim,

    /** Specifies the alignment constraints per tensor dimension.
     * Number of elements in value array of this attribute should not be less
     * than value specified by NvSciBufTensorAttrKey_NumDims attribute. Value of
     * every element in the value array should be power of two.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value in
     * every index of value array of this attribute in reconciled
     * NvSciBufAttrList is greater than or equal to the value in the same index
     * of the same attribute in every input unreconciled NvSciBufAttrList that
     * set it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t[]
     *
     * valid input value: value is power of 2.
     */
    NvSciBufTensorAttrKey_AlignmentPerDim,

    /** Returns the stride value (in bytes) for each tensor dimension.
     * @note The returned array contains stride values in decreasing order.
     * In other words, the index @em 0 of the array will have the largest
     * stride while [@em number-of-dims - 1] index will have the smallest stride.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufTensorAttrKey_StridesPerDim,

    /** Attribute providing pixel format of the tensor. This key needs to be
     * set only if NvSciBufType_Image and NvSciBufType_Tensor are involved in
     * reconciliation.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match. Additionally, reconciliation succeeds only if value of
     * this attribute matches with the value of
     * NvSciBufImageAttrKey_PlaneColorFormat in all the input unreconciled
     * NvSciBufAttrList(s) that have set it.
     * Image/Tensor reconciliation only supports NvSciColor_A8B8G8R8 and
     * NvSciColor_Float_A16B16G16R16 color formats as of now.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList matches with the value of
     * the same attribute in every input unreconciled NvSciBufAttrList that set
     * it. Additionally, reconciliation succeeds only if value of this attribute
     * matches with NvSciBufImageAttrKey_PlaneColorFormat attribute in the same
     * reconciled NvSciBufAttrList and value is set to either
     * NvSciColor_A8B8G8R8 or NvSciColor_Float_A16B16G16R16.
     *
     * Type: Input/Output attribute
     *
     * Datattype: @ref NvSciBufAttrValColorFmt
     *
     * valid input value: NvSciColor_LowerBound < value < NvSciColor_UpperBound
     */
    NvSciBufTensorAttrKey_PixelFormat,

    /** Attribute providing base address alignment requirements for tensor.
     * Input value provided for this attribute must be power of two. Output
     * value of this attribute is always power of two.
     *
     * During validation of reconciled NvSciBufAttrList against input
     * unreconciled NvSciBufAttrList(s), validation succeeds only if value of
     * this attribute in reconciled NvSciBufAttrList is greater than or equal to
     * the value of the same attribute in every input unreconciled
     * NvSciBufAttrList that set it.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t
     *
     * valid input value: value is power of 2.
     */
    NvSciBufTensorAttrKey_BaseAddrAlign,

    /** Size of buffer allocated for 'N' tensors.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufTensorAttrKey_Size,

#if (NV_IS_SAFETY == 0)
    /** Specifies the data type of a NvSciBufType_Array.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @ref NvSciBufAttrValDataType
     *
     * valid input value: NvSciDataType_Int4 <= value <= NvSciDataType_Bool
     */
    NvSciBufArrayAttrKey_DataType   =  NV_SCI_BUF_ARRAY_ATTR_KEY_START,

    /** Specifies the stride of each element in the NvSciBufType_Array.
     * Stride must be greater than or equal to size of datatype specified by
     * NvSciBufArrayAttrKey_DataType.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufArrayAttrKey_Stride,

    /** Specifies the NvSciBufType_Array capacity.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufArrayAttrKey_Capacity,

    /** Indicates the total size of a NvSciBufType_Array.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufArrayAttrKey_Size,

    /** Indicates the base alignment of a NvSciBufType_Array.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t
     */
    NvSciBufArrayAttrKey_Alignment,

    /** Specifies the number of levels of images in a pyramid.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c uint32_t
     *
     * valid input value: 1 <= value <= NV_SCI_BUF_PYRAMID_MAX_LEVELS
     */
    NvSciBufPyramidAttrKey_NumLevels  =  NV_SCI_BUF_PYRAMID_ATTR_KEY_START,

    /** Specifies the scaling factor by which each successive image in a
     * pyramid must be scaled.
     * If more than one unreconciled NvSciBufAttrLists specify this input
     * attribute, reconciliation is successful if all the input attributes of
     * this type match.
     *
     * Type: Input/Output attribute
     *
     * Datatype: @c float
     *
     * valid input value: 0.0f < value <= 1.0f
     */
    NvSciBufPyramidAttrKey_Scale,

    /** NvSciBuf allocates the total buffer size considering all levels in an
     * image pyramid and returns an array of buffer offsets for each level.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufPyramidAttrKey_LevelOffset,

    /** NvSciBuf allocates the total buffer size considering all levels in an
     * image pyramid and returns an array of buffer sizes for each level.
     *
     * Type: Output attribute
     *
     * Datatype: @c uint64_t[]
     */
    NvSciBufPyramidAttrKey_LevelSize,

    /** Alignment attribute of pyramid
     *
     * Type: Output attribute
     * Datatype: uint64_t
     */
    NvSciBufPyramidAttrKey_Alignment,
#endif

    /** Specifies the maximum number of NvSciBuf attribute keys.
     * The total space for keys is 32K.
     *
     * Datatype: None
     */
    NvSciBufAttrKey_UpperBound = 0x3ffffffU,

} NvSciBufAttrKey;

/**
 * @}
 */

/**
 * @addtogroup nvscibuf_datatype
 * @{
 */

/**
 * @brief Defines buffer access permissions for NvSciBufObj.
 *
 * @implements{18840072}
 */
typedef enum {
    NvSciBufAccessPerm_Readonly = 1,
    NvSciBufAccessPerm_ReadWrite = 3,
    /*! Usage of Auto permissions is restricted only for export,
     *! import APIs and shouldn't be used to set value for
     *! NvSciBufGeneralAttrKey_RequiredPerm Attribute
     */
    NvSciBufAccessPerm_Auto,
    NvSciBufAccessPerm_Invalid,
} NvSciBufAttrValAccessPerm;

/**
 * @brief Defines the image layout type for NvSciBufType_Image.
 *
 * @implements{18840075}
 */
typedef enum {
    NvSciBufImage_BlockLinearType,
    NvSciBufImage_PitchLinearType,
} NvSciBufAttrValImageLayoutType;

/**
 * @brief Defines the image scan type for NvSciBufType_Image.
 *
 * @implements{18840078}
 */
typedef enum {
    NvSciBufScan_ProgressiveType,
    NvSciBufScan_InterlaceType,
} NvSciBufAttrValImageScanType;

/**
 * @brief Defines the image color formats for NvSciBufType_Image.
 *
 * @implements{18840081}
 */
typedef enum {
    NvSciColor_LowerBound,
    /* RAW PACKED */
    NvSciColor_Bayer8RGGB,
    NvSciColor_Bayer8CCCC,
    NvSciColor_Bayer8BGGR,
    NvSciColor_Bayer8GBRG,
    NvSciColor_Bayer8GRBG,
    NvSciColor_Bayer16BGGR,
    NvSciColor_Bayer16CCCC,
    NvSciColor_Bayer16GBRG,
    NvSciColor_Bayer16GRBG,
    NvSciColor_Bayer16RGGB,
    NvSciColor_Bayer16RCCB,
    NvSciColor_Bayer16BCCR,
    NvSciColor_Bayer16CRBC,
    NvSciColor_Bayer16CBRC,
    NvSciColor_Bayer16RCCC,
    NvSciColor_Bayer16CCCR,
    NvSciColor_Bayer16CRCC,
    NvSciColor_Bayer16CCRC,
    NvSciColor_X2Bayer14GBRG,
    NvSciColor_X4Bayer12GBRG,
    NvSciColor_X6Bayer10GBRG,
    NvSciColor_X2Bayer14GRBG,
    NvSciColor_X4Bayer12GRBG,
    NvSciColor_X6Bayer10GRBG,
    NvSciColor_X2Bayer14BGGR,
    NvSciColor_X4Bayer12BGGR,
    NvSciColor_X6Bayer10BGGR,
    NvSciColor_X2Bayer14RGGB,
    NvSciColor_X4Bayer12RGGB,
    NvSciColor_X6Bayer10RGGB,
    NvSciColor_X2Bayer14CCCC,
    NvSciColor_X4Bayer12CCCC,
    NvSciColor_X6Bayer10CCCC,
    NvSciColor_X4Bayer12RCCB,
    NvSciColor_X4Bayer12BCCR,
    NvSciColor_X4Bayer12CRBC,
    NvSciColor_X4Bayer12CBRC,
    NvSciColor_X4Bayer12RCCC,
    NvSciColor_X4Bayer12CCCR,
    NvSciColor_X4Bayer12CRCC,
    NvSciColor_X4Bayer12CCRC,
    NvSciColor_Signed_X2Bayer14CCCC,
    NvSciColor_Signed_X4Bayer12CCCC,
    NvSciColor_Signed_X6Bayer10CCCC,
    NvSciColor_Signed_Bayer16CCCC,
    NvSciColor_FloatISP_Bayer16CCCC,
    NvSciColor_FloatISP_Bayer16RGGB,
    NvSciColor_FloatISP_Bayer16BGGR,
    NvSciColor_FloatISP_Bayer16GRBG,
    NvSciColor_FloatISP_Bayer16GBRG,
    NvSciColor_FloatISP_Bayer16RCCB,
    NvSciColor_FloatISP_Bayer16BCCR,
    NvSciColor_FloatISP_Bayer16CRBC,
    NvSciColor_FloatISP_Bayer16CBRC,
    NvSciColor_FloatISP_Bayer16RCCC,
    NvSciColor_FloatISP_Bayer16CCCR,
    NvSciColor_FloatISP_Bayer16CRCC,
    NvSciColor_FloatISP_Bayer16CCRC,
    NvSciColor_X12Bayer20CCCC,
    NvSciColor_X12Bayer20BGGR,
    NvSciColor_X12Bayer20RGGB,
    NvSciColor_X12Bayer20GRBG,
    NvSciColor_X12Bayer20GBRG,
    NvSciColor_X12Bayer20RCCB,
    NvSciColor_X12Bayer20BCCR,
    NvSciColor_X12Bayer20CRBC,
    NvSciColor_X12Bayer20CBRC,
    NvSciColor_X12Bayer20RCCC,
    NvSciColor_X12Bayer20CCCR,
    NvSciColor_X12Bayer20CRCC,
    NvSciColor_X12Bayer20CCRC,
    NvSciColor_Signed_X12Bayer20CCCC,
    NvSciColor_Signed_X12Bayer20GBRG,

    /* Semiplanar formats */
    NvSciColor_U8V8,
    NvSciColor_U8_V8,
    NvSciColor_V8U8,
    NvSciColor_V8_U8,
    NvSciColor_U10V10,
    NvSciColor_V10U10,
    NvSciColor_U12V12,
    NvSciColor_V12U12,
    NvSciColor_U16V16,
    NvSciColor_V16U16,

    /* PLANAR formats */
    NvSciColor_Y8,
    NvSciColor_Y10,
    NvSciColor_Y12,
    NvSciColor_Y16,
    NvSciColor_U8,
    NvSciColor_V8,
    NvSciColor_U10,
    NvSciColor_V10,
    NvSciColor_U12,
    NvSciColor_V12,
    NvSciColor_U16,
    NvSciColor_V16,

    /* Packed YUV formats */
    NvSciColor_A8Y8U8V8,
    NvSciColor_Y8U8Y8V8,
    NvSciColor_Y8V8Y8U8,
    NvSciColor_U8Y8V8Y8,
    NvSciColor_V8Y8U8Y8,
    NvSciColor_A16Y16U16V16,

    /* RGBA PACKED */
    NvSciColor_A8,
    NvSciColor_Signed_A8,
    NvSciColor_B8G8R8A8,
    NvSciColor_A8R8G8B8,
    NvSciColor_A8B8G8R8,
    NvSciColor_A2R10G10B10,
    NvSciColor_A16,
    NvSciColor_Signed_A16,
    NvSciColor_Signed_R16G16,
    NvSciColor_A16B16G16R16,
    NvSciColor_Signed_A16B16G16R16,
    NvSciColor_Float_A16B16G16R16,
    NvSciColor_A32,
    NvSciColor_Signed_A32,
    NvSciColor_Float_A16,
    NvSciColor_UpperBound
} NvSciBufAttrValColorFmt;

/**
 * @brief Defines the image color standard for NvSciBufType_Image.
 *
 * @implements{18840084}
 */
typedef enum {
    NvSciColorStd_SRGB,
    NvSciColorStd_REC601_SR,
    NvSciColorStd_REC601_ER,
    NvSciColorStd_REC709_SR,
    NvSciColorStd_REC709_ER,
    NvSciColorStd_REC2020_RGB,
    NvSciColorStd_REC2020_SR,
    NvSciColorStd_REC2020_ER,
    NvSciColorStd_YcCbcCrc_SR,
    NvSciColorStd_YcCbcCrc_ER,
    NvSciColorStd_SENSOR_RGBA,
    NvSciColorStd_REQ2020PQ_ER,
} NvSciBufAttrValColorStd;

/**
 * @brief Defines various numeric datatypes for NvSciBuf.
 *
 * @implements{18840087}
 */
typedef enum {
    NvSciDataType_Int4,
    NvSciDataType_Uint4,
    NvSciDataType_Int8,
    NvSciDataType_Uint8,
    NvSciDataType_Int16,
    NvSciDataType_Uint16,
    NvSciDataType_Int32,
    NvSciDataType_Uint32,
    NvSciDataType_Float16,
    NvSciDataType_Float32,
    NvSciDataType_FloatISP,
    NvSciDataType_Bool,
    NvSciDataType_UpperBound
} NvSciBufAttrValDataType;

/**
 * @brief Defines GPU ID structure. This structure is used to
 * set the value for NvSciBufGeneralAttrKey_GpuId attribute.
 *
 * @implements{18840093}
 */
typedef struct {
    /** GPU ID. This member is initialized by the successful
     * call to cuDeviceGetUuid() for CUDA usecases */
    uint8_t bytes[16];
} NvSciRmGpuId;

/**
 * @}
 */

/**
 * @defgroup nvscibuf_attr_datastructures NvSciBuf Data Structures
 * Specifies NvSciBuf data structures.
 * @{
 */

/**
 * @brief top-level container for the following set of
 *        resources: NvSciBufAttrLists, memory objects, and NvSciBufObjs.
 * Any @ref NvSciBufAttrList created or imported using a particular @ref NvSciBufModule
 * is bound to it, along with any @ref NvSciBufObj created or
 * imported using those NvSciBufAttrList(s).
 *
 * @note For any NvSciBuf API call that has more than one input of type
 * NvSciBufModule, NvSciBufAttrList, and/or NvSciBufObj, all such inputs
 * must agree on the NvSciBufModule instance.
 */
typedef struct NvSciBufModuleRec* NvSciBufModule;

/**
 * @brief This structure defines a key/value pair used to get or set
 * the NvSciBufAttrKey(s) and their corresponding values from or to
 * NvSciBufAttrList.
 *
 * @note An array of this structure need to be set in order to
 * allocate a buffer.
 *
 * @implements{18840090}
 */
typedef struct {
     /** NvSciBufAttrKey for which value needs to be set/retrieved. This member
      * is initialized to any one of the NvSciBufAttrKey other than
      * NvSciBufAttrKey_LowerBound and NvSciBufAttrKey_UpperBound */
      NvSciBufAttrKey key;

      /** Memory which contains the value corresponding to the key */
      const void* value;

      /** Length of the value in bytes */
      size_t len;
} NvSciBufAttrKeyValuePair;

/**
 * A memory object is a container holding the reconciled NvSciBufAttrList
 * defining constraints of the buffer, the handle of the allocated buffer
 * enforcing the buffer access permissions represented by
 * NvSciBufGeneralAttrKey_ActualPerm key in reconciled NvSciBufAttrList
 * and the buffer properties.
 */

/**
 * @brief A reference to a particular Memory object.
 *
 * @note Every @ref NvSciBufObj that has been created but not freed
 * holds a reference to the @ref NvSciBufModule, preventing it
 * from being de-initialized.
 */
typedef struct NvSciBufObjRefRec* NvSciBufObj;

/**
 * @brief A reference, that is not modifiable, to a particular Memory Object.
 */
typedef const struct NvSciBufObjRefRec* NvSciBufObjConst;


/**
 * @brief A container constituting an attribute list which contains
 * - set of NvSciBufAttrKey attributes defining buffer constraints
 * - slotcount defining number of slots in an attribute list
 * - flag specifying if attribute list is reconciled or unreconciled
 *
 * @note Every @ref NvSciBufAttrList that has been created but not freed
 * holds a reference to the @ref NvSciBufModule, preventing it
 * from being de-initialized.
 */
typedef struct NvSciBufAttrListRec* NvSciBufAttrList;

/**
 * @brief Defines the exported form of NvSciBufObj intended to be
 * shared across an NvSciIpc channel.
 *
 * @implements{18840114}
 */
typedef struct {
      /** Exported data (blob) for NvSciBufObj */
      uint64_t data[NVSCIBUF_EXPORT_DESC_SIZE];
} __attribute__((packed)) NvSciBufObjIpcExportDescriptor;

/**
 * @}
 */

/**
 * @defgroup nvscibuf_attr_list_api NvSciBuf Attribute List APIs
 * Methods to perform operations on NvSciBuf attribute lists.
 * @{
 */

/**
 * @brief Creates a new, single slot, unreconciled NvSciBufAttrList associated
 * with the input NvSciBufModule with empty NvSciBufAttrKeys.
 *
 * @param[in] module NvSciBufModule to associate with the newly
 * created NvSciBufAttrList.
 * @param[out] newAttrList The new NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion status of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a module is NULL
 *      - @a newAttrList is NULL
 * - ::NvSciError_InsufficientMemory if insufficient system memory to
 *   create a NvSciBufAttrList.
 * - ::NvSciError_InvalidState if a new NvSciBufAttrList cannot be associated
 *   with the given NvSciBufModule.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a module is invalid.
 */
NvSciError NvSciBufAttrListCreate(
    NvSciBufModule module,
    NvSciBufAttrList* newAttrList);

/**
 * @brief Frees the NvSciBufAttrList and removes its association with the
 * NvSciBufModule with which it was created.
 *
 * @note Every owner of the NvSciBufAttrList shall call NvSciBufAttrListFree()
 * only after all the functions invoked by the owner with NvSciBufAttrList as
 * an input are completed.
 *
 * @param[in] attrList The NvSciBufAttrList to be freed.
 *
 * @return void
 * - panics if NvSciBufAttrList is invalid.
 */
void NvSciBufAttrListFree(
    NvSciBufAttrList attrList);

/**
 * @brief Sets the values for NvSciBufAttrKey(s) in the NvSciBufAttrList.
 * It only reads values from NvSciBufAttrKeyValuePair array and
 * saves copies during this call.
 *
 * @note All combinations of NvSciBufAttrListSetAttrs(),
 * NvSciBufAttrListGetAttrs(), NvSciBufAttrListAppendUnreconciled()
 * and NvSciBufAttrListReconcile() can be called concurrently,
 * however, function completion order is not guaranteed by NvSciBuf
 * and thus outcome of calling these functions concurrently is
 * undefined.
 *
 * @param[in] attrList Unreconciled NvSciBufAttrList.
 * @param[in] pairArray Array of NvSciBufAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and
 * key member of every NvSciBufAttrKeyValuePair in the array is a valid enumeration
 * value defined by the NvSciBufAttrKey enum and value member of every
 * NvSciBufAttrKeyValuePair in the array is not NULL.
 * @param[in] pairCount Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::NvSciError, the completion status of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrList is NULL
 *      - @a attrList is reconciled
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - any of the NvSciBufAttrKey(s) specified in the NvSciBufAttrKeyValuePair
 *        array is output only
 *      - any of the NvSciBufAttrKey(s) specified in the NvSciBufAttrKeyValuePair
 *        array has already been set
 *      - any of the NvSciBufAttrKey(s) specified in @a pairArray is not a
 *        valid enumeration value defined by the NvSciBufAttrKey enum
 *      - value(s) set for NvSciBufAttrKey(s) in @a pairArray are invalid
 * - ::NvSciError_InsufficientMemory if not enough system memory.
 * - Panics if any of the following occurs:
 *      - @a attrList is not valid
 */
NvSciError NvSciBufAttrListSetAttrs(
    NvSciBufAttrList attrList,
    NvSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Returns the slot count per NvSciBufAttrKey in a NvSciBufAttrList.
 *
 * @param[in] attrList The NvSciBufAttrList to retrieve the slot count from.
 *
 * @return size_t
 * - Number of slots in the NvSciBufAttrList
 * - panics if @a attrList is invalid
 */
size_t NvSciBufAttrListGetSlotCount(
    NvSciBufAttrList attrList);

/**
 * @brief Returns an array of NvSciBufAttrKeyValuePair for a given set of NvSciBufAttrKey(s).
 * This function accepts a set of NvSciBufAttrKey(s) passed in the @ref NvSciBufAttrKeyValuePair
 * structure. The return values, stored back into @ref NvSciBufAttrKeyValuePair, consist of
 * @c const @c void* pointers to the attribute values from the @ref NvSciBufAttrList.
 * The application must not write to this data.
 *
 * @param[in] attrList NvSciBufAttrList to fetch the NvSciBufAttrKeyValuePair(s) from.
 * @param[in,out] pairArray Array of NvSciBufAttrKeyValuePair.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every NvSciBufAttrKeyValuePair in the array is a valid enumeration value
 * defined by the NvSciBufAttrKey enum.
 * @param[in] pairCount Number of elements/entries in @a pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - @a pairCount is 0
 *      - any of the NvSciBufAttrKey(s) in @a pairArray is not a valid
 *        enumeration value defined by the NvSciBufAttrKey enum
 *      - @a attrList is reconciled and any of the NvSciBufAttrKey(s) specified
 *        in NvSciBufAttrKeyValuePair is input only
 *      - @a attrList is unreconciled and any of the NvSciBufAttrKey(s)
 *        specified in NvSciBufAttrKeyValuePair is output only
 * - Panics if any of the following occurs:
 *      - @a attrList is invalid
 */
NvSciError NvSciBufAttrListGetAttrs(
    NvSciBufAttrList attrList,
    NvSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

/**
 * @brief Returns an array of NvSciBufAttrKeyValuePair(s) from input
 * NvSciBufAttrList at the given slot index. The return values, stored in @ref
 * NvSciBufAttrKeyValuePair, consist of @c const @c void* pointers to the attribute values
 * from the NvSciBufAttrList. The application must not write to this data.
 *
 * @note When exporting an array containing multiple unreconciled NvSciBufAttrList(s),
 * the importing endpoint still imports just one unreconciled NvSciBufAttrList.
 * This unreconciled NvSciBufAttrList is referred to as a multi-slot
 * NvSciBufAttrList. It logically represents an array of NvSciBufAttrList(s), where
 * each key has an array of values, one per slot.
 *
 * @param[in] attrList NvSciBufAttrList to fetch the NvSciBufAttrKeyValuePair(s) from.
 * @param[in] slotIndex Index in the NvSciBufAttrList.
 * Valid value: 0 to slot count of NvSciBufAttrList - 1.
 * @param[in,out] pairArray Array of NvSciBufAttrKeyValuePair. Holds the NvSciBufAttrKey(s)
 * passed into the function and returns an array of NvSciBufAttrKeyValuePair structures.
 * Valid value: pairArray is valid input if it is not NULL and key member
 * of every NvSciBufAttrKeyValuePair is a valid enumeration value defined by the
 * NvSciBufAttrKey enum
 * @param[in] pairCount Number of elements/entries in pairArray.
 * Valid value: pairCount is valid input if it is non-zero.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrList is NULL
 *      - @a pairArray is NULL
 *      - any of the NvSciBufAttrKey(s) in @a pairArray is not a valid
 *        enumeration value defined by the NvSciBufAttrKey enum
 *      - @a pairCount is 0
 *      - @a slotIndex >= slot count of NvSciBufAttrList
 *      - NvSciBufAttrKey specified in @ pairArray is invalid.
 *      - @a attrList is reconciled and any of the NvSciBufAttrKey(s) specified
 *        in NvSciBufAttrKeyValuePair is input only
 *      - @a attrList is unreconciled and any of the NvSciBufAttrKey(s)
 *        specified in NvSciBufAttrKeyValuePair is output only
 * - Panics if any of the following occurs:
 *      - @a attrList is invalid
 */
NvSciError NvSciBufAttrListSlotGetAttrs(
    NvSciBufAttrList attrList,
    size_t slotIndex,
    NvSciBufAttrKeyValuePair* pairArray,
    size_t pairCount);

#if (NV_IS_SAFETY == 0)
/**
 * @brief Allocates a buffer and then dumps the contents of the specified
 * attribute list into the buffer.
 *
 * @param[in] attrList Attribute list to fetch contents from.
 * @param[out] buf A pointer to the buffer allocated for the debug dump.
 * @param[out] len The length of the buffer allocated for the debug dump.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if @a attrList is invalid.
 */
NvSciError NvSciBufAttrListDebugDump(
    NvSciBufAttrList attrList,
    void** buf,
    size_t* len);
#endif

/**
 * @brief Reconciles the given unreconciled NvSciBufAttrList(s) into a new
 * reconciled NvSciBufAttrList.
 * On success, this API call returns reconciled NvSciBufAttrList, which has to
 * be freed by the caller using NvSciBufAttrListFree().
 *
 * @param[in] inputArray Array containing unreconciled NvSciBufAttrList(s) to be
 *            reconciled. @a inputArray is valid if it is non-NULL.
 * @param[in] inputCount The number of unreconciled NvSciBufAttrList(s) in
 *            @a inputArray. This value must be non-zero. For a single
 *            NvSciBufAttrList, the count must be set 1.
 * @param[out] newReconciledAttrList Reconciled NvSciBufAttrList. This field
 *             is populated only if the reconciliation succeeded.
 */
#if (NV_IS_SAFETY == 0)
/**
 * @param[out] newConflictList Unreconciled NvSciBufAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * @param[out] newConflictList unused.
 */
#endif
/**
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a inputArray[] is NULL
 *      - @a inputCount is 0
 *      - @a newReconciledAttrList is NULL
 *      - any of the NvSciBufAttrList in @a inputArray is reconciled.
 *      - not all the NvSciBufAttrLists in @a inputArray are bound to the
 *        same NvSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *        type(s) of the NvSciBufAttrList(s) involved in reconciliation is
 *        unset
 *      - an attribute key is set to an unsupported value considering the data
 *        type(s) of the NvSciBufAttrList(s) involved in reconciliation
 */
#if (NV_IS_SAFETY == 0)
/**      - @a newConflictList is NULL
 */
#endif
/**
 * - ::NvSciError_InsufficientMemory if not enough system memory.
 * - ::NvSciError_InvalidState if a new NvSciBufAttrList cannot be associated
 *   with the NvSciBufModule associated with the NvSciBufAttrList(s) in the
 *   given @a inputArray to create a new reconciled NvSciBufAttrList
 * - ::NvSciError_NotSupported if an attribute key is set resulting in a
 *   combination of given constraints that are not supported
 * - ::NvSciError_Overflow if internal integer overflow is detected.
 * - ::NvSciError_ReconciliationFailed if reconciliation failed.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - Panic if:
 *      - @a unreconciled NvSciBufAttrList(s) in inputArray is not valid.
 */
NvSciError NvSciBufAttrListReconcile(
    const NvSciBufAttrList inputArray[],
    size_t inputCount,
    NvSciBufAttrList* newReconciledAttrList,
    NvSciBufAttrList* newConflictList);

/**
 * @brief Clones an unreconciled/reconciled NvSciBufAttrList. The resulting
 * NvSciBufAttrList contains all the values of the input NvSciBufAttrList.
 * If the input NvSciBufAttrList is an unreconciled NvSciBufAttrList, then
 * modification to the output NvSciBufAttrList will be allowed using
 * NvSciBufAttrListSetAttrs().
 *
 * @param[in] origAttrList NvSciBufAttrList to be cloned.
 *
 * @param[out] newAttrList The new NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a origAttrList is NULL
 *      - @a newAttrList is NULL
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory
 *   to create a new NvSciBufAttrList.
 * - ::NvSciError_InvalidState if a new NvSciBufAttrList cannot be associated
 *   with the NvSciBufModule of @a origAttrList to create the new
 *   NvSciBufAttrList.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a origAttrList is invalid
 */
NvSciError NvSciBufAttrListClone(
    NvSciBufAttrList origAttrList,
    NvSciBufAttrList* newAttrList);

/**
 * @brief Appends multiple unreconciled NvSciBufAttrList(s) together, forming a
 *  single new unreconciled NvSciBufAttrList with a slot count equal to the
 *  sum of all the slot counts of NvSciBufAttrList(s) in the input array.
 *
 * @param[in] inputUnreconciledAttrListArray[] Array containing the
 *  unreconciled NvSciBufAttrList(s) to be appended together.
 *  Valid value: Array of valid NvSciBufAttrList(s) where the array
 *  size is at least 1.
 * @param[in] inputUnreconciledAttrListCount Number of unreconciled
 * NvSciBufAttrList(s) in @a inputUnreconciledAttrListArray.
 * Valid value: inputUnreconciledAttrListCount is valid input if it
 * is non-zero.
 *
 * @param[out] newUnreconciledAttrList Appended NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a inputUnreconciledAttrListArray is NULL
 *      - @a inputUnreconciledAttrListCount is 0
 *      - @a newUnreconciledAttrList is NULL
 *      - any of the NvSciBufAttrList(s) in @a inputUnreconciledAttrListArray
 *        is reconciled
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if a new NvSciBufAttrList cannot be associated
 *   with the NvSciBufModule associated with the NvSciBufAttrList(s) in the
 *   given @a inputUnreconciledAttrListArray to create the new NvSciBufAttrList.
 * - ::NvSciError_ResourceError if system lacks resource other than memory.
 * - panics if @a any NvSciBufAttrList in the @a
 *   inputUnreconciledAttrListArray is invalid
 */
NvSciError NvSciBufAttrListAppendUnreconciled(
    const NvSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    NvSciBufAttrList* newUnreconciledAttrList);

/**
 * @brief Checks if the NvSciBufAttrList is reconciled.
 *
 * @param[in] attrList NvSciBufAttrList to check.
 * @param[out] isReconciled boolean value indicating whether the
 * @a attrList is reconciled or not.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrList is NULL
 *      - @a isReconciled is NULL
 * - panics if @a attrList is invalid
 */
NvSciError NvSciBufAttrListIsReconciled(
    NvSciBufAttrList attrList,
    bool* isReconciled);

/**
 * @brief Validates a reconciled NvSciBufAttrList against a set of
 *        unreconciled NvSciBufAttrList(s).
 *
 * @param[in] reconciledAttrList Reconciled NvSciBufAttrList list to be
 *            validated. @a unreconciledAttrListArray is valid if it is
 *            non-NULL.
 * @param[in] unreconciledAttrListArray Set of unreconciled NvSciBufAttrList(s)
 *            that need to be used for validation.
 * @param[in] unreconciledAttrListCount Number of unreconciled
 *            NvSciBufAttrList(s). This value must be non-zero.
 *            For a single NvSciBufAttrList, the count must be set to 1.
 * @param[out] isReconcileListValid Flag indicating if the reconciled
 *             NvSciBufAttrList satisfies the constraints of set of
 *             unreconciled NvSciBufAttrList(s).
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a reconciledAttrList is NULL or
 *      - @a unreconciledAttrListArray[] is NULL or
 *      - @a unreconciledAttrListCount is zero or
 *      - @a isReconcileListValid is NULL
 *      - any of the NvSciBufAttrList in @a unreconciledAttrListArray is
 *        reconciled.
 *      - not all the NvSciBufAttrLists in @a unreconciledAttrListArray are
 *        bound to the same NvSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *        type(s) of the NvSciBufAttrList(s) involved in reconciliation is
 *        unset
 *      - an attribute key is set to an unsupported value considering the data
 *        type(s) of the NvSciBufAttrList(s) involved in reconciliation
 * - ::NvSciError_ReconciliationFailed if validation of reconciled
 *   NvSciBufAttrList failed against input unreconciled NvSciBufAttrList(s).
 * - ::NvSciError_InsufficientMemory if internal memory allocation failed.
 * - ::NvSciError_Overflow if internal integer overflow occurs.
 * - Panics if:
 *      - @a unreconciled NvSciBufAttrList(s) in unreconciledAttrListArray
 *        is invalid.
 *      - @a reconciledAttrList is not valid.
 */
NvSciError NvSciBufAttrListValidateReconciled(
    NvSciBufAttrList reconciledAttrList,
    const NvSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    bool* isReconcileListValid);

/**
 * @}
 */

/**
 * @defgroup nvscibuf_obj_api NvSciBuf Object APIs
 * List of APIs to create/operate on NvSciBufObj.
 * @{
 */

/**
 * @brief Creates a new NvSciBufObj holding reference to the same
 * Memory object to which input NvSciBufObj holds the reference.
 *
 * @note The new NvSciBufObj created with NvSciBufObjDup() has same
 * NvSciBufAttrValAccessPerm as the input NvSciBufObj.
 *
 * @param[in] bufObj NvSciBufObj from which new NvSciBufObj needs
 * to be created.
 * @param[out] dupObj The new NvSciBufObj.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL
 *      - @a dupObj is NULL
 * - ::NvSciError_InsufficientMemory if memory allocation is failed.
 * - ::NvSciError_InvalidState if the total number of NvSciBufObjs referencing
 *   the memory object is INT32_MAX and the caller tries to take one more
 *   reference using this API.
 * - ::NvSciError_ResourceError if system lacks resource other than memory
 * - Panics if @a bufObj is invalid.
 */
NvSciError NvSciBufObjDup(
    NvSciBufObj bufObj,
    NvSciBufObj* dupObj);

/**
 * @brief Reconciles the input unreconciled NvSciBufAttrList(s) into a new
 * reconciled NvSciBufAttrList and allocates NvSciBufObj that meets all the
 * constraints in the reconciled NvSciBufAttrList.
 *
 * @note This interface just combines NvSciBufAttrListReconcile() and
 * NvSciBufObjAlloc() interfaces together.
 *
 * @param[in] attrListArray Array containing unreconciled NvSciBufAttrList(s) to
 * reconcile. Valid value: Array of valid unreconciled NvSciBufAttrList(s) where
 * array size is at least 1.
 * @param[in] attrListCount The number of unreconciled NvSciBufAttrList(s) in
 * @c attrListArray. Valid value: 1 to SIZE_MAX.
 *
 * @param[out] bufObj The new NvSciBufObj.
 */
#if (NV_IS_SAFETY == 0)
/**
 * @param[out] newConflictList Unreconciled NvSciBufAttrList consisting of the
 * key/value pairs which caused the reconciliation failure. This field is
 * populated only if the reconciliation failed.
 */
#else
/**
 * @param[out] newConflictList unused.
 */
#endif
/**
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrListCount is 0
 *      - @a attrListArray is NULL
 *      - @a bufObj is NULL
 *      - any of the NvSciBufAttrList in @a attrListArray is reconciled.
 *      - not all the NvSciBufAttrLists in @a attrListArray are bound to
 *        the same NvSciBufModule.
 *      - an attribute key necessary for reconciling against the given data
 *      type(s) of the NvSciBufAttrList(s) involved in reconciliation is
 *      unset
 *      - an attribute key is set to an unsupported value considering the data
 *      type(s) of the NvSciBufAttrList(s) involved in reconciliation
 */
#if (NV_IS_SAFETY == 0)
/**
 *      - @a newConflictList is NULL
 */
#endif
/**
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if a new NvSciBufAttrList cannot be associated
 *   with the NvSciBufModule associated with the NvSciBufAttrList(s) in the
 *   given @a attrListArray to create the new NvSciBufAttrList.
 * - ::NvSciError_NotSupported if an attribute key is set specifying a
 *   combination of constraints that are not supported
 * - ::NvSciError_Overflow if internal integer overflow is detected.
 * - ::NvSciError_ReconciliationFailed if reconciliation failed.
 * - ::NvSciError_ResourceError if any of the following occurs:
 *      - NVIDIA driver stack failed while assigning new permission to the buffer handle
 *      - system lacks resource other than memory
 * - Panics if any of the unreconciled NvSciBufAttrLists is invalid.
 */
NvSciError NvSciBufAttrListReconcileAndObjAlloc(
    const NvSciBufAttrList attrListArray[],
    size_t attrListCount,
    NvSciBufObj* bufObj,
    NvSciBufAttrList* newConflictList);

/**
 * @brief Removes reference to the Memory object by destroying the NvSciBufObj.
 *
 * @note Every owner of the NvSciBufObj shall call NvSciBufObjFree()
 * only after all the functions invoked by the owner with NvSciBufObj
 * as an input are completed.
 *
 * \param[in] bufObj The NvSciBufObj to deallocate.
 *
 * @return void
 * - Panics if @a bufObj is invalid.
 */
void NvSciBufObjFree(
    NvSciBufObj bufObj);

/**
 * @brief Retrieves the reconciled NvSciBufAttrList whose attributes define
 * the constraints of the allocated buffer from the NvSciBufObj.
 *
 * @note The retrieved NvSciBufAttrList from an NvSciBufObj is read-only,
 * and the attribute values in the list cannot be modified using
 * set attribute APIs. In addition, the retrieved NvSciBufAttrList must
 * not be freed with NvSciBufAttrListFree.
 *
 * @param[in] bufObj The NvSciBufObj to retrieve the NvSciBufAttrList from.
 * @param[out] bufAttrList The retrieved reconciled NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL.
 *      - @a bufAttrList is NULL.
 * - Panics if @a bufObj is invalid.
 */
NvSciError NvSciBufObjGetAttrList(
    NvSciBufObj bufObj,
    NvSciBufAttrList* bufAttrList);

/**
 * @brief Gets the CPU virtual address (VA) of the read/write buffer
 * referenced by the NvSciBufObj.
 *
 * @note This interface can be called successfully only if NvSciBufObj
 * was obtained from successful call to NvSciBufObjAlloc() or
 * NvSciBufObj was obtained from successful call to NvSciBufObjIpcImport()/
 * NvSciBufIpcImportAttrListAndObj() where NvSciBufAccessPerm_ReadWrite
 * permissions are granted to the imported NvSciBufObj (The permissions
 * of the NvSciBufObj are indicated by NvSciBufGeneralAttrKey_ActualPerm
 * key in the reconciled NvSciBufAttrList associated with it) and CPU
 * access is requested by setting NvSciBufGeneralAttrKey_NeedCpuAccess
 * to true.
 *
 * @param[in] bufObj The NvSciBufObj.
 *
 * @param[out] ptr The CPU virtual address (VA).
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL.
 *      - @a ptr is NULL.
 * - ::NvSciError_BadParameter NvSciBufObj either did not request
 *   for CPU access by setting NvSciBufGeneralAttrKey_NeedCpuAccess
 *   to true OR does not have NvSciBufAccessPerm_ReadWrite to the
 *   buffer.
 * - Panics if @a bufObj is invalid.
 */
NvSciError NvSciBufObjGetCpuPtr(
    NvSciBufObj bufObj,
    void**  ptr);

/**
 * @brief Gets the CPU virtual address (VA) of the read-only buffer
 * referenced by the NvSciBufObj.
 *
 * @note This interface can be called successfully only if NvSciBufObj
 * was obtained from successful call to NvSciBufObjAlloc() or
 * NvSciBufObj was obtained from successful call to NvSciBufObjIpcImport()/
 * NvSciBufIpcImportAttrListAndObj() where at least NvSciBufAccessPerm_Readonly
 * permissions are granted to the imported NvSciBufObj (The permissions of the
 * NvSciBufObj are indicated by NvSciBufGeneralAttrKey_ActualPerm key in the
 * reconciled NvSciBufAttrList associated with it) and CPU access is
 * requested by setting NvSciBufGeneralAttrKey_NeedCpuAccess to true.
 *
 * @param[in] bufObj The NvSciBufObj.
 *
 * @param[out] ptr the CPU virtual address (VA).
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL.
 *      - @a ptr is NULL.
 * - ::NvSciError_BadParameter NvSciBufObj either did not request
 *   for CPU access by setting NvSciBufGeneralAttrKey_NeedCpuAccess
 *   to true OR does not have at least NvSciBufAccessPerm_ReadOnly
 *   permissions to the buffer.
 * - Panics if @a bufObj is invalid.
 */
NvSciError NvSciBufObjGetConstCpuPtr(
    NvSciBufObj bufObj,
    const void**  ptr);

/**
 * @brief Flushes the given @c len bytes at starting @c offset in the
 * buffer referenced by the NvSciBufObj. Flushing is done only when
 * NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency key is set in
 * reconciled NvSciBufAttrList to true.
 *
 * @param[in] bufObj The NvSciBufObj.
 * @param[in] offset The starting offset in memory of the NvSciBufObj.
 * Valid value: 0 to buffer size - 1.
 * @param[in] len The length (in bytes) to flush.
 * Valid value: 1 to buffer size - offset.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL
 *      - @a len is zero
 *      - @a offset + @a len > buffer size.
 * - ::NvSciError_NotPermitted if buffer referenced by @a bufObj is
 *   not mapped to CPU.
 * - ::NvSciError_NotSupported if NvSciBufAllocIfaceType associated with the
 *   NvSciBufObj is not supported.
 * - ::NvSciError_Overflow if @a offset + @a len exceeds UINT64_MAX
 * - ::NvSciError_ResourceError if NVIDIA driver stack could not flush the
 *   CPU cache range.
 * - Panics if @a bufObj is invalid.
 */
NvSciError NvSciBufObjFlushCpuCacheRange(
    NvSciBufObj bufObj,
    uint64_t offset,
    uint64_t len);

/**
 * @brief Allocates a buffer that satisfies all the constraints defined by
 * the attributes of the specified reconciled NvSciBufAttrList, and outputs
 * a new NvSciBufObj referencing the Memory object containing the allocated
 * buffer properties.
 *
 * @param[in] reconciledAttrList The reconciled NvSciBufAttrList.
 *
 * @param[out] bufObj The new NvSciBufObj.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a reconciledList is NULL
 *      - @a reconciledList is not a reconciled NvSciBufAttrList
 *      - @a bufObj is NULL
 * - ::NvSciError_InsufficientMemory if there is insufficient memory
 *   to complete the operation.
 * - ::NvSciError_InvalidState if a new NvSciBufObj cannot be associated
 *   with the NvSciBufModule with which @a reconciledAttrList is associated to
 *   create the new NvSciBufObj.
 * - ::NvSciError_ResourceError if any of the following occurs:
 *      - NVIDIA driver stack failed during buffer allocation
 *      - system lacks resource other than memory
 * - Panics if @a reconciledAttrList is invalid.
 */
NvSciError NvSciBufObjAlloc(
    NvSciBufAttrList reconciledAttrList,
    NvSciBufObj* bufObj);

/**
 * @brief Creates a new memory object containing a buffer handle representing
 * new NvSciBufAttrValAccessPerm to the same buffer for which buffer handle
 * is contained in the input memory object referenced by input NvSciBufObj and
 * creates a new NvSciBufObj referencing it provided NvSciBufAttrValAccessPerm
 * are less than permissions represented by buffer handle in the memory object
 * referenced by input NvSciBufObj. This interface has same effect as calling
 * NvSciBufObjDup() if NvSciBufAttrValAccessPerm are same as permissions
 * represented by buffer handle in the memory object referenced by input
 * NvSciBufObj.
 *
 * @param[in] bufObj NvSciBufObj.
 * @param[in] reducedPerm Reduced access permissions that need to be imposed on
 * the new NvSciBufObj (see @ref NvSciBufAttrValAccessPerm).
 * Valid value: NvSciBufAttrValAccessPerm enum value >=
 * NvSciBufAccessPerm_Readonly and <= NvSciBufAttrValAccessPerm represented by
 * NvSciBufGeneralAttrKey_ActualPerm key in reconciled NvSciBufAttrList
 * associated with the input NvSciBufObj.
 * \param[out] newBufObj The new NvSciBufObj with new permissions.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL
 *      - @a newBufObj is NULL
 *      - @a reducedPerm is greater than the permissions specified in the value
 *        of the NvSciBufGeneralAttrKey_ActualPerm key
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidState if the total number of NvSciBufObjs referencing
 *   the memory object is INT32_MAX and the caller tries to take one more
 *   reference using this API.
 * - ::NvSciError_ResourceError if any of the following occurs:
 *      - NVIDIA driver stack failed while assigning new permission to the buffer handle
 *      - system lacks resource other than memory
 * - Panics of @a bufObj is invalid.
 */
NvSciError NvSciBufObjDupWithReducePerm(
    NvSciBufObj bufObj,
    NvSciBufAttrValAccessPerm reducedPerm,
    NvSciBufObj* newBufObj);

/**
 * @}
 */

/**
 * @defgroup nvscibuf_transport_api NvSciBuf APIs
 * List of APIs to transport NvSciBuf buffers and attribute list objects across
 * various communication boundaries that interact using NvSciIpc.
 * @{
 */


/**
 * @brief Exports NvSciBufAttrList and NvSciBufObj into an
 * NvSciIpc-transferable object export descriptor. The blob can be
 * transferred to the other processes to create a matching NvSciBufObj.
 *
 * @param[in] bufObj NvSciBufObj to export.
 * @param[in] permissions Flag indicating the expected access permission
 *            (see @ref NvSciBufAttrValAccessPerm). The valid value range is
 *            such that the value of NvSciBufGeneralAttrKey_ActualPerm set in
 *            the reconciled NvSciBufAttrList exported to the peer to which
 *            NvSciBufObj is being exported is less than or equal to @a accPerm
 *            and @a accPerm is less than or equal to underlying NvSciBufObj
 *            permission. Additionally, NvSciBufAccessPerm_Auto value is
 *            unconditionally valid.
 * @param[in] ipcEndpoint NvSciIpcEndpoint to identify the peer process.
 * \param[out] attrListAndObjDesc NvSciBuf allocates and fills in the
 *             exportable form of NvSciBufObj and its corresponding
 *             NvSciBufAttrList to be shared across an NvSciIpc channel.
 * \param[out] attrListAndObjDescSize Size of the exported blob.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDescSize is NULL
 *      - @a bufObj is NULL
 *      - @a ipcEndpoint is invalid
 * - ::NvSciError_InsufficientMemory if memory allocation failed
 * - ::NvSciError_InvalidOperation if reconciled NvSciBufAttrList of @a bufObj
 *   has greater permissions for the @a ipcEndpoint peer than the
 *   @a permissions
 * - ::NvSciError_Overflow if an arithmetic overflow occurs due to an invalid
 *     export descriptor
 * - ::NvSciError_NotPermitted if NvSciBufObj and NvSciBufAttrList associated
 *   with it are not being exported in the reverse direction of IPC path in
 *   which unreconciled NvSciBufAttrLists involved in reconciliation of
 *   NvSciBufAttrList associated with the input NvScibufObj were exported.
 * - ::NvSciError_ResourceError if the NVIDIA driver stack failed.
 * - Panic if @a bufObj is invalid
 *
 */
NvSciError NvSciBufIpcExportAttrListAndObj(
    NvSciBufObj bufObj,
    NvSciBufAttrValAccessPerm permissions,
    NvSciIpcEndpoint ipcEndpoint,
    void** attrListAndObjDesc,
    size_t* attrListAndObjDescSize);

/**
 * @brief This API is invoked by the importing process after it receives the
 * object export descriptor sent by the other process who has created
 * descriptor.
 * The importing process will create its own NvSciBufObj and return as
 * output.
 *
 * @param[in] module NvSciBufModule to be used for importing NvSciBufObj.
 * @param[in] ipcEndpoint NvSciIpcEndpoint to identify the peer process.
 * @param[in] attrListAndObjDesc The exported form of NvSciBufAttrList and
 *            NvSciBufObj. The valid value must be non NULL.
 * @param[in] attrListAndObjDescSize Size of the imported blob. This value must
 *            be non-zero.
 * @param[in] attrList[] Receiver side array of NvSciBufAttrList(s) against
 *            which the imported NvSciBufAttrList has to be validated. NULL is
 *            valid value here if the validation of the received
 *            NvSciBufAttrList needs to be skipped.
 * @param[in] count Number of NvSciBufAttrList objects in the array. This value
 *            must be non-zero.
 * @param[in] minPermissions Minimum permissions of the buffer that the process
 *            is expecting to import the buffer (see @ref
 *            NvSciBufAttrValAccessPerm). The valid data range is from
 *            NvSciBufAccessPerm_Readonly to less than
 *            NvSciBufAccessPerm_Invalid and should be less than
 *            NvSciBufAttrValAccessPerm, with which it was exported.
 * @param[in] timeoutUs Maximum delay (in microseconds) before an NvSciBuf
 *            object times out. The value of the variable is ignored currently.
 * \param[out] bufObj NvSciBuf Object duplicated and exported during the
 *             importing process. This object is associated with the reconciled
 *             NvSciBufAttrList imported from the attrListAndObjDesc.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a attrListAndObjDesc is NULL
 *      - @a attrListAndObjDesc represents an NvSciBufAttrList with invalid
 *        attribute key values set
 *      - @a attrListAndObjDesc is invalid
 *      - @a attrListAndObjDescSize is 0
 *      - @a count is 0
 *      - @a bufObj is NULL
 * - ::NvSciError_NotSupported if any of the following occurs:
 *      - @a attrListAndObjDesc represents an NvSciBufAttrList with same key
 *        multiple times
 *      - @a attrListAndObjDesc represents an NvSciBufAttrList with mandatory
 *        key needed for interpreting the buffer not set
 *      - @a attrListAndObjDesc is incompatible
 * - ::NvSciError_AccessDenied if @a minPermissions are greater than permissions
 *     with which object was exported
 * - ::NvSciError_AttrListValidationFailed if input unreconciled attribute
 *     lists' contraints are not satisfied by attributes associated with
 *     imported NvSciBuf object
 * - ::NvSciError_InsufficientMemory if memory allocation failed
 * - ::NvSciError_ResourceError if the NVIDIA driver stack failed.
 * - Panic if:
 *    - @a any of the unreconciled attribute lists are not valid
 *    - @a module is invalid
 *
 */
NvSciError NvSciBufIpcImportAttrListAndObj(
    NvSciBufModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* attrListAndObjDesc,
    size_t attrListAndObjDescSize,
    const NvSciBufAttrList attrList[],
    size_t count,
    NvSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    NvSciBufObj* bufObj);

/**
 * @brief Frees the descriptor used for exporting both NvSciBufAttrList and
 * NvSciBufObj together.
 *
 * @param[in] attrListAndObjDescBuf Descriptor to be freed. The valid value is
 *            the one returned by successful call to
 *            NvSciBufIpcAttrlistAndObjExport.
 *
 * @return void
 *
 */
void NvSciBufAttrListAndObjFreeDesc(
    void* attrListAndObjDescBuf);

/**
 * @brief Exports the NvSciBufObj into an NvSciIpc-transferable object
 * export descriptor.
 * Descriptor can be transferred to other end of IPC where matching
 * NvSciBufObj can be created from the descriptor.
 *
 * @param[in] bufObj NvSciBufObj to export.
 * @param[in] accPerm Flag indicating the expected access permission
 *            (see @ref NvSciBufAttrValAccessPerm). The valid value range is
 *            such that the value of NvSciBufGeneralAttrKey_ActualPerm set in
 *            the reconciled NvSciBufAttrList exported to the peer to which
 *            NvSciBufObj is being exported is less than or equal to @a accPerm
 *            and @a accPerm is less than or equal to underlying NvSciBufObj
 *            permission. Additionally, NvSciBufAccessPerm_Auto value is
 *            unconditionally valid.
 * @param[in] ipcEndpoint NvSciIpcEndpoint.
 * \param[out] exportData NvSciBuf populates the return value with exportable
 *             form of NvSciBufObj shared across an NvSciIpc channel.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a bufObj is NULL
 *      - @a accPerm is invalid
 *      - @a ipcEndpoint is invalid
 *      - @a exportData is NULL
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidOperation if reconciled NvSciBufAttrList of @a bufObj
 *   has greater permissions for the @a ipcEndpoint peer than the
 *   @a permissions
 * - ::NvSciError_NotPermitted if NvSciBufObj is not being exported in the
 *   reverse direction of IPC path in which unreconciled NvSciBufAttrLists
 *   involved in reconciliation of NvSciBufAttrList associated with the input
 *   NvScibufObj were exported.
 * - ::NvSciError_ResourceError if the NVIDIA driver stack failed.
 * - Panic if @a bufObj is invalid
 *
 */
NvSciError NvSciBufObjIpcExport(
    NvSciBufObj bufObj,
    NvSciBufAttrValAccessPerm accPerm,
    NvSciIpcEndpoint ipcEndpoint,
    NvSciBufObjIpcExportDescriptor* exportData);

/**
 * @brief Creates the NvSciBufObj based on supplied object export descriptor
 * and returns the NvSciBufObj bound to the reconciled NvSciBufAttrList.
 *
 * @param[in] ipcEndpoint NvSciIpcEndpoint.
 * @param[in] desc A pointer to an NvSciBufObjIpcExportDescriptor. The valid
 *            value is non-NULL that points to descriptor received on NvSciIpc
 *            channel.
 * @param[in] reconciledAttrList Reconciled NvSciBufAttrList returned by
 *            NvSciBufAttrListIpcImportReconciled().
 * @param[in] minPermissions Flag indicating the expected access permission.
 *            The valid value range is such that @a minPermissions <= permission
 *            with which NvSciBufObj was exported.
 * @param[in] timeoutUs Maximum delay (in microseconds) before an NvSciBufObj
              times out. The value of the variable is ignored currently.
 * @param[out] bufObj Imported NvSciBufObj created from the descriptor.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_AccessDenied if minPermissions are greater than permissions
 *     with which object was exported
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a ipcEndpoint is invalid
 *      - @a desc is NULL or invalid
 *      - @a reconciledAttrList is NULL
 *      - @a bufObj is NULL
 * - ::NvSciError_InsufficientMemory if there is insufficient system memory.
 * - ::NvSciError_Overflow if an arithmetic overflow occurs due to an invalid
 *     export descriptor
 * - ::NvSciError_ResourceError if the NVIDIA driver stack failed.
 * - Panic if @a reconciledAttrList is invalid
 */
NvSciError NvSciBufObjIpcImport(
    NvSciIpcEndpoint ipcEndpoint,
    const NvSciBufObjIpcExportDescriptor* desc,
    NvSciBufAttrList reconciledAttrList,
    NvSciBufAttrValAccessPerm minPermissions,
    int64_t timeoutUs,
    NvSciBufObj* bufObj);

/**
 * @brief Transforms the input unreconciled NvSciBufAttrList(s) to an exportable
 * unreconciled NvSciBufAttrList descriptor that can be transported by the
 * application to any remote process as a serialized set of bytes over an
 * NvSciIpc channel.
 *
 * @param[in] unreconciledAttrListArray The unreconciled NvSciBufAttrList(s) to
 *            be exported. The valid value is non NULL.
 * @param[in] unreconciledAttrListCount Number of unreconciled
 *            NvSciBufAttrList(s) in @a unreconciledAttrListArray. This value
 *            must be non-zero. For a single list, the count must be set 1.
 * @param[in] ipcEndpoint The NvSciIpcEndpoint.
 * @param[out] descBuf A pointer to the new unreconciled NvSciBufAttrList
 *             descriptor, which the caller can deallocate later using
 *             NvSciBufAttrListFreeDesc().
 * @param[out] descLen The size of the new unreconciled NvSciBufAttrList
 *             descriptor.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a unreconciledAttrListArray is NULL
 *      - @a unreconciledAttrListCount is 0
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - Panic if @a any of the NvSciBufAttrList(s) in @a unreconciledAttrListArray
 *   is invalid.
 *
 */
NvSciError NvSciBufAttrListIpcExportUnreconciled(
    const NvSciBufAttrList unreconciledAttrListArray[],
    size_t unreconciledAttrListCount,
    NvSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * @brief Transforms the reconciled NvSciBufAttrList to an exportable reconciled
 * NvSciBufAttrList descriptor that can be transported by the application to any
 * remote process as a serialized set of bytes over an NvSciIpc channel.
 *
 * @param[in] reconciledAttrList The reconciled NvSciBufAttrList to be exported.
 * @param[in] ipcEndpoint NvSciIpcEndpoint.
 * @param[out] descBuf A pointer to the new reconciled NvSciBufAttrList
 *             descriptor, which the caller can deallocate later using
 *             NvSciBufAttrListFreeDesc().
 * @param[out] descLen The size of the new reconciled NvSciBufAttrList
 *             descriptor.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a reconciledAttrList is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descLen is NULL
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_NotPermitted if reconciled NvSciBufAttrList is not being
 *   exported in the reverse direction of IPC path in which unreconciled
 *   NvSciBufAttrLists involved in reconciliation of input NvSciBufAttrList were
 *   exported.
 * - Panic if @a reconciledAttrList is invalid.
 *
 */
NvSciError NvSciBufAttrListIpcExportReconciled(
    NvSciBufAttrList reconciledAttrList,
    NvSciIpcEndpoint ipcEndpoint,
    void** descBuf,
    size_t* descLen);

/**
 * @brief Translates an exported unreconciled NvSciBufAttrList descriptor
 * (potentially received from any process) into an unreconciled NvSciBufAttrList.
 *
 * @param[in] module NvScibufModule with which to associate the
 *            imported NvSciBufAttrList.
 * @param[in] ipcEndpoint NvSciIpcEndpoint.
 * @param[in] descBuf The unreconciled NvSciBufAttrList descriptor to be
 *            translated into an unreconciled NvSciBufAttrList.  The valid value
 *            is non-NULL that points to descriptor received on NvSciIpc
 *            channel.
 * @param[in] descLen The size of the unreconciled NvSciBufAttrList descriptor.
 *            This value must be non-zero.
 * @param[out] importedUnreconciledAttrList The imported unreconciled
 *             NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descBuf represents an NvSciBufAttrList with attribute key that is
 *        not supported by current version of NvSciBuf
 *      - @a descBuf represents an NvSciBufAttrList with invalid attribute key
 *        values set
 *      - @a descBuf is invalid
 *      - @a descLen is 0
 *      - @a importedUnreconciledAttrList is NULL
 * - ::NvSciError_NotSupported if @a descBuf represents an NvSciBufAttrList with
 *     same key multiple times.
 * - ::NvSciError_InsufficientMemory if sufficient system memory.
 * - Panic if @a module is invalid
 *
 */
NvSciError NvSciBufAttrListIpcImportUnreconciled(
    NvSciBufModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    NvSciBufAttrList* importedUnreconciledAttrList);

/**
 * @brief Translates an exported reconciled NvSciBufAttrList descriptor
 * (potentially received from any process) into a reconciled NvSciBufAttrList.
 *
 * It also validates that the reconciled NvSciBufAttrList to be imported will
 * be a reconciled NvSciBufAttrList that is consistent with the constraints in
 * an array of input unreconciled NvSciBufAttrList(s). This is recommended
 * while importing what is expected to be a reconciled NvSciBufAttrList to
 * cause NvSciBuf to validate the reconciled NvSciBufAttrList against the input
 * un-reconciled NvSciBufAttrList(s), so that the importing process can be sure
 * that an NvSciBufObj will satisfy the input constraints.
 *
 * @param[in] module NvScibufModule with which to associate the
 *            imported NvSciBufAttrList.
 * @param[in] ipcEndpoint NvSciIpcEndpoint.
 * @param[in] descBuf The reconciled NvSciBufAttrList descriptor to be
 *            translated into a reconciled NvSciBufAttrList.  The valid value is
 *            non-NULL that points to descriptor received on NvSciIpc channel.
 * @param[in] descLen The size of the reconciled NvSciBufAttrList descriptor.
 *            This value must be non-zero.
 * @param[in] inputUnreconciledAttrListArray The array of unreconciled
 *            NvSciBufAttrList against which the new reconciled
 *            NvSciBufAttrList is to be validated. NULL pointer is acceptable
 *            as a parameter if the validation needs to be skipped.
 * @param[in] inputUnreconciledAttrListCount The number of unreconciled
 *            NvSciBufAttrList(s) in @a inputUnreconciledAttrListArray. If
 *            @a inputUnreconciledAttrListCount is non-zero, then this operation
 *            will fail with an error unless all the constraints of all the
 *            unreconciled NvSciBufAttrList(s) in inputUnreconciledAttrListArray
 *            are met by the imported reconciled NvSciBufAttrList.
 * @param[out] importedReconciledAttrList Imported reconciled NvSciBufAttrList.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a module is NULL
 *      - @a ipcEndpoint is invalid
 *      - @a descBuf is NULL
 *      - @a descBuf represents an NvSciBufAttrList with invalid attribute key
 *        values set
 *      - @a descBuf is invalid
 *      - @a descLen is 0
 *      - @a importedReconciledAttrList is NULL
 * - ::NvSciError_NotSupported if any of the following occurs:
 *      - @a descBuf represents an NvSciBufAttrList with same key multiple times
 *      - @a descBuf represents an NvSciBufAttrList with mandatory key needed
 *        for interpreting the buffer not set
 *      - @a descBuf is incompatible
 * - ::NvSciError_InsufficientMemory if memory allocation failed.
 * - ::NvSciError_InvalidOperation if the import process does not
 *     have permission to initiate the action to import the reconciled attribute
 *     list.
 * - ::NvSciError_AttrListValidationFailed if input unreconciled attribute
 *     lists' constraints are not satisfied by attributes associated with
 *     imported importedReconciledAttrList.
 * - Panic if:
 *      - @a any of the NvSciBufAttrList in
 *        inputUnreconciledAttrListArray is invalid
 *      - @a module is invalid
 *
 */
NvSciError NvSciBufAttrListIpcImportReconciled(
    NvSciBufModule module,
    NvSciIpcEndpoint ipcEndpoint,
    const void* descBuf,
    size_t descLen,
    const NvSciBufAttrList inputUnreconciledAttrListArray[],
    size_t inputUnreconciledAttrListCount,
    NvSciBufAttrList* importedReconciledAttrList);


/**
 * @brief Frees the NvSciBuf exported NvSciBufAttrList descriptor.
 *
 * @param[in] descBuf NvSciBufAttrList descriptor to be freed.  The valid value
 * is non-NULL.
 *
 * @return void
 *
 */
void NvSciBufAttrListFreeDesc(
    void* descBuf);

/**
 * @}
 */

/**
 * @defgroup nvscibuf_init_api NvSciBuf Initialization APIs
 * List of APIs to initialize/de-initialize NvSciBuf module.
 * @{
 */

/**
 * @brief Initializes and returns a new NvSciBufModule with no
 * NvSciBufAttrLists, buffers, or NvSciBufObjs bound to it.
 * @note A process may call this function multiple times.
 * Each successful invocation will yield a new NvSciBufModule.
 *
 * @param[out] newModule The new NvSciBufModule.
 *
 * @return ::NvSciError, the completion code of this operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if @a newModule is NULL.
 * - ::NvSciError_InsufficientMemory if memory is not available.
 * - ::NvSciError_ResourceError if any of the following occurs:
 *      - NVIDIA driver stack failed
 *      - system lacks resource other than memory
 */
NvSciError NvSciBufModuleOpen(
    NvSciBufModule* newModule);

/**
 * @brief Releases the NvSciBufModule obtained through
 * an earlier call to NvSciBufModuleOpen(). Once the NvSciBufModule is closed
 * and all NvSciBufAttrLists and NvSciBufObjs bound to it
 * are freed, the NvSciBufModule will be de-initialized in
 * the calling process.
 *
 * @param[in] module The NvSciBufModule to close.
 *
 * @return void
 * - Panic if @a module is invalid
 */
void NvSciBufModuleClose(
    NvSciBufModule module);

/**
 * \brief Checks if loaded NvSciBuf library version is compatible with
 * NvSciBuf library version with which elements dependent on NvSciBuf
 * were built.
 *
 * This function checks loaded NvSciBuf library version with input NvSciBuf
 * library version as well as versions of libraries that NvSciBuf depends on
 * and sets the output variable to true if all libraries are compatible, else
 * sets output to false.
 *
 * \param[in] majorVer build major version.
 * \param[in] minorVer build minor version.
 * \param[out] isCompatible boolean value stating if loaded NvSciBuf library is
 * compatible or not.
 * \return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a isCompatible is NULL or
 *      - failed to check dependent library versions.
 */
NvSciError NvSciBufCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

/**
 * @}
 */

/** @} */

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif /* INCLUDED_NVSCIBUF_H */
