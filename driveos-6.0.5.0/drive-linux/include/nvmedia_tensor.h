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
 * \brief <b> NVIDIA Media Interface: Tensor Processing </b>
 *
 * @b Description: This file contains the
 *                 \ref nvmedia_tensor_api "NvMedia Tensor Processing API".
 */

#ifndef NVMEDIA_TENSOR_H
#define NVMEDIA_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_tensormetadata.h"

/**
 * \defgroup nvmedia_tensor_top Tensor Handling API
 *
 * The Tensor Processing API encompasses all NvMedia tensor related functionality.
 *
 * @ingroup nvmedia_top
 * @{
 */

 /**
 * \defgroup nvmedia_tensor_api Tensor Handling API
 *
 * The Tensor Processing API encompasses all NvMedia tensor related functionality.
 *
 * @ingroup nvmedia_tensor_top
 * @{
 */

/** \brief Major version number. */
#define NVMEDIA_TENSOR_VERSION_MAJOR   (1u)
/** \brief Minor version number. */
#define NVMEDIA_TENSOR_VERSION_MINOR   (13u)
/** \brief Patch version number. */
#define NVMEDIA_TENSOR_VERSION_PATCH   (0u)

/** \brief Defines the maximum supported number of tensor surfaces */
#define NVMTENSOR_4D_MAX_N       (65536u)
/** \brief Defines the maximum supported number of channels of a 4D tensor */
#define NVMTENSOR_4D_MAX_C       (8192u)
/** \brief Defines the maximum supported height of a 4D tensor */
#define NVMTENSOR_4D_MAX_H       (8192u)
/** \brief Defines the maximum supported width of a 4D tensor */
#define NVMTENSOR_4D_MAX_W       (8192u)
/** \brief Defines the maximum supported x value of a 4D tensor. This value is only for NCxHWx order. */
#define NVMTENSOR_4D_MAX_X       (1024u)

/**
 * \hideinitializer
 * \brief This macro tells NvMediaTensorGetStatus() and NvMediaTensorLock()
 *  APIs to block infinitely till the operation finishes.
 */
#define NVMEDIA_TENSOR_TIMEOUT_INFINITE       (0xFFFFFFFFu)

/**
 * \brief Holds the status of the latest operation for a tensor.
 *
 */
typedef struct {
    /** \brief Holds the return status of the operation as an error code of type - \ref NvMediaStatus. */
    NvMediaStatus status;
    /** \brief Duration of the operation in microseconds. */
    uint32_t durationUs;
} NvMediaTensorTaskStatus;

/**
 * \brief A handle representing tensor objects.
 */
typedef struct NvMediaTensor NvMediaTensor;

/**
 * \brief Defines attribute types for creating NvMedia Tensor.
 * The maximum supported NvMediaTensor size is (UINT32_MAX) = 4294967295 bytes.
 */
typedef enum {
    /** \brief Specifies the tensor data type. */
    NVM_TENSOR_ATTR_DATA_TYPE = 0,
    /** \brief Defines the tensor bits per element. */
    NVM_TENSOR_ATTR_BITS_PER_ELEMENT,
    /** \brief Defines the tensor dimension order. */
    NVM_TENSOR_ATTR_DIMENSION_ORDER,
    /** \brief Defines the CPU access to tensor. (default: uncached) */
    NVM_TENSOR_ATTR_CPU_ACCESS,
    /** \brief Defines the allocation type, reserved. (default: none) */
    NVM_TENSOR_ATTR_ALLOC_TYPE,
    /** \brief Attribute types for 4D tensors.
     * Defines the number of tensor surfaces.
     */
    NVM_TENSOR_ATTR_4D_N,
    /** \brief Defines the number of channels of a 4D tensor. */
    NVM_TENSOR_ATTR_4D_C,
    /** \brief Defines the height of a 4D tensor. */
    NVM_TENSOR_ATTR_4D_H,
    /** \brief Defines the width of a 4D tensor. */
    NVM_TENSOR_ATTR_4D_W,
    /** \brief Defines the x value of a 4D tensor. This value is only for NCxHWx order */
    NVM_TENSOR_ATTR_4D_X,
    /** \brief Defines the maximum number of tensor creation attributes */
    NVM_TENSOR_ATTR_MAX
} NvMediaTensorAttrType;

/**
 * \brief Specifies that the tensor CPU accesses are uncached.
 * In this case, the tensor memory will be mapped and can be accessed with a mapping
 * into the current process's virtual address space.
 */
#define NVM_TENSOR_ATTR_CPU_ACCESS_UNCACHED                 (0x00000001u)
/**
 * \brief Specifies that the tensor CPU accesses are cacheable.
 * In this case, the tensor memory will be mapped and can be accessed with a mapping
 * into the current process's virtual address space.
 */
#define NVM_TENSOR_ATTR_CPU_ACCESS_CACHED                   (0x00000002u)
/**
 * \brief Specifies that the tensor CPU accesses are unmapped from the virtual
 *        address space of the current process.
 */
#define NVM_TENSOR_ATTR_CPU_ACCESS_UNMAPPED                 (0x00000003u)

#if NV_BUILD_CONFIGURATION_EXPOSING_T19X
/**
 * \brief Specifies that the tensor allocation is on CVSRAM.
 * CVSRAM is currently not supported in the Safety build.
 */
#define NVM_TENSOR_ATTR_ALLOC_RESERVED                      (0x00000010u)
#endif

/**
 * \brief Specifies that the tensor allocation is on Soc DRAM.
 */
#define NVM_TENSOR_ATTR_ALLOC_NONE                          (0x00000000u)

/**
 * \brief Holds tensor creation attributes.
 */
typedef struct {
    /** \brief Holds tensor creation attribute type. */
    NvMediaTensorAttrType type;
    /** \brief Holds tensor creation attribute value.*/
    uint32_t value;
} NvMediaTensorAttr;

/**
 * \brief A helper macro to initialize tensor creation attributes.
 * \param x The array to initialize with tensor creation attributes.
 */
#define NVM_TENSOR_INIT_ATTR(x)                                                       \
{                                                                                     \
        x[0].type  = NVM_TENSOR_ATTR_DATA_TYPE;                                       \
        x[0].value = 0;                                                               \
                                                                                      \
        x[1].type  = NVM_TENSOR_ATTR_4D_N;                                            \
        x[1].value = 0;                                                               \
                                                                                      \
        x[2].type  = NVM_TENSOR_ATTR_4D_C;                                            \
        x[2].value = 0;                                                               \
                                                                                      \
        x[3].type  = NVM_TENSOR_ATTR_4D_H;                                            \
        x[3].value = 0;                                                               \
                                                                                      \
        x[4].type  = NVM_TENSOR_ATTR_4D_W;                                            \
        x[4].value = 0;                                                               \
                                                                                      \
        x[5].type  = NVM_TENSOR_ATTR_4D_X;                                            \
        x[5].value = 0;                                                               \
                                                                                      \
        x[6].type  = NVM_TENSOR_ATTR_BITS_PER_ELEMENT;                                \
        x[6].value = 0;                                                               \
                                                                                      \
        x[7].type  = NVM_TENSOR_ATTR_DIMENSION_ORDER;                                 \
        x[7].value = 0;                                                               \
                                                                                      \
        x[8].type  = NVM_TENSOR_ATTR_CPU_ACCESS;                                      \
        x[8].value = 0;                                                               \
                                                                                      \
        x[9].type  = NVM_TENSOR_ATTR_ALLOC_TYPE;                                      \
        x[9].value = 0;                                                               \
}

/**
 * \brief A helper macro to define tensor creation attributes.
 */
#define NVM_TENSOR_DEFINE_ATTR(x)                                                     \
    NvMediaTensorAttr x[NVM_TENSOR_ATTR_MAX];                                         \
    NVM_TENSOR_INIT_ATTR(x);                                                          \

/**
 * \brief A helper macro to set 4-D tensor creation attributes.
 * The `attr` parameter must be defined before setting the values
 * using \ref NVM_TENSOR_SET_ATTR_4D.
 * For example, to set the attributes for `NCHW 4-D FP16` tensor:

 * \code
 * NVM_TENSOR_SET_ATTR_4D(attr, N, C, H, W, NCHW, FLOAT, 16, UNCACHED, NONE, 32);
 * \endcode

 * \param attr        The attribute to set.
 * \param N           A tensor layout dimension for number of surfaces.
 * \param C           A tensor layout dimension for number of channels on the surface.
 * \param H           A tensor layout dimension for the height of the surface.
 * \param W           A tensor layout dimension for the width of the surface.
 * \param order       The order of the tensor elements.
 * \param datatype    The datatype for this tensor.
 * \param bpe         Specifies the bits per element.
 * \param accesstype  Specifies the access type.
 * \param alloctype   Specifies the allocation type.
 * \param X           A tensor layout dimension for X.
 */
#define NVM_TENSOR_SET_ATTR_4D(attr, N, C, H, W, order, datatype, bpe, accesstype, alloctype, X)\
{                                                                                               \
    attr[0].type = NVM_TENSOR_ATTR_DATA_TYPE;                                                   \
    attr[0].value = NVM_TENSOR_ATTR_DATA_TYPE_##datatype;                                       \
                                                                                                \
    attr[1].type = NVM_TENSOR_ATTR_4D_N;                                                        \
    attr[1].value = N;                                                                          \
                                                                                                \
    attr[2].type = NVM_TENSOR_ATTR_4D_C;                                                        \
    attr[2].value = C;                                                                          \
                                                                                                \
    attr[3].type = NVM_TENSOR_ATTR_4D_H;                                                        \
    attr[3].value = H;                                                                          \
                                                                                                \
    attr[4].type = NVM_TENSOR_ATTR_4D_W;                                                        \
    attr[4].value = W;                                                                          \
                                                                                                \
    attr[5].type = NVM_TENSOR_ATTR_4D_X;                                                        \
    attr[5].value = X;                                                                          \
                                                                                                \
    attr[6].type = NVM_TENSOR_ATTR_BITS_PER_ELEMENT;                                            \
    attr[6].value = NVM_TENSOR_ATTR_BITS_PER_ELEMENT_##bpe;                                     \
                                                                                                \
    attr[7].type = NVM_TENSOR_ATTR_DIMENSION_ORDER;                                             \
    attr[7].value = NVM_TENSOR_ATTR_DIMENSION_ORDER_##order;                                    \
                                                                                                \
    attr[8].type = NVM_TENSOR_ATTR_CPU_ACCESS;                                                  \
    attr[8].value = NVM_TENSOR_ATTR_CPU_ACCESS_##accesstype;                                    \
                                                                                                \
    attr[9].type = NVM_TENSOR_ATTR_ALLOC_TYPE;                                                  \
    attr[9].value = NVM_TENSOR_ATTR_ALLOC_##alloctype;                                          \
}

/**
 * \brief Defines tensor lock access types.
 * \ingroup lock_unlock
 */
typedef enum {
    /** \brief Read access */
    NVMEDIA_TENSOR_ACCESS_READ       = (1 << 0),
    /** \brief Write access */
    NVMEDIA_TENSOR_ACCESS_WRITE      = (1 << 1),
    /** \brief Read/Write access */
    NVMEDIA_TENSOR_ACCESS_READ_WRITE = (NVMEDIA_TENSOR_ACCESS_READ | NVMEDIA_TENSOR_ACCESS_WRITE)
} NvMediaTensorLockAccess;

/**
 * \brief Defines the tensor surface map descriptor used by NvMediaTensorLock().
 *
 * The descriptor holds a maximum of \ref NVMEDIA_TENSOR_MAX_DIMENSIONS dimensions.
 *
 * \ingroup lock_unlock
 */
typedef struct {
    /** \brief Total size of the tensor. */
    uint32_t size;
    /** \brief CPU accessible memory pointer of Tensor. */
    void *mapping;
} NvMediaTensorSurfaceMap;

/**
 * \brief Destroys a tensor object previously created by
 * NvMediaTensorCreateFromNvSciBuf().
 *
 * \param[in] tensor The tensor to destroy.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
void
NvMediaTensorDestroy(
    NvMediaTensor *tensor
);

/**
 * \brief Locks a tensor and returns the associated mapped pointers
 * pointing to the tensor surface data.
 *
 * The CPU can only access tensors created with the
 * \ref NVM_TENSOR_ATTR_CPU_ACCESS_UNCACHED or \ref NVM_TENSOR_ATTR_CPU_ACCESS_CACHED
 * attributes.
 *
 * If a tensor is currently in use by an internal engine, this function waits
 * until the operation completes.
 *
 * \param[in] tensor           A pointer to the tensor object.
 * \param[in] lockAccessType   Specifies the NvMediaTensorLockAccess type.
 * \param[out] surfaceMap      A pointer to the surface descriptors.
 *
 * \return \ref NvMediaStatus The completion status of the operation:
 * @retval NVMEDIA_STATUS_OK if the function is successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if any of the arguments are NULL or invalid.
 * @retval NVMEDIA_STATUS_TIMEOUT if the wait on the tensor timed out.
 * @retval NVMEDIA_STATUS_ERROR if an error occurred.
 * \ingroup lock_unlock
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaTensorLock(
    NvMediaTensor *tensor,
    NvMediaTensorLockAccess lockAccessType,
    NvMediaTensorSurfaceMap *surfaceMap
);

/**
 * \brief Unlocks a tensor.
 *
 * Releases the lock applied on NvMediaTensor using NvMediaTensorLock.
 *
 * \param[in] tensor The tensor object to unlock.
 * \ingroup lock_unlock
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
void
NvMediaTensorUnlock(
    NvMediaTensor *tensor
);

/**
 * \brief Gets the status of the last operation for the tensor,
 *        and optionally waits for the operation to complete or time out.
 *
 * \param[in] tensor          The handle to the tensor object.
 * \param[in] millisecondWait Time in milliseconds to wait for the operation to
 *                            complete before getting the status.
 *                            Recommended value is NVMEDIA_TENSOR_TIMEOUT_INFINITE.
 * \param[out] status The status of the operation.
 * \return \ref NvMediaStatus, the completion status of the operation:
 * @retval NVMEDIA_STATUS_OK if the function is successful.
 * @retval NVMEDIA_STATUS_TIMEOUT if the wait on the tensor timed out.
 * @retval NVMEDIA_STATUS_ERROR if an error occurred.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if any of the arguments are NULL or invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaTensorGetStatus(
    NvMediaTensor *tensor,
    uint32_t millisecondWait,
    NvMediaTensorTaskStatus *status
);

/**
 * \brief Fills in the metadata information for the tensor.
 *
 * \param[in] tensor               The tensor object to get metadata from.
 * \param[in, out] tensormetadata  A pointer to a NvMediaTensorMetaData
 *                                 structure where tensor metadata is copied.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * @retval NVMEDIA_STATUS_OK if the function is successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if any of the arguments are NULL or invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaTensorGetMetaData(
    const NvMediaTensor *tensor,
    NvMediaTensorMetaData *tensormetadata
);

/**
 * \brief  Returns version information for the %NvMediaTensor library.
 *
 * \param[out] version  A pointer to a structure in which the function may
 *                       store version information.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * @retval NVMEDIA_STATUS_OK if the function call is successful.
 * @retval NVMEDIA_STATUS_BAD_PARAMETER if @a version is invalid.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaTensorGetVersion(
    NvMediaVersion *version
);

/**@} <!-- Ends nvmedia_tensor_api Tensor Handling API --> */

/*
 * \defgroup history_nvmedia_tensor History
 * Provides change history for the NvMedia Tensor API.
 *
 * \section history_nvmedia_tensor Version History
 *
 * <b> Version 1.0 </b> May 22, 2017
 * - Initial release
 *
 * <b> Version 1.2 </b> Jun 21, 2019
 * - Fix Minor Misra Violations
 *
 * <b> Version 1.3 </b> Jun 22, 2019
 * - Deprecate the NvMediaTensorCreate API
 *   in support of NvSciBuf APIs
 *
 * <b> Version 1.4 </b> Dec 9, 2019
 * - Add const to NvMediaGetTensorMetaData
 *   in support of Misra rule 8.13
 *
 * <b> Version 1.5 </b> Jan 10, 2020
 * - In NvMediaTensorLock API fix data type of
 *   lockAccessType parameter
 *
 * <b> Version 1.6 </b> Jan 15, 2020
 * - Fix the comments for: NvMediaTensorGetMetaData,
 *   NvMediaTensorGetStatus, NvMediaTensorLock
 *
 * <b> Version 1.7 </b> Feb 13, 2020
 * - Fix the comments for: NvMediaTensorLock,
 *   NvMediaTensorGetStatus, NvMediaTensorGetMetaData
 * - Rearranged NvMediaTensorDestroy
 *
 * <b> Version 1.8 </b> Mar 25, 2020
 * - Fix the doxygen comments for most functions
 * - Fixed NvMediaTensorDestroy misra violation 8.3
 *
 * <b> Version 1.9 </b> Apr 14, 2020
 * - Updated Doxygen comments for enums, macros and structs
 *
 * <b> Version 1.10 </b> May 7, 2020
 * - Updated Doxygen comments for NvMediaTensor struct
 *
 * <b> Version 1.11 </b> June 5, 2020
 * - Added NvMediaTensorGetVersion API
 *
 * <b> Version 1.12 </b> June 17, 2020
 * - Added Max Tensor dimension macros
 *
 * <b> Version 1.13 </b> August 27, 2021
 * - Update doxygen comments for all APIs to have Thread safety information and API Group information
 *
 * <b> Version 1.13 </b> February 08, 2022
 * - Updated the doxygen comments with usage considerations for all APIs.
 *
 * <b> Version 1.13 </b> April 04, 2022
 * - Update doxygen comments for NvMediaTensorLock and NvMediaTensorGetStatus APIs
 *
 * <b> Version 1.13.0 </b> May 10, 2022
 * - Added patch version number macro: NVMEDIA_TENSOR_VERSION_PATCH.
 *
 **/
/**@} <!-- Ends nvmedia_tensor_top Tensor Handling API --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_TENSOR_H */
