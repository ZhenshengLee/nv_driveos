// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef SIVC_INSTANCE_H
#define SIVC_INSTANCE_H

/**
 * @addtogroup SIVC_API
 * @{
 */

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#include "ivclib-static-analysis.h"
#include "ivclib-fwd-decl.h"

#define SIVC_ALIGN_SHIFT 6U

#ifdef __cplusplus
#define SIVC_ALIGN       (static_cast<uint32_t>(1U) << SIVC_ALIGN_SHIFT)
namespace Ivc {
extern "C" {
#else
#define SIVC_ALIGN       (((uint32_t)1U) << SIVC_ALIGN_SHIFT)
#endif

/**
 * @brief Align a number.
 * @param[in] value An unsigned integer
 *
 * @restriction @id{Init_Align} @asil{D}
 * The caller must ensure that the argument is less than or equal to
 * (UINT32_MAX - SIVC_ALIGN + 1).
 *
 * @return The closest integer that is greater than or equal to @p value and
 * divisible by SIVC_ALIGN.
 *
 * @outcome @id{Align_Result} @asil{D}
 * @c sivc_align returns the closest integer that is greater than or equal to
 * @p value and divisible by SIVC_ALIGN.
 */
uint32_t sivc_align(uint32_t value);

/**
 * @brief Calculate size of the memory needed for IVC fifo.
 * @param[in] nframes    Number of IVC queue frames
 * @param[in] frame_size Size of one frame in bytes
 *
 * @return Number of bytes needed for IVC fifo memory area, or 0 if fails.
 *
 * Function fails if:
 * @li @p frame_size is not a multiple of SIVC_ALIGN.
 * @li Computed IVC FIFO size exceeds UINT32_MAX
 *
 * @outcome @id{FifoSize_Success} @asil{D}
 * If succeeded @c sivc_align returns number of bytes needed for IVC fifo memory
 * area.
 *
 * @outcome @id{FifoSize_Failure} @asil{D}
 * In case of error @c sivc_align returns 0.
 */
uint32_t sivc_fifo_size(uint32_t nframes, uint32_t frame_size);

struct sivc_fifo_header;

/**
 * Callback: signal the remote endpoint.
 * @param[in] queue IVC queue endpoint from which to signal.
 */
STAN_A7_1_6_PD_HYP_6328
typedef void (*sivc_notify_function)(struct sivc_queue* queue);

/**
 * @brief Callback: Invalidate cache for a given memory range.
 * @param[in] addr Start address
 * @param[in] addr Memory range size
 */
STAN_A2_11_1_FP_HYP_5204 STAN_A7_1_6_PD_HYP_6328
typedef void (*sivc_cache_invalidate_function)(const volatile void* addr,
        size_t size);

/**
 * @brief Callback: Flush cache for a given memory range.
 * @param[in] addr Start address
 * @param[in] addr Memory range size
 */
STAN_A2_11_1_FP_HYP_5204 STAN_A7_1_6_PD_HYP_6328
typedef void (*sivc_cache_flush_function)(const volatile void* addr,
        size_t size);

/* WARNING: This structure is a part of private IVC interface.
 *          You should not access any of its member directly. Please use helper
 *          functions instead.
 *          See sivc_get_nframes() and sivc_get_frame_size().
 */
struct sivc_queue {
    STAN_A9_6_1_FP_HV_070 STAN_A2_11_1_FP_HYP_5204
    volatile struct sivc_fifo_header* recv_fifo;
    STAN_A9_6_1_FP_HV_070 STAN_A2_11_1_FP_HYP_5204
    volatile struct sivc_fifo_header* send_fifo;
    uint32_t w_pos;
    uint32_t r_pos;
    uint32_t nframes;
    uint32_t frame_size;
    STAN_A9_6_1_FP_HV_070
    sivc_notify_function notify;
    STAN_A9_6_1_FP_HV_070
    sivc_cache_invalidate_function mem_cache_invalidate;
    STAN_A9_6_1_FP_HV_070
    sivc_cache_flush_function mem_cache_flush;
};

/**
 * @brief  Initialize IVC queue control structure.
 * @param[in] queue            IVC queue
 * @param[in] recv_base        Shared memory address of receive IVC FIFO
 * @param[in] send_base        Shared memory address of transmit IVC FIFO
 * @param[in] nframes          Number of frames in a queue
 * @param[in] frame_size       Frame size in bytes
 * @param[in] notify           Notification callback, can be NULL
 * @param[in] cache_invalidate Memory cache invalidation callback, can be NULL
 * @param[in] sivc_cache_flush Memory cache flush callback, can be NULL
 *
 * IVC queue control structure is considered to be private, even though is
 * is declared in public header. This function should be used to set it up.
 *
 * @return 0 on success, or a negative error value on failure.
 *
 * Function fails if:
 * @li @c -EINVAL    The @p queue is NULL
 * @li @c -EINVAL    @p recv_base or @p send_base are zero
 * @li @c -EINVAL    @p recv_base or @p send_base are not aligned to SIVC_ALIGN
 * @li @c -EINVAL    @p frame_size is not aligned to SIVC_ALIGN
 * @li @c -EINVAL    Receive FIFO memory area and send FIFO memory area overlap
 * @li @c -EINVAL    Expected IVC FIFO size is bigger than 2^32
 *
 * @restriction @id{Init_SharedMemoryMapping} @asil{D}
 * The caller is responsible for ensuring that both IVC queue endpoints
 * are compatibly initialized as specified by following preconditions:
 * The IVC Library execution environment shall provide a region of memory that
 * is mapped into the address space (execution domain) of both sides of the IVC
 * channel with read-write access, of sufficient size to contain the two FIFOs
 * described in the configuration information. This region shall be used
 * exclusively by the IVC Library, except as documented by
 * @c sivc_get_read_frame and @c sivc_get_write_frame.
 * @rationale Both send and receive FIFOs require both read and write access for
 * transitional, backwards compatibility with Legacy IVC implementations.
 *
 * @restriction @id{Cfg_FifoSymmetry} @asil{D}
 * Sending and receiving FIFO buffers at the local endpoint must correspond
 * receiving and sending FIFO buffers at the remote endpoint:
 *      @li @p recv_base must correspond to the same underlying physical memory
 *      as @p send_base for the other IVC queue endpoint.
 *      @li @p send_base must correspond to the same underlying physical memory
 *      as @p recv_base for the other IVC queue endpoint.
 *
 * @restriction @id{Cfg_MaxFrameCount} @asil{D}
 * Number of frames in a queue @p nframes must be identical for both IVC queue
 * endpoints.
 *
 * @restriction @id{Cfg_FrameSize} @asil{D}
 * Frame size @p frame_size must be identical for both IVC queue endpoints.
 *
 * @restriction @id{Init_CacheCoherency} @asil{D}
 * If hardware cache coherency is not guaranteed, @c cache_invalidate and
 * @c sivc_cache_flush cache management callbacks must be supplied.
 *
 * @restriction @id{Init_AtMostOnce} @asil{D}
 * @c sivc_init shall be called at most once unless both endpoints do
 * coordinated re-initialization. For example, on SC7 resume peers may call
 * @c sivc_init to restart communication, but it must be done on both sides and
 * before calling any other IVC function.
 *
 * @restriction @id{Init_TheFirst} @asil{D}
 * @c sivc_init must be called before invoking any other IVC function on the
 * endpoint.
 *
 * @outcome @id{Init_Error} @asil{D}
 * If @c sivc_init cannot successfully performs endpoint initialization, it
 * must return error.
 *
 * @outcome @id{Init_ZeroedSharedMemory} @asil{B}
 * When shared memory is initialized with zeros and @c sivc_init successfully
 * performs endpoint initialization, the endpoint must be in @c Established
 * state.
 */
int sivc_init(struct sivc_queue* queue,
        uintptr_t recv_base, uintptr_t send_base,
        uint32_t nframes, uint32_t frame_size, sivc_notify_function notify,
        sivc_cache_invalidate_function cache_invalidate,
        sivc_cache_flush_function sivc_cache_flush);

/** @} */

#ifdef __cplusplus
} // extern "C"
} // namespace Ivc
#endif

#endif // SIVC_INSTANCE_H
