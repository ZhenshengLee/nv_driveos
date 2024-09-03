/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IINTERRUPTNOTIFY_HPP
#define IINTERRUPTNOTIFY_HPP

#include "NvSIPLInterrupts.hpp"

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Interrupt Notify </b>
 *
 */

namespace nvsipl
{

/** @defgroup ddi_interrupt_notify Interrupt Notify API
 *
 * @brief Provides interfaces for Interrupt Notify.
 *
 * @deprecated This callback-based interface is to be deprecated in a future
 * release in favour of a direct return interface. (Jira CAMERASW-11822.)
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * Interface defining Interrupt Notify functionality
 */
class IInterruptNotify
{
public:

    /**
     * @brief Callback function to dispatch a notification to the client queue.
     *
     * The @ref NvSIPLPipelineNotifier::NotificationType attributed is
     * automatically decided based the value of the @ref code parameter.
     *
     * @param[in]   code        The interrupt code to send to the client in the
     *                          notification. Valid values are defined in
     *                          @ref InterruptCode.
     * @param[in]   data        Fixed-sized storage for custom driver-defined
     *                          payload. Valid range is [0, ULLONG_MAX].
     * @param[in]   gpioIdx     The interrupt CDAC GPIO index.
     *                          Valid range is [0, UINT_MAX].
     * @param[in]   linkMask    The one-hot link masks indicating which Camera
     *                          Module link(s) this notification describes.
     *
     * @retval  NVSIPL_STATUS_OK on completion
     * @retval  NVSIPL_STATUS_BAD_ARGUMENT if the linkMask has bits set above
     *          @ref MAX_DEVICEBLOCKS_PER_PLATFORM
     * @retval  (SIPLStatus) other propagated error
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Re-entrant: No
     *   - Async/Sync: Async
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as
     *     mentioned in the NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus Notify(const InterruptCode code,
                             uint64_t const data,
                             uint32_t const gpioIdx,
                             uint32_t const linkMask) = 0;

protected:

    /**
     * @brief Constructor.
     */
    IInterruptNotify() = default;

    /**
     * @brief Prevent IInterruptNotify from being copy constructed.
     */
    IInterruptNotify(IInterruptNotify const &) = delete;

    /**
     * @brief Prevent IInterruptNotify from being move constructed.
     */
    IInterruptNotify(IInterruptNotify &&) = delete;

    /**
     * @brief Prevent IInterruptNotify from being copy assigned.
     */
    IInterruptNotify& operator=(IInterruptNotify const &) & = delete;

    /**
     * @brief Prevent IInterruptNotify from being move assigned.
     */
    IInterruptNotify& operator=(IInterruptNotify&&) & = delete;

    /**
     * @brief destructor 
     */
    virtual ~IInterruptNotify() = default;

};

/** @} */

} // end of namespace nvsipl

#endif // IINTERRUPTNOTIFY_HPP
