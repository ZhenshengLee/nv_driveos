/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef IINTERRUPTSTATUS_HPP
#define IINTERRUPTSTATUS_HPP

#include <array>

#include "NvSIPLInterrupts.hpp"

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Interrupt Status </b>
 *
 */

namespace nvsipl
{

/** @defgroup ddi_interrupt_status Interrupt Status API
 *
 * @brief Provides interfaces for Interrupt Status.
 *
 * @ingroup ddi_api_grp
 * @{
 */

/**
 * @brief Maximum number of error notifications to dispatch to the client
 * per device group (i.e. Deserializer or Camera Module links) interrupt
 * event.
 */
static constexpr uint32_t MAX_DEVGRP_NTFNS_PER_INTR { 8U };

/**
 * Interface defining Interrupt Status functionality
 */
class IInterruptStatus
{
public:
    /**
     * @brief Interrupt Notification from hardware device drivers to be
     * delivered to the client.
     */
    struct InterruptNotification {
        /** The interrupt code to send to the client in the notification. */
        InterruptCode code;
        /** Fixed-sized storage for custom driver-defined payload. */
        uint64_t data;
        /** The interrupt CDAC GPIO index. */
        uint32_t gpioIdx;
        /**
         * The one-hot link masks indicating which Camera Module link(s) this
         * notification describes.
         */
        uint32_t linkMask;
        /**
         * Whether this interrupt notification is valid (filled). Ignored if set
         * to false.
         */
        bool valid;
    };

    /**
     * @brief Requests the driver to query the status of a device in response to
     * an asserted interrupt, and return a list of interrupt notifications to be
     * propagated to the client.
     *
     * If the device emits interrupt(s) muxed to the asserted GPIO pin, its
     * driver is expected to perform a brief and minimal query to the hardware
     * (e.g. to read select status registers) over the communication bus to
     * determine whether it is in a state of error.
     *
     * Locking and synchronization may be required in the driver, in order to
     * protect the context data and communication bus.
     *
     * If no error statuses or evidence of interrupt assertions were found in
     * the device(s), @ref NVSIPL_STATUS_OK should be returned unless there were
     * errors in the process of querying for this information.
     *
     * If neither @ref NVSIPL_STATUS_OK nor @ref NVSIPL_STATUS_NOT_SUPPORTED is
     * returned, then an error notification with code @ref
     * InterruptCode::INTR_STATUS_FAILURE is dispatched to the client and no
     * notifications in @ref intrNtfns are dispatched.
     *
     * @param[in]   gpioIdx     CDAC error GPIO index, of type @ref uint32_t.
     *                          Valid range is [0, UINT_MAX].
     * @param[out]  intrNtfns   List of interrupt notifications to propagate to
     *                          the client. The @ref valid parameter of each
     *                          element must be set to true, otherwise it is
     *                          considered empty.
     *
     *                          The list capacity is @ref
     *                          MAX_DEVGRP_NTFNS_PER_INTR.
     *
     * @retval  NVSIPL_STATUS_OK on completion
     * @retval  NVSIPL_STATUS_NOT_SUPPORTED on invalid configuration
     * @retval  (SIPLStatus) other propagated error
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Re-entrant: No
     *   - Async/Sync: Sync
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
    virtual SIPLStatus GetInterruptStatus(
        uint32_t const gpioIdx,
        std::array<InterruptNotification, MAX_DEVGRP_NTFNS_PER_INTR> &intrNtfns) const
    {
        static_cast<void>(gpioIdx);
        static_cast<void>(intrNtfns);
        return NVSIPL_STATUS_NOT_SUPPORTED;
    }

protected:
    /**
     * @brief Define default constructor.
     */
    IInterruptStatus() =  default;
    /**
     * @brief Define default destructor.
     */
    virtual ~IInterruptStatus() = default;
    /**
     * @brief Prevent IInterruptStatus from being copy constructed.
     */
    IInterruptStatus(const IInterruptStatus& status) = delete;
    /**
     * @brief Prevent IInterruptStatus from being copy assigned.
     */
    IInterruptStatus& operator=(const IInterruptStatus& status) & = delete;
    /**
     * @brief Prevent IInterruptStatus from being move constructed.
     */
    IInterruptStatus(IInterruptStatus&& status) = delete;
    /**
     * @brief Prevent IInterruptStatus from being move assigned.
     */
    IInterruptStatus& operator=(IInterruptStatus&& status) & = delete;

};

/**
 * @brief Empty (invalid) @ref InterruptNotification.
 */
static constexpr struct IInterruptStatus::InterruptNotification EMPTY_INTR_NTFN
    { InterruptCode::INTR_STATUS_FAILURE, 0ULL, 0U, 0U, false };

/** @} */

} // end of namespace nvsipl

#endif // IINTERRUPTSTATUS_HPP