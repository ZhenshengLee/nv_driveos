/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef IINTERRUPTSTATUS_HPP
#define IINTERRUPTSTATUS_HPP

#include "NvSIPLInterrupts.hpp"
#include "IInterruptNotify.hpp"

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
 * Interface defining Interrupt Status functionality
 */
class IInterruptStatus
{
public:

    /**
     * @brief Requests the driver to query the status of a device in response to
     * an asserted interrupt, it may issue notifications to the client directly
     * through the provided callback interface.
     *
     * If the device emits interrupt(s) muxed to the asserted GPIO pin, its
     * driver is expected to perform a brief and minimal query to the hardware
     * (e.g. to read select status registers) over the communication bus to
     * determine whether it is in a state of error.
     *
     * Locking and synchronization may be required in the driver, in order to
     * protect the context and communication bus.
     *
     * The function execution may be terminated by the caller if its runtime
     * exceeds the configured timeout duration, in order to meet FDTI or other
     * timing deadlines.
     *
     * @param[in]   gpioIdx         CDAC error GPIO index, of type @ref
     *                              uint32_t. Valid range is [0, UINT_MAX].
     * @param[in]   intrNotifier    Callback interface to dispatch an interrupt
     *                              notification to the client.
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
        const uint32_t gpioIdx,
        IInterruptNotify &intrNotifier) = 0;

protected:
    IInterruptStatus() =  default;
    virtual ~IInterruptStatus() = default;

};

/** @} */

} // end of namespace nvsipl

#endif // IINTERRUPTSTATUS_HPP
