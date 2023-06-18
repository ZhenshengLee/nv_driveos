/*
   * Copyright (c) 2020-2022 NVIDIA Corporation.  All rights reserved.
   *
   * NVIDIA Corporation and its licensors retain all intellectual property
   * and proprietary rights in and to this software and related documentation
   * and any modifications thereto.  Any use, reproduction, disclosure or
   * distribution of this software and related documentation without an express
   * license agreement from NVIDIA Corporation is strictly prohibited.
   */

/* NVIDIA SIPL Control Auto Interface */

#ifndef NVSIPLCONTROLAUTOINTERFACE_HPP
#define NVSIPLCONTROLAUTOINTERFACE_HPP


#include "NvSIPLCommon.hpp"
#include "NvSiplControlAutoDef.hpp"

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Auto Control Interface - @ref NvSIPLAutoControl </b>
 *
 */

/** @defgroup NvSIPLAutoControl SIPL Auto Control
 *
 * @brief Describes interfaces for SIPL Auto Control implementation.
 *
 * @ingroup NvSIPL */

namespace nvsipl
{

/** @ingroup NvSIPLAutoControl
 * @{
 */

/** @class ISiplControlAuto INvSiplControlAuto.hpp
  *
  * @brief Defines SIPL Control Auto Interface Class
  */
class ISiplControlAuto {

public:
    /** @brief Function to process auto (AE/AWB) algorithm
     *
     * This is plugin API that is called by SIPL.
     *
     * @pre None.
     *
     * @param[in]  inParams Sipl Control Auto input parameters. Valid range: See @ref SiplControlAutoInputParam
     * @param[out] outParam Sipl Control Auto output parameters.
     *
     * @returns
     * - NVSIPL_STATUS_OK: on successful completion
     * - NVSIPL_STATUS_BAD_ARGUMENT: an error status if input parameters are invalid.
     * - NVSIPL_STATUS_NOT_INITIALIZED: an error status if not initialized.
     * - NVSIPL_STATUS_ERROR: an error status if there are other errors.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes, with the following conditions:
     *     - Provided the same instance is not used on multiple threads at the same time.
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus Process(const SiplControlAutoInputParam& inParams,
                               SiplControlAutoOutputParam& outParams) = 0;

    /** @brief Function to get captured frame noise profile
     *
     * This is plugin API that is called by SIPL.
     *
     * @pre None.
     *
     * @param[in]  currFrameEmbedInfo          Captured frame embedded information.
     *                                         Valid range: See @ref SiplControlEmbedInfo
     * @param[in]  maxSupportedNoiseProfiles   Maximum number of supported noise profiles.
     *                                         Supported range:[1, 32].
     * @param[out] noiseProfile output noise profile for captured frame ISP processing
     *
     * @returns
     * - NVSIPL_STATUS_OK: on successful completion
     * - NVSIPL_STATUS_NOT_INITIALIZED: if object is not initialized.
     * - NVSIPL_STATUS_BAD_ARGUMENT: an error status if input parameters are invalid.
     * - NVSIPL_STATUS_ERROR: an error status if there are other errors.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes, with the following conditions:
     *     - Provided the same instance is not used on multiple threads at the same time.
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus GetNoiseProfile(const SiplControlEmbedInfo& currFrameEmbedInfo,
                                       const uint32_t maxSupportedNoiseProfiles,
                                       uint32_t& noiseProfile) {
       noiseProfile = 0U;
       return NVSIPL_STATUS_OK;
    }

    /**
     * @brief Function to reset to state right after initialization.
     *
     * This is plugin API that is called by SIPL.
     *
     * @pre None.
     *
     * @returns
     * - NVSIPL_STATUS_OK: on successful completion
     * - NVSIPL_STATUS_NOT_INITIALIZED: an error status if not initialized.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes, with the following conditions:
     *     - Provided the same instance is not used on multiple threads at the same time.
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus Reset() {
       return NVSIPL_STATUS_NOT_SUPPORTED;
    }

    /** @brief Default destructor. */
    virtual ~ISiplControlAuto() = default;

protected:
    /** @brief Default constructor. */
    ISiplControlAuto() = default;

private:

    /** @brief disable copy constructor */
    ISiplControlAuto(const ISiplControlAuto&) = delete;
    /** @brief disable assignment operation */
    ISiplControlAuto& operator= (const ISiplControlAuto&) = delete;
};

/** @} */

} // namespace nvsipl

#endif // NVSIPLCONTROLAUTOINTERFACE_HPP
