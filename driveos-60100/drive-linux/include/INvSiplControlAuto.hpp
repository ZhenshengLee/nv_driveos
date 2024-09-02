/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    /** @brief Function to process auto (AE-AWB) algorithm
     *
     * @note This is plugin API that is called by SIPL at run-time for processing auto(AE-AWB) algorithm.
     *
     * @pre None.
     *
     * @param[in]  inParams  Sipl Control Auto input parameters.
     *                       Valid range: See @ref SiplControlAutoInputParam
     * @param[out] outParam  Sipl Control Auto output parameters.
     *
     * @retval NVSIPL_STATUS_OK on successful completion.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT if input parameters are invalid (range check failure).
     * @retval (SIPLStatus) if there are other errors.
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

    /** @brief Function to get noise profile from captured frame embedded data.
     *
     * @note This is plugin API that is called by SIPL for getting noise profile for the captured frame.
     *
     * @pre None.
     *
     * @param[in]  currFrameEmbedInfo          Captured frame embedded information.
     *                                         Valid range: See @ref SiplControlEmbedInfo
     * @param[in]  maxSupportedNoiseProfiles   Maximum number of supported noise profiles.
     *                                         Valid range:[1, 32].
     * @param[out] noiseProfile                captured frame noise profile.
     *
     * @retval NVSIPL_STATUS_OK on successful completion.
     * @retval NVSIPL_STATUS_BAD_ARGUMENT if input parameters are invalid.
     * @retval (SIPLStatus) if there are other errors.
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
       static_cast<void>(currFrameEmbedInfo);
       static_cast<void>(maxSupportedNoiseProfiles);
       noiseProfile = 0U;
       return NVSIPL_STATUS_OK;
    }

    /**
     * @brief Function to reset to the state right after initialization.
     *
     * @note This is plugin API that is called by SIPL to reset SIPL Control Auto
     * to the state right after initialization.
     *
     * @pre None.
     *
     * @retval NVSIPL_STATUS_OK on successful completion.
     * @retval (SIPLStatus) if there are other errors.
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

    /** @brief SIPL Control Auto Interface class default destructor. */
    virtual ~ISiplControlAuto() = default;

protected:
    /** @brief SIPL Control Auto Interface class default constructor. */
    ISiplControlAuto() = default;

private:

    /** @brief disable SIPL Control Auto Interface class copy constructor */
    ISiplControlAuto(ISiplControlAuto const &) = delete;
    /** @brief disable SIPL Control Auto Interface class copy assignment operation */
    ISiplControlAuto& operator= (ISiplControlAuto const &) & = delete;

    /** @brief disable SIPL Control Auto Interface class move constructor */
    ISiplControlAuto(ISiplControlAuto &&) = delete;
    /** @brief disable SIPL Control Auto Interface class move assignment operation */
    ISiplControlAuto& operator= (ISiplControlAuto &&) & = delete;
};

/** @} */

} // namespace nvsipl

#endif // NVSIPLCONTROLAUTOINTERFACE_HPP
