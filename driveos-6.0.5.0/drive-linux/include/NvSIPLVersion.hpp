/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVSIPLVERSION_HPP
#define NVSIPLVERSION_HPP

#include <cstdint>

namespace nvsipl
{

/**
 *  @defgroup NvSIPLVersion NvSIPL Version
 *
 * Holds the version information for @ref NvSIPLCamera_API and
 * @ref NvSIPLClient_API.
 *
 * @ingroup NvSIPLCamera_API
 * @{
 */

/** @brief Holds the version information of @ref NvSIPLCamera_API and @ref NvSIPLClient_API.
*
* @par
* Version history of @ref NvSIPLCamera_API and @ref NvSIPLClient_API
* @par
* Version <b> 1.0.0 </b>
*/
struct NvSIPLVersion
{
    uint32_t uMajor; /**< Holds the major revision. */
    uint32_t uMinor; /**< Holds the minor revision. */
    uint32_t uPatch; /**< Holds the patch revision. */
};

constexpr uint32_t NVSIPL_MAJOR_VER = 1U; /**< Indicates the major revision. */
constexpr uint32_t NVSIPL_MINOR_VER = 0U; /**< Indicates the minor revision. */
constexpr uint32_t NVSIPL_PATCH_VER = 0U; /**< Indicates the patch revision. */

/** @brief Returns the version of the SIPL library.
 *
 * This API does the following:
 *  - Populates the version of SIPL from NVSIPL_MAJOR_VER for the major version,
 *    NVSIPL_MINOR_VER for the minor version, and NVSIPL_PATCH_VER for the patch version.
 *
 * @param[out] rVersion A reference to the object to copy the version information.
 */
void NvSIPLGetVersion(NvSIPLVersion& rVersion);

/** @} */

}  // namespace nvsipl

#endif // NVSIPLVERSION_HPP
