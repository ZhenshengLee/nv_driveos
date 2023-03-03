/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef DESER_DRIVERS_EXPORT_HPP
#define DESER_DRIVERS_EXPORT_HPP

/*VCAST_DONT_INSTRUMENT_START*/
#include "DeserializerIF/CNvMDeserializer.hpp"
/*VCAST_DONT_INSTRUMENT_END*/
#include "sipl_error.h"
/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface: Deserializer Export </b>
 *
 */

namespace nvsipl {

/** @defgroup ddi_deserializer_export Deserializer Export API
 *
 * @brief Provides interfaces for the Deserializer Export.
 *
 * @ingroup ddi_api_grp
 * @{
 */

    /**
     * @brief Get a newly allocated deserializer object, as implemented by the
     * driver library.
     *
     * The returned pointer should be allocated in the default heap, as it will
     * be freed using `delete`.
     *
     * @retval (CNvMDeserializer*) An object implementing the \ref
     *                             CNvMDeserializer interface on success.
     * @retval nullptr             On bad allocation or other failure.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    extern "C" CNvMDeserializer*
    CNvMDeserializer_Create();

    /**
     * @brief Gets a null terminated C-style string containing the name of the
     * deserializer.
     *
     * The returned string is expected to be valid at any time, so it should
     * be a string constant and not dynamically allocated.
     *
     * @retval (char_t*)      A null-terminated C-style string containing the name of
     *                        deserializer device.
     * @retval nullptr        On bad allocation or other failure.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    extern "C" const char_t**
    CNvMDeserializer_GetName();

/** @} */

}

#endif /* DESER_DRIVERS_EXPORT_HPP */
