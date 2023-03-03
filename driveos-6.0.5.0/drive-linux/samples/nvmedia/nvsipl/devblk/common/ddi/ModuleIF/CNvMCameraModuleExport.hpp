/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CAMERA_MODULE_EXPORT_HPP
#define CAMERA_MODULE_EXPORT_HPP

#include "NvSIPLDeviceBlockTrace.hpp"
#include "CNvMCameraModule.hpp"
/*VCAST_DONT_INSTRUMENT_START*/
#include "sipl_error.h"
/*VCAST_DONT_INSTRUMENT_END*/

/**
 * @file
 *
 * @brief <b> Camera Device Driver Interface : Camera Module Export API</b>
 *
 */

namespace nvsipl {

/** @defgroup ddi_cam_module_export Camera Module Export API
 *
 * @brief Provides interfaces for Camera Module Export.
 *
 * @ingroup ddi_api_grp
 * @{
 */

    /**
     * @brief Gets a newly allocated camera module object, as implemented by the
     * driver library.
     *
     * The returned pointer should be allocated in the default heap, as it will
     * be freed using `delete`.
     *
     * @retval nullptr             On bad allocation or other failure.
     * @retval (CNvMDeserializer*) An object implementing the \ref
     *                             CNvMCameraModule interface on success.
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
    extern "C" CNvMCameraModule*
    CNvMCameraModule_Create();

    /**
     * @brief Gets a list of camera modules which should use this camera module driver library.
     *
     * The returned pointer is expected to point to a null-terminated list of
     * null-terminated C-style strings, which remains valid for the life of the
     * program (and thus should be a constant).
     *
     * @retval (char_t**) A list of null-terminated C-style strings, containing supported
     *                    driver names.
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
    CNvMCameraModule_GetNames();

    /**
     * @brief Gets the version of the CNvMCameraModule API used by the library.
     *
     * The returned pointer is expected to remains valid for the life of the
     * program (and thus should point to a compile-time constant).
     *
     * @retval (CNvMCameraModule::Version*) A pointer to camera module version.
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
    extern "C" const CNvMCameraModule::Version*
    CNvMCameraModule_GetVersion();

    /**
     * @brief Set the debug level in the module driver library.
     *
     * Driver libraries should use this value to enable or disable log messages
     * based on the provided trace level.
     *
     * @param[in] level    enum which defines tracing/logging levels \ref TraceLevel.
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
    extern "C" SIPLStatus
    CNvMCameraModule_SetDebugLevel(INvSIPLDeviceBlockTrace::TraceLevel const level);

/** @} */

}

#endif /* CAMERA_MODULE_EXPORT_HPP */
