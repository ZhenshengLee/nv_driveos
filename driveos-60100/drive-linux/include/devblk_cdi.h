/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * \brief <b> NVIDIA Device Block Interface: Camera Device Interface (CDI)</b>
 *
 * This file contains the Camera Device Interface API.
 */

#ifndef DEVBLK_CDI_H
#define DEVBLK_CDI_H

#include "nvmedia_core.h"
#include "NvSIPLCapStructs.h"
#include "NvSIPLCDICommon.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup devblk_cdi_api Camera Device Interface
 *
 * The Camera Device Interface API encompasses all DevBlk I2C control related
 * functions, including programming of all I2C controlled components
 * such as deserializers, serializers, EEPROMs, and image sensors.
 *
 * DevBlkCDI needs a device driver for each attached device. This provides the
 * flexibility of adding new devices easily.
 *
 * @ingroup cdi_api_grp
 * @{
 */

/**
 * \defgroup devblk_cdi_types Basic CDI Types
 * The Camera Device Interface API provides common CDI processing functions.
 * @ingroup basic_api_top
 *
 * @{
 */

/** \brief Device address to use for an CDI simulator device. Select the
 * \ref DEVBLK_CDI_I2C_SIMULATOR port to use simulator mode for all devices.
 */
#define DEVBLK_CDI_SIMULATOR_ADDRESS   0xFF1U
/** Bits reserved for the  I2C bus number in \ref CDI_RDEV_CFG(csi, i2c). */
#define RDEV_CFG_I2C_BITS           8U
/** Bits reserved for the CSI port in \ref CDI_SLV_RDEV_CFG(csi, i2c). */
#define RDEV_CFG_CSI_BITS           (RDEV_CFG_I2C_BITS + 8U)
/** Bit reserved for the passive mode flag in \ref CDI_SLV_RDEV_CFG(csi, i2c). */
#define RDEV_CFG_SLV_BIT            (RDEV_CFG_CSI_BITS + 1U)

/** \brief Macro to create root device configuration with the connected CSI port
 * and I2C bus.
 * \param[in] csi The CSI port number defined in \ref NvSiplCapInterfaceType.
 * \param[in] i2c The I2C bus number defined in \ref DevBlkCDI_I2CPort.
 */
#define CDI_RDEV_CFG(csi, i2c)   (((uint32_t)(csi) << (uint32_t)RDEV_CFG_I2C_BITS) | (i2c))

/**
 * \brief  Extended macro to create root device configuration with the connected
 *  CSI port, I2C bus, and an option to disable power control from root device.
 *
 * \param[in] csi        The CSI port number defined in
 *                       \ref NvSiplCapInterfaceType.
 * \param[in] i2c        The I2C bus number defined in \ref DevBlkCDI_I2CPort.
 * \param[in] disPwrCtrl A flag to disable power control. Value must be 0
 *                       (root device turns on power for devices in
 *                       DevBlkCDIRootDeviceCreate() and turns off power
 *                       for devices in DevBlkCDIRootDeviceDestroy())
 *                       or 1 (root device does not control power for devices
 *                       in %DevBlkCDIRootDeviceCreate() and
 *                       %DevBlkCDIRootDeviceDestroy()).
 *                       Valid range: [0, 1].
 */
#define CDI_RDEV_CFG_EX(csi, i2c, disPwrCtrl) \
         ((i2c & 0xffU) | \
          ((uint32_t)(csi & 0xffU) << (uint32_t)RDEV_CFG_I2C_BITS) | \
          ((uint32_t)(disPwrCtrl & 1U) << (uint32_t)RDEV_CFG_SLV_BIT))

/**
 * \brief  Macro to create a passive root device configuration with the
 *  connected CSI port and I2C bus when the application is run on a passive SoC.
 *
 * \param[in] csi   The CSI port number defined in \ref NvSiplCapInterfaceType.
 * \param[in] i2c   The I2C bus number defined in \ref DevBlkCDI_I2CPort.
 */
#define CDI_SLV_RDEV_CFG(csi, i2c) \
        ((i2c) | ((uint32_t)(csi) << RDEV_CFG_I2C_BITS) | ((uint32_t)(1U) << RDEV_CFG_CSI_BITS))


struct DevBlkCDISensorControl;

struct DevBlkCDIEmbeddedDataChunk;

struct DevBlkCDIEmbeddedDataInfo;

struct DevBlkCDISensorAttributes;

#if !NV_IS_SAFETY
struct DevBlkCDIModuleConfig;
#endif

/** @} */ /* Ends Basic CDI Types group */

/**
 * \brief  Holds the handles for an DevBlkCDIDevice object.
 */
typedef struct {
    void *deviceHandle;
    void *deviceDriverHandle;
} DevBlkCDIDevice;

/**
 * \defgroup cdi_root_device_api CDI Root Device
 *
 * Manage \ref DevBlkCDIRootDevice objects, which represent the root of the
 * SIPL device block.
 *
 * The DevBlkCDIRootDevice object manages an I2C port on the host hardware
 * device.
 * @{
 */

/**
 * \hideinitializer
 * \brief Defines the I2C buses on the host hardware device.
 */
typedef enum {
    DEVBLK_CDI_I2C_BUS_0 = 0, /**< Specifies i2c-0. */
    DEVBLK_CDI_I2C_BUS_1 = 1, /**< Specifies i2c-1. */
    DEVBLK_CDI_I2C_BUS_2 = 2, /**< Specifies i2c-2. */
    DEVBLK_CDI_I2C_BUS_3 = 3, /**< Specifies i2c-3. */
    DEVBLK_CDI_I2C_BUS_4 = 4, /**< Specifies i2c-4. */
    DEVBLK_CDI_I2C_BUS_5 = 5, /**< Specifies i2c-5. */
    DEVBLK_CDI_I2C_BUS_6 = 6, /**< Specifies i2c-6. */
    DEVBLK_CDI_I2C_BUS_7 = 7, /**< Specifies i2c-7. */
    DEVBLK_CDI_I2C_BUS_8 = 8, /**< Specifies i2c-8. */
    DEVBLK_CDI_I2C_BUS_9 = 9, /**< Specifies i2c-9. */
    DEVBLK_CDI_I2C_BUS_10 = 10, /**< Specifies i2c-10. */
    DEVBLK_CDI_I2C_BUS_11 = 11, /**< Specifies i2c-11. */
    DEVBLK_CDI_I2C_SIMULATOR = 255, /**< Port SIMULATOR (20) */
} DevBlkCDI_I2CPort;

/**
 * \brief The maximum number of GPIOs supported on a CDI Root Device.
 */
#define DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS (16U)

/**
 * \brief The maximum number of power links per device block.
 */
#define MAX_POWER_LINKS_PER_BLOCK (4U)

/**
 * \brief Structure to hold power control information
 */
typedef struct {
    /** \brief Power control method */
    uint8_t method;
    /**
     * \brief Power controller i2c slave address.
     */
    uint8_t i2cAddr;
    /**
     * \brief Camera module to power controller link mapping.
     *        Valid range: [0, \ref MAX_POWER_LINKS_PER_BLOCK - 1].
     */
    uint8_t links[MAX_POWER_LINKS_PER_BLOCK];
} DevBlkCDIPowerControlInfo;

/**
 * \brief Structure to describe a RAW image buffer
 */
typedef struct {
    /**
     * \brief A CPU pointer to an image buffer. Same as imageDataDMA
     *        Valid value: non-NULL.
     */
    uint8_t *imageData;
    /**
     * \brief An image buffer address which can be used for direct
     * crypto HW access. Same as imageData.
     *        Valid value: non-0.
     */
    uint64_t imageDataDMA;
    /** \brief Length of a single line of an image (with any padding),
     *         in bytes.
     *         The pitch must be > 0.
     */
    uint32_t pitch;
    /**
     * \brief Total height of an image, including top and/or bottom
     *        embedded data (if embedded data is put into the same
     *        buffer).
     *        The height must be > 0
     */
    uint32_t height;
    /**
     * \brief Pointer to start of a top embedded data, if enabled
     *        on a sensor.
     *        Only makes sense if embedded data is put into a buffer
     *        separate from imageData buffer.
     *        Valid value: [non-NULL] if top embedded data is active.
     */
    uint8_t const *embDataTop;
    /**
     * \brief Size of a top embedded data in bytes, if enabled.
     *        The embedded data len must be >= 0.
     */
    uint32_t embSizeTop;
    /**
     * \brief Pointer to start of a bottom embedded data, if enabled
     *        on a sensor.
     *        Only makes sense if embedded data is put into a buffer
     *        separate from imageData buffer.
     *        Valid value: [non-NULL] if bottom embedded data is active.
     */
    uint8_t const *embDataBottom;
    /**
     * \brief Size of a bottom embedded data in bytes, if enabled.
     *        The embedded data len must be >= 0
     */
    uint32_t embSizeBottom;
} DevBlkImageDesc;

/**
 * \brief CDI codes for CAMGPIO GPIO Levels.
 *
 * \ingroup cdi_gpio_levels
 *
 * @{
 */
#define DEVBLK_CDI_GPIO_LEVEL_LOW     (1U)
#define DEVBLK_CDI_GPIO_LEVEL_HIGH    (2U)
/** @} */

/**
 * \brief CDI codes for CAMGPIO Interrupt Events.
 */
typedef enum {
    /** An interrupt has occurred. */
    DEVBLK_CDI_GPIO_EVENT_INTR = 0,
    /** An interrupt timeout period has elapsed. */
    DEVBLK_CDI_GPIO_EVENT_INTR_TIMEOUT,
    /**
     * An error occurred in CAMGPIO code, potentially resulting in permanent loss
     * of functionality. (Error)
     */
    DEVBLK_CDI_GPIO_EVENT_ERROR_CAMGPIO,
    /**
     * An error occurred in backend code, potentially resulting in permanent
     * loss of functionality. (Error)
     */
    DEVBLK_CDI_GPIO_EVENT_ERROR_BACKEND,
    /**
     * A generic error occurred, potentially resulting in permanent loss of
     * functionality. (Error)
     */
    DEVBLK_CDI_GPIO_EVENT_ERROR_UNKNOWN,
} DevBlkCDIGpioEvent;

/**
 * \brief Structure to hold array of CAMGPIO indices.
 */
typedef struct {
    /**
     * List of CAMGPIO pins.
     * @anon_struct
    */
    struct {
        /**
         * \brief CAMGPIO index. Range: [0,UINT_MAX]
         * @anon_struct_member
         */
        uint32_t idx;
        /**
         * \brief Error localization timeout duration [ms], 0 to disable.
         * Range: [0,UINT_MAX]
         * @anon_struct_member
         */
        uint32_t timeout_ms;
        /**
         * \brief CAMGPIO index
         * @anon_struct_member
         */
        DevBlkCDIGpioEvent evt;
    } pins[DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS];
    /**
     * \brief Number of items in indices in @ref pins.
     *        Valid range: [0, \ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS].
     */
    uint32_t count;
} DevBlkCDIGPIOIndices;

/**
 * \brief  An opaque handle for an DevBlkCDIRootDevice object.
 */
typedef void DevBlkCDIRootDevice;

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Creates an @ref DevBlkCDIRootDevice object for a root device.
 *
 * This API does the following:
 * -# Verifies the CSI/I2C ports information.
 * -# Creates and initializes the device context for the root device.
 * -# Create platform context and clears its memory by calling @ref NvOsAlloc and @ref NvOsMemset
 *    and performs initialization.
 * -# Returns the handle of the root device, or NULL if error occurred.
 *
 * @pre None.
 *
 * @param[in] portCfg
 *          The CSI/I2C ports that this root device use. It is strongly
 *          recommended that you use the macro @ref CDI_RDEV_CFG(csi, i2c) or
 *          @ref CDI_RDEV_CFG_EX(csi, i2c, disPwrCtrl) to
 *          generate this value. If the application runs on a passive SoC while
 *          a master SoC controls CDI devices, use CDI_SLV_RDEV_CFG(csi, i2c).
 *          Supported I2C are:
 *          - @ref DEVBLK_CDI_I2C_BUS_0
 *          - @ref DEVBLK_CDI_I2C_BUS_1
 *          - @ref DEVBLK_CDI_I2C_BUS_2
 *          - @ref DEVBLK_CDI_I2C_BUS_3
 *          - @ref DEVBLK_CDI_I2C_BUS_4
 *          - @ref DEVBLK_CDI_I2C_BUS_5
 *          - @ref DEVBLK_CDI_I2C_BUS_6
 *          - @ref DEVBLK_CDI_I2C_BUS_7
 *          - @ref DEVBLK_CDI_I2C_BUS_8
 *          - @ref DEVBLK_CDI_I2C_BUS_9
 *          - @ref DEVBLK_CDI_I2C_BUS_10
 *          - @ref DEVBLK_CDI_I2C_BUS_11
 *          - @ref DEVBLK_CDI_I2C_SIMULATOR
 *
 * Supported CSI ports are:
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_A
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_B
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_C
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_D
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_CD
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_F
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_G
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_H
 *          - @ref NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_GH
 * @param[in] gpios
 *          A @ref DevBlkCDIGPIOIndices structure that lists
 *          all the CAMGPIO indices; Valid Range: [0, @ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS - 1].
 *          This field is only used when @a useCDIv2API is set to true.
 * @param[in] useCDIv2API
 *          Holds a flag to indicate which version of the CDI API to use.
 *          false indicates version 1, while true indicates version 2;
 *          For all safety use cases, this flag must set to true to
 *          use CDI version 2 API.
 * @retval  The new root device's handle if successful
 * @retval  NULL if error occurred.
 *
 * \usage
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
 *
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
DevBlkCDIRootDevice *
DevBlkCDIRootDeviceCreate(
    uint32_t portCfg,
    DevBlkCDIGPIOIndices gpios,
    const bool useCDIv2API
);
/// @endcond

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Destroys an @ref DevBlkCDIRootDevice object for a root device.
 *
 * This API does the following:
 * -# Release all platform-specific resources.
 * -# Releases all the resources allocated when the root device is created.
 * -# Frees the root device by calling @ref NvOsFree.
 *
 * @pre None
 *
 * @param[in] device A pointer to the root device to be destroyed;
 *                   Valid range: [non-NULL].
 *
 * \usage
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
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
void
DevBlkCDIRootDeviceDestroy(
    DevBlkCDIRootDevice *device
);
/// @endcond

/**
 * @brief Queries the logic level of a Tegra GPIO input pin associated with the
 * root device.
 *
 * This API does the following:
 * -# Verifies the input parameters. Makes sure a memory is allocated to
 *    hold the return GPIO pin level.
 * -# Retrieve GPIO pin level.
 *
 * @pre None
 *
 * @param[in]   device      A pointer to the root device to use; Valid value: [non-NULL].
 * @param[in]   gpio_idx    The index of the GPIO pin; Valid range: [0, @ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS - 1].
 *                          The range of @a gpio_idx will be checked in the supporting function.
 * @param[out]  level       The GPIO pin @a level queried from CamGPIO;
 *                          The pin @a level should be @ref DEVBLK_CDI_GPIO_LEVEL_LOW or
 *                          @ref DEVBLK_CDI_GPIO_LEVEL_HIGH.
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that parameter(s) were invalid.
 * @retval  NVMEDIA_STATUS_ERROR         Indicate that some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the API is not
 *          supported on the current platform
 *
 * \usage
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIRootDeviceGetGPIOPinLevel(
    DevBlkCDIRootDevice const *device,
    uint32_t gpio_idx,
    uint32_t *level
);

/**
 * @brief Compares the logic level of a Tegra GPIO input pin associated with the
 * root device with expected pin level.
 *
 * This API does the following:
 * -# Retrieves GPIO's current pin level by Calling @ref DevBlkCDIRootDeviceGetGPIOPinLevel
 * which also helps verify the input parameters.
 * -# Compares the retrieved GPIO pin level with expected pin level.
 *
 * @pre None
 *
 * @param[in]   device          A pointer to the root device to use;
 *                              Valid value: [non-NULL].
 * @param[in]   gpio_idx        The index of the GPIO pin; Valid range:
 *                              [0, @ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS - 1].
 *                              The range of @a gpio_idx will be checked in the supporting function.
 * @param[in]   expected_level  The expected GPIO pin @a level of a Tegra GPIO input pin to compare
 *                              with. Valid range:
 *                              [ @ref DEVBLK_CDI_GPIO_LEVEL_LOW, @ref DEVBLK_CDI_GPIO_LEVEL_HIGH]
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that parameter(s) were invalid.
 * @retval  NVMEDIA_STATUS_ERROR         Indicate GPIO's current pin level did not match with the
 *                                       expected pin level or some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the API is not supported on the
 *                                       current platform
 *
 * \usage
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIRootDeviceCheckGPIOPinLevel(
    DevBlkCDIRootDevice const *device,
    uint32_t gpio_idx,
    uint32_t expected_level
);

/**
 * @brief Sets the logic level of Tegra GPIO output pin associated with this
 * root device.
 *
 * This API does the following:
 * -# Verifies the input parameters.
 * -# Set GPIO pin level.
 *
 * @pre None
 *
 * @param[in]   device      A pointer to the root device to use; Valid value: [non-NULL].
 * @param[in]   gpio_idx    The index of the GPIO pin; Valid range: [0, @ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS - 1].
 *                          The range of @a gpio_idx will be checked in the supporting function.
 * @param[in]   level       The GPIO pin @a level to be set in CamGPIO;
 *                          The pin @a level must be @ref DEVBLK_CDI_GPIO_LEVEL_LOW or
 *                          @ref DEVBLK_CDI_GPIO_LEVEL_HIGH.
 *                          The value of @a level will be checked in the supporting function.
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that parameter(s) were invalid.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that this API is not
 *          supported on the current platform.
 *
 * \usage
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIRootDeviceSetGPIOPinLevel(
    DevBlkCDIRootDevice const *device,
    uint32_t gpio_idx,
    uint32_t level
);

/**
 * @brief Verifies that the level of a GPIO pin configured as an interrupt is at
 * the correct pre-transition level, and then clears any pending interrupt event.
 *
 * This API does the following:
 * -# Verifies the input parameters.
 * -# Checks interrupt pin.
 *
 * @pre None
 *
 * @param[in]   device      A pointer to the root device to use; Valid value: [non-NULL].
 * @param[in]   gpio_idx    The index of the GPIO pin; Valid range: [0, @ref DEVBLK_CDI_ROOT_DEVICE_MAX_GPIOS - 1].
 *                          The value of @a gpio_idx will be verified in the supporting function.
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that this API is not
 *          supported on the current platform.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that parameter(s) were invalid.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that some other error occurred.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: Yes
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
NvMediaStatus
DevBlkCDIRootDeviceCheckIntrPin(
    DevBlkCDIRootDevice const *device,
    uint32_t gpio_idx
);

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief On QNX, this function helps determine if pulse error reporting is supported.
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that pulse error reporting is supported
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that pulse error reporting is not supported
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: Yes, with the following conditions:
 *   - Grants: nonroot, allow
 *   - Abilities: public_channel
 *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
 *     NVIDIA DRIVE OS Safety Developer Guide
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
NvMediaStatus
DevBlkCDIRootDeviceIsPulseErrorSupported(void);
/// @endcond


/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Blocks waiting on a the pulse channel's error semaphore to post,
 * indicating an error.
 *
 * @pre None
 *
 * @param[in]       device  A pointer to the root device to use;
 *                          Valid value: [non-NULL].
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that @a device was NULL.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that this API is not
 *          supported on the current platform.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: Yes
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
 *
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
NvMediaStatus
DevBlkCDIRootDeviceWaitForPulseError(
    DevBlkCDIRootDevice const *device);
///  @endcond

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Waits until an error condition is reported by means of a CamGPIO
 * device node interrupt assertion or @ref DevBlkCDIRootDeviceAbortWaitForError() is called.
 *
 * Waits until an error condition is reported by means of a CamGPIO
 * device node interrupt assertion or @ref DevBlkCDIRootDeviceAbortWaitForError() is called.
 *
 * For safety use cases, application software shall first verify the successful
 * reception of camera error GPIO interrupt as a first step of programming the
 * deserializer. This can be implemented by calling
 * @ref DevBlkCDIRootDeviceWaitForError() followed by programming the
 * deserializer to toggle the camera error GPIO pin which would cause
 * @ref DevBlkCDIRootDeviceWaitForError() to return.
 *
 * This API uses platform-specific supporting
 * function to wait for errors.
 *
 * @pre None
 *
 * @note The `gpios` structure is modified in every invocation of this
 *       function, therefore it must be reset by the caller before rearming.
 *
 * @param[in]       device  A pointer to the root device to use;
 *                          Valid value: [non-NULL].
 * @param[in,out]   gpios   A pointer to the list of CamGPIO indices to monitor, upon
 *                          unblocking, this list will only contain indices for
 *                          CamGPIO(s) on which interrupt event(s) were
 *                          triggered, if @ref NVMEDIA_STATUS_OK is returned;
 *                          Valid value: [non-NULL].
 *
 *                          @note If non-interrupt type GPIOs are specified,
 *                          this call must never unblock for them.
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the call was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that @a device was NULL.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that this API is not
 *          supported on the current platform.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: Yes
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
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
NvMediaStatus
DevBlkCDIRootDeviceWaitForError(
    DevBlkCDIRootDevice const *device,
    DevBlkCDIGPIOIndices *gpios
);
/// @endcond

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Aborts a call to DevBlkCDIRootDeviceWaitForError().
 *
 * This API does the following:
 * -# Verifies the input parameters.
 * -# Abort the wait process.
 * -# Reports error if @ref DevBlkCDIRootDeviceWaitForError() failed to
 *    abort after 2.5 seconds.
 *
 * @pre Should be called after @ref DevBlkCDIRootDeviceWaitForError
 *
 * @param[in] device A pointer to the root device to use;
 *                   Valid value: [non-NULL].
 * @retval  NVMEDIA_STATUS_OK            Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that @a device was NULL.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that some other error occurred.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that this API is not
 *          supported on the current platform.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: Yes, with the following conditions:
 *   - Grants: nonroot, allow
 *   - Abilities: public_channel
 *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
 *     NVIDIA DRIVE OS Safety Developer Guide
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
NvMediaStatus
DevBlkCDIRootDeviceAbortWaitForError(
    DevBlkCDIRootDevice const *device
);
/// @endcond
/**@} <!-- Ends cdi_root_device_api CDI Root Device --> */

/**
 * \defgroup cdi_device_driver_api CDI Device Driver
 *
 * Program elements related to \ref DevBlkCDIDeviceDriver, which defines a
 * device driver.
 * The core DevBlkCDI calls the driver when the client calls the related
 * public DevBlkCDI function.
 *
 * Before the client can create an DevBlkCDIDevice object (a device), it must
 * provide a device driver.
 * The DevBlkCDIDeviceDriver object contains the following data fields and
 * function pointers.
 *
 * @par Data Fields
 * \li deviceName The name of the device. This is a null-terminated string.
 * \li regLength  Target device offset length in bytes. Valid Range: [1, 2].
 * \li dataLength Target device data length in bytes. Valid Range: [1, 2].
 *
 * @par Function Pointers
 *  - DriverCreate (mandatory). Invoked when the client calls
 *    DevBlkCDIDeviceCreate().
 *  - DriverDestroy (mandatory). Invoked when the client calls
 *    DevBlkCDIDeviceDestroy().
 *  - GetModuleConfig (optional). Invoked when the client calls
 *    DevBlkCDIGetModuleConfig().
 *    Not supported for safety use cases.
 *  - ParseTopEmbDataInfo (optional). Invoked when the client calls
 *    DevBlkCDIParseTopEmbDataInfo().
 *  - ParseTopEmbDataInfo (optional). Invoked when the client calls
 *    DevBlkCDIParseTopEmbDataInfo().
 *  - SetSensorControls (optional). Invoked when the client calls
 *    DevBlkCDISetSensorControls().
 *  - GetSensorAttributes (optional). Invoked when the client calls
 *    DevBlkCDIGetSensorAttributes().
 *  - SetSensorCharMode (optional). Invoked when the client calls
 *    DevBlkCDISetSensorCharMode().
 *    Not supported for safety use cases.
 *  - ReadRegister (optional).
 *    Not supported for safety use cases.
 *  - WriteRegister (optional).
 *    Not supported for safety use cases.
 *
 * Here is a sample device driver implementation. The source file defines the
 * driver by creating an DevBlkCDIDeviceDriver struct and setting
 * its function pointers. The header file provides a function that retrieves
 * a pointer to the driver struct.
 *
 * @par Header File
 *
 * \code
 *
 * #include <devblk_cdi.h>
 *
 * DevBlkCDIDeviceDriver *GetSAMPLEDEVICEDriver(void);
 *
 * \endcode
 *
 * <b>Source File</b>
 * \code
 *
 * #include "cdi_sample_device.h"
 *
 * static NvMediaStatus
 * DriverCreate(
 *     DevBlkCDIDevice *handle,
 *     void *clientContext)
 * {
 *     if(!handle)
 *         return NVMEDIA_STATUS_BAD_PARAMETER;
 *
 *     Can be used to maintain local device context
 *     or can be set to NULL.
 *     handle->deviceDriverHandle = NULL;
 *
 *     return NVMEDIA_STATUS_OK;
 * }
 *
 * static DevBlkCDIDeviceDriver deviceDriver = {
 *     .deviceName = "Sample Sensor Device",
 *     .DriverCreate = DriverCreate
 * };
 *
 * DevBlkCDIDeviceDriver *
 * GetSAMPLEDEVICEDriver(void)
 * {
 *     Return device driver descriptor structure
 *     return &deviceDriver;
 * }
 *
 * \endcode
 *
 * @{
 */

/**
 * \brief  Holds device driver data.
 */
typedef struct {
    /** Holds the device name. */
    const char *deviceName;
    /** Holds the target device offset length in bytes. */
    uint32_t regLength;
    /** Holds the target device data length in bytes. */
    uint32_t dataLength;

    /**
     * \brief Holds the function that creates device driver
     *
     * The implementation should allocate and initialize the device driver
     * internal handle and module configuration.
     * `DriverCreate` should not communicate with the device, only set up
     * driver data structures.
     *
     * \pre @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *
     * \param[in]  handle A pointer to the device to use; Valid value: [non-NULL].
     * \param[in]  clientContext A non-NULL pointer to the device context to use when driver is created.
     *                           If NULL, the default context will be used in driver.
     * \retval NVMEDIA_STATUS_OK when all internal structures are initialized successfully.
     * \retval NVMEDIA_STATUS_OUT_OF_MEMORY when an internal allocation fails.
     * \retval NVMEDIA_STATUS_BAD_PARAMETER when `handle` is NULL.
     * \retval (NvMediaStatus) May return a different NvMedia error status for implementation-defined reasons.
     *
     * \usage
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
    NvMediaStatus (* DriverCreate)(
        DevBlkCDIDevice *handle,
        void const* clientContext);

    /**
     * \brief Holds the function that destroy the driver for a device
     *
     * The implementation should release all resources allocated in \ref DriverCreate().
     * It should not communicate with the device.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \param[in]  handle A pointer to the device to use; Valid value: [non-NULL].
     *
     * \retval NVMEDIA_STATUS_OK when all resources are successfully released.
     * \retval (NvMediaStatus) May return an NvMedia error status for implementation-defined reasons.
     *
     * \usage
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
     *   - Init: No
     *   - Runtime: No
     *   - De-Init: Yes
     */
    NvMediaStatus (* DriverDestroy)(
        DevBlkCDIDevice *handle);

    /**
     * \brief Holds the function that sets sensor controls
     *
     * This function is invoked by \ref DevBlkCDISetSensorControls function call.
     *
     * `sensrCtrlStructSize` can be used to determine the API version, as
     * its size will change on every revision. Drivers should use this to
     * ensure they are used only with compatible clients.
     *
     * The implementation may receive a `DevBlkCDISensorControl` which requests
     * more sensor contexts than are supported by the driver or sensor, in
     * which case an error should be returned and no sensor configuration modified.
     *
     * If supported by the device, drivers should set hold acquire during
     * device configuration.
     *
     * For each requested sensor context `i`, each of the following parameters should be programmed:
     *   - Exposure time, if `exposureControl[i].expTimeValid` is set
     *   - Exposure gain, if `exposureControl[i].gainValid` is set
     *   - White balance gain, if `wbControl[i].wbValid` is set
     *
     * If `frameReportControl.frameReportValid` is set, frame reporting should
     * be enabled with given configuration.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \param[in]  handle A pointer to the device to use; Valid value: [non-NULL].
     * \param[in]  sensorControl  A pointer to a sensor control structure for @a device;
     *                            Valid value: [non-NULL].
     * \param[in]  sensrCtrlStructSize Size of the @a sensorControl structure; The size must > 0.
     * \retval NVMEDIA_STATUS_OK when all operations succeed.
     * \retval (NvMediaStatus) May return an NvMedia error status for implementation-defined reasons.
     *
     * \usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
    NvMediaStatus (* SetSensorControls)(
        DevBlkCDIDevice const* handle,
        const struct DevBlkCDISensorControl *sensorControl,
        const size_t sensrCtrlStructSize);

    /**
     * \brief  Holds a pointer to the function that parses top section embedded data
     *  returned as part of a captured buffer.
     *
     * DevBlkCDIParseTopEmbDataInfo() invokes this function.
     *
     * The `embeddedDataChunkStructSize` and `dataInfoStructSize` parameters
     * can be used to validate that the client is using the same CDI version as
     * the driver.
     *
     * The provided `topEmbDataChunk` provides access to the top data chunks.
     * These should be used to fill out the information requested by `embeddedDataInfo`,
     * as supported by the sensor. Unsupported elements should have their
     * corresponding `valid` flag set to false. For example, if a given sensor
     * doesn't support reporting temperature information,
     * `embeddedDataInfo->sensorTempInfo->tempValid` should be set to false
     * after this function runs.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \param[in]  handle  A pointer to the device to use; Valid value: [non-NULL.
     * \param[in]  topEmbDataChunk  A pointer to the top sensor embedded data
     *                              \ref DevBlkCDIEmbeddedDataChunk structure.
     *                              Valid value: [non-NULL].
     * \param[in]  topChunkStructSize  Size of the @a topEmbDataChunk in bytes;
     *                                 The size must > 0.
     * \param[in]  embeddedDataInfo   A pointer to the buffer that holds the parsed embedded
     *                                data info. Valid value: [non-NULL].
     * \param[in]  dataInfoStructSize  Size of the @a embeddedDataInfo structure, in bytes;
     *                                 The size must > 0.
     * \retval NVMEDIA_STATUS_OK when all operations succeed.
     * \retval (NvMediaStatus) May return an NvMedia error status for implementation-defined reasons.
     *
     * \usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
    NvMediaStatus (* ParseTopEmbDataInfo)(
        DevBlkCDIDevice const* handle,
        const struct DevBlkCDIEmbeddedDataChunk *topEmbDataChunk,
        const size_t topChunkStructSize,
        struct DevBlkCDIEmbeddedDataInfo *embeddedDataInfo,
        const size_t dataInfoStructSize);

    /**
     * \brief  Holds a pointer to the function that parses bottom section embedded data
     *  returned as part of a captured buffer.
     *
     * DevBlkCDIParseBotEmbDataInfo() invokes this function.
     *
     * The `embeddedDataChunkStructSize` and `dataInfoStructSize` parameters
     * can be used to validate that the client is using the same CDI version as
     * the driver.
     *
     * The provided `botEmbDataChunk` provides access to the bottom data chunks.
     * These should be used to fill out the information requested by `embeddedDataInfo`,
     * as supported by the sensor. Unsupported elements should have their
     * corresponding `valid` flag set to false. For example, if a given sensor
     * doesn't support reporting temperature information,
     * `embeddedDataInfo->sensorTempInfo->tempValid` should be set to false
     * after this function runs.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \param[in]  handle  A pointer to the device to use; Valid value: [non-NULL.
     * \param[in]  botEmbDataChunk  A pointer to the bottom sensor embedded data
     *                              \ref DevBlkCDIEmbeddedDataChunk structure.
     *                              Valid value: [non-NULL].
     * \param[in]  botChunkStructSize  Size of the @a botEmbDataChunk in bytes;
     *                                 The size must > 0.
     * \param[in]  embeddedDataInfo   A pointer to the buffer that holds the parsed embedded
     *                                data info. Valid value: [non-NULL].
     * \param[in]  dataInfoStructSize  Size of the @a embeddedDataInfo structure, in bytes;
     *                                 The size must > 0.
     * \retval NVMEDIA_STATUS_OK when all operations succeed.
     * \retval (NvMediaStatus) May return an NvMedia error status for implementation-defined reasons.
     *
     * \usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
    NvMediaStatus (* ParseBotEmbDataInfo)(
        DevBlkCDIDevice const* handle,
        const struct DevBlkCDIEmbeddedDataChunk *botEmbDataChunk,
        const size_t botChunkStructSize,
        struct DevBlkCDIEmbeddedDataInfo *embeddedDataInfo,
        const size_t dataInfoStructSize);

    /** Holds the function that gets sensor attributes.
     *
     * The `sensorAttrStructSize` parameter can be used to check if the
     * application was built against the same version of CDI as the driver.
     *
     * The function should fill out the provided \ref DevBlkCDISensorAttributes
     * completely.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \param[in]  handle A pointer to the device to use; Valid value: [non-NULL].
     * \param[in]  sensorAttr A pointer to the \ref DevBlkCDISensorAttributes structure.
     *                        Valid value: [non-NULL].
     * \param[in]  sensorAttrStructSize Size of the @a sensorAttr structure, in bytes;
     *                                  The size must > 0.
     * \retval NVMEDIA_STATUS_OK on success.
     * \retval (NvMediaStatus) if parameter validation or another internal operation fails.
     *
     * \usage
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
    NvMediaStatus (* GetSensorAttributes)(
        DevBlkCDIDevice const* handle,
        struct DevBlkCDISensorAttributes *sensorAttr,
        const size_t sensorAttrStructSize);

#if !NV_IS_SAFETY
    /** Holds the function that gets module configuration.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \usage
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
    NvMediaStatus (* GetModuleConfig)(
        DevBlkCDIDevice *handle,
        struct DevBlkCDIModuleConfig *moduleConfig);

    /** Holds the function that sets sensor in characterization mode.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \n This function is invoked by \ref DevBlkCDISetSensorCharMode function call.
     * \n
     * @par Sample Usage:
     *
     * \code
     *  Pseudo code for cdi device driver function SetSensorCharMode invoked by DevBlkCDISetSensorCharMode function call.
     *
     * NvMediaStatus
     * SetSensorCharMode(
     *     DevBlkCDIDevice *handle,
     *     uint8_t expNo);
     *  {
     *     NvMediaStatus status = NVMEDIA_STATUS_OK;
     *
     *     check input parameters
     *     if (!handle || !expNo)
     *     {
     *        return NVMEDIA_STATUS_BAD_PARAMETER;
     *     }
     *
     *     set sensor in characterization mode for expNo
     *     status = ExpBypass(handle, expNo, ...)
     *     if (status is NOT NVMEDIA_STATUS_OK) {
     *         Error Handling
     *     }
     *
     *     update driver internal state and sensor attributes
     *     drvrHandle->numActiveExposures = 1;
     *     drvrHandle->charModeEnabled = NVMEDIA_TRUE;
     *     drvrHandle->charModeExpNo = expNo;
     *
     *     return status;
     *  }
     *
     * \endcode
     *
     * \usage
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
    NvMediaStatus (* SetSensorCharMode)(
        DevBlkCDIDevice *handle,
        uint8_t expNo);
#endif

    /**
     * Holds the function that reads a block from an I2C device
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
     *   - De-Init: Yes
     */
    NvMediaStatus (* ReadRegister)(
        DevBlkCDIDevice const* handle,
        uint32_t deviceIndex,
        uint32_t registerNum,
        uint32_t dataLength,
        uint8_t *dataBuff);

    /**
     * Holds the function that writes a block to an I2C device
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
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
     *   - De-Init: Yes
     */
    NvMediaStatus (* WriteRegister)(
        DevBlkCDIDevice const* handle,
        uint32_t deviceIndex,
        uint32_t registerNum,
        uint32_t dataLength,
        uint8_t const* dataBuff);


    /** Holds the function which authenticates passed image data.
     *
     * The function authenticates pixel and embedded data passed in as parameters.
     * Expectation is that driver has already established an authentication session
     * with a sensor prior to a call. If session is not established - function will
     * return error.
     *
     * The AuthenticateImage() method will always be called from a single thread.
     * The same thread can handle multiple driver objects in a sequence, as specified
     * by CNvMCameraModule::SensorProperty::imgAuthThreadID.
     *
     * \param[in]  handle A pointer to the device to use; Valid value: [non-NULL].
     * \param[in]  imageDesc Description of a RAW image to authenticate;
     * \retval NVMEDIA_STATUS_OK Authentication match.
     * \retval NVMEDIA_STATUS_INCOMPATIBLE_VERSION Image verification failure (authentication
     *                                             mismatch).
     * \retval NVMEDIA_STATUS_UNDEFINED_STATE Out-of-order image is detected.
     * \retval NVMEDIA_STATUS_ERROR Internal failure in crypto operation.
     * \retval NVMEDIA_STATUS_INVALID_SIZE Input buffer sizes are invalid.
     * \retval NVMEDIA_STATUS_NOT_INITIALIZED Crypto state was not initialized before
     *                                        requesting image verification.
     *
     * \pre
     *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
     *   - A valid device driver of the CDI device is created with DriverCreate().
     *
     * \usage
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
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    NvMediaStatus (* AuthenticateImage)(
        DevBlkCDIDevice const * const handle,
        DevBlkImageDesc const * const imageDesc);

} DevBlkCDIDeviceDriver;

/**@} <!-- Ends cdi_device_driver_api CDI Device driver --> */

/**
 * \defgroup cdi_device_api CDI Device
 *
 * An CDI device represents a device that is attached or linked to the root I2C
 * port.
 *
 * @{
 */

/**
 * Holds the description of the target I2C device.
 */
typedef struct {
    /** Holds the client context. */
    void *clientContext;
} DevBlkCDIAdvancedConfig;

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Creates a CDI device object.
 *
 * This API does the following:
 * -# Verifies the input parameters.
 * -# Allocates and initializes the device context for the CDI device by calling @ref
 *    NvOsAlloc and @ref NvOsMemset.
 * -# Create each subdevice object with @ref NvOsAlloc and clears its memory by calling
 *    @ref NvOsMemset. The handles for the subdevices will be stored in the device context.
 * -# Initializes the device driver for the CDI device. Overrides
 *    driver's default configuration if @a advancedConfig is provided.
 * -# Returns the device handle, if no errors are encountered.
 *
 * @pre None
 *
 * @param[in] rootDevice A pointer to the root device that you created
 *                       with @ref DevBlkCDIRootDeviceCreate();
 *                       Valid value: [non-NULL].
 * @param[in] deviceAddressList A pointer to the list of I2C device addresses for
 *                              the subdevices; Valid value: [non-NULL].
 *                              The number of I2C device addresses in the
 *                              list is specified by `numDevices`.
 * @param[in] numDevices The number of subdevices. Valid range: [0, UINT32_MAX].
 * @param[in] deviceDriverParam A pointer to the driver structure that defines the behavior of the
 *                              device; Valid value: [non-NULL].
 * @param[in] advancedConfig A non-NULL pointer to the advanced configuration;
 *                           A NULL pointer indicates no advanced configuration is available.
 * @param[in] linkIndex The data link index; Valid range: [0, (Maximum Links Supported per Deserializer - 1)].
 * @param[in] isDeserializer Holds a flag to indicate the device is
 *            deserializer or not; true indicates the device is a deserializer, false indicates the
 *            device is not a deserializer(for example: sensor, serializer, EEPROM).
 * @param[in] useNativeI2CAddress Holds a flag which enables or disables
 *            using native i2c address for a device; true indicates the device is using a native i2c
 *            address, false indicates the device is using a virtual i2c address.
 * @param[in] useCDIv2API Holds a flag to indicate which version of the CDI API to use.
 *                        false indicates version 1, while true indicates version 2;
 *                        For all safety use cases, this flag must set to true to
 *                        use CDI version 2 API.
 * @retval The new device's handle.
 * @retval NULL if error occurred.
 *
 * \usage
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
 * @endif
 */
/// @cond (SWDOCS_CDI_INTERNAL)
DevBlkCDIDevice *
DevBlkCDIDeviceCreate(
    DevBlkCDIRootDevice *rootDevice,
    uint32_t *deviceAddressList,
    uint32_t numDevices,
    DevBlkCDIDeviceDriver *deviceDriverParam,
    DevBlkCDIAdvancedConfig const *advancedConfig,
    uint8_t linkIndex,
    NvMediaBool isDeserializer,
    NvMediaBool useNativeI2CAddress,
    NvMediaBool useCDIv2API
);
/// @endcond

/**
 * @if (SWDOCS_CDI_INTERNAL)
 * @brief Destroys the object that describes an CDI device.
 *
 * This API does the following:
 * -# Deinitializes the device driver for the device.
 * -# Delete each subdevice object.
 * -# Releases all the resources allocated when the device is created.
 * -# Frees the device by calling @ref NvOsFree.
 *
 * @pre None
 *
 * @param[in] device A pointer to the CDI device to destroy;
 *                   Valid value: [non-NULL].
 * @endif
 *
 * \usage
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
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */
/// @cond (SWDOCS_CDI_INTERNAL)
void
DevBlkCDIDeviceDestroy(
    DevBlkCDIDevice *device
);
/// @endcond

#if !(NV_IS_SAFETY)
/**
 * @brief Performs a read operation over I2C.
 *
 * For safety use cases, application software shall call @ref DevBlkCDIDeviceRead() 2 times to read
 * the contents of the same register location. If a register value is not expected to change,
 * return values of the two reads must match.
 *
 * @pre None
 *
 * @param[in] device A pointer to the CDI device to read;
 *                   Valid value: [non-NULL].
 * @param[in] deviceIndex Index of the sub-device to use; Valid range: [0, UINT32_MAX].
 * @param[in] regLength Length of the register address, in bytes; Valid range: [0, UINT32_MAX].
 * @param[in] regData A pointer to the register address; Valid value: [non-NULL].
 * @param[in] dataLength Length of data to be read, in bytes; Valid range: [0, UINT32_MAX].
 * @param[out] data A pointer to the location for storing the read data; Valid value: [non-NULL].
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *           pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
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
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDIDeviceRead(
    DevBlkCDIDevice const *device,
    uint32_t deviceIndex,
    uint32_t regLength,
    uint8_t *regData,
    uint32_t dataLength,
    uint8_t *data
);
#endif

#if !(NV_IS_SAFETY)
/**
 * @brief Performs a write operation over I2C.
 *
 * For safety use cases, application software shall call @ref DevBlkCDIDeviceRead() after calling
 * @ref DevBlkCDIDeviceWrite() to verify whether the write operation was successful. If a register
 * value is not expected to change, read value must match the value written.
 *
 * @pre None
 *
 * @param[in] device A pointer to the CDI device to write;
 *                   Valid value: [non-NULL].
 * @param[in] deviceIndex Index of the sub-device to use; Valid range: [0, UINT32_MAX].
 * @param[in] dataLength Length of data to be written, in bytes; Valid range: [0, UINT32_MAX].
 * @param[in] data       A pointer to data to be written to device via I2C; Valid value: [non-NULL].
 *
 * @retval  NVMEDIA_STATUS_OK            Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *           pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that any other error occurred.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
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
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDIDeviceWrite(
    DevBlkCDIDevice const *device,
    uint32_t deviceIndex,
    uint32_t dataLength,
    const uint8_t *data
);
#endif

/**
 * @brief Queries the sensor attributes.
 *
 * Sensor attributes are static properties like sensor name,
 * exposure-gain ranges supported, and number of active exposures.
 *
 * @pre None
 * @note This function invokes the device driver function specified by the
 * call to @ref GetSensorAttributes().
 * @param[in]  device               A pointer to the device to use;
 *                                  Valid value: [non-NULL].
 * @param[out] sensorAttr           A pointer to the sensor attributes structure;
 *                                  Valid value: [non-NULL].
 * @param[in]  sensorAttrStructSize Size of the @a sensorAttr, in bytes;
 *                                  The size must > 0.
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the device driver
 *                                       does not support this functionality.
 * @retval  (NvMediaStatus) An error code returned by the dependencies in case
 *                          of failure.
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
NvMediaStatus
DevBlkCDIGetSensorAttributes(
    DevBlkCDIDevice *device,
    DevBlkCDISensorAttributes *sensorAttr,
    const size_t sensorAttrStructSize);

/**
 * \brief Holds the sensor control structure.
 * \note
 *   \parblock
 *      To activate a sensor control block, set the corresponding @a valid flag
 *      to TRUE and populate the control settings to be set.
 *
 *      To disable a sensor control block, set the corresponding @a valid flag
 *      to FALSE.
 *
 *      For example, to activate the white balance control block, set the
 *      @a wbValid flag in the @a wbControl structure to TRUE and populate the
 *      white balance settings to be programmed. To disable white balance
 *      control block, set the @a wbValid flag to FALSE.
 *   \endparblock
 */
typedef struct DevBlkCDISensorControl {
    /**
     * Holds the number of sensor contexts to activate.
     * A sensor context is a mode of operation, supported by some sensors,
     * in which multiple set of settings (contexts) are programmed and
     * the sensor toggles between them at run time.
     *
     * Must be in the range [1, \ref DEVBLK_CDI_MAX_SENSOR_CONTEXTS].
     * For sensors that do not support sensor context, set to 1.
     */
    uint8_t                 numSensorContexts;

    /**
     * Holds the sensor exposure settings to set for each context.
     */
    DevBlkCDIExposure       exposureControl[DEVBLK_CDI_MAX_SENSOR_CONTEXTS];

    /**
     * Holds the sensor white balance settings to set for each context.
     */
    DevBlkCDIWhiteBalance   wbControl[DEVBLK_CDI_MAX_SENSOR_CONTEXTS];

    /**
     * Holds the sensor frame report value to be programmed.
     * Sensor frame report control enables you to program
     * custom user information per frame into sensor scratch space (registers)
     * provided by the sensor for private use.
     */
    DevBlkCDIFrameReport    frameReportControl;

    /**
     * Holds the illumination info for the captured frame.
     */
    DevBlkCDIIllumination   illuminationControl;
} DevBlkCDISensorControl;

/**
 * @brief Sets sensor control parameters.
 *
 * This function needs to be implemented in the sensor device driver to control sensor
 * image settings like exposure time, sensor gain, and white balance gain.
 * All parameters provided to this function are applied together at a frame boundary
 * through the "group hold" functionality, if supported by the sensor.
 *
 * @pre None
 *
 * @note This function invokes the device driver function specified by the
 * call to @ref SetSensorControls().
 *
 * @param[in] device  A pointer to the device to use;
 *                    Valid value: [non-NULL].
 * @param[in] sensorControl  A pointer to a sensor control structure for @a device;
 *                           Valid value: [non-NULL].
 * @param[in] sensrCtrlStructSize Size of the @a sensorControl structure, in bytes.
 *                                The size must > 0.
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the device driver
 *                                       does not support this functionality.
 * @retval  (NvMediaStatus) An error code returned by the dependencies in case
 *                          of failure.
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
 *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
 *     NVIDIA DRIVE OS Safety Developer Guide
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDISetSensorControls(
    DevBlkCDIDevice *device,
    const DevBlkCDISensorControl *sensorControl,
    const size_t sensrCtrlStructSize
);

/**
 * @brief  Parses top sensor embedded data info and
 * provides sensor image settings information for the captured frame.
 *
 * This function needs to be implemented in the sensor device driver to
 * retrieve sensor image settings like exposure, gain and white balance gain
 * information applied to the frame.
 *
 * @note This function invokes the device driver function specified by the
 *  call to @ref ParseTopEmbedDataInfo().
 *
 * @pre None
 *
 * @param[in]  device               A pointer to the device to use;
 *                                  Valid value: [non-NULL].
 * @param[in]  embeddedTopDataChunk  A pointer to the top sensor embedded data
 *                                   @ref DevBlkCDIEmbeddedDataChunk structure;
 *                                   Valid value: [non-NULL].
 * @param[in]  embeddedDataChunkStructSize  Size of the @a embeddedTopDataChunk structures, in bytes;
 *                                          The size must > 0.
 * @param[out]  embeddedDataInfo   A pointer to the parsed embedded data info
 *                                 structure; Valid value: [non-NULL].
 * @param[in]  dataInfoStructSize  Size of the @a embeddedDataInfo structure, in bytes;
 *                                 The size must > 0.
 * @retval  NVMEDIA_STATUS_OK            Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *          pointer parameters was NULL or invalid.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the device driver
 *          does not support this functionality.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
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
NvMediaStatus
DevBlkCDIParseTopEmbDataInfo(
    DevBlkCDIDevice *device,
    const DevBlkCDIEmbeddedDataChunk *embeddedTopDataChunk,
    const size_t embeddedDataChunkStructSize,
    DevBlkCDIEmbeddedDataInfo *embeddedDataInfo,
    const size_t dataInfoStructSize);

/**
 * @brief  Parses Bottom sensor embedded data info and
 * provides sensor image settings information for the captured frame.
 *
 * This function needs to be implemented in the sensor device driver to
 * retrieve sensor image settings like exposure, gain and white balance gain
 * information applied to the frame.
 *
 *@note This function invokes the device driver function specified by the
 * call to @ref ParseBotEmbDataInfo().
 *
 * @pre None
 *
 * @param[in]  device               A pointer to the device to use;
 *                                  Valid value: [non-NULL].
 * @param[in]  embeddedBotDataChunk  A pointer to the bottom sensor embedded data
 *                                   @ref DevBlkCDIEmbeddedDataChunk structure;
 *                                   Valid value: [non-NULL].
 * @param[in]  embeddedDataChunkStructSize  Size of the @a embeddedBotDataChunk structures, in bytes;
 *                                          The size must > 0.
 * @param[out]  embeddedDataInfo   A pointer to the parsed embedded data info
 *                                 structure; Valid value: [non-NULL].
 * @param[in]  dataInfoStructSize  Size of the @a embeddedDataInfo structure, in bytes;
 *                                 The size must > 0.
 * @retval  NVMEDIA_STATUS_OK            Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *          pointer parameters was NULL or invalid.
 * @retval  NVMEDIA_STATUS_NOT_SUPPORTED Indicates that the device driver
 *          does not support this functionality.
 * @retval  NVMEDIA_STATUS_ERROR         Indicates that any other error occurred.
 *
 * \usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
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
NvMediaStatus
DevBlkCDIParseBotEmbDataInfo(
    DevBlkCDIDevice *device,
    const DevBlkCDIEmbeddedDataChunk *embeddedBotDataChunk,
    const size_t embeddedDataChunkStructSize,
    DevBlkCDIEmbeddedDataInfo *embeddedDataInfo,
    const size_t dataInfoStructSize);

/**
 * @brief Set the deserializer module power
 *
 * Enable or disable deserializer power based on the enable parameter.
 *
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @param[in] enable The flag to enable/disable.
 *                   Valid value: [ @ref NVMEDIA_TRUE | @ref NVMEDIA_FALSE ].
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER If any of the input parameter is NULL.
 *                                       By the device driver.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDISetDeserPower(
    DevBlkCDIDevice *device,
    NvMediaBool enable);

/**
 * @brief Enable the error report
 *
 * Used to enable the error report.
 *
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER If any of the input parameter is NULL.
 *                                       By the device driver.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIEnableErrorReport(
    DevBlkCDIRootDevice const *device);

/**
 * @brief Get the deserialzer's power control information
 *
 * Get the deserialzer's power control information.
 *
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @param[out] desPwrControlInfo A structure that holds the power
 *                               control information for deserializer.
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
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
 *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
 *     NVIDIA DRIVE OS Safety Developer Guide
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIGetDesPowerControlInfo(
    DevBlkCDIDevice *device,
    DevBlkCDIPowerControlInfo *desPwrControlInfo);

/**
 * @brief Get the camera's power control information
 *
 * Get the camera's power control information.
 *
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                  Valid value: [non-NULL].
 * @param[out] camPwrControlInfo A structure that holds the power
 *                              control information for camera module.
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
 *
 *@usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
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
NvMediaStatus
DevBlkCDIGetCamPowerControlInfo(
    DevBlkCDIDevice *device,
    DevBlkCDIPowerControlInfo *camPwrControlInfo);

/**
 * @brief Reserve device I2C address.
 *
 * Supplied I2C address is reserved either as physical or virtual
 * as per request.
 * If caller wants to get new virtual address for the device, it
 * should supply valid physical I2C address of the device, in the
 * request to reserve address as virtual.
 *
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @param[in] address An I2C address to be reserved.
 *                    Valid range: [ 0, UINT8_MAX ].
 * @param[in] useNativeI2C A flag to denote if address needs to be
 *                         reserved as physical (native) or virtual address.
 *                         Valid value: [ 0 | 1].
 * @param[out] reservedI2CAddr A pointer to reserved I2C address.
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER If the pointer of device is NULL.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIReserveI2CAddr(
    DevBlkCDIDevice *device,
    uint8_t address,
    bool useNativeI2C,
    uint32_t *reservedI2CAddr);

#if !(NV_IS_SAFETY)
/**
 * @brief Set Multiplexer to select the FRSYNC source
 *
 * Set Multiplexer to select the FRSYNC source.
 * @pre None
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @param[in] fsyncMuxSel A selection to Multiplexer.
 *                        Valid range: [ 0, UINT32_MAX ].
 * @param[in] camGrpIdx An index to the camera group
 *                      Valid range: [ 0, UINT32_MAX ].
 * @retval  NVMEDIA_STATUS_OK Indicates that the operation was successful.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_ERROR Indicates that any other error occurred.
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDISetFsyncMux(
    DevBlkCDIDevice *device,
    uint32_t fsyncMuxSel,
    uint32_t camGrpIdx);
#endif

/**
 * @brief Authenticate an image data passed in a parameters.
 *
 * Authenticate an image data passed in a parameters.
 *
 * @pre @a The struct must be valid @ref DevBlkImageDesc handle created width @ref CNvMAuthMgr::AuthOneImage.
 *
 * @param[in] device A pointer to the device to use.
 *                   Valid value: [non-NULL].
 * @param[in] imageDesc Description of an image to authenticate.
 *                      Valid value: [non-NULL].
 *
 * @retval NVMEDIA_STATUS_INCOMPATIBLE_VERSION Image verification failure (authentication
 *                                              mismatch).
 * @retval NVMEDIA_STATUS_UNDEFINED_STATE Out-of-order image is detected.
 * @retval NVMEDIA_STATUS_ERROR Internal failure in crypto operation.
 * @retval NVMEDIA_STATUS_NOT_SUPPORTED authentication is not supported by
 *                                       the device.
 * @retval NVMEDIA_STATUS_INVALID_SIZE if any of the input size parameters do
 *                                      not present values which make sense.
 * @retval NVMEDIA_STATUS_NOT_INITIALIZED Crypto state was not initialized before
 *                                         requesting image verification.
 * @retval  NVMEDIA_STATUS_BAD_PARAMETER Indicates that one or more
 *                                       pointer parameters was NULL.
 * @retval  NVMEDIA_STATUS_OK Authentication is successful.
 * @retval  (NvMediaStatus) An error code returned by the dependencies in case
 *                          of failure.
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
DevBlkCDIAuthenticateImage(
        DevBlkCDIDevice const * const device,
        DevBlkImageDesc const * const imageDesc);

/**@} <!-- Ends cdi_device_api CDI Device --> */

#if !NV_IS_SAFETY
/**
 * \hideinitializer
 * \brief CDI Power control items
 */
typedef enum {
    /** Aggregator Power */
    DEVBLK_CDI_PWR_AGGREGATOR,
    /** LINK 0 Power */
    DEVBLK_CDI_PWR_LINK_0,
    /** LINK 1 PWR */
    DEVBLK_CDI_PWR_LINK_1,
    /** LINK 2 PWR */
    DEVBLK_CDI_PWR_LINK_2,
    /** LINK 3 PWR */
    DEVBLK_CDI_PWR_LINK_3,
} DevBlkCDIPowerItems;

/**
 * \hideinitializer
 * \brief Holds the CDI Module ISP configuration.
 */
typedef struct DevBlkCDIModuleConfig {
    /** Holds the camera module name. */
    char cameraModuleCfgName[128];
    /** Holds the camera-specific configuration string. */
    const char *cameraModuleConfigPass1;
    const char *cameraModuleConfigPass2;
} DevBlkCDIModuleConfig;

/** Set sensor in characterization mode.
 *
 * @par Description
 * DevBlkCDISetSensorCharMode API provides ability for the user to configure
 * the sensor for characterization. Sensor characterization provides optimal
 * parameters, corresponding to sensor physical and functional
 * characteristics, for image processing.
 * \n Sensor characterization for High Dynamic Range (HDR) sensors with
 * multiple exposures (T1, T2, … , Tn ) involves characterizing individual
 * exposures separately, if required by the sensor. This API provides the
 * ability to configure sensor to capture each exposure separately,
 * if required by sensor characterization.
 * This function re-configures the sensor i.e. changes the sensor static
 * attributes like numActiveExposures, sensorExpRange, sensorGainRange
 * and hence, should be called during sensor initialization time.
 * In order to characterize the sensor exposure number ‘n’,
 * where n = {1,2,3, … , N} for N-exposure HDR sensor, the input parameter
 * ‘expNo’ should be set to ‘n’.
 * \n For a non-HDR sensor, the input parameter ‘expNo’ should always be set to ‘1’.
 *
 * \pre @a device must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *
 * \param[in]  device A pointer to the sensor control device in use.
 * \param[in]  expNo  Sensor exposure number to be used for characterization.
 * Valid range for expNo : [0, (DEVBLK_CDI_MAX_EXPOSURES-1)]
 * For Non-HDR sensor, this should be set to '1'
 *
 * \return \ref NvMediaStatus  The completion status of the operation.
 * Possible values are:
 * \li \ref NVMEDIA_STATUS_OK if successful.
 * \li \ref NVMEDIA_STATUS_BAD_PARAMETER if an input parameter is NULL or invalid.
 * \li \ref NVMEDIA_STATUS_NOT_SUPPORTED if the functionality is not supported
 * by the device driver.
 * \li \ref NVMEDIA_STATUS_ERROR if some other error occurred.
 *
 * \usage
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
NvMediaStatus
DevBlkCDISetSensorCharMode(
    DevBlkCDIDevice *device,
    uint8_t expNo);

/**
 * \brief Gets the Module ISP configuration.
 *
 * \pre @a device must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *
 * \param[in] device A pointer to the device to use.
 * \param[out] moduleConfig A pointer to the module ISP configuration.
 * \return \ref NvMediaStatus The completion status of the operation.
 * Possible values are:
 * \li \ref NVMEDIA_STATUS_OK
 * \li \ref NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * \li \ref NVMEDIA_STATUS_NOT_SUPPORTED if the functionality is not supported
 * by the device driver.
 * \li \ref NVMEDIA_STATUS_ERROR if any other error occurred.
 *
 * \usage
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
NvMediaStatus
DevBlkCDIGetModuleConfig(
    DevBlkCDIDevice *device,
    DevBlkCDIModuleConfig *moduleConfig);

#endif /* #if !NV_IS_SAFETY */

/**@} <!-- Ends devblk_cdi_api Camera Device Interface --> */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* DEVBLK_CDI_H */
