/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef DEVBLK_CDI_I2C_H
#define DEVBLK_CDI_I2C_H

#include "devblk_cdi.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup devblk_cdi_i2c_api Camera Device Interface I2C Programmer
 *
 * The Camera Device Interface I2C Programmer API encompasses all I2C related
 * functions for programming all I2C controlled components
 * such as deserializers, serializers, EEPROMs, and image sensors.
 *
 * @ingroup cdi_api_grp
 * @{
 */

/**
 * @brief Macro for determining the number of address/data pairs in a register table
 *
 * This macro should be used by the device driver to set @ref DevBlkCDII2CRegList::numRegs.
 *
 * Do not use this macro when referencing register tables through pointers, otherwise
 * the wrong number of address/data pairs will be returned.
 *
 */
#define I2C_ARRAY_SIZE(x) (uint32_t)(sizeof(x) / sizeof((x)[0]))

/** The maximum number of i2c commands for performing an I2C bulk read/write */
#define CDI_I2C_MAX_BATCH_SIZE          16U

/** @brief Defines the I2C address/data pair and an optional delay. */
typedef struct {
    /**
     * The address of an I2C register;
     * Valid range: [0, UINT16_MAX].
     */
    uint16_t address;

    /**
     * The value of an I2C register;
     * Valid range: [0, UINT16_MAX].
     */
    uint16_t data;

    /**
     * The number of microseconds to delay between this read/write operation
     * and the next one; Valid range: [0, UINT32_MAX].
     */
    uint32_t delayUsec;
} DevBlkCDII2CReg;

/** @brief Defines the I2C register table to write to the device. */
typedef struct {
    /**
     * An array of DevBlkCDII2CReg structures, of length `numRegs`.
     *
     * The array should be declared with "const" so the values in the
     * array cannot be modified.
     */
    const DevBlkCDII2CReg *regs;
    /**
     * The number of registers in the `regs` array; Valid range: [0, UINT32_MAX].
     */
    uint32_t numRegs;
} DevBlkCDII2CRegList;

/** @brief Defines the I2C register table to read from the device */
typedef struct {
    /**
     * An array of DevBlkCDII2CReg structures, of length `numRegs`.
     *
     * The array must not be declared with "const" so the values
     * read back from device can be stored in the array.
     */
    DevBlkCDII2CReg *regs;
    /**
     * The number of registers in the `regs` array; Valid range: [0, UINT32_MAX].
     */
    uint32_t numRegs;
} DevBlkCDII2CRegListWritable;

/**
 * @brief  An opaque handle for an I2C programmer object.
 */
typedef void* DevBlkCDII2CPgmr;

/**
 * @brief i2c WRITE transactions callback function.
 *
 * Callback type used by i2cProgrammer listeners. DevBlkCDII2CPgmrListenerReg is used
 * to register a callback with i2cProgrammer.
 *
 * The callback is called by i2cProgrammer after every successful i2c write transaction
 * performed by the i2cProgrammer, passing i2c write data which was sent over i2c.
 *
 * The callback is expected to be safe for re-entrancy.
 *
 * @pre
 *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *   - A valid DevBlkCDII2CPgmr object created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   handle A pointer to the device that registered a listener.
 *                     Valid value: [non-NULL].
 * @param[in]   data A buffer pointer containing data which was just sent over i2c
 *                   bus by a corresponding i2cProgrammer.
 *                   Valid value: [non-NULL].
 * @param[in]   len Length of i2c write transaction buffer pointed to by data;
 *                  Valid range: [>0].
 *
 * @retval      NVMEDIA_STATUS_OK If listener processed notification successfully.
 * @retval      not-NVMEDIA_STATUS_OK for any listener-specific error while processing
 *                                    notification.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: no
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
 */
typedef NvMediaStatus (*DevBlkCDII2CWriteCb)(DevBlkCDIDevice const * const handle,
                                             uint8_t const *data, uint32_t len);

/**
 * @brief i2c READ transactions callback function.
 *
 * Callback type used by i2cProgrammer listeners. DevBlkCDII2CPgmrListenerReg is used
 * to register a callback with i2cProgrammer.
 *
 * The callback is called by i2cProgrammer after every successful i2c read transaction
 * performed by the i2cProgrammer, passing i2c read data which was received over i2c.
 *
 * The callback is expected to be safe for re-entrancy.
 *
 * @pre
 *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *   - A valid DevBlkCDII2CPgmr object created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   handle A pointer to the device that registered a listener.
 *                     Valid value: [non-NULL].
 * @param[in]   address A start address in device register space for which the read
 *                      transaction was done.
 * @param[in]   data A buffer pointer containing data which was just read from i2c
 *                   bus by a corresponding i2cProgrammer.
 *                   Valid value: [non-NULL].
 * @param[in]   len Length of i2c read transaction buffer pointed to by data.
 *                  Valid range: [>0].
 *
 * @retval      NVMEDIA_STATUS_OK If listener processed notification successfully.
 * @retval      not-NVMEDIA_STATUS_OK for any listener-specific error while processing
 *                                    notification.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: no
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
 */
typedef NvMediaStatus (*DevBlkCDII2CReadCb)(DevBlkCDIDevice const * const handle,
                                            uint16_t address,
                                            uint8_t const *data, uint32_t len);
/**
 * @brief Pre-sleep notification callback function.
 *
 * Callback type used by i2cProgrammer listeners. DevBlkCDII2CPgmrListenerReg is used
 * to register a callback with i2cProgrammer.
 *
 * The callback is called by i2cProgrammer before the programmer enters a sleep during
 * a sequence of I2C writes.
 * Usually sleep is required to make sure previous I2C writes are effective before
 * issuing next i2c writes. Listener has a chance to perform driver-specific actions
 * before sleeping.
 *
 * The callback is expected to be safe for re-entrancy.
 *
 * @pre
 *   - @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *   - A valid DevBlkCDII2CPgmr object created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   handle A pointer to the device that registered a listener.
 *                     Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If listener processed notification successfully.
 * @retval      not-NVMEDIA_STATUS_OK for any listener-specific error while processing
 *                                    notification.
 *
 * @usage
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
 */
typedef NvMediaStatus (*DevBlkCDII2CSleepCb)(DevBlkCDIDevice const * const handle);

/**
 * @brief Creates an I2C Programmer to read/write I2C registers of a device.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Allocates and initializes the context for the I2C Programmer.
 * -# Returns the handle of the new I2C Programmer.
 *
 * @pre @a handle must be valid DevBlkCDIDevice handle created with DevBlkCDIDeviceCreate().
 *
 * @param[in]   handle A pointer to the device that needs I2C access;
 *                     Valid value: [non-NULL].
 * @param[in]   addrLength I2C register address length in bytes; Valid range: [1, 2].
 * @param[in]   dataLength I2C register data length in bytes; Valid range: [1, 2].
 *
 * @return      An opaque handle of an I2C programmer object.
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
DevBlkCDII2CPgmr
DevBlkCDII2CPgmrCreate(
    DevBlkCDIDevice *handle,
    const uint8_t addrLength,
    const uint8_t dataLength
);

/**
 * @brief Destroys an I2C Programmer.
 *
 * This API does the following:
 * -# Releases all the resources allocated when the I2C Programmer is created.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer to destroy.
 *                            Valid value: [non-NULL].
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
 *   - Runtime: No
 *   - De-Init: Yes
 */
void
DevBlkCDII2CPgmrDestroy(
    DevBlkCDII2CPgmr i2cProgrammer
);

/**
 * @brief Register an i2c transactions listener
 *
 * Register Read and/or Write notification callbacks on the i2cProgrammer.
 * wrCallback is invoked by i2cProgrammer after each successful i2c Write
 * transaction to the device, providing data buffer which was sent over i2c.
 * rdCallback is invoked by i2cProgrammer after each successful i2c Read
 * transaction with the device, providing data buffer which was read from i2c.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer.
 *                            Valid value: [non-NULL].
 * @param[in]   handle A handle for a CDI device with registers as a listener.
 *                     The handle is passed as a parameter to RD and WR callabcks.
 *                     Valid value: [non-NULL].
 * @param[in]   wrCallback A pointer to a callback function to be called after i2c write.
 *                         Valid value: a pointer to DevBlkCDII2CWriteCb function or NULL.
 * @param[in]   rdCallback A pointer to a callback function to be called after i2c read.
 *                         Valid value: a pointer to DevBlkCDII2CReadCb function or NULL.
 * @param[in]   sleepCallback A pointer to a callback function to be called before and after
 *                            the programmer enters a sleep.
 *                         Valid value: a pointer to DevBlkCDII2CSleepCb function or NULL.
 *
 * @retval      NVMEDIA_STATUS_OK If listener was registered successfully.
 * @retval      NVMEDIA_STATUS_OUT_OF_MEMORY If no more listeners can be registered on
 *                                           a given i2cProgrammer.
 *
 * @usage
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
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus DevBlkCDII2CPgmrListenerReg(DevBlkCDII2CPgmr const i2cProgrammer,
                                          DevBlkCDIDevice const * const handle,
                                          DevBlkCDII2CWriteCb wrCallback,
                                          DevBlkCDII2CReadCb rdCallback,
                                          DevBlkCDII2CSleepCb sleepCallback);

/**
 * @brief Writes 8-bit data to an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 8-bit data to an I2C register.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 8-bit register data; Valid range: [0, UINT8_MAX]
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteUint8(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint8_t data
);

/**
 * @brief Writes 8-bit data to an I2C register along with
 * I2C Readback Verification
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 8-bit data to an I2C register.
 * -# The call to CDAC also requests I2C Readback verification.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 8-bit register data; Valid range: [0, UINT8_MAX]
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteUint8Verify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint8_t data
);
/**
 * @brief Writes 16-bit data to an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write a 16-bit data to an I2C register.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 16-bit register data; Valid range: [0, UINT16_MAX].
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteUint16(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint16_t data
);

/**
 * @brief Writes an array to i2c device starting from a register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an array of values starting from a specified i2c address.
 * Register address is incremented for each next byte in array
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit starting register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data a pointer to an array to send over i2c; Valid range: [non-NULL].
 * @param[in]   dataLen length of an array to write; Valid range: [0, UINT16_MAX].
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
 *
 * @usage
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
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteBlock(
    DevBlkCDII2CPgmr const i2cProgrammer,
    uint16_t const address,
    uint8_t const * const data,
    uint16_t const dataLen
);

 /**
 * @brief Writes a block of data to i2c device starting from a register along with
 * I2C Readback Verification
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an array of values starting from a specified i2c address.
 * Register address is incremented for each next byte in array
 * -# The call to CDAC also requests I2C Readback verification.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit starting register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data a pointer to an array to send over i2c; Valid range: [non-NULL].
 * @param[in]   dataLen length of an array to write; Valid range: [0, UINT16_MAX].
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
 *              or if the I2C Readback Verification failed.
 *
 * @usage
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
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteBlockVerify(
    DevBlkCDII2CPgmr const i2cProgrammer,
    uint16_t const address,
    uint8_t const * const data,
    uint16_t const dataLen
);

/**
 * @brief Writes 16-bit data to an I2C register along with
 * I2C Readback Verification
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 16-bit data to an I2C register.
 * -# The call to CDAC also requests I2C Readback verification.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 16-bit register data; Valid range: [0, UINT16_MAX]
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteUint16Verify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint16_t data
);

/**
 * @brief Reads 8-bit data from an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to read an 8-bit data from an I2C register.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[out]   data  A pointer to an 8-bit buffer that holds the read value;
 *                     Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadUint8(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    uint8_t *data
);

/**
 * @brief Reads 8-bit data to an I2C register along with
 * I2C Readback Verification
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 8-bit data to an I2C register.
 * -# The call to CDAC also requests I2C Readback verification.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 8-bit register data; Valid range: [0, UINT8_MAX]
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadUint8Verify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    uint8_t *data
);

/**
 * @brief Reads 16-bit data from an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to read a 16-bit data from an I2C register.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[out]   data A pointer to a 16-bit buffer that holds the read value;
 *                    Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadUint16(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    uint16_t *data
);

/**
 * @brief Reads 16-bit data to an I2C register along with
 * I2C Readback Verification
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 16-bit data to an I2C register.
 * -# The call to CDAC also requests I2C Readback verification.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   data 16-bit register data; Valid range: [0, UINT16_MAX]
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadUint16Verify(
    const DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    uint16_t *data
);

/**
 * @brief Performs write operation for a register table.
 *
 * This function calls CDAC to write each of the registers in a register table.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will use the address/data pair information to program a sequence of I2C registers.
 * To improve efficiency, this function batches writes and flushes them to the
 * hardware in one of three situations:
 *
 *   - An entry in the register list has a `delayUsec` > 0. In this case the
 *   batch is immediately flushed and the thread is put to sleep for the
 *   specified time.
 *   - The batch size reaches \ref CDI_I2C_MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   regList A pointer to the register table;
 *                      Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteArray(
    DevBlkCDII2CPgmr i2cProgrammer,
    const DevBlkCDII2CRegList *regList
);

/**
 * @brief Performs write operation for a register table.
 *
 * This function calls CDAC to write each of the registers in a register table
 * and also requests I2C readback verification from CDAC.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will use the address/data pair information to program a sequence of I2C registers.
 * To improve efficiency, this function batches writes and flushes them to the
 * hardware in one of three situations:
 *
 *   - An entry in the register list has a `delayUsec` > 0. In this case the
 *   batch is immediately flushed and the thread is put to sleep for the
 *   specified time.
 *   - The batch size reaches \ref CDI_I2C_MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   regList A pointer to the register table;
 *                      Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If write to register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrWriteArrayVerify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const DevBlkCDII2CRegList *regList
);

/**
 * @brief Performs read operation for a register table.
 *
 * This function calls CDAC to read each of the registers in a register table.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will use the address/data pair information to read a sequence of I2C register values.
 * To improve efficiency, this function batches reads from the hardware in one of
 * two situations:
 *
 *   - The batch size reaches \ref CDI_I2C_MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
 *
 * \note In order to store the read register values, the register table should be
 *       declared using \ref DevBlkCDII2CRegListWritable structure.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[out]   regList A pointer to the writable register table.
 *                      Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadArray(
    DevBlkCDII2CPgmr i2cProgrammer,
    const DevBlkCDII2CRegListWritable *regList
);

/**
 * @brief Performs read operation for a register table.
 *
 * This function calls CDAC to read each of the registers in a register table
 * and also requests I2C readback verification from CDAC.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will use the address/data pair information to read a sequence of I2C register values.
 * To improve efficiency, this function batches reads from the hardware in one of
 * two situations:
 *
 *   - The batch size reaches \ref CDI_I2C_MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
 *
 * \note In order to store the read register values, the register table should be
 *       declared using \ref DevBlkCDII2CRegListWritable structure.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[out]   regList A pointer to the writable register table.
 *                      Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadArrayVerify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const DevBlkCDII2CRegListWritable *regList
);

/**
 * @brief Reads a block of data from I2C device
 *
 * This function calls CDAC to read data from consecutive register addresses. It is
 * best used for reading a block of data from devices like EEPROM.
 *
 * To improve performance, this function performs the I2C bulk read operation,
 * by specifying the starting address and the length of data in bytes.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   dataLength Size of the data to read; Valid range: [1, \ref MAX_CDAC_I2C_BUFFER].
 * @param[out]   dataBuff A pointer to a buffer that must be bigger than `dataLength` bytes to
 *                        hold all the read values from the device; Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadBlock(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint16_t dataLength,
    uint8_t *dataBuff
);

/**
 * @brief Reads a block of data from I2C device starting at a register along with
 * I2C Readback Verification
 *
 * This function calls CDAC to read data from consecutive register addresses. It is
 * best used for reading a block of data from devices like EEPROM.
 * The call to CDAC also requests I2C Readback verification.
 *
 * To improve performance, this function performs the I2C bulk read operation,
 * by specifying the starting address and the length of data in bytes.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer;
 *                            Valid value: [non-NULL].
 * @param[in]   address 16-bit register address; Valid range: [0, UINT16_MAX].
 * @param[in]   dataLength Size of the data to read; Valid range: [1, \ref MAX_CDAC_I2C_BUFFER].
 * @param[out]   dataBuff A pointer to a buffer that must be bigger than `dataLength` bytes to
 *                        hold all the read values from the device; Valid value: [non-NULL].
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
 *              or if the I2C Readback Verification failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrReadBlockVerify(
    DevBlkCDII2CPgmr i2cProgrammer,
    const uint16_t address,
    const uint16_t dataLength,
    uint8_t *dataBuff
);

#if !NV_IS_SAFETY
/**
 * @brief Dumps the values for all registers in the register table.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will dump the I2C regsister value for each address/data pair for debugging purpose.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer An opaque handle for I2C Programmer.
 * @param[in]   regList Pointer to the register table.
 *
 * @retval      NVMEDIA_STATUS_OK If read from register is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
 * @retval      NVMEDIA_STATUS_ERROR If the devctl IPC call to CDAC device node failed.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrDumpArray(
    DevBlkCDII2CPgmr i2cProgrammer,
    const DevBlkCDII2CRegList *regList
);

/**
 * @brief Enables debug logs for I2C Programmer.
 *
 * @pre @a i2cProgrammer must be valid DevBlkCDII2CPgmr handle created with DevBlkCDII2CPgmrCreate().
 *
 * @param[in]   i2cProgrammer Device handle for I2C Programmer.
 * @param[in]   enable flag to enable/disable debug logs.
 *
 * @retval      NVMEDIA_STATUS_OK If enable/disable debug log is successful.
 * @retval      NVMEDIA_STATUS_BAD_PARAMETER If one or more params was NULL or invalid.
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
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
DevBlkCDII2CPgmrDebugLogControl(
    DevBlkCDII2CPgmr i2cProgrammer,
    NvMediaBool enable
);
#endif /* #if !NV_IS_SAFETY */

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* DEVBLK_CDI_I2C_H */
