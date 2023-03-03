/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved. All
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
 * @brief Creates an I2C Programmer to read/write I2C registers of a device.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Allocates and initializes the context for the I2C Programmer.
 * -# Returns the handle of the new I2C Programmer.
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
 * @brief Writes 8-bit data to an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write an 8-bit data to an I2C register.
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
 * @brief Writes 16-bit data to an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to write a 16-bit data to an I2C register.
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
 * @brief Reads 8-bit data from an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to read an 8-bit data from an I2C register.
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
 * @brief Reads 16-bit data from an I2C register.
 *
 * This API does the following:
 * -# Verifies all the input parameters.
 * -# Calls CDAC to read a 16-bit data from an I2C register.
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
 *   - The batch size reaches \ref MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
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
 * @brief Performs read operation for a register table.
 *
 * This function calls CDAC to read each of the registers in a register table.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will use the address/data pair information to read a sequence of I2C register values.
 * To improve efficiency, this function batches reads from the hardware in one of
 * two situations:
 *
 *   - The batch size reaches \ref MAX_BATCH_SIZE.
 *   - The final entry in the register list has been reached.
 *
 * \note In order to store the read register values, the register table should be
 *       declared using \ref DevBlkCDII2CRegListWritable structure.
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
 * @brief Reads a block of data from I2C device
 *
 * This function calls CDAC to read data from consecutive register addresses. It is
 * best used for reading a block of data from devices like EEPROM.
 *
 * To improve performance, this function performs the I2C bulk read operation,
 * by specifying the starting address and the length of data in bytes.
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

#if !NV_IS_SAFETY
/**
 * @brief Dumps the values for all registers in the register table.
 *
 * The register table consists of multiple register address/data pairs. This function
 * will dump the I2C regsister value for each address/data pair for debugging purpose.
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
 *   - Runtime: No
 *   - De-Init: No
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
