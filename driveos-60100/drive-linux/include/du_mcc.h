/*
 * Copyright (c) 2022, NVIDIA Corporation. All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation. Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */
#ifndef __DU_MCC_H__
#define __DU_MCC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>


/*==================[macros]==================================================*/

/** Return types for APIs.
 */

/** Success. The requested operation is successfully completed.
 */
#define DU_MCC_E_OK                        0x55U

/** Out of range error. Response data validation failed at MCU.
 */
#define DU_MCC_E_DATA_OOR                  0xF2U

/** Incorrect Interface ID sent from DU_MCC library to MCU.
 */
#define DU_MCC_E_INVALID_IF_ID             0xF3U

/** Update firmware (UFW) on MCU is not valid
 */
#define DU_MCC_E_INVALID_UPDATE_FW         0xF4U

/** Condition not valid at MCU to perform request operation
 */
#define DU_MCC_E_COND_NOT_MET              0xF5U

/** Incorrect message length sent from DU_MCC library to MCU
 */
#define DU_MCC_E_SIZE_ERR                  0xF6U

/** Flash write operation failed at MCU due to flash lock
 */
#define DU_MCC_E_WR_FLASH_LOCK_ERR         0xF7U

/** Flash write operation failed at MCU as flash is busy
 */
#define DU_MCC_E_WR_FLASH_BUSY_ERR         0xF8U

/** Flash write operation generic failure at MCU
 */
#define DU_MCC_E_WR_FLASH_ERR              0xF9U

/** Flash erase operation failed at MCU due to flash lock
 */
#define DU_MCC_E_ERASE_FLASH_LOCK_ERR      0xFAU

/** Flash erase operation generic failure at MCU
 */
#define DU_MCC_E_ERASE_FLASH_ERR           0xFBU

/** Programming sequence error
 */
#define DU_MCC_E_SEQ_ERR                   0xFCU

/** CRC computed at MCU doesn't not match with
 *  received CRC from DU_MCC library
 */
#define DU_MCC_E_CHKSUM_ERR                0xFDU

/** Invalid data sent from DU_MCC library to MCU
 */
#define DU_MCC_E_INVALID_DATA_ERR          0xFEU

/** Incorrect block ID sent from DU_MCC library to MCU
 */
#define DU_MCC_E_BLKID_INVALID_ERR         0xFFU

/** Response not received by DU_MCC within timeout
 */
#define DU_MCC_E_TIMEOUT                   0x0FU

/** API called with invalid parameter
 */
#define DU_MCC_E_INVALID_PARAMS            0x0EU

/** Flashed Aurix FW does not support du mcc interface
 */
#define DU_MCC_E_NOT_SUPPORTED              0x0DU

/** DU_MCC library failed to transmit request to MCU
 */
#define DU_MCC_E_COMM_ERR                  0x0CU

/** HW step of provided Hex file does not match with flashed Aurix FW
 */
#define DU_MCC_E_INCOMPATIBLE_MCU          0x0BU

/** Board type of provided Hex file does not match with flashed Aurix FW
 */
#define DU_MCC_E_INCOMPATIBLE_BOARD        0x0AU

/** Generic failure in du mcc library
 */
#define DU_MCC_E_NOK                       0x09U


/** Data types
 */
typedef uint8_t         U8;
typedef uint16_t        U16;
typedef uint32_t        U32;


/** Used to verify du and MCC lib are compatible - [Required MCC lib symbol] */
extern const U32 DU_MCC_version;

/** API return type
 */
typedef U8 DU_MCC_Ret_Type;

/** Tegra device IDs
 */
typedef enum  {
    DU_MCC_TEGRA_X1,
    DU_MCC_TEGRA_X2,
    DU_MCC_TEGRA_INVALID_ID
} DU_MCC_Tegra_Device_ID;

/** Drive Update Boot Chain IDs
 */
typedef enum  {
    DU_MCC_BootChain_A,
    DU_MCC_BootChain_B,
    DU_MCC_BootChain_C,
    DU_MCC_BootChain_D,
    DU_MCC_BootChain_INVALID_ID
} DU_MCC_BootChain_ID;

//==================================================================================================
// The following section defines the DU MCC library interface. The DU will then call DU_MCC_Init() 
// and then good to call all other APIs to start communicating with the MCU
//
// ...Sample code flow from DU...
// [DU]
//  /* check the du_mcc_version */
//  ASSERT(DU_MCC_LIB_VERSION == DU_MCC_version)
//
//  /* Initialize the MCC lib */
//  void* libHandle = DU_MCC_Init(0, NULL);
//
//  /* Start communicating with MCU and process requests */
//  DU_MCC_SetDefaultBootChain(libHandle, ...);
//
//  /* De init */
//  DU_MCC_DeInit(libHandle);
//==================================================================================================

/**
 * DU_MCC_Init() is the interface to Initialize library instance
 *
 *
 * @param argc: No of args
 * @param argv : arguments
 * @return : void* : NULL if there is any error in Initializing, 
 *      pointer to internal handle would be returned in case of successful initialization
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
void* DU_MCC_Init(int argc, char *argv[]);

/**
 * DU_MCC_DeInit() is the interface to DeInitialize library instance
 *
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @return : DU_MCC_Ret_Type : success/failure in DeInit
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */
DU_MCC_Ret_Type DU_MCC_DeInit(void* handle);

/**
 * DU_MCC_TegraReboot() is the interface for requesting reboot of Tegra
 *
 * This function requests MCU to Reboot Tegra.
 * An option is provided to hold Tegra in reset state.
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param Tegra_id: Tegra ID for Tegra x1 or x2
 * @param timeout : Timeout value in Seconds, MCU shall wait before performing Tegra reset.
 * @return : DU_MCC_Ret_Type : success/failure in Tegra Reboot
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_TegraReboot(
   void* handle,
   DU_MCC_Tegra_Device_ID Tegra_id,
   U8 timeout);

/**
 * DU_MCC_SystemReset() is the interface to reset Aurix.
 *
 * This function requests MCU FW to reset the Aurix controller.
 * This will result in reset of all the Tegras and the peripherals
 * on the board
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param timeDelay : Timeout value in Seconds, MCU shall wait before issuing Tegra reset.
 * @return : DU_MCC_Ret_Type : success/failure in Aurix reset / complete board reset
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_SystemReset(
    void* handle,
    U8 timeDelay);


/**********************************************************************/
/* Interface APIs for GPIO based Boot-chain selection for Tegra x1/x2 */
/**********************************************************************/

/**
 * DU_MCC_SetNext_BootChain() is the interface to request MCU
 * to switch the BootChain of Tegra to default or alternate.
 * (only for the next immediate reboot of Xavier).
 *
 * This function requests MCU to switch the BootChain of Tegra to default or
 * alternate boot-chain, by changing the state of the GPIO line.
 * This will not change the default BootChain in persistent memory.
 * This shall be followed by a MCU_TegraReboot request, to boot the
 * Tegra in the selected BootChain.
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param Tegra_id : Id of Tegra x1 or x2 for which the BootChain
 * switching is requested
 * @param DU_MCC_BootChain_ID : A|B|C|D
 * @return : DU_MCC_Ret_Type : success/failure in set BootChain for next reboot
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_SetNextBootChain(
    void* handle,
    DU_MCC_Tegra_Device_ID Tegra_id,
    DU_MCC_BootChain_ID Boot_id);


/**
 * DU_MCC_SetDefault_BootChain() is the interface to request MCU
 * to set the default BootChain of the Tegra.
 *
 * This function requests MCU to set the default BootChain selection
 * status in persistent memory to the supplied Boot_id values.
 *
 * This function will change status in persistent memory only,
 * does not set the GPIO line accordingly.
 *
 * MCU will evaluate boot chain configuration from persistent memory
 * to set the GPIO line during every boot-up of Tegra.
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param Tegra_id : Id of Tegra x1 or x2
 * @param Boot_id : Id of default BootChain to be set
 * @return : DU_MCC_Ret_Type : success/failure in set default BootChain
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_SetDefaultBootChain(
    void* handle,
    DU_MCC_Tegra_Device_ID Tegra_id,
    DU_MCC_BootChain_ID Boot_id);

/**********************************************************************/
/* Interface APIs for getting Boot-chain information for Tegra x1/x2  */
/**********************************************************************/

/**
 * DU_MCC_GetDefault_BootChain() is the interface to request MCU
 * to get default boot chain configuration for Tegra.
 *
 * This function requests MCU to get default boot chain configuration for Tegra.
 * MCU gets the boot chain status by reading default boot chain configuration
 * in persistent memory for supplied Tegra Id.
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param Tegra_id: Id of Tegra x1 or x2 for which the Default BootChain
 *                  configuration is requested.
 * @param DefaultBootChain (out): out param to store obtained default
 *        BootChain configuration. = 0x0 for BootChain_A, 0x1 for BootChain_B
 * @return : DU_MCC_Ret_Type : success/failure in getting default BootChain configuration.
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_GetDefaultBootChain(
    void* handle,
    DU_MCC_Tegra_Device_ID Tegra_id,
    U8 *DefaultBootChain);

/**
 * DU_MCC_GetActive_BootChain() is the interface to request MCU
 * to get active boot chain selection for Tegra.
 *
 * This function requests MCU to get active boot chain selection for Tegra.
 * MCU gets the boot chain selected while booting the Tegra.
 *
 * @param handle: Handle obtained from DU_MCC_Init() should be passed
 * @param Tegra_id: Id of Tegra x1 or x2 for which the active BootChain
 *                  selection is requested.
 * @param ActiveBootChain (out): out param to store obtained active
 *        BootChain configuration. = 0x0 for BootChain_A, 0x1 for BootChain_B
 * @return : DU_MCC_Ret_Type : success/failure in getting active BootChain selection.
 *
 * @pre
 *   DU_MCC_Init should already be called
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
DU_MCC_Ret_Type DU_MCC_GetActiveBootChain(
    void* handle,
    DU_MCC_Tegra_Device_ID Tegra_id,
    U8 *ActiveBootChain);

#ifdef __cplusplus
}
#endif

#endif // __DU_MCC_H__
