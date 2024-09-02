/*
 * Copyright (c) 2022-2023, NVIDIA Corporation. All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation. Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */
#ifndef __IST_CLIENT_MCC_H__
#define __IST_CLIENT_MCC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//==================================================================================================
// The following section defines the ISTClient callback functions that are registered with an MCC
// library and then get called by MCC lib to handle commands received from IST_Manager
//==================================================================================================

/**
 * @brief IST diagnostic result structure (opaque)
 */
typedef struct ist_client_result {
	uint8_t hw_result;        /**< hw_result value (opaque to mcc) */
	uint8_t reserved_0;       /**< reserved_0 value (opaque to mcc) */
	uint8_t sw_rpl_status;    /**< sw_rpl_status value (opaque to mcc) */
	uint8_t sw_preist_status; /**< sw_preist_status value (opaque to mcc) */
} ist_client_result_t;

/**
 * @addtogroup ist_client_mcc_api
 * @{
 */

/**
 * Setup communication with MCU and send KIST result
 *
 * This primitive is responsible for setting up communication with MCU, and
 * sending KIST result to MCU. In the sample implementation, this is done by
 * setting up an NvSciIpc channel to MCU communication coordinator daemon
 * process (MCC daemon), notifying MCU that results are ready, then sending
 * those results when requested by MCU.
 * If this function returns an error, it is considered fatal. In particular,
 * IST_Client will not retry sending the result by calling this API again.
 * The sample implementation makes multiple attempts until it can establish
 * NvSciIpc channel with MCC daemon. Once connected it retries notifying MCU
 * until MCU requests the results, and results are transferred successfully.
 * When KIST result is sent to MCU, or when a fatal error is encountered,
 * MCC library shall drop non-needed privileges before returning.
 *
 * @param [in] result: KIST result data
 * @returns 0 on success, <0 on fatal error.
  *
 * @note
 * - Allowed context for the API call
 *    - Interrupt: No
 *    - Signal handler: No
 *    - Thread-safe: No
 *    - Sync/Async: Sync
 *    - Re-entrant: No
 * - Required Privileges:
 *    - Sample implementation requires PROCMGR_AID_INTERRUPTEVENT, PROCMGR_AID_PUBLIC_CHANNEL, NvSciIpcEndpoint.
 *    - OEM implementation might require different privileges.
 * - Operation Mode
 *    - Init: Yes
 *    - Run time: Yes
 *    - Deinit: No
 */
extern int ISTClient_mcc_send_results(ist_client_result_t result);

/** @} */ /* ist_client_mcc_api */

#ifdef __cplusplus
}
#endif

#endif // __IST_CLIENT_MCC_H__
