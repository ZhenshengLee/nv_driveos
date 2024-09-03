/*
 * Copyright (c) 2022, NVIDIA Corporation. All Rights Reserved.
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
 * IST diagnostic result structure
 */
typedef struct ist_client_result {
	uint8_t hw_result;        /**< hw_result value (opaque to mcc) */
	uint8_t reserved_0;       /**< reserved_0 value (opaque to mcc) */
	uint8_t sw_rpl_status;    /**< sw_rpl_status value (opaque to mcc) */
	uint8_t sw_preist_status; /**< sw_preist_status value (opaque to mcc) */
} ist_client_result_t;

/**
 * Do everything needed for MCC and send KIST result
 *
 * @param [in] result: KIST result data
 * @returns 0 on success
 */
extern int ISTClient_mcc_send_results(ist_client_result_t result);

#ifdef __cplusplus
}
#endif

#endif // __IST_CLIENT_MCC_H__
