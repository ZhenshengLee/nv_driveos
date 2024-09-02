/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file NvEpl.h
 * @brief EPL resource manager external header
 */

#ifndef NVEPL_H
#define NVEPL_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include <SafetyServiceType.h>

/*
 * GID offsets
 */
#define GID_OFFSET_ERR_REPORT (1U)
#define MAX_DEVCTL_NBYTES     (2U * 1024U)

/**
 * @brief Device node for resource manager
 *
 */
#define EPL_DEVPATH "/dev/epdaemon"

/**
 * @brief API to report SW error to FSI via TOP2 HSP
 *
 *
 * @param[in]  ErrorReport Error report that contains Instance id, Error Id
 *                         and Metadata
 *
 * @return ::SS_E_OK - if error report is successfully propagated to FSI
 *           SS_E_PRECON - Invalide input param or
 *                         If channel between EPL and daemon is not established.
 *           SS_E_NOK - any other failure
 *
 * @pre Error propagation daemon should be running and NvEplInit() must be successfully completed.
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync mode: Sync
 *   - Required privileges: EPD/SwErrReport:<reporter_id> custom ability is required.
 *   - API group
 *      - Init: No
 *      - Runtime: Yes
 *      - De-Init: No
 */
SS_ReturnType_t NvEpl_ReportError(const SS_ErrorReportFrame_t ErrorReport);

#if __QNX__
/**
 * @brief API to report SW error to FSI using MISC EC
 *
 *
 * @param[in]  errNumber - Generic SW error number through which error
 *             needs to be reported.
 * @param[in]  swErrCode - Client Defined Error Code, which will be
 *             forwarded to application.
 *
 * @return ::SS_E_OK - if register write is successful.
 *           SS_E_PRECON – Invalid input param or previous error report is in progress.
 *           SS_E_NOK - any other failure
 *
 * @pre Error propagation daemon should be running and NvEplInit() must be successfully completed.
 *      Misc registers (MISCREG_MISC_EC_ERRX_SW_ERR_CODE_0 and MISCREG_MISC_EC_ERRX_SW_ERR_ASSERT_0)
 *      should be mapped to process address space (done in NvEplInit()).
 *      Generic sw error number passed as input argument to this API should be enabled in the
 *      epl client configuration in dt.
 *      This API is only available for Qnx.
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 *   - Required privileges: mem_phys ability for Misc EC error and assert registers is required
 *   - API group
 *    - Init: No
 *    - Runtime: Yes
 *    - De-Init: No
 */
SS_ReturnType_t NvEpl_Report_MISC_EC_Error(uint8_t errNumber, uint32_t swErrCode);

/**
 * @brief API to check if SW error can be reported via Misc EC
 *        by reading and checking Misc EC error status register value.
 *        It is required to be used only if process need to access MISC_EC
 *
 *
 * @param[in]  errNumber - Generic SW error number for which status needs to
 *                         enquired - [0 to 4].
 * @param[out] status - true - SW error can be reported
 *                      false - SW error can not be reported because previous error
 *                              is still active. User need to retry later.
 * @return ::SS_E_OK - if register read is successful.
 *           SS_E_PRECON – Invalid input param
 *           SS_E_NOK - any other failure
 *
 * @pre Error propagation daemon should be running and NvEplInit() must be successfully completed.
 *      Misc EC error status register MISC_EC_ERRSLICE0_MISSIONERR_STATUS_0 should
 *      be mapped to process address space (done in NvEplInit()). NvEplInit() must be successful.
 *      This API is only available for Qnx.
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 *   - Required privileges: mem_phys ability for Misc EC mission error status register is required
 *   - API group
 *    - Init: No
 *    - Runtime: Yes
 *    - De-Init: No
 */
SS_ReturnType_t NvEpl_GetMiscEcErrStatus(uint8_t errNumber, bool* status);
#endif

/**
 * @brief API to initialize NvEpl
 *
 * @param None
 * @return : SS_E_OK - if init is successful.
 *           SS_E_NOK - any other failure
 *
 * @pre Error propagation daemon should be running.
 *
 * @usage
 * - Allowed context for the API call
 *   - Cancellation point: No
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No, called as constructor
 *   - Re-entrant: No
 *   - Async/Sync mode: Sync
 *   - Required privileges: None.
 *   - API group
 *    - Init: Yes
 *    - Runtime: No
 *    - De-Init: No
 */
SS_ReturnType_t NvEplInit(void);
#if defined(__cplusplus)
}
#endif

#endif /* NVEPL_H */

