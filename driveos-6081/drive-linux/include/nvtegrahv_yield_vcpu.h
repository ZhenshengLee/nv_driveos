/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVTEGRAHV_YIELD_VCPU_H
#define NVTEGRAHV_YIELD_VCPU_H


#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup nvtegrahv_yield_vcpu_api VCPU Yielding API
 * API to yield VCPUs to the low priority VMs.
 * @ingroup nvtegrahv_library
 * @{
 */

/**
 * @page page_access_control NvHv VCPU-Yield Access Control
 * @section sec_access_control Common Privileges
 *
 * Description of custom abilites
 * 1) nvhv/yield_vcpu
 *     - Custom ability to yield VCPU of Guest VM to a low priority VM
 *
 * Common privileges required for all NvHv VCPU-Yield APIs
 *   - Service/Group Name: nvhv
 */

/**
 * @brief API to request Guest VM to yield VCPU to a low priority VM.\n\n
 *        This API yields the VCPU to a low priority VM if no other higher priority process than the caller is ready to\n
 *        run on this VCPU in the Guest VM. The API blocks till the time VCPU is yielded to the low priority VM is\n
 *        expired (i.e. timeout) or till the low priority VM returns the VCPU actively.\n
 *        An IVC communication channel is established between Guest VM and the low priority VM to yield and return VCPU.\n
 *        When the low priority VM releases it through the IVC communication channel or timeout occurs on Guest VM,\n
 *        the VCPU is returned to Guest VM.\n
 *        This API is only allowed during run-time. If user requests VCPU yielding in initialization or deinitialization phase,\n
 *        it returns an error. And timeout value 0 is an invalid input parameter because VCPU yielding for 0 microsecond is\n
 *        inappropriate. Minimum timeout value is 1 microsecond and maximum timeout value is configured in device tree\n
 *        "nvhv/max_timeout_us", the default value of the configuration is 1000000 in microseconds.\n
 *        If any invalid VM ID or timeout is given as a parameter, this API returns an error.
 *
 * @param[in] vm_id VM ID for low priority VM
 * @param[in] timeout_us Time for which the user is requesting to yield the VCPU to low priority VM\n
 *                       The minimum value is 1 microsecond and maximum value is configured in device tree
 *
 * @return
 * 		EOK       Success - VCPU is yielded successfully and returned to Guest VM by low priority VM releasing the VCPU\n
 * 		ENODEV    Failed to open NvHv device node\n
 * 		ETIMEDOUT Success - VCPU is yielded successfully and taken back to Guest VM by timeout\n
 * 		EFAULT    VCPU yielding operation failed\n
 * 		EINVAL    Invalid timeout_us parameter (zero or exceeded maximum value)
 *@pre
 *
 * @usage
 * - Allowed context for the API call
 *  - Interrupt handler: No
 *  - Signal handler: No
 *  - Thread-safe: Yes
 *  - Async/Sync: Sync
 *  - Re-entrant: Yes
 * - Required Privileges:
 *   - Custom abilities: nvhv/yield_vcpu
 * - API Group
 *  - Initialization: No
 *  - Run time: Yes
 *  - De-initialization: No
 *
 */
int32_t NvHvYieldVcpu(uint32_t vm_id, uint32_t timeout_us);

/**@} <!-- nvtegrahv_yield_vcpu_api VCPU Yielding --> */
#if defined(__cplusplus)
}
#endif

#endif /* NVTEGRAHV_YIELD_VCPU_H */
