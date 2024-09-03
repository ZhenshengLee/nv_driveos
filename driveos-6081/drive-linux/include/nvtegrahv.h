/*
 * Copyright (c) 2016-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * Description: NvTegraHv library provides interfaces to request virtualization
 * system for various services.
 * @note
 * Users should belong to 'nvhv' group defined in /etc/groups to access these interfaces.
 */

#ifndef INCLUDED_NV_TEGRA_HV_H
#define INCLUDED_NV_TEGRA_HV_H

#if defined(__cplusplus)
extern "C"
{
#endif

#include <stdint.h>

/**
 * @defgroup nvhv_user_api_group NVHV::User interface
 * @ingroup qnx_top_group
 * @{
 */

#define BUF_SIZE 	20 /* Max digits for vmid */
#define DISPLAY_VM 	"DISPLAY_VM" /* environ var that stores display vmid */

/** @cond HIDDEN_SYMBOLS */
/**
 * @brief function pointer to NvHvGetOsVmId
 */
typedef int32_t (*PfnNvHvGetOsVmId)(uint32_t *vmid);
/**
 * @brief function pointer to NvHvCheckOsNative
 */
typedef int32_t (*PfnNvHvCheckOsNative)(void);
#if (NV_IS_SAFETY == 0)
/**
 * @brief function pointer to NvHvGetDisplayVmId
 */
typedef int32_t (*PfnNvHvGetDisplayVmId)(uint32_t *dpvmid);
int32_t NvHvGetDisplayVmId(uint32_t *dpvmid);
#endif
/** @endcond */

/**
 * @page page_access_control NvHv Common Access Control
 * @section sec_access_control Common Privileges
 *
 * Description of custom abilites
 * 1) nvhv/get_gid
 *     - Custom ability to get Vitual Machine ID
 *
 * 2) nvhv/get_hv_sysinfo
 *     - Custom ability to get hypervisor system information
 *
 * Common privileges required for all NvHv Common APIs
 *   - Service/Group Name: nvhv
 */


/**
 * @brief API to check whether running on Virtualized System or Native.
 * This API internally checks if "/dev/nvhv" node is present or not. This
 * devnode is present only on virtualized system.
 *
 * @return
 * 		0 Running on Virtualized System\n
 * 		1 Native\n
 * 		-1 Failed
 *
 * @usage
 * - Allowed context for the API call
 *  - Interrupt handler: No
 *  - Signal handler: No
 *  - Thread safe: Yes
 *  - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *  - Initialization: Yes
 *  - Run time: Yes
 *  - De-initialization: No
 *
 * SWUD_ID: QNXBSP_NVHV_NVTEGRAHV_LIB_01
 */
int32_t NvHvCheckOsNative(void);

/**
 * @brief API to get Vitual Machine ID.
 * This API requests virtualization system to get current VM ID.
 *
 * @param[out] vmid Guest VM ID. Memory for this shall be allocated by calling function/process.
 *
 * @return
 * 		EOK Success\n
 * 		ENODEV Failed to open NvHv device node\n
 * 		EINVAL Invalid argument\n
 * 		EFAULT devctl call to NvHv device node returned failure
 *
 * @usage
 * - Allowed context for the API call
 *  - Interrupt handler: No
 *  - Signal handler: No
 *  - Thread safe: Yes
 *  - Async/Sync: Sync
 * - Required Privileges:
 *   - Custom abilities: nvhv/get_gid
 * - API Group
 *  - Initialization: Yes
 *  - Run time: Yes
 *  - De-initialization: No
 *
 * SWUD_ID: QNXBSP_NVHV_NVTEGRAHV_LIB_02
 */
int32_t NvHvGetOsVmId(uint32_t *vmid);

/**
 * @brief enum for OS type
 */
typedef enum
{
    /** OS is running as non native */
    Os_Non_Native = 0,
    /** OS is running as Native */
    Os_Native = 1,
    /** OS detection failed */
    Os_Detection_Failed = -1
} NvOsType;

/** @} */ /* nvhv_user_api_group */

#if defined(__cplusplus)
}
#endif  /* __cplusplus */

#endif
