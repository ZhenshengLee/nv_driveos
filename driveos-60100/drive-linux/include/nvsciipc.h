/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_NVSCIIPC_H
#define INCLUDED_NVSCIIPC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include "nvscierror.h"
#include "nvscievent.h"

/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSci Inter-Process Communication </b>
 *
 * @version 1.4
 *
 * @section 1.4 Jun/29/2023
 * - Add NvSciIpcEnableNotification()
 * ---------------------------------------------------------------------------
 * @section 1.3 Jun/26/2023
 * - Add NV_SCI_IPC_EVENT_ASYNC_ERROR to NvSciIpcGetEventSafe()
 * - Add NvSciIpcAsyncError()
 * - Remove Runtime support of NvSciIpcResetEndpointSafe()
 * ---------------------------------------------------------------------------
 * @section 1.2 Mar/07/2023
 * - Deprecate non-safe APIs in safety build
 *      NvSciIpcCloseEndpoint
 *      NvSciIpcResetEndpoint
 *      NvSciIpcRead
 *      NvSciIpcWrite
 *      NvSciIpcGetEvent
 *      NvSciIpcSetQnxPulseParam
 * ---------------------------------------------------------------------------
 * @section 1.1 Dec/15/2022
 * - Add NvSciError_NotPermitted error for incorrect API group usage
 *   in QNX safety only (NvDVMS).
 *       NvSciIpcInit()
 *       NvSciIpcOpenEndpoint()
 *       NvSciIpcOpenEndpointWithEventService()
 *       NvSciIpcGetEventNotifier()
 *       NvSciIpcCloseEndpointSafe()
 *       NvSciIpcSetQnxPulseParamSafe()
 * ---------------------------------------------------------------------------
 * @section 1.0 Jun/23/2022
 * - Add NvSciIpcGetEventSafe()
 * - Add NvSciIpcWaitEventQnx() : QNX only
 * - Add NvSciIpcSetQnxPulseParamSafe() : QNX only
 * - Add NvSciIpcInspectEventQnx() : QNX only
 * - Add NvSciIpcCheckVersionCompatibility()
 * - Add NvSciIpcMajorVersion, NvSciIpcMinorVersion constant
 */

/* use constant global version variable instead of macro for consistency with
 * version check API of existing NvSci family
 */

/** @brief NvSciIpc API Major version number */
static const uint32_t NvSciIpcMajorVersion = 1U;

/** @brief NvSciIpc API Minor version number */
static const uint32_t NvSciIpcMinorVersion = 4U;

/**
 * @defgroup nvsci_group_ipc Inter-Process Communication
 * IPC and Event Service APIs
 *
 * @ingroup nvsci_top
 * @{
 */
/**
 * @defgroup nvsci_ipc_api IPC APIs
 *
 *
 * @ingroup nvsci_group_ipc
 * @{
 *
 * The NvSciIpc library provides interfaces for any two entities in a system to
 * communicate with each other irrespective of where they are placed. Entities
 * can be in:
 * - Different threads in the same process
 * - The same process
 * - Different processes in the same VM
 * - Different VMs on the same SoC
 * @if (SWDOCS_NVSCIIPC_STANDARD)
 * - Different SoCs
 * @endif
 *
 * Each of these different boundaries will be abstracted by a library providing
 * unified communication (Read/Write) APIs to entities. The communication
 * consists of two bi-directional send/receive queues.
 *
 * Programming model in using NvSciIpc library is explained in SDK developer
 * guide with examples.
 *
 * @pre Start-up sequence
 * - When NvSciIpc APIs are used, the initialization sequence must be exactly
 * the same as below:
 * Here two cases, with or without NvSciEventService.
 * - Call flow with NvSciIpc
 * ~~~~~~~~~~~~~~~~~~~~~
 *    NvSciIpcInit()
 *    NvSciIpcOpenEndpoint()
 *    NvSciIpcSetQnxPulseParamSafe() (QNX OS-specific) or
 *    NvSciIpcGetLinuxEventFd() (Linux OS-specific)
 *    NvSciIpcGetEndpointInfo()
 *    NvSciIpcResetEndpointSafe()
 * ~~~~~~~~~~~~~~~~~~~~~
 * - Call flow with NvSciIpc and NvSciEventService library
 * NvSciEventService provides APIs that replace OS-specific event-blocking API.
 * They are only compatible with an endpoint which is opened with
 * NvSciOpenEndpointWithEventService().
 * ~~~~~~~~~~~~~~~~~~~~~
 *    NvSciEventLoopServiceCreateSafe() to get eventLoopService
 *    NvSciIpcInit()
 *    NvSciIpcOpenEndpointWithEventService()
 *    NvSciIpcGetEventNotifier()
 *    NvSciEventNotifier::SetHandler()
 *    NvSciIpcGetEndpointInfo()
 *    NvSciIpcResetEndpointSafe()
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * @pre Single thread handling Tx and Rx data
 * - For Inter-VM and Inter-Process backend applications on QNX OS,
 * only a single event-blocking OS API (e.g. NvSciIpcWaitEventQnx(),
 * NvSciEventLoopService::WaitForEvent()) is allowed per endpoint
 * in the same process boundary.
 * If a client application tries to use receiving and sending thread separately
 * for the same endpoint handle, the event-blocking OS APIs must be used in a
 * single thread in order to receive remote notification.
 * Once a remote notification arrives in the thread, the notification should be
 * forwarded to the other thread using the same endpoint handle through any OS
 * synchronization method (e.g. sem_post, pthread_cond_signal or
 * MsgSendPulse etc.)
 * Single thread usage is recommended to handle both TX and RX data.
 *
 * @pre Getting events before using Read/Write NvSciIpc API
 * - Before using any Read/Write APIs, the user must check if
 * @ref NV_SCI_IPC_EVENT_READ or @ref NV_SCI_IPC_EVENT_WRITE event is available
 * through NvSciIpcGetEventSafe().
 * NvSciIpcGetEventSafe() has additional support to establish connection between
 * two endpoint software entities.
 *
 * @pre Checksum for data integrity
 * - NvSciIpc does not provide checksum for data to be transferred or check for
 * data integrity error.
 * If such a mechanism is required, the client user should implement checksum
 * processing routine for data.
 *
 * @pre Use ChannelCreatePulsePool() - QNX OS
 * - In using legacy event handling (without NvSciEventService) on NvSciIpc,
 * user has to create chid and coid using ChannelCreate_r() and
 * ConnectAttach_r() before calling NvSciIpcSetQnxPulseParamSafe().
 * This ChannelCreate() uses global pulse message pool. If privileged and
 * non-privileged processes use the same global pool, unprivileged processes
 * can potentially interfere with privileged processes by receiving and not
 * handling large numbers of pulses (thereby causing denial of service attack).
 * So, ChannelCreatePulsePool() API is recommended to use fixed private pool of
 * pulses instead of using ChannelCreate_r().
 * ~~~~~~~~~~~~~~~~~~~~~
 * struct nto_channel_config {
 *     struct sigevent event;    // initialize using SIGEV_SEM_INIT() macro
 *     unsigned num_pulses;      // threshold to trigger semaphore event
 *     unsigned rearm_threshold; // 0 ~ num_pulses
 *     unsigned options;         // _NTO_CHO_CUSTOM_EVENT
 *     unsigned reserved[3];
 * }
 * ~~~~~~~~~~~~~~~~~~~~~
 * - The num_pulses should be based on the number of event notifications that can occur at the same time.
 * The notifications that can occur on an endpoint are reset /read /write and are independent of each other,
 * so at least 3 notifications can occur on one endpoint.
 * ~~~~~~~~~~~~~~~~~~~~~
 * num_pulses = 3 * number of endpoints
 * ~~~~~~~~~~~~~~~~~~~~~
 * - More information can be found in QNX OS manual page.
 * - In order to detect IVC signalling storm, user needs to create separate thread
 * to receive semaphore event which is set in nto_channel_config structure and
 * call sem_wait() in that thread.
 * NvSciIpcInspectEventQnx() API returns NvSciIpcEndpoint handle in which IVC
 * signalling storm happens.
 * User can decide post action (e.g. close endpoint, restart process, deinit
 * NvSciIpc etc.) per system usecase scenario after detecting issue.
 *
 * @note
 * <b>Configuring thread pool of resource manager - QNX OS</b>
 * - NvSciIpc resource manager (io-nvsciipc) uses thread pool to manage
 * concurrent request from multiple NvSciIpc client processes using NvSciIpc
 * library. io-nvsciipc is used during opening endpoint.
 * Users should evaluate thread pool capacity of io-nvsciipc then
 * configure them with -t option in startup script.
 * Thread pool capacity for NvSciIPC can be evaluated based on number of
 * parallel outstanding NvSciIPC requests, at any point of time, that are
 * expected in the system. Default value of thread pool capacity is 10.
 *
 * @note
 * <b>When to use blocking API</b>
 * - Users must call OS event-blocking API to wait for an event when
 * NvSciIpcGetEventSafe() does not return desired events.
 * The following are OS event-blocking API examples:
 *  - QNX  : NvSciIpcWaitEventQnx()
 *  - LINUX: select(), poll() etc.
 *  - NvSciEventService: NvSciEventLoopService::WaitForEvent(),<br/>
 *                       NvSciEventLoopService::WaitForMultipleEvents()
 *
 * - If user process needs to wait for events from multiple remote NvSciIpc
 * endpoint processes, use single blocking call from single thread instead of
 * using blocking call per endpoint thread. This is recommended to improve
 * performance by avoiding thread creation per endpoint.
 * NvSciEventLoopService::WaitForMultipleEvents() blocking call is suitable for
 * this use case.
 *
 * @note
 * <b>Consideration when choosing Read/Write APIs</b>
 * - Using NvSciIpcReadSafe() and NvSciIpcWriteSafe() is recommended rather than following
 * Read/Write APIs. See detail constraints of API in each function description.
 *  - NvSciIpcReadGetNextFrame()
 *  - NvSciIpcWriteGetNextFrame()
 *  - NvSciIpcReadAdvance()
 *  - NvSciIpcWriteAdvance()
 * However, above functions are better to avoid extra memory copy.
 *
 * @note
 * <b>Maximum number of endpoints</b>
 * - One NvSciIpc client process is allowed to open up to 500 endpoints.
 * QNX NvSciIpc opens two device nodes in opening endpoint.
 * QNX OS kernel supports 1024
 * 100 open channels without disabling kernel preemption. User client needs
 * one channel/connection pair to receive an endpoint notification.
 *
 * @note
 * <b>Concurrent read/write</b>
 * - Client processes who want concurrent read and write operation on endpoints
 * need to open two endpoints, one for read and the other for write operation.
 * Read and write operation on different endpoint work exclusively without any
 * external locking mechanism in multiple threads. Channel memory consumption
 * will be doubled in using two endpoints.
 *
 */

/*******************************************************************/
/********************* ACCESS CONTROL ******************************/
/*******************************************************************/

/**
 * @page page_access_control NvSciIpc Access Control
 * @section sec_access_control Common Privileges
 *
 * <b>Description of QNX custom abilites</b>
 *
 *  1) NvSciIpcEndpoint:{List of VUIDs}
 *     - required VUID list is set by QNX_BSP::IOLauncher with -T option
 *     - This ability is used to authenticate client process from NvSciIpc
 *       Driver
 *
 *  2) NvSciC2cPcieEndpoint:{List of SGIDs}
 *     - required SGIDs list is set by QNX_BSP::IOLauncher with -T option
 *     - This ability is used to authenticate client process which try to access any
 *       Inter-Chip PCIe device-node provided by NvSciC2c resmgr/process.
 *
 * <b>Common QNX privileges required for all NvSciIpc APIs</b>
 *
 *   1) Intra-VM and Inter-VM
 *   - Service/Group Name:
 *     proc_boot (GID: 40029), libc (GID: 40002), libslog2 (GID: 40006),
 *     libnvivc (GID: 45031), libnvsciipc (GID: 45047),
 *     libnvos_s3_safety (GID: 45037), libnvdvms_client (GID: 45112)
 *
 *   2) Inter-Chip PCIE
 *   - Service/Group Name:
 *     nvsciipc (GID: 2000), nvsys (GID: 3000), devg_nvrm_nvmap (GID: 10100), /usr/ (GID: 40032),
 *     devg_nvrm_nvhost (GID: 10140), /usr/libnvidia/ (GID: 55046),
 *     nvscic2c_pcie_epc_1/2/.../12 (GID: 26001, 26002,..., 26012),
 *     nvscic2c_pcie_epf_1/2/.../12 (GID: 26101, 26002,..., 26112),
 *     libc (GID: 40002), libm (GID: 40005), libslog2 (GID: 40006), libgnat (GID: 40012),
 *     libcatalog (GID: 40052), libnvivc (GID: 45031), libnvos_s3_safety (GID: 45037),
 *     libnvos (GID: 45038), libnvrm_gpu (GID: 45042), libnvrm_host1x (GID: 45043),
 *     libnvscievent (GID: 45046), libnvsciipc (GID: 45047), libnvrm_mem (GID: 45069),
 *     libnvsocsys (GID: 45071), libnvdvms_client (GID: 45112), libnvscibuf (GID: 55016),
 *     libnvscicommon (GID: 55017), libnvscisync (GID: 55019), libnvscic2ccommon (GID: 55075),
 *     libnvscic2c (GID: 55076), libnvscic2cpcie (GID: 55077)
 */

/*******************************************************************/
/************************ DATA TYPES *******************************/
/*******************************************************************/

/**
 * @brief Handle to the NvSciIpc endpoint.
 */
typedef uint64_t NvSciIpcEndpoint;

typedef struct NvSciIpcEndpointInfo NvSciIpcEndpointInfo;

/**
 * @brief Defines information about the NvSciIpc endpoint.
 */
struct NvSciIpcEndpointInfo {
    /** Holds the number of frames. */
    uint32_t nframes;
    /** Holds the frame size in bytes. */
    uint32_t frame_size;
};

/**
 * Specifies maximum Endpoint name length
 * including null terminator
 */
#define NVSCIIPC_MAX_ENDPOINT_NAME   64U

/* NvSciIPC Event type */
/** Specifies the IPC read event. */
#define	NV_SCI_IPC_EVENT_READ           0x01U
/** Specifies the IPC write event. */
#define	NV_SCI_IPC_EVENT_WRITE          0x02U
/** Specifies the IPC connection established event. */
#define	NV_SCI_IPC_EVENT_CONN_EST       0x04U
/** Specifies the IPC connection reset event. */
#define	NV_SCI_IPC_EVENT_CONN_RESET     0x08U
/** Specifies the IPC write fifo empty event. */
#define	NV_SCI_IPC_EVENT_WRITE_EMPTY    0x10U
/** Specifies the IPC asynchronous error event. */
#define NV_SCI_IPC_EVENT_ASYNC_ERROR    0x20U
/** Specifies single event mask to check IPC connection establishment */
#define	NV_SCI_IPC_EVENT_CONN_EST_ALL (NV_SCI_IPC_EVENT_CONN_EST | \
    NV_SCI_IPC_EVENT_WRITE | NV_SCI_IPC_EVENT_WRITE_EMPTY | \
    NV_SCI_IPC_EVENT_READ)

/** infinite timeout for NvSciIpcWaitEventQnx() */
#define NVSCIIPC_INFINITE_WAIT -1LL

/* NvSciIpc Asynchronous erros */
/** Indicates there is eDMA error during PCIE operation. */
#define NV_SCI_ASYNC_PCIE_EDMA_XFER_ERROR            0x1U
/** Indicates there is uncorrectable fatal error during PCIE operation. */
#define NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_FATAL    0x2U
/** Indicates there is uncorrectable non fatal error during PCIE operation. */
#define NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_NONFATAL 0x4U
/** Indicates there is validation error. */
#define NV_SCI_ASYNC_PCIE_VALIDATION_ERROR           0x8U

/*******************************************************************/
/********************* FUNCTION TYPES ******************************/
/*******************************************************************/

/**
 * @brief Initializes the NvSciIpc library.
 *
 * This function parses the NvSciIpc configuration file and creates
 * an internal database of NvSciIpc endpoints that exist in a system.
 *
 * @return ::NvSciError, the completion code of the operation.
 * - ::NvSciError_Success      Indicates a successful operation.
 * - ::NvSciError_NotPermitted Indicates initialization has failed.
 *                             Indicates incorrect API group usage.
 * - ::NvSciError_InvalidState Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcInit(void);

/**
 * @brief De-initializes the NvSciIpc library.
 *
 * This function cleans up the NvSciIpc endpoint internal database
 * created by NvSciIpcInit().
 *
 * @note This API can be called in Init mode to release resources
 *       in error handling or to test functionality.
 *
 * @return @c void
 *
 * @pre Invocation of NvSciIpcInit() must be successful.
 *
 * @pre Closing endpoints before deinitialization
 * - Before calling this API, the user shall ensure all existing opened
 * endpoints are closed by NvSciIpcCloseEndpointSafe().
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
void NvSciIpcDeinit(void);

/**
 * @brief Opens an endpoint with the given name.
 *
 * The function locates the NvSciIpc endpoint with the given name in the
 * NvSciIpc configuration table in the internal database, and returns a handle
 * to the endpoint if found. When the operation is successful, endpoint can
 * utilize the allocated shared data area and the corresponding signaling
 * mechanism setup. If the operation fails, the state of the NvSciIpc endpoint
 * is undefined.
 * In case of QNX OS, in order to authenticate user client process, NvSciIpc
 * uses custom ability "NvSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  endpoint The name of the NvSciIpc endpoint to open.
 * @param[out] handle   A handle to the endpoint on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 * - ::NvSciError_Busy               Indicates the @a endpoint is already in
 *                                   use.
 * - ::NvSciError_InsufficientMemory Indicates memory allocation failed for
 *                                   the operation.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *                                   Indicates incorrect API group usage.
 *
 * @pre Invocation of NvSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX):
 *   - "nvdvms/ClientCommGetState"
 *   - "NvSciIpcEndpoint"
 *   - "NvSciC2cPcieEndpoint" (Inter-Chip only)
 *   - "nvsys/system_info" (Inter-Chip only)
 *   - "NvMap/Interfaces:17,19" (Inter-Chip only)
 *   - When used in streaming mode (Inter-Chip only):
 *     - "NvHost/Interfaces:1-2"
 *     - "NvHost/Waiter:1"
 *       OR "NvHost/Waiter:N", where 'N' is the
 *       number of Inter-Chip endpoints application uses in the same process.
 *   - PROCMGR_AID_PROT_EXEC (Inter-Chip only)
 *   - PROCMGR_AID_MAP_FIXED (Inter-Chip only)
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcOpenEndpoint(const char *endpoint, NvSciIpcEndpoint *handle);

/**
 * @brief Opens an endpoint with the given name and event service.
 *
 * This API provides same functionality as NvSciIpcOpenEndpoint().
 * But, it requires additional event service abstract object as an input
 * parameter to utilize NvSciEventService infrastructure.
 * NvSciEventService can be created through NvSciEventLoopServiceCreateSafe().
 * NvSciIpcGetEventNotifier() can be used only when this API is invoked
 * successfully.
 * In case of QNX OS, in order to authenticate user client process, NvSciIpc
 * uses custom ability "NvSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  endpoint      The name of the NvSciIpc endpoint to open.
 * @param[out] handle        A handle to the endpoint on success.
 * @param[in]  eventService  An abstract object to use NvSciEventService
 *                           infrastructure.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 * - ::NvSciError_Busy               Indicates the @a endpoint is already in
 *                                   use.
 * - ::NvSciError_InsufficientMemory Indicates memory allocation failed for
 *                                   the operation.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *                                   Indicates incorrect API group usage.
 *
 * @pre Invocation of NvSciEventLoopServiceCreateSafe() must be successful.
 *      Invocation of NvSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX):
 *   - "nvdvms/ClientCommGetState"
 *   - "NvSciIpcEndpoint"
 *   - "NvSciC2cPcieEndpoint" (Inter-Chip only)
 *   - "nvsys/system_info" (Inter-Chip only)
 *   - "NvMap/Interfaces:17,19" (Inter-Chip only)
 *   - When used in streaming mode (Inter-Chip only):
 *     - "NvHost/Interfaces:1-2"
 *     - "NvHost/Waiter:1"
 *       OR "NvHost/Waiter:N", where 'N' is the
 *       number of Inter-Chip endpoints application uses in the same process.
 *   - PROCMGR_AID_PROT_EXEC (Inter-Chip only)
 *   - PROCMGR_AID_MAP_FIXED (Inter-Chip only)
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcOpenEndpointWithEventService(const char *endpoint,
    NvSciIpcEndpoint *handle, NvSciEventService *eventService);

/**
 * @brief Get NvSciIpc event notifier.
 *
 * This API is used to connect NvSciIpc event handling with OS-provided
 * event interface.
 * It also utilizes NvSciEventService infrastructure.
 *
 * @note This API is only compatible with an endpoint that is opened with
 *       NvSciIpcOpenEndpointWithEventService()
 *
 * @param[in]  handle         NvSciIpc endpoint handle.
 * @param[out] eventNotifier  A pointer to NvSciEventNotifier object on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_InsufficientMemory Indicates memory allocation failed for the
 *                                   operation.
 * - ::NvSciError_ResourceError      Indicates not enough system resources.
 * - ::NvSciError_NotPermitted       Indicates incorrect API group usage.
 *
 * @pre Invocation of NvSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX):
 *   - PROCMGR_AID_INTERRUPTEVENT (Inter-VM only)
 *   - PROCMGR_AID_PUBLIC_CHANNEL
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcGetEventNotifier(NvSciIpcEndpoint handle,
               NvSciEventNotifier **eventNotifier);

/**
 * @brief Closes an endpoint with the given handle.
 *
 * @warning This API is deprecated and returns without doing anything in safety
 * build.
 * Use NvSciIpcCloseEndpointSafe() instead of this.
 */
void NvSciIpcCloseEndpoint(NvSciIpcEndpoint handle);

/**
 * @brief Closes an endpoint with the given handle (safety version)
 *
 * The function frees the NvSciIpc endpoint associated with the given @a handle.
 *
 * @note This API can be called in Init mode to release resources
 *       in error handling or to test functionality.
 *
 * @param[in] handle A handle to the endpoint to close.
 * @param[in] clear  Reserved for future use and any value has no change to the
 *                   function behavior.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_NotPermitted       Indicates incorrect API group usage.
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() or
 *      NvSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @pre Deleting NvSciEventNotifier
 * - Before calling NvSciIpcCloseEndpointSafe(), event notifier associated with
 * the closing endpoint through NvSciIpcGetEventNotifier() must be deleted by
 * NvSciEventNotifier::Delete().
 *
 * @pre Closing Inter-Chip Endpoint
 * - When used in streaming mode, before calling
 * NvSciIpcCloseEndpointSafe() for (INTER_CHIP, PCIE) endpoint,
 * the user must ensure all fences submitted to NvIpc for Inter-Chip
 * communication for signaling are expired and also all NvSciBufObjs
 * and NvSciSyncObjs mappings registered, exported and imported with
 * NvIpc for Inter-Chip communication are un-registered and released.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvSciError NvSciIpcCloseEndpointSafe(NvSciIpcEndpoint handle, bool clear);

/**
 * @brief Resets an endpoint.
 *
 * @warning This API is deprecated and returns without doing anything in safety
 * build.
 * Use NvSciIpcResetEndpointSafe() instead of this.
 */
void NvSciIpcResetEndpoint(NvSciIpcEndpoint handle);

/**
 * @brief Resets an endpoint. (safety version)
 *
 * Initiates a reset on the endpoint and notifies the remote endpoint.
 * Once this API is called, all existing data in channel will be discarded.
 * After invoking this function, client user shall call NvSciIpcGetEventSafe()
 * to get specific event type (READ, WRITE etc.).
 *
 * @param[in] handle A handle to the endpoint to reset.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 *
 * @pre Invocation of NvSciIpcSetQnxPulseParamSafe() or
 *      NvSciIpcGetLinuxEventFd() or
 *      NvSciIpcGetEventNotifier() must be successful.
 *
 * @pre When to reset endpoint
 * - This API must be called exactly once per endpoint handle during Init
 * mode to complete the initialization sequence before using the endpoint for
 * communication,and at most once during De-init mode only if both endpoints
 * are in the established state (e.g. synchronization for closing channel). In case of
 * De-Init mode, NvSciIpcGetEventSafe() must NOT be called after
 * NvSciIpcResetEndpointSafe() is called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvSciError NvSciIpcResetEndpointSafe(NvSciIpcEndpoint handle);

/**
 * @brief Reads a frame from the endpoint.
 *
 * @warning This API is deprecated and returns NvSciError_NotSupported without
 * doing anything in safety build.
 * Use NvSciIpcReadSafe() instead of this.
 */
NvSciError NvSciIpcRead(NvSciIpcEndpoint handle, void *buf, size_t size,
    int32_t *bytes);

/**
 * @brief Reads a frame from the endpoint (safety version)
 *
 * This function copies a new frame contents into a buffer and advances to the
 * next frame. If the destination buffer is smaller than the configured
 * frame size of the endpoint, the trailing bytes are discarded.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If read channel of the endpoint was previously full, then the function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpointSafe(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * @param[in]  handle The handle to the endpoint to read from.
 * @param[out] buf    A pointer to a destination buffer to receive the contents
 *                    of the next frame.
 * @param[in]  size   The number of bytes to copy from the frame,
 *                    not to exceed the length of the destination buffer and
 *                    configured frame size of the endpoint.
 * @param[out] bytes  The number of bytes read on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates read channel is empty and
 *                                   the read operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcReadSafe(NvSciIpcEndpoint handle, void *buf, uint32_t size,
    uint32_t *bytes);

/**
 * @brief Get a pointer to the read frame from the endpoint.
 *
 * This is a non-blocking call.
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpointSafe(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * @param[in]  handle The handle to the endpoint to read from.
 * @param[out] buf    A pointer to a destination buffer to receive
 *                    the contents of the next frame on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates read channel is empty and
 *                                   the read operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @pre No overlapped read operation
 * - Between NvSciIpcReadGetNextFrame() and NvSciIpcReadAdvance(), the user must not perform
 * any other NvSciIpc read operations with the same endpoint handle.
 *
 * @pre No use of invalid pointer to read frame
 * - Once a read frame is released by NvSciIpcReadAdvance(), the user must not use previously
 * returned pointer of NvSciIpcReadGetNextFrame() since it is already invalid.
 *
 * @pre No write operation with pointer to read frame
 * - The user shall not write through a returned pointer of NvSciIpcReadGetNextFrame().
 * This is protected by a const volatile pointer return type.
 *
 * @pre Copy data before using it
 * - The user shall not read the same memory location multiple times. If required, copy
 * specific memory location to a local buffer before using it.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcReadGetNextFrame(NvSciIpcEndpoint handle,
    const volatile void **buf);

/**
 * @brief Advance to the next read frame of the endpoint.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If a read channel of the endpoint was previously full, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpointSafe(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * @note This API is used together with NvSciIpcReadGetNextFrame().
 * Refer to precondition of NvSciIpcReadGetNextFrame()
 *
 * @param[in] handle The handle to the endpoint to read from.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates read channel is empty and
 *                                   the read operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcReadAdvance(NvSciIpcEndpoint handle);

/**
 * @brief Writes a frame to the endpoint.
 *
 * @warning This API is deprecated and returns NvSciError_NotSupported without
 * doing anything in safety build.
 * Use NvSciIpcWriteSafe() instead of this.
 */
NvSciError NvSciIpcWrite(NvSciIpcEndpoint handle, const void *buf, size_t size,
    int32_t *bytes);

/**
 * @brief Writes a frame to the endpoint. (safety version)
 *
 * If space is available in the endpoint, this function posts a new frame,
 * copying the contents from the provided data buffer.
 * If @a size is less than the frame size, then the remaining bytes of the frame
 * are filled with zero.
 *
 * This is a non-blocking call.
 * If write channel of the endpoint was previously empty, then the function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset.
 *
 * The user shall make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle The handle to the endpoint to write to.
 * @param[in]  buf    A pointer to a source buffer for the contents of
 *                    the next frame.
 * @param[in]  size   The number of bytes to be copied to the frame,
 *                    not to exceed the length of the destination buffer and
 *                    configured frame size of the endpoint.
 * @param[out] bytes  The number of bytes written on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates write channel is full and
 *                                   the write operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcWriteSafe(NvSciIpcEndpoint handle, const void *buf,
    uint32_t size, uint32_t *bytes);

/**
 * @brief Get a pointer to the write frame from the endpoint.
 *
 * This is a non-blocking call. write channel of the endpoint must not be full.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpointSafe(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * @param[in]  handle The handle to the endpoint to write to.
 * @param[out] buf    A pointer to a destination buffer to hold the contents of
 *                    the next frame on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates write channel is full and
 *                                   the write operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @pre No overlapped write operation
 * - Between NvSciIpcWriteGetNextFrame() and NvSciIpcWriteAdvance(), do not
 * perform any other NvSciIpc write operations with the same endpoint handle.
 *
 * @pre No use of invalid pointer to write
 * - Once a transmit message is committed by NvSciIpcWriteAdvance(), do not use
 * previously returned pointer of NvSciIpcWriteGetNextFrame() since it is
 * already invalid.
 *
 * @pre No read operation with pointer to write frame
 * - Do not read through a returned pointer of NvSciIpcWriteGetNextFrame().
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcWriteGetNextFrame(NvSciIpcEndpoint handle,
    volatile void **buf);

/**
 * @brief Advance to the next write frame of the endpoint.
 *
 * This is a non-blocking call.
 * If write channel of the endpoint is not full, then post the next frame.
 * If write channel of the endpoint was previously empty, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpointSafe(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * @note This API is used together with NvSciIpcWriteGetNextFrame().
 * Refer to precondition of NvSciIpcWriteGetNextFrame()
 *
 * @param[in] handle The handle to the endpoint to write to.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates write channel is full and
 *                                   the write operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @pre Populated with data
 * - The user shall ensure entire outgoing frame is populated with application
 * data or filled with zeros prior call to NvScipcWriteAdvance().
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcWriteAdvance(NvSciIpcEndpoint handle);

/**
 * @brief Returns endpoint information.
 *
 *
 * @param[in]  handle NvSciIpc endpoint handle.
 * @param[out] info   A pointer to NvSciIpcEndpointInfo object that
 *                    this function copies the info to on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() or
 *      NvSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcGetEndpointInfo(NvSciIpcEndpoint handle,
    NvSciIpcEndpointInfo *info);

#ifndef __QNX__
/**
 * Returns the NvSciIpc file descriptor for a given endpoint.
 *
 * <b> This API is specific to Linux OS. </b>
 * Event handle will be used to plug OS event notification
 * (can be read, can be written, established, reset etc.)
 *
 * @param handle NvSciIpc endpoint handle
 * @param fd     A pointer to the endpoint file descriptor.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type or OS environment.
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() or
 *      NvSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcGetLinuxEventFd(NvSciIpcEndpoint handle, int32_t *fd);
#endif /* !__QNX__ */

/**
 * @brief Get Events
 *
 * @warning This API is deprecated and returns NvSciError_NotSupported without
 * doing anything in safety build.
 * Use NvSciIpcGetEventSafe() instead of this.
 */
NvSciError NvSciIpcGetEvent(NvSciIpcEndpoint handle, uint32_t *events);

/**
 * @brief Get Events
 * (safety version)
 *
 * Returns a bitwise OR operation on new events that occurred since the
 * last call to this function.
 *
 * This function sets @a events to the result of a bitwise OR operation of zero
 * or more @c NV_SCI_IPC_EVENT_* constants corresponding to all new events that
 * have occurred on the endpoint since:
 * - the preceding call to this function on the endpoint or
 * - opening the endpoint, if this is the first call to this function on the
 *   endpoint since it was opened.
 *
 * The parameter @a events is set to zero if no new events have
 * occurred.
 *
 * There are 6 types of events:
 * - @c NV_SCI_IPC_EVENT_CONN_EST   : IPC connection established
 * - @c NV_SCI_IPC_EVENT_WRITE      : IPC write
 * - @c NV_SCI_IPC_EVENT_READ       : IPC read
 * - @c NV_SCI_IPC_EVENT_CONN_RESET : IPC connection reset
 * - @c NV_SCI_IPC_EVENT_WRITE_EMPTY : IPC write FIFO empty
 * - @c NV_SCI_IPC_EVENT_ASYNC_ERROR : Asynchronous error
 *
 * @c NV_SCI_IPC_EVENT_CONN_EST and @c NV_SCI_IPC_EVENT_CONN_RESET events are
 * connection event. They're edge-triggered events and once they're read by
 * user, events are cleared.
 *
 * @c NV_SCI_IPC_EVENT_WRITE, @c NV_SCI_IPC_EVENT_READ and
 * @c NV_SCI_IPC_EVENT_WRITE_EMPTY events are FIFO status event.
 * As long as free buffer is available on write FIFO or data are available in
 * read FIFO, API keeps reporting same events. All these events also mean that
 * connection is established.
 *
 * An @c NV_SCI_IPC_EVENT_CONN_EST event occurs on an endpoint each time a
 * connection is established through the endpoint (between the endpoint and
 * the other end of the corresponding channel).
 *
 * An @c NV_SCI_IPC_EVENT_WRITE event occurs on an endpoint:
 * -# In conjunction with the delivery of each @c NV_SCI_IPC_CONN_EST event.
 * -# Each time the endpoint's sending FIFO ceases to be full.
 *
 * An @c NV_SCI_IPC_EVENT_READ event occurs on an endpoint:
 * -# In conjunction with the delivery of each @c NV_SCI_IPC_EVENT_CONN_EST
 * event, if frames can already be read as of delivery.
 * -# Each time the endpoint's receiving FIFO ceases to be empty.
 *
 * An @c NV_SCI_IPC_EVENT_CONN_RESET event occurs on an endpoint when the user
 * calls NvSciIpcResetEndpoint.
 *
 * An @c NV_SCI_IPC_EVENT_WRITE_EMPTY event occurs on an endpoint when write
 * FIFO is empty. user can utilize this event to check if remote endpoint reads
 * all data which local endpoint sent.
 *
 * An @c NV_SCI_IPC_EVENT_ASYNC_ERROR event occurs on an endpoint when there is
 * a asynchronouse error, e.g. failure on eDMA or PCIE. To get
 * detailed error information, the user can call @c NvSciIpcGetAsyncError().
 *
 * If this function doesn't return desired events, user must call
 * OS-provided blocking API to wait for notification from remote endpoint.
 *
 * The following are blocking API examples:
 * - QNX  : NvSciIpcWaitEventQnx()
 * - LINUX: select(), poll() etc.
 * - NvSciEventService: NvSciEventLoopService::WaitForEvent(), <br/>
 *                      NvSciEventLoopService::WaitForMultipleEvents()
 *
 * In case of QNX OS, in order to authenticate user client process, NvSciIpc
 * uses custom ability "NvSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @note Before using any read and write NvSciIpc API, user should call this
 *       API to make sure if connection is established.
 *
 * @param[in]  handle NvSciIpc endpoint handle.
 * @param[out] events  A pointer to the variable into which to store
 *                    the bitwise OR result of new events on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter       Indicates an invalid or NULL argument.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 *
 * @pre NvSciIpcResetEndpointSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): "NvSciIpcEndpoint"
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcGetEventSafe(NvSciIpcEndpoint handle, uint32_t *events);

#if defined(__QNX__)
/**
 * @brief Wait for an event with timeout
 *
 * <b>This API is specific to QNX OS.</b>
 * Users of NvSciIpc must call this API to wait for an event when
 * NvSciIpcGetEventSafe() does not return desired events.
 * This API is a blocking call.
 *
 * In using NvSciEventService, call other blocking APIs instead of this API.
 * - NvSciEventLoopService::WaitForEvent()
 * - NvSciEventLoopService::WaitForMultipleEvents()
 * - NvSciEventLoopService::WaitForMultipleEventsExt()
 *
 * @note It is only compatible with an endpoint that is opened with
 *       NvSciIpcOpenEndpoint().
 *
 * @param[in] chid     The ID of a channel that you established by calling
 *                     ChannelCreate_r() or ChannelCreatePulsePool().
 * @param[in] microseconds  A 64-bit integer timeout in microsecond.
 *                          Set to NVSCIIPC_INFINITE_WAIT for an infinite
 *                          timeout.
 * @param[in] bytes    The size of @pulse buffer
 * @param[out] pulse   A void * pointer to struct _pulse structure where the
 *                     fuction can store the received data.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::EOK            Indicates a successful operation.
 * - ::EFAULT         Indicates A fault occurred when the kernel tried to access
 *                    the buffer provided, or the size of the receive buffer is
 *                    less than the size of @a pulse. The pulse is lost in this
 *                    case.
 * - ::EPERM          Indicates NvSciIpc is uninitialized.
 * - ::EINTR          Indicates API was interrupted by a signal.
 * - ::ESRCH          Indicates The channel indicated by chid doesn't exist.
 * - ::ETIMEDOUT      Indicates A kernel timeout unblocked the call.
 *                    See TimerTimeout().
 * - ::EINVAL         Indicates a invalid parameter
 *
 * @pre ChannelCreate_r() or ChannelCreatePulsePool() must be successful.
 *      NvSciIpcSetQnxPulseParamSafe() must be successful.
 *
 * @pre Suitable API to wait for multiple events
 * - If user process needs to wait for events from multiple remote NvSciIpc
 * endpoint processes, the user has to use single blocking call from
 * single thread instead of using blocking call per endpoint thread.
 * NvSciEventLoopService::WaitForMultipleEvents() blocking call is suitable for
 * this use case.
 *
 * @pre Suspend a thread when accessing with same chid
 * - If the user of NvSciIpc uses same chid between this API and
 * NvSciIpcInspectEventQnx(),the user has to suspend a thread which calls this
 * API before calling NvSciIpcInspectEventQnx() in other monitor thread.
 * You can suspend thread using delay() or other method.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
int32_t NvSciIpcWaitEventQnx(int32_t chid, int64_t microseconds, uint32_t bytes,
    void *pulse);

/**
 * @brief Sets the event pulse parameters for QNX.
 *
 * <b>This API is specific to QNX OS.</b>
 * @warning This API is deprecated and just returns NvSciError_NotSupported without
 * doing anything in safety build.
 * Use NvSciIpcSetQnxPulseParamSafe() instead of this.
 */
NvSciError NvSciIpcSetQnxPulseParam(NvSciIpcEndpoint handle,
    int32_t coid, int16_t pulsePriority, int16_t pulseCode,
    void *pulseValue);

/**
 * @brief Sets the event pulse parameters for QNX.
 * (safety version)
 *
 * <b>This API is specific to QNX OS.</b>
 * When a notification from a peer endpoint is available, the NvSciIpc library
 * sends a pulse message to the application.
 * This API is to connect @a coid to the endpoint, plug OS event notification
 * and set pulse parameters (@a pulsePriority and @a pulseCode),
 * thereby enabling the application to receive peer notifications from the
 * NvSciIpc library.
 * An application can receive notifications from a peer endpoint using
 * @c NvSciIpcWaitEventQnx() which is blocking call.
 *
 * Prior to calling this function, both @c ChannelCreate_r() /
 * @c ChannelCreatePulsePool() and @c ConnectAttach_r() must be called in
 * the application to obtain the value for @a coid to pass to this function.
 *
 * To use the priority of the calling thread, set @a pulsePriority to
 * @c SIGEV_PULSE_PRIO_INHERIT(-1). The priority must fall within the valid
 * range, which can be determined by calling @c sched_get_priority_min() and
 * @c sched_get_priority_max().
 *
 * Applications can define any value per endpoint for @a pulseCode.
 * @a pulseCode will be used by NvSciIpc to signal IPC events and should be
 * reserved for this purpose by the application.
 *
 * @note It is only compatible with an endpoint that is opened with
 *       NvSciIpcOpenEndpoint().
 *
 * @param[in] handle        NvSciIpc endpoint handle.
 * @param[in] coid          The connection ID created from calling
 *                          @c ConnectAttach_r().
 * @param[in] pulsePriority The value for pulse priority.
 * @param[in] pulseCode     The 8-bit positive pulse code specified by the user.
 *                          The values must be between @c _PULSE_CODE_MINAVAIL
 *                          and @c _PULSE_CODE_MAXAVAIL.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_NotInitialized  Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter    Indicates an invalid @a handle.
 * - ::NvSciError_NotSupported    Indicates API is not supported in provided
 *                                endpoint backend type or OS environment.
 * - ::NvSciError_ResourceError   Indicates not enough system resources.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 * - ::NvSciError_NotPermitted    Indicates incorrect API group usage.
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() must be successful.
 *      Invocation of ConnectAttach_r() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX):
 *   - PROCMGR_AID_INTERRUPTEVENT (Inter-VM only)
 *   - PROCMGR_AID_PUBLIC_CHANNEL
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcSetQnxPulseParamSafe(NvSciIpcEndpoint handle,
    int32_t coid, int16_t pulsePriority, int16_t pulseCode);


/**
 * @brief Inspect Event for QNX.
 *
 * <b>This API is specific to QNX OS.</b>
 * When legacy event handling is used, if IVC signalling storm happens in
 * specific Intra-VM endpoint which is not bound to NvSciEventService object,
 * this API returns that endpoint handle and unregisters relavant events.
 * This API is used in a thread to receive semaphore event of
 * ChannelCreatePulsePool() when pulse can't be obtained from the fixed pool
 * any more.
 * In order to prevent unwanted IVC signaling path breakage, Do not call this
 * API when semaphore event is not triggered by ChannelCreatePulsePool().
 *
 * @note This API is used together with NvSciIpcWaitEventQnx().
 * Refer to precondition of NvSciIpcWaitEventQnx()
 *
 * @note It is only compatible with endpoints that are opened with
 *       NvSciIpcOpenEndpoint().
 *
 * @param[in] chid           A chid which is created by ChannelCreatePulsePool()
 * @param[in] numEvents      A threshold value to unregister Intra-VM IVC
 *                           signalling events. This shall be less than
 *                           num_pulses which is configured in
 *                           ChannelCreatePulsePool().
 * @param[in] epCount        endpoint handle count in @epHandleArray.
 * @param[out] epHandleArray Array of NvSciIpc endpoint handle which has
 *                           excessive IVC signalling.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_NotInitialized  Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_NoSuchEntry     Indicates unknown endpoint.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 *
 * @pre Invocation of ChannelCreatePulsePool() must be successful.
 *      Invocation of ConnectAttach_r() must be successful.
 *      Invocation of NvSciIpcSetQnxPulseParamSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvSciError NvSciIpcInspectEventQnx(int32_t chid, uint16_t numEvents,
    uint32_t epCount, NvSciIpcEndpoint **epHandleArray);
#endif /* __QNX__ */

/**
 * @brief Get asynchronouse errors.
 *
 * Returns error codes that can happen asynchronousely in the backend
 * component. It could be errors from eDMA or PCIE in case of C2C backend.
 *
 * There are four errors reported in @a mask and this API returns bitwise OR
 * operaton on new errors that has occurred since the last call to it.
 * - @c NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_FATAL
 * - @c NV_SCI_ASYNC_PCIE_AER_UNCORRECTABLE_NONFATAL
 * - @c NV_SCI_ASYNC_PCIE_EDMA_XFER_ERROR
 * - @c NV_SCI_ASYNC_PCIE_VALIDATION_ERROR
 *
 * @param[in]  handle NvSciIpc endpoint handle.
 * @param[out] mask   place holder to store bitwise OR errors
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_BadParameter    Indicates an invalid @a handle or @a error.
 * - ::NvSciError_NotInitialized  Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 * - ::NvSciError_NotSupported    Indicates API is not supported in provided
 *                                endpoint backend type or OS environment.
 *
 * @pre Invocation of NvSciIpcGetEventSafe() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvSciError NvSciIpcGetAsyncErrors(NvSciIpcEndpoint handle,  uint32_t* errors);

/**
 * @brief Enable event notification (or polling)
 *
 * Control notification when connection or FIFO status is changed.
 * If notification is disabled, NvSciIpc doesn't send notification to remote
 * endpoint.
 * notification is enabled by default unless user disables it using
 * this API explicitly.
 * User can use this API to choose asynchronous event notification mode or
 * polling mode.
 * Once notification is disabled in both endpoints, user shall not call
 * event blocking API (e.g. poll, NvSciIpcWaitEventQnx, WaitForEvent,
 * WaitForMultipleEvents etc.).
 *
 * @note This API supports Intra-VM and Inter-VM backend only.
 *
 * @param[in] handle        NvSciIpc endpoint handle.
 * @param[in] flag          flag to enable IPC notification.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_NotInitialized  Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter    Indicates an invalid @a handle.
 * - ::NvSciError_NotSupported    Indicates API is not supported in provided
 *                                endpoint backend type or OS environment.
 *
 * @pre Invocation of NvSciIpcResetEndpointSafe() must be successful.
 *
 * @pre Pending event in disabling notification
 * - Before calling this API to disable notification,
 * The user shall ensure that there are no remaining outstanding data
 * transactions between the two endpoints using their own handshaking mechanism
 * at the client level. If this is not guaranteed, pending notification might be
 * delivered from local to remote peer.
 * In that case, user shall handle (receive or ignore) such spurious
 * incoming notifications.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciIpcEnableNotification(NvSciIpcEndpoint handle, bool flag);

/**
 * @brief Check NvSciIpc library version compatibility
 *
 * Checks if loaded NvSciIpc library version is compatible with
 * the version the application was compiled against.
 * This function checks loaded NvSciIpc library version with input NvSciIpc
 * library version and sets output variable true provided major version of the
 * loaded library is same as @a majorVer and minor version of the
 * loaded library is not less than @a minorVer, else sets output to false
 *
 * @param[in]  majorVer build major version.
 * @param[in]  minorVer build minor version.
 * @param[out] isCompatible boolean value stating if loaded NvSciIpc library
 *             is compatible or not.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success if successful.
 * - ::NvSciError_BadParameter if any of the following occurs:
 *      - @a isCompatible is NULL
 *
 * @pre None
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvSciError NvSciIpcCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

/** @} <!-- End nvsci_ipc_api --> */
/** @} <!-- End nvsci_group_ipc --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_NVSCIIPC_H */

