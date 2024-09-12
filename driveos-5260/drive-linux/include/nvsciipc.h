/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <nvscierror.h>
#include <nvscievent.h>

/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSci Inter-Process Communication </b>
 *
 */
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
 * When Init operation group APIs are used, the user should call them in the
 * following order, with or without NvSciEventService.
 *
 * <b> Typical call flow with NvSciIpc library </b>
 *
 * 1) Init mode
 *    - NvSciIpcInit()
 *    - NvSciIpcOpenEndpoint()
 *    - Set event reporting path
 *      NvSciIpcSetQnxPulseParam() (QNX OS-specific) or
 *      NvSciIpcGetLinuxEventFd() (Linux OS-specific)
 *    - NvSciIpcGetEndpointInfo()
 *    - NvSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          NvSciIpcGetEvent()
 *          if (event & NV_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              MsgReceivePulse_r() (QNX OS-specific) or
 *              select(), epoll() (Linux OS-specific)
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 2) Runtime mode (loop)
 *    - NvSciIpcGetEvent()
 *    - If an event is not desired,
 *      call OS-blocking API
 *      MsgReceivePulse_r() (QNX OS-specific) or
 *      select(), epoll() (Linux OS-specific)
 *    - NvSciIpcRead() or NvSciIpcWrite()
 *
 * 3) De-Init mode
 *    - NvSciIpcCloseEndpoint()
 *    - NvSciIpcDeinit()
 *
 * <b> Typical call flow with NvSciIpc and NvSciEventService library </b>
 *
 * NvSciEventService provides APIs that replace OS-specific event-blocking API.
 * They are only compatible with an endpoint which is opened with
 * NvSciOpenEndpointWithEventService().
 *
 * 1) Init mode
 *    - NvSciEventLoopServiceCreate() to get eventLoopService
 *    - NvSciIpcInit()
 *    - NvSciIpcOpenEndpointWithEventService()
 *    - NvSciIpcGetEventNotifier() to get eventNotifier
 *    - NvSciIpcGetEndpointInfo()
 *    - NvSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          NvSciIpcGetEvent()
 *          if (event & NV_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              eventLoopService->WaitForEvent(eventNotifier)
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 2) Runtime mode (loop)
 *    - NvSciIpcGetEvent()
 *    - If an event is not desired,
 *    - eventLoopService->WaitForEvent()
 *    - NvSciIpcRead() or NvSciIpcWrite()
 *
 * 3) De-Init mode
 *    - eventNotifier->Delete()
 *    - NvSciIpcCloseEndpoint()
 *    - NvSciIpcDeinit()
 *    - eventLoopService->EventService.Delete()
 *
 * <b>Using multi-threading in NvSciIpc - QNX OS</b>
 *
 * For Inter-VM and Inter-Process backend applications on QNX OS,
 * only a single event-blocking OS API (i.e. MsgReceivePulse_r(),
 * NvSciEventLoopService::WaitForEvent()) is allowed per endpoint
 * in the same process boundary.
 * If a client application tries to use receiving and sending thread separately for
 * the same endpoint handle, the event-blocking OS APIs must be used in a
 * single thread in order to receive remote notification.
 * Once a remote notification arrives in the thread, the notification should be forwarded
 * to the other thread using the same endpoint handle through any OS synchronization
 * method (e.g. sem_post, pthread_cond_signal or MsgSendPulse etc.)
 * Single thread usage is recommended to handle both TX and RX data.
 *
 * <b>Configuring thread pool of resource manager - QNX OS</b>
 *
 * NvSciIpc resource manager (io-nvsciipc) and IVC resource manager (devv-nvivc)
 * use thread pool to manage concurrent request from multiple NvSciIpc client
 * processes using NvSciIpc library.
 * io-nvsciipc is used during opening endpoint and devv-nvivc is used for
 * Inter-VM IVC signaling.
 * Drive OS users should evaluate thread pool capacity of io-nvsciipc and
 * devv-nvivc then configure them with -t option in startup script.
 * Thread pool capacity for NvSciIPC can be evaluated based on number of
 * parallel outstanding NvSciIPC requests, at any point of time, that are
 * expected in the system. Default value of thread pool capacity is 10.
 *
 * <b>Getting events before using Read/Write NvSciIpc API</b>
 *
 * Before using any Read/Write APIs, the user must check if @ref NV_SCI_IPC_EVENT_READ
 * or @ref NV_SCI_IPC_EVENT_WRITE event is available through NvSciIpcGetEvent().
 * NvSciIpcGetEvent() has additional support to establish connection between
 * two endpoint software entities.
 *
 * <b>When to use blocking API</b>
 *
 * Users of NvSciIpc must call OS event-blocking API to wait for an event when
 * NvSciIpcGetEvent() does not return desired events.
 * The following are OS event-blocking API examples:
 * - QNX  : MsgReceivePulse_r()
 * - LINUX: select(), epoll() etc.
 * - NvSciEventService: NvSciEventLoopService::WaitForEvent(),<br/>
 *                      NvSciEventLoopService::WaitForMultipleEvents()
 *
 * If user process needs to wait for events from multiple remote NvSciIpc
 * endpoint processes, use single blocking call from single thread instead of
 * using blocking call per endpoint thread. This is recommended to improve
 * performance by avoiding thread creation per endpoint.
 * NvSciEventLoopService::WaitForMultipleEvents() blocking call is suitable for
 * this use case.
 *
 * <b>How to check if peer endpoint entity receives a message</b>
 *
 * NvSciIpc library does not provide information about whether a peer endpoint
 * entity receives all sent messages from a local endpoint entity.
 * If such a mechanism is required, the client user should implement separate
 * message acknowledgment in the application layer.
 *
 * <b>Recommended Read/Write APIs</b>
 *
 * Using NvSciIpcRead() and NvSciIpcWrite() is recommended rather than following
 * Read/Write APIs. See detail constraints of API in each function description.
 * - NvSciIpcReadGetNextFrame()
 * - NvSciIpcWriteGetNextFrame()
 * - NvSciIpcReadAdvance()
 * - NvSciIpcWriteAdvance()
 * However, above functions are better to avoid extra memory copy.
 *
 * <b>Provide valid buffer pointers</b>
 *
 * The user of NvSciIpc must provide valid buffer pointers to NvSciIpcRead(),
 * NvSciIpcWrite() and other Read/Write NvSciIpc APIs as NvSciIpc library
 * validation to these parameters is limited to a NULL pointer check.
 *
 * <b>Maximum number of endpoints</b>
 *
 * One NvSciIpc client process is allowed to open up to 50 endpoints.
 * QNX OS safety manual imposes a restriction no process SHALL have more than
 * 100 open channels without disabling kernel preemption. User client needs
 * one channel/connection pair to receive an endpoint notification.
 *
 */

/*******************************************************************/
/************************ DATA TYPES *******************************/
/*******************************************************************/
/* implements_unitdesign QNXBSP_NVSCIIPC_LIBNVSCIIPC_40 */

/**
 * @brief Handle to the NvSciIpc endpoint.
 */
typedef uint64_t NvSciIpcEndpoint;

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
#define	NV_SCI_IPC_EVENT_READ       1U
/** Specifies the IPC write event. */
#define	NV_SCI_IPC_EVENT_WRITE      2U
/** Specifies the IPC connection established event. */
#define	NV_SCI_IPC_EVENT_CONN_EST   4U
/** Specifies the IPC connection reset event. */
#define	NV_SCI_IPC_EVENT_CONN_RESET 8U
/** Specifies single event mask to check IPC connection establishment */
#define	NV_SCI_IPC_EVENT_CONN_EST_ALL (NV_SCI_IPC_EVENT_CONN_EST | NV_SCI_IPC_EVENT_WRITE | NV_SCI_IPC_EVENT_READ)

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
 * - ::NvSciError_InvalidState Indicates an invalid operation state.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
NvSciError NvSciIpcInit(void);

/**
 * @brief De-initializes the NvSciIpc library.
 *
 * This function cleans up the NvSciIpc endpoint internal database
 * created by NvSciIpcInit().
 * Before calling this API, all existing opened endpoints must be closed
 * by NvSciIpcCloseEndpoint().
 *
 * @return @c void
 *
 * @pre Invocation of NvSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
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
 * - ::NvSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::NvSciError_InsufficientMemory Indicates memory allocation failed for the operation.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *
 * @pre Invocation of NvSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "NvSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
NvSciError NvSciIpcOpenEndpoint(const char *endpoint, NvSciIpcEndpoint *handle);

/**
 * @brief Opens an endpoint with the given name and event service.
 *
 * This API provides same functionality as NvSciIpcOpenEndpoint().
 * But, it requires additional event service abstract object as an input
 * parameter to utilize NvSciEventService infrastructure.
 * NvSciEventService can be created through NvSciEventLoopServiceCreate().
 * NvSciIpcGetEventNotifier() can be used only when this API is invoked
 * successfully.
 * In case of QNX OS, in order to authenticate user client process, NvSciIpc
 * uses custom ability "NvSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
 *
 * @param[in]  endpoint      The name of the NvSciIpc endpoint to open.
 * @param[out] handle        A handle to the endpoint on success.
 * @param[in]  eventService  An abstract object to use NvSciEventService infrastructure.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates any argument was NULL or invalid.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_NoSuchEntry        Indicates the @a endpoint was not found.
 * - ::NvSciError_Busy               Indicates the @a endpoint is already in use.
 * - ::NvSciError_InsufficientMemory Indicates memory allocation failed for the operation.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 * - ::NvSciError_NotPermitted       Indicates process doesn't have the required
 *                                   privilege.
 *
 * @pre Invocation of NvSciEventLoopServiceCreate() must be successful.
 *      Invocation of NvSciIpcInit() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "NvSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
NvSciError NvSciIpcOpenEndpointWithEventService(const char *endpoint,
    NvSciIpcEndpoint *handle, NvSciEventService *eventService);

/**
 * @brief Get NvSciIpc event notifier.
 *
 * This API is used to connect NvSciIpc event handling with OS-provided
 * event interface.
 * It also utilizes NvSciEventService infrastructure.
 * Before calling NvSciIpcCloseEndpoint(), event notifier should be deleted
 * through Delete callback of NvSciEventNotifier.
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
 *
 * @pre Invocation of NvSciIpcOpenEndpointWithEventService() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
NvSciError NvSciIpcGetEventNotifier(NvSciIpcEndpoint handle,
               NvSciEventNotifier **eventNotifier);

/**
 * @brief Closes an endpoint with the given handle.
 *
 * The function frees the NvSciIpc endpoint associated with the given @a handle.
 *
 * @param[in] handle A handle to the endpoint to close.
 *
 * @return @c void
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: No
 *   - De-initialization: Yes
 */
void NvSciIpcCloseEndpoint(NvSciIpcEndpoint handle);

/**
 * @brief Resets an endpoint.
 *
 * Initiates a reset on the endpoint and notifies the remote endpoint.
 * Applications must call this function and complete the reset operation before
 * using the endpoint for communication.
 * Once this API is called, all existing data in channel will be discarded.
 * After invoking this function, client user shall call NvSciIpcGetEvent()
 * to get specific event type (READ, WRITE etc.). if desired event is not
 * returned from GetEvent API, OS-specific blocking call (select/poll/epoll
 * or MsgReceivePulse) should be called to wait remote notification.
 * This sequence must be done repeatedly to get event type that
 * endpoint wants.
 *
 * @param[in] handle A handle to the endpoint to reset.
 *
 * @return @c void
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
void NvSciIpcResetEndpoint(NvSciIpcEndpoint handle);

/**
 * @brief Returns the contents of the next frame from an endpoint.
 *
 * This function removes the next frame and copies its contents
 * into a buffer. If the destination buffer is smaller than the configured
 * frame size of the endpoint, the trailing bytes are discarded.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If read channel of the endpoint was previously full, then the function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * The user shall make sure if actual input buffer size is equal or bigger than
 * requested size before using this API.
 *
 * @param[in]  handle The handle to the endpoint to read from.
 * @param[out] buf    A pointer to a destination buffer to receive the contents
 *                    of the next frame.
 * @param[in]  size   The number of bytes to copy from the frame. If @a size
 *                    is greater than the size of the destination buffer, the
 *                    remaining bytes are discarded.
 * @param[out] bytes  The number of bytes read on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates read channel is empty and the read
 *                                   operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported on provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcRead(NvSciIpcEndpoint handle, void *buf, size_t size,
	int32_t *bytes);

/**
 * @brief Returns a pointer to the location of the next frame from an endpoint.
 *
 * This is a non-blocking call.
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 * Between NvSciIpcReadGetNextFrame() and NvSciIpcReadAdvance(), do not perform
 * any other NvSciIpc read operations with the same endpoint handle.
 * Once a read frame is released by NvSciIpcReadAdvance(), do not use previously
 * returned pointer of NvSciIpcReadGetNextFrame() since it is already invalid.
 * Do not write through a returned pointer of NvSciIpcReadGetNextFrame().
 * This is protected by a const volatile pointer return type.
 * Do not read the same memory location multiple times. If required, copy
 * specific memory location to a local buffer before using it.
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
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcReadGetNextFrame(NvSciIpcEndpoint handle,
    const volatile void **buf);

/**
 * @brief Removes the next frame from an endpoint.
 *
 * This is a non-blocking call. Read channel of the endpoint must not be empty.
 * If a read channel of the endpoint was previously full, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * Between NvSciIpcReadGetNextFrame() and NvSciIpcReadAdvance(), do not perform
 * any other NvSciIpc read operations with the same endpoint handle.
 * Once a read frame is released by NvSciIpcReadAdvance(), do not use previously
 * returned pointer of NvSciIpcReadGetNextFrame() since it is already invalid.
 *
 * @param[in] handle The handle to the endpoint to read from.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates the frame was removed successfully.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates read channel is empty and the read
 *                                   operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcReadAdvance(NvSciIpcEndpoint handle);

/**
 * @brief Writes a new frame to the endpoint.
 *
 * If space is available in the endpoint, this function posts a new frame,
 * copying the contents from the provided data buffer.
 * If @a size is less than the frame size, then the remaining bytes of the frame
 * are undefined.
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
 *                    not to exceed the length of the destination buffer.
 * @param[out] bytes  The number of bytes written on success.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates a successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle or @a size.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates write channel is full and the write
 *                                   operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcWrite(NvSciIpcEndpoint handle, const void *buf, size_t size,
	int32_t *bytes);

/**
 * @brief Returns a pointer to the location of the next frame for writing data.
 *
 * This is a non-blocking call. write channel of the endpoint must not be full.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 * Between NvSciIpcWriteGetNextFrame() and NvSciIpcWriteAdvance(), do not
 * perform any other NvSciIpc write operations with the same endpoint handle.
 * Once a transmit message is committed by NvSciIpcWriteAdvance(), do not use
 * previously returned pointer of NvSciIpcWriteGetNextFrame() since it is already
 * invalid.
 * Do not read through a returned pointer of NvSciIpcWriteGetNextFrame().
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
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcWriteGetNextFrame(NvSciIpcEndpoint handle,
    volatile void **buf);

/**
 * @brief Writes the next frame to the endpoint.
 *
 * This is a non-blocking call.
 * If write channel of the endpoint is not full, then post the next frame.
 * If write channel of the endpoint was previously empty, then this function
 * notifies the remote endpoint.
 *
 * This operation cannot proceed if the endpoint is being reset. However,
 * if the remote endpoint has called NvSciIpcResetEndpoint(), calls to this
 * function can still succeed until the next event notification on the local
 * endpoint.
 *
 * Between NvSciIpcWriteGetNextFrame() and NvSciIpcWriteAdvance(), do not
 * perform any other NvSciIpc write operations with the same endpoint handle.
 * Once transmit message is committed by NvSciIpcWriteAdvance(), do not use
 * previously returned pointer of NvSciIpcWriteGetNextFrame() since it is already
 * invalid.
 *
 * @param[in] handle The handle to the endpoint to write to.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success            Indicates successful operation.
 * - ::NvSciError_BadParameter       Indicates an invalid @a handle.
 * - ::NvSciError_NotInitialized     Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_InsufficientMemory Indicates write channel is full and the write
 *                                   operation aborted.
 * - ::NvSciError_ConnectionReset    Indicates the endpoint is being reset.
 * - ::NvSciError_NotSupported       Indicates API is not supported in provided
 *                                   endpoint backend type.
 * - ::NvSciError_InvalidState       Indicates an invalid operation state.
 *
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: No
 *   - Runtime: Yes
 *   - De-initialization: No
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
 * @pre Invocation of NvSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): None
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcGetEndpointInfo(NvSciIpcEndpoint handle,
                struct NvSciIpcEndpointInfo *info);

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
 */
NvSciError NvSciIpcGetLinuxEventFd(NvSciIpcEndpoint handle, int32_t *fd);
#endif /* !__QNX__ */

/**
 * @brief Get Events
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
 * There are four types of events:
 * - @c NV_SCI_IPC_EVENT_CONN_EST   : IPC connection established
 * - @c NV_SCI_IPC_EVENT_WRITE      : IPC write
 * - @c NV_SCI_IPC_EVENT_READ       : IPC read
 * - @c NV_SCI_IPC_EVENT_CONN_RESET : IPC connection reset
 *
 * These may occur in arbitrary combinations, except for the following:
 * - @c NV_SCI_IPC_EVENT_CONN_EST is always combined with @c NV_SCI_IPC_EVENT_WRITE.
 * - @c NV_SCI_IPC_EVENT_CONN_RESET cannot be combined with any other events.

 * There are seven possible event combinations:
 * - 0
 * - @c NV_SCI_IPC_EVENT_CONN_EST and @c NV_SCI_IPC_EVENT_WRITE
 * - @c NV_SCI_IPC_EVENT_CONN_EST and @c NV_SCI_IPC_EVENT_WRITE and
 *   @c NV_SCI_IPC_EVENT_READ
 * - @c NV_SCI_IPC_EVENT_READ
 * - @c NV_SCI_IPC_EVENT_WRITE
 * - @c NV_SCI_IPC_EVENT_WRITE and @c NV_SCI_IPC_EVENT_READ
 * - @c NV_SCI_IPC_EVENT_CONN_RESET
 *
 * An @c NV_SCI_IPC_EVENT_CONN_EST event occurs on an endpoint each time a
 * connection is established through the endpoint (between the endpoint and
 * the other end of the corresponding channel).
 *
 * An @c NV_SCI_IPC_EVENT_WRITE event occurs on an endpoint:
 * -# In conjunction with the delivery of each @c NV_SCI_IPC_CONN_EST event.
 * -# Each time the endpoint ceases to be full after a prior @c NvSciIpcWrite*
 * call returned @c NvSciError_InsufficientMemory. Note however that an
 * implementation is permitted to delay the delivery of this type of
 * @c NV_SCI_IPC_EVENT_WRITE event, e.g., for purposes of improving throughput.
 *
 * An @c NV_SCI_IPC_EVENT_READ event occurs on an endpoint:
 * -# In conjunction with the delivery of each @c NV_SCI_IPC_EVENT_CONN_EST event,
 * if frames can already be read as of delivery.
 * -# Each time the endpoint ceases to be empty after a prior @c NvSciRead*
 * call returned @c NvSciError_InsufficientMemory. Note however that an
 * implementation is permitted to delay the delivery of this type of
 * @c NV_SCI_IPC_EVENT_READ event, e.g., for purposes of improving throughput.
 *
 * An @c NV_SCI_IPC_EVENT_CONN_RESET event occurs on an endpoint when the user
 * calls NvSciIpcResetEndpoint.
 *
 * If this function doesn't return desired events, user must call
 * OS-provided blocking API to wait for notification from remote endpoint.
 *
 * The following are blocking API examples:
 * - QNX  : MsgReceivePulse_r()
 * - LINUX: select(), epoll() etc.
 * - NvSciEventService: NvSciEventLoopService::WaitForEvent(), <br/>
 *                      NvSciEventLoopService::WaitForMultipleEvents()
 *
 * In case of QNX OS, in order to authenticate user client process, NvSciIpc
 * uses custom ability "NvSciIpcEndpoint". Use procmgr_ability_lookup()
 * QNX OS API to get ability ID.
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
 * @pre NvSciIpcResetEndpoint() must be called.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_MEM_PHYS, "NvSciIpcEndpoint"
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: Yes
 *   - De-initialization: No
 */
NvSciError NvSciIpcGetEvent(NvSciIpcEndpoint handle, uint32_t *events);

#ifdef __QNX__
/**
 * @brief Sets the event pulse parameters for QNX.
 *
 * <b>This API is specific to QNX OS.</b>
 * When a notification from a peer endpoint is available, the NvSciIpc library
 * sends a pulse message to the application.
 * This API is to connect @a coid to the endpoint, plug OS event notification
 * and set pulse parameters (@a pulsePriority, @a pulseCode and @a pulseValue),
 * thereby enabling the application to receive peer notifications from the
 * NvSciIpc library.
 * An application can receive notifications from a peer endpoint using
 * @c MsgReceivePulse_r() which is blocking call.
 *
 * Prior to calling this function, both @c ChannelCreate_r() and @c ConnectAttach_r()
 * must be called in the application to obtain the value for @a coid to pass to
 * this function.
 *
 * To use the priority of the calling thread, set @a pulsePriority to
 * @c SIGEV_PULSE_PRIO_INHERIT(-1). The priority must fall within the valid
 * range, which can be determined by calling @c sched_get_priority_min() and
 * @c sched_get_priority_max().
 *
 * Applications can define any value per endpoint for @a pulseCode and @a pulseValue.
 * @a pulseCode will be used by NvSciIpc to signal IPC events and should be
 * reserved for this purpose by the application. @a pulseValue can be used
 * for the application cookie data.
 *
 * @note It is only compatible with an endpoint that is opened with
 *       NvSciIpcOpenEndpoint().
 *
 * @param[in] handle        NvSciIpc endpoint handle.
 * @param[in] coid          The connection ID created from calling @c ConnectAttach_r().
 * @param[in] pulsePriority The value for pulse priority.
 * @param[in] pulseCode     The 8-bit positive pulse code specified by the user. The
 *                          values must be between @c _PULSE_CODE_MINAVAIL and
 *                          @c _PULSE_CODE_MAXAVAIL
 * @param[in] pulseValue    A pointer to the user-defined pulse value.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_NotInitialized  Indicates NvSciIpc is uninitialized.
 * - ::NvSciError_BadParameter    Indicates an invalid @a handle.
 * - ::NvSciError_NotSupported    Indicates API is not supported in provided
 *                                endpoint backend type or OS environment.
 * - ::NvSciError_ResourceError   Indicates not enough system resources.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 *
 * @pre Invocation of NvSciIpcOpenEndpoint() must be successful.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt: No
 *   - Signal handler(QNX): No
 *   - Thread: Yes
 * - Is thread safe: Yes
 * - Required Privileges(QNX): PROCMGR_AID_INTERRUPTEVENT
 * - API Group
 *   - Initialization: Yes
 *   - Runtime: No
 *   - De-initialization: No
 */
NvSciError NvSciIpcSetQnxPulseParam(NvSciIpcEndpoint handle,
	int32_t coid, int16_t pulsePriority, int16_t pulseCode,
	void *pulseValue);
#endif /* __QNX__ */
/** @} <!-- End nvsci_ipc_api --> */
/** @} <!-- End nvsci_group_ipc --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_NVSCIIPC_H */
