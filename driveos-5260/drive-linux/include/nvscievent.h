/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_NVSCIEVENT_H
#define INCLUDED_NVSCIEVENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <nvscierror.h>

/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSci Event Service </b>
 *
 */
/**
 * @defgroup nvsci_ipc_event Event Service APIs
 *
 * @ingroup nvsci_group_ipc
 * @{
 *
 * The NvSciEventService library provides the ability to build portable
 * event-driven applications.
 * An event is any significant occurrence or change in the state for system hardware
 * or software. An event notification is a message or notification sent by one
 * software part to another to convey that an event has taken place.
 * An event-driven model consists of an event producer and event consumers.
 * Once an event producer detects an event, it represents the event as
 * a message (or notification). An event is transmitted from an event producer to
 * event consumers through an OS-specific event channel.
 * Event consumers must be informed when an event has occurred.
 * NvSciEventService will execute the correct response (or callback)
 * to an event.
 *
 * NvSciEventService provides a mandatory abstract interface between
 * other SCI technologies (especially NvSciIpc and NvSciStreams) and
 * the application-provided event loop that services them.
 *
 * The following common object type(s) are implemented:
 *
 * - User-visible object types (for application use)
 *    - NvSciEventService: An object that subsumes all state that commonly would
 *      have been maintained in global variables.
 *    - NvSciEventNotifier: An object that a library creates using an
 *      NvSciEventService and then provides to the user, and with which the user
 *      registers an event handler that is invoked whenever the library
 *      generates an event.
 *
 * @if (SWDOCS_NVSCIIPC_INTERNAL)
 * - Non-user-visible object types (for integrating libraries with an
 *    NvSciEventService)
 *    - NvSciNativeEvent: An object that a library fills in with
 *      environment-specific information that is necessary for an event service
 *      to wait for environment-specific events (from OS or other event
 *      producers).
 *    - NvSciLocalEvent: An object with which a library can signal events
 *      directly, without going through environment-specific mechanisms. Local
 *      events are limited to cases where the signaler and waiter are in the
 *      same process, but may be more efficient than environment-specific
 *      notifiers (which typically pass through an OS).
 * @endif
 *
 * <b>Typical call flow with NvSciIpc library</b>
 *
 * 1) Init mode
 *    - NvSciEventLoopServiceCreate()
 *    - NvSciIpcInit()
 *    - NvSciIpcOpenEndpointWithEventService()
 *    - NvSciIpcGetEventNotifier()
 *    - NvSciIpcGetEndpointInfo()
 *    - NvSciIpcResetEndpoint()
 *    - Ensure a channel is established
 * ~~~~~~~~~~~~~~~~~~~~~
 *      loop {
 *          NvSciIpcGetEvent()
 *          if (event & NV_SCI_IPC_EVENT_CONN_EST_ALL) break
 *          else {
 *              NvSciEventLoopService::WaitForEvent()
 *              or
 *              NvSciEventLoopService::WaitForMultipleEvents()
 *          }
 *      }
 * ~~~~~~~~~~~~~~~~~~~~~
 * 2) Runtime mode (loop)
 *    - NvSciIpcGetEvent()
 *    - If an event is not desired,
 *      NvSciEventLoopService::WaitForEvent()
 *      or
 *      NvSciEventLoopService::WaitForMultipleEvents()
 *    - NvSciIpcRead() or NvSciIpcWrite()
 *
 * 3) Deinit mode
 *    - If an eventNotifier is not required any more,
 *      NvSciEventNotifier::Delete()
 *    - NvSciIpcCloseEndpoint()
 *    - NvSciIpcDeinit()
 *    - NvSciEventService::Delete()
 */

/*****************************************************************************/
/*                               DATA TYPES                                  */
/*****************************************************************************/

/**
 * \brief Infinite timeout for NvSciEventLoopService::WaitForEvent() or
 * NvSciEventLoopService::WaitForMultipleEvents().
 */
#define NV_SCI_EVENT_INFINITE_WAIT -1
#define NV_SCI_EVENT_PRIORITIES 4

typedef struct NvSciEventService NvSciEventService;
typedef struct NvSciEventNotifier NvSciEventNotifier;
typedef struct NvSciEventLoopService NvSciEventLoopService;

/// @cond (SWDOCS_NVSCIIPC_INTERNAL)
typedef struct NvSciNativeEvent NvSciNativeEvent;
typedef struct NvSciLocalEvent NvSciLocalEvent;
typedef struct NvSciTimerEvent NvSciTimerEvent;
typedef struct NvSciEventLoop NvSciEventLoop;
/// @endcond

/**
 * \struct NvSciEventService
 * \brief An abstract interface for a program's event handling infrastructure.
 *
 * An NvSciEventService is an abstraction that a library can use to interact
 * with the event handling infrastructure of the containing program.
 *
 * If a library needs to handle asynchronous events or report asynchronous
 * events to its users, but the library does not wish to impose a threading
 * model on its users, the library can require each user to provide an
 * NvSciEventService when the user initializes the library (or a portion
 * thereof).
 *
 * An NvSciEventService provides two categories of services related to event
 * handling:
 *
 * (1) The ability to define "event notifiers", which are objects that can
 *     notify event handling infrastructure each time an event has occurred.
 *     Note that event notifications carry no payload; it is expected that any
 *     event payload information is conveyed separately.
 *
 * (2) The ability to bind an "event handler" to each event notifier. An event
 *     handler is essentially a callback that is invoked each time the bound
 *     event notifier reports the occurrence of an event.
 */
struct NvSciEventService {
    /**
     * @if (SWDOCS_NVSCIIPC_INTERNAL)
     * \brief Defines an event notifier for a native notifier.
     *
     * @note This API is for internal use only.
     *
     * The new NvSciEventNotifier will report the occurrence of an event to
     * the event service each time the provided native notifier reports an
     * event from the OS environment.
     *
     * This function creates event notifier which reports the occurrence of
     * an event from the OS environment to the event service.
     * To configure the event bound to OS environment, it calls the function
     * in @a nativeEvent with the notifier pointer, which is a supported function
     * in the NvSciIpc library.
     *
     * @param[in]   thisEventService NvSciEventService object pointer created by
     *                               NvSciEventLoopServiceCreate().
     * @param[in]   nativeEvent      NvSciNativeEvent object pointer.
     * @param[out]  newEventNotifier NvSciEventNotifier object pointer on
     *                               success.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success         Indicates a successful operation.
     * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
     * - ::NvSciError_BadParameter    Indicates an invalid input parameters.
     * - ::NvSciError_ResourceError   Indicates not enough system resources.
     * - ::NvSciError_InvalidState    Indicates an invalid operation state.
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
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_INTERNAL)
    NvSciError (*CreateNativeEventNotifier)(
            NvSciEventService* thisEventService,
            NvSciNativeEvent* nativeEvent,
            NvSciEventNotifier** newEventNotifier);
    /// @endcond

    /**
     * @if (SWDOCS_NVSCIIPC_INTERNAL)
     * \brief Creates an intra-process local event with an event notifier
     *        that reports each event signaled through it.
     *
     * @note This API is for internal use only.
     *
     * @param[in]   thisEventService NvSciEventService object pointer created by
     *                               NvSciEventLoopServiceCreate().
     * @param[out]  newLocalEvent    NvSciLocalEvent object pointer on
     *                               success.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success             Indicates a successful operation.
     * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
     * - ::NvSciError_BadParameter        Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState        Indicates an invalid operation state.
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
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_INTERNAL)
    NvSciError (*CreateLocalEvent)(
            NvSciEventService* thisEventService,
            NvSciLocalEvent** newLocalEvent);
    /// @endcond

    /**
     * @if (SWDOCS_NVSCIIPC_NOTSUPPORT)
     * \brief Creates a timer event with an event notifier that reports each
     *        event signaled through it.
     *
     * @note This API is not yet supported.
     *
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_NOTSUPPORT)
    NvSciError (*CreateTimerEvent)(
            NvSciEventService* thisEventService,
            NvSciTimerEvent** newTimerEvent);
    /// @endcond

    /**
     * \brief Releases any resources associated with this event service.
     *
     * Before this member function is called, the caller must ensure that all
     * other member function calls on @a thisEventService have completed and the
     * caller must never again invoke any member functions on
     * @a thisEventService.
     *
     * If there any NvSciEventNotifier objects created from this event service that
     * have not been deleted yet, the resources allocated for this event
     * service will not necessarily be released until all those
     * NvSciEventNotifier objects are first deleted.
     *
     * There may also be implementation-specific conditions that result in a
     * delay in the release of resources.
     *
     * Release resources associated with NvSciEventService and NvSciEventService
     * which is created by NvSciEventLoopServiceCreate().
     *
     * @note This API must be called after releasing notifier and NvSciEventService is
     * no longer required.
     *
     * @param[in]  thisEventService NvSciEventService object pointer created by
     *                              NvSciEventLoopServiceCreate().
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
    void (*Delete)(NvSciEventService* thisEventService);
};

/**
 * \struct NvSciEventNotifier
 *
 * \brief An abstract interface to notify event to event consumer and
 * to register event handler of the event consumer client process.
 */
struct NvSciEventNotifier {
    /**
     * @if (SWDOCS_NVSCIIPC_NOTSUPPORT)
     * \brief Registers or unregisters a handler for a particular event notifier.
     *
     * @note This API is not yet supported.
     *
     * In general, handlers for distinct event notifiers may run
     * concurrently with one another. The NvSciEventService promises however
     * that no single event notifier will have its handler invoked concurrently.
     *
     * \param[in] eventNotifier The event notifier that reports each event. Must
     *                          not already be in use by another event loop.
     *
     * \param[in] callback The function to call to handle the event. If NULL,
     *                     handler will be unregistered.
     *
     * \param[in] cookie The parameter to pass to the callback.
     *
     * \param[in] priority The priority of the handler relative to other
     *                     handlers registered with eventLoop. Must be less
     *                     than @ref NV_SCI_EVENT_PRIORITIES.
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_NOTSUPPORT)
    NvSciError (*SetHandler)(NvSciEventNotifier* thisEventNotifier,
            void (*callback)(void* cookie),
            void* cookie,
            uint32_t priority);
    /// @endcond

    /**
     * \brief Unregisters any previously-registered event handler and delete
     * this event notifier.
     *
     * If the event handler's callback is concurrently executing in another
     * thread, then this function will still return immediately, but the event
     * handler will not be deleted until after the callback returns.
     *
     * This function releases the NvSciEventNotifier and unregisters the event handler.
     * It should be called when the NvSciEventNotifier is no longer required.
     *
     * @param[in]  thisEventNotifier The event handler to unregister and delete.
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
    void (*Delete)(NvSciEventNotifier* thisEventNotifier);
};

/**
 * \brief Creates a new event loop service. The number of event loops that can
 * be created in the new event loop service will be limited to at most
 * @a maxEventLoops.
 *
 * This function creates a new event loop service @a newEventLoopService which is
 * a primary instance of event service. An application must call event service
 * functions along with @a newEventLoopService.
 * The number of event loops that can be created in the new event loop service
 * will be limited to at most @a maxEventLoops.
 *
 * @param[in]   maxEventLoops       The number of event loops, it must be 1.
 * @param[out]  newEventLoopService NvSciEventNotifier object double pointer.
 *
 * @return ::NvSciError, the completion code of operations:
 * - ::NvSciError_Success             Indicates a successful operation.
 * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::NvSciError_NotSupported        Indicates a condition is unsupported.
 * - ::NvSciError_InvalidState        Indicates an invalid operation state.
 * - ::NvSciError_BadParameter        Indicates an invalid or NULL argument.
 * - ::NvSciError_ResourceError       Indicates not enough system resources.
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
NvSciError NvSciEventLoopServiceCreate(
        size_t maxEventLoops,
        NvSciEventLoopService** newEventLoopService);

/**
 * \struct NvSciEventLoopService

 * \brief An abstract interface that event consumer can wait for
 * events using event notifier in event loop.
 */
struct NvSciEventLoopService {
    NvSciEventService EventService;

    /**
     * @if (SWDOCS_NVSCIIPC_NOTSUPPORT)
     * \brief Creates an event loop that can handle events for NvSciEventLoopService.
     *
     * @note This API is not yet supported.
     *
     * The user is responsible for running the event loop from a thread by
     * calling the new event loop's Run() function.
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_NOTSUPPORT)
    NvSciError (*CreateEventLoop)(NvSciEventLoopService* eventLoopService,
            NvSciEventLoop** eventLoop);
    /// @endcond

    /**
     * \brief Waits up to a configurable timeout for a particular event
     * notification, servicing events with configured callbacks in the interim.
     *
     * Any asynchronous event notifiers that are pending before calling
     * this function will be claimed by some thread for handling before
     * this function returns.
     *
     * @a eventNotifier must have been created through EventService.
     *
     * @note This function must not be called from an event notifier
     *          callback.
     *
     * This function waits up to a configurable timeout to receive a pulse event
     * which is configured on NvSciQnxEventService_CreateNativeEventNotifier().
     * @a eventNotifier must have been created through EventService before calling.
     *
     * @param[in]  eventNotifier NvSciEventNotifier object pointer.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for an infinite timeout, for example,
     *                           the value @ref NV_SCI_EVENT_INFINITE_WAIT.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success            Indicates a successful operation.
     * - ::NvSciError_BadParameter       Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState       Indicates an invalid operation state.
     * - ::NvSciError_NotSupported       Indicates a condition is unsupported.
     * - ::NvSciError_Timeout            Indicates a timeout occurrence.
     * - ::NvSciError_ResourceError      Indicates not enough system resources.
     * - ::NvSciError_InterruptedCall    Indicates an interrupt occurred.
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
    NvSciError (*WaitForEvent)(
            NvSciEventNotifier* eventNotifier,
            int64_t microseconds);

    /**
     * \brief Waits up to a configurable timeout for any of a set of
     * particular event notifications, servicing events with configured
     * callbacks in the interim.
     *
     * Any asynchronous event notifiers that are pending before calling
     * this function will be claimed by some thread for handling before
     * this function returns.
     *
     * Each event notifier in @a eventNotifierArray must have been created
     * through EventService.
     *
     * On a successful return, for each integer `i` in the range
     * `[0, eventNotifierCount]`, `newEventArray[i]` will be true only if
     * `eventNotifierArray[i]` had a new event.
     *
     * @note This function must not be called from an event notifier
     *          callback.
     *
     * @param[in]  eventNotifierArray Array of NvSciEventNotifier object
     *                                pointers.
     * @param[in]  eventNotifierCount Event notifier count in eventNotifierArray.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for infinite timeout, for example,
     *                           the value @ref NV_SCI_EVENT_INFINITE_WAIT.
     * @param[out] newEventArray Array of event occurrence.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success            Indicates a successful operation.
     * - ::NvSciError_BadParameter       Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState       Indicates an invalid operation state.
     * - ::NvSciError_NotSupported       Indicates a condition is not supported.
     * - ::NvSciError_Timeout            Indicates a timeout occurrence.
     * - ::NvSciError_ResourceError      Indicates not enough system resources.
     * - ::NvSciError_InterruptedCall    Indicates an interrupt occurred.
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
    NvSciError (*WaitForMultipleEvents)(
            NvSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool* newEventArray);
};

/** @} <!-- End nvsci_ipc_event --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_NVSCIEVENT_H */
