/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "nvscierror.h"

/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSci Event Service </b>
 *
 * @version 1.4
 *
 * @section 1.4 Sep/05/2023
 * - Add NvSciEventLoopServiceCreateSafeX() in Linux build
 * ---------------------------------------------------------------------------
 * @section 1.3 Jun/01/2023
 * - Set NvSciLocalEvent object to public
 * - Add NvSciEventMoveNotifier()
 * - Add NvSciEventLoopServiceCreateSafeX() in QNX build
 *       and update with atomic operation for local event
 *       NvSciLocalEvent::Signal()
 *       NvSciEventLoopService::WaitForEvent()
 *       NvSciEventLoopService::WaitForMultipleEvents()
 *       NvSciEventLoopService::WaitForMultipleEventsExt()
 * ---------------------------------------------------------------------------
 * @section 1.2 Mar/07/2023
 * - Deprecate NvSciEventLoopServiceCreate in safety build
 * ---------------------------------------------------------------------------
 * @section 1.1 Dec/15/2022
 * - Add NvSciError_NotPermitted error for incorrect API group usage
 *   in QNX safety only (NvDVMS).
 *       NvSciEventNotifier::SetHandler()
 *       NvSciEventLoopServiceCreateSafe()
 * ---------------------------------------------------------------------------
 * @section 1.0 Jun/23/2022
 * - Add NvSciEventLoopServiceCreateSafe()
 * - Add NvSciEventInspect()
 * - Add NvSciEventCheckVersionCompatibility()
 * - Add NvSciEventMajorVersion, NvSciEventMinorVersion const var
 */

/*
 * use constant global version variable instead of macro for consistency with
 * version check API of existing NvSci family
 */

/** @brief NvSciEvent API Major version number */
static const uint32_t NvSciEventMajorVersion = 1U;

/** @brief NvSciEvent API Minor version number */
static const uint32_t NvSciEventMinorVersion = 4U;

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
 *    - NvSciLocalEvent: An object representing a local event with which
 *      a library can signal events directly, without going through
 *      environment-specific mechanisms. Local events are limited to cases where
 *      the signaler and waiter are in the same process, but may be more
 *      efficient than environment-specific notifiers (which typically pass
 *      through an OS).
 *
 * - Non-user-visible object types (for integrating libraries with an
 *    NvSciEventService)
 *    - NvSciNativeEvent: An object representing a native event that a library
 *      fills in with environment-specific information that is necessary for
 *      an event service to wait for environment-specific events (from OS or
 *      other event producers).
 *
 * @note <b>For startup-up sequence with NvSciEventService library, refer to
 * "Start-up sequence" in NvSciIpc library</b>
 *
 * @pre No nested call to waiting for events
 * - These three APIs in waiting for events must not be called from
 * an event handler registered with NvSciEventNotifier.
 *      - NvSciEventLoopService::WaitForEvent()
 *      - NvSciEventLoopService::WaitForMultipleEvents()
 *      - NvSciEventLoopService::WaitForMultipleEventsExt()
 *
 * @pre No multi-thread call to waiting for events
 * - These three APIs in waiting for events must not be called from
 * multiple thread if same NvSciEventLoopSerivce instance is used.
 *      - NvSciEventLoopService::WaitForEvent()
 *      - NvSciEventLoopService::WaitForMultipleEvents()
 *      - NvSciEventLoopService::WaitForMultipleEventsExt()
 */

/*******************************************************************/
/********************* ACCESS CONTROL ******************************/
/*******************************************************************/

/**
 * @page page_access_control NvSciEventService Access Control
 * @section sec_access_control Common Privileges
 *
 * <b>Common QNX privileges required for all NvSciEventService APIs</b>
 *   - Service/Group Name: proc_boot, libc, libslog2, libnvivc, libnvsciipc
 *                         libnvscievent, libnvos_s3_safety, libnvdvms_client
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
typedef struct NvSciNativeEvent NvSciNativeEvent;
typedef struct NvSciLocalEvent NvSciLocalEvent;

/// @cond (SWDOCS_NVSCIIPC_INTERNAL)
typedef struct NvSciTimerEvent NvSciTimerEvent;
typedef struct NvSciEventLoop NvSciEventLoop;
/// @endcond

/**
 * \struct NvSciLocalEvent
 * \brief An OS-agnostic object that sends signal to another thread
 *        in the same process.
 */
struct NvSciLocalEvent {
    /** \brief Event notifier associated with this local event. */
    NvSciEventNotifier* eventNotifier;

    /**
     * \brief Sends an intra-process local event signal.
     *
     * Any thread which is blocked by local event notifier associated with
     * local event will be unblocked by this signal.
     *
     * @param[in]  thisLocalEvent NvSciLocalEvent object pointer created by
     *                            NvSciEventService::CreateLocalEvent()
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success         Indicates a successful operation.
     * - ::NvSciError_BadParameter    Indicates an invalid input parameter.
     * - ::NvSciError_TryItAgain       Indicates an kernel pulse queue shortage.
     * - ::NvSciError_InvalidState      Indicates an invalid operation state.
     *
     * @pre NvSciEventService::CreateLocalEvent() must be called.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: Yes
     *   - Async/Sync: Sync
     * - Required Privileges: None
     * - API Group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    NvSciError (*Signal)(NvSciLocalEvent* thisLocalEvent);

    /**
     * \brief Releases any resources associated with this local event.
     *
     * This function must be called after releasing notifier and when
     * NvSciLocalEvent is no longer required.
     *
     * @note This API can be called in Init mode to release resources
     *       in error handling or to test functionality.
     *
     * @param[in]  thisLocalEvent NvSciLocalEvent object pointer created by
     *                            NvSciEventService::CreateLocalEvent().
     *
     * @return @c void
     *
     * @pre NvSciEventService::CreateLocalEvent() must be called.
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
    void (*Delete)(NvSciLocalEvent* thisLocalEvent);
};

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
     *                               NvSciEventLoopServiceCreateSafe().
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
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
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
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_INTERNAL)
    NvSciError (*CreateNativeEventNotifier)(
            NvSciEventService* thisEventService,
            NvSciNativeEvent* nativeEvent,
            NvSciEventNotifier** newEventNotifier);
    /// @endcond

    /**
     * \brief Creates an intra-process local event with an event notifier
     *        that reports each event signaled through it.
     *
     * @param[in]   thisEventService NvSciEventService object pointer created by
     *                               NvSciEventLoopServiceCreateSafe().
     * @param[out]  newLocalEvent    NvSciLocalEvent object pointer on
     *                               success.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success             Indicates a successful operation.
     * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
     * - ::NvSciError_BadParameter        Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState        Indicates an invalid operation state.
     * - ::NvSciError_NotPermitted        Indicates incorrect API group usage.
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
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
    NvSciError (*CreateLocalEvent)(
            NvSciEventService* thisEventService,
            NvSciLocalEvent** newLocalEvent);

    /**
     * @if (SWDOCS_NVSCIIPC_INTERNAL)
     * \brief Creates a timer event with an event notifier that reports each
     *        event signaled through it.
     *
     * @note This is for internal use only in Linux.
     *
     * @endif
     */
    /// @cond (SWDOCS_NVSCIIPC_INTERNAL)
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
     * which is created by NvSciEventLoopServiceCreateSafe().
     *
     * @note This API must be called after releasing notifier and
     *       NvSciEventService is no longer required.
     *
     * @note This API can be called in Init mode to release resources
     *       in error handling or to test functionality.
     *
     * @param[in]  thisEventService NvSciEventService object pointer created by
     *                              NvSciEventLoopServiceCreateSafe().
     *
     * @return @c void
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
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
     *   - De-Init: Yes
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
     * \brief Registers or unregisters a handler for a particular event notifier.
     *
     * In general, handlers for distinct event notifiers may run
     * concurrently with one another. The NvSciEventService promises however
     * that no single event notifier will have its handler invoked concurrently.
     *
     * Handlers for both NvSciNativeEvent and NvSciLocalEvent can be registered.
     *
     * \param[in] eventNotifier The event notifier that reports each event. Must
     *                          not already be in use by another event loop.
     *
     * \param[in] callback The function to call to handle the event. If NULL,
     *                     handler will be unregistered.
     *
     * \param[in] cookie The parameter to pass to the callback.
     *
     * \param[in] priority The parameter is not supported and any value is
     *                     ignored.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success             Indicates a successful operation.
     * - ::NvSciError_BadParameter        Indicates an invalid or NULL argument.
     * - ::NvSciError_NotPermitted        Indicates incorrect API group usage.
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
     *      NvSciIpcGetEventNotifier() must be called for NvSciNativeEvent.
     *      NvSciEventService::CreateLocalEvent() must be called
     *      for NvSciLocalEvent.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: Yes
     *   - Signal handler: Yes
     *   - Thread-safe: Yes
     *   - Async/Sync: Sync
     * - Required Privileges(QNX): None
     * - API Group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    NvSciError (*SetHandler)(NvSciEventNotifier* thisEventNotifier,
            void (*callback)(void* cookie),
            void* cookie,
            uint32_t priority);

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
     * @note This API can be called in Init mode to release resources
     *       in error handling or to test functionality.
     *
     * @param[in]  thisEventNotifier The event handler to unregister and delete.
     *
     * @return @c void
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
     *      NvSciIpcGetEventNotifier() must be called for NvSciNativeEvent.
     *      NvSciEventService::CreateLocalEvent() must be called
     *      for NvSciLocalEvent.
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
     *   - De-Init: Yes
     */
    void (*Delete)(NvSciEventNotifier* thisEventNotifier);
};

/**
 * \brief Creates a new event loop service. The number of event loops that can
 * be created in the new event loop service will be limited to at most
 * @a maxEventLoops.
 *
 * @warning This API is deprecated and returns NvSciError_NotSupported without
 * doing anything in safety build.
 * Use NvSciEventLoopServiceCreateSafe() instead of this.
 */
NvSciError NvSciEventLoopServiceCreate(
        size_t maxEventLoops,
        NvSciEventLoopService** newEventLoopService);

/**
 * \brief Creates a new event loop service. The number of event loops that can
 * be created in the new event loop service will be limited to at most
 * @a maxEventLoops.
 *
 * This function creates a new event loop service @a newEventLoopService which
 * is a primary instance of event service. An application must call event
 * service functions along with @a newEventLoopService.
 * The number of event loops that can be created in the new event loop service
 * will be limited to at most @a maxEventLoops.
 *
 * @note The following information is specific to QNX and about @a config.
 * It uses ChannelCreatePulsePool() to create channel with the private pulse
 * pool. The ChannelCreatePulsePool() needs this parameter to specify the
 * attributes of the private pulse pool, including what to do when there are no
 * available pulses and @a config is used to pass it.
 * ~~~~~~~~~~~~~~~~~~~~~
 * struct nto_channel_config {
 *     struct sigevent event;
 *     unsigned num_pulses;
 *     unsigned rearm_threshold;
 *     unsigned options;
 *     unsigned reserved[3];
 * }
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * More information can be found in QNX OS manual page.
 *
 * @param[in]   maxEventLoops       The number of event loops, it must be 1.
 * @param[in]   config              OS-specific configuration parameter.
 *                                  It should NULL in Linux.
 * @param[out]  newEventLoopService NvSciEventNotifier object double pointer.
 *
 * @return ::NvSciError, the completion code of operations:
 * - ::NvSciError_Success             Indicates a successful operation.
 * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::NvSciError_NotSupported        Indicates a condition is unsupported.
 * - ::NvSciError_InvalidState        Indicates an invalid operation state.
 * - ::NvSciError_BadParameter        Indicates an invalid or NULL argument.
 * - ::NvSciError_ResourceError       Indicates not enough system resources.
 * - ::NvSciError_NotPermitted        Indicates incorrect API group usage.
 *
 * @pre Threshold of pulse number in private pulse pool
 * - @a num_pulses in nto_channel_config shall be decided based on how many
 * event notifications can happen simulatenously. For native event, there are
 * three event notifications, reset/read/write. Considering native and local
 * event together, the formula is
 * ~~~~~~~~~~~~~~~~~~~~~
 * num_pulses = 3 * native events (same as endpoints) + local events
 * ~~~~~~~~~~~~~~~~~~~~~
 * This formula gives you a bottom line of how many pulses you need. If you
 * experience a pulse shortage in normal operation, you may need more pulses,
 * e.g. increasing num_pulses by double of previous setting.
 *
 * @pre Handle an event storm
 * - If a user wants to detect an event strom and stop the event from being
 * notified, the user needs to create a separate thread to receive semaphore
 * event which is set in nto_channel_config structure and call sem_wait() in
 * that thread. When the thread is awaken by the semaphore event, it can call
 * NvSciEventInspect() API to handle event storm.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): PROCMGR_AID_PUBLIC_CHANNEL
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciEventLoopServiceCreateSafe(
        size_t maxEventLoops,
        void* config,
        NvSciEventLoopService** newEventLoopService);

/**
 * \brief Creates a new event loop service that has the same properties as
 * created by NvSciEventLoopServiceCreateSafe(), but supports faster local
 * event.
 *
 * For a detailed description of the API, see NvSciEventLoopServiceCreateSafe().
 *
 * @note In case of Linux/x86, This is just for API parity and not benefit
 *       in terms of local event performance.
 *
 * @param[in]   maxEventLoops       The number of event loops, it must be 1.
 * @param[in]   config              OS-specific configuration parameter.
 *                                  It should NULL in Linux.
 * @param[out]  newEventLoopService NvSciEventNotifier object double pointer.
 *
 * @return ::NvSciError, the completion code of operations:
 * - ::NvSciError_Success             Indicates a successful operation.
 * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::NvSciError_NotSupported        Indicates a condition is unsupported.
 * - ::NvSciError_InvalidState        Indicates an invalid operation state.
 * - ::NvSciError_BadParameter        Indicates an invalid or NULL argument.
 * - ::NvSciError_ResourceError       Indicates not enough system resources.
 * - ::NvSciError_NotPermitted        Indicates incorrect API group usage.
 *
 * @pre No infinite timeout with fast local event
 * - The following waiting functions of the event loop service created with
 * NvSciEventLoopServiceCreateSafeX() do not support infinite timeout and
 * will return NvSciError_BadParameter when they are called with infinite
 * timeout.
 *      - WaitForEvent()
 *      - WaitForMultipleEvents()
 *      - WaitForMultipleEventsExt()
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): PROCMGR_AID_PUBLIC_CHANNEL
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciEventLoopServiceCreateSafeX(
        size_t maxEventLoops,
        void* config,
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
     * This function waits up to a configurable timeout to receive a pulse event
     * which is configured on NvSciQnxEventService_CreateNativeEventNotifier().
     * @a eventNotifier must have been created through EventService before calling.
     *
     * @param[in]  eventNotifier NvSciEventNotifier object pointer.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for an infinite timeout, for example,
     *                           the value @ref NV_SCI_EVENT_INFINITE_WAIT.
     *                           Timeout must be 2msec or greater as high
     *                           precision timer is not supported in DRIVE OS.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success            Indicates a successful operation.
     * - ::NvSciError_BadParameter       Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState       Indicates an invalid operation state.
     * - ::NvSciError_NotSupported       Indicates another thread already
     *                                   waiting for event.
     * - ::NvSciError_Timeout            Indicates a timeout occurrence.
     * - ::NvSciError_ResourceError      Indicates not enough system resources.
     * - ::NvSciError_InterruptedCall    Indicates an interrupt occurred.
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
     *      NvSciIpcGetEventNotifier() must be called for NvSciNativeEvent.
     *      NvSciEventService::CreateLocalEvent() must be called
     *      for NvSciLocalEvent.
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
     * `[0, eventNotifierCount-1]`, `newEventArray[i]` will be true only if
     * `eventNotifierArray[i]` had a new event.
     *
     * @note This function will be deprecated in furture and user must use
     *          the newer version of the API which is
     *          NvSciEventWaitForMultipleEventsExt
     *
     * @param[in]  eventNotifierArray Array of NvSciEventNotifier object
     *                                pointers. It should not be NULL.
     * @param[in]  eventNotifierCount Event notifier count in eventNotifierArray.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for infinite timeout, for example,
     *                           the value @ref NV_SCI_EVENT_INFINITE_WAIT.
     *                           Timeout must be 2msec or greater as high
     *                           precision timer is not supported in DRIVE OS.
     * @param[out] newEventArray Array of event occurrence.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success            Indicates a successful operation.
     * - ::NvSciError_BadParameter       Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState       Indicates an invalid operation state.
     * - ::NvSciError_NotSupported       Indicates another thread already
     *                                   waiting for events.
     * - ::NvSciError_Timeout            Indicates a timeout occurrence.
     * - ::NvSciError_ResourceError      Indicates not enough system resources.
     * - ::NvSciError_InterruptedCall    Indicates an interrupt occurred.
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
     *      NvSciIpcGetEventNotifier() must be called for NvSciNativeEvent.
     *      NvSciEventService::CreateLocalEvent() must be called
     *      for NvSciLocalEvent.
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
    NvSciError (*WaitForMultipleEvents)(
            NvSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool* newEventArray);

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
     * `[0, eventNotifierCount-1]`, `newEventArray[i]` will be true only if
     * `eventNotifierArray[i]` had a new event.
     *
     * @a eventNotifierArray can be NULL then this function will not update
     * @a newEventArray but still service events with configured callbacks,
     * which are created with @a eventService until the specified timeout period.
     * For this NULL case, timeout in  @a microseconds works in this way.
     * 1. @a microseconds > 0
     *    Callbacks will continue to be served until the timeout happens. If
     *    any callback takes long more than timeout, other callbacks associated
     *    with events which arrives before timeout will be served even after
     *    timeout.
     * 2. @a microseconds = -1 (NV_SCI_EVENT_INFINITE_WAIT)
     *    Callbacks will continue to be served and this API will never
     *    returns.
     *
     * @param[in]  eventService Pointer to the event service object
     * @param[in]  eventNotifierArray Array of NvSciEventNotifier object
     *                                pointers. If it is NULL,
     *                                @a eventNotifierCount should be zero and
     *                                @a newEventArray should be NULL together.
     * @param[in]  eventNotifierCount Event notifier count in eventNotifierArray.
     * @param[in]  microseconds  A 64-bit integer timeout in microsecond.
     *                           Set to -1 for infinite timeout, for example,
     *                           the value @ref NV_SCI_EVENT_INFINITE_WAIT.
     *                           Timeout must be 2msec or greater as high
     *                           precision timer is not supported in DRIVE OS.
     * @param[out] newEventArray Array of event occurrence.
     *
     * @return ::NvSciError, the completion code of operations:
     * - ::NvSciError_Success            Indicates a successful operation.
     * - ::NvSciError_BadParameter       Indicates an invalid input parameter.
     * - ::NvSciError_InvalidState       Indicates an invalid operation state.
     * - ::NvSciError_NotSupported       Indicates another thread already
     *                                   waiting for events.
     * - ::NvSciError_Timeout            Indicates a timeout occurrence.
     * - ::NvSciError_ResourceError      Indicates not enough system resources.
     * - ::NvSciError_InterruptedCall    Indicates an interrupt occurred.
     *
     * @pre NvSciEventLoopServiceCreateSafe() must be called.
     *      NvSciIpcGetEventNotifier() must be called for NvSciNativeEvent.
     *      NvSciEventService::CreateLocalEvent() must be called
     *      for NvSciLocalEvent.
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
    NvSciError (*WaitForMultipleEventsExt)(
            NvSciEventService *eventService,
            NvSciEventNotifier* const * eventNotifierArray,
            size_t eventNotifierCount,
            int64_t microseconds,
            bool* newEventArray);
};

#ifdef __QNX__
/**
 * \brief Inspect events
 *
 * <b>This API is specific to QNX OS.</b>
 *
 * If event storm happens in specific event notifiers,
 * this API unregisters relevant events and returns the event notifiers
 * bound to the events. This API is used in a thread receiving semaphore
 * event when event can't be obtained from the pool any more due to
 * event storm. In order to prevent unwanted breakage on handling event,
 * Do not call this API when semaphore event is not triggered.
 *
 * @param[in]  thisEventService   NvSciEventService object pointer created
 *                                by NvSciEventLoopServiceCreateSafe().
 * @param[in]  numEvents          A threshold value to unregister events.
 *                                This shall be less than num_pulses which
 *                                is configured in .
 * @param[in]  eventNotifierCount Notifier count in @eventNotifierArray.
 * @param[out] eventNotifierArray Array of NvSciNotifier which has
 *                                excessive event signalling
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_BadParameter    Indicates an invalid or NULL argument.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 *
 * @pre NvSciEventLoopServiceCreateSafe() must be called.
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
NvSciError NvSciEventInspect(
        NvSciEventService *thisEventService,
        uint32_t numEvents,
        uint32_t eventNotifierCount,
        NvSciEventNotifier** eventNotifierArray);
#endif /* __QNX__ */

/**
 * \brief Move event notifier
 *
 * This API exports event Notifier from old EventService and import the Notifier
 * to new EventService. After the event notifier is migrated,
 * the old EventService can not wait the event associated with the notifier.
 * Instead, use the new EventService waits for the event.
 *
 * @note This API supports local event and native event notifier only.
 *       In case of native event, limited to Intra-VM and Inter-VM backend only.
 *
 * @param[in] oldEventService   NvSciEventService object pointer to export
 *                              event notifier
 * @param[in] newEventService   NvSciEventService object pointer to import
 *                              event notifier
 * @param[in] eventNotifier     Event notifier object pointer.
 *
 * @return ::NvSciError, the completion code of the operation:
 * - ::NvSciError_Success         Indicates a successful operation.
 * - ::NvSciError_BadParameter    Indicates an invalid or NULL argument.
 * - ::NvSciError_InvalidState    Indicates an invalid operation state.
 * - ::NvSciError_InsufficientMemory  Indicates memory is not sufficient.
 * - ::NvSciError_NotPermitted    Indicates incorrect API group usage.
 *
 * @pre NvSciEventLoopServiceCreateSafe() must be successful per
 *      oldEventService and newEventService.
 *
 * @pre No pending event before moving notifier
 * - When this API is called, no local and native event associated with the
 * event notifier must be waited in the EventService and sent by remote peer.
 * NvSciIpc APIs using the same endpoint will be blocked till this API returns.
 *
 * @pre Complete moving notifier
 * - To complete sequence of this API for native event, client process shall
 * call NvSciIpcGetEventSafe() till NvSciIpcGetEventSafe() returns one of
 * NV_SCI_IPC_EVENT_CONN_EST_ALL events after this API. This can be done
 * by calling NvSciIpcGetEventSafe() and WaitForEvent() family API pair,
 * or calling NvSciIpcGetEventSafe() with sleep in polling mode (with disabled
 * notification mode).
 *
 * @pre Handshake to ensure no data transaction before
 * moving notifier
 * - Before calling this API to move native event notifier,
 * user shall make sure that channel connection is established and
 * there is no more outstanding data transaction between two endpoints
 * through any handshaking mechanism at client level.
 * If this is not guranteed, pending notification of remote endpoint might
 * fail since local endpoint already enters move-notifier sequence.
 * NV_SCI_IPC_EVENT_WRITE_EMPTY of NvSciIpcGetEventSafe() can be used
 * to make sure if remote endpoint read all sent data.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: No
 *   - Async/Sync: Sync
 * - Required Privileges(QNX): None
 * - API Group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvSciError NvSciEventMoveNotifier(
        NvSciEventService *oldEventService,
        NvSciEventService *newEventService,
        NvSciEventNotifier *eventNotifier);

/**
 * @brief Check NvSciEventSerivice library version compatibility
 *
 * Checks if loaded NvSciEvent library version is compatible with
 * the version the application was compiled against.
 * This function checks loaded NvSciEvent library version with input NvSciEvent
 * library version and sets output variable true provided major version of the
 * loaded library is same as @a majorVer and minor version of the
 * loaded library is not less than @a minorVer, else sets output to false
 *
 * @param[in]  majorVer build major version.
 * @param[in]  minorVer build minor version.
 * @param[out] isCompatible boolean value stating if loaded NvSciEvent library
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
NvSciError NvSciEventCheckVersionCompatibility(
    uint32_t majorVer,
    uint32_t minorVer,
    bool* isCompatible);

/** @} <!-- End nvsci_ipc_event --> */

#ifdef __cplusplus
}
#endif
#endif /* INCLUDED_NVSCIEVENT_H */

