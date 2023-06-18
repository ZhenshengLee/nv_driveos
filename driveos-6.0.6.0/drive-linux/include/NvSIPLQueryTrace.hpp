/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVSIPLQUERYTRACE_H
#define NVSIPLQUERYTRACE_H

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Query Trace Interface - @ref NvSIPLQuery_API </b>
 *
 */

namespace nvsipl
{
/** @ingroup NvSIPLQuery_API
 * @{
 */

/** @class INvSIPLQueryTrace NvSIPLQueryTrace.hpp
 *
 * Defines the public interfaces to control the logging/tracing of the
 * @ref NvSIPLQuery_API for debugging.
 * The interfaces are implemented by class CNvSIPLQueryTrace. */
class INvSIPLQueryTrace
{
 public:

    /** @brief Defines tracing/logging levels. */
    enum TraceLevel
    {
        LevelNone = 0, /**< Indicates logging is turned off. */
        LevelError, /**< Indicates logging is turned on for errors. */
        LevelCritical, /**< Indicates logging is turned on for critical messages. */
        LevelInfo, /**< Indicates logging is turned on for information level messages. */
        LevelVerbose, /**< Indicates logging is turned on for verbose information. */
        LevelSpew, /**< Indicates logging is turned on for very verbose information. */
        LevelAll /**< Indicates logging is turned on for all messages. */
    };

    using TraceFuncPtr = void(*)(const char*, int);

    /** @brief Gets a handle to INvSIPLTrace instance.
     *
     * Static function to get a handle to singleton INvSIPLTrace implementation object.
     *
     * @pre None.
     *
     * @returns pointer to INvSIPLTrace.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: No
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    static INvSIPLQueryTrace* GetInstance(void);

    /** @brief Sets a callable trace hook.
     *
     * Function to set a callable hook to receive the messages from the library.
     * @param[in] traceHook @c std::function object, which could be a functor,
     * function pointer, or a lambda. The function object should take
     * @c const @ char* message and number of chars as arguments.
     *
     * @pre None.
     *
     * @param[in] bCallDefaultRenderer Boolean flag indicating if the message should be printed
     * to the default renderer (stderr).
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: No
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual void SetHook(TraceFuncPtr traceHook,
                         bool bCallDefaultRenderer) = 0;

    /** @brief Sets the log level.
     *
     * Function to set the level of logging.
     * Each trace statement specifies a trace level for that statement, and all traces
     * with a level greater than or equal to the current application trace level will be
     * rendered at runtime.  Traces with a level below the application trace level will
     * be ignored. The application trace level can be changed at any time to render additional
     * or fewer trace statements.
     *
     * @pre None.
     *
     * @param[in] eLevel Trace level \ref TraceLevel.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: No
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual void SetLevel(TraceLevel eLevel) = 0;

    /** @brief Disable line info (__FUNCTION__ : __LINE__: ) prefix
     *
     * Function to disable line information prefix.
     * Each log/trace is prefixed with function name and the line number.
     * Calling this function will disable the prefix.
     *
     * @pre None.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: No
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual void DisableLineInfo(void) = 0;

    /** @brief Default destructor. */
    virtual ~INvSIPLQueryTrace() = default;
};

/** @} */

} // namespace nvsipl

#endif // NVSIPLQUERYTRACE_H
