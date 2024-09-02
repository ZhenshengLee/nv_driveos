/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NVSIPLDEVBLKTRACE_H
#define NVSIPLDEVBLKTRACE_H

#include <cstdint>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: DeviceBlock Trace Interface - @ref NvSIPLDevBlkTrace_API </b>
 *
 */

/** @defgroup NvSIPLDevBlkTrace_API NvSIPL Device Block Trace
 *
 * @brief Provides interfaces to configure the trace level for SIPL Device Block.
 *
 * @ingroup NvSIPL */

namespace nvsipl
{

/**@ingroup NvSIPLDevBlkTrace_API
 * @{
 */

/** @class INvSIPLDeviceBlockTrace NvSIPLDeviceBlockTrace.hpp
 *
 * @brief Describes the interfaces of NvSIPLDeviceBlockTrace.
 *
 * Defines the public interfaces to control the logging/tracing
 * of the @ref NvSIPLDevBlkTrace_API for debug purposes.
 */
class INvSIPLDeviceBlockTrace
{
public:

   /** @brief Defines tracing/logging levels. */
   enum TraceLevel : std::uint8_t
   {
       /** Indicates logging is turned off. */
       LevelNone = 0,
       /** Indicates logging is turned on for errors. */
       LevelError,
       /** Indicates logging is turned on for critical warnings. */
       LevelWarning,
       /** Indicates logging is turned on for information level messages. */
       LevelInfo,
       /** Indicates logging is turned on for every print statement. */
       LevelDebug
   };

#if !(NV_IS_SAFETY)

   /** @brief Gets a handle to @ref INvSIPLDeviceBlockTrace instance.
    *
    * Static function to get a handle to singleton @ref INvSIPLDeviceBlockTrace implementation object.
    *
    * @note On safety build, this function always returns NULL.
    *
    * @pre None.
    *
    * @retval (INvSIPLDeviceBlockTrace*) Pointer to @ref INvSIPLDeviceBlockTrace. */
   static INvSIPLDeviceBlockTrace* GetInstance(void);
   using TraceFuncPtr = void(*)(const char*, int);
   /** @brief Sets a callable trace hook.
    *
    * Function to set a callable hook to receive the messages from the library.
    *
    * @pre None.
    *
    * @param[in] traceHook @c std::function object, which could be a functor,
    *                      function pointer, or a lambda. The function object must include
    *                      @c const @c char* message and number of chars as arguments.
    * @param[in] bCallDefaultRenderer Boolean flag indicating if the message should be printed
    *                                 to the default renderer (stderr).
    */
   virtual void SetHook(TraceFuncPtr traceHook,
                        bool const bCallDefaultRenderer) = 0;

   /** @brief Sets the log level.
    *
    * Function to set the level of logging.
    * Each trace statement specifies a trace level for that statement, and all traces
    * with a level greater than or equal to the current application trace level will be
    * rendered at runtime. Traces with a level below the application trace level will
    * be ignored. The application trace level can be changed at any time to render additional
    * or fewer trace statements.
    *
    * @pre None.
    *
    * @param[in] eLevel Trace level @ref TraceLevel.
    */
   virtual void SetLevel(TraceLevel const eLevel) = 0;

   /**
    * @brief Gets the log level.
    *
    * Function to fetch the level of logging.
    *
    * @pre None.
    *
    * @retval (TraceLevel) Level of logging.
    */
   virtual TraceLevel GetLevel() = 0;

   /** @brief Disable line info (__FUNCTION__ : __LINE__: ) prefix
    *
    * Function to disable line information prefix.
    * Each log/trace is prefixed with function name and the line number.
    * Calling this function will disable the prefix.
    *
    * @pre None.
    */
    virtual void DisableLineInfo(void) = 0;

    /**
     * @brief Log a trace message.
     *
     * Function to log a trace message.
     *
     * @pre None.
     *
     * @param[in] eLevel    Level to log the trace at.
     * @param[in] func      Function name.
     * @param[in] file      File name.
     * @param[in] line      Line number.
     * @param[in] pformat   Format string.
     * @param[in] ...       Variadic list of arguments corresponding to the format string.
     */
    virtual void Trace(TraceLevel eLevel,
                       const char* func,
                       const char* file,
                       int line,
                       const char *pformat,
                       ...) = 0;
#endif //!(NV_IS_SAFETY)

   /** @brief Default destructor. */
   virtual ~INvSIPLDeviceBlockTrace() = default;

protected :
   /**
     * @brief Default constructor.
     */
    INvSIPLDeviceBlockTrace() = default;

    /**
     * @brief Default INvSIPLDeviceBlockTrace copy assignment operator.
     */
    INvSIPLDeviceBlockTrace(INvSIPLDeviceBlockTrace const &) = default;

    /**
     * @brief Default INvSIPLDeviceBlockTrace move assignment operator.
     */
    INvSIPLDeviceBlockTrace(INvSIPLDeviceBlockTrace &&) = default;

    /**
     * @brief Default INvSIPLDeviceBlockTrace copy assignment operator.
     */
    INvSIPLDeviceBlockTrace& operator=( INvSIPLDeviceBlockTrace const &) & = default;

    /**
     * @brief Default INvSIPLDeviceBlockTrace from move assignment operator.
     */
    INvSIPLDeviceBlockTrace& operator=(INvSIPLDeviceBlockTrace&&) & = default;
};

/** @} */

} // namespace nvsipl


#endif // NVSIPLDEVBLKTRACE_H
