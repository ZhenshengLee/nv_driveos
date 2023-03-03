/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVMTRACE_H
#define NVMTRACE_H

#include "NvSIPLDeviceBlockTrace.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdarg>
#include <cstring>

#if NV_IS_SAFETY
#include "sipl_error.h"

namespace nvsipl {
    /**
     * @brief Dummy implementation of the Device Block Trace interface.
     *
     * This class is never instantiated or exposed publically, but needs to
     * exist so that the compiler has somewhere to generate RTTI metadata for
     * nvsipl::INvSIPLDeviceBlockTrace (which is part of the ABI for
     * libnvsipl_devblk_ddi).
     */
    class DummyTraceImpl : public nvsipl::INvSIPLDeviceBlockTrace {
        DummyTraceImpl();
    };
}

#else //NV_IS_SAFETY

#define _FILENAME_ ((strrchr(__FILE__, '/') != NULL) ? (strrchr(__FILE__, '/') + 1) : __FILE__)

//! Quick-log a message at debugging level
#define LOG_DEBUG(...) \
    INvSIPLDeviceBlockTrace::GetInstance()->Trace(LEVEL_DBG, __FUNCTION__, _FILENAME_, __LINE__, __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...) \
    INvSIPLDeviceBlockTrace::GetInstance()->Trace(LEVEL_INFO, __FUNCTION__, _FILENAME_, __LINE__, __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...) \
    INvSIPLDeviceBlockTrace::GetInstance()->Trace(LEVEL_WARN, __FUNCTION__, _FILENAME_, __LINE__, __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...) \
    INvSIPLDeviceBlockTrace::GetInstance()->Trace(LEVEL_ERR, __FUNCTION__, _FILENAME_, __LINE__, __VA_ARGS__)

/** @brief Logging level for no traces output. */
#define LEVEL_NONE INvSIPLDeviceBlockTrace::LevelNone

/** @brief Logging level for errors. */
#define LEVEL_ERR INvSIPLDeviceBlockTrace::LevelError

/** @brief Logging level for warnings. */
#define LEVEL_WARN INvSIPLDeviceBlockTrace::LevelWarning

/** @brief Logging level for info. */
#define LEVEL_INFO INvSIPLDeviceBlockTrace::LevelInfo

/** @brief Logging level for debug output. */
#define LEVEL_DBG INvSIPLDeviceBlockTrace::LevelDebug

namespace nvsipl
{

/**
 * @brief Singleton class for handling all tracing functions.
 *
 * The tracing functionality is not thread-safe. Logs may be incorrect if
 * traced at the same time. Prints to cout by default. A custom function for
 * logs to be sent to may be set, and the cout print may be disabled.
 * The line info (thread, file, line, function name) is prepended to logs by
 * default. Line info may be disabled.
 *
 * Tracing is only enabled if the logging severity level is higher or equal
 * than the set level. Other traces are ignored. Traces are constructed with
 * line info (if enabled) and the logging message. The constructed trace message
 * is stored internal to the function, and sent to the hook function
 * (if enabled), cout (if enabled).
 *
 */
class CNvSIPLDeviceBlockTrace: public INvSIPLDeviceBlockTrace
{
public:

    /**
     * @brief Gets the reference to the singleton class.
     *
     * Gets the static instance of the class. This function is thread-safe as
     * the singleton is static-initalized in the function.
     */
    static CNvSIPLDeviceBlockTrace& GetTrace();

    /**
     * @brief Log a trace message.
     *
     * Logs a trace message, which gets sent to the set trace hook.
     * The function also will print to the default renderer (cout) if
     * m_bDefaultRender is set to true. Line info is limited and truncated
     * to m_uMaxLineInfo and the total length of the trace message is limited
     * and truncated to m_uMaxLength.
     *
     * @param[in] eLevel    Level to log the trace at.
     * @param[in] func      Function name.
     * @param[in] file      File name.
     * @param[in] line      Line number.
     * @param[in] pformat   Format string.
     * @param[in] ...       Variadic list of arguments corresponding to the format string.
     */
    void Trace(TraceLevel const eLevel,
               const char* const func,
               const char* const chfile,
               int32_t const line,
               const char *pformat,
               ...);

    /**
     * @brief Sets a callable trace hook.
     *
     * Function to set a callable hook and m_bDefaultRender.
     * If m_bDefaultRender is set to true, traces will be printed
     * to the default renderer (cout) as well as the trace hook.
     *
     * @param[in] pfnTraceHook          Function to send traces to.
     * @param[in] bCallDefaultRenderer  Setter for m_bDefaultRender.
     */
    virtual void SetHook(TraceFuncPtr traceHook,
                 bool const bCallDefaultRenderer) override;


    /**
     * @brief Sets the log level.
     *
     * Function to set the level of logging.
     * All traces with a level greater than or equal severity level
     * will be rendered at runtime. Other traces will be ignored.
     * Also sets the log level of CDI logging.
     *
     * @param[in] eLevel          Logging level to set.
     */
    virtual void SetLevel(TraceLevel const eLevel) override;

    /**
     * @brief Disables line info.
     *
     * Function to disable the __FUNCTION__ : __LINE__: prefix.
     *
     */
    virtual void DisableLineInfo(void) override;

    /**
     * @brief Setting to print to the default renderer (cout).
     */
    bool m_bDefaultRender = true;

    /**
     * @brief Setting to append function/line prefix to the traces.
     */
    bool m_appendLineInfo = true;

    /**
     * @brief Function to send traces to.
     */
    TraceFuncPtr m_pfnTraceHook = nullptr;

    /**
     * @brief Maximum trace string length.
     */
    static const std::uint32_t m_uMaxLength = 8192U;

    /**
     * @brief Maximum prefix string length.
     */
    static const std::uint32_t m_uMaxLineInfo = 128U;

    /**
     * @brief Internal storage to construct strings.
     */
    char m_message[m_uMaxLength] {};

    /**
     * @brief Loglevel.
     */
    TraceLevel m_level = LevelError;
};

} // namespace nvsipl

#endif // NV_IS_SAFETY

#endif // NVMTRACE_H

