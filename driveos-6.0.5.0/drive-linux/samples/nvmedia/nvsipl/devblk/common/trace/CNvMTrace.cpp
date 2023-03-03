/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "trace/CNvMTrace.hpp"

extern "C" {
    #include "cdi_debug_log.h"
}

#include <pthread.h>

#if NV_IS_SAFETY

nvsipl::INvSIPLDeviceBlockTrace* nvsipl::INvSIPLDeviceBlockTrace::GetInstance(void)
{
    return NULL;
}

// This constructor effectively tells qcc/gcc that this is the file in which we
// should generate the RTTI metadata for INvSIPLDeviceBlockTrace (by virtue of
// it being the first non-virtual implementation of any child class.)
// See the following mailing for details:
//    https://gcc.gnu.org/legacy-ml/gcc-help/2005-04/msg00221.html
nvsipl::DummyTraceImpl::DummyTraceImpl() : INvSIPLDeviceBlockTrace() {
}

#else

namespace nvsipl
{

INvSIPLDeviceBlockTrace* INvSIPLDeviceBlockTrace::GetInstance(void)
{
    static CNvSIPLDeviceBlockTrace sTrace;
    return &sTrace;
}

CNvSIPLDeviceBlockTrace& CNvSIPLDeviceBlockTrace::GetTrace(void)
{
    CNvSIPLDeviceBlockTrace* const instance =
        (CNvSIPLDeviceBlockTrace*)INvSIPLDeviceBlockTrace::GetInstance();
    return *instance;
}

void CNvSIPLDeviceBlockTrace::Trace(TraceLevel const eLevel,
                                    const char* const func,
                                    const char* const chfile,
                                    int32_t const line,
                                    const char *pformat,
                                    ...)
{
    if (eLevel <= m_level) {
        if (pformat == nullptr) {
            return;
        }

        va_list args;
        va_start(args, pformat);
        char* msg = m_message;
        uint32_t msgSize = 0U;

        if (m_appendLineInfo) {
            char thread_name[128] {};
            (void) pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
            int32_t const lineInfoLength = snprintf(m_message, m_uMaxLineInfo, "%s: %s: %d: %s: ", thread_name, chfile, line, func);
            msg += lineInfoLength;
            msgSize += lineInfoLength;
        }

        int const nCharacters = vsnprintf(msg, m_uMaxLength - msgSize, pformat, args);
        if (nCharacters < 0) {
            va_end(args);
            return;
        }
        msgSize += nCharacters;

        // Send to renderer
        if (m_pfnTraceHook != nullptr) {
            m_pfnTraceHook(m_message, msgSize);
        }
        if (m_bDefaultRender) {
            std::cout << m_message;
        }
        va_end(args);
    }
    return;
}

void CNvSIPLDeviceBlockTrace::SetHook(TraceFuncPtr traceHook,
                                      bool const bCallDefaultRenderer)
{
    m_pfnTraceHook = traceHook;
    m_bDefaultRender = bCallDefaultRenderer;
}

void CNvSIPLDeviceBlockTrace::SetLevel(TraceLevel const eLevel)
{
    m_level = eLevel;

    // Set the trace level used by the CDI drivers
    switch(eLevel) {
        case LEVEL_ERR:
            SetCDILogLevel(CDI_LOG_LEVEL_ERR);
            break;
        case LEVEL_WARN:
            SetCDILogLevel(CDI_LOG_LEVEL_WARN);
            break;
        case LEVEL_INFO:
            SetCDILogLevel(CDI_LOG_LEVEL_INFO);
            break;
        case LEVEL_DBG:
            SetCDILogLevel(CDI_LOG_LEVEL_DBG);
            break;
        case LEVEL_NONE:
            break;
    }
}

void CNvSIPLDeviceBlockTrace::DisableLineInfo(void)
{
    m_appendLineInfo = false;
}

} // nvsipl namespace

#endif
