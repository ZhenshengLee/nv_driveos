/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "cLogger.h"
#include <cstring>

using namespace std;

CLogger::CLogger() :
    m_level(LEVEL_ERR),
    m_style(LOG_STYLE_NORMAL),
    m_logFile(stdout)
{

}

CLogger::~CLogger()
{

}

CLogger& CLogger::GetInstance()
{
    static CLogger instance;
    return instance;
}

void CLogger::SetLogLevel(LogLevel level)
{
    m_level = (level > LEVEL_DBG) ? LEVEL_DBG : level;
}

void CLogger::SetLogStyle(LogStyle style)
{
    m_style = (style > LOG_STYLE_FUNCTION_LINE) ? LOG_STYLE_FUNCTION_LINE
                                                : style;
}

void CLogger::SetLogFile(FILE *logFilePtr)
{
    if (logFilePtr) {
         m_logFile = logFilePtr;
    }
}

void CLogger::LogLevelMessageVa(LogLevel level, const char *functionName,
                                       uint32_t lineNumber, const char *format,
                                                                    va_list ap)
{
    char str[256] = {'\0',};

    if (level > m_level) {
        return;
    }

    strcpy(str, "NvMTest: ");
    switch (level) {
        case LEVEL_NONE:
            break;
        case LEVEL_ERR:
            strcat(str, "ERROR: ");
            break;
        case LEVEL_WARN:
            strcat(str, "WARNING: ");
            break;
        case LEVEL_INFO:
            break;
        case LEVEL_DBG:
            // Empty
            break;
    }

    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    if (m_style == LOG_STYLE_NORMAL) {
        if (strlen(str) != 0 && str[strlen(str) - 1] == '\n') {
            strcat(str, "\n");
        }
    } else if (m_style == LOG_STYLE_FUNCTION_LINE) {
        if (strlen(str) != 0 && str[strlen(str) - 1] == '\n') {
            str[strlen(str) - 1] = 0;
        }
        snprintf(str + strlen(str), sizeof(str) - strlen(str), " at %s():%d\n",
                                                     functionName, lineNumber);
    }

    fprintf(m_logFile, "%s", str);
}

void CLogger::LogLevelMessage(LogLevel level, const char *functionName,
                               uint32_t lineNumber, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, format, ap);
    va_end(ap);
}

void CLogger::LogLevelMessage(LogLevel level, std::string functionName,
                               uint32_t lineNumber, std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber,
                                                       format.c_str(), ap);
    va_end(ap);
}

void CLogger::LogMessageVa(const char *format, va_list ap)
{
    char str[128] = {'\0',};
    vsnprintf(str, sizeof(str), format, ap);
    fprintf(m_logFile, "%s", str);
}

void CLogger::LogMessage(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format, ap);
    va_end(ap);
}

void CLogger::LogMessage(std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format.c_str(), ap);
    va_end(ap);
}
