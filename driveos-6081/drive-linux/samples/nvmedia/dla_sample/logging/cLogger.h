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

#ifndef _CLOGGER_H_
#define _CLOGGER_H_

#include <stdint.h>
#include <cstdarg>
#include <cstdio>
#include <string>

//! Set log level
#define SET_LOG_LEVEL(level) \
    CLogger::GetInstance().SetLogLevel(level)

#define LINE_INFO __FUNCTION__, __LINE__

//! Quick-log a message at debugging level
#define LOG_DBG(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...) \
    CLogger::GetInstance().LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__)

//! Quick-log a message at preset level
#define LOG_MSG(...) \
    CLogger::GetInstance().LogMessage(__VA_ARGS__)

#define LEVEL_NONE CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG CLogger::LogLevel::LEVEL_DEBUG

//! \brief Logger utility class
//! This is a singleton class - at most one instance can exist at all times.
class CLogger{
public:
    //! enum describing the different levels for logging
    enum LogLevel {
        /** no log */
        LEVEL_NO_LOG = 0,
        /** error level */
        LEVEL_ERROR,
        /** warning level */
        LEVEL_WARNING,
        /** info level */
        LEVEL_INFORMATION,
        /** debug level */
        LEVEL_DEBUG
    };

    //! enum describing the different styles for logging
    enum LogStyle {
        LOG_STYLE_NORMAL        = 0,
        LOG_STYLE_FUNCTION_LINE = 1
    };

    //! Get the logging instance.
    //! \return Reference to the Logger object.
    static CLogger& GetInstance();

    //! Destructor of the Logger class
    virtual ~CLogger();

    //! Set the level for logging.
    //! \param[in] eLevel The logging level.
    void SetLogLevel(LogLevel eLevel);

    //! Set the style for logging.
    //! \param[in] eStyle The logging style.
    void SetLogStyle(LogStyle eStyle);

    //! Set the logging file (C-style).
    //! \param[in] hLogFilePtr Handle of the file to be used for logging.
    void SetLogFile(FILE *hLogFilePtr);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] pszFormat Format string as a cstring.
    void LogLevelMessage(LogLevel eLevel, const char *pszFunctionName,
                         uint32_t sLineNumber, const char *pszFormat, ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] sFormat Format string as a C++ string.
    void LogLevelMessage(LogLevel eLevel, std::string sFunctionName,
                         uint32_t sLineNumber, std::string sFormat, ...);

    //! Log a message (cstring) at preset level.
    //! \param[in] pszFormat Format string as a cstring.
    void LogMessage(const char *pszFormat, ...);

    //! Log a message (C++ string) at preset level.
    //! \param[in] sFormat Format string as a C++ string.
    void LogMessage(std::string sFormat, ...);

private:
    //! Need private constructor because this is a singleton.
    CLogger();
    LogLevel m_level;
    LogStyle m_style;
    FILE *m_logFile;

    void LogLevelMessageVa(LogLevel eLevel, const char *pszFunctionName,
                       uint32_t sLineNumber, const char *pszFormat, va_list ap);
    void LogMessageVa(const char *pszFormat, va_list ap);
}; // CLogger class

#endif // _CLOGGER_H_
