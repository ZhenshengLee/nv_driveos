/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_LOG_UTILS_H_
#define _NVMEDIA_TEST_LOG_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include "nvm_cov_whitelist.h"

enum LogLevel {
    LEVEL_ERR  = 0,
    LEVEL_WARN = 1,
    LEVEL_INFO = 2,
    LEVEL_DBG  = 3,
};

enum LogStyle {
    LOG_STYLE_NORMAL = 0,
    LOG_STYLE_FUNCTION_LINE
};

#define LINE_INFO       __FUNCTION__, __LINE__
#define LOG_DBG(...)    do { NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule,1_2), "Advisory") LogLevelMessage(LEVEL_DBG, LINE_INFO, __VA_ARGS__); } while (1 == 0)
#define LOG_INFO(...)   do { NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule,1_2), "Advisory") LogLevelMessage(LEVEL_INFO, LINE_INFO, __VA_ARGS__); } while (1 == 0)
#define LOG_WARN(...)   do { NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule,1_2), "Advisory") LogLevelMessage(LEVEL_WARN, LINE_INFO, __VA_ARGS__); } while (1 == 0)
#define LOG_ERR(...)    do { NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule,1_2), "Advisory") LogLevelMessage(LEVEL_ERR, LINE_INFO, __VA_ARGS__); } while (1 == 0)
#define LOG_MSG(...)    do { NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule,1_2), "Advisory") LogMessage(__VA_ARGS__); } while (1 == 0)

//  SetLogLevel
//
//    SetLogLevel()  Set logging level
//
//  Arguments:
//
//   level
//      (in) Logging level

void
SetLogLevel(
    enum LogLevel level);

//  SetLogStyle
//
//    SetLogStyle()  Set logging print slyle
//
//  Arguments:
//
//   level
//      (in) Logging style

void
SetLogStyle(
    enum LogStyle style);

//  SetLogFile
//
//    SetLogFile()  Set logging file handle
//
//  Arguments:
//
//   level
//      (in) Logging file handle

void
SetLogFile(
    FILE *logFileHandle);

//  LogLevelMessage
//
//    LogLevelMessage()  Print message if logging level is higher than message level
//
//  Arguments:
//
//   LogLevel
//      (in) Message level
//
//   format
//      (in) Message format
//
//   ...
//      (in) Parameters list

void
LogLevelMessage(
    enum LogLevel level,
    const char *functionName,
    int lineNumber,
    const char *format,
    ...);

//  LogMessage
//
//    LogMessage()  Print message
//
//  Arguments:
//
//   format
//      (in) Message format
//
//   ...
//      (in) Parameters list

void
LogMessage(
    const char *format,
    ...);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_LOG_UTILS_H_ */
