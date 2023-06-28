/*
 * Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software and related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <string.h>
#ifdef NVMEDIA_ANDROID
#define LOG_TAG "nvmedia_common"
#define LOG_NDEBUG 1
#include <utils/Log.h>
#endif
#ifdef NVMEDIA_QNX
#include <sys/slog.h>
#endif

#include "log_utils.h"

#ifdef NVMEDIA_QNX
#define NV_SLOGCODE 0xAAAA
#endif
#define MAX_STATS_LEN 500

#define LOG_BUFFER_BYTES 1024

static enum LogLevel msg_level = LEVEL_ERR;
static enum LogStyle msg_style = LOG_STYLE_NORMAL;
static FILE *msg_file = NULL;

void SetLogLevel(enum LogLevel level)
{
   if (level > LEVEL_DBG)
     return;

   msg_level = level;
}

void SetLogStyle(enum LogStyle style)
{
   if (style > LOG_STYLE_FUNCTION_LINE)
     return;

   msg_style = style;
}

void SetLogFile(FILE *logFileHandle)
{
    if(!logFileHandle)
        return;

    msg_file = logFileHandle;
}

void LogLevelMessage(enum LogLevel level, const char *functionName,
                     int lineNumber, const char *format, ...)
{
    va_list ap;
    char str[LOG_BUFFER_BYTES] = {'\0',};
    FILE *logFile = msg_file ? msg_file : stdout;

    if (level > msg_level)
        return;

#ifndef NVMEDIA_ANDROID
/** In the case of Android ADB log, if LOG_TAG is defined,
 * before 'Log.h' is included in source file,
 * LOG_TAG is automatically concatenated at the beginning of log message,
 * so, we don't copy 'nvmedia: ' into 'str'.
 */
    strcpy(str, "nvmedia: ");

/** As LOG_TAG is concatednated, log level is also automatically concatenated,
 * by calling different ADB log function such as ALOGE(for eror log message),
 * ALOGW(for warning log message).
 */
    switch (level) {
        case LEVEL_ERR:
            strcat(str, "ERROR: ");
            break;
        case LEVEL_WARN:
            strcat(str, "WARNING: ");
            break;
        case LEVEL_INFO:
        case LEVEL_DBG:
            // Empty
            break;
    }
#endif

    va_start(ap, format);
    vsnprintf(str + strlen(str), sizeof(str) - strlen(str), format, ap);

    if(msg_style == LOG_STYLE_NORMAL) {
        // Add trailing new line char
        if(strlen(str) && str[strlen(str) - 1] != '\n')
            strcat(str, "\n");

    } else if(msg_style == LOG_STYLE_FUNCTION_LINE) {
        // Remove trailing new line char
        if(strlen(str) && str[strlen(str) - 1] == '\n')
            str[strlen(str) - 1] = 0;

        // Add function and line info
        snprintf(str + + strlen(str), sizeof(str) - strlen(str), " at %s():%d\n", functionName, lineNumber);
    }

#ifdef NVMEDIA_ANDROID
    switch (msg_level) {
        case LEVEL_ERR:
            ALOGE("%s", str);
            break;
        case LEVEL_WARN:
            ALOGW("%s", str);
            break;
        case LEVEL_INFO:
            ALOGI("%s", str);
           break;
        case LEVEL_DBG:
            ALOGD("%s", str);
            break;
    }
#else
    fprintf(logFile, "%s", str);
#endif
#ifdef NVMEDIA_QNX
    /* send to system logger */
    slogf(_SLOG_SETCODE(NV_SLOGCODE, 0), _SLOG_ERROR, str);
#endif
    va_end(ap);
}

void LogMessage(const char *format, ...)
{
    va_list ap;
    char str[LOG_BUFFER_BYTES] = {'\0',};
    FILE *logFile = msg_file ? msg_file : stdout;

    va_start(ap, format);
    vsnprintf(str, sizeof(str), format, ap);

#ifdef NVMEDIA_ANDROID
    switch (msg_level) {
        case LEVEL_ERR:
            ALOGE("%s", str);
            break;
        case LEVEL_WARN:
            ALOGW("%s", str);
            break;
        case LEVEL_INFO:
            ALOGI("%s", str);
           break;
        case LEVEL_DBG:
            ALOGD("%s", str);
            break;
    }
#else
    fprintf(logFile, "%s", str);
#endif
#ifdef NVMEDIA_QNX
    /* send to system logger */
    slogf(_SLOG_SETCODE(NV_SLOGCODE, 0), _SLOG_ERROR, str);
#endif
    va_end(ap);
}
