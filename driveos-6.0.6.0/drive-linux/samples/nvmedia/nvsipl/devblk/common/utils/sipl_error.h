/* Copyright (c) 2020-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef SIPL_ERROR_H
#define SIPL_ERROR_H

#include "nvos_s3_tegra_log.h"
#include "NvSIPLCapStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file
 *
 * @brief This file contains the variables, functions and macros used for
 * safety-compliant common error logging in SIPL modules.
 * Every error log begins with a line showing the file path under sipl and
 * the line number where the error log originates. It is then followed by
 * one or more lines that show the error strings coded by the user and any
 * associated numbers of various integer types. The actual logging is done
 * using appropriate NvOsDebugPrintStr* functions.
 */

/**
 * @brief char_t is to clarify the coding intent,
 * particularly as required by MISRA C and C++.
 */
typedef char char_t;

static const uint16_t SIPL_LOG_CODE = (uint16_t)NVOS_SLOG_CODE_CAMERA;
static const uint8_t  SIPL_SEVERITY_ERROR = (uint8_t)NVOS_LOG_SEVERITY_ERROR;

/*
 *  @brief Get a non-null string from the input pointer.
 *
 *  @param[in] msg Pointer that represents the message string
 *                 Valid range [NULL or non-NULL]
 *
 *  @retval The input string if the pointer parameter is non-null, else the null string
 */
static inline const char_t* getValidStr(const char_t *const msg)
{
    return (msg == NULL) ? "" : msg;
}

/**
 *  @brief Search fileNameStr for the SIPL directory path substring. If found,
 *  return everything after the substring, otherwise return fileNameStr.
 *
 *  @param[in] fileNameStr  A null-terminated C-string denoting the full SIPL file path
 *
 *  @retval The file name string that remains after the SIPL prefix is successfully removed,
 *          else return the original input string as a non-null pointer
 */
static inline const char_t* removePath(const char_t fileNameStr[])
{
    const char_t *const safe_fname = getValidStr(fileNameStr);

    const char_t *ret = safe_fname;
    const char_t *searchStr = safe_fname;
    const char_t eos = '\0';
    const size_t maxLength = 1024U; // max. characters to search for the substring
    size_t pos = 0U;

    while ((searchStr[0] != eos) && (pos < maxLength))
    {
        const char_t path[] = "camera/fusa/sipl/";
        size_t cnt = 0U;
        /*
         *  If (searchStr[cnt] == path[cnt]) and (path[cnt] != 0),
         *  then it is safe to conclude that
         *  searchStr[cnt] != 0.
         */
        while ((path[cnt] != eos) && (searchStr[cnt] == path[cnt]))
        {
            cnt++;
        }
        if (path[cnt] == eos)
        {
            ret = &searchStr[cnt];
            break;
        }
        pos++;
        searchStr = &safe_fname[pos];
    }
    return ret;
}

#if NV_IS_SAFETY
/**
 * @defgroup LOG_GROUP Group of logging levels.
 *
 * @brief Different logging levels which are set to no-operation on safety build.
 * @{
 */
/** Quick-log a message at debugging level*/
#define LOG_DEBUG(...)

/** Quick-log a message at debugging level*/
#define LOG_DBG(...)

/** Quick-log a message at info level*/
#define LOG_INFO(...)

/** Quick-log a message at warning level*/
#define LOG_WARN(...)

/** Quick-log a message*/
#define LOG_MSG(...)
/** @} */
#endif //NV_IS_SAFETY

/**
 * @brief Logs a string passed by the caller.
 *
 * @param[in] msg Message to be logged.
 */
#define SIPL_LOG_ERR_STR(msg) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg)); \
}

/**
 * @brief Logs two strings passed by the caller.
 *
 * @param[in] str1 First string to be logged.
 * @param[in] str2 Second string to be logged.
 */
#define SIPL_LOG_ERR_2STR(str1, str2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(str1)); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(str2)); \
}

/**
 * @brief Logs three strings passed by the caller.
 *
 * @param[in] str1 First string to be logged.
 * @param[in] str2 Second string to be logged.
 * @param[in] str3 Third string to be logged.
 */
#define SIPL_LOG_ERR_3STR(str1, str2, str3) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(str1)); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(str2)); \
    NvOsDebugPrintStr(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(str3)); \
}

/**
 * @brief Logs a string and a signed integer value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_INT(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and an unsigned integer value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Unsigned integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_UINT(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and an unsigned integer value passed by the caller in the
 * hexadecimal format.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Unsigned integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_HEX_UINT(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrHexUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and signed long value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Long integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_SLONG(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrSLong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and an unsigned long value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Unsigned long integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_ULONG(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and two signed integer values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Signed integer value to be logged.
 * @param[in] val2 Signed integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_2INT(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2Int(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \
}

/**
 * @brief Logs a string and two signed long values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Signed long value to be logged.
 * @param[in] val2 Signed long value to be logged.
 */
#define SIPL_LOG_ERR_STR_2SLONG(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2SLong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \
}

/**
 * @brief Logs a string and two unsigned integer values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Unsigned integer value to be logged.
 * @param[in] val2 Unsigned integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_2UINT(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2UInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \
}

/**
 * @brief Logs a string and two unsigned long values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Unsigned long value to be logged.
 * @param[in] val2 Unsigned long value to be logged.
 */
#define SIPL_LOG_ERR_STR_2ULONG(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2ULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \
}

/**
 * @brief Logs a string and a float value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Float value to be logged.
*/
#define SIPL_LOG_ERR_STR_FLOAT(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrFloat(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and two float values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 First Float value to be logged.
 * @param[in] val2 Second Float value to be logged.
 */
#define SIPL_LOG_ERR_STR_2FLOAT(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2Float(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \
}

/**
 * @brief Logs a string and a double value passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val Double value to be logged.
 */
#define SIPL_LOG_ERR_STR_DOUBLE(msg, val) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrDouble(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val)); \
}

/**
 * @brief Logs a string and two double values passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 First Double value to be logged.
 * @param[in] val2 Second Double value to be logged.
 */
#define SIPL_LOG_ERR_STR_2DOUBLE(msg, val1, val2) \
{ \
    NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
    NvOsDebugPrintStrWith2Double(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1), (val2)); \

/**
 * @brief Logs a string with unsigned integer and hexadecimal passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Unsigned Integer value to be logged.
 * @param[in] val2 Hexadecimal value to be logged.
 */
#define SIPL_LOG_ERR_STR_UINT_AND_HEX(msg, val1, val2) \
{ \
      NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
      NvOsDebugPrintStrUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
      NvOsDebugPrintStrHexUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val2)); \
}

/**
 * @brief Logs a string with two hexadecimal passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 First Hexadecimal value to be logged.
 * @param[in] val2 Second Hexadecimal value to be logged.
 */
#define SIPL_LOG_ERR_STR_2HEX(msg, val1, val2) \
{ \
      NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
      NvOsDebugPrintStrHexUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
      NvOsDebugPrintStrHexUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val2)); \
}

/**
 * @brief Logs a string with hexadecimal and unsigned integer passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Hexadecimal value to be logged.
 * @param[in] val2 Unsigned Integer value to be logged.
 */
#define SIPL_LOG_ERR_STR_HEX_AND_UINT(msg, val1, val2) \
{ \
      NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
      NvOsDebugPrintStrHexUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
      NvOsDebugPrintStrUInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
}

/**
 * @brief Logs a string with hexadecimal unsigned long passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Hexadecimal unsigned long value to be logged.
 * @param[in] val2 Hexadecimal unsigned long value to be logged.
 */
#define SIPL_LOG_ERR_STR_HEX_2ULONG(msg, val1, val2) \
{ \
      NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
      NvOsDebugPrintStrHexULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
      NvOsDebugPrintStrHexULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val2)); \
}

/**
 * @brief Logs a string with hexadecimal unsigned long passed by the caller.
 *
 * @param[in] msg Message to be logged.
 * @param[in] val1 Hexadecimal unsigned long value to be logged.
 * @param[in] val2 Hexadecimal unsigned long value to be logged.
 * @param[in] val3 Hexadecimal unsigned long value to be logged.
 */
#define SIPL_LOG_ERR_STR_HEX_3ULONG(msg, val1, val2, val3) \
{ \
      NvOsDebugPrintStrInt(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, removePath(__FILE__), __LINE__); \
      NvOsDebugPrintStrHexULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val1)); \
      NvOsDebugPrintStrHexULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val2)); \
      NvOsDebugPrintStrHexULong(SIPL_LOG_CODE, SIPL_SEVERITY_ERROR, getValidStr(msg), (val3)); \
}

/* TODO : Temporary Placeholder */
/*VCAST_DONT_INSTRUMENT_START*/
static inline void criticalFailureDetected(void)
{
    SIPL_LOG_ERR_STR("critial failure detected!");
}
/*VCAST_DONT_INSTRUMENT_END*/

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif // SIPL_ERROR_H
