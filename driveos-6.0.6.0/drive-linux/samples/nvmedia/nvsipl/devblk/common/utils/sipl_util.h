/*
   * Copyright (c) 2020-2022 NVIDIA Corporation.  All rights reserved.
   *
   * NVIDIA Corporation and its licensors retain all intellectual property
   * and proprietary rights in and to this software and related documentation
   * and any modifications thereto.  Any use, reproduction, disclosure or
   * distribution of this software and related documentation without an express
   * license agreement from NVIDIA Corporation is strictly prohibited.
   */

/* SIPL Utility Interface */

#ifndef SIPL_UTIL_H
#define SIPL_UTIL_H

#include "nvmedia_core.h"
#include "sipl_error.h"
#include <float.h>
#include <string.h>
#include <stdbool.h>
#include "nverror.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert a float to uint16_t
 *
 * Check if the float is within [0.0, 0xFFFFU] range. If the float is within the range, it casts
 * the \p f to the \p u16 output and returns \p instatus if \p retainStatus is true, else it returns
 * NVMEDIA_STATUS_OK. If the float is not within the range, it returns NVMEDIA_STATUS_BAD_PARAMETER.
 *
 * @param[in]  f             float_t parameter to convert to uint16_t.
 *                           See description for valid range.
 * @param[out] u16           pointer location to store uint16_t conversion result. Set to 0U if
 *                           float was out of valid range.
 *                           valid range [non-NULL]
 * @param[in]  instatus      client set default status to be returned if conversion is successful
 *                           and \p retainStatus is set to true
 * @param[in]  retainStatus  set to true to override the default status returned on successful
 *                           conversion.
 *
 * @retval    NVMEDIA_STATUS_OK             on successful conversion with \p retainStatus set to
 *                                          false
 * @retval    instatus                      on successful conversion with \p retainStatus set to
 *                                          true
 * @retval    NVMEDIA_STATUS_BAD_PARAMETER  if \p f is out of range
 *
 * @note Client must ensure that \p u16 is non-NULL as the unit does not check.
 */
static inline NvMediaStatus floatToUint16(const float_t f, uint16_t *u16, const NvMediaStatus instatus, bool retainStatus)
{
    NvMediaStatus status;

    if (retainStatus == true) {
        status = instatus;
    } else {
        status = NVMEDIA_STATUS_OK;
    }

    if((f <= (float_t)0xFFFFU) && (f >= 0.0F)) {
        *u16 = (uint16_t)f;
    } else {
        *u16 = 0U;
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("floatToUint16: Bad input parameter");
    }

    return status;
}

/**
 * Copies the specified number of bytes from memory area src to memory area dest if both src and
 * dest are non-NULL.
 *
 * @param[in]  dest      To where the content shall be copied.
 *                       valid range [NULL or non-NULL]
 * @param[in]  src       From where the content shall be copied.
 *                       valid range [NULL or non-NULL]
 * @param[in]  bytes     Number of bytes to be copied.
 */
static inline void fusa_memcpy(void * const dest, void const * const src, size_t const bytes)
{
    if ((src != NULL) && (dest != NULL))
    {
        (void) memmove(dest, src, bytes);
    }
}

/*
 * Copy up to dest_array_sz_bytes from memory area src_str
 * to memory area dest, including terminating NULL char
 *
 * @param[in] dest_array
 *          Char array, into which the src_str needs to be copied,
 * including terminating NULL char
 *
 * @param[in] src_str
 *          Source string, from where the content needs to be copied
 *
 * @param[in] dest_array_sz_bytes
 *          Number of bytes in the dest_array
 *
 * @return NVMEDIA_STATUS_OK
 *          Successfully copied
 *
 * @return NVMEDIA_STATUS_OUT_OF_MEMORY
 *          Failure occurred due to out of memory
 *
 * @return NVMEDIA_STATUS_BAD_PARAMETER
 *          Failure occurred due to invalid input
 */

static inline NvMediaStatus fusa_strncpy
(
    char * const dest_array,
    char const * const src_str,
    size_t const dest_array_sz_bytes
)
{
    if ((dest_array != NULL) && (src_str != NULL)) {
        const size_t lenSrc = strnlen(src_str, dest_array_sz_bytes) + 1U;
        if (dest_array_sz_bytes >= lenSrc) {
            (void) memmove(dest_array, src_str, lenSrc);
            return NVMEDIA_STATUS_OK;
        } else {
            return NVMEDIA_STATUS_OUT_OF_MEMORY;
        }
    } else {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
}

/**
 *  @brief Convert NvMediaBool to bool
 *
 *  Eliminate the use of NVMEDIA_TRUE/NVMEDIA_FALSE
 *  Usage - Replace all the NvMediaBool comparison code with the following:
 *
 *  NvMediaBool value;
 *  value == NVMEDIA_TRUE    =>   toBool(value)
 *  value == NVMEDIA_FALSE   =>   !(toBool(value))
 *
 *  @param[in] value NvMediaBool variable for comparison, valid inputs are NVMEDIA_TRUE and
 *                   NVMEDIA_FALSE.
 *
 *  @retval true  if value is NVMEDIA_TRUE
 *  @retval false if value is NVMEDIA_FALSE
 */
static inline bool toBool(const NvMediaBool value)
{
    return (value != (NvMediaBool)(0U));
}

/**
 * @brief Checks if two float values are within an epsilon value
 *
 * @param[in] a float_t variable used for comparision
 * @param[in] b float_t variable used for comparision
 * @param[in] epsilonVal float_t variable used for calculating rounding error
 *
 * @retval true if a and b are within epsilonVal
 * @retval false if a and b do not test within epsilonVal
 *
 * @note Client must ensure that NaN is not used as input for any parameter
 */
static inline bool checkFloatEquality(float_t const a, float_t const b, float_t const epsilonVal)
{
    return (a >= b) ? ((b + epsilonVal) >= a) : ((a + epsilonVal) >= b);
}

/**
 * Checks if two double values are within an epsilon value
 *
 * @param[in] a double_t variable used for comparision
 * @param[in] b double_t variable used for comparision
 * @param[in] epsilonVal double_t variable used for calculating rounding error
 *
 * @retval true if a and b are within epsilonVal
 * @retval false if a and b do not test within epsilonVal
 *
 * @note Client must ensure that NaN is not used as input for any parameter
 */
static inline bool checkDoubleEquality(double_t const a, double_t const b, double_t const epsilonVal)
{
    return (a >= b) ? ((b + epsilonVal) >= a) : ((a + epsilonVal) >= b);
}

/**
 *  @brief Convert bool to NvMediaBool
 *
 *  Eliminate the use of NVMEDIA_TRUE/NVMEDIA_FALSE
 *  Usage - Replace all the NvMediaBool assignment code with the following:
 *
 *  = NVMEDIA_TRUE    =>   = toNvMediaBool(true)
 *  = NVMEDIA_FALSE   =>   = toNvMediaBool(false)
 *
 *  @param[in] value bool variable to be converted to NvMediaBool, valid inputs are true and false
 *
 *  @retval NVMEDIA_TRUE if value is true
 *  @retval NVMEDIA_FALSE if value is false
 */
static inline NvMediaBool toNvMediaBool(const bool value)
{
    return (value ? (NvMediaBool)(1U) : (NvMediaBool)(0U));
}

/**
 *  @brief Convert double to float
 *
 * Eliminate CERT FLT34-C violation for narrowing conversions
 *
 * @param[in] d double variable to be converted to float
 *
 * @retval float_t value after conversion
 */
static inline float_t toFloatFromDouble(double_t const d)
{
    if(isgreater(fabs(d), FLT_MAX) != 0) {
       SIPL_LOG_ERR_STR("float range check for double to float conversion");
    }
    return (float_t)d;
}

/**
 *  @brief left shift the given value by number of bits
 *
 *  Eliminate CERT INT34-C violation
 *
 *  @param[in] value uint32_t variable to be left-shifted
 *
 *  @param[in] bits uint32_t number of bits to be shifted
 *
 *  @retval uint32_t value after shifting, 0U if \p bits is 32U or greater
 *
 */
static inline uint32_t leftBitsShift(uint32_t const value, uint32_t const bits)
{
    return (bits < 32U) ? (value << bits) : 0U;
}

/**
 *  @brief addition of two unsigned integers without MISRA C++ violation
 *
 *  Adding two uint32_t values together and clamp to UINT32_MAX
 *
 *  @param[in] a  uint32_t value to be added
 *  @param[in] b  uint32_t value to be added
 *
 *  @retval the uint32_t saturated addition clamping to UINT32_MAX
 */
static inline uint32_t saturatingAddUint32(uint32_t const a, uint32_t const b)
{
    uint32_t ret;
    if (a > (UINT32_MAX - b))
    {
         ret = UINT32_MAX;
    }
    else
    {
         ret = a + b;
    }

    return ret;
}

/**
 *  @brief addition of two unsigned integers without MISRA C++ violation
 *
 *  Adding two uint64_t values together and clamp to UINT64_MAX
 *
 *  @param[in] a  uint64_t value to be added
 *  @param[in] b  uint64_t value to be added
 *
 *  @retval the uint64_t saturated addition clamping to UINT64_MAX
 */
static inline uint64_t saturatingAddUint64(uint64_t const a, uint64_t const b)
{
    uint64_t ret;
    if (a > (UINT64_MAX - b))
    {
         ret = UINT64_MAX;
    }
    else
    {
         ret = a + b;
    }

    return ret;
}

/**
 *  @brief addition of two unsigned integers without MISRA C++ violation
 *
 *  Adding two uint16_t values together and clamp to UINT16_MAX
 *
 *  @param[in] a uint16_t value to be added
 *  @param[in] b uint16_t value to be added
 *
 *  @retval the uint16_t sum of the saturated addition
 */
static inline uint16_t saturatingAddUint16(uint16_t const a, uint16_t const b)
{
    uint16_t ret;
    if (a > (UINT16_MAX - b))
    {
         ret = UINT16_MAX;
    }
    else
    {
         ret = a + b;
    }

    return ret;
}

/**
 *  @brief subtract two unsigned integers without CERT C violation
 *
 *  Subtract two uint32_t values and clamp to 0U
 *
 *  @param[in] a uint32_t value to be subtracted from
 *  @param[in] b uint32_t value to subtract
 *
 *  @retval the uint32_t result of the saturated subtraction
 */
static inline uint32_t saturatingSubtractUint32(uint32_t const a, uint32_t const b)
{
    uint32_t ret;

    if(a < b) {
        ret = 0U;
    } else {
        ret = (a - b);
    }

    return ret;
}

/**
 *  @brief subtract three unsigned integers without CERT C violation
 *
 *  Subtract three uint32_t values and clamp to 0U
 *
 *  @param[in] a uint32_t value to be subtracted from
 *  @param[in] b uint32_t value to subtract
 *  @param[in] c uint32_t value to subtract
 *
 *  @retval the uint32_t result of the saturated subtraction
 */
static inline uint32_t saturatingSubtractUint32_3(uint32_t const a, uint32_t const b, uint32_t const c)
{
    uint32_t const ret = saturatingSubtractUint32(a, b);

    return saturatingSubtractUint32(ret, c);
}

/**
 *  @brief multiplication of two unsigned integers without CERT C violation
 *
 *  Multiplying two uint32_t values together and clamp to UINT32_MAX
 *
 *  @param[in] a uint32_t value to be multiplied
 *  @param[in] b uint32_t value to be multiplied
 *
 *  @retval the uint32_t product of the saturated multiplication
 */
static inline uint32_t saturatingMultUint32(uint32_t const a, uint32_t const b)
{
    uint32_t ret = 0U;

    if ((b > 0U) && (a > (UINT32_MAX/b))) {
        ret = UINT32_MAX;
    } else {
        ret = a * b;
    }

    return ret;
}

/**
 *  @brief multiplication of two integers without CERT C violation
 *
 *  Multiplying two int32_t values together and clamp to INT32_MAX or INT32_MIN
 *
 *  @param[in] a int32_t value to be multiplied
 *  @param[in] b int32_t value to be multiplied
 *
 *  @retval the int32_t product of the saturated multiplication
 */
static inline int32_t saturatingMultInt32(int32_t const a, int32_t const b)
{

    int32_t ret;
    /* This code purposely avoids (INT32_MIN/-1) which is undefined behavior,
     * because that value would not be representable as an int32_t type
     * (INT32_MAX+1).
     *
     * This code also relies heavily on integer division rounding toward zero.
     * This is guaranteed by the language.
     */
    if (a >= 0) {
        if ((b > 0) && (a > (INT32_MAX/b))) {
            ret = INT32_MAX;
        } else if ((b < -1) && (a > (INT32_MIN/b))) {
            ret = INT32_MIN;
        } else {
            ret = a * b;
        }
    } else {  // a < 0
        if ((b > 0) && (a < (INT32_MIN/b))) {
            ret = INT32_MIN;
        } else if ((b < 0) && (a < (INT32_MAX/b))) {
            ret = INT32_MAX;
        } else {
            ret = a * b;
        }
    }
    return ret;
}

/**
 *  @brief multiplication of three unsigned integers without CERT C violation
 *
 *  Multiplying three uint32_t values and clamp to UINT32_MAX
 *
 *  @param[in] a uint32_t value to be multiplied
 *  @param[in] b uint32_t value to be multiplied
 *  @param[in] c uint32_t value to be multiplied
 *
 *  @retval the uint32_t product of the saturated multiplication
 */
static inline uint32_t saturatingMultUint32_3(uint32_t const a, uint32_t const b, uint32_t const c)
{
    uint32_t const ret = saturatingMultUint32(a, b);

    return saturatingMultUint32(ret, c);
}

/**
 *  @brief Convert uint32_t to float_t
 *
 *  Eliminate CERT FLP36-C violation
 *
 *  @param[in] x uint32_t value to be converted to float_t type
 *
 *  @retval float_t LSBs discarded if x >= ((uint32_t)(1U) << FLT_MANT_DIG)
 *
 */
static inline float_t toFloatFromUint32(uint32_t const x)
{
    if (x >= ((uint32_t)(1U) << FLT_MANT_DIG)) {
        SIPL_LOG_ERR_STR("LSBs of uint32_t type are being discarded");
    }
    return (float_t)(x);
}

/**
 *  @brief Convert uint64_t to double_t
 *
 *  Eliminate CERT FLP36-C violation
 *
 *  @param[in] x uint64_t value to be converted to double_t type
 *
 *  @retval double_t LSBs discarded if x >= ((uint64_t)(1UL) << DBL_MANT_DIG)
 *
 */
static inline double_t toDoubleFromUint64(uint64_t const x)
{
    if (x >= ((uint64_t)(1UL) << DBL_MANT_DIG)) {
        SIPL_LOG_ERR_STR("LSBs of uint64_t type are being discarded");
    }
    return (double_t)(x);
}

/**
 *  @brief Convert float_t to uint32_t
 *
 *  Eliminate CERT FLP34-C and FLP36-C violation
 *
 *  @param[in] x float_t value to be converted to uint32_t type
 *
 *  @retval input parameter x is cast to uint32_t if x <= 4294967040.0F && 0.0F <= x, else
 *          return 0xFFFFFFFFU if x > 4294967040.0F is true and return 0U if false
 *
 */
static inline uint32_t toUint32FromFloat(float_t const x)
{
    uint32_t ret = 0U;

/**
 *  max_float_u32 is the largest float value that still fits
 *  into a uint32_t type.
 *  4294967040.0F = 0xffffff00U
 *
 *  The next value is:
 *
 *  4294967296.0F = 0x100000000U
 *
 *  We are very careful to avoid undefined behavior.
 *  Comparing to NaN results in false.
 *  So an input of NaN returns 0U.
 */
    float_t const max_float_u32 = 4294967040.0F;
    if ((0.0F <= x) && (x <= max_float_u32)) {
        ret = (uint32_t)(x);
    } else {
        ret = (x > max_float_u32) ? 0xFFFFFFFFU : 0U;
    }
    return ret;
}

/**
 *  @brief Convert uint32_t to int32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint32_t value to be converted to int32_t type
 *
 *  @retval input parameter x cast to int32_t if x <= INT32_MAX else return INT32_MAX
 *
 */
static inline int32_t toInt32FromUint32(uint32_t const x)
{
    int32_t ret = 0;

    if (x <= (uint32_t)INT32_MAX) {
        ret = (int32_t)(x);
    } else {
        ret = INT32_MAX;
    }
    return ret;
}

/**
 *  @brief Convert int64_t to int32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x int64_t value to be converted to int32_t type
 *
 *  @retval input parameter x cast to int32_t if x <= INT32_MAX and x >= INT32_MIN
 *          else if x > INT32_MAX, return INT32_MAX
 *          else (which implies x < INT32_MIN), return INT32_MIN
 */
static inline int32_t toInt32FromInt64(int64_t const x)
{
    int32_t ret = 0;
    if ((x <= (int64_t)INT32_MAX) && (x >= (int64_t)INT32_MIN))
    {
        ret = (int32_t)(x);
    }
    else if (x > (int64_t)INT32_MAX)
    {
        ret = INT32_MAX;
    }
    else
    {
        ret = INT32_MIN;
    }
    return ret;
}

/**
 *  @brief Convert int64_t to uint32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x int64_t value to be converted to uint32_t type
 *
 *  @retval input parameter x cast to uint32_t if x <= UINT32_MAX and x >= 0
 *          else if x > UINT32_MAX return UINT32_MAX
 *          else (which implies  x < 0) return 0U
 *
 */
static inline uint32_t toUint32FromInt64(int64_t const x)
{
    uint32_t ret = 0U;
    if ((x <= (int64_t)UINT32_MAX) && (x >= 0))
    {
        ret = (uint32_t)(x);
    }
    else if (x > (int64_t)UINT32_MAX)
    {
        ret = UINT32_MAX;
    }
    else
    {
        ret = 0U;
    }
    return ret;
}

/**
 *  @brief Convert uint32_t to uint16_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint32_t value to be converted to uint16_t type
 *
 *  @retval input parameter x cast to uint16_t if x <= UINT16_MAX, else return UINT16_MAX
 *
 */
static inline uint16_t toUint16FromUint32(uint32_t const x)
{
    uint16_t ret = 0U;

    if (x <= (uint32_t)UINT16_MAX) {
        ret = (uint16_t)(x);
    } else {
        ret = UINT16_MAX;
    }
    return ret;
}

/**
 *  @brief Convert int32_t to uint16_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x int32_t value to be converted to uint16_t type
 *
 *  @retval input parameter x cast to uint16_t if x <= UINT16_MAX and x >= 0
 *          else if x < 0 return 0U
 *          else (which implies x > UINT16_MAX) return UINT16_MAX
 *
 */
static inline uint16_t toUint16FromInt32(int32_t const x)
{
    uint16_t ret = 0;

    if (x > UINT16_MAX) {
        ret = (uint16_t)UINT16_MAX;
    } else if (x < 0) {
        ret = (uint16_t)0U;
    } else {
        ret = (uint16_t)(x);
    }
    return ret;
}

/**
 *  @brief Convert int32_t to uint32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x int32_t value to be converted to uint32_t type
 *
 *  @retval input parameter x cast to uint32_t if x >= 0, else return 0U
 *
 */
static inline uint32_t toUint32FromInt32(int32_t const x)
{
    uint32_t ret = 0U;

    if (x >= 0) {
        ret = (uint32_t)(x);
    }

    return ret;
}

/**
 *  @brief Convert uint64_t to uint32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint64_t value to be converted to uint32_t type
 *
 *  @retval input parameter x cast to uint32_t if x <= UINT32_MAX, else return UINT32_MAX
 *
 */
static inline uint32_t toUint32FromUint64(uint64_t const x)
{
    uint32_t ret = 0U;

    if (x > (uint64_t)UINT32_MAX) {
        ret = (uint32_t)UINT32_MAX;
    } else {
        ret = (uint32_t)(x);
    }
    return ret;
}

/**
 *  @brief Convert uint64_t to uint32_t with status
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint64_t value to be converted
 *  @param[out] y uint32_t converted uint32_t value
 *
 *  @retval input parameter x cast to uint32_t if x <= UINT32_MAX and return NvSuccess,
 *  else return NvError_OverFlow
 *
 */
static inline NvError getUint32FromUint64(uint64_t const x, uint32_t *y)
{
    if (x > (uint64_t)UINT32_MAX) {
        SIPL_LOG_ERR_STR("Truncating value from uint64_t to uint32_t");
        return NvError_OverFlow;
    } else {
        *y = (uint32_t)(x);
    }
    return NvSuccess;
}

/**
 *  @brief Convert uint32_t to uint8_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint32_t value to be converted to uint8_t type
 *
 *  @retval input parameter x cast to uint8_t if x <= UINT8_MAX, else return UINT8_MAX
 *
 */
static inline uint8_t toUint8FromUint32(uint32_t const x)
{
    uint8_t ret = 0;

    if (x <= (uint32_t)UINT8_MAX) {
        ret = (uint8_t)(x);
    } else {
        ret = (uint8_t)UINT8_MAX;
    }
    return ret;
}

/**
 *  @brief Convert uint16_t to uint8_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x uint16_t value to be converted to uint8_t type
 *
 *  @retval input parameter x cast to uint8_t if x <= UINT8_MAX, else return UINT8_MAX
 *
 */
static inline uint8_t toUint8FromUint16(uint16_t const x)
{
    uint8_t ret = 0;

    if (x <= (uint16_t)UINT8_MAX) {
        ret = (uint8_t)(x);
    } else {
        ret = (uint8_t)UINT8_MAX;
    }
    return ret;
}

/**
 *  @brief Convert size_t to uint32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x size_t value to be converted to uint32_t type
 *
 *  @retval input parameter x cast to uint32_t if x <= UINT32_MAX, else return UINT32_MAX
 *
 */
static inline uint32_t toUint32FromSize_t(size_t const x)
{
    uint32_t ret = 0U;
    if (x > (size_t)UINT32_MAX) {
        ret = (uint32_t)UINT32_MAX;
    } else {
        ret = (uint32_t)(x);
    }
    return ret;
}

/**
 *  @brief Convert size_t to int32_t
 *
 *  Eliminate CERT INT31-C violation
 *
 *  @param[in] x size_t value to be converted to int32_t type
 *
 *  @retval input parameter x cast to int32_t if x <= INT32_MAX, else return INT32_MAX
 *
 */
static inline int32_t toInt32FromSize_t(size_t const x)
{
    int32_t ret = 0;
    if (x > (size_t)INT32_MAX) {
        ret = (int32_t)INT32_MAX;
    } else {
        ret = (int32_t)(x);
    }
    return ret;
}

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif // SIPL_UTIL_H
