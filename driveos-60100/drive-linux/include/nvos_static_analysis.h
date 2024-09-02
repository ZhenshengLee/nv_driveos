/*
* SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

#ifndef INCLUDED_NVOS_STATIC_ANALYSIS_H
#define INCLUDED_NVOS_STATIC_ANALYSIS_H

/**
 * @file
 *
 * Macros/functions/etc for static analysis of code.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#if defined(__QNX__)
#include <unistd.h>
#include <hw/inout.h>
#define NVOS_EXIT(x) _exit(x)
#endif /* End of __QNX__ */

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * These macros are used for whitelisting coverity violations. The macros are
 * only enabled when a coverity scan is being run.
 */
#ifdef NV_IS_COVERITY
/**
 * NVOS_MISRA - Define a MISRA rule for NVOS_COV_WHITELIST.
 *
 * @param type - This should be Rule or Directive depending on if you're dealing
 *               with a MISRA rule or directive.
 * @param num  - This is the MISRA rule/directive number. Replace hyphens and
 *               periods in the rule/directive number with underscores. Example:
 *               14.2 should be 14_2.
 *
 * This is a convenience macro for defining a MISRA rule for the
 * NVOS_COV_WHITELIST macro.
 *
 * Example 1: For defining MISRA rule 14.2, use NVOS_MISRA(Rule, 14_2).
 * Example 2: For defining MISRA directive 4.7, use NVOS_MISRA(Directive, 4_7).
 */
#define NVOS_MISRA(type, num) MISRA_C_2012_##type##_##num

/**
 * NVOS_CERT - Define a CERT C rule for NVOS_COV_WHITELIST.
 *
 * @param num - This is the CERT C rule number. Replace hyphens and periods in
 *              the rule number with underscores. Example: INT30-C should be
 *              INT30_C.
 *
 * This is a convenience macro for defining a CERT C rule for the
 * NVOS_COV_WHITELIST macro.
 *
 * Example: For defining CERT C rule INT30-C, use NVOS_CERT(INT30_C).
 */
#define NVOS_CERT(num) CERT_##num

/**
 * Helper macro for stringifying the _Pragma() string
 */
#define NVOS_COV_STRING(x) #x

/**
 * NVOS_COV_WHITELIST - Whitelist a coverity violation on the next line.
 *
 * @param type        - This is the whitelisting category. Valid values are
 *                      deviate or false_positive.
 *                      deviate is for an approved rule deviation.
 *                      false_positive is normally used for a bug in coverity
 *                      which causes a false violation to appear in the scan.
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVOS_MISRA() or NVOS_CERT() macro to define
 *                      this field.
 * @param comment_str - This is the comment that you want associated with this
 *                      whitelisting. This should normally be a bug number
 *                      (ex: coverity bug) or JIRA task ID (ex: RFD). Unlike the
 *                      other arguments, this argument must be a quoted string.
 *
 * Use this macro to whitelist a coverity violation in the next line of code.
 *
 * Example 1: Whitelist a MISRA rule 14.2 violation due to a deviation
 * documented in the JIRA TID-123 RFD:
 * NVOS_COV_WHITELIST(deviate, NVOS_MISRA(Rule, 14_2), "JIRA TID-123")
 * <Next line of code with a rule 14.2 violation>
 *
 * Example 2: Whitelist violations for CERT C rules INT30-C and STR30-C caused
 * by coverity bugs:
 * NVOS_COV_WHITELIST(false_positive, NVOS_CERT(INT30_C), "Bug 123456")
 * NVOS_COV_WHITELIST(false_positive, NVOS_CERT(STR30_C), "Bug 123457")
 * <Next line of code with INT30-C and STR30-C violations>
 */
#define NVOS_COV_WHITELIST(type, checker, comment_str) \
        _Pragma(NVOS_COV_STRING(coverity compliance type checker comment_str))

/**
 * NVOS_COV_WHITELIST_BLOCK_BEGIN - Whitelist a coverity violation for a block
 *                                   of code.
 *
 * @param type        - This is the whitelisting category. Valid values are
 *                      deviate or false_positive.
 *                      deviate is for an approved rule deviation.
 *                      false_positive is normally used for a bug in coverity
 *                      which causes a false violation to appear in the scan.
 * @param num         - This is number of violations expected within the block.
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVOS_MISRA() or NVOS_CERT() macro to define
 *                      this field.
 * @param comment_str - This is the comment that you want associated with this
 *                      whitelisting. This should normally be a bug number
 *                      (ex: coverity bug) or JIRA task ID (ex: RFD). Unlike the
 *                      other arguments, this argument must be a quoted string.
 *
 * Use this macro to whitelist a coverity violation for a block of code. It
 * must be terminated by an NVOS_COV_WHITELIST_BLOCK_END()
 *
 * Example: Whitelist 10 MISRA rule 14.2 violation due to a deviation
 * documented in the JIRA TID-123 RFD:
 * NVOS_COV_WHITELIST_BLOCK_BEGIN(deviate, 10, NVOS_MISRA(Rule, 14_2), "JIRA TID-123")
 *  > Next block of code with 10 rule 14.2 violations
 * NVOS_COV_WHITELIST_BLOCK_END(NVOS_MISRA(Rule, 14_2))
 *
 */
#define NVOS_COV_WHITELIST_BLOCK_BEGIN(type, num, checker, comment_str) \
_Pragma(NVOS_COV_STRING(coverity compliance block type:num checker comment_str))

/**
 * NVOS_COV_WHITELIST_BLOCK_END - End whitelist a block of code.that is
 *                                 whitelisted with a
 *                                 NVOS_COV_WHITELIST_BLOCK_BEGIN
 *
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVOS_MISRA() or NVOS_CERT() macro to define
 *                      this field.
 *
 * Use this macro to mark the end of the block whitelisted by
 * NVOS_COV_WHITELIST_BLOCK_END()
 *
 */
#define NVOS_COV_WHITELIST_BLOCK_END(checker) \
	_Pragma(NVOS_COV_STRING(coverity compliance end_block checker))

/**
 * NVOS_COV_PEND_WHITELIST - Whitelist a coverity violation on the next line.
 * Use this for violations whose deviation request is not approved.
 *
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVOS_MISRA() or NVOS_CERT() macro to define
 *                      this field.
 * @param comment_str - This is the comment that you want associated with this
 *                      whitelisting. This should normally be a bug number
 *                      (ex: coverity bug) or JIRA task ID (ex: RFD). Unlike the
 *                      other arguments, this argument must be a quoted string.
 *
 * Use this macro to whitelist a coverity violation in the next line of code.
 *
 * Example 1: Whitelist a MISRA rule 14.2 violation due to a deviation
 * documented in the pending JIRA TID-123 RFD:
 * NVOS_COV_PEND_WHITELIST(NVOS_MISRA(Rule, 14_2), "JIRA TID-123")
 * <Next line of code with a rule 14.2 violation>
 *
 */
#define NVOS_COV_PEND_WHITELIST(checker, comment_str) \
        NVOS_COV_WHITELIST(deviate, checker, comment_str)

#else
/**
 * no-op macros for normal compilation - whitelisting is disabled when a
 * coverity scan is NOT being run
 */
#define NVOS_MISRA(type, num)
#define NVOS_CERT(num)
#define NVOS_COV_STRING(x)
#define NVOS_COV_WHITELIST(type, checker, comment_str)
#define NVOS_COV_WHITELIST_BLOCK_BEGIN(type, num, checker, comment_str)
#define NVOS_COV_WHITELIST_BLOCK_END(checker)
#define NVOS_COV_PEND_WHITELIST(checker, comment_str)
#endif

/**
 * Exit codes for function that call NVOS_EXIT().
 * Only LSBits are visible to parent process
 */

#define ERR_INT30_C_ADDU16  0x378
#define ERR_INT30_C_ADDU32  0x377
#define ERR_INT30_C_ADDU64  0x376
#define ERR_INT30_C_SUBU32  0x375
#define ERR_INT30_C_SUBU64  0x374
#define ERR_INT30_C_MULTU32 0x373
#define ERR_INT30_C_MULTU64 0x372
#define ERR_INT32_C_ADDS32  0x371
#define ERR_INT32_C_ADDS64  0x370
#define ERR_INT32_C_SUBS32  0x369
#define ERR_INT32_C_SUBS64  0x368
#define ERR_INT32_C_MULTS32 0x367
#define ERR_INT32_C_MULTS64 0x366
#define ERR_INT32_C_DIVS32  0x365
#define ERR_INT32_C_DIVS64  0x364
#define ERR_INT31_C_CAST_U64TOU8   0x361
#define ERR_INT31_C_CAST_U32TOS32  0x360
#define ERR_INT31_C_CAST_U32TOU16  0x359
#define ERR_INT31_C_CAST_U32TOU8   0x358
#define ERR_INT31_C_CAST_U64TOU32  0x357
#define ERR_INT31_C_CAST_U64TOS32  0x356
#define ERR_INT31_C_CAST_S32TOU32  0x355
#define ERR_INT31_C_CAST_S64TOU32  0x354
#define ERR_INT31_C_CAST_S32TOU64  0x353
#define ERR_INT31_C_CAST_S32TOU8   0x352
#define ERR_INT31_C_CAST_S64TOU64  0x351
#define ERR_INT31_C_CAST_S64TOS32  0x350
#define ERR_INT31_C_CAST_U64TOS64  0x34F
#define ERR_INT30_C_ADDU64UPTR 0x34E
#define ERR_INT30_C_ADDU32UPTR 0x34D
#define ERR_INT30_C_ADDU32VPTR ERR_INT30_C_ADDU32UPTR
#define ERR_INT30_C_MULTSIZET 0x34C

/**
 * Define macro for virtual address bit mask
 */
#define VA_ADDR_MASK      0xFFFFFFFFFFFFUL
#define OFFSET_MASK       0xFFFFUL

typedef uint32_t NvBoolVar;
/**
 * Define macro to replace boolean True
 */
#ifndef NvBoolTrue
#define NvBoolTrue ((NvBoolVar)0xBABAFACEU)
#endif

/**
 * Define macro to replace boolean False
 */
#ifndef NvBoolFalse
#define NvBoolFalse ((NvBoolVar)0x45450531U)
#endif

/** @brief convert signedness of 32bit value */
union flag_converter {
    int32_t flag_int32;
    uint32_t flag_uint32;
};

/*
 * Utility function to check for CERT-C violation.
 */

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer add operation do not wrap.
 */
static inline bool AddU16(uint16_t op1, uint16_t op2, uint16_t *result)
{
    bool e = false;

    if ((((uint16_t)UINT16_MAX - op1) < op2) == false) {
        *result = (uint16_t)(op1 + op2);
        e = true;
    }

    return e;
}

#if defined(INT16_MAX)
/** CERT INT30-C
 *  Precondition test to ensure that integer add operation do not wrap.
 */
static inline bool AddS16(int16_t op1, int16_t op2, int16_t *result)
{
    bool e = false;

    if (((op2 > 0) && (op1 > (INT16_MAX - op2))) ||
       ((op2 < 0) && (op1 < (INT16_MIN - op2)))) {
         return e;
    } else {
        *result = op1 + op2;
        e = true;
    }
    return e;
}
#endif

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer add operation do not wrap.
 */
static inline bool AddU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = false;

    if (((UINT32_MAX - op1) < op2) == false) {
        *result = op1 + op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C
 *  Precondition test to ensure that integer add operation do not wrap.
 */
static inline bool AddS32(int32_t op1, int32_t op2, int32_t *result)
{
    bool e = false;

    if (((op2 > 0) && (op1 > (INT32_MAX - op2))) ||
       ((op2 < 0) && (op1 < (INT32_MIN - op2)))) {
         return e;
    } else {
        *result = op1 + op2;
        e = true;
    }
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer add operation do not wrap.
 */
static inline bool AddU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = false;

    if (((UINT64_MAX - op1) < op2) == false) {
        *result = op1 + op2;
        e = true;
    }
    return e;
}

/** CERT INT30-C
 *  Precondition test to ensure that integer add operation do not wrap.
 */
static inline bool AddS64(int64_t op1, int64_t op2, int64_t *result)
{
    bool e = false;

    if (((op2 > 0) && (op1 > (INT64_MAX - op2))) ||
       ((op2 < 0) && (op1 < (INT64_MIN - op2)))) {
         return e;
    } else {
        *result = op1 + op2;
        e = true;
    }
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer multiplication operation
 *  do not wrap.
 */
static inline bool MultU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = false;

    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT32_MAX / op2)) == false)  {
        *result = op1 * op2;
    } else {
        goto fail;
    }

    e = true;

fail:
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer multiplication operation
 *  do not wrap.
 */
static inline bool MultU16(uint16_t op1, uint16_t op2, uint16_t *result)
{
    bool e = false;

    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    NVOS_COV_WHITELIST(deviate, NVOS_CERT(INT31_C), "<QNXBSP>:<qnx_asil_header>:<1>:<TID-448>")
    } else if ((op1 > (uint16_t)((uint16_t)UINT16_MAX / op2)) == false)  {
        *result = (uint16_t)(op1 * op2);
    } else {
        goto fail;
    }

    e = true;

fail:
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure casting unsigned long to unsigned integer
 *  do not wrap.
 */
static inline bool CastU64toU32(uint64_t op1, uint32_t *result)
{
    bool e = false;

    if (op1 <= (uint64_t)UINT32_MAX) {
         *result = (uint32_t)op1;
         e = true;
    }

    return e;
}


/** Increment uint32_t by one
 *  Precondition test to ensure value is within RANGE_MAX
 */
static inline void incrementUInt32(uint32_t *x)
{
    if (*x < UINT32_MAX) {
        (*x)++;
    } else {
        *x = UINT32_MAX;
    }
}


/** Increment uint64_t by one
 *  Precondition test to ensure value is within RANGE_MAX
 */
static inline void incrementUInt64(uint64_t *x)
{
    if (*x < UINT64_MAX) {
        (*x)++;
    } else {
        *x = UINT64_MAX;
    }
}

/** CERT INT30-C:
 *  Precondition test to ensure casting unsigned long to unsigned integer
 *  do not wrap.
 */
static inline bool CastU64toS32(uint64_t op1, int32_t *result)
{
    bool e = false;

    if (op1 <= (uint64_t)INT32_MAX) {
         *result = (int32_t)op1;
         e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure casting unsigned integer to signed integer
 *  do not wrap.
 */
static inline bool CastU32toS32(uint32_t op1, int32_t *result)
{
    bool e = false;

    if (op1 <= (uint32_t)INT32_MAX) {
         *result = (int32_t)op1;
         e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure casting signed integer to unsigned integer
 *  do not wrap.
 */
static inline bool CastS32toU32(int32_t op1, uint32_t *result)
{
    bool e = false;

    if (op1 >= 0) {
         *result = (uint32_t)op1;
         e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure casting signed integer to unsigned integer
 *  do not wrap.
 */
static inline bool CastS32toU16(int32_t op1, uint16_t *result)
{
    bool e = false;

    if ((op1 >= 0) && (op1 <= (int32_t)UINT16_MAX)) {
        *result = (uint16_t)op1;
        e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer multiplication operation
 *  do not wrap.
 */
static inline bool MultU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = false;

    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT64_MAX / op2)) == false)  {
        *result = op1 * op2;
    } else {
        goto fail;
    }

    e = true;

fail:
    return e;
}

/**
 * @brief Precondition test to ensure that signed integer multiplication
 *        operation do not wrap.
 *
 * @param[in] op1 Input operand to be Multiplied
 * @param[in] op2 Value multiplied to input operand
 * @param[out] result Reference of variable to store product of two input
 *             operands
 *
 * @returns false if operation result wraps, else true.
 */
static inline bool MultS32(int32_t op1, int32_t op2, int32_t *result)
{
    bool e = false;

    if ((op1 == 0) || (op2 == 0)) {
        *result = 0;
    } else if (op1 > (INT32_MAX / op2))  {
        goto fail;
    } else {
        *result = op1 * op2;
    }

    e = true;

fail:
    return e;
}


/** CERT INT30-C
 *  Precondition test to ensure that signed integer multiplication operation
 *  do not wrap.
 */
static inline bool MultS64(int64_t op1, int64_t op2, int64_t *result)
{
    bool e = false;

    if (op1 > 0) { /* op1 is positive */
        if (op2 > 0) { /* op1 and op2 are positive */
            if (op1 > (INT64_MAX / op2)) {
                return e;
            }
        } else { /* op1 positive, op2 nonpositive */
            if (op2 < (INT64_MIN / op1)) {
                return e;
            }
         } /* op1 positive, op2 nonpositive */
    } else { /* op1 is nonpositive */
        if (op2 > 0) { /* op1 is nonpositive, op2 is positive */
           if (op1 < (INT64_MIN / op2)) {
               return e;
           }
        } else { /* op1 and op2 are nonpositive */
            if ( (op1 != 0) && (op2 < (INT64_MAX / op1))) {
                return e;
            }
        } /* End if op1 and op2 are nonpositive */
    } /* End if op1 is nonpositive */

    e = true;
    *result = op1 * op2;
    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer subtraction operation
 *  do not wrap.
 */
static inline bool SubU32(uint32_t op1, uint32_t op2, uint32_t *result)
{
    bool e = false;

    if ((op1 < op2) == false) {
        *result = op1 - op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer subtraction operation
 *  do not wrap.
 */
static inline bool SubU16(uint16_t op1, uint16_t op2, uint16_t *result)
{
    bool e = false;

    if ((op1 < op2) == false) {
        *result = (uint16_t)(op1 - op2);
        e = true;
    }

    return e;
}

/**
 * CERT INT32-C:
 * Precondition test to ensure that signed integer subtraction operation
 * does not overflow.
 */
static inline bool SubS32(int32_t op1, int32_t op2, int32_t *result)
{
    bool e;

    if (((op2 > 0) && (op1 < (INT32_MIN + op2))) ||
    ((op2 < 0) && (op1 > (INT32_MAX + op2)))) {
        e = false;
    } else {
        *result = op1 - op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C:
 *  Precondition test to ensure that unsigned integer subtraction operation
 *  do not wrap.
 */
static inline bool SubU64(uint64_t op1, uint64_t op2, uint64_t *result)
{
    bool e = false;

    if ((op1 < op2) == false) {
        *result = op1 - op2;
        e = true;
    }

    return e;
}

/** CERT INT30-C
 *  Precondition test to ensure that signed integer subtraction operation
 *  do not wrap.
 */
static inline bool SubS64(int64_t op1, int64_t op2, int64_t *result)
{
    bool e = false;

    if (((op2 > 0) && (op1 < (INT64_MIN + op2))) ||
       ((op2 < 0) && (op1 > (INT64_MAX + op2)))) {
         return e;
    } else {
        *result = op1 - op2;
        e = true;
    }
    return e;
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that size_t subtraction operation
 * do not wrap.
 */
static inline
bool SubSizet(size_t op1, size_t op2, size_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
        return true;
    } else {
        return false;
    }
}

/**
 * Exit API is not supported in LK and BL.
 */

#ifdef NVOS_EXIT

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer add operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU16WithExit(uint16_t op1, uint16_t op2, uint16_t *result)
{
    if ((((uint16_t)UINT16_MAX - op1) < op2) == false) {
        *result = (uint16_t)(op1 + op2);
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU16);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer add operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if (((UINT32_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU32);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer add operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if (((UINT64_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU64);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that uintptr_t and unsigned integer add
 * operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU32UPTRWithExit(uintptr_t op1, uint32_t op2, uintptr_t *result)
{
    if (((UINTPTR_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU32UPTR);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that void * and unsigned integer add
 * operation does not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU32OffsetToVoidPtrWithExit(
    const void *op1,
    uint32_t op2, const void **result)
{
    NVOS_COV_WHITELIST(deviate, NVOS_MISRA(Rule, 11_6), "<QNXBSP>:<nvidia>:<1>:<TID-371>")
    if (op1 > (const void *)((uintptr_t)UINTPTR_MAX - (uintptr_t)op2)) {
        NVOS_EXIT(ERR_INT30_C_ADDU32VPTR);
    } else {
        *result = (const void *)((const uint8_t *)op1 + op2);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that uintptr_t and size_t add
 * operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddUPTRSizetWithExit(uintptr_t op1, size_t op2, uintptr_t *result)
{
    if (((UINTPTR_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU32UPTR);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that size_t add
 * operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddSizetWithExit(size_t op1, size_t op2, size_t *result)
{
    if (((SIZE_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU32UPTR);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that size_t
 * multiplication operation do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void MultSizetWithExit(size_t op1, size_t op2, size_t *result)
{
    if ((op1 == (size_t)0) || (op2 == (size_t)0)) {
       *result = (size_t)0;
    } else if (op1 > (SIZE_MAX / op2))  {
        NVOS_EXIT(ERR_INT30_C_MULTSIZET);
    } else {
        *result = op1 * op2;
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that uintptr_t unsigned integer add
 * operation do not wrap
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddU64UPTRWithExit(uintptr_t op1, uint64_t op2, uintptr_t *result)
{
    if (((UINTPTR_MAX - op1) < op2) == false) {
        *result = op1 + op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_ADDU64UPTR);
    }
}
/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer multiplication operation
 * do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void MultU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if ((op1 == 0U) || (op2 == 0U)) {
       *result = 0U;
    } else if ((op1 > (UINT32_MAX / op2)) == true)  {
        NVOS_EXIT(ERR_INT30_C_MULTU32);
    } else {
        *result = op1 * op2;
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer multiplication operation
 * do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void MultU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if ((op1 == 0UL) || (op2 == 0UL)) {
        *result = 0UL;
    } else if ((op1 > (UINT64_MAX / op2)) == true) {
        NVOS_EXIT(ERR_INT30_C_MULTU64);
    } else {
        *result = op1 * op2;
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer subtraction operation
 * do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when underflow cannot happen by design.
 */
static inline
void SubU32WithExit(uint32_t op1, uint32_t op2, uint32_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_SUBU32);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that unsigned integer subtraction operation
 * do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when underflow cannot happen by design.
 */
static inline
void SubU64WithExit(uint64_t op1, uint64_t op2, uint64_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_SUBU64);
    }
}

/**
 * CERT INT30-C:
 * Precondition test to ensure that size_t subtraction operation
 * do not wrap.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when underflow cannot happen by design.
 */
static inline
void SubSizetWithExit(size_t op1, size_t op2, size_t *result)
{
    if ((op1 < op2) == false) {
        *result = op1 - op2;
    } else {
        NVOS_EXIT(ERR_INT30_C_SUBU64);
    }
}

/**
 * CERT INT32-C:
 * Precondition test to ensure that signed integer add operation do not overflow.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void AddS32WithExit(int32_t op1, int32_t op2, int32_t *result)
{
    if (((op2 > 0) && (op1 > (INT32_MAX - op2))) ||
    ((op2 < 0) && (op1 < (INT32_MIN - op2)))) {
        NVOS_EXIT(ERR_INT32_C_ADDS32);
    } else {
        *result = op1 + op2;
    }
}

static inline
void AddS64WithExit(int64_t op1, int64_t op2, int64_t *result)
{
    if (((op2 > 0) && (op1 > (INT64_MAX - op2))) ||
    ((op2 < 0) && (op1 < (INT64_MIN - op2)))) {
        NVOS_EXIT(ERR_INT32_C_ADDS64);
    } else {
        *result = op1 + op2;
    }
}


/**
 * CERT INT32-C:
 * Precondition test to ensure that signed integer subtraction operation
 * does not overflow.
 * Function calls NVOS_EXIT() on overflow.
 * Only use when overflow cannot happen by design.
 */
static inline
void SubS32WithExit(int32_t op1, int32_t op2, int32_t *result)
{
    if (((op2 > 0) && (op1 < (INT32_MIN + op2))) ||
    ((op2 < 0) && (op1 > (INT32_MAX + op2)))) {
        NVOS_EXIT(ERR_INT32_C_SUBS32);
    } else {
        *result = op1 - op2;
    }
}

static inline
void SubS64WithExit(int64_t op1, int64_t op2, int64_t *result)
{
    if (((op2 > 0) && (op1 < (INT64_MIN + op2))) ||
    ((op2 < 0) && (op1 > (INT64_MAX + op2)))) {
        NVOS_EXIT(ERR_INT32_C_SUBS64);
    } else {
        *result = op1 - op2;
    }
}


/**
 * CERT INT32-C:
 * Precondition test to ensure that signed integer division operation
 * does not overflow or divide by zero.
 * Function calls NVOS_EXIT() on overflow or divide by zero.
 * Only use when overflow/divide by zero cannot happen by design.
 */
static inline
void DivS32WithExit(int32_t op1, int32_t op2, int32_t *result)
{
    if ((op2 == 0) || ((op1 == INT32_MIN) && (op2 == -1))) {
        NVOS_EXIT(ERR_INT32_C_DIVS32);
    } else {
        *result = op1 / op2;
    }
}

static inline
void DivS64WithExit(int64_t op1, int64_t op2, int64_t *result)
{
    if ((op2 == 0) || ((op1 == INT64_MIN) && (op2 == -1))) {
        NVOS_EXIT(ERR_INT32_C_DIVS64);
    } else {
        *result = op1 / op2;
    }
}


/** Casting functions to fix CERT INT31-C:
 *  Ensure integer conversions do not result in lost or misinterpreted data.
 */
static inline
uint32_t CastS32toU32WithExit(int32_t op)
{
    if (op < 0) {
        NVOS_EXIT(ERR_INT31_C_CAST_S32TOU32);
    }
    return (uint32_t)op;
}

static inline
int32_t CastU32toS32WithExit(uint32_t op)
{
    if (op > (uint32_t)INT32_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U32TOS32);
    }
    return (int32_t)op;
}

static inline
uint16_t CastU32toU16WithExit(uint32_t op)
{
    if (op > (uint32_t)UINT16_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U32TOU16);
    }
    return (uint16_t)op;
}

static inline
uint8_t CastU32toU8WithExit(uint32_t op)
{
    if (op > (uint32_t)UINT8_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U32TOU8);
    }
    return (uint8_t)op;
}

static inline
char CastU32toCharWithExit(uint32_t op)
{
    if (op > (uint32_t)INT8_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U32TOU8);
    }
    return (char)op;
}

static inline
uint8_t CastU64toU8WithExit(uint64_t op)
{
    if (op > (uint64_t)UINT8_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U64TOU8);
    }
    return (uint8_t)op;
}

static inline
uint32_t CastU64toU32WithExit(uint64_t op)
{
    if (op > UINT32_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U64TOU32);
    }
    return (uint32_t)op;
}

static inline
int32_t CastU64toS32WithExit(uint64_t op)
{
    if (op > (uint64_t)INT32_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U64TOS32);
    }
    return (int32_t)op;
}

static inline
uint32_t CastS64toU32WithExit(int64_t op)
{
    if ((op > (int64_t)UINT32_MAX) || (op < 0)) {
        NVOS_EXIT(ERR_INT31_C_CAST_S64TOU32);
    }
    return (uint32_t)op;
}

static inline
uint64_t CastS32toU64WithExit(int32_t op)
{
    if (op < 0) {
        NVOS_EXIT(ERR_INT31_C_CAST_S32TOU64);
    }
    return (uint64_t)op;
}

static inline
uint8_t CastS32toU8WithExit(int32_t op)
{
    if ((op > (int32_t)UINT8_MAX) || (op < 0)) {
        NVOS_EXIT(ERR_INT31_C_CAST_S32TOU8);
    }
    return (uint8_t)op;
}

static inline
uint64_t CastS64toU64WithExit(int64_t op)
{
    if (op < 0) {
        NVOS_EXIT(ERR_INT31_C_CAST_S64TOU64);
    }
    return (uint64_t)op;
}

static inline
int32_t CastS64toS32WithExit(int64_t op)
{
    int32_t max = INT32_MAX;
    int32_t min = INT32_MIN;

    if ((op > (int64_t)max) || (op < (int64_t)min)) {
        NVOS_EXIT(ERR_INT31_C_CAST_S64TOS32);
    }
    return (int32_t)op;
}

static inline
int64_t CastU64toS64WithExit(uint64_t op)
{
    if (op > (uint64_t)INT64_MAX) {
        NVOS_EXIT(ERR_INT31_C_CAST_U64TOS64);
    }
    return (int64_t)op;
}
#endif /* End of NVOS_EXIT */

#ifdef UINTPTR_MAX
/**
 * CERT INT30-C:
 * Ensure that uintptr_t and unsigned integer add do not
 *          result in lost or misinterpreted data
 */
static inline
bool AddU32UPTR(uintptr_t op1, uint32_t op2, uintptr_t *result)
{
    if (((UINTPTR_MAX - op1) < op2) == false) {
        *result = op1 + op2;
        return true;
    }
    return false;
}

/**
 * CERT INT30-C:
 * Ensure that adding a 32-bit integer offset to a void pointer does not result
 * in integer overflow
 */
static inline
bool AddU32OffsetToVoidPtr(const void *op1, uint32_t op2, const void **result)
{
    NVOS_COV_WHITELIST(deviate, NVOS_MISRA(Rule, 11_6), "<QNXBSP>:<nvidia>:<1>:<TID-371>")
    if (op1 <= (const void *)((uintptr_t)UINTPTR_MAX - (uintptr_t)op2)) {
        *result = (const void *)((const uint8_t *)op1 + op2);
        return true;
    }
    return false;
}
#endif /* End of UINTPTR_MAX */

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint64toUint32(uint64_t op, uint32_t *result)
{
    bool e = false;

    if ((op > UINT32_MAX) == false) {
        *result = (uint32_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint64toInt64(uint64_t op, int64_t *result)
{
    bool e = false;

    if ((op > (uint64_t)INT64_MAX) == false) {
        *result = (int64_t)op;
        e = true;
    }

    return e;
}

/** MISRA RULE 10.8
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline void NvConvertUint32toInt64(uint32_t op, int64_t *result)
{

    *result = (int64_t)op;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint32toUint16(uint32_t op, uint16_t *result)
{
    bool e = false;

    if ((op > (uint32_t)UINT16_MAX) == false) {
        *result = (uint16_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint32toInt32(uint32_t op, int32_t *result)
{
    bool e = false;

    if ((op > (uint32_t)INT32_MAX) == false) {
        *result = (int32_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint32toUint8(uint32_t op, uint8_t *result)
{
    bool e = false;

    if ((op > (uint32_t )UINT8_MAX) == false) {
        *result = (uint8_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertLint64toULint64(int64_t op, uint64_t *result)
{
    bool e = false;

    if (op >= 0) {
        *result = (uint64_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertLint64toUint32(int64_t op, uint32_t *result)
{
    bool e = false;

    if ((op >= 0) && ((op > (int64_t)UINT32_MAX) == false)) {
        *result = (uint32_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertLint64toInt32(int64_t op, int32_t *result)
{
    bool e = false;
    int32_t max = INT32_MAX;
    int32_t min = INT32_MIN;

    if ((op >= (int64_t)min) && (op <= (int64_t)max)) {
        *result = (int32_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertInt32toUint32(int32_t op, uint32_t *result)
{
    bool e = false;

    if ((op < 0) == false) {
        *result = (uint32_t)op;
        e = true;
    }

    return e;
}

#if defined(INT16_MAX)
/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertInt32toInt16(int32_t op, int16_t *result)
{
    bool e = false;

    if ((op > INT16_MAX) == false) {
        *result = (int16_t)op;
        e = true;
    }

    return e;
}

/** CERT INT31-C:
 * Ensure that integer conversions do not result in lost or misinterpreted data.
 */
static inline
bool NvConvertUint16toInt16(uint16_t op, int16_t *result)
{
    bool e = false;

    if ((op > (uint16_t)INT16_MAX) == false) {
        *result = (int16_t)op;
        e = true;
    }

    return e;
}
#endif /* End of INT16_MAX */

/** CERT INT30-C:
 *  Readl and writel APIs to avoid CERT-C warnings
 */

#if defined(__QNX__)

static inline
void NvWritel(uintptr_t base, uint64_t offset, uint32_t val)
{
    out32((base & VA_ADDR_MASK) + (offset & OFFSET_MASK), val);
}

static inline
uint32_t NvReadl(uintptr_t base, uint64_t offset)
{
    return in32((base & VA_ADDR_MASK) + (offset & OFFSET_MASK));
}

#endif /* End of __QNX__ */

/** CERT INT31-C:
 *
 * Convert Flags from Unsigned to Signed integer
 *
 * Note:- NvConvertUint32toInt32() cannot be used for this purpose
 */
static inline int32_t NvConvertFlagUInt32toInt32(uint32_t flag_uint32)
{

    union flag_converter flag;

    flag.flag_uint32 = flag_uint32;

    return flag.flag_int32;
}

/** CERT INT31-C:
 *
 * Convert Flags from Signed to Unsigned integer
 *
 * Note:- NvConvertInt32toUint32() cannot be used for this purpose
 */
static inline uint32_t NvConvertFlagInt32toUInt32(int32_t flag_int32)
{
    union flag_converter flag;

    flag.flag_int32 = flag_int32;

    return flag.flag_uint32;
}

NVOS_COV_WHITELIST(deviate, NVOS_MISRA(Rule, 3_1), "<QNXBSP>:<nvidia>:<1>:<TID-320>")
/** CERT INT34-C:
 *
 * Do not shift an expression by a negative number of bits or by greater than
 * or equal to the number of bits that exist in the operand.
 *
 * See: https://wiki.sei.cmu.edu/confluence/display/c/INT34-C.+Do+not+shift+an+expression+by+a+negative+number+of+bits+or+by+greater+than+or+equal+to+the+number+of+bits+that+exist+in+the+operand
 */
static inline bool ShiftLeftU32(
    uint32_t operand,
    uint32_t shift,
    uint32_t *result)
{
    bool e = false;

    if (shift < 32U /* bits in a uint32_t */) {
        *result = (operand << shift);
        e = true;
    }

    return e;
}

NVOS_COV_WHITELIST(deviate, NVOS_MISRA(Rule, 3_1), "<QNXBSP>:<nvidia>:<1>:<TID-320>")
/** CERT INT34-C:
 *
 * Do not shift an expression by a negative number of bits or by greater than
 * or equal to the number of bits that exist in the operand.
 *
 * See: https://wiki.sei.cmu.edu/confluence/display/c/INT34-C.+Do+not+shift+an+expression+by+a+negative+number+of+bits+or+by+greater+than+or+equal+to+the+number+of+bits+that+exist+in+the+operand
 */
static inline bool ShiftRightU32(
    uint32_t operand,
    uint32_t shift,
    uint32_t *result)
{
    bool e = false;

    if (shift < 32U /* bits in a uint32_t */) {
        *result = (operand >> shift);
        e = true;
    }

    return e;
}

#if defined(__cplusplus)
}
#endif
#endif // INCLUDED_NVOS_STATIC_ANALYSIS_H
