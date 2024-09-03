/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited
 */

#ifndef INCLUDED_NVM_COV_WHITELIST_H
#define INCLUDED_NVM_COV_WHITELIST_H

/**
 * These macros are used for whitelisting coverity violations. The macros are
 * only enabled when a coverity scan is being run.
 */
#ifdef NV_IS_COVERITY
/**
 * NVM_MISRA - Define a MISRA rule for NVM_COV_WHITELIST.
 *
 * @param type - This should be Rule or Directive depending on if you're dealing
 *               with a MISRA rule or directive.
 * @param num  - This is the MISRA rule/directive number. Replace hyphens and
 *               periods in the rule/directive number with underscores. Example:
 *               14.2 should be 14_2.
 *
 * This is a convenience macro for defining a MISRA rule for the
 * NVM_COV_WHITELIST macro.
 *
 * Example 1: For defining MISRA rule 14.2, use NVM_MISRA(Rule, 14_2).
 * Example 2: For defining MISRA directive 4.7, use NVM_MISRA(Directive, 4_7).
 */
#define NVM_MISRA(type, num) MISRA_C_2012_##type##_##num

/**
 * NVM_CERT - Define a CERT C rule for NVM_COV_WHITELIST.
 *
 * @param num - This is the CERT C rule number. Replace hyphens and periods in
 *              the rule number with underscores. Example: INT30-C should be
 *              INT30_C.
 *
 * This is a convenience macro for defining a CERT C rule for the
 * NVM_COV_WHITELIST macro.
 *
 * Example: For defining CERT C rule INT30-C, use NVM_CERT(INT30_C).
 */
#define NVM_CERT(num) CERT_##num

/**
 * Helper macro for stringifying the _Pragma() string
 */
#define NVM_COV_STRING(x) #x

/**
 * NVM_COV_WHITELIST - Whitelist a coverity violation on the next line.
 *
 * @param type        - This is the whitelisting category. Valid values are
 *                      deviate or false_positive.
 *                      deviate is for an approved rule deviation.
 *                      false_positive is normally used for a bug in coverity
 *                      which causes a false violation to appear in the scan.
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVM_MISRA() or NVM_CERT() macro to define
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
 * NVM_COV_WHITELIST(deviate, NVM_MISRA(Rule, 14_2), "JIRA TID-123")
 * <Next line of code with a rule 14.2 violation>
 *
 * Example 2: Whitelist violations for CERT C rules INT30-C and STR30-C caused
 * by coverity bugs:
 * NVM_COV_WHITELIST(false_positive, NVM_CERT(INT30_C), "Bug 123456")
 * NVM_COV_WHITELIST(false_positive, NVM_CERT(STR30_C), "Bug 123457")
 * <Next line of code with INT30-C and STR30-C violations>
 */
#define NVM_COV_WHITELIST(type, checker, comment_str) \
        _Pragma(NVM_COV_STRING(coverity compliance type checker comment_str))

/**
 * NVM_COV_PEND_WHITELIST - Whitelist a coverity violation on the next line.
 * Use this for violations whose deviation request is not approved.
 *
 * @param checker     - This is the MISRA or CERT C rule causing the violation.
 *                      Use the NVM_MISRA() or NVM_CERT() macro to define
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
 * NVM_COV_PEND_WHITELIST(NVM_MISRA(Rule, 14_2), "JIRA TID-123")
 * <Next line of code with a rule 14.2 violation>
 *
 */
#define NVM_COV_PEND_WHITELIST(checker, comment_str) \
        NVM_COV_WHITELIST(deviate, checker, comment_str)

#else
/**
 * no-op macros for normal compilation - whitelisting is disabled when a
 * coverity scan is NOT being run
 */
#define NVM_MISRA(type, num)
#define NVM_CERT(num)
#define NVM_COV_STRING(x)
#define NVM_COV_WHITELIST(type, checker, comment_str)
#define NVM_COV_PEND_WHITELIST(checker, comment_str)
#endif

#endif /* INCLUDED_NVM_COV_WHITELIST_H */
