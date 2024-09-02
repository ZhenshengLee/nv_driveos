//
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#ifndef IVCLIB_STATIC_ANALYSIS_H
#define IVCLIB_STATIC_ANALYSIS_H

// Inline annotations to suppress Partial Deviations (PDs) and False Positives
// (FPs) reported by Coverity (since version 2019.06). Enabled as follows:
// $ NV_BUILD_CONFIGURATION_IS_COVERITY=1 tmp

#ifdef NV_IS_COVERITY
// Disable MISRA C and CERT C/CPP violations caused by the leading underscore in macro names.
_Pragma("coverity compliance block (deviate CERT_DCL51_CPP) (deviate CERT_DCL37_C)")
_Pragma("coverity compliance block (deviate MISRA_C_2012_Rule_21_1) (deviate MISRA_C_2012_Rule_21_2)")
// Disable MISRA C violations caused by sharing this header with C++ compiled binaries.
_Pragma("coverity compliance block (deviate MISRA_C_2012_Rule_2_5)")

// Disable violation(s) in the helper macro not affecting the final binary.
_Pragma("coverity compliance deviate MISRA_C_2012_Rule_20_10")
#define _str(_x) #_x
// The following _stan*() helpers should not be used outside of this file.
#define _stan(rule,fpdev,comment) _Pragma(_str(coverity compliance fpdev rule comment))
#ifdef STAN_SUPPRESS_APPROVED
#  define _stanA(rule,fpdev,id) _stan(rule,fpdev,_str(approved^id))
#else
#  define _stanA(rule,fpdev,id)
#endif
#ifdef STAN_SUPPRESS_PENDING
#  define _stanP(rule,fpdev,id) _stan(rule,fpdev,_str(pending^id))
#else
#  define _stanP(rule,fpdev,id)
#endif
#ifdef STAN_SUPPRESS_REJECTED
#  define _stanR(rule,fpdev,id) _stan(rule,fpdev,_str(rejected^id))
#else
#  define _stanR(rule,fpdev,id)
#endif
#ifdef STAN_SUPPRESS_INSPECTED
#  define _stanI(rule,fpdev,id) _stan(rule,fpdev,_str(inspected^id))
#else
#  define _stanI(rule,fpdev,id)
#endif

#define STAN_COVERITY_BUG_SATW_5118_begin \
    _Pragma("coverity compliance block (fp AUTOSAR_Cpp14_A2_11_1) (fp AUTOSAR_Cpp14_A7_1_6) (fp AUTOSAR_Cpp14_A7_1_8) (fp AUTOSAR_Cpp14_A7_4_1) (fp AUTOSAR_Cpp14_A11_3_1)")

#define STAN_COVERITY_BUG_SATW_5118_end \
    _Pragma("coverity compliance end_block AUTOSAR_Cpp14_A2_11_1 AUTOSAR_Cpp14_A7_1_6 AUTOSAR_Cpp14_A7_1_8 AUTOSAR_Cpp14_A7_4_1 AUTOSAR_Cpp14_A11_3_1")

#else

#define _stan(rule,fpdev,comment)
#define _stanA(rule,fpdev,id)
#define _stanP(rule,fpdev,id)
#define _stanR(rule,fpdev,id)
#define _stanI(rule,fpdev,id)

#define STAN_COVERITY_BUG_SATW_5118_begin
#define STAN_COVERITY_BUG_SATW_5118_end

#endif

// Tags for identifying reference/pointer function parameters.
// Should be paired with STAN_A8_4_8_PD_HV_053 annotations right before
// function definitions containing any of these tags.
#define OUT
#define INOUT

// All approved, pending, rejected and inspected FPs and PDs for ivclib shall be listed below.

// Approved deviations

#define STAN_EXP55_CPP_FP_HYP_7529 _stanA(CERT_EXP55_CPP,fp,HYP-7529)
#define STAN_INT30_C_PD_HYP_4978   _stanA(CERT_INT30_C,deviate,HYP-4978)
#define STAN_A0_1_6_FP_HYP_8787    _stanA(AUTOSAR_Cpp14_A0_1_6,fp,HYP-8787)
#define STAN_A5_2_2_PD_HYP_6132    _stanA(AUTOSAR_Cpp14_A5_2_2,deviate,HYP-6132)
#define STAN_A7_1_1_FP_HV_068      _stanA(AUTOSAR_Cpp14_A7_1_1,fp,HV-068)
#define STAN_A7_1_6_PD_HYP_6328    _stanA(AUTOSAR_Cpp14_A7_1_6,deviate,HYP-6328)
#define STAN_A7_2_2_PD_HV_100      _stanA(AUTOSAR_Cpp14_A7_2_2,deviate,HV-100)
#define STAN_A7_2_3_PD_HV_109      _stanA(AUTOSAR_Cpp14_A7_2_3,deviate,HV-109)
#define STAN_A7_4_1_PD_VS_069      _stanA(AUTOSAR_Cpp14_A7_4_1,deviate,VS-069)
#define STAN_A8_4_8_PD_HV_053      _stanA(AUTOSAR_Cpp14_A8_4_8,deviate,HV-053)
#define STAN_A9_6_1_FP_HV_070      _stanA(AUTOSAR_Cpp14_A9_6_1,fp,HV-070)
#define STAN_2_3_PD_HYP_10576      _stanA(MISRA_C_2012_Rule_2_3,deviate,HYP-10576)
#define STAN_2_5_PD_HYP_10576      _stanA(MISRA_C_2012_Rule_2_5,deviate,HYP-10576 )
#define STAN_4_3_PD_HYP_8049       _stanA(MISRA_C_2012_Directive_4_3,deviate,HYP-8049)
#define STAN_8_13_PD_HYP_10576     _stanA(MISRA_C_2012_Rule_8_13,deviate,HYP-10576 )
#define STAN_11_1_PD_HYP_4975      _stanA(MISRA_C_2012_Rule_11_1,deviate,HYP-4975)

// Pending deviations

// Rejected deviations

#define STAN_EXP32_C_PD_HYP_4977   _stanR(CERT_EXP32_C,deviate,HYP-4977)
#define STAN_A2_11_1_FP_HYP_5204   _stanR(AUTOSAR_Cpp14_A2_11_1,fp,HYP-5204)
#define STAN_4_7_PD_HYP_4979       _stanR(MISRA_C_2012_Directive_4_7,deviate,HYP-4979)
#define STAN_8_6_PD_HYP_4976       _stanR(MISRA_C_2012_Rule_8_6,deviate,HYP-4976)
#define STAN_11_3_PD_HYP_8056      _stanR(MISRA_C_2012_Rule_11_3,deviate,HYP-8056)
#define STAN_11_8_PD_HYP_4971      _stanR(MISRA_C_2012_Rule_11_8,deviate,HYP-4971)

// Violations Inspected/To inspect

#define STAN_INT31_C_FP_HYP_3854   _stanI(CERT_INT31_C,fp,HYP-3854)
#define STAN_A0_1_3_FP_HYP_8465    _stanI(AUTOSAR_Cpp14_A0_1_3,fp,HYP-8465)
#define STAN_A4_7_1_FP_HYP_4752    _stanI(AUTOSAR_Cpp14_A4_7_1,fp,HYP-4752)
#define STAN_M5_0_15_PD_HYP_8675   _stanI(AUTOSAR_Cpp14_M5_0_15,deviate,HYP-8675)
#define STAN_A5_2_3_PD_HYP_7746    _stanI(AUTOSAR_Cpp14_A5_2_3,deviate,HYP-7746)
#define STAN_M5_2_8_PD_HYP_7098    _stanI(AUTOSAR_Cpp14_M5_2_8,deviate,HYP-7098)
#define STAN_M7_4_3_PD_HYP_8507    _stanI(AUTOSAR_Cpp14_M7_4_3,deviate,HYP-8507)

// TBD
// L3 rule, cannot be treated as advisory
#define STAN_EXP37_C_PD_HYP_10579  _stanI(CERT_EXP37_C,deviate,HYP-10579)
// L3 rule, cannot be treated as advisory
#define STAN_EXP40_C_PD_HYP_10579  _stanI(CERT_EXP40_C,deviate,HYP-10579)
// L3 rule, cannot be treated as advisory
#define STAN_EXP44_C_PD_HYP_10579  _stanI(CERT_EXP44_C,deviate,HYP-10579)
// L3 rule, cannot be treated as advisory
#define STAN_INT36_C_PD_HYP_10579  _stanI(CERT_INT36_C,deviate,HYP-10579)
#define STAN_M0_1_10_FP_HYP_8741   _stanI(AUTOSAR_Cpp14_M0_1_10,fp,HYP-8741)

#ifdef NV_IS_COVERITY
_Pragma("coverity compliance end_block MISRA_C_2012_Rule_2_5")
_Pragma("coverity compliance end_block MISRA_C_2012_Rule_21_1 MISRA_C_2012_Rule_21_2")
_Pragma("coverity compliance end_block CERT_DCL51_CPP CERT_DCL37_C")
#endif

#endif // include guard
