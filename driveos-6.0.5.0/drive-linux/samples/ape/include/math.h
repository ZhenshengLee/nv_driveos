/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __MATH_H
#define __MATH_H

#ifdef __cplusplus
extern "C" {
#endif

double fabs(double x);
double pow(double x, double y);
double sqrt(double x);
double atan(double x);
double log(double x);
double sin(double x);
double cos(double x);
double exp(double x);
double atan2(double y, double x);
double acos(double x);
double tan(double x);
double floor(double x);
double ceil(double x);

#ifdef __cplusplus
}
#endif

#endif
