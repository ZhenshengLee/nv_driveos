/*
 * pingPong.h
 *
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __PINGPONG_H
#define __PINGPONG_H

#define ABNORMAL_TERMINATION_TIMEOUT 2
#define ALIVE_MSG_CHAR 'H'
#define ALIVE_MSG_LEN 1

#define PINGPONG_RATE_PER_SECOND 5
#define SLEEP_TIME_IN_MICROSECONDS ((1000 * 1000) / PINGPONG_RATE_PER_SECOND)

#endif //__PINGPONG_H
