/*
 * Copyright (c) 2012, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _TESTUTIL_I2C_H_
#define _TESTUTIL_I2C_H_

typedef void* I2cHandle;

int
testutil_i2c_open(
    int i2c_num,
    I2cHandle *handle);

void
testutil_i2c_close(
    I2cHandle handle);

int
testutil_i2c_write_subaddr(
    I2cHandle handle,
    unsigned int addr,
    unsigned char *buf,
    unsigned int len);

int
testutil_i2c_read_subaddr(
    I2cHandle handle,
    int addr,
    unsigned char *offset,
    unsigned int offset_len,
    unsigned char *buf,
    unsigned int buf_len);

#endif /* _TESTUTIL_I2C_H_ */
