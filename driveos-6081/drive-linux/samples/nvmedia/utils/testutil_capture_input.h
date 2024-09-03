/*
 * Copyright (c) 2012-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _TESTUTIL_CAPTURE_INPUT_H_
#define _TESTUTIL_CAPTURE_INPUT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "testutil_i2c.h"

typedef enum
{
    I2C0 = 0,
    I2C1,
    I2C2,
    I2C3,
    I2C4,
    I2C5,
    I2C6,
    I2C7
} I2cId;

typedef enum
{
    AnalogDevices_ADV7180,
    AnalogDevices_ADV7182,
    AnalogDevices_ADV7281,
    AnalogDevices_ADV7282,
    NationalSemi_DS90UR910Q,
    Toshiba_TC358743,
    OV10635,
    OV10640,
    TI_DS90UH940,
    Toshiba_TC358791,
    Toshiba_TC358791_CVBS,
    AnalogDevices_ADV7480,
    AnalogDevices_ADV7481C,
    AnalogDevices_ADV7481H,
    CapureInputDevice_NULL,
    MaxCapureInputDevice,
} CaptureInputDeviceId;

typedef void * CaptureInputHandle;

typedef enum
{
    HDMI,
    CVBS,
    FPDLINK,
    VIP,
} InputType;

typedef enum
{
    YUV_420,
    YUV_422,
    YUV_444,
    RGB
} FrameFormat;

typedef enum
{
    NTSC,
    PAL
} VideoStd;

typedef enum
{
    PROGRESSIVE,
    INTERLACED
} PictureStructure;

typedef struct
{
    unsigned int width;
    unsigned int height;
    union
    {
        struct
        {
            unsigned int lanes;
            FrameFormat format;
            PictureStructure structure;
        } hdmi2csi;
        struct
        {
            VideoStd std;
            PictureStructure structure;
            unsigned int inputCh;
        } cvbs2csi;
        struct
        {
            unsigned int lanes;
            FrameFormat format;
        } fpdlink2csi;
        struct
        {
            VideoStd std;
        } vip;
    };
    InputType input;
} CaptureInputConfigParams;

int
testutil_capture_input_open(
    I2cId i2cId,
    CaptureInputDeviceId deviceId,
    unsigned int captureMode,
    CaptureInputHandle *handle);

int
testutil_capture_input_detect(
    CaptureInputHandle handle,
    CaptureInputConfigParams *params);

int
testutil_capture_input_configure(
    CaptureInputHandle handle,
    CaptureInputConfigParams *params);

int
testutil_capture_input_start(
    CaptureInputHandle handle);

int
testutil_capture_input_stop(
    CaptureInputHandle handle);

int
testutil_capture_input_status(
    CaptureInputHandle handle);

void
testutil_capture_input_close(
    CaptureInputHandle handle);

#ifdef __cplusplus
}
#endif
#endif /* _TESTUTIL_CAPTURE_INPUT_H_ */
