/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _TPG_SENSOR_SERIALIZER_UTILITY_H_
#define _TPG_SENSOR_SERIALIZER_UTILITY_H_


NvMediaStatus
TPGReadErrorData(size_t errGrpSize,
                 const char** errGrpLUT,
                 const char* errCodeFileName,
                 uint16_t* errGrpCode);

#endif
