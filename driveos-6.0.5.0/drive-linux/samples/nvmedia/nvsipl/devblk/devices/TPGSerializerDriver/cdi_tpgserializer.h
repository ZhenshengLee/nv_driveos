/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_MAXTPGSERIALIZER_H_
#define _CDI_MAXTPGSERIALIZER_H_

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"

#define LINE_BUFFER_SIZE 1024

#define TPGSERIALIZER_REGISTER(REGISTER) \
    REGISTER(TPGSERIALIZER_STATUS_REG_0) \
    REGISTER(TPGSERIALIZER_STATUS_REG_1) \
    REGISTER(TPGSERIALIZER_STATUS_REG_2) \
    REGISTER(TPGSERIALIZER_STATUS_REG_3) \
    REGISTER(TPGSERIALIZER_STATUS_MAX_ERR)

#define GENERATE_ENUM(ENUM) ENUM,
#define GENERATE_ENUM_STRING(STRING) #STRING,

enum ErrGrpTPGSerializer {
    TPGSERIALIZER_REGISTER(GENERATE_ENUM)
};


typedef struct {
    uint8_t dummy;
} _DriverHandle;

DevBlkCDIDeviceDriver *GetTPGSerializerDriver(void);

size_t
TPGSerializerReadErrorSize(void);

NvMediaStatus
TPGSerializerReadErrorData(
    size_t bufSize,
    uint8_t* const buffer);


#endif /* _CDI_MAXTPGSERIALIZER_H_ */
