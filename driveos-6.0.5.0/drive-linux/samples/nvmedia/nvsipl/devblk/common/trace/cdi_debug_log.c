/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#if !NV_IS_SAFETY

#include "cdi_debug_log.h"
#include "log_utils.h"

void
SetCDILogLevel(CDILogLevel level)
{
    switch (level) {
        case CDI_LOG_LEVEL_ERR:
            SetLogLevel(LEVEL_ERR);
            break;
        case CDI_LOG_LEVEL_WARN:
            SetLogLevel(LEVEL_WARN);
            break;
        case CDI_LOG_LEVEL_INFO:
            SetLogLevel(LEVEL_INFO);
            break;
        case CDI_LOG_LEVEL_DBG:
            SetLogLevel(LEVEL_DBG);
            break;
        default:
            (void)0; //NO-OP
            break;
    }
    return;
}

#endif /* !NV_IS_SAFETY */