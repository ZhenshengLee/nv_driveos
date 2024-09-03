/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CDI_DS90UB971_TPG_H_
#define _CDI_DS90UB971_TPG_H_

#include "devblk_cdi.h"

typedef struct {
    unsigned int configSetIdx;
#if !NV_IS_SAFETY
    DevBlkCDIModuleConfig moduleConfig;
#endif
} ContextDS90UB971TPG;

DevBlkCDIDeviceDriver *GetDS90UB971TPGDriver(void);

#endif /* _CDI_DS90UB971_TPG_H_ */
