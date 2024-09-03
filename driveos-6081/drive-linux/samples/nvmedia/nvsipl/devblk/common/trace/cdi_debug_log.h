/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#if !NV_IS_SAFETY

#ifndef CDI_DEBUG_LOG_H
#define CDI_DEBUG_LOG_H

/** @brief Defines tracing/logging levels. */
typedef enum {
/** @brief Error message level: for use when logging messages related to fatal/unrecoverable errors. */
    CDI_LOG_LEVEL_ERR  = 0,
/** @brief Warning message level: for use when logging messages related to inert/recovered errors. */
    CDI_LOG_LEVEL_WARN = 1,
/** @brief Warning message level: for use when logging messages related to state of the system. */
    CDI_LOG_LEVEL_INFO = 2,
/** @brief Warning message level: for use when logging messages related to debug information. */
    CDI_LOG_LEVEL_DBG  = 3,
} CDILogLevel;

/**
 * @brief Sets the log level for CDI drivers, which are logged by the slogger
 *
 * @param[in] level     Level of logging for CDI drivers.
 */
void
SetCDILogLevel(CDILogLevel level);

#endif /* CDI_DEBUG_LOG_H */

#endif /* !NV_IS_SAFETY */
