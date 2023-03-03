/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __ADSP_OS_CONFIG_H_
#define __ADSP_OS_CONFIG_H_

/*
 * Note:
 *
 * ADSP_OS_CONFIG_HWMBOX contains a duplicate reference to HWMBOX5
 * which is also present in hwmailbox.h. Updates to mailbox information
 * must be reflected in both places to ensure proper functionality.
 *
 * Duplication is done to fix missing header file build issues in PDK
 * since it is not possible to share all header files required
 */
#define ADSP_OS_CONFIG_HWMBOX         0x29D8000

/*
 * ADSP OS Config
 *
 * DECOMPRESS  (Bit 0) : Set if ADSP FW needs to be decompressed
 * VIRT CONFIG (Bit 1) : Set if virtualized configuration
 * DMA PAGE (Bits 7:4) : Contains DMA page information
 */

#define ADSP_CONFIG_DECOMPRESS_SHIFT  0
#define ADSP_CONFIG_DECOMPRESS_EN     1
#define ADSP_CONFIG_DECOMPRESS_MASK   (1 << ADSP_CONFIG_DECOMPRESS_SHIFT)

#define ADSP_CONFIG_VIRT_SHIFT        1
#define ADSP_CONFIG_VIRT_EN           1
#define ADSP_CONFIG_VIRT_MASK         (1 << ADSP_CONFIG_VIRT_SHIFT)

#define ADSP_CONFIG_DMA_PAGE_SHIFT    4
#define ADSP_CONFIG_DMA_PAGE_MASK     (0xF << ADSP_CONFIG_DMA_PAGE_SHIFT)

#endif /* __ADSP_OS_CONFIG_H_ */
