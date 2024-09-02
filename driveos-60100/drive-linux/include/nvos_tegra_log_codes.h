/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef INCLUDED_NVOS_TEGRA_LOG_CODES_H
#define INCLUDED_NVOS_TEGRA_LOG_CODES_H

/*
 * This file should not be included directly, include "include/nvos_s3_tegra_safety.h"
 */

/* @brief slog code for PCT parser */
#define NVOS_SLOG_CODE_PCT          (NVOS_LOG_CODE_START)
/* @brief slog code for CAN driver */
#define NVOS_SLOG_CODE_CAN          (NVOS_LOG_CODE_START + 1)
/* brief slog code for NvCVNAS driver */
#define NVOS_SLOG_CODE_CVNAS        (NVOS_LOG_CODE_START + 2)
/* @brief slog code for NvDT library */
#define NVOS_SLOG_CODE_DT           (NVOS_LOG_CODE_START + 3)
/* @brief slog code for HSP library */
#define NVOS_SLOG_CODE_HSP          (NVOS_LOG_CODE_START + 4)
/* @brief slog code for I2C driver */
#define NVOS_SLOG_CODE_I2C          (NVOS_LOG_CODE_START + 5)
/* @brief slog code for Iolauncher */
#define NVOS_SLOG_CODE_IOLAUNCHER   (NVOS_LOG_CODE_START + 6)
/* @brief slog code for NvSciipc */
#define NVOS_SLOG_CODE_IPC          (NVOS_LOG_CODE_START + 7)
/* @brief slog code for NvSKU driver */
#define NVOS_SLOG_CODE_SKU          (NVOS_LOG_CODE_START + 8)
/* @brief slog code for SPI driver */
#define NVOS_SLOG_CODE_SPI          (NVOS_LOG_CODE_START + 9)
/* @brief slog code for BPMP Comms driver */
#define NVBPMPC_SLOG_CODE           (NVOS_LOG_CODE_START + 10)
/* @brief slog code for VSCD driver */
#define NVVSCD_SLOG_CODE            (NVOS_LOG_CODE_START + 11)
/* @brief slog code for nvpwm */
#define NVOS_SLOG_CODE_NVPWM        (NVOS_LOG_CODE_START + 12)
/* @brief slog code for GPCDMA library */
#define NVOS_SLOG_CODE_GPCDMA       (NVOS_LOG_CODE_START + 13)
/* @brief slog code for timesync */
#define NVOS_SLOG_CODE_TIMESYNC     (NVOS_LOG_CODE_START + 14)
/* @brief slog code for nvmnand library */
#define NVOS_SLOG_CODE_NVMNAND      (NVOS_LOG_CODE_START + 15)
/* @brief slog code for QNX HV Driver */
#define NVOS_SLOG_CODE_NVHV         (NVOS_LOG_CODE_START + 16)
/* @brief slog code for NvMem library */
#define NVOS_SLOG_CODE_NVMEM        (NVOS_LOG_CODE_START + 17)
/* @brief slog code for NvGpio Driver */
#define NVOS_SLOG_CODE_NVGPIO       (NVOS_LOG_CODE_START + 18)
/* @brief slog code for NvClock */
#define NVOS_SLOG_CODE_NVCLOCK      (NVOS_LOG_CODE_START + 19)
/* @brief slog code for NvMedia Core unit */
#define NVOS_SLOG_CODE_NVMEDIA_CORE  (NVOS_LOG_CODE_START + 20)
/* @brief slog code for NvMedia Surface sub-element */
#define NVOS_SLOG_CODE_NVMEDIA_SURFACE (NVOS_LOG_CODE_START + 21)
/* @brief slog code for NvMedia ICP unit */
#define NVOS_SLOG_CODE_NVMEDIA_ICP (NVOS_LOG_CODE_START + 22)
/* @brief slog code for NvMedia ISP unit */
#define NVOS_SLOG_CODE_NVMEDIA_ISP (NVOS_LOG_CODE_START + 23)
/* @brief slog code for NvMedia SciSync unit */
#define NVOS_SLOG_CODE_NVMEDIA_NVMSCISYNC (NVOS_LOG_CODE_START + 24)
/* @brief slog code for NvMedia SciBuf unit */
#define NVOS_SLOG_CODE_NVMEDIA_NVMSCIBUF (NVOS_LOG_CODE_START + 25)
/* @brief slog code for NvMedia Utility unit */
#define NVOS_SLOG_CODE_NVMEDIA_UTILITY (NVOS_LOG_CODE_START + 26)
/* @brief slog code for NvMedia TVMR unit */
#define NVOS_SLOG_CODE_NVMEDIA_TVMR (NVOS_LOG_CODE_START + 27)
/* @brief slog code for plat safety monitor */
#define NVOS_SLOG_CODE_PLATSAFETY   (NVOS_LOG_CODE_START + 28)
/* @brief slog code for NvOS internal logging */
#define NVOS_SLOG_CODE_NVOS         (NVOS_LOG_CODE_START + 29)
/* @brief slog code for Camera driver */
#define NVOS_SLOG_CODE_CAMERA       (NVOS_LOG_CODE_START + 30)
/* @brief slog code for NvSysEventClient internal logging */
#define NVOS_SLOG_CODE_NVSYSEVENTCLIENT  (NVOS_LOG_CODE_START + 31)
/* @brief slog code for NvSysMgr internal logging */
#define NVOS_SLOG_CODE_NVSYSMGR     (NVOS_LOG_CODE_START + 32)
/* @brief slog code for NvMedia DLA unit */
#define NVOS_SLOG_CODE_NVMEDIA_DLA  (NVOS_LOG_CODE_START + 33)
/* @brief slog code for NvMedia TENSOR unit */
#define NVOS_SLOG_CODE_NVMEDIA_TENSOR (NVOS_LOG_CODE_START + 34)
/* @brief slog code for IST Client driver */
#define NVOS_SLOG_CODE_ISTCLIENT    (NVOS_LOG_CODE_START + 35)
/* @brief slog code for RPL library */
#define NVOS_SLOG_CODE_RPL          (NVOS_LOG_CODE_START + 36)
/* @brief code to read secure MMU-500 parity error register */
#define NVOS_LOG_CODE_SMMU_SDL   (NVOS_LOG_CODE_START + 37)
/* @brief slog code for NvHost internal logging */
#define NVOS_SLOG_CODE_NVHOST     (NVOS_LOG_CODE_START + 38)
/* @brief slog code for NvMap internal logging */
#define NVOS_SLOG_CODE_NVMAP     (NVOS_LOG_CODE_START + 39)
/* @brief slog code for TVMR IEP unit */
#define NVOS_SLOG_CODE_TVMR_IEP     (NVOS_LOG_CODE_START + 40)
/* @brief slog code for TVMR IOFST unit */
#define NVOS_SLOG_CODE_TVMR_IOFST   (NVOS_LOG_CODE_START + 41)
/* @brief slog code for NvMedia IEP unit */
#define NVOS_SLOG_CODE_NVMEDIA_IEP  (NVOS_LOG_CODE_START + 42)
/* @brief slog code for NvMedia IOFST unit */
#define NVOS_SLOG_CODE_NVMEDIA_IOFST (NVOS_LOG_CODE_START + 43)
/* @brief slog code for Thermal driver */
#define NVOS_SLOG_CODE_THERMAL_DRIVER   (NVOS_LOG_CODE_START + 44)
/* @brief slog code for NVSCI driver */
#define NVOS_SLOG_CODE_NVSCI         (NVOS_LOG_CODE_START + 45)
/* @brief slog code for Display driver */
#define NVOS_SLOG_CODE_NVDISP       (NVOS_LOG_CODE_START + 46)
/* @brief slog code for TVMR 2D unit */
#define NVOS_SLOG_CODE_TVMR_2D (NVOS_LOG_CODE_START + 47)
/* @brief slog code for NvMedia 2D unit */
#define NVOS_SLOG_CODE_NVMEDIA_2D (NVOS_LOG_CODE_START + 48)
/* @brief slog code for TVMR LDC unit */
#define NVOS_SLOG_CODE_TVMR_LDC (NVOS_LOG_CODE_START + 49)
/* @brief slog code for VIC related NvMedia units */
#define NVOS_SLOG_CODE_NVMEDIA_VIC (NVOS_LOG_CODE_START + 50)
/* @brief slog code for NvTZVault driver */
#define NVOS_SLOG_CODE_NVTZVAULT    (NVOS_LOG_CODE_START + 51)
/* @brief slog code for libteec driver */
#define NVOS_SLOG_CODE_LIBTEEC      (NVOS_LOG_CODE_START + 52)
/* @brief slog code for PVA driver */
#define NVOS_SLOG_CODE_NVPVA      (NVOS_LOG_CODE_START + 53)
/* @brief slog code for PVA Interface Unit */
#define NVOS_SLOG_CODE_PVASYS_INTERFACE  (NVOS_LOG_CODE_START + 54)
/* @brief slog code for PVA UMD Unit */
#define NVOS_SLOG_CODE_PVASYS_UMD  (NVOS_LOG_CODE_START + 55)
/* @brief slog code for NvVIC driver */
#define NVOS_SLOG_CODE_NVVIC (NVOS_LOG_CODE_START + 56)
/* @brief slog code for CCPLEX_HDS sub-element */
#define NVOS_SLOG_CODE_CCPLEX_HDS  (NVOS_LOG_CODE_START + 57)
/* @brief slog code for OIST */
#define NVOS_SLOG_CODE_OIST (NVOS_LOG_CODE_START + 58)
/* @brief slog code for nvdc */
#define NVOS_SLOG_CODE_NVDC      (NVOS_LOG_CODE_START + 59)
/* @brief slog code for nvpkcs11 */
#define NVOS_SLOG_CODE_NVPKCS11  (NVOS_LOG_CODE_START + 60)
/* @brief slog code for nvvse */
#define NVOS_SLOG_CODE_NVVSE  (NVOS_LOG_CODE_START + 61)
/* @brief slog code for NVVIDEO IOFA */
#define NVOS_SLOG_CODE_NVVIDEO_IOFA  (NVOS_LOG_CODE_START + 62)
/* @brief slog code for NVMEDIA IOFA */
#define NVOS_SLOG_CODE_NVMEDIA_IOFA  (NVOS_LOG_CODE_START + 63)
/* @brief slog code for NVDVMS */
#define NVOS_SLOG_CODE_NVDVMS  (NVOS_LOG_CODE_START + 64)
/* @brief slog code for NVDVMS */
#define NVOS_SLOG_CODE_NVPM2  (NVOS_LOG_CODE_START + 65)
/* @brief slog code for NvDTStartupCmd library */
#define NVOS_SLOG_CODE_NVDTS         (NVOS_LOG_CODE_START + 66)
/* @brief slog code for C2CPCIE */
#define NVOS_SLOG_CODE_C2CPCIE  (NVOS_LOG_CODE_START + 67)
/* @brief slog code for NVVIDEO COMMON */
#define NVOS_SLOG_CODE_NVVIDEO_CMN (NVOS_LOG_CODE_START + 68)

#if defined(NV_IS_TRACER_ENABLED) && (NV_IS_TRACER_ENABLED == 1)
/* @brief slog code for function tracer */
#define NVOS_SLOG_CODE_NVTRACER  (NVOS_LOG_CODE_START + 69)
#endif
/* @brief slog code for NVCERT parser */
#define NVOS_SLOG_CODE_NVCERT  (NVOS_LOG_CODE_START + 70)
/* @brief slog code for QNX Resmgr Utils */
#define NVOS_SLOG_CODE_RESMGR_UTILS  (NVOS_LOG_CODE_START + 71)
/* @brief slog code for nvplayfair_util */
#define NVOS_SLOG_CODE_NVPLAYFAIR_UTIL  (NVOS_LOG_CODE_START + 72)
/* @brief slog code for tests.
 *  This is a generic log code intended for usage from tests only. */
#define NVOS_SLOG_CODE_TEST  (NVOS_LOG_CODE_START + 73)
/* @brief slog code for nvhsierrrptinj */
#define NVOS_SLOG_CODE_NVHSIERRRPTINJ  (NVOS_LOG_CODE_START + 74)
/* @brief slog code for DLA */
#define NVOS_SLOG_CODE_NVDLA  (NVOS_LOG_CODE_START + 75)
/* @brief slog code for Drive Update */
#define NVOS_SLOG_CODE_NVDU  (NVOS_LOG_CODE_START + 76)
/* @brief slog code for VulkanSC library */
#define NVOS_SLOG_CODE_VULKANSC (NVOS_LOG_CODE_START + 77)
/* @brief slog code for nvrm_gpu */
#define NVOS_SLOG_CODE_NVRMGPU (NVOS_LOG_CODE_START + 78)
/* @brief slog code for NVRM */
#define NVOS_SLOG_CODE_NVRM  (NVOS_LOG_CODE_START + 79)
/* @brief slog code for NVETHERNET DT */
#define NVOS_SLOG_CODE_NVETHERNET_DT  (NVOS_LOG_CODE_START + 80)
/* @brief slog code for NVVIDEO IEP */
#define NVOS_SLOG_CODE_NVVIDEO_IEP  (NVOS_LOG_CODE_START + 81)
/* @brief slog code for nvrm sync */
#define NVOS_SLOG_CODE_NVRMSYNC  (NVOS_LOG_CODE_START + 82)
/* @brief slog code for OpenWFD */
#define NVOS_SLOG_CODE_OPENWFD  (NVOS_LOG_CODE_START + 83)
/* @brief slog code for nvrm_chip */
#define NVOS_SLOG_CODE_NVRMCHIP (NVOS_LOG_CODE_START + 84)
/* @brief slog code for nvrm_surface */
#define NVOS_SLOG_CODE_NVRMSURFACE (NVOS_LOG_CODE_START + 85)
/* @brief slog code for libbpmp */
#define NVOS_SLOG_CODE_LIBBPMP (NVOS_LOG_CODE_START + 86)
/* @brief slog code for automount */
#define NVOS_SLOG_CODE_AUTOMOUNT (NVOS_LOG_CODE_START + 87)
/* @brief slog code for datafs */
#define NVOS_SLOG_CODE_DATAFS (NVOS_LOG_CODE_START + 88)

#endif /* INCLUDED_NVOS_TEGRA_LOG_CODES_H */
