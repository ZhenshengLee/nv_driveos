/*
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <asm.h>
#include <adsp_os_config.h>

.section ".text.loader"
.globl _loader_start

_loader_start:
	cpsid   iaf       // interrupts disabled

	/* Perform ADSP FW decompression based on Guest Reset
	 * information from ADSP_OS_CONFIG_HWMBOX */
	mov		r0, #ADSP_CONFIG_DECOMPRESS_EN
	ldr		r2, =ADSP_OS_CONFIG_HWMBOX
	ldr		r3, [r2]
	and		r1, r3, #ADSP_CONFIG_DECOMPRESS_MASK
	cmp		r0, r1
	beq		.Ldecom_os
	b		.Lboot_os
.Ldecom_os:
	mrs		r0, cpsr
	bic		r0, r0, #0x1f

	ldr		r2, =loader_stack_top

	/* set up the stack for supervisor mode */
	mrs		r0, cpsr
	bic		r0, r0, #0x1f
	orr		r1, r0, #0x13 // supervisor
	msr		cpsr_c, r1

	ldr		r2, =loader_stack_top
	mov		sp, r2

_loader_setup:
#Setup sequence for ARM with CP15
	/* do some cpu setup */
	mrc		p15, 0, r0, c1, c0, 0
		/* XXX this is currently for arm926, revist with armv6 cores */
		/* new thumb behavior, low exception vectors, i/d cache disable, mmu disabled */
	bic		r0, r0, #(1<<15| 1<<13 | 1<<12)
	bic		r0, r0, #(1<<2 | 1<<0)
		/* enable alignment faults */
	orr		r0, r0, #(1<<1)
	mcr		p15, 0, r0, c1, c0, 0

#Setup sequence from WITH_CPU_EARLY_INIT
__cpu_early_init:
	mrc	p15, 0, r0, c0, c0, 0	@ read main ID register
	and	r1, r0, #0xff000000	@ ARM?
	teq	r1, #0x41000000
	bne	3f
	and	r2, r0, #0x00f00000	@ variant
	and	r3, r0, #0x0000000f	@ revision
	orr	r3, r3, r2, lsr #20-4	@ combine variant and revision
	ubfx	r0, r0, #4, #12		@ primary part number

	/* Cortex-A9 Errata */
	ldr	r1, =0x00000c09		@ Cortex-A9 primary part number
	teq	r0, r1
	bne	3f
	cmp	r3, #0x10		@ power ctrl reg added r1p0
	mrcge	p15, 0, r1, c15, c0, 0	@ read power control register
	orrge	r1, r1, #1		@ enable dynamic clock gating
	mcrge	p15, 0, r1, c15, c0, 0	@ write power control register

	/* CONFIG_ARM_ERRATA_742230 */
	cmp	r3, #0x22		@ only present up to r2p2
	mrcle	p15, 0, r1, c15, c0, 1	@ read diagnostic register
	orrle	r1, r1, #1 << 4		@ set bit #4
	mcrle	p15, 0, r1, c15, c0, 1	@ write diagnostic register

	/* CONFIG_ARM_ERRATA_743622 */
	teq	r3, #0x20		@ present in r2p0
	teqne	r3, #0x21		@ present in r2p1
	teqne	r3, #0x22		@ present in r2p2
	teqne	r3, #0x27		@ present in r2p7
	teqne	r3, #0x29		@ present in r2p9
	mrceq	p15, 0, r1, c15, c0, 1	@ read diagnostic register
	orreq	r1, r1, #1 << 6		@ set bit #6
	mcreq	p15, 0, r1, c15, c0, 1	@ write diagnostic register

	/* CONFIG_ARM_ERRATA_751472 */
	cmp	r3, #0x30		@ present prior to r3p0
	mrclt	p15, 0, r1, c15, c0, 1	@ read diagnostic register
	orrlt	r1, r1, #1 << 11	@ set bit #11
	mcrlt	p15, 0, r1, c15, c0, 1	@ write diagnostic register

	/* CONFIG_ARM_ERRATA_752520 */
	cmp	r3, #0x29		@ present prior to r2p9
	mrclt	p15, 0, r1, c15, c0, 1	@ read diagnostic register
	orrlt	r1, r1, #1 << 20	@ set bit #20
	mcrlt	p15, 0, r1, c15, c0, 1	@ write diagnostic register

#Clearing out the bss
	ldr		r0, =__bss_start
	ldr		r1, =__bss_end
	mov		r2, #0
__bss_loop:
	cmp		r0, r1
	strlt		r2, [r0], #4
	blt		__bss_loop

#Branch to decompression sequence
	bl loader_main

	/* Clear Decompression bit in ADSP_OS_CONFIG_HWMBOX */
	ldr		r2, =ADSP_OS_CONFIG_HWMBOX
	ldr		r3, [r2]
	bic		r3, r3, #ADSP_CONFIG_DECOMPRESS_MASK
	str		r3, [r2]
.Lboot_os:
	b 		MEMBASE

3:
	b	.			@ unexpected case

.section .containers
.incbin "%OUTBIN%.xz"

.bss
.align 3
LOCAL_DATA(loader_stack)
	.skip 2048
LOCAL_DATA(loader_stack_top)

