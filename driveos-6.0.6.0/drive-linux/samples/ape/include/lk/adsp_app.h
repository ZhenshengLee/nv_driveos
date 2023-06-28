/*
* Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef _ADSP_APPS_H_
#define _ADSP_APPS_H_

#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>


#ifdef __cplusplus
extern "C" {
#endif


/*
 * applications are allowed up to five data segments
 *
 *      .dram_data:      allocated at instance load time from DRAM.
 *                       cached.
 *                       local to ADSP.
 *      .dram_shared:    allocated at instance load time from DRAM.
 *                       unchached.
 *                       host CPU access.
 *                       loader returnes the CPU accessable address if the shared
 *                       memory segment to the caller.
 *      .dram_shared_wc: allocated at instance load time from DRAM.
 *                       memory region is buffered.
 *                       host CPU access.
 *                       loader returnes the CPU accessable address of the shared
 *                       memory segment to the caller.
 *      .aram_data:      allocated at instance load time from ARAM of possible.
 *                       from DRAM otherwise.
 *      .aram_x_data:    aram exclusive. allocated at instance load time from ARAM.
 *                       loader may prevent app from loading if allocation fails.
 *                       user defined action.
 *
 * the loader allocates these segments on behalf of the application and passes the
 * pointers in the applications args structure.
 */

/*  macros for section attributes */
#define __used			__attribute__((used))
#define __section_used(x)	__SECTION(x) __used

#define ADSP_DRAM		__section_used(".dram_data");
#define ADSP_DRAM_SHARED	__section_used(".dram_shared");
#define ADSP_DRAM_SHARED_WC	__section_used(".dram_shared_wc");
#define ADSP_ARAM		__section_used(".aram_data");
#define ADSP_ARAM_EXCLUSIVE	__section_used(".aram_x_data");

#define ARGV_SIZE_IN_WORDS	128

#define NVADSP_NAME_SZ		64

/* adsp_app startup flags */
#define ADSP_APP_FLAG_START_ON_BOOT		0x01
#define ADSP_APP_FLAG_CUSTOM_STACK_SIZE		0x02

struct app_mem_size {
	uint64_t dram;
	uint64_t dram_shared;
	uint64_t dram_shared_wc;
	uint64_t aram;
	uint64_t aram_x;
};

/*
 * Application instance args structure.  Filled by loader at instance
 * loading time. Interpretation is up to app.
 */
typedef struct adsp_app_args {
	 /* number of arguments passed in */
	int32_t  argc;
	 /* binary representation of arguments */
	int32_t  argv[ARGV_SIZE_IN_WORDS];
} adsp_app_args_t;

struct adsp_app_descriptor;

/*
 *  adsp_app_init: Init entry point for an ADSP app. it is expected that after this routine
 *  exectutes, the app is ready to communicate using its primary IPC mechanisms,
 *  e.g. mailbox.
 *
 *  return value:  if the app returns anything but NO_ERROR, the app will not
 *                  be launched.
 */
typedef status_t (*adsp_app_init) (const struct adsp_app_descriptor *,
		void *place_holder);


/*
 *  adsp_app_entry: main entry point for ADSP application. returns when app terminates.
 *
 *  return value:   app exit code.
 */
typedef status_t (*adsp_app_entry)(const struct adsp_app_descriptor *,
		void *place_holder);

#define ADSP_APP_START(appname) struct adsp_app_descriptor _app_ex_##appname __SECTION(".adsp_apps") = { .name = #appname, .args= NULL,
#define ADSP_APP_END };


/*
 *  application instance memory structure.  filled by loader at instance loading time.
 */
typedef struct adsp_app_mem{
	void		*dram;		/* DRAM segment*/
	void		*shared;	/* DRAM in shared memory segment. uncached */
	void		*shared_wc;	/* DRAM in shared memory segment. write combined */
	void		*aram;		/* ARAM if available, DRAM OK */
	uint32_t	aram_flag;	/* set to 1 if allocated from ARAM else 0 for DRAM */
	void		*aram_x;	/* ARAM Segment. exclusively */
	uint32_t	aram_x_flag;	/* set to 1 if ARAM exclusive allocation succeeded */
} adsp_app_mem_t;

/*
 *  adsp  application decsriptor. filled by loader for dynamically loadable apps.
 *	initialized by app for statically loaded apps.
 */
struct adsp_app_descriptor {
	const char		*name;
	adsp_app_init		init;
	adsp_app_entry		entry;
	adsp_app_mem_t		mem;
	struct app_mem_size	mem_size;
	const adsp_app_args_t	*args;
	unsigned int		flags;
	size_t			stack_size;
	const void              *handle;
	const uint64_t		host_ref;
	char			version[16];
};

#ifdef __cplusplus
}
#endif

#endif
