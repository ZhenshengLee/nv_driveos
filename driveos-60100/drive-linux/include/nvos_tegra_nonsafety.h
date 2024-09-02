/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited
 */

#ifndef INCLUDED_NVOS_TEGRA_NONSAFETY_H
#define INCLUDED_NVOS_TEGRA_NONSAFETY_H

/**
 * @defgroup nvos_group Operating System Abstraction (NvOS)
 *
 * This provides a basic set of interfaces to unify code
 * across all supported operating systems. This layer does @b not
 * handle any hardware specific functions, such as interrupts.
 * "Platform" setup and GPU access are done by other layers.
 *
 * @warning Drivers and applications should @b not make any operating system
 * calls outside of this layer, @b including stdlib functions. Doing so will
 * result in non-portable code.
 *
 * For APIs that take key parameters, keys may be of ::NVOS_KEY_MAX length.
 * Any characters beyond this maximum is ignored.
 *
 * All strings passed to or from NvOS functions are encoded in UTF-8. For
 * character values below 128, this is the same as simple ASCII.
 *
 * @par Important:
 *
 *  At interrupt time there are only a handful of NvOS functions that are safe
 *  to call:
 *  - ::NvOsSemaphoreSignal
 *  - ::NvOsIntrMutexLock
 *  - ::NvOsIntrMutexUnlock
 *
 * <a name="nvos_cfg_support"></a>
 * <h2>NvOs Configuration Data Storage Support</h2>
 *
 * NvOs supports named configuration variables that may have
 * string values or unsigned integer values.
 *
 * The storage of the configuration data is highly platform
 * dependent. The platform may provide the means to set the initial
 * data for the storage in the platform build phase. Also it may
 * be possible to set and retrieve the values in a running system
 * via OS provided mechanisms.
 *
 * Information about the configuration data storages for various
 * operating systems follows.
 *
 * @par Linux / Win32 x86 Host
 *
 * On x86 host systems and Linux the configuration data storage is the
 * environment inherited from the shell that launches the application.
 * To set configuration variable values export the appropriate
 * environment variables from the launching shell.
 *
 * @par Windows CE / Window Mobile
 *
 * On target Windows systems the configuration data is stored in
 * the Windows registry, under the registry path:
 * <pre>
 *     HKEY_LOCAL_MACHINE\\Software\\NVIDIA Corporation\\NvOs
 * </pre>
 * The configuration data can be set by \c .reg files in the image build.
 *
 * @par Android
 *
 * On Android the configuration store is the Android property
 * mechanism. NvOs config variables are stored in the "persist.tegra"
 * namespace. Values can be set via the "setprop" tool and values
 * queried via the "getprop" tool.
 *
 * As with other Linux systems, the environment acts as a read-only
 * configuration data store. Values read from the environment override
 * those read from Android properties.
 *
 * @par Other Target Systems
 *
 * The default for target systems is to use a file on the file
 * system as the data storage. The path to the file on the
 * target file system is:
 * <pre>
 *    /etc/tegra_config.txt
 * </pre>
 * @{
 */

#include <stdarg.h>
#include "nvcommon.h"
#include "nverror.h"
/* common logging APIs */
#include "nvos_s3_tegra_log.h"
#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * A physical address. Must be 64 bits for OSs that support more than 64 bits
 * of physical addressing, not necessarily correlated to the size of a virtual
 * address.
 *
 * Currently, 64-bit physical addressing is supported by NvOS on WinNT only.
 *
 * XXX 64-bit phys addressing really should be supported on Linux/x86, since
 * all modern x86 CPUs have 36-bit (or more) physical addressing. We might
 * need to control a PCI card that the SBIOS has placed at an address above
 * 4 GB.
 */
#if NVOS_IS_WINDOWS
typedef NvU64 NvOsPhysAddr;
#else
typedef NvU32 NvOsPhysAddr;
#endif

/** The maximum length of a shared resource identifier string.
 */
#define NVOS_KEY_MAX 128

/** The maximum length for a file system path.
 */
#define NVOS_PATH_MAX 256

/** Indicates that ::NvOsStatType structure has \c mtime.
 */
#define NVOS_HAS_MTIME 1

/** Indicates that NvOs has NvOsMkdir().
 */
#define NVOS_HAS_MKDIR 1

/** Indicates that NvOs has NvOsFtruncate().
 */
#define NVOS_HAS_FTRUNCATE 1

/** Indicates that NvOs has condition variables.
 */
#define NVOS_HAS_CONDITION 1

/** @name Print Operations
 */
/*@{*/

/** Printf family. */
typedef struct NvOsFileRec *NvOsFileHandle;

/** Prints a string to a file stream.
 *
 *  @param stream The file stream to which to print.
 *  @param format The format string.
 */
NvError
NvOsFprintf(NvOsFileHandle stream, const char *format, ...);

// Doxygen requires escaping backslash characters (\) with another \ so in
// @return, ignore the first backslash if you are reading this in the header.
/** Expands a string into a given string buffer.
 *
 *  @param str A pointer to the target string buffer.
 *  @param size The size of the string buffer.
 *  @param format A pointer to the format string.
 *
 *  @return The number of characters printed (not including the \\0).
 *  The buffer was printed to successfully if the returned value is
 *  greater than -1 and less than \a size.
 */
NvS32
NvOsSnprintf(char *str, size_t size, const char *format, ...);

/** Prints a string to a file stream using a va_list.
 *
 *  @param stream The file stream.
 *  @param format A pointer to the format string.
 *  @param ap The va_list structure.
 */
NvError
NvOsVfprintf(NvOsFileHandle stream, const char *format, va_list ap);

/** Expands a string into a string buffer using a va_list.
 *
 *  @param str A pointer to the target string buffer.
 *  @param size The size of the string buffer.
 *  @param format A pointer to the format string.
 *  @param ap The va_list structure.
 *
 *  @return The number of characters printed (not including the \\0).
 *  The buffer was printed to successfully if the returned value is
 *  greater than -1 and less than \a size.
 */
NvS32
NvOsVsnprintf(char *str, size_t size, const char *format, va_list ap);

/**
 * Outputs a message to the debugging console, if present. All device driver
 * debug printfs should use this. Do not use this for interacting with a user
 * from an application; in that case, use NvTestPrintf() instead.
 *
 * @param format A pointer to the format string.
 */
void
NvOsDebugPrintf(const char *format, ...);

/**
 * Same as ::NvOsDebugPrintf, except takes priority and tag fields.
 */
void
NvOsLogPrintf(int prio, const char *tag, const char *format, ...);

/**
 * Same as ::NvOsDebugPrintf, except takes a va_list.
 */
void
NvOsDebugVprintf( const char *format, va_list ap );

/**
 * Same as ::NvOsDebugPrintf, except returns the number of chars written.
 *
 * @return number of chars written or -1 if that number is unavailable.
 */
NvS32
NvOsDebugNprintf( const char *format, ...);

void
NvOsDebugString(const char *str);

enum {
    NVOS_VERB_LVL_0 = 0,
    NVOS_VERB_LVL_1 = 1,
    NVOS_VERB_LVL_2 = 2,
    NVOS_VERB_LVL_3 = 3,
    NVOS_VERB_LVL_4 = 4,
};

int
NvOsSetVerboseLevel(NvS32 VerbLevel);

int
NvOsGetVerboseLevel(NvS32 *VerbLevel);

void
NvOsVerbosePrintf(NvS32 VerbLevel, const char *Fmt, ...);

/**
 * Prints an error and the line it appeared on.
 * Does nothing if \a err is @c NvSuccess.
 *
 * @param err The error to return.
 * @param file A pointer to the file the error occurred in.
 * @param line The line number the error occurred on.
 * @returns The \a err value, on success.
 */
NvError
NvOsShowError(NvError err, const char *file, int line);

/**
 * Prints err if it is an error (does nothing if err==NvSuccess).
 * Always returns err unchanged
 * never prints anything if err==NvSuccess)
 *
 * NOTE: Do not use this with errors that are expected to occur under normal
 * situations.
 *
 * @param err - the error to return
 * @returns err
 */
#define NV_SHOW_ERRORS  NV_DEBUG
#if     NV_SHOW_ERRORS
#define NV_SHOW_ERROR(err)  NvOsShowError(err,__FILE__,__LINE__)
#else
#define NV_SHOW_ERROR(err)  (err)
#endif

// Doxygen requires escaping # with a backslash, so in the examples below
// ignore the backslash before the # if reading this in the header file.
/**
 * Helper macro to go along with ::NvOsDebugPrintf. Usage:
 * <pre>
 *     NV_DEBUG_PRINTF(("foo: %s\n", bar));
   </pre>
 *
 * The debug print will be disabled by default in all builds, debug and
 * release. @note Usage requires double parentheses.
 *
 * To enable debug prints in a particular .c file, add the following
 * to the top of the .c file and rebuild:
 * <pre>
 *     \#define NV_ENABLE_DEBUG_PRINTS 1
   </pre>
 *
 * To enable debug prints in a particular module, add the following
 * to the makefile and rebuild:
 * <pre>
 *     LCDEFS += -DNV_ENABLE_DEBUG_PRINTS=1
   </pre>
 *
 */
#if !defined(NV_ENABLE_DEBUG_PRINTS)
#define NV_ENABLE_DEBUG_PRINTS 0
#endif
#if NV_ENABLE_DEBUG_PRINTS
#define NV_DEBUG_PRINTF(x) do { NvOsDebugPrintf x; } while (NV_FALSE)
#else
#define NV_DEBUG_PRINTF(x) do {} while (NV_FALSE)
#endif

/*@}*/
/** @name OS Version
 */
/*@{*/

typedef enum
{
    NvOsOs_Unknown = 0x0UL,
    NvOsOs_Windows = 0x1UL,
    NvOsOs_Linux = 0x2UL,
    NvOsOs_Qnx = 0x4UL,
    NvOsOs_Integrity = 0x5UL,
    NvOsOs_Force32 = 0x7fffffffUL,
} NvOsOs;

typedef enum
{
    NvOsSku_Unknown = 0x0UL,
    NvOsSku_Android = 0x1UL,
    NvOsSku_Force32 = 0x7fffffffUL,
} NvOsSku;

typedef struct NvOsOsInfoRec
{
    NvOsOs  OsType;
    NvOsSku Sku;
} NvOsOsInfo;

/**
 * Gets the current OS version.
 *
 * @param pOsInfo A pointer to the operating system information structure.
 */
NvError
NvOsGetOsInformation(NvOsOsInfo *pOsInfo);

/*@}*/

/** @name String Operations
 */
/*@{*/

/** Copies a string.
 *
 *  The \a src string must be NUL terminated.
 *
 * @deprecated Instead use strcpy().
 *
 *  @param dest A pointer to the destination of the copy.
 *  @param src A pointer to the source string.
 */
void
NvOsStrcpy(char *dest, const char *src);

/** Copies a string, length-limited.
 *
 * @deprecated Instead use strncpy().
 *
 *  @param dest A pointer to the destination of the copy.
 *  @param src A pointer to the source string.
 *  @param size The length of the \a dest string buffer plus NULL terminator.
 */
void
NvOsStrncpy(char *dest, const char *src, size_t size);

/** Defines straight-forward mappings to international language encodings.
 *  Commonly-used encodings on supported operating systems are provided.
 *  @note NvOS string (and file/directory name) processing functions expect
 *  UTF-8 encodings. If the system-default encoding is not UTF-8,
 *  conversion may be required. @see NvUStrConvertCodePage.
 *
 **/
typedef enum
{
    NvOsCodePage_Unknown = 0x0UL,
    NvOsCodePage_Utf8 = 0x1UL,
    NvOsCodePage_Utf16 = 0x2UL,
    NvOsCodePage_Windows1252 = 0x3UL,
    NvOsCodePage_Force32 = 0x7fffffffUL,
} NvOsCodePage;

/** Gets the length of a string.
 *
 * @deprecated Instead use strlen().
 *
 *  @param s A pointer to the string.
 */
size_t
NvOsStrlen(const char *s);

/** Gets the length of a string. length-limited.
 *
 *  @param s A pointer to the string.
 *  @param size the maximum length to check
 */
size_t
NvOsStrnlen(const char *s, size_t size);

/*@}*/
/** @name Memory Operations (Basic)
 */
/*@{*/

/** Copies memory.
 *
 * @deprecated Instead use memcpy().
 *
 *  @param dest A pointer to the destination of the copy.
 *  @param src A pointer to the source memory.
 *  @param size The length of the copy.
 */
void NvOsMemcpy(void *dest, const void *src, size_t size);

/** Compares two memory regions.
 *
 * @deprecated Instead use memcmp().
 *
 *  @param s1 A pointer to the first memory region.
 *  @param s2 A pointer to the second memory region.
 *  @param size The length to compare.
 *
 *  This returns 0 If the memory regions are identical
 */
int
NvOsMemcmp(const void *s1, const void *s2, size_t size);

/** Sets a region of memory to a value.
 *
 * This API is deprecated -- use memset() instead in new code!
 *
 *  @param s A pointer to the memory region.
 *  @param c The value to set.
 *  @param size The length of the region.
 */
void
NvOsMemset(void *s, NvU8 c, size_t size);

/*@}*/
/** @name File Input/Output
 */
/*@{*/

/**
 *  Defines wrappers over stdlib's file stream functions,
 *  with some changes to the API.
 */
typedef enum
{
    /** NvOS equivalent of SEEK_SET */
    NvOsSeek_Set = 0,
    /** NvOS equivalent of SEEK_CUR */
    NvOsSeek_Cur = 1,
    /** NvOS equivalent of SEEK_END */
    NvOsSeek_End = 2,

    /** Max value for NvOsSeekEnum */
    NvOsSeek_Force32 = 0x7FFFFFFF
} NvOsSeekEnum;

typedef enum
{
    NvOsFileType_Unknown = 0,
    NvOsFileType_File = 1,
    NvOsFileType_Directory = 2,
    NvOsFileType_Fifo = 3,
    NvOsFileType_CharacterDevice = 4,
    NvOsFileType_BlockDevice = 5,

    NvOsFileType_Force32 = 0x7FFFFFFF
} NvOsFileType;

typedef struct NvOsStatTypeRec
{
    NvU64 size;
    NvOsFileType type;
    /// last modified time of the file
    NvU64 mtime;
} NvOsStatType;

/** Opens a file with read permissions. */
#define NVOS_OPEN_READ    0x1

/** Opens a file with write persmissions. */
#define NVOS_OPEN_WRITE   0x2

/** Creates a file if is not present on the file system. */
#define NVOS_OPEN_CREATE  0x4

/** Open file in append mode. Implies WRITE. */
#define NVOS_OPEN_APPEND  0x8

/** Opens a file stream.
 *
 *  If the ::NVOS_OPEN_CREATE flag is specified, ::NVOS_OPEN_WRITE must also
 *  be specified.
 *
 *  If \c NVOS_OPEN_WRITE is specified, the file will be opened for write and
 *  will be truncated if it was previously existing.
 *
 *  If \c NVOS_OPEN_WRITE and ::NVOS_OPEN_READ are specified, the file will not
 *  be truncated.
 *
 *  @param path A pointer to the path to the file.
 *  @param flags Or'd flags for the open operation (\c NVOS_OPEN_*).
 *  @param [out] file A pointer to the file that will be opened, if successful.
 */
NvError
NvOsFopen(const char *path, NvU32 flags, NvOsFileHandle *file);

/** Closes a file stream.
 *
 *  @param stream The file stream to close.
 *  Passing in a null handle is okay.
 *
 *  @return NvSuccess if successful.
 *  @return NvError_BadParamter when stream is NULL.
 *  @return NvError_FileOperationFailed when closing file is failed.
 */
NvError NvOsFcloseEx(NvOsFileHandle stream);

/** Closes a file stream.
 *
 *  @param stream The file stream to close.
 *  Passing in a null handle is okay.
 */
void NvOsFclose(NvOsFileHandle stream);

/** Writes to a file stream.
 *
 *  @param stream The file stream.
 *  @param ptr A pointer to the data to write.
 *  @param size The length of the write.
 *
 *  @retval NvError_FileWriteFailed Returned on error.
 */
NvError
NvOsFwrite(NvOsFileHandle stream, const void *ptr, size_t size);

/** Reads a file stream.
 *
 *  Buffered read implementation if available for a particular OS may
 *  return corrupted data if multiple threads read from the same
 *  stream simultaneously.
 *
 *  To detect short reads (less that specified amount), pass in \a bytes
 *  and check its value to the expected value. The \a bytes parameter may
 *  be null.
 *
 *  @param stream The file stream.
 *  @param ptr A pointer to the buffer for the read data.
 *  @param size The length of the read.
 *  @param [out] bytes A pointer to the number of bytes readd; may be null.
 *
 *  @retval NvError_FileReadFailed If the file read encountered any
 *      system errors.
 *  @retval NvError_EndOfFile Indicates end of file reached. Bytes
 *      must be checked to determine whether data was read into the
 *      buffer.
 */
NvError
NvOsFread(NvOsFileHandle stream, void *ptr, size_t size, size_t *bytes);

/** Gets a character from a file stream.
 *
 *  @param stream The file stream.
 *  @param [out] c A pointer to the character from the file stream.
 *
 *  @retval NvError_EndOfFile When the end of file is reached.
 */
NvError
NvOsFgetc(NvOsFileHandle stream, NvU8 *c);

/** Changes the file position pointer.
 *
 *  @param file The file.
 *  @param offset The offset from whence to seek.
 *  @param whence The starting point for the seek.
 *
 *  @retval NvError_FileOperationFailed On error.
 */
NvError
NvOsFseek(NvOsFileHandle file, NvS64 offset, NvOsSeekEnum whence);

/** Gets the current file position pointer.
 *
 *  @param file The file.
 *  @param [out] position A pointer to the file position.
 *
 *  @retval NvError_FileOperationFailed On error.
 */
NvError
NvOsFtell(NvOsFileHandle file, NvU64 *position);

/** Gets file information.
 *
 *  @param filename A pointer to the file about which to get information.
 *  @param [out] stat A pointer to the information structure.
 */
NvError
NvOsStat(const char *filename, NvOsStatType *stat);

/** Gets file information from an already open file.
 *
 *  @param file The open file.
 *  @param [out] stat A pointer to the information structure.
 */
NvError
NvOsFstat(NvOsFileHandle file, NvOsStatType *stat);

/** Flushes any pending writes to the file stream.
 *
 *  @param stream The file stream.
 */
NvError
NvOsFflush(NvOsFileHandle stream);

/** Commits any pending writes to storage media.
 *
 *  After this completes, any pending writes are guaranteed to be on the
 *  storage media associated with the stream (if any).
 *
 *  @param stream The file stream.
 */
NvError
NvOsFsync(NvOsFileHandle stream);

/** Removes a file from the storage media. If the file is open,
 *  this function marks the file for deletion upon close.
 *
 *  @param filename A pointer to the file to remove.
 *
 *  @retval NvError_FileOperationFailed If cannot remove file.
 */
NvError
NvOsFremove(const char *filename);

/** Causes the file to have a size of length bytes. The file must
 *  be opened for writing.
 *
 *  @param stream Holds the file stream.
 *  @param length Specifies the size of file in bytes to which to truncate.
 *
 *  @retval NvError_FileOperationFailed If file is not open for writing.
 *  @retval NvError_ResourceError If I/O error occurred when writing to the file.
 */
NvError
NvOsFtruncate(NvOsFileHandle stream, NvU64 length);

/*@}*/
/** @name Directories
 */
/*@{*/

/** A handle to a directory. */
typedef struct NvOsDirRec *NvOsDirHandle;

/** Opens a directory.
 *
 *  @param path A pointer to the path of the directory to open.
 *  @param [out] dir A pointer to the directory that will be opened, if successful.
 *
 *  @retval NvError_DirOperationFailed Returned upon failure.
 */
NvError
NvOsOpendir(const char *path, NvOsDirHandle *dir);

/** Gets the next entry in the directory.
 *
 *  @param dir The directory pointer.
 *  @param [out] name A pointer to the name of the next file.
 *  @param size The size of the name buffer.
 *
 *  @retval NvError_EndOfDirList When there are no more entries in the
 *      directory.
 *  @retval NvError_DirOperationFailed If there is a system error.
 */
NvError
NvOsReaddir(NvOsDirHandle dir, char *name, size_t size);

/** Closes the directory.
 *
 *  @param dir The directory to close.
 *      Passing in a null handle is okay.
 */
void NvOsClosedir(NvOsDirHandle dir);

/** Creates the directory.
 *
 *  This is currently only implemented on Linux.
 *
 *  @param dirname A pointer to the name of the directory to create.
 *
 *  @retval NvError_PathAlreadyExists If directory could not be
 *  created because the path already exists.
 *  @retval NvSuccess If the directory was created.
 *  @return Other nonzero error code if the directory could not be
 *  created for some other reason.
 */
NvError NvOsMkdir(char *dirname);

/** Retrieves an unsigned integer variable from the config store.
 *
 * See also: <a class="el" href="#nvos_cfg_support">
 *     NvOs Configuration Data Storage Support</a>
 *
 *  @param name A pointer to the name of the variable.
 *  @param [out] value A pointer to the value to write.
 *
 *  @retval NvError_ConfigVarNotFound If the name is not found in the
 *      config store.
 *  @retval NvError_InvalidConfigVar If the configuration variable cannot
 *      be converted into an unsigned integer.
 */
NvError
NvOsGetConfigU32(const char *name, NvU32 *value);

/** Retrieves a string variable from the config store.
 *
 * See also: <a class="el" href="#nvos_cfg_support">
 *     NvOs Configuration Data Storage Support</a>
 *
 *  @param name A pointer to the name of the variable.
 *  @param value A pointer to the value to write into.
 *  @param size The size of the value buffer.
 *
 *  @retval NvError_ConfigVarNotFound If the name is not found in the
 *      config store.
 */
NvError
NvOsGetConfigString(const char *name, char *value, NvU32 size);

/** Retrieves a string variable from the config store, with
 * a persist.sys prefix.
 *
 * See also: <a class="el" href="#nvos_cfg_support">
 *     NvOs Configuration Data Storage Support</a>
 *
 *  @param name A pointer to the name of the variable.
 *  @param value A pointer to the value to write into.
 *  @param size The size of the value buffer.
 *
 *  @retval NvError_ConfigVarNotFound If the name is not found in the
 *      config store.
 */
NvError
NvOsGetSysConfigString(const char *name, char *value, NvU32 size);

/*@}*/
/** @name Memory Allocation
 */
/*@{*/

/** Dynamically allocates memory.
 *  Alignment, if desired, must be done by the caller.
 *
 *  @param size The size of the memory to allocate.
 */
void *NvOsAlloc(size_t size);

/** Dynamically allocates memory with specific alignment.
 *
 *  @param align The alignment of the memory to allocate. Must be a power of two.
 *  @param size The size of the memory to allocate, in bytes.
 */
void *NvOsAllocAlign(size_t align, size_t size);

/** Re-sizes a previous dynamic allocation.
 *
 *  @param ptr A pointer to the original allocation.
 *  @param size The new size to allocate.
 */
void *NvOsRealloc(void *ptr, size_t size);

/** Frees a dynamic memory allocation.
 *
 *  Freeing a null value is okay.
 *
 *  @param ptr A pointer to the memory to free, which should be from
 *      NvOsAlloc().
 */
void NvOsFree(void *ptr);

/** An opaque handle returned by shared memory allocations.
 */
typedef struct NvOsSharedMemRec *NvOsSharedMemHandle;

/** Dynamically allocates multiprocess shared memory.
 *
 *  The memory will be zero initialized when it is first created.
 *
 *  @param key A pointer to the global key to identify the shared allocation.
 *  @param size The size of the allocation.
 *  @param [out] descriptor A pointer to the result descriptor.
 *
 *  @return If the shared memory for \a key already exists, then this returns
 *  the already allocated shared memory; otherwise, it creates it.
 */
NvError
NvOsSharedMemAlloc(const char *key, size_t size,
    NvOsSharedMemHandle *descriptor);

/**
 *
 * Enables sharing ::NvOsSharedMemHandle memory that was created by
 * a call to NvOsSharedMemAlloc().
 *
 *  @param fd A pointer to the global key to identify the shared allocation.
 *  @param [out] descriptor A pointer to the result descriptor.
 *
 *  @retval NvError_SharedMemMapFailed on failure.
 */
NvError
NvOsSharedMemHandleFromFd(int fd, NvOsSharedMemHandle *descriptor);

/** Extracts the file descriptor from a handle.
 *
 *  @param descriptor The memory descriptor to map.
 *  @param [out] fd A pointer to the result.
 *
 *  @return A file descriptor that can be used to create a new
 *  handle.
 */
NvError
NvOsSharedMemGetFd(NvOsSharedMemHandle descriptor, int *fd);

/** Maps a shared memory region into the process virtual memory.
 *
 *  @param descriptor The memory descriptor to map.
 *  @param offset The offset in bytes into the mapped area.
 *  @param size The size area to map.
 *  @param [out] ptr A pointer to the result pointer.
 *
 *  @retval NvError_SharedMemMapFailed on failure.
 */
NvError
NvOsSharedMemMap(NvOsSharedMemHandle descriptor, size_t offset,
    size_t size, void **ptr);

/** Unmaps a mapped region of shared memory.
 *
 *  @param ptr A pointer to the pointer to virtual memory.
 *  @param size The size of the mapped region.
 */
void NvOsSharedMemUnmap(void *ptr, size_t size);

/** Frees shared memory from NvOsSharedMemAlloc().
 *
 *  It is valid to call \c NvOsSharedMemFree while mappings are still
 *  outstanding.
 *
 *  @param descriptor The memory descriptor.
 */
void NvOsSharedMemFree(NvOsSharedMemHandle descriptor);

/** Defines memory attributes. */
typedef enum
{
#ifdef INCLUDED_NVOS_TEGRA_NONSAFETY_H
    NvOsMemAttribute_Reserved           = 0,
#else
    NvOsMemAttribute_Uncached           = 0,
#endif
    NvOsMemAttribute_WriteBack          = 1,
    NvOsMemAttribute_WriteCombined      = 2,
    NvOsMemAttribute_InnerWriteBack     = 3,
    NvOsMemAttribute_Secured            = 4,
    NvOsMemAttribute_DeviceMemory       = 5,
    NvOsMemAttribute_SO_DeviceMemory    = 6,
    NvOsMemAttribute_End                = 7,

    NvOsMemAttribute_Force32 = 0x7FFFFFFF
} NvOsMemAttribute;

#ifdef INCLUDED_NVOS_TEGRA_NONSAFETY_H
#if defined(__LP64__) && !defined(__DEPRECATE_NVOS_MEM_ATTR_UNCACHED__) && (!NVOS_IS_QNX)
    #if defined(__GNUC__) && (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
        #pragma GCC diagnostic warning "-Wdeprecated-declarations"
        #define __DEPRECATE_NVOS_MEM_ATTR_UNCACHED__ __attribute__ ((deprecated(\
                    "Use NvOsMemAttribute_WriteCombined instead!!\n"\
                    "Use proper barrier protection when moved to writecombined\n"\
                    "VPR is treated as strongly ordered in kernel even if you use\n"\
                    "writecombined. So, no need for barrier protection for VPR allocations")))
        __DEPRECATE_NVOS_MEM_ATTR_UNCACHED__ static const NvOsMemAttribute NvOsMemAttribute_Uncached = NvOsMemAttribute_Reserved;
    #else
        static const NvOsMemAttribute NvOsMemAttribute_Uncached = NvOsMemAttribute_Reserved;
    #endif
#else
        static const NvOsMemAttribute NvOsMemAttribute_Uncached = NvOsMemAttribute_Reserved;
#endif
#endif

/** Specifies no memory flags. */
#define NVOS_MEM_NONE     0x0U

/** Specifies the memory may be read. */
#define NVOS_MEM_READ     0x1U

/** Specifies the memory may be written to. */
#define NVOS_MEM_WRITE    0x2U

/** Specifies the memory may be executed. */
#define NVOS_MEM_EXECUTE  0x4U

/**
 * The memory must be visible by all processes, this is only valid for
 * WinCE 5.0.
 */
#define NVOS_MEM_GLOBAL_ADDR 0x8U

/** The memory may be both read and writen. */
#define NVOS_MEM_READ_WRITE (NVOS_MEM_READ | NVOS_MEM_WRITE)

/** Specifies the memory must be mapped as fixed for mmap. */
#define NVOS_MEM_MMAP_FIXED 0x40U

/** Support MAP_LAZY for mmap in QNX. */
#define NVOS_MEM_MMAP_LAZY 0x80U

/** Support MAP_POPULATE for mmap */
#define NVOS_MEM_MMAP_POPULATE 0x100U

/** Maps computer resources into user space.
 *
 *  @param phys The physical address start.
 *  @param size The size of the aperture.
 *  @param attrib Memory attributes (caching).
 *  @param flags Bitwise OR of \c NVOS_MEM_*.
 *  @param [out] ptr A pointer to the result pointer.
 */
NvError
NvOsPhysicalMemMap(NvOsPhysAddr phys, size_t size,
    NvOsMemAttribute attrib, NvU32 flags, void **ptr);

/**
 * Releases resources previously allocated by NvOsPhysicalMemMap().
 *
 * @param ptr The virtual pointer returned by \c NvOsPhysicalMemMap. If this
 *     pointer is null, this function has no effect.
 * @param size The size of the mapped region.
 */
void NvOsPhysicalMemUnmap(void *ptr, size_t size);

/*@}*/
/** @name Page Allocator
 */
/*@{*/

/**
 *  Low-level memory allocation of the external system memory.
 */
typedef enum
{
    NvOsPageFlags_Contiguous    = 0,
    NvOsPageFlags_NonContiguous = 1,

    NvOsMemFlags_Forceword = 0x7ffffff,
} NvOsPageFlags;

typedef struct NvOsPageAllocRec *NvOsPageAllocHandle;

/** Allocates memory via the page allocator.
 *
 *  @param size The number of bytes to allocate.
 *  @param attrib Page caching attributes.
 *  @param flags Various memory allocation flags.
 *  @param protect Page protection attributes (\c NVOS_MEM_*).
 *  @param [out] descriptor A pointer to the result descriptor.
 *
 *  @return A descriptor (not a pointer to virtual memory),
 *  which may be passed into other functions.
 */
NvError
NvOsPageAlloc(size_t size, NvOsMemAttribute attrib,
    NvOsPageFlags flags, NvU32 protect, NvOsPageAllocHandle *descriptor);

/** Frees pages from NvOsPageAlloc().
 *
 *  It is not valid to call NvOsPageFree() while there are outstanding
 *  mappings.
 *
 *  @param descriptor The descriptor from \c NvOsPageAlloc.
 */
void
NvOsPageFree(NvOsPageAllocHandle descriptor);

/** Maps pages into the virtual address space.
 *
 *  Upon successful completion, \a *ptr holds a virtual address
 *  that may be accessed.
 *
 *  @param descriptor Allocated pages from NvOsPageAlloc(), etc.
 *  @param offset Offset in bytes into the page range.
 *  @param size The size of the mapping.
 *  @param [out] ptr A pointer to the result pointer.
 *
 * @retval NvSuccess If successful, or the appropriate error code.
 */
NvError
NvOsPageMap(NvOsPageAllocHandle descriptor, size_t offset, size_t size,
    void **ptr);

/** Unmaps the virtual address from NvOsPageMap().
 *
 *  @param descriptor Allocated pages from NvOsPageAlloc(), etc.
 *  @param ptr A pointer to the virtual address to unmap that was returned
 *      from \c NvOsPageMap.
 *  @param size The size of the mapping, which should match what
 *   was passed into \c NvOsPageMap.
 */
void
NvOsPageUnmap(NvOsPageAllocHandle descriptor, void *ptr, size_t size);

/** Returns the physical address given an offset.
 *
 *  This is useful for non-contiguous page allocations.
 *
 *  @param descriptor The descriptor from NvOsPageAlloc(), etc.
 *  @param offset The offset in bytes into the page range.
 */
NvOsPhysAddr
NvOsPageAddress(NvOsPageAllocHandle descriptor, size_t offset);

/*@}*/
/** @name Dynamic Library Handling
 */
/*@{*/

/** A handle to a dynamic library. */
typedef struct NvOsLibraryRec *NvOsLibraryHandle;

/** Load a dynamic library.
 *
 *  No operating system specific suffixes or paths should be used for the
 *  library name. So do not use:
 *
 *  <pre>
        /usr/lib/libnvos.so
        libnvos.dll
    </pre>
 *
 * Just use:
 *  <pre>
    libnvos
   </pre>
 *
 *  @param name A pointer to the library name.
 *  @param [out] library A pointer to the result library.
 *
 *  @retval NvError_LibraryNotFound If the library cannot be opened.
 */
NvError
NvOsLibraryLoad(const char *name, NvOsLibraryHandle *library);

/** Gets an address of a symbol in a dynamic library.
 *
 *  @param library The dynamic library.
 *  @param symbol A pointer to the symbol to lookup.
 *
 *  @return The address of the symbol, or NULL if the symbol cannot be found.
 */
void*
NvOsLibraryGetSymbol(NvOsLibraryHandle library, const char *symbol);

/** Unloads a dynamic library.
 *
 *  @param library The dynamic library to unload.
 *      It is okay to pass a null \a library value.
 */
void NvOsLibraryUnload(NvOsLibraryHandle library);

/** Unloads a dynamic library.
 *
 *  @param library The dynamic library to unload.
 *
 *  @return NvSuccess if successful.
 *  @return NvError_BadParamter when library is NULL.
 *  @return NvError_LibraryNotFound when unlaoding library is failed.
 */
NvError NvOsLibraryUnloadEx(NvOsLibraryHandle library);

/*@}*/
/** @name Syncronization Objects and Thread Management
 */
/*@{*/

typedef struct NvOsMutexRec *NvOsMutexHandle;
typedef struct NvOsIntrMutexRec *NvOsIntrMutexHandle;
typedef struct NvOsConditionRec *NvOsConditionHandle;
typedef struct NvOsSemaphoreRec *NvOsSemaphoreHandle;
typedef struct NvOsThreadRec *NvOsThreadHandle;

/** Unschedules the calling thread for at least the given
 *      number of milliseconds.
 *
 *  Other threads may run during the sleep time.
 *
 *  @param msec The number of milliseconds to sleep.
 */
void
NvOsSleepMS(NvU32 msec);

/** Stalls the calling thread for at least the given number of
 *  microseconds. The actual time waited might be longer; you cannot
 *  depend on this function for precise timing.
 *
 *  @note It is safe to use this function at ISR time.
 *
 *  @param usec The number of microseconds to wait.
 */
void NvOsWaitUS(NvU32 usec);

NvError NvOsSleepUS(NvU32 usec);
NvError NvOsSleepNS(NvU32 nsec);

/** Stalls the calling thread for at least the given number of
 *  microseconds. The actual time waited might be longer; you cannot
 *  depend on this function for precise timing.
 *
 *  @note It is safe to use this function at ISR time.
 *
 *  @param usec The number of microseconds to wait.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsBusyWaitUS(NvU32 usec);
NvError NvOsBusyWaitNS(NvU32 nsec);

/**
 * Allocates a new process-local mutex.
 *
 * @note Mutexes can be locked recursively; if a thread owns the lock,
 * it can lock it again as long as it unlocks it an equal number of times.
 *
 * @param mutex The mutex to initialize.
 *
 * @return \a NvError_MutexCreateFailed, or one of common error codes on
 * failure.
 */
NvError NvOsMutexCreate(NvOsMutexHandle *mutex);

/** Locks the given unlocked mutex.
 *
 *  If a process is holding a lock on a multi-process mutex when it terminates,
 *  this lock will be automatically released.
 *
 *  @param mutex The mutex to lock; note that this is a recursive lock.
 */
void NvOsMutexLock(NvOsMutexHandle mutex);

/** Locks the given unlocked mutex.
 *
 *  If a process is holding a lock on a multi-process mutex when it terminates,
 *  this lock will be automatically released.
 *
 *  @param mutex The mutex to lock; note that this is a recursive lock.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsMutexLockEx(NvOsMutexHandle mutex);

/** Unlocks a locked mutex.
 *
 *  A mutex must be unlocked exactly as many times as it has been locked.
 *
 *  @param mutex The mutex to unlock.
 */
void NvOsMutexUnlock(NvOsMutexHandle mutex);

/** Unlocks a locked mutex.
 *
 *  A mutex must be unlocked exactly as many times as it has been locked.
 *
 *  @param mutex The mutex to unlock.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsMutexUnlockEx(NvOsMutexHandle mutex);

/** Frees the resources held by a mutex.
 *
 *  Mutecies are reference counted and a given mutex will not be destroyed
 *  until the last reference has gone away.
 *
 *  @param mutex The mutex to destroy. Passing in a null mutex is okay.
 */
void NvOsMutexDestroy(NvOsMutexHandle mutex);

/** Frees the resources held by a mutex.
 *
 *  Mutecies are reference counted and a given mutex will not be destroyed
 *  until the last reference has gone away.
 *
 *  @param mutex The mutex to destroy.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsMutexDestroyEx(NvOsMutexHandle mutex);

/**
 * Creates a mutex that is safe to aquire in an ISR.
 *
 * @param mutex A pointer to the mutex is stored here on success.
 */
NvError NvOsIntrMutexCreate(NvOsIntrMutexHandle *mutex);

/**
 * Aquire an ISR-safe mutex.
 *
 * @param mutex The mutex to lock. For kernel (OAL) implementations,
 *     NULL implies the system-wide lock will be used.
 */
void NvOsIntrMutexLock(NvOsIntrMutexHandle mutex);

/**
 * Releases an ISR-safe mutex.
 *
 * @param mutex The mutex to unlock. For kernel (OAL) implementations,
 *     NULL implies the system-wide lock will be used.
 */
void NvOsIntrMutexUnlock(NvOsIntrMutexHandle mutex);

/**
 * Destroys an ISR-safe mutex.
 *
 * @param mutex The mutex to destroy. If \a mutex is NULL, this API has no
 *     effect.
 */
void NvOsIntrMutexDestroy(NvOsIntrMutexHandle mutex);

/**
 * Creates a condition variable.
 *
 * @param cond A pointer to the condition variable to initialize.
 *
 * @retval NvSuccess if successful, or the appropriate error code.
 * @retval NvError_InsufficientMemory if insufficient memory exists.
 * @retval NvError_Busy if there are insufficient resources.
 */
NvError NvOsConditionCreate(NvOsConditionHandle *cond);

/**
 * Frees resources held by the condition variable.
 *
 * @param cond The condition variable to destroy.
 *
 * @retval NvSuccess if successful.
 * @retval NvError_BadParameter if condition variable is not valid.
 * @retval NvError_Busy if the condition variable is currently being used in
 *     another thread.
 */
NvError NvOsConditionDestroy(NvOsConditionHandle cond);

/**
 * Unblocks all threads currently blocked on the specified condition variable.
 *
 * @param cond The condition variable on which other threads may be blocked.
 *
 * @retval NvError_BadParameter if the condition variable is not valid.
 */
NvError NvOsConditionBroadcast(NvOsConditionHandle cond);

/**
 * Unblocks atleast one of the threads that are blocked on the specified
 *    condition variable.
 *
 * @param cond The condition variable on which other threads may be blocked.
 *
 * @retval NvError_BadParameter if condition variable is not valid.
 */
NvError NvOsConditionSignal(NvOsConditionHandle cond);

/**
 * Atomically releases the mutex and causes the calling thread to block on
 *    the condition variable. This function must be called with mutex locked
 *    by the calling thread or undefined behaviour results. Upon successful
 *    return, the mutex shall have been locked and be owned by the calling
 *    thread.
 *
 *  @param cond The condition variable to wait on.
 *  @param mutex The associated mutex that needs to be locked by the calling
 *      thread.
 *
 *  @retval NvSuccess if successful.
 *  @retval NVError_AccessDenied if the mutex was not owned by the calling
 *      thread.
 * @retval NvError_BadParameter if condition variable or mutex is not valid.
 */
NvError NvOsConditionWait(NvOsConditionHandle cond, NvOsMutexHandle mutex);

/**
 * Atomically releases the mutex and causes the calling thread to block on
 *     the condition variable with a timeout. Must be called with mutex locked
 *     by the calling thread or undefined behaviour results. Upon successful
 *     return, the mutex shall have been locked and be owned by the calling
 *     thread
 *
 *  @param cond The condition variable to wait on.
 *  @param mutex The associated mutex that needs to be locked by the calling
 *     thread.
 *  @param microsecs Time specified in microseconds for timeout to occur.
 *
 *  @retval NvSuccess if successful.
 *  @retval NVError_AccessDenied if the mutex was not owned by the calling
 *      thread.
 *  @retval NVError_Timeout if the time specified by microsecs has passed
 *      before the condition variable is signalled  or broadcast.
 * @retval NvError_BadParameter if condition variable or mutex is not valid.
 */
NvError NvOsConditionWaitTimeout(NvOsConditionHandle cond, NvOsMutexHandle mutex, NvU32 microsecs);

/**
 * Creates a counting semaphore.
 *
 * @param semaphore A pointer to the semaphore to initialize.
 * @param value The initial semaphore value.
 *
 * @retval NvSuccess If successful, or the appropriate error code.
 */
NvError
NvOsSemaphoreCreate(NvOsSemaphoreHandle *semaphore, NvU32 value);

/**
 * Creates a duplicate semaphore from the given semaphore.
 * Freeing the original semaphore has no effect on the new semaphore.
 *
 * @param orig The semaphore to duplicate.
 * @param semaphore A pointer to the new semaphore.
 *
 * @retval NvSuccess If successful, or the appropriate error code.
 */
NvError
NvOsSemaphoreClone( NvOsSemaphoreHandle orig,  NvOsSemaphoreHandle *semaphore);

/** Waits until the semaphore value becomes non-zero, then
 *  decrements the value and returns.
 *
 *  @param semaphore The semaphore to wait for.
 */
void NvOsSemaphoreWait(NvOsSemaphoreHandle semaphore);

/** Waits until the semaphore value becomes non-zero, then
 *  decrements the value and returns.
 *
 *  @param semaphore The semaphore to wait for.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsSemaphoreWaitEx(NvOsSemaphoreHandle semaphore);

/**
 * Waits for the given semaphore value to become non-zero with timeout. If
 * the semaphore value becomes non-zero before the timeout, then the value is
 * decremented and \a NvSuccess is returned.
 *
 * @param semaphore The semaphore to wait for.
 * @param msec Timeout value in milliseconds.
 *     \c NV_WAIT_INFINITE can be used to wait forever.
 *
 * @retval NvError_Timeout If the wait expires.
 */
NvError
NvOsSemaphoreWaitTimeout(NvOsSemaphoreHandle semaphore, NvU32 msec);

/** Increments the semaphore value.
 *
 *  @param semaphore The semaphore to signal.
 */
void NvOsSemaphoreSignal(NvOsSemaphoreHandle semaphore);

/** Increments the semaphore value.
 *
 *  @param semaphore The semaphore to signal.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsSemaphoreSignalEx(NvOsSemaphoreHandle semaphore);

/** Frees resources held by the semaphore.
 *
 *  Semaphores are reference counted across the computer (multiproceses),
 *  and a given semaphore will not be destroyed until the last reference has
 *  gone away.
 *
 *  @param semaphore The semaphore to destroy.
 *      Passing in a null semaphore is okay (no op).
 */
void NvOsSemaphoreDestroy(NvOsSemaphoreHandle semaphore);

/** Frees resources held by the semaphore.
 *
 *  Semaphores are reference counted across the computer (multiproceses),
 *  and a given semaphore will not be destroyed until the last reference has
 *  gone away.
 *
 *  @param semaphore The semaphore to destroy.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsSemaphoreDestroyEx(NvOsSemaphoreHandle semaphore);

typedef enum
{
    NvOsThreadPriorityType_Normal = 1,
    NvOsThreadPriorityType_NearInterrupt = 2,
    // NvOsThreadPriorityType_Native is ONLY valid when used with NvOsThreadCreateWithAttr
    // Using this value means that the priority of thread will be set to "nativePriority"
    NvOsThreadPriorityType_Native = 3,

    NvOsThreadPriorityType_Last = 4,
    NvOsThreadPriorityType_Force32 = 0x7FFFFFFF
} NvOsThreadPriorityType;

/** Entry point for a thread.
 */
typedef void (*NvOsThreadFunction)(void *args);

/** Creates a thread.
 *
 *  @param function The thread entry point.
 *  @param args A pointer to the thread arguments.
 *  @param [out] thread A pointer to the result thread ID structure.
 */
NvError
NvOsThreadCreate( NvOsThreadFunction function, void *args,
    NvOsThreadHandle *thread);

/** Creates a near interrupt priority thread.
 *
 *  @param function The thread entry point.
 *  @param args A pointer to the thread arguments.
 *  @param [out] thread A pointer to the result thread ID structure.
 */
NvError
NvOsInterruptPriorityThreadCreate( NvOsThreadFunction function, void *args,
    NvOsThreadHandle *thread);

/** Returns the name of the given thread.
 *
 *  @param thread The thread whose name needs to be returned.
 */
const char*
NvOsThreadGetName( NvOsThreadHandle thread);

/** Assigns the given name to the given thread.
 *
 *  @param thread The thread to assign the name to.
 *  @param name String codifying the thread name.
 */
NvError
NvOsThreadSetName( NvOsThreadHandle thread, const char *name);

/** Waits for the given thread to exit.
 *
 *  The joined thread will be destroyed automatically. All OS resources
 *  will be reclaimed. There is no method for terminating a thread
 *  before it exits naturally.
 *
 *  @param thread The thread to wait for.
 *  Passing in a null thread ID is okay (no op).
 */
void NvOsThreadJoin(NvOsThreadHandle thread);

/** Waits for the given thread to exit.
 *
 *  The joined thread will be destroyed automatically. All OS resources
 *  will be reclaimed. There is no method for terminating a thread
 *  before it exits naturally.
 *
 *  @param thread The thread to wait for.
 *
 *  @retval NvSuccess If successful, or the appropriate error code.
 */
NvError NvOsThreadJoinEx(NvOsThreadHandle thread);

/** Yields to another runnable thread.
 */
void NvOsThreadYield(void);

/**
 * Returns current thread ID.
 *
 *  @retval ThreadId
 */

NvU64 NvOsGetCurrentThreadId(void);

/**
 * Atomically compares the contents of a pointer-sized memory location
 * with a value, and if they match, updates it to a new value. This
 * function is the equivalent of the following code, except that other
 * threads or processors are effectively prevented from reading or
 * writing \a *pTarget while we are inside the function.
 *
 * @code
 * void *OldTarget = *pTarget;
 * if (OldTarget == OldValue)
 *     *pTarget = NewValue;
 * return OldTarget;
 * @endcode
 */
void *NvOsAtomicCompareExchangePtr(void **pTarget, void *OldValue,
                                   void *NewValue);

/**
 * Atomically compares the contents of a 32-bit memory location with a value,
 * and if they match, updates it to a new value. This function is the
 * equivalent of the following code, except that other threads or processors
 * are effectively prevented from reading or writing \a *pTarget while we are
 * inside the function.
 *
 * @code
 * NvS32 OldTarget = *pTarget;
 * if (OldTarget == OldValue)
 *     *pTarget = NewValue;
 * return OldTarget;
 * @endcode
 */
NvS32 NvOsAtomicCompareExchange32(NvS32 *pTarget, NvS32 OldValue, NvS32
    NewValue);

/**
 * Atomically swaps the contents of a 32-bit memory location with a value. This
 * function is the equivalent of the following code, except that other threads
 * or processors are effectively prevented from reading or writing \a *pTarget
 * while we are inside the function.
 *
 * @code
 * NvS32 OldTarget = *pTarget;
 * *pTarget = Value;
 * return OldTarget;
 * @endcode
 */
NvS32 NvOsAtomicExchange32(NvS32 *pTarget, NvS32 Value);

/**
 * Atomically increments the contents of a 32-bit memory location by a specified
 * amount. This function is the equivalent of the following code, except that
 * other threads or processors are effectively prevented from reading or
 * writing \a *pTarget while we are inside the function.
 *
 * @code
 * NvS32 OldTarget = *pTarget;
 * *pTarget = OldTarget + Value;
 * return OldTarget;
 * @endcode
 */
NvS32 NvOsAtomicExchangeAdd32(NvS32 *pTarget, NvS32 Value);

/**
 * A NvOsTlsKey Handle. On Integrity \c pthread_key_t is a pointer type,
 * whereas it is an unsigned int on other platforms.
 *
 * For Integrity case, NvOsTlsKeyHandle will be typedef'ed to NvUPtr later
 * in a subsequent change.
 */

#if NVOS_IS_INTEGRITY
typedef NvU64 NvOsTlsKeyHandle;
#else
typedef NvU32 NvOsTlsKeyHandle;
#endif
/** A TLS index that is guaranteed to be invalid. */
#define NVOS_INVALID_TLS_INDEX 0xFFFFFFFF
#define NVOS_TLS_CNT            4

/**
 * Allocates a thread-local storage variable. All TLS variables have initial
 * value NULL in all threads when first allocated.
 *
 * @returns The TLS index of the TLS variable if successful, or
 *     ::NVOS_INVALID_TLS_INDEX if not.
 */
NvOsTlsKeyHandle NvOsTlsAlloc(void);

/**
 * Allocates a thread-local storage variable, with destructor when thread exits.
 * All TLS variables have initial value NULL in all threads when first allocated.
 *
 * @param destructor Callback to invoke when thread is destroyed.
 * @returns The TLS index of the TLS variable if successful, or
 *     ::NVOS_INVALID_TLS_INDEX if not.
 */
NvOsTlsKeyHandle NvOsTlsAllocWithDestructor(void (*destructor)(void*));

/**
 * Frees a thread-local storage variable.
 *
 * @param TlsIndex The TLS index of the TLS variable. This function is a no-op
 *     if TlsIndex equals ::NVOS_INVALID_TLS_INDEX.
 */
void NvOsTlsFree(NvOsTlsKeyHandle TlsIndex);

/**
 * Gets the value of a thread-local storage variable.
 *
 * @param TlsIndex The TLS index of the TLS variable.
 *     The current value of the TLS variable is returned.
 */
void *NvOsTlsGet(NvOsTlsKeyHandle TlsIndex);

/**
 * Sets the value of a thread-local storage variable.
 *
 * @param TlsIndex The TLS index of the TLS variable.
 * @param Value A pointer to the new value of the TLS variable.
 */
void NvOsTlsSet(NvOsTlsKeyHandle TlsIndex, void *Value);

/**
 * Registers a function that should be called when the thread terminates
 * to clean up any structures stored in thread-local storage.
 *
 * @param func The callback function called when the thread terminates.
 * @param context A pointer to the data to pass to the callback function.
 */
NvError NvOsTlsAddTerminator(void (*func)(void *), void *context);

/**
 * Checks the list of existing terminator functions for one that matches
 * both the function and the data, and then removes that function from the
 * list.
 *
 * @param func The callback function called when the thread terminates.
 * @param context A pointer to the data to pass to the callback function.
 * @returns True if the terminator exists and was removed, otherwise false if not.
 */
NvBool NvOsTlsRemoveTerminator(void (*func)(void *), void *context);

/*@}*/
/** @name Time Functions
 */
/*@{*/

/**
 * @brief Defines the system time structure.
 */
typedef struct NvOsSystemTimeRec
{
    NvU32   Seconds;
    NvU32   Milliseconds;
} NvOsSystemTime;

/** Gets system realtime clock.
 *
 * @param hNvOsSystemtime A pointer to the system handle used to set the time.
 */
NvError
NvOsGetSystemTime(NvOsSystemTime *hNvOsSystemtime);

/** @return The system time in milliseconds.
 *
 *  The returned values are guaranteed to be monotonically increasing,
 *  but may wrap back to zero (after about 50 days of runtime).
 *
 *  In some systems, this is the number of milliseconds since power-on,
 *  or may actually be an accurate date.
 */
NvU32
NvOsGetTimeMS(void);

/** @return The system time in microseconds.
 *
 *  The returned values are guaranteed to be monotonically increasing,
 *  but may wrap back to zero.
 *
 *  Some systems cannot gauantee a microsecond resolution timer.
 *  Even though the time returned is in microseconds, it is not gaurnateed
 *  to have micro-second resolution.
 *
 *  Please be advised that this API is mainly used for code profiling and
 *  meant to be used direclty in driver code.
 */
NvU64
NvOsGetTimeUS(void);

/** @return The system time in nanoseconds.
 *
 *  The returned values are guaranteed to be monotonically increasing,
 *  but may wrap back to zero.
 *
 *  Some systems cannot gauantee a nanosecond resolution timer.
 *  Even though the time returned is in nanoseconds, it is not guaranteed
 *  to have nano-second resolution.
 *
 *  Please be advised that this API is mainly used for code profiling and
 *  meant to be used direclty in driver code.
 */
NvU64
NvOsGetTimeNS(void);

/*@}*/
/** @name CPU Cache
 *  Cache operations for both instruction and data cache, implemented
 *  per processor.
 */
/*@{*/

/** Writes back the entire data cache.
 */
void
NvOsDataCacheWriteback(void);

/** Writes back and invalidates the entire data cache.
 */
void
NvOsDataCacheWritebackInvalidate(void);

/** Writes back a range of the data cache.
 *
 *  @param start A pointer to the start address.
 *  @param length The number of bytes to write back.
 */
void
NvOsDataCacheWritebackRange(void *start, NvU32 length);

/** Writes back and invlidates a range of the data cache.
 *
 *  @param start A pointer to the start address.
 *  @param length The number of bytes to write back.
 */
void
NvOsDataCacheWritebackInvalidateRange(void *start, NvU32 length);

/** Invalidates the entire instruction cache.
 */
void
NvOsInstrCacheInvalidate(void);

/** Invalidates a range of the instruction cache.
 *
 *  @param start A pointer to the start address.
 *  @param length The number of bytes.
 */
void
NvOsInstrCacheInvalidateRange(void *start, NvU32 length);

/** Checks and returns whether the given memory address is within
 *  memory of given attribute.
 *
 *  @param Attrib Memory attribute to check.
 *  @param Addr Address of the memory, which is to be checked.
 *  @retval NV_TRUE if memory pointed by \a Addr is of given attribute,
 *          NV_FASLE otherwise.
 */
NvBool
NvOsIsMemoryOfGivenType(NvOsMemAttribute Attrib, NvU32 Addr);

/** Flushes the CPU's write combine buffer.
 */
void
NvOsFlushWriteCombineBuffer(void);

/** Interrupt handler function.
 */
typedef void (*NvOsInterruptHandler)(void *args);

/** Interrupt handler type.
 */
typedef struct NvOsInterruptRec *NvOsInterruptHandle;

/**
 * Registers the interrupt handler with the IRQ number.
 *
 * @param IrqListSize Size of the \a IrqList passed in for registering the IRQ
 *      handlers for each IRQ number.
 * @param pIrqList Array of IRQ numbers for which interupt handlers are to be
 *     registerd.
 * @param pIrqHandlerList A pointer to an array of interrupt routines to be
 *      called when an interrupt occurs.
 * @param context A pointer to the register's context handle.
 * @param handle A pointer to the interrupt handle.
 * @param InterruptEnable If true, immediately enable interrupt.  Otherwise
 *      enable interrupt only after calling NvOsInterruptEnable().
 *
 * @retval NvError_IrqRegistrationFailed If the interrupt is already registered.
 * @retval NvError_BadParameter If the IRQ number is not valid.
 */
NvError
NvOsInterruptRegister(NvU32 IrqListSize,
    const NvU32 *pIrqList,
    const NvOsInterruptHandler *pIrqHandlerList,
    void *context,
    NvOsInterruptHandle *handle,
    NvBool InterruptEnable);

/**
 * Unregisters the interrupt handler from the associated IRQ number.
 *
 * @note This function is intended to @b only be called
 *       from NvOsInterruptUnregister().
 *
 * @param handle interrupt Handle returned when a successfull call is made to
 *     NvOsInterruptRegister().
 */
void
NvOsInterruptUnregister(NvOsInterruptHandle handle);

/**
 * Enables the interrupt handler with the IRQ number.
 *
 * @note This function is intended to @b only be called
 *       from NvOsInterruptRegister().
 *
 * @param handle Interrupt handle returned when a successfull call is made to
 *     \c NvOsInterruptRegister.
 *
 * @retval NvError_BadParameter If the handle is not valid.
 * @retval NvError_InsufficientMemory If interrupt enable failed.
 * @retval NvSuccess If interrupt enable is successful.
 */
NvError
NvOsInterruptEnable(NvOsInterruptHandle handle);

/**
 *  Called when the ISR/IST is done handling the interrupt.
 *
 * @param handle Interrupt handle returned when a successfull call is made to
 *     NvOsInterruptRegister().
 */
void
NvOsInterruptDone(NvOsInterruptHandle handle);

/**
 * Mask/unmask an interrupt.
 *
 * Drivers can use this API to fend off interrupts. Mask means no interrupts
 * are forwarded to the CPU. Unmask means, interrupts are forwarded to the
 * CPU. In case of SMP systems, this API masks the interrutps to all the CPUs,
 * not just the calling CPU.
 *
 * @param handle    Interrupt handle returned by NvOsInterruptRegister().
 * @param mask      NV_FALSE to forward the interrupt to CPU; NV_TRUE to
 *     mask the interrupts to CPU.
 */
void NvOsInterruptMask(NvOsInterruptHandle handle, NvBool mask);

struct NvOsAllocatorRec;
typedef void* (*NvOsAllocFunc)(struct NvOsAllocatorRec *, size_t size);
typedef void* (*NvOsAllocAlignFunc)(struct NvOsAllocatorRec *, size_t align, size_t size);
typedef void* (*NvOsReallocFunc)(struct NvOsAllocatorRec *, void *ptr, size_t size);
typedef void  (*NvOsFreeFunc)(struct NvOsAllocatorRec *, void *ptr);

typedef struct NvOsAllocatorRec {
    NvOsAllocFunc Alloc;
    NvOsAllocAlignFunc AllocAlign;
    NvOsReallocFunc Realloc;
    NvOsFreeFunc Free;
} NvOsAllocator;

struct NvOsAllocatorRec *NvOsGetCurrentAllocator(void);


/*
 * Debug support.
 */
#if NV_DEBUG

// The first parameter of the debug allocator functions is a function pointer to the
// actual allocator function.
void *NvOsAllocLeak(NvOsAllocator *, size_t size, const char *f, int l);
void *NvOsAllocAlignLeak(NvOsAllocator *, size_t align, size_t size, const char *f, int l);
void *NvOsReallocLeak(NvOsAllocator *, void *ptr, size_t size, const char *f, int l);
void NvOsFreeLeak(NvOsAllocator *, void *ptr, const char *f, int l);

#define NvOsAlloc(size)             NvOsAllocLeak(NvOsGetCurrentAllocator(), size, __FILE__, __LINE__)
#define NvOsAllocAlign(align, size) NvOsAllocAlignLeak(NvOsGetCurrentAllocator(), align, size, __FILE__, __LINE__)
#define NvOsRealloc(ptr, size)      NvOsReallocLeak(NvOsGetCurrentAllocator(), ptr, size, __FILE__, __LINE__)
#define NvOsFree(ptr)               NvOsFreeLeak(NvOsGetCurrentAllocator(), ptr, __FILE__, __LINE__)

#endif /* NVOS_DEBUG */


#if NV_DEBUG

/**
 * Sets the file and line corresponding to a resource allocation.
 * Call will fill file and line for the most recently stored
 * allocation location, if not already set.
 *
 * @param userptr A pointer to used by client to identify resource.
 *     Can be NULL, which leads to no-op.
 * @param file A pointer to the name of the file from which allocation
 *     originated. Value cannot be NULL; use "" for an empty string.
 * @param l The line.
 */
void NvOsSetResourceAllocFileLine(void* userptr, const char* file, int line);

static NV_INLINE NvError
NvOsSharedMemAllocTraced(const char *key, size_t size,
    NvOsSharedMemHandle *descriptor, const char *f, int l )
{
    NvError status;
    status = (NvOsSharedMemAlloc)(key, size, descriptor);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*descriptor, f, l);
    return status;
}

static NV_INLINE NvError
NvOsSharedMemHandleFromFdTraced(int fd,
    NvOsSharedMemHandle *descriptor, const char *f, int l )
{
    NvError status;
    status = (NvOsSharedMemHandleFromFd)(fd, descriptor);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*descriptor, f, l);
    return status;
}

static NV_INLINE NvError
NvOsPhysicalMemMapTraced(NvOsPhysAddr phys, size_t size,
    NvOsMemAttribute attrib, NvU32 flags, void **ptr, const char *f, int l )
{
    NvError status;
    status = (NvOsPhysicalMemMap)(phys, size, attrib, flags, ptr);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*ptr, f, l);
    return status;
}

static NV_INLINE NvError
NvOsPageAllocTraced(size_t size, NvOsMemAttribute attrib,
    NvOsPageFlags flags, NvU32 protect, NvOsPageAllocHandle *descriptor,
    const char *f, int l )
{
    NvError status;
    status = (NvOsPageAlloc)(size, attrib, flags, protect, descriptor);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*descriptor, f, l);
    return status;
}

static NV_INLINE NvError
NvOsMutexCreateTraced(NvOsMutexHandle *mutex, const char *f, int l )
{
    NvError status;
    status = (NvOsMutexCreate)(mutex);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*mutex, f, l);
    return status;
}

static NV_INLINE NvError
NvOsIntrMutexCreateTraced(NvOsIntrMutexHandle *mutex, const char *f, int l )
{
    NvError status;
    status = (NvOsIntrMutexCreate)(mutex);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*mutex, f, l);
    return status;
}

static NV_INLINE NvError
NvOsSemaphoreCreateTraced( NvOsSemaphoreHandle *semaphore,  NvU32 value,
    const char *f, int l )
{
    NvError status;
    status = (NvOsSemaphoreCreate)(semaphore, value);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*semaphore, f, l);
    return status;
}

static NV_INLINE NvError
NvOsSemaphoreCloneTraced( NvOsSemaphoreHandle orig, NvOsSemaphoreHandle *clone,
    const char *f, int l )
{
    NvError status;
    status = (NvOsSemaphoreClone)(orig, clone);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*clone, f, l);
    return status;
}

static NV_INLINE NvError
NvOsThreadCreateTraced( NvOsThreadFunction function, void *args,
    NvOsThreadHandle *thread, const char *f, int l )
{
    NvError status;
    status = (NvOsThreadCreate)(function, args, thread);
    if (status == NvSuccess)
        NvOsSetResourceAllocFileLine(*thread, f, l);
    return status;
}

static NV_INLINE NvError
NvOsInterruptRegisterTraced(NvU32 IrqListSize, const NvU32 *pIrqList,
    const NvOsInterruptHandler *pIrqHandlerList, void *context,
    NvOsInterruptHandle *handle, NvBool InterruptEnable, const char *f, int l )
{
    NvError status;
    (void) f; /* potentially unused parameter */
    (void) l; /* potentially unused parameter */
    status = (NvOsInterruptRegister)(IrqListSize, pIrqList, pIrqHandlerList,
        context, handle, InterruptEnable);
    return status;
}

#define NvOsSharedMemAlloc(key, size, descriptor) \
    NvOsSharedMemAllocTraced(key, size, descriptor, __FILE__, __LINE__)
#define NvOsSharedMemHandleFromFd(fd, descriptor) \
    NvOsSharedMemHandleFromFdTraced(fd, descriptor, __FILE__, __LINE__)
#define NvOsPhysicalMemMap(phys, size, attrib, flags, ptr)   \
    NvOsPhysicalMemMapTraced(phys, size, attrib, flags, ptr, \
            __FILE__, __LINE__)
#define NvOsPageAlloc(size, attrib, flags, protect, descriptor)   \
    NvOsPageAllocTraced(size, attrib, flags, protect, descriptor, \
            __FILE__, __LINE__)
#define NvOsMutexCreate(mutex) NvOsMutexCreateTraced(mutex, __FILE__, __LINE__)
#define NvOsIntrMutexCreate(mutex) \
    NvOsIntrMutexCreateTraced(mutex, __FILE__, __LINE__)
#define NvOsSemaphoreCreate(semaphore, value)   \
    NvOsSemaphoreCreateTraced(semaphore, value, __FILE__, __LINE__)
#define NvOsSemaphoreClone(orig, semaphore)   \
    NvOsSemaphoreCloneTraced(orig, semaphore, __FILE__, __LINE__)
#define NvOsThreadCreate(func, args, thread)    \
    NvOsThreadCreateTraced(func, args, thread, __FILE__, __LINE__)
#define NvOsInterruptRegister(IrqListSize, pIrqList, pIrqHandlerList, \
        context, handle, InterruptEnable) \
    NvOsInterruptRegisterTraced(IrqListSize, pIrqList, pIrqHandlerList, \
        context, handle, InterruptEnable, __FILE__, __LINE__)

#endif // NVOS_DEBUG

// Forward declare resource tracking struct.
typedef struct NvCallstackRec     NvCallstack;

typedef enum
{
    NvOsCallstackType_NoStack = 1,
    NvOsCallstackType_HexStack = 2,
    NvOsCallstackType_SymbolStack = 3,

    NvOsCallstackType_Last = 4,
    NvOsCallstackType_Force32 = 0x7FFFFFFF
} NvOsCallstackType;

typedef void (*NvOsDumpCallback)(void* context, const char* line);

void NvOsDumpToDebugPrintf(void* context, const char* line);

NvCallstack* NvOsCallstackCreate      (NvOsCallstackType stackType);
void         NvOsCallstackDestroy     (NvCallstack* callstack);
NvU32        NvOsCallstackGetHeight   (NvCallstack* stack);
void         NvOsCallstackGetFrame    (char* buf, NvU32 len, NvCallstack* stack, NvU32 level);
void         NvOsCallstackDump        (NvCallstack* stack, NvU32 skip, NvOsDumpCallback callBack, void* context);
NvU32        NvOsCallstackHash        (NvCallstack* stack);
NvBool       NvOsCallstackContainsPid (NvCallstack* stack, NvU32 pid);
void         NvOsGetProcessInfo       (char* buf, NvU32 len);

static NV_INLINE void NvOsDebugCallstack(NvU32 skip)
{
    NvCallstack* stack = NvOsCallstackCreate(NvOsCallstackType_SymbolStack);
    if (stack)
    {
        NvOsCallstackDump(stack, skip, NvOsDumpToDebugPrintf, NULL);
        NvOsCallstackDestroy(stack);
    }
}

/**
 * To be deprecated.
 *
 * Sets a frame rate target that matches the CPU frequency scaling to the
 * measured performance.
 *
 * @param target  The fps target.
 * @return A file descriptor that, on exiting, you pass to the NvOsCancelFpsTarget function.
 */
int NvOsSetFpsTarget(int target);

/**
 * To be deprecated.
 *
 * Sets the frame rate target with an existing file descriptor (fd).
 *
 * @param fd      An open fd returned by the NvOsSetFpsTarget() function.
 * @param target  The fps target.
 */
int NvOsModifyFpsTarget(int fd, int target);

/**
 * To be deprecated.
 *
 * Cancels the fps target previously supplied with the NvOsSetFpsTarget() function.
 *
 * @param fd The file descriptor generated by \c NvOsSetFpsTarget.
 */
void NvOsCancelFpsTarget(int fd);


#if NVOS_IS_HOS
/**
 * HOS doesn't have ioctl/devctl - we emulate similar semantics using
 * NvOsDrvOpen, NvOsDrvClose and NvOsDrvIoctl.
 */
typedef int NvOsDrvHandle;

/**
 * Opens a driver node, e.g. /dev/nvmap.  Returns a handle that can be passed
 * to NvOsDrvIoctl and must be closed with NvOsDrvClose.
 */
NvOsDrvHandle NvOsDrvOpen(const char *pathname);

/**
 * Closes an NvOsDrvHandle.
 */
void NvOsDrvClose(NvOsDrvHandle);

/**
 * Performs an IOCTL on the NvOsDrvHandle.
 * Accepts a single IN/OUT/INOUT buffer.
 */
NvError NvOsDrvIoctl(NvOsDrvHandle, unsigned int iocode, void *arg, size_t argsize);

/**
 * Performs an IOCTL on the NvOsDrvHandle.
 * Accepts a single buffer IN/OUT/INOUT + single buffer IN.
 */
NvError NvOsDrvIoctl2(NvOsDrvHandle, unsigned int iocode, void *arg, size_t argsize, const void *inbuf, size_t inbufsize);

/**
 * Performs an IOCTL on the NvOsDrvHandle.
 * Accepts a single buffer IN/OUT/INOUT + single buffer OUT.
 */
NvError NvOsDrvIoctl3(NvOsDrvHandle, unsigned int iocode, void *arg, size_t argsize, void *outbuf, size_t outbufsize);

/**
 * Sets up memory associated with hMem for sharing.
 */
NvError NvOsDrvMapSharedMem(NvOsDrvHandle, unsigned int handle, unsigned int hMem);

/** Creates a pool of events.
 *
 *  An event pool is comprised of one or more events that can be collectively
 *  waited upon.  Supported event types include:
 *
 *  - system event---triggered when system-level action (e.g., hardware error) occurs.
 *  - timer event---triggered when specified timeout expires.
 *  - user event---triggered when user-defined action occurs.
 *
 *  @param numEvents Specifies the maximum number of events in the pool.
 *  @param eventPool Returns event pool on success.
 *
 *  @retval NvError_InsufficientResources If there is no memory for pool.
 */
typedef struct NvOsEventPoolRec NvOsEventPool;
typedef struct NvOsEventRec NvOsEvent;

NvError NvOsEventPoolCreate(NvU32 numEvents, NvOsEventPool **eventPool);

/** Destroys specified event pool.
 *
 *  @param eventPool Event pool to destroy.
 */
void NvOsEventPoolDestroy(NvOsEventPool *eventPool);

/** Links event into specified pool.
 *
 * When an event has been successfully linked into a pool then subsequent
 * calls to NvOsEventPoolWait will include any signals of that event.
 *
 *  @param eventPool Event pool in which to link event.
 *  @param event Event to link into specified pool.
 *
 *  @retval NvError_InsufficientResources Returned if no room for event in pool.
 */
NvError NvOsEventPoolLink(NvOsEventPool *eventPool, NvOsEvent *event);

/** Waits on event(s) in pool.
 *
 *  @param eventPool Event pool on which to wait.
 *  @param timeoutUs Duration in microseconds before wait times out.
 *  @param event Returns event that was signaled.
 */
void NvOsEventPoolWait(NvOsEventPool *eventPool, NvU32 timeoutUs, NvOsEvent **event);

/** Creates a system event.
 *
 *  @param fd File descriptor of driver node.
 *  @param id System event identifier.
 *  @param event Returns os-dependent handle of new system event.
 *
 *  @retval NvError_InsufficientResources Returned if no memory for event.
 */
NvError NvOsSystemEventCreate(int fd, NvU32 id, NvOsEvent **event);

/** Creates a timer event.
 *
 *  @param timeoutMs Timer event Timeout in milliseconds.
 *  @param event Returns new timer event.
 *
 *  @retval NvError_InsufficientResources Returned if no memory for event.
 */
NvError NvOsTimerEventCreate(NvU32 timeoutMs, NvOsEvent **event);

/** Creates a user-level event for application->application use.
 *
 *  @param event Returns new user event.
 *
 *  @retval NvError_InsufficientResources Returned if no memory for event.
 */
NvError NvOsUserEventCreate(NvOsEvent **event);

/** Signals user event.
 *
 *  @param event Event to signal.
 *
 *  @retval NvError_InsufficientResources Returned if no memory for event.
 */
NvError NvOsUserEventSignal(NvOsEvent *event);

/** Destroys an event.
 *
 *  @param event Event to destroy.
 */
void NvOsEventDestroy(NvOsEvent *event);

/** Clears event.
 *
 *  If an event is pending, issuing a clear resets the pending state.
 *
 *  @param event Event to clear.
 */
void NvOsEventClear(NvOsEvent *event);

/** Waits for an event to be signaled.
 *
 *  Unlike NvOsEventPoolWait, this function is thread safe. If multiple
 *  threads are waiting for the same event, they will all be woken up when
 *  the event is signaled.
 *
 *  @param event Event to wait for.
 *  @param timeoutUs Timeout in microseconds, or NV_WAIT_INFINITE.
 *                   Does not apply to timer events.
 *  @retval NvError_Timeout Returned when the wait times out.
 */
NvError NvOsEventWait(NvOsEvent* event, NvU32 timeoutUs);

typedef struct NvOsThreadCreateAttrRec
{
    // If NULL, driver will allocate memory for stack
    void *threadStack;
    // If 0, driver will use OS-dependent default stack size
    size_t threadStackSize;
    // If NvOsThreadPriorityType_Native, nativePriority is used instead
    NvOsThreadPriorityType priority;
    // OS-dependent priority value of thread
    int nativePriority;
    // Preferred CPU core number to run on. -1 indicates no preference
    int idealCore;
    // Bitmask of CPU cores allowed for the thread to run on. 0 indicates core mask is not specified
    uint64_t affinityMask;
    // If NULL, thread name is not set
    const char *threadName;
} NvOsThreadCreateAttr;

#define NVOS_DEFINE_THREAD_CREATE_ATTR(x) \
    NvOsThreadCreateAttr x = { NULL, 0, NvOsThreadPriorityType_Normal, 0, -1, 0, NULL }

/** Assigns the core affinity mask indicating the set of cores
 *  allowed for the given thread along with ideal core number
 *  preferred.
 *  When bit N in affinityMask param is set to 1, the thread
 *  can run on core N.
 *  Affinity mask and ideal core number needs to be set before
 *  every thread creation.
 *
 *  @param idealCore Preferred ideal core number.
 *  @param affinityMask Affiniy mask for each core.
 */
NvError
NvOsThreadCreateWithAttr(NvOsThreadFunction function, void *args,
    NvOsThreadHandle *thread, const NvOsThreadCreateAttr *attr);

typedef void (*NvOsExitHandlerFunc)(void);
typedef struct NvOsExitHandlerContextRec
{
    NvOsExitHandlerFunc Func;
    struct NvOsExitHandlerContextRec *Prev;
} NvOsExitHandlerContext;

/** Enumerate Telemetry context categories
 *
 *  NvOsTelemetryCategory_Test: Test context
 */
typedef enum
{
    NvOsTelemetryCategory_Test,
    NvOsTelemetryCategory_VideoInfo,
} NvOsTelemetryCategory;

/** Submit context data to Telemetry architecture
 *
 *  Telemetry report is a way of reporting significant data when
 *  an error is present. A category and the corresonding struct
 *  will be assigned so that necessary data can be submitted to
 *  the Telemetry architecture.
 *
 *  @param category Context category to be submitted to Telemetry.
 *  @param arg      Defined struct for the specific category to be submitted.
 */
NvError NvOsTelemetrySubmitData(NvOsTelemetryCategory category, void *arg);

/** Platform specific implementation of exit handler API.
 *
 *  For platforms that do not support the atexit() function, this provides a mechanism
 *  for NVIDIA components to register a callback. The NvOs platform backend
 *  is responsible for triggering these callbacks at an appropriate time during
 *  NvOs deinitialisation. For example, it may provide a platform-specific API
 *  that the process should call during shutdown. The callbacks will be executed
 *  in reverse order to when they were registered.
 *
 *  Unlike the C standard atexit() function, this function takes two parameters.
 *  The first is the client function that is called when the process
 *  exits. The second is a pointer to an NvOsExitHandlerContext structure. The
 *  client does not need to initialize the members of this struct, instead it
 *  is * initialized inside the call to NvOsExitHandler().
 *
 *  The NvOsExitHandlerContext structure can be considered opaque by the client,
 *  the client does not directly use the members of the structure. Its
 *  purpose is to allow the NvOs library to internally store a linked list of
 *  registered callback functions, without using dynamic memory
 *  allocations.
 *
 *  This puts extra restrictions on how the client must declare the variable
 *  for the NvOsExitHandlerContext structure. It must be guaranteed to be valid
 *  until the exit callback has been executed. Generally, this means declaring
 *  it as a static variable. For this reason, using the macro
 *  NVOS_REGISTER_EXIT_HANDLER() and NVOS_REGISTER_EXIT_HANDLER_LAST are the
 *  preferred way for clients to register their atexit() handler function.
 *
 * @param func  Function to be called when process exits.
 * @param ctx   A pointer to the object for NvOs internal storage of callback function.
 * @param toLast A bool flag to register a handler to be called at last.
 */
void NvOsRegisterExitHandler(NvOsExitHandlerFunc func, NvOsExitHandlerContext *ctx, NvBool toLast);

#define NVOS_REGISTER_EXIT_HANDLER(_func) \
    do { \
        static NvOsExitHandlerContext _ctx = {0,0}; \
        NvOsRegisterExitHandler(_func, &_ctx, false); \
    } while(0)

#define NVOS_REGISTER_EXIT_HANDLER_LAST(_func) \
    do { \
        static NvOsExitHandlerContext _ctx = {0,0}; \
        NvOsRegisterExitHandler(_func, &_ctx, true); \
    } while(0)

#endif

NvU32 NvOsReadReg(NvU64 Reg, NvU32 Off);
void NvOsWriteReg(NvU64 reg, NvU32 off, NvU32 val);
NvError NvOsMapRegSpace(NvU64 *VirtReg, NvU32 Base, NvU32 Size);
NvError NvOsUnMapRegSpace(NvU64 *VirtReg, NvU32 Base, NvU32 Size);

/**
 * @brief This function does slog initialization for current executing context.
 * The Log init operation is performed internally in the NvOs library constructor.
 * So this function call is not required for process where NvOs library loaded.
 * However this function should be called when child process is executed using
 * fork().
 *
 * SWUD ID:
 *
 * @return int return zero on Success and -1 An error occurred log Init failed.
 *
 * @note
 *   - This function should be called when current executing context is forked
 *   process. In any other case this function is not expected to be called.
 *
 * - Is thread safe: No
 * - Required Privileges: N/A
 * - API Mode
 *   - Initialization: Yes
 *   - Run time: No
 *   - De-initialization: No
 */
int NvOsDebugExternalLogInit(void);

/*@}*/
/** @} */

#if defined(__cplusplus)
}
#endif

/* Additional or redefined NvOs APIs for HOS */
#if NVOS_IS_HOS
#include "../../core-hos/include/nvos_hos.h"
#endif

#endif // INCLUDED_NVOS_TEGRA_NONSAFETY_H
