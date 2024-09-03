/*
 * Copyright (c) 2016 NVIDIA Corporation.  All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */

/**
 * @file
 * <b>NVIDIA Tegra Screen Capture Interface</b>
 *
 * @b Description: This file contains the screen capture API for Tegra.
 */

/**
 * @defgroup scrncapt_api_group Screen Capture API
 *
 * The Screen Capture API is implemented by the Screen Capture wrapper library,
 * `libnvscrncapt.so`. Its functions and data structures are declared by
 * the this header file, which applications must include to use the
 * Screen Capture API.
 *
 * A screen capture operation consists of three API calls to:
 * - Initiate
 * - Perform capture, and
 * - Conclude the operation.
 *
 * <h3>Initiate Screen Capture</h3>
 *
 * A screen capture operation is initiated by a call to NvScrncaptInit(),
 * which creates an instance of ::NvScrncaptResult and initializes some of
 * its members. The Screen Capture API returns screen images to the application
 * in this buffer. All subsequent calls to the Screen Capture API require an
 * argument that points to the structure.
 *
 * After a successful return from \c %NvScrncaptInit(), the application may
 * populate \a userAddress and \a userDataSize to specify a preallocated,
 * user-supplied screen capture buffer. If the application supplies a screen
 * capture buffer it is responsible for managing the buffer and
 * freeing the buffer when the screen capture operation concludes. If the
 * application does not supply a buffer, the API allocates and manages
 * the buffer itself.
 *
 * <h3>Perform Screen Capture</h3>
 *
 * To perform a capture operation, call the NvScrncaptCapture() function.
 *
 * If the call succeeds, it:
 *
 * -# Allocates a screen capture buffer.
 * -# Captures the requested screen images in the buffer.
 * -# Sets \c result->numHeads to indicate the number of heads captured.
 * -# Sets the elements of \c result->heads[] to represent the screen capture
 *   on each captured head. Element 0 represents the first captured head,
 *   element 1 represents the second, and so on. Thus the function sets elements
 *   0 through numHeads 1.
 * -# Returns a status code of zero.
 *
 * If the call fails, it returns a non-zero status code.
 *
 * If screen captures are stored in ::NvScrncaptSurfaceLayout_BlockLinear
 * surface layout format, then applications must call the
 * NvScrncaptGetBlocklinearOffset() function to calculate the byte offset
 * to a pixel.
 *
 * <h3>Conclude Screen Capture</h3>
 *
 * Call the NvScrncaptCleanup() function to conclude the screen capture
 * operation. This call frees the screen capture buffer created by the
 * NvScrncaptCapture() function and releases control of the kernel
 * screen capture functionality.
 *
 * <h3>Allocating and Managing the Screen Capture Buffer</h3>
 *
 * An application may allocate and manage its own screen buffer rather
 * than let the Screen Capture API do so. If the application allocates its
 * own buffer, before it calls the \c %NvScrncaptCapture() function, it must
 * store the buffer's address in \a result->userAddress and its size in
 * \a result->userDataSize.
 *
 * The buffer must be large enough to hold all of the screen images to be
 * captured to succeed. A good way to estimate the size required is to
 * perform a screen capture operation in which the API is allowed to manage
 * the buffer and then examine \a result->stats.memSize before calling the
 * NvScrncaptCleanup() function.
 *
 * If the application allocates its own screen buffer, it is responsible
 * for freeing the buffer after it calls the NvScrncaptCleanup() function.
 *
 * The following sections describe the major data structures and functions in
 * the Screen Capture API, which assume that the API creates and manages the
 * buffer. A sample application is provided as an example of
 * how to use the API.
 *
 * @ingroup scrncapt_group
 * @{
 */

#ifndef _NVSCRNCAPT_H
#define _NVSCRNCAPT_H

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Major Version number */
#define NVSCRNCAPT_VERSION_MAJOR     1
/** \brief Minor Version number */
#define NVSCRNCAPT_VERSION_MINOR     0

/** \hideinitializer \brief A true \ref NvScrncaptBool value */
#define NVSCRNCAPT_TRUE  (0 == 0)
/** \hideinitializer \brief A false \ref NvScrncaptBool value */
#define NVSCRNCAPT_FALSE (0 == 1)

/** \hideinitializer \brief Max # of Display Heads */
#define NVSCRNCAPT_MAX_HEADS         4
/** \hideinitializer \brief Max # of Windows */
#define NVSCRNCAPT_MAX_WINS          6

/**
 * \brief A boolean value, holding \ref NVSCRNCAPT_TRUE or
 * \ref NVSCRNCAPT_FALSE.
 */
typedef int NvScrncaptBool;

/**
 * \hideinitializer
 * \brief Defines the set of all possible error codes.
 */
typedef enum {
    /** \hideinitializer The operation completed successfully; no error. */
    NVSCRNCAPT_STATUS_OK = 0,
    /** Indicates a bad parameter was passed. */
    NVSCRNCAPT_STATUS_BAD_PARAMETER,
    /** Indicates a the resource is busy. */
    NVSCRNCAPT_STATUS_BUSY,
    /** Indicates a operation timed out. */
    NVSCRNCAPT_STATUS_TIMED_OUT,
    /** Indicates a an out-of-memory condition. */
    NVSCRNCAPT_STATUS_OUT_OF_MEMORY,
    /** Indicates not initialized. */
    NVSCRNCAPT_STATUS_NOT_INITIALIZED,
    /** Indicates not supported. */
    NVSCRNCAPT_STATUS_NOT_SUPPORTED,
    /** Indicates a catch-all error, used when no other error code applies. */
    NVSCRNCAPT_STATUS_ERROR,
    /** Indicates invalid size. */
    NVSCRNCAPT_STATUS_INVALID_SIZE,
    /** Indicates an incompatible version. */
    NVSCRNCAPT_STATUS_INCOMPATIBLE_VERSION,
} NvScrncaptStatus;

/**
 * \brief Defines the set of frame buffer memory layout types.
 */
typedef enum {
  NvScrncaptSurfaceLayout_Tiled = 0,
  NvScrncaptSurfaceLayout_PitchLinear,
  NvScrncaptSurfaceLayout_BlockLinear,
} NvScrncaptSurfaceLayout;

/**
 * \brief Defines the set of pixel color formats, which must
 *        be kept in sync with the kernel defines.
 */
typedef enum {
    NvScrncaptColorFormat_RGB_AUTO        = ~0,
    /* NOTE: these enums need to be kept in sync with the kernel defines */
    NvScrncaptColorFormat_P8              = 3,
    NvScrncaptColorFormat_B4G4R4A4        = 4,
    NvScrncaptColorFormat_B5G5R5A         = 5,
    NvScrncaptColorFormat_B5G6R5          = 6,
    NvScrncaptColorFormat_AB5G5R5         = 7,
    NvScrncaptColorFormat_B8G8R8A8        = 12,
    NvScrncaptColorFormat_R8G8B8A8        = 13,
    NvScrncaptColorFormat_B6x2G6x2R6x2A8  = 14,
    NvScrncaptColorFormat_R6x2G6x2B6x2A8  = 15,
    NvScrncaptColorFormat_YCbCr422        = 16,
    NvScrncaptColorFormat_YUV422          = 17,
    NvScrncaptColorFormat_YCbCr420P       = 18,
    NvScrncaptColorFormat_YUV420P         = 19,
    NvScrncaptColorFormat_YCbCr422P       = 20,
    NvScrncaptColorFormat_YUV422P         = 21,
    NvScrncaptColorFormat_YCbCr422R       = 22,
    NvScrncaptColorFormat_YUV422R         = 23,
    NvScrncaptColorFormat_YCbCr422RA      = 24,
    NvScrncaptColorFormat_YUV422RA        = 25,
    NvScrncaptColorFormat_R4G4B4A4        = 27,
    NvScrncaptColorFormat_R5G5B5A         = 28,
    NvScrncaptColorFormat_AR5G5B5         = 29,
    NvScrncaptColorFormat_B5G5R5X         = 30,
    NvScrncaptColorFormat_XB5G5R5         = 31,
    NvScrncaptColorFormat_R5G5B5X         = 32,
    NvScrncaptColorFormat_XR5G5B5         = 33,
    NvScrncaptColorFormat_R5G6B5          = 34,
    NvScrncaptColorFormat_A8B8G8R8        = 36,
    NvScrncaptColorFormat_B8G8R8X8        = 37,
    NvScrncaptColorFormat_R8G8B8X8        = 38,
    NvScrncaptColorFormat_YCbCr444P       = 41,
    NvScrncaptColorFormat_YCrCb420SP      = 42,
    NvScrncaptColorFormat_YCbCr420SP      = 43,
    NvScrncaptColorFormat_YCrCb422SP      = 44,
    NvScrncaptColorFormat_YCbCr422SP      = 45,
    NvScrncaptColorFormat_YCrCb422SPR     = 46,
    NvScrncaptColorFormat_YCbCr422SPR     = 47,
    NvScrncaptColorFormat_YCrCb444SP      = 48,
    NvScrncaptColorFormat_YCbCr444SP      = 49,
    NvScrncaptColorFormat_YUV444P         = 52,
    NvScrncaptColorFormat_YVU420SP        = 53,
    NvScrncaptColorFormat_YUV420SP        = 54,
    NvScrncaptColorFormat_YVU422SP        = 55,
    NvScrncaptColorFormat_YUV422SP        = 56,
    NvScrncaptColorFormat_YVU422SPR       = 57,
    NvScrncaptColorFormat_YUV422SPR       = 58,
    NvScrncaptColorFormat_YVU444SP        = 59,
    NvScrncaptColorFormat_YUV444SP        = 60,
} NvScrncaptColorFormat;

/**
 * \brief Defines the set of blend modes possible for a window.
 */
typedef enum {
    NvScrncaptBlend_None,
    NvScrncaptBlend_Premult,
    NvScrncaptBlend_Coverage,
    NvScrncaptBlend_Colorkey,
} NvScrncaptBlend;

/**
 * \brief Holds a pixel value.
 */
typedef struct {
    /*! Holds the R/Y plane pixel value. */
    union {
        unsigned char r;
        unsigned char y;
    };
    /*! Holds the G/U plane pixel value. */
    union {
        unsigned char g;
        unsigned char u;
    };
    /*! Holds the B/V plane pixel value. */
    union {
        unsigned char b;
        unsigned char v;
    };
    /*! Holds the alpha pixel value. */
    unsigned char alpha;
} NvScrncaptPixel;

/**
 * Holds planes within the frame buffer. Depending on
 * the \a pixelFormat, access to certain planes may
 * be invalid. Use the following guidance when accessing:
 *
 * - For RGB formats, pU & pV planes shall not be used. Both
 *   lengthU & lengthV fields will be set to 0.
 * - For YUV formats, U and/or V planes may not be used.
 *   For example, in YUV420 Semiplanar:
 *    - pY points to Y values
 *    - PU points to U/V interleaved values
 *    - pV shall not be used (lengthV will be 0)
 *
 *    In general, the \a lengthX parameter should be checked to
 *    see if a given plane is valid for accessing.
 */
typedef struct {
    /*! Holds the surface layout. */
    NvScrncaptSurfaceLayout surfaceLayout;
    /*! Holds the pixel color format. */
    NvScrncaptColorFormat pixelFormat;
    /*! Holds the RGB/Y surface pointer. */
    union {
        unsigned char *pRGB;
        unsigned char *pY;
    };
    /*! Holds the RGB/Y buffer length. */
    union {
        unsigned int lengthRGB;
        unsigned int lengthY;
    };
    /*! RGB/Y surface stride */
    union {
        unsigned int strideRGB;
        unsigned int strideY;
    };
    /*! Holds the U surface pointer. */
    unsigned char *pU;
    /*! Holds the U buffer length. */
    unsigned int lengthU;
    /*! Holds the U/V surface stride. */
    unsigned int strideUV;
    /*! Holds the V surface pointer. */
    unsigned char *pV;
    /*! Holds the V buffer length. */
    unsigned int lengthV;
    /*! Holds the block Height. */
    unsigned char blockHeight;
} NvScrncaptSurfaceMap;


/**
 * \brief Holds the representation of a display aperture.
 */
typedef struct {
    /*! Holds the X-coordinate of start position. */
    int startX;
    /*! Holds the Y-coordinate of start position. */
    int startY;
    /*! Holds the width. */
    int width;
    /*! Holds the height. */
    int height;
} NvScrncaptAperture;

/**
 * \brief Holds the representation of a window's current state.
 */
typedef struct {
    /*! Holds a flag indicating if this window is enabled. */
    NvScrncaptBool enabled;
    /*! Holds the hardware window index whith which this handle is associated. */
    unsigned int windowIdx;
    /*! Holds the handle to the associated frame buffer surface map. */
    NvScrncaptSurfaceMap surfaceMap;
    /*! Holds the blend mode. */
    NvScrncaptBlend blendMode;
    /*! Holds the window depth. */
    unsigned int winDepth;
    /*! Holds the global alpha value for window. */
    unsigned int alpha;
    /*! Holds the horizontal inversion. */
    NvScrncaptBool invertH;
    /*! Holds the vertical inversion. */
    NvScrncaptBool invertV;
    /*! Holds the framebuffer aperture. */
    NvScrncaptAperture fbAperture;
    /*! Holds the output aperture. */
    NvScrncaptAperture outputAperture;
} NvScrncaptWindowState;

/**
 * \brief Holds a head's current state.
 */
typedef struct {
    /*! Holds the index of this display head. */
    unsigned int headIdx;
    /*! Holds the a flag indicating whether this head is enabled. */
    NvScrncaptBool enabled;
    /*! Holds the horizaontal display resolution. */
    unsigned int resolutionH;
    /*! Holds the vertical display resolution. */
    unsigned int resolutionV;
    /*! Holds the number of windows assigned to this head. */
    int numWins;
    /*! Holds the window state information (for \a numWins). */
    NvScrncaptWindowState wins[NVSCRNCAPT_MAX_WINS];
} NvScrncaptHeadState;

/**
 * \brief Holds statistics for a single capture.
 */
typedef struct {
    /*! Holds the size of memory allocated for this capture in bytes. */
    unsigned int memSize;
    /*! Holds the uSec taken for capture. */
    unsigned int captureTimeUsec;
} NvScrncaptStatistics;

/**
 * \brief Holds the screen capture result.
 */
typedef struct {
    /*! Holds the number of active heads. */
    int numHeads;
    /*! Holds the head state information. */
    NvScrncaptHeadState heads[NVSCRNCAPT_MAX_HEADS];
    /*! Holds the user-provided address of pre-allocated memory. */
    unsigned char *userAddress;
    /*! Holds the size of user-allocated memory. */
    unsigned int userDataSize;
    /*! Holds the capture statistics. */
    NvScrncaptStatistics stats;
    /*! Holds the reserved for library use: do not manipulate. */
    void *reserved;
} NvScrncaptResult;

/**
 * Initializes data structures required for screen capture.
 *
 * This call creates an instance of struct ::NvScrncaptResult, which
 * is the central data structure for the Screen Capture API.
 *
 * \note This API takes exclusive hold of screen capture
 * functionality from the kernel. Any future call to the kernel screen capture
 * API will fail without a preceding them first with a call to
 * \ref NvScrncaptCleanup.
 *
 * \param[out] pResult A double pointer to a \c %NvScrncaptResult object to
 * get the allocated one
 *
 * \return \ref NvScrncaptStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVSCRNCAPT_STATUS_OK
 * \n \ref NVSCRNCAPT_STATUS_OUT_OF_MEMORY
 * \n \ref NVSCRNCAPT_STATUS_ERROR
 */
NvScrncaptStatus NvScrncaptInit(NvScrncaptResult **pResult);

/**
 * Conducts screen capture on heads selected by input bitmask.
 *
 * To use a pre-allocated buffer for capture, populate the
 * result data structure with \a userAddress and \a userDataSize prior
 * to this call.
 *
 * \pre
 * The ::NvScrncaptResult structure has been created by a
 * call to the NvScrncaptInit() function.
 *
 * \param[in] result A pointer to a \c %NvScrncaptResult object.
 * \param[in] headMask A bitmask specifying on which heads to perform the
 *            screen capture. A "1" bit specifies to capture the
 *            corresponding head. The low order bit represents head 0,
 *            the next bit represents head 1, and so on. Thus, for example,
 *            a bit mask of 0x05 tells the API to capture images for head 0
 *            and head 2.
 *
 * \return \ref NvScrncaptStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVSCRNCAPT_STATUS_OK
 * \n \ref NVSCRNCAPT_STATUS_NOT_INITIALIZED
 * \n \ref NVSCRNCAPT_STATUS_BAD_PARAMETER
 * \n \ref NVSCRNCAPT_STATUS_OUT_OF_MEMORY
 * \n \ref NVSCRNCAPT_STATUS_ERROR
 */
NvScrncaptStatus NvScrncaptCapture(NvScrncaptResult *result, unsigned int headMask);

/**
 * Cleans up associated data structures and releases exclusive
 * hold on screen capture functionality. Library-allocated
 * buffers are freed upon this call. In case of pre-allocated
 * user buffers, caller is responsible for freeing these buffers
 * upon return of this call.
 *
 * \param[in] result A pointer to a NvScrncaptResult object
 *
 * \return \ref NvScrncaptStatus The completion status of the operation.
 * Possible values are:
 * \n \ref NVSCRNCAPT_STATUS_OK
 * \n \ref NVSCRNCAPT_STATUS_NOT_INITIALIZED
 * \n \ref NVSCRNCAPT_STATUS_ERROR
 */
NvScrncaptStatus NvScrncaptCleanup(NvScrncaptResult *result);

/**
 * Auxiliary function to calculate byte offset for a pixel
 * for \ref NvScrncaptSurfaceLayout_BlockLinear surface format.
 *
 * The return value is the byte offset to the pixel from the start of
 * screen image. The values of \a stride and \a blockHeight can be obtained
 * from the ::NvScrncaptSurfaceMap structure for the plane.
 *
 * See the sample application \c GetPixelRGB() and \c GetPixelYUV() functions
 * for examples of how to obtain these values.
 *
 * \param[in] x Specifies the horizontal x byte position
 * \param[in] y Specifies the vertical y position
 * \param[in] stride Specifies the stride of current plane (length of a row
 *            of pixels in bytes.
 * \param[in] blockHeight Specifies the block height value of the block
 *            linear surface.
 *
 * \return Byte offset of the pixel from start of buffer.
 */
unsigned int NvScrncaptGetBlocklinearOffset(unsigned int x,
        unsigned int y, unsigned int stride, unsigned int blockHeight);

#ifdef __cplusplus
};      /* extern "C" */
#endif

/** @} */
#endif  /* _NVSCRNCAPT_H */
