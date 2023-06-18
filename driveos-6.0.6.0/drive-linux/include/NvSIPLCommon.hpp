/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLCOMMON_HPP
#define NVSIPLCOMMON_HPP

#include <memory>
#include <math.h>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Common Data Structures  - @ref NvSIPL </b>
 *
 */

/** @namespace nvsipl
 *  @brief Contains the classes and variables for implementation of @ref NvSIPL.
 */
namespace nvsipl
{
/** @defgroup NvSIPL SIPL
 *
 * @brief  SIPL provides abstract and simple API to capture the output of image sensors
 * with optional image processing.
 *
 */
/** @addtogroup NvSIPL
 * @{
 */


/**
 * \brief Holds a rectangular region of a surface.
 *
 * The co-ordinates are top-left inclusive, bottom-right
 * exclusive.
 *
 * The SIPL co-ordinate system has its origin at the top left corner
 * of a surface, with x and y components increasing right and
 * down.
 */
typedef struct {
    /*! Left X co-ordinate. Inclusive. */
    uint16_t x0;
    /*! Top Y co-ordinate. Inclusive. */
    uint16_t y0;
    /*! Right X co-ordinate. Exclusive. */
    uint16_t x1;
    /*! Bottom Y co-ordinate. Exclusive. */
    uint16_t y1;
} NvSiplRect;

/**
 * \brief Media global time, measured in microseconds.
 */
typedef uint64_t NvSiplGlobalTime;

/**
 * \brief Defines the location of a point on a two-dimensional object.
 */
typedef struct {
    /*! Holds the horizontal location of the point. */
    int32_t x;
    /*! Holds the vertical location of the point. */
    int32_t y;
} NvSiplPoint;

/**
 * \brief Defines the float-precision location of a point on a two-dimensional
 *  object.
 */
typedef struct {
    /*! Holds the horizontal location of the point. */
    float_t x;
    /*! Holds the vertical location of the point. */
    float_t y;
} NvSiplPointFloat;

/** \hideinitializer \brief A true \ref NvSiplBool value.
 */
#define SIPL_TRUE  (0 == 0)
/** \hideinitializer \brief A false \ref NvSiplBool value.
 */
#define SIPL_FALSE (0 == 1)

/**
 * \brief A boolean value, holding \ref SIPL_TRUE or \ref
 * SIPL_FALSE.
 */
typedef uint32_t NvSiplBool;

/** \brief Defines clock base for NvSiplTime.
 */
typedef enum {
    /** \hideinitializer \brief Specifies that PTP clock is used
     for base time calculation. */
    NVSIPL_TIME_BASE_CLOCK_PTP = 0,
    /** \hideinitializer \brief Specifies that kernel monotonic clock is used
     for base time calculation. */
    NVSIPL_TIME_BASE_CLOCK_MONOTONIC,
    /** \hideinitializer \brief Specifies that a user defined clock is used
     for base time calculation. */
    NVSIPL_TIME_BASE_CLOCK_USER_DEFINED,
} NvSiplTimeBase;

/** @brief Defines the status codes returned by functions in @ref NvSIPL modules. */
enum SIPLStatus
{
    // New status code must be added after NVSIPL_STATUS_OK and before NVSIPL_STATUS_ERROR.

    /** Indicates the operation completed successfully without errors. */
    NVSIPL_STATUS_OK = 0,

    // Error codes.
    /** Indicates one or more invalid arguments was encountered. */
    NVSIPL_STATUS_BAD_ARGUMENT,
    /** Indicates an unsupported operation or argument was encountered. */
    NVSIPL_STATUS_NOT_SUPPORTED,
    /** Indicates an out of memory or other system resource error was encountered. */
    NVSIPL_STATUS_OUT_OF_MEMORY,
    /** Indicates a resource error was encountered. */
    NVSIPL_STATUS_RESOURCE_ERROR,
    /** Indicates an operation timed out. */
    NVSIPL_STATUS_TIMED_OUT,
    /** Indicates a module is in an invalid state. */
    NVSIPL_STATUS_INVALID_STATE,
    /** Indicates that end of file has been reached. */
    NVSIPL_STATUS_EOF,
    /** Indicates a module was not initialized. */
    NVSIPL_STATUS_NOT_INITIALIZED,
    /** Indicates module is in non-recoverable fault state. */
    NVSIPL_STATUS_FAULT_STATE,
    /** Indicates an unspecified error that is used when no other error code applies. */
    NVSIPL_STATUS_ERROR
};

/** @brief CDAC GPIO event codes. */
enum SIPLGpioEvent
{
    /** There is no pending event. */
    NVSIPL_GPIO_EVENT_NOTHING = 0,
    /** An interrupt has occured. */
    NVSIPL_GPIO_EVENT_INTR,
    /**
     * Waiting for an interrupt was unexpectedly cancelled, potentially
     * resulting in permanent loss of functionality. (Error)
     */
    NVSIPL_GPIO_EVENT_ERROR_WAIT_CANCELLED,
    /**
     * An error occurred in CDAC code, potentially resulting in permanent loss
     * of functionality. (Error)
     */
    NVSIPL_GPIO_EVENT_ERROR_CDAC,
    /**
     * An error occurred in backend code, potentially resulting in permanent
     * loss of functionality. (Error)
     */
    NVSIPL_GPIO_EVENT_ERROR_BACKEND,
    /**
     * A generic error occurred, potentially resulting in permanent loss of
     * functionality. (Error)
     */
    NVSIPL_GPIO_EVENT_ERROR_UNKNOWN,
};

/**
 * Error details for a particular device
 */
typedef struct {
    /**
     * Buffer which will be filled by driver with error information.
     * Expected to be initialized by the client with size bufferSize.
     */
    std::unique_ptr<uint8_t []> upErrorBuffer;

    /**
     * Holds the maximum size of error data which can be contained in the buffer.
     */
    size_t bufferSize;

    /**
     * Holds size of error written to the buffer, filled by driver.
     */
    size_t sizeWritten;
} SIPLErrorDetails;

/** @brief Flag indicating which module errors to read. */
enum SIPLModuleErrorReadFlag
{
    /**
     * Read only sensor error information when getting module error details.
     * Skip reading serializer error information.
     */
    NVSIPL_MODULE_ERROR_READ_SENSOR,
    /**
     * Read only serializer error information when getting error details.
     * Skip reading sensor error information.
     */
    NVSIPL_MODULE_ERROR_READ_SERIALIZER,
    /**
     * Read both sensor and serializer error information when getting error details.
     */
    NVSIPL_MODULE_ERROR_READ_ALL,
};

/** @} */

}  // namespace nvsipl

#endif // NVSIPLCOMMON_HPP
