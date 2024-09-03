/*
 * Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef __NVPLAYFAIR_H__
#define __NVPLAYFAIR_H__

#ifdef __cplusplus
extern "C" {
#endif

#undef NVPLAYFAIR_ARCH_IS_X86

#if (defined(__i386__) || defined(__amd64__))
#define NVPLAYFAIR_ARCH_IS_X86
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>

#ifdef NVPLAYFAIR_ARCH_IS_X86
#include <time.h>

#else
/* @brief   The Tegra counter used for timestamping by NvPlayfair library runs
 *          at a frequency of 31.25-MHz which translates to 1-cycle = 32-nsec.
 *          The following macro can be used to convert a cycle count value to
 *          nsec value by doing a shift left by the specified amount.
 */
#define TEGRA_EXPECTED_CLOCK_FREQ       (31250000UL)
#define TEGRA_CYCLES_TO_NSEC_SHIFT      (5U)

#endif /* NVPLAYFAIR_ARCH_IS_X86 */

#define NVPLAYFAIR_LIB_VERSION_MAJOR    (1U)
#define NVPLAYFAIR_LIB_VERSION_MINOR    (2U)

#define FOREACH_UNIT(CMD)                                                           \
    CMD(SEC)                                                                        \
    CMD(MSEC)                                                                       \
    CMD(USEC)                                                                       \
    CMD(NSEC)

#define FOREACH_STATUS(CMD)                                                         \
    CMD(NVP_PASS)                                                                   \
    CMD(NVP_FAIL_ALLOC)                                                             \
    CMD(NVP_FAIL_NOINIT)                                                            \
    CMD(NVP_FAIL_FILEOP)                                                            \
    CMD(NVP_FAIL_NULLPTR)                                                           \
    CMD(NVP_FAIL_NO_SAMPLES)                                                        \
    CMD(NVP_FAIL_VERSION_MISMATCH)                                                  \
    CMD(NVP_FAIL_INVALID_TIME_UNIT)                                                 \
    CMD(NVP_FAIL_INVALID_LOG_BACKEND)                                               \
    CMD(NVP_FAIL_SAMPLE_COUNT_MISMATCH)

#define GENERATE_ENUM(ENUM)             ENUM,
#define GENERATE_STRING(STRING)         #STRING,

/** @brief  This enum defines the possible return statuses of NvPlayfair APIs.
 */
typedef enum {
    FOREACH_STATUS(GENERATE_ENUM)
} NvpStatus_t;

/** @brief This array is used to print library error codes as strings.
 */
extern const char *nvpStatusStr[];

/** @brief  This enum defines the time-units understood by relevant NvPlayfair
 *          APIs.
 */
typedef enum {
    FOREACH_UNIT(GENERATE_ENUM)
} NvpTimeUnits_t;

/** @brief  This enum defines the supported logging backends of the library.
 */
typedef enum {
    CONSOLE,
    NVOS
} NvpLogBackend_t;

/** @brief  This is used to store various statistics related to the gathered
 *          data.
 */
typedef struct
{
    double   min;
    double   max;
    double   mean;
    double   pct99;
    double   stdev;
    uint32_t count;
} NvpPerfStats_t;

/** @brief  This data-structure defines a rate-limit object which can be used
 *          to make a benchmark repeat its compute loop at a given period.
 */
typedef struct NvpRateLimitInfo
{
    uint32_t periodUs;
    uint32_t periodNumber;
    uint64_t periodicExecStartTimeUs;
} NvpRateLimitInfo_t;

/** @brief  Main data-structure provided by the NvPlayfair library. It can be
 *          configured as a benchmarking data object that can be used to store
 *          all the latency data gathered for a particular metric by a
 *          benchmark.
 */
typedef struct NvpPerfData
{
    uint64_t sampleNumber;
    uint64_t *timestamps;
    uint64_t *latencies;
    uint32_t maxSamples;
    bool initialized;
    char *filename;
} NvpPerfData_t;

/** @brief  Data-structure for capturing the version info. of the library.
 */
typedef struct NvpLibVersion
{
    uint32_t major;
    uint32_t minor;
} NvpLibVersion_t;

/** @brief  A benchmark can define this macro which will disable all the
 *          book-keeping activities of NvPlayfair and avoid loading
 *          nvplayfair.so at runtime.
 */
#ifndef DISABLE_NVPLAYFAIR

/** @brief  Check the return status of NvPlayfair APIs. It makes the caller
 *          program exit in case of error (hence it is not suitable when the
 *          program must do some cleanup when an error occurs).
 */
#define NVP_CHECKERR_EXIT(e) {                                                      \
    if (e != NVP_PASS) {                                                            \
        fprintf(stderr, "%s, %s:%d, NvPlayfair Error: %s\n",                        \
                __FILE__, __func__, __LINE__, nvpStatusStr[e]);                     \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
}

/** @brief  Macro for getting the total num. of samples in the ring-buffer of
 *          the given NvpPerfData_t* object.
 */
#define NVP_GET_SAMPLE_COUNT(perfData)                                              \
    (perfData->sampleNumber < (uint64_t)perfData->maxSamples)?                      \
        perfData->sampleNumber : (uint64_t)perfData->maxSamples


/** @brief  A convenience macro which can be used to iterate over the timestamp
 *          and latency values of all the samples recorded so far in the given
 *          NvpPerfData_t* object.
 *
 *  @param  obj             An NvpPerfData_t* object
 *  @param  ts              Name of a 64-bit timestamp value (declared in the macro)
 *  @param  lat             Name of a 64-bit latency value (declared in the macro)
 */
#define for_each_sample(obj, ts, lat)                                               \
    for(uint64_t i = 0UL, ts = (obj)->timestamps[i], lat = (obj)->latencies[i],     \
            sampleCount = NVP_GET_SAMPLE_COUNT(obj);                                \
            i < sampleCount;                                                        \
            ++i, ts = (obj)->timestamps[i], lat = (obj)->latencies[i])

/** @brief  Similar to "for_each_sample" macro but this one only iterates over
 *          the latency value for each sample in the given object.
 *
 *  @param  obj             An NvpPerfData_t* object
 *  @param  lat             Name of a 64-bit latency value (declared in the macro)
 */
#define for_each_sample_latency(obj, lat)                                           \
    for(uint64_t i = 0UL, lat = (obj)->latencies[i],                                \
            sampleCount = NVP_GET_SAMPLE_COUNT(obj);                                \
            i < sampleCount;                                                        \
            ++i, lat = (obj)->latencies[i])

/** @brief  Obtain an opaque timestamp in a safe-manner under out-of-order
 *          execution. The given timestamp is to be understood by NvPlayfair
 *          library only.
 *
 *  @return uint64_t        Opaque timestamp value
 */
static inline uint64_t NvpGetTimeMark(void)
{
    uint64_t mark;

#ifdef NVPLAYFAIR_ARCH_IS_X86
    int ret;
    struct timespec tp;

    ret = clock_gettime(CLOCK_MONOTONIC, &tp);
    if (ret != 0) {
        fprintf(stderr, "%s, %s:%d, NvPlayfair Error clock_gettime %d\n",
                __FILE__, __func__, __LINE__, ret);
        exit(EXIT_FAILURE);
    }

    mark = tp.tv_sec * 1000000000UL + tp.tv_nsec;

#else /* NVPLAYFAIR_ARCH_IS_X86 */
    __asm__ __volatile__ ("ISB;                                                     \
                           mrs %[result], cntvct_el0;                               \
                           ISB"
                           : [result] "=r" (mark)
                           :
                           : "memory");

#endif /* NVPLAYFAIR_ARCH_IS_X86 */

    return mark;
}

/** @brief  Convert the opaque time-mark value to nsec timestamp. Note
 *          that, on ARM architecture, the function assumes that 1-cycle =
 *          32-nsec which is true for the Tegra counters running at the
 *          frequency of 31.25-MHz.
 *
 *  @param  timeMark            64-bit opaque time mark value
 *  @return uint64_t            Current timestamp in nsec
 */
static inline uint64_t
NvpConvertTimeMarkToNsec(uint64_t timeMark)
{
#ifdef NVPLAYFAIR_ARCH_IS_X86
    return timeMark;

#else
    return (timeMark << TEGRA_CYCLES_TO_NSEC_SHIFT);

#endif /* NVPLAYFAIR_ARCH_IS_X86 */
}

/** @brief  Primary function for populating a given NvpPerfData_t* object. It
 *          allocates necessary memory for internal buffers based on the given
 *          num. of samples and stores the given information for internal
 *          book-keeping.
 *
 *  @param  perfData        An NvpPerfData_t* object
 *  @param  numOfSamples    An integer specifying the total number of
 *                          samples that can be stored in the
 *                          NvpPerfData_t* object
 *  @param  filename        A character string specifying the name of a
 *                          file which can be used to store the recorded
 *                          performance data. Can be NULL, in which case,
 *                          the data will not be recorded
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpConstructPerfData(NvpPerfData_t *perfData,
        uint32_t numOfSamples,
        char const *filename);

/** @brief  Complementary function to NvpConstructPerfData; it can be used to
 *          free up all resources associated with the given NvpPerfData_t*
 *          object. It should be called once the user is done with the object.
 *
 *  @param  perfData        An NvpPerfData_t* object
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpDestroyPerfData(NvpPerfData_t *perfData);

/** @brief  Provide a short-hand way to invoke the respective function in the
 *          given NvpPerfData_t* object. The default behavior of this function
 *          is to record the given timestamp and latency values (and convert
 *          them to nsec if needed) in the internal buffers of NvpPerfData_t*
 *          object at the appropriate index.
 *
 *  @param  perfData        An NvpPerfData_t* object
 *  @param  sampleStartTime 64-bit opaque time value marking start of sample
 *  @param  sampleEndTime   64-bit opqaue time value marking end of sample
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpRecordSample(NvpPerfData_t *perfData,
        uint64_t sampleStartTime,
        uint64_t sampleEndTime);

/** @brief  Record the data gathered (up-till the moment when this call
 *          is made) in NvpPerfData_t* object into a file in the file-system.
 *          The name of the file is taken from the object itself; as provided
 *          at the time of object initialization via NvpConstructPerfData API.
 *
 *  @param  perfData        An NvpPerfData_t* object
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpDumpData(NvpPerfData_t *perfData);

/** @brief  Calculate latency percentiles in the given perfData structure. The
 *          percentile points which need to be calculated are specified as an
 *          array. The values of those percentile points are returned in
 *          another array of equal length; whose reference is provided as an
 *          argument to the function.
 *
 *  NOTE    The caller is responsible for freeing the memory of the array
 *          referenced by "percentileValues" pointer: free(*percentileValues)
 *
 *  @param  perfData        An NvpPerfData_t* object
 *  @param  numOfPercentilePoints
 *                          Num. of percentile points to calculate. This
 *                          variable is used to traverse the below variable
 *                          "percentilePointsArray" and to allocate memory
 *                          for the "percentileValues" array
 *  @param  percentilePointsArray
 *                          An array of doubles specifying the percentile
 *                          points (e.g., 50, 75, 99.99 etc.) to calculate of
 *                          the latency array
 *  @param  percentileValues
 *                          Return array which will be allocated dynamically
 *                          and populated with percentile values. The index of
 *                          each value will correspond to the respective index
 *                          in the "percentilePointsArray" e.g.,
 *                          percentilePointsArray[0] = 75;
 *                          *percentileValues[0] = 75th percentile value in
 *                                                 latency array of perfData
 *  @param  unit            An enum value specifying the unit of time in
 *                          which the calculated percentile values should be
 *                          reported
 *
 *  @return NvpStatus_t     NVP_PASS
 *                              API executed successfully
 *                          NVP_FAIL_NULLPTR
 *                              One ore more of the input pointers are NULL
 *                          NVP_FAIL_NO_SAMPLES
 *                              The given perfData structure does not contain
 *                              any latency samples
 *                          NVP_FAIL_INVALID_TIME_UNIT
 *                              Library does not recognize the specified unit
 */
NvpStatus_t
NvpCalcPercentiles(NvpPerfData_t *perfData,
        uint32_t numOfPercentilePoints,
        double *percentilePointsArray,
        double **percentileValues,
        NvpTimeUnits_t unit);

/** @brief  Calculate common stat. metrics (min, mean, max, standard deviation,
 *          99.99-percentile) for the given NvpPerfData_t* object.  The results
 *          are converted to the given time unit as needed before the function
 *          returns
 *
 *  @param  perfData        An NvpPerfData_t* object
 *  @param  stats           Pointer to a stats structure for returning
 *                          calculated statistics on the given data
 *  @param  unit            An enum value specifying the unit of time in
 *                          which stats should be reported
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpCalcStats(NvpPerfData_t *perfData,
        NvpPerfStats_t *stats,
        NvpTimeUnits_t unit);

/** @brief  Print a report on the console about the statistics calculated for
 *          the collected perf. data. It can be very handy to get quick insight
 *          into the data w/o resorting to detailed offline analysis.
 *
 *  @param  perfData        An NvpPerfData_t* object
 *  @param  stats           Pointer to a stats structure. If not NULL, the
 *                          values in this structure are assumed to be the
 *                          stats.  calculated for the given data object.
 *                          If it is NULL, the function will calculate
 *                          stats. for the data by calling NvpCalcStats
 *                          function internally
 *  @param  unit            An enum value specifying the unit of time in
 *                          which stats should be printed
 *  @param  msg             A character string specifying a message to be
 *                          printed at the beginning of the perf. report
 *  @param  csv             Print the report in CSV format
 *  @param  logBackend      Backend where the stats. report should be logged
 *  @param  reportFilename  Name of file where stats. report should be saved.
 *                          Can be NULL in which case, the report will not be
 *                          saved to a file
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpPrintStatsExt(NvpPerfData_t *perfData,
        NvpPerfStats_t *stats,
        NvpTimeUnits_t unit,
        char const *msg,
        bool csv,
        NvpLogBackend_t logBackend,
        char const *reportFilename);

/*
 * @brief   This is legacy API which is now superseded by NvpPrintStatsExt
 *
 * TODO Make NvpPrintStats == NvpPrintStatsExt and eliminate NvpPrintStatsExt
 */
NvpStatus_t
NvpPrintStats(NvpPerfData_t *perfData,
        NvpPerfStats_t *stats,
        NvpTimeUnits_t unit,
        char const *msg,
        bool csv);

/** @brief  Initialize an NvpRateLimitInfo_t* object with the information
 *          required to enforce a desired periodicity to the target benchmark.
 *
 *  @param  rtInfo          An NvpRateLimitInfo_t* object
 *  @param  fps             Frequency of taret benchmark's control loop
 *                          execution
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpRateLimitInit(NvpRateLimitInfo_t *rtInfo,
        uint32_t fps);

/** @brief  Calculate a timestamp in usec and store it as the start time
 *          of the first period of the calling benchmark inside the given
 *          object. This start time value is then used to calculate the
 *          boundaries for all the subsequent periods of the benchmark.
 *
 *  @param  rtInfo          An NvpRateLimitInfo_t* object
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpMarkPeriodicExecStart(NvpRateLimitInfo_t *rtInfo);

/** @brief  Capture a timestamp in usec and use it to calculate the time
 *          remaining till the beginning of the next period of the benchmark.
 *          If the remaining time is non-negative, then a sleep is invoked for
 *          the respective duration.
 *
 *  @param  rtInfo          An NvpRateLimitInfo_t* object
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpRateLimitWait(NvpRateLimitInfo_t *rtInfo);

/** @brief  Return the library version info. to the caller.
 *
 *  @param  version         An NvpLibVersion_t* object
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
NvpStatus_t
NvpGetLibVersion(NvpLibVersion_t *version);

/** @brief  Check that the library version matches between the public header
 *          and a compiled library object file.
 *
 *  @return NvpStatus_t     NVP_PASS on success or an appropriate error status
 */
static inline NvpStatus_t
NvpCheckLibVersion(void)
{
    NvpStatus_t ret = NVP_PASS;
    NvpLibVersion_t version;

    ret = NvpGetLibVersion(&version);
    if (ret != NVP_PASS) {
        return ret;
    }

    if (version.major != NVPLAYFAIR_LIB_VERSION_MAJOR ||
        version.minor != NVPLAYFAIR_LIB_VERSION_MINOR) {
        return NVP_FAIL_VERSION_MISMATCH;
    }

    return ret;
}

/** @brief  Allow aggregating latencies from different perfData objects into a
 *          given structure on a per-sample basis.
 *
 *  @param  netPerfData     NvpPerfData_t* object in which aggregate per-sample
 *                          latencies will be stored
 *  @param  inputPerfDataArray
 *                          Array of pointers to NvpPerfDat_t* objects. The
 *                          latency of each sample in the objects in this array
 *                          is summed up to generate respective sample latency
 *                          in the "netPerfData" object
 *  @param  numOfPerfDataObjs
 *                          Num. of perfData objects in the "inputPerfDataArray"
 *
 * @return  NvpStatus_t     NVP_PASS
 *                              API executed successfully
 *                          NVP_FAIL_NULLPTR
 *                              One ore more of the input pointers are NULL
 *                          NVP_FAIL_NOINIT
 *                              "netPerfData" object is not initialized
 *                          NVP_FAIL_SAMPLE_COUNT_MISMATCH
 *                              Max. samples in "netPerfData" do not match
 *                              with max. samples in one of the input perfData
 *                              objects OR Sample count in one of the input
 *                              perfData objects is different from the sample
 *                              count in the first input perfData object
 */
NvpStatus_t
NvpAggregatePerfData(NvpPerfData_t *netPerfData,
        NvpPerfData_t **inputPerfDataArray,
        uint32_t numOfPerfDataObjs);

#else /* DISABLE_NVPLAYFAIR */

#define NVP_CHECKERR_EXIT(e)                                    ( e )

#define NvpGetTimeMark()                                        ( 1UL )
#define NvpConvertTimeMarkToNsec(tm)                            ( 1UL )

#define for_each_sample(obj, ts, lat)                                               \
    for (uint64_t lat, ts; false; )

#define for_each_sample_latency(obj, lat)                                           \
    for (uint64_t lat; false; )

#define DISABLE_FUNC(func)                                                          \
    static inline NvpStatus_t func {                                                \
        return NVP_PASS;                                                            \
    }

DISABLE_FUNC(NvpConstructPerfData(NvpPerfData_t *perfData,
        uint32_t numOfSamples, char const *filename));

DISABLE_FUNC(NvpDumpData(NvpPerfData_t *perfData));
DISABLE_FUNC(NvpDestroyPerfData(NvpPerfData_t *perfData));
DISABLE_FUNC(NvpRecordSample(NvpPerfData_t *perfData, uint64_t start, uint64_t end));

DISABLE_FUNC(NvpCalcPercentiles(NvpPerfData_t *perfData,
            uint32_t numOfPercentilePoints, double *percentilePointsArray,
            double **percentileValues, NvpTimeUnits_t unit));

DISABLE_FUNC(NvpCalcStats(NvpPerfData_t *perfData, NvpPerfStats_t *stats,
            NvpTimeUnits_t unit));

DISABLE_FUNC(NvpPrintStatsExt(NvpPerfData_t *perfData, NvpPerfStats_t *stats,
            NvpTimeUnits_t unit, char const *msg, bool csv, NvpLogBackend_t
            logBackend, char const *reportFilename));

DISABLE_FUNC(NvpPrintStats(NvpPerfData_t *perfData, NvpPerfStats_t *stats,
            NvpTimeUnits_t unit, char const *msg, bool csv));

DISABLE_FUNC(NvpRateLimitWait(NvpRateLimitInfo_t *rtInfo));
DISABLE_FUNC(NvpMarkPeriodicExecStart(NvpRateLimitInfo_t *rtInfo));
DISABLE_FUNC(NvpRateLimitInit(NvpRateLimitInfo_t *rtInfo, uint32_t fps));

DISABLE_FUNC(NvpGetLibVersion(NvpLibVersion_t *version));
DISABLE_FUNC(NvpCheckLibVersion(void));

DISABLE_FUNC(NvpAggregatePerfData(NvpPerfData_t *netPerfData,
            NvpPerfData_t **inputPerfDataArray, uint32_t numOfPerfDataObjs));

#endif /* DISABLE_NVPLAYFAIR */

#ifdef __cplusplus
}
#endif

#endif /* __NVPLAYFAIR_H__ */
