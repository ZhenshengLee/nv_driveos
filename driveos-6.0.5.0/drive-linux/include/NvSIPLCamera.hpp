/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef NVSIPLCAMERA_HPP
#define NVSIPLCAMERA_HPP

#include "NvSIPLCommon.hpp"
#include "NvSIPLPlatformCfg.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "INvSiplControlAuto.hpp"
#include "NvSIPLClient.hpp"
#include "INvSIPLDeviceInterfaceProvider.hpp"

#include "nvscisync.h"
#include "nvscistream.h"
#include "nvscibuf.h"

#include <cstdint>
#include <memory>
#include <vector>

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Camera Interface - @ref NvSIPLCamera_API </b>
 *
 */

/** @defgroup NvSIPLCamera_API NvSIPL Camera
 *
 * @brief Provides top-level interfaces to program external image devices and Tegra to create and
 * manage image processing pipelines to receive outputs in NvSciBufObj Images.
 *
 * @ingroup NvSIPL */

namespace nvsipl
{

/** @ingroup NvSIPLCamera_API
 * @{
 */

#if !NV_IS_SAFETY
static constexpr size_t NITO_PARAMETER_SET_ID_SIZE {16U};
static constexpr size_t NITO_SCHEMA_HASH_SIZE {32U};
static constexpr size_t NITO_DATA_HASH_SIZE {32U};

/**
 * NvSIPLNitoMetadata defines the 3-tuple returned by a successful call
 * to GetNitoMetadataFromMemory().
 */
struct NvSIPLNitoMetadata
{
    uint8_t parameterSetID[NITO_PARAMETER_SET_ID_SIZE]; //< Identifier of the parameter set
    uint8_t schemaHash[NITO_SCHEMA_HASH_SIZE];          //< Hash value of the parameter set schema
    uint8_t dataHash[NITO_DATA_HASH_SIZE];              //< Hash value of parameter values
};

/** @brief Get NITO Metadata (knobset UUID, schema hash, data hash) from a NITO memory buffer.
 *
 * @pre None.
 *
 * The possible return values from this functions are:
 * - NVSIPL_STATUS_OK
 * - NVSIPL_STATUS_BAD_ARGUMENT
 * - NVSIPL_STATUS_INVALID_STATE
 * - NVSIPL_STATUS_OUT_OF_MEMORY
 * - NVSIPL_STATUS_ERROR
 *
 * @param[in]     nitoMem              Pointer to location of memory to load from.
 * @param[in]     nitoMemLength        Size of memory pointed to by nitoMem.
 * This value must be in the range [1, 6MB (6UL * 1024UL * 1024UL)].
 * @param[in,out] metadataArray        An array of NitoMetadata tuples to store the result.
 * @param[in]     metadataArrayLength  The size of array @c metadataArray
 * @param[out]    metadataCount        The number of tuples stored in @c metadataArray on success.
 *
 * @returns::SIPLStatus the completion status of the operation.
 *
 * @usage
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions:
 *     - Two threads are not using the same metadata array.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: Yes, with the following conditions:
 *   - Grants: nonroot, allow
 *   - Abilities: public_channel
 *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
 *     NVIDIA DRIVE OS Safety Developer Guide
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
SIPLStatus GetNitoMetadataFromMemory(uint8_t const *const nitoMem,
                                     size_t const nitoMemLength,
                                     NvSIPLNitoMetadata *const metadataArray,
                                     size_t const metadataArrayLength,
                                     size_t *const metadataCount);

#endif // NV_IS_SAFETY

typedef enum
{
    /** For a given SyncObj SIPL acts as a signaler. This type corresponds to
     ** postfences from SIPL */
    SIPL_SIGNALER,
    /* For a given SyncObj SIPL acts as a waiter. This type corresponds to
     * prefences to SIPL */
    SIPL_WAITER,
    /* For a given SyncObj SIPL acts as both signaler and waiter. */
    SIPL_SIGNALER_WAITER,
} NvSiplNvSciSyncClientType;

/**
 * \brief Defines SIPL \ref NvSciSyncObj types.
 */
typedef enum {
    /** Specifies an NvSciSyncObj type for which SIPL acts as a waiter. */
    NVSIPL_PRESYNCOBJ,
    /** Specifies an NvSciSyncObj type for which SIPL acts as a signaler, signaling EOFFence. */
    NVSIPL_EOFSYNCOBJ,
    /** Specifies an NvSciSyncObj type for which SIPL acts as a signaler, signaling SOFFence. */
    NVSIPL_SOFSYNCOBJ,
    /**
     * Specifies an NvSciSyncObj type for which SIPL acts both as a signaler, signaling EOFFence,
     * and as a waiter.
     * Use this type in usecases where an EOFfence from a SIPL handle in one iteration is used as a
     * PREfence for the same handle in the next iteration.
     */
    NVSIPL_EOF_PRESYNCOBJ,
    /**
     * Specifies an NvSciSyncObj type for which a SIPL component acts as a signaler, signaling
     * SOFFence, as a waiter.
     * Use this type in usecases where an SOFfence from a SIPL handle in one iteration is used as a
     * PREfence for the same handle in the next iteration.
     */
    NVSIPL_SOF_PRESYNCOBJ

} NvSiplNvSciSyncObjType;


/** @class INvSIPLCamera NvSIPLCamera.hpp
 *
 * @brief The top-level API for SIPL.
 *
 * A SIPL client acquires this API by calling GetInstance() exactly once.
 *
 */
class INvSIPLCamera
{
 public:
    /** @brief Get a handle to an INvSIPLCamera instance.
     *
     * Create an instance of the implementation class and return the handle. The
     * object is automatically destroyed when the variable holding the return value goes out of
     * scope.
     *
     * @pre None.
     *
     * @returns a pointer to a new INvSIPLCamera instance.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    static std::unique_ptr<INvSIPLCamera> GetInstance();

    /** @brief Set a platform configuration.
     *
     * This method sets a @ref PlatformCfg camera platform configuration.
     * The configuration specifies all sensors that will be used by this client.
     * This method must be called before @ref SetPipelineCfg().
     *
     * @pre None.
     *
     * @param[in] platformConfig The platform configuration.
     * The external devices referenced in the platform configuration must be supported by the
     * SIPL Device Block drivers.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus SetPlatformCfg(PlatformCfg const* const platformConfig) = 0;

    /** @brief Set a platform configuration and returns the device block notification queues.
     *
     * This method sets a @ref PlatformCfg camera platform configuration and returns the device
     * block notification queues.
     * The configuration specifies all sensors that will be used by this client.
     *
     * This method must be called before @ref SetPipelineCfg().
     *
     * @pre None.
     *
     * @param[in] platformConfig @ref PlatformCfg
     * The external devices referenced in the platform configuration must be supported by the
     * SIPL Device Block drivers.
     * @param[out] queues The queues that will deliver device block notifications.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus SetPlatformCfg(PlatformCfg const* const platformConfig, NvSIPLDeviceBlockQueues &queues) = 0;

    /** @brief Set a pipeline configuration.
     *
     * This method sets a camera pipeline configuration.
     *
     * @pre This function must be called after @ref SetPlatformCfg() but before @ref Init().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] pipelineCfg The pipeline configuration to set.
     * @param[out] queues The queues that will deliver completed frames and events to the client.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus SetPipelineCfg(uint32_t index,
                                      NvSIPLPipelineConfiguration const &pipelineCfg,
                                      NvSIPLPipelineQueues& queues) = 0;

    /** @brief Register the Auto Control plugin to be used for a specific pipeline.
     *
     * This method must be called for every pipeline with ISP output enabled.
     *
     * @pre This function must be called after @ref RegisterImages() but before @ref Start().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] type Auto Control plugin type.
     * @param[in] autoControl Handle to plugin implementation,
     * or nullptr if type is NV_PLUGIN.
     * @param[in] blob Reference to binary blob containing the ISP configuration.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus RegisterAutoControlPlugin(uint32_t index,
                                                 PluginType type,
                                                 ISiplControlAuto *const autoControl,
                                                 std::vector<uint8_t> const &blob) = 0;

    /** @brief Initialize the API for the selected platform configuration.
     *
     * The method internally initializes the camera module(s) and deserializer for each device block
     * in the selected platform configuration, and creates and initializes the image processing
     * pipelines based on the number and type of the outputs set via @ref SetPipelineCfg.
     *
     * @pre This function must be called must be called after @ref SetPipelineCfg() but before @ref RegisterImages.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus Init() = 0;

#if !NV_IS_SAFETY
    /** @brief Set sensor in characterization mode.
     *
     * This function re-configures the sensor
     * i.e. changes the sensor static attributes
     * like numActiveExposures, sensorExpRange, sensorGainRange
     * and hence, should be called during sensor initialization time.
     * In order to characterize the sensor exposure number 'n',
     * where n = {1,2,3, ... , N} for N-exposure HDR sensor,
     * the input parameter 'expNo' should be set to 'n'.
     * For a non-HDR sensor, the input parameter 'expNo' should always be set to '1'.
     *
     * @pre This function must be called after @ref Init() and before @ref Start().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] expNo Sensor exposure number to be used for characterization.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus SetSensorCharMode(uint32_t index, uint8_t expNo) = 0;
#endif // !NV_IS_SAFETY

    /** @brief Get image attributes.
     *
     * This method is used to get the attributes of the images to be used with the image
     * processing pipeline. The user must reconcile the attributes returned by this function with
     * the attributes required by the downstream consumers of the output of the pipeline and
     * allocate the images.
     *
     * @pre This function must be called after @ref Init() but before @ref Start().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] outType The output for which image attributes are being fetched.
     * @param[in, out] imageAttr Reference to the image attributes structure.
     * @li The surface type for ICP output is determined by the properties of the image sensor
     * and must not be overridden by the user
     * @li The surface type for ISP0 output is set to the following by default for
     * both RGB Bayer and RGB-IR sensor if not already set by the client.
     * Note, if set by client it must only be to one of the supported surface types of
     * ISP output listed in @ref RegisterImages or an error will be generated.
     *
     * Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard
     * ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | --------------
     * YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER
     *
     * YUV 420 SEMI-PLANAR UINT8 Block Linear is defined by NvSciBuf with:
     * NvSciBufImageAttrKey_SurfType set to NvSciSurfType_YUV
     * NvSciBufImageAttrKey_SurfBPC set to NvSciSurfBPC_8
     * NvSciBufImageAttrKey_SurfMemLayout set to NvSciSurfMemLayout_SemiPlanar
     * NvSciBufImageAttrKey_SurfSampleType set to NvSciSurfSampleType_420
     * NvSciBufImageAttrKey_SurfComponentOrder set to NvSciSurfComponentOrder_YUV
     * NvSciBufImageAttrKey_SurfColorStd set to NvSciColorStd_REC709_ER
     * NvSciBufImageAttrKey_Layout set to NvSciBufImage_BlockLinearType
     *
     * @li The surface type for ISP1 output is set to the following by default if not
     * already set by the client. Note, if set by client it must only be to one of the supported
     * surface types of ISP output listed in @ref RegisterImages or an error will be generated.
     *
     * Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard | Sensor Type
     * ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | -------------- | -----------
     * YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * Y            | PITCH LINEAR | UINT      | PACKED      | N/A               | 16                 | Y               | REC709_ER      | RGB-IR
     *
     * YUV 420 SEMI-PLANAR UINT8 Block Linear is defined by NvSciBuf same as shown above for ISP0
     * Y PACKED UINT16 Pitch Linear is defined by NvSciBuf with:
     * NvSciBufImageAttrKey_PlaneCount set to 1U
     * NvSciBufImageAttrKey_PlaneColorFormat set with NvSciColor_Y16
     * NvSciBufImageAttrKey_PlaneColorStd set to NvSciColorStd_REC709_ER
     * NvSciBufImageAttrKey_Layout set to NvSciBufImage_PitchLinearType
     *
     * @li The surface type for ISP2 output is set to the
     * following by default if not already set by the client. Note, if set by client it must only be
     * to one of the supported surface types of ISP output listed in @ref RegisterImages or an error
     * will be generated.
     * note: ISP2 output is not supported for RGB-IR sensor
     * Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard | Sensor Type
     * ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | -------------- | -----------
     * RGBA         | PITCH LINEAR | FLOAT     | PACKED      | NONE              | 16                 | RGBA            | SENSOR_RGBA    | RGB Bayer
     *
     * RGBA PACKED FLOAT16 Pitch Linear is defined by NvSciBuf with:
     * NvSciBufImageAttrKey_PlaneCount set to 1U
     * NvSciBufImageAttrKey_PlaneColorFormat set with NvSciColor_Float_A16
     * NvSciBufImageAttrKey_PlaneColorStd set to NvSciColorStd_SENSOR_RGBA
     * NvSciBufImageAttrKey_Layout set to NvSciBufImage_PitchLinearType
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus GetImageAttributes(uint32_t index,
                                          INvSIPLClient::ConsumerDesc::OutputType const outType,
                                          NvSciBufAttrList &imageAttr) = 0;

    /** @brief Read from an EEPROM in a camera module.
     *
     * This method is used to perform data reads from an EEPROM in a camera
     * module.
     *
     *  @pre This function can only be called after @ref Init() but before @ref Start().
     *
     * @param[in] index The ID of the sensor to which the EEPROM is associated.
     * @param[in] address The start address to read from in the EEPROM.
     * @param[in] length Contiguous size of data to be read. [byte]
     * @param[out] buffer Buffer that EEPROM data is to be written into, must be
     *                    at least size length.
     *
     * @returns::SIPLStatus. The completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus ReadEEPROMData(uint32_t const index,
                                      uint16_t const address,
                                      uint32_t const length,
                                      uint8_t * const buffer) = 0;

    /** @brief Register images.
     *
     * This method is used to register the images to be used within the image processing
     * pipelines. These images serve as the output of ISP or as the output of ICP and input to ISP.
     *
     * @pre This function must be called after @ref Init() and before @ref Start(). Additionally, if ISP
     * output is enabled it must be called before @ref RegisterAutoControlPlugin().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] outType The output which images are being registered; can be ICP, ISP0 or ISP1.
     * @param[in] images Vector of @ref NvSciBufObj to be registered.
     * @li Supported number of images that can be registered: [1, 64]
     * @li Supported surface formats for ISP0 output for both RGB Bayer and RGB-IR sensor:
     *
     * Row # | Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard
     * ----- | ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | --------------
     * 1     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER
     * 2     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER
     * 3     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER
     * 4     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER
     * 5     | YUV          | BLOCK LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER
     * 6     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER
     * 7     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 16                 | VUYX            | REC709_ER
     * 8     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER
     * 9     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER
     * 10    | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER
     * 11    | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER
     *
     * @li Supported surface formats for ISP1 output:
     *
     * Row # | Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard | Sensor Type
     * ----- | ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | -------------- | ------------
     * 1     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 2     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 3     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 4     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 5     | YUV          | BLOCK LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER      | RGB Bayer
     * 6     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER      | RGB Bayer
     * 7     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 16                 | VUYX            | REC709_ER      | RGB Bayer
     * 8     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 9     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 10    | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 11    | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 12    | Y            | PITCH LINEAR | UINT      | PACKED      | N/A               | 16                 | Y               | REC709_ER      | RGB-IR
     *
     * @li Supported surface formats for ISP2 output:
     * note: ISP2 output is not supported for RGB-IR sensor
     *
     * Row # | Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order | Color Standard | Sensor Type
     * ----- | ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | --------------- | -------------- | ------------
     * 1     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 2     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 3     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 4     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 420               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 5     | YUV          | BLOCK LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER      | RGB Bayer
     * 6     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 8                  | VUYX            | REC709_ER      | RGB Bayer
     * 7     | YUV          | PITCH LINEAR | UINT      | PACKED      | NONE              | 16                 | VUYX            | REC709_ER      | RGB Bayer
     * 8     | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 9     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 8                  | YUV             | REC709_ER      | RGB Bayer
     * 10    | YUV          | BLOCK LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 11    | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 444               | 16                 | YUV             | REC709_ER      | RGB Bayer
     * 12    | RGBA         | PITCH LINEAR | FLOAT     | PACKED      | NONE              | 16                 | RGBA            | SENSOR_RGBA    | RGB-Bayer
     *
     * @li Supported surface formats for ICP output:
     *
     * Row # | Surface Type | Layout       | Data Type | Memory      | Sub-sampling Type | Bits Per Component | Component Order
     * ----- | ------------ | ------------ | --------- | ----------- | ----------------- | ------------------ | ---------------
     * 1     | RAW          | PITCH LINEAR | UINT      | PACKED      | NONE              | 10                 | {RGGB, BGGR, GRBG, GBRG, RCCB, BCCR, CRBC, CBRC, CCCC, BGGI_RGGI, GBIG_GRIG, GIBG_GIRG, IGGB_IGGR, RGGI_BGGI, GRIG_GBIG, GIRG_GIBG, IGGR_IGGB}
     * 2     | RAW          | PITCH LINEAR | UINT      | PACKED      | NONE              | 12                 | {RGGB, BGGR, GRBG, GBRG, RCCB, BCCR, CRBC, CBRC, CCCC}
     * 3     | RAW          | PITCH LINEAR | UINT      | PACKED      | NONE              | 16                 | {RGGB, BGGR, GRBG, GBRG, RCCB, BCCR, CRBC, CBRC, CCCC}
     * 4     | YUV          | PITCH LINEAR | UINT      | SEMI PLANAR | 422               | 8                  | {YUV, YVU}
     * 5     | YUV          | PITCH LINEAR | UINT      | PACKED      | 422               | 8                  | {YUYV, YVYU, VYUY, UYVY}
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus RegisterImages(uint32_t const index,
                                      INvSIPLClient::ConsumerDesc::OutputType const outType,
                                      std::vector<NvSciBufObj> const &images) = 0;

    /** @brief Begin streaming from all sensors in the selected platform configuration.
     *
     * This method starts the streaming from sensors belonging to each device block
     * in the selected platform configuration, and starts the associated image processing pipelines.
     *
     * @pre This function must be called after @ref RegisterImages() is called for capture and enabled ISP outputs,
     * @ref RegisterAutoControlPlugin() (if ISP output is enabled), and @ref RegisterNvSciSyncObj().
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Async
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus Start() = 0;

    /** @brief Stop streaming from all sensors in the selected platform configuration.
     *
     * This method stops the streaming from sensors belonging to each device block
     * in the selected platform configuration, and stops the associated image processing pipelines.
     *
     * @pre This function must be called after @ref Start() and before @ref Deinit().
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Async
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: No
     *   - De-Init: Yes
     */
    virtual SIPLStatus Stop() = 0;

    /** @brief Deinitialize the API for the selected platform configuration.
     *
     * This method deinitializes the camera module(s) and deserializer for each device block in the
     * selected platform configuration, and deinitializes and destroys the image processing
     * pipelines.
     *
     * Any registered images are automatically deregistered and can be safely destroyed.
     *
     * @pre This function can be called anytime after Init().
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Async
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: No
     *   - De-Init: Yes
     */
    virtual SIPLStatus Deinit() = 0;

    /** @brief Get maximum size of error information
     *
     * This method queries the drivers for sizes of error information and returns the largest.
     * This size should be used to allocate buffers for requesting detailed errors.
     *
     * @pre This function must be called after @ref Init() and before @ref Start().
     *
     * @param[in]  devBlkIndex              Index of the device block associated with
     *                                      the deserializer to retrieve error size from.
     * @param[out] maxErrorSize             Maximum size of device error information, in bytes
     *                                      (0 if no valid size found).
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @see SIPLErrorDetails
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus GetMaxErrorSize(uint32_t const devBlkIndex,
                                       size_t & maxErrorSize) = 0;

    /** @brief Get the error interrupt event information for a GPIO activation.
     *
     * This method queries CDAC for the latest event code of a GPIO pin, called in response
     * to Deserializer, Serializer and/or Sensor error notification(s).
     *
     * This API is only supported on QNX with the Version 2 CDI API.
     * NVSIPL_STATUS_NOT_SUPPORTED is returned on all other platforms.
     *
     * @pre This function must be called after @ref Init() and before @ref Stop().
     *
     * @param[in]   devBlkIndex Index of the device block associated with
     *                          the error notification.
     * @param[in]   gpioIndex   Index of the CDAC Error GPIO that issued an
     *                          interrupt event notification.
     * @param[out]  event       The latest-updated CDAC GPIO event code.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus GetErrorGPIOEventInfo(uint32_t const devBlkIndex,
                                             uint32_t const gpioIndex,
                                             SIPLGpioEvent &event) = 0;

    /** @brief Get generic deserializer error information
     *
     * This method queries the driver for detailed error information and populates a provided buffer
     * for the deserializer associated with the device block index.
     * The contents, size written, and order are determined by the driver.
     *
     * If no error info is expected (max error size is 0), this may be called with null error info
     * to retrieve only the remote and link error information.
     *
     * It is expected that the provided buffer is the correct size for the driver-provided errors.
     *
     * @pre This function must be called after @ref Init() and before @ref Stop().
     *
     * @param[in]  devBlkIndex              Index of the device block associated with
     *                                      the deserializer to retrieve information from.
     * @param[out] deserializerErrorInfo    SIPLErrorDetails buffer to populate
     *                                      with error information
     *                                      and the size of data written.
     *                                      Zero size means that no valid data was written.
     * @param[out] isRemoteError            bool set to true if remote (serializer) error detected.
     * @param[out] linkErrorMask            uint8_t to store link mask for link error state
     *                                      (1 in index position indicates error;
     *                                      all 0 means no link error detected).
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus GetDeserializerErrorInfo(uint32_t const devBlkIndex,
                                                SIPLErrorDetails * const deserializerErrorInfo,
                                                bool & isRemoteError,
                                                uint8_t& linkErrorMask) = 0;

    /** @brief Gets generic module error information
     *
     * This method queries the drivers for detailed error information and populates a provided buffer
     * for module devices (sensor, serializer) associated with the index.
     * The contents, size written, and order are determined by each driver.
     *
     * It is expected that the provided buffers are the correct size for the driver-provided errors.
     *
     * A flag is provided indicating whether sensor, serializer, or both errors should be read.
     * If not read, the errorInfo may be null for that device.
     *
     * @pre This function must be called after @ref Init() and before @ref Stop().
     *
     * @param[in]  index                    ID of the sensor associated with the devices
     *                                      to retrieve error information from.
     * @param[out] serializerErrorInfo      Buffer to populate
     *                                      with serializer error information
     *                                      and the size of data written.
     *                                      Zero size means that no valid data was written.
     * @param[out] sensorErrorInfo          Buffer to populate
     *                                      with sensor error information
     *                                      and the size of data written.
     *                                      Zero size means that no valid data was written.
     * @param[in]  errorsToRead             Flag indicating which errors to read -
     *                                      sensor, serializer, or both.
     *                                      If this flag indicates that errors should be read,
     *                                      the corresponding SIPLErrorDetails must be valid.
     *                                      Default behavior is to read both.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus GetModuleErrorInfo(uint32_t const index,
                                          SIPLErrorDetails * const serializerErrorInfo,
                                          SIPLErrorDetails * const sensorErrorInfo,
                                          SIPLModuleErrorReadFlag const errorsToRead
                                           = NVSIPL_MODULE_ERROR_READ_ALL) = 0;

    /** @brief Disable a given link.
     *
     * This method disables a given link.
     *
     * @pre This function should only be called after @ref Start() and before @ref Stop().
     *
     * Error notifications to the client are dropped until @ref EnableLink() is called.
     *
     * @param[in] index The ID of the sensor.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus DisableLink(uint32_t index) = 0;

    /** @brief Enable a given link.
     *
     * This method enables a given link and, if reset is asserted, reconfigures
     * the camera module to reestablish the link.
     *
     * @pre This function should only be called after @ref Start() and before @ref Stop().
     *
     * Error notifications to the client that were disabled by @ref DisableLink() are resumed.
     *
     * @param[in] index The ID of the sensor.
     * @param[in] resetModule If true, reconfigure the camera module before enabling the link.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus EnableLink(uint32_t index, bool const resetModule) = 0;

#if !NV_IS_SAFETY
    /** @brief Control the LED on the associated camera module.
     *
     * This method tries to enable or disable the LED on the specific module.
     * It is valid only if there is an LED on the camera module and it is controlled by the sensor.
     *
     * @pre This function should only be called after @ref Start() and before @ref Stop().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] enable Enable or disable LED.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: Yes
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: No
     *   - Runtime: Yes
     *   - De-Init: No
     */
    virtual SIPLStatus ToggleLED(uint32_t index, bool enable) = 0;
#endif // !NV_IS_SAFETY

    /** @brief Default destructor. */
    virtual ~INvSIPLCamera() = default;

    /** @brief Fill an @ref NvSciSyncAttrList.
     *
     * The method is used to fetch the NvSciSync attributes required for compatiblility
     * with the underlying image processing pipelines.
     *
     * @pre This function must be called after @ref Init() and before @ref
     * RegisterNvSciSyncObj().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] outType The output for which NvSciSync attributes are being fetched.
     * @param[out] attrList Attribute list to be filled.
     * @param[in] clientType Waiter, signaler, or both.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus FillNvSciSyncAttrList(uint32_t index,
                                             INvSIPLClient::ConsumerDesc::OutputType const outType,
                                             NvSciSyncAttrList const attrList,
                                             NvSiplNvSciSyncClientType const clientType) = 0;

    /** @brief Register an @ref NvSciSyncObj.
     *
     * @pre This function must be called after @ref FillNvSciSyncAttrList() and before @ref
     * Start().
     *
     * @param[in] index The ID of the sensor.
     * @param[in] outType The output for which the NvSciSyncObj is being registered.
     * @param[in] syncobjtype Presync, EOF sync, or presync and EOF sync.
     * @param[in] syncobj The object to be registered.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus RegisterNvSciSyncObj(uint32_t index,
                                            INvSIPLClient::ConsumerDesc::OutputType const outType,
                                            NvSiplNvSciSyncObjType const syncobjtype,
                                            NvSciSyncObj const syncobj) = 0;



    /** @brief Retrieve custom interface provider for deserializer
     *
     * Retrieve the custom interface provider for the deserializer
     * associated with the specified device block index.
     * This allows for direct access to custom deserializer functionality.
     *
     * @pre This function must be called after @ref Init() and before @ref Start().
     *
     * @param[in] devBlkIndex           Index of the device block associated with the deserializer.
     * @param[out] interfaceProvider    The custom interface provider for this deserializer.
     *                                  May be nullptr if no custom interfaces are implemented.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus GetDeserializerInterfaceProvider(uint32_t const devBlkIndex,
                                                        IInterfaceProvider *&interfaceProvider) = 0;

    /** @brief Retrieve custom interface provider for module
     *
     * Retrieve the custom interface provider for the module
     * associated with the specified sensor index.
     * This allows for direct access to custom module functionality.
     *
     * @pre This function must be called after @ref Init() and before @ref Start().
     *
     * @param[in] index                 ID of the sensor associated with the module
     *                                  to retrieve interface from.
     * @param[out] interfaceProvider    The customer interface provider for this module.
     *                                  May be nullptr if no custom interfaces are implemented.
     *
     * @returns::SIPLStatus the completion status of the operation.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual SIPLStatus GetModuleInterfaceProvider(uint32_t const index,
                                                  IInterfaceProvider *&interfaceProvider) = 0;

}; // INvSIPLCamera

/** @} */
} // namespace nvsipl



#endif // NVSIPLCAMERA_HPP
