/*
 * Copyright (c) 2013-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVCOMPOSER_H
#define NVCOMPOSER_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#define NVCOMPOSER_MAX_LAYERS 63
#define NVCOMPOSER_MAX_DRM_FORMATS 128
#define NVCOMPOSER_MAX_DRM_MODIFIERS 7

/**
 * Defines the reasons DeepISP upscale failed.
 */
typedef enum {
    DEEPISP_REASON_NONE,
    DEEPISP_REASON_UNSUPPORTED_COLORSPACE,
    DEEPISP_REASON_UNSUPPORTED_FRAMERATE,
    DEEPISP_REASON_UNSUPPORTED_RESOLUTION,
    DEEPISP_REASON_INTERLACED_SOURCE,
    DEEPISP_REASON_NOT_REQUIRED,
    DEEPISP_REASON_NOT_SUPPORTED,
    DEEPISP_REASON_GFN,
    DEEPISP_REASON_BLACKLISTED_LAYER,
    DEEPISP_REASON_BLACKLISTED_KEYEVENT,
    DEEPISP_REASON_UNKNOWN_ERROR,
} deepisp_reason_t;

/**
 * Defines important information that affects the behaviour of composition.
 */
typedef enum {
    /**
     * Optimisation flag to inform the composer that the layer configuration
     * of the contents has changed. If this is not set then the composer can
     * assume that only the buffer contents have changed since the last frame,
     * and so certain optimisations can be performed.
     */
    NVCOMPOSER_FLAG_GEOMETRY_CHANGED = (1<<0),

    /**
     * Property flag to inform the composer that one or more of the input
     * buffers is located in VPR memory.
     */
    NVCOMPOSER_FLAG_PROTECTED = (1<<1),

    /**
     * Optimisation flag to inform the composer that the contents represent
     * the last pass of composition, meaning that the target buffer is the
     * bottom-most layer and will not subsequently be blending on top of any
     * other layers. The composer can use this information to optimise how it
     * writes the alpha component to the target buffer.
     */
    NVCOMPOSER_FLAG_LAST_PASS = (1<<2),

    /**
     * Requirement flag indicating that a content includes a layer with
     * YUV format with 16 bits per component.
     *
     * Note: The YUV format with 16 bits per component accommodates both
     *       YUV format with 10 bits per component and with 12 bits per
     *       component, resp. The bits are shifted towards MSB.
     */
    NVCOMPOSER_FLAG_YUV16 = (1<<3),

    /**
     * Requirement flag indicating that composition should be performed
     * with gamma correction (ie. the blend happens in linear space).
     */
    NVCOMPOSER_FLAG_GAMMA_CORRECTION = (1<<4),

    /**
     * Indicates that maximum display luminance is provided by client and
     * it should be used instead of the default value based on internal
     * nvcomposer backend logic.
     */
    NVCOMPOSER_FLAG_OUTPUT_CUSTOM_LUMINANCE = (1<<5),

    /**
     * Indicate the compositor should perform read pixels
     */
    NVCOMPOSER_FLAG_PREPARE_READ_PIXELS = (1 << 6),

    /**
     * Indicate that a content includes a solid color layer.
     */
    NVCOMPOSER_FLAG_SOLID_COLOR_LAYER = (1<<15),

    /**
     * Indicate the type of YUV packing requested.
     */
    NVCOMPOSER_FLAG_PACK_YUV420_8  = (1<<16),
    NVCOMPOSER_FLAG_PACK_YUV420_10 = (1<<17),
    NVCOMPOSER_FLAG_PACK_YUV422_12 = (1<<18),

    /**
     * Indicate that an output is expected in Rec.2020 color space. Hence
     * if doing RGB to YUV conversion, a conversion from RGB to Rec.2020 YUV
     * is performed instead of default conversion from RGB to Rec.709 YUV.
     */
    NVCOMPOSER_FLAG_OUTPUT_2020 = (1<<19),

    /**
     * Indicate a YUV444 target instead of RGB444.
     */
    NVCOMPOSER_FLAG_OUTPUT_YUV444 = (1<<20),

    /**
     * Indicate that the output is expected to be PQ encoded
     */
    NVCOMPOSER_FLAG_OUTPUT_HDR = (1<<21),

    /**
     * Indicate that a composition should be performed into limited range.
     */
    NVCOMPOSER_FLAG_OUTPUT_LIMITED = (1<<22),

    /**
     * Indicate that the linear space blending needs to be done for hdr layers
     */
    NVCOMPOSER_FLAG_FORCE_HDR_GCB = (1<<23),

    /**
     * Indicate that the content includes some layer with YUV format for which
     * composer should invoke deepisp to up-scale the luma plane
     */
    NVCOMPOSER_FLAG_SUPERRES_YUV = (1<<24),

    /**
     * Enable dumps of various internal composer information to file(s)
     * for this frame.
     */
    NVCOMPOSER_FLAG_ENABLE_DUMPS = (1<<25),

    /**
     * Indicate that intermediate RGB should be output alongside YUV.
     */
    NVCOMPOSER_FLAG_OUTPUT_AUXRGB = (1<<26),

    /**
     * Indicate that a metadata blob sent in target structure should be
     * scrambled into a target surface.
     */
    NVCOMPOSER_FLAG_SCRAMBLE_METADATA = (1<<27),

    /**
     * Indicates that the matrix for conversion to output primaries is provided
     * by client. This client matrix overrides the default conversion based
     * on internal nvcomposer backend logic.
     */
    NVCOMPOSER_FLAG_OUTPUT_CUSTOM_PRIMARIES = (1<<28),

    /**
     * Indicate that the output is expected to be IPT
     */
    NVCOMPOSER_FLAG_OUTPUT_IPT = (1<<29),

    /**
     * WAR Bug 2657990: Fill the target layer with black, but
     * otherwise change the state of the compositor as little as
     * possible in comparison to the situation, where this flag is not
     * set.
     */
    NVCOMPOSER_FLAG_OUTPUT_BLANK = (1<<30),

    /**
     * Indicate when blending mode is NVCOMPOSER_BLENDING_NONE, plane alpha value
     * is disabled
     */
     NVCOMPOSER_FLAG_PLANE_ALPHA_REQUIRES_BLENDING = (1 << 31),
} nvcomposer_flags_t;

#define NVCOMPOSER_FLAG_PACK_YUV                \
    (NVCOMPOSER_FLAG_PACK_YUV420_8 |            \
     NVCOMPOSER_FLAG_PACK_YUV420_10 |           \
     NVCOMPOSER_FLAG_PACK_YUV422_12)

#define NVCOMPOSER_FLAG_PACK_OUTPUT             \
    (NVCOMPOSER_FLAG_PACK_YUV |                 \
     NVCOMPOSER_FLAG_OUTPUT_YUV444)

#define NVCOMPOSER_FLAG_OUTPUT                  \
    (NVCOMPOSER_FLAG_PACK_OUTPUT |              \
     NVCOMPOSER_FLAG_OUTPUT_2020 |              \
     NVCOMPOSER_FLAG_OUTPUT_HDR |               \
     NVCOMPOSER_FLAG_OUTPUT_LIMITED |           \
     NVCOMPOSER_FLAG_OUTPUT_AUXRGB |            \
     NVCOMPOSER_FLAG_SCRAMBLE_METADATA |        \
     NVCOMPOSER_FLAG_OUTPUT_CUSTOM_PRIMARIES |  \
     NVCOMPOSER_FLAG_OUTPUT_IPT |               \
     NVCOMPOSER_FLAG_OUTPUT_BLANK)

/**
 * Defines the capabilites of a composer.
 */
typedef enum {
    /**
     * Composer can transform input layers.
     */
    NVCOMPOSER_CAP_TRANSFORM = (1<<0),

    /**
     * Composer can access buffers in VPR memory.
     */
    NVCOMPOSER_CAP_PROTECT = (1<<1),

    /**
     * Composer can write to a buffer in pitch linear layout.
     */
    NVCOMPOSER_CAP_PITCH_LINEAR = (1<<2),

    /**
     * Composer can read YUV surfaces with up to 16 bits per component.
     * See notes on NVCOMPOSER_FLAG_YUV16 for the formats this covers.
     */
    NVCOMPOSER_CAP_YUV16 = (1<<3),

    /**
     * Composer can write compressed content to the target buffer.
     */
    NVCOMPOSER_CAP_COMPRESS = (1<<4),

    /**
     * Composer can perform gamma correction (ie. blend in linear space).
     */
    NVCOMPOSER_CAP_GAMMA_CORRECTION = (1<<5),

    /**
     * Composer can perform csc using client provided lookup tables or csc matrix.
     */
    NVCOMPOSER_CAP_CLIENT_CSC = (1<<6),

    /**
     * Composer can process client provided metadata.
     */
    NVCOMPOSER_CAP_METADATA = (1<<7),

    /**
     * Composer supports gralloc buffers.
     */
    NVCOMPOSER_CAP_GRALLOC = (1<<8),

    /**
     * Composer supports dma-buf buffers.
     */
    NVCOMPOSER_CAP_DMABUF = (1<<9),

    /**
     * Composer supports chroma-loc in upsampling.
     */

    NVCOMPOSER_CAP_YUV_UPSAMPLE_MPEG1 = (1<<10),
    NVCOMPOSER_CAP_YUV_UPSAMPLE_BT601 = (1<<11),

    /**
     * Compositor supports read pixels
     */
    NVCOMPOSER_CAP_READ_PIXELS = (1<<12),

    /*
     * Keep values of subsequent caps and corresponding flags in sync.
     */

    /**
     * Composer can handle solid color layer.
     */
    NVCOMPOSER_CAP_SOLID_COLOR_LAYER  = NVCOMPOSER_FLAG_SOLID_COLOR_LAYER,

    /**
     * Composer can pack for HDMI YUV formatted TVs.
     */
    NVCOMPOSER_CAP_PACK_YUV420_8  = NVCOMPOSER_FLAG_PACK_YUV420_8,
    NVCOMPOSER_CAP_PACK_YUV420_10 = NVCOMPOSER_FLAG_PACK_YUV420_10,
    NVCOMPOSER_CAP_PACK_YUV422_12 = NVCOMPOSER_FLAG_PACK_YUV422_12,

    /**
     * Composer can output in Rec.2020 color space.
     */
    NVCOMPOSER_CAP_OUTPUT_2020 = NVCOMPOSER_FLAG_OUTPUT_2020,

    /**
     * Composer can output YUV444 format.
     */
    NVCOMPOSER_CAP_OUTPUT_YUV444  = NVCOMPOSER_FLAG_OUTPUT_YUV444,

    /**
     * Composer can perform composition for HDR buffers.
     */
    NVCOMPOSER_CAP_OUTPUT_HDR = NVCOMPOSER_FLAG_OUTPUT_HDR,

    /**
     * Composer can perform composition into limited range.
     */
    NVCOMPOSER_CAP_OUTPUT_LIMITED = NVCOMPOSER_FLAG_OUTPUT_LIMITED,

    /**
     * Composer can perform deepisp scale for some layer
     * TODO: Does cap bit for composer need to specify which format or BPC?
     */
    NVCOMPOSER_CAP_SUPERRES_YUV = NVCOMPOSER_FLAG_SUPERRES_YUV,

    /**
     * Composer can output intermediate RGB.
     */
    NVCOMPOSER_CAP_OUTPUT_AUXRGB = NVCOMPOSER_FLAG_OUTPUT_AUXRGB,

    /**
     * Composer can scramble metadata blob into target surface.
     */
    NVCOMPOSER_CAP_SCRAMBLE_METADATA = NVCOMPOSER_FLAG_SCRAMBLE_METADATA,

    /**
     * Composer accepts a custom matrix for conversion to output primaries.
     */
    NVCOMPOSER_CAP_OUTPUT_CUSTOM_PRIMARIES = NVCOMPOSER_FLAG_OUTPUT_CUSTOM_PRIMARIES,

    /**
     * Composer can output IPT color space.
     */
    NVCOMPOSER_CAP_OUTPUT_IPT = NVCOMPOSER_FLAG_OUTPUT_IPT,

    /**
     * Composer can output pure black.
     */
    NVCOMPOSER_CAP_OUTPUT_BLANK = NVCOMPOSER_FLAG_OUTPUT_BLANK,
} nvcomposer_caps_t;

/**
 * Tells what defines color conversion for layer.
 */
typedef enum {
    /**
     * Apply luma offset provided in nvcomposer_csc_t prior applying client
     * look-up table(s).
     */
    NVCOMPOSER_CSC_LUMA_OFFSET = (1<<0),
    /**
     * Conversions are defined through client provided look-up table(s).
     *
     * Blend lut is a standalone lut. Combined lut requires that both blend
     * and display luts are set.
     */
    NVCOMPOSER_CSC_USE_BLEND_LUT = (1<<1),
    NVCOMPOSER_CSC_USE_COMBINED_LUT = (1<<2),

    NVCOMPOSER_CSC_USE_LUT_MASK = (NVCOMPOSER_CSC_USE_BLEND_LUT |
                                   NVCOMPOSER_CSC_USE_COMBINED_LUT),
} nvcomposer_csc_flags_t;

/**
 * Defines names of common scalar data formats.
 */
typedef enum {
    NVCOMPOSER_DATA_FORMAT_UNKNOWN,
    NVCOMPOSER_DATA_FORMAT_UBYTE,
    NVCOMPOSER_DATA_FORMAT_USHORT,
    NVCOMPOSER_DATA_FORMAT_SSHORT,
    NVCOMPOSER_DATA_FORMAT_FLOAT,
    NVCOMPOSER_DATA_FORMAT_FP16,
} nvcomposer_data_format_t;

/**
 * Returns printable names of data formats.
 */
static inline const char*
nvcomposer_data_format_to_str(nvcomposer_data_format_t format)
{
    #define CASE(f) case NVCOMPOSER_DATA_FORMAT_##f: return #f

    switch (format) {
        CASE(UNKNOWN);
        CASE(UBYTE);
        CASE(USHORT);
        CASE(SSHORT);
        CASE(FLOAT);
        CASE(FP16);
    }

    #undef CASE

    // TODO: Print the numerical value of unknown format into the
    //       returned string. Jira MGTWOD-587.
    return "Undefined NVCOMPOSER data format";
}

/**
 * Lookup table(s) and matrices for color space conversion.
 */
typedef struct nvcomposer_lut {
    /* Dimension of LUT, set dim[2] = 1 for 2D LUT. */
    uint8_t dim[3];
    /* # of components per LUT element. Usually 3 or 4 (with alpha). */
    uint8_t numComponents : 3;
    /* # of bytes taken by each element, thus num_components*sizeof("format"). */
    uint8_t entrySize : 5;
    /* Size of dim[0] in bytes incl. alignment */
    int rowPitch;
    /* Size of dim[0]xdim[1] in bytes incl. alignment. */
    int slicePitch;
    /* Name of basic format of LUT components. */
    nvcomposer_data_format_t format;
    /* client owns the allocation. */
    uint8_t *data;
} nvcomposer_lut_t;

/**
 * Information for layer color space conversion defined by client.
 */
typedef struct nvcomposer_csc_luts {
    /* luma offset to be applied prior LUT instead of the 'regular' offset per
     * layer color space.
     */
    bool lumaOffsetValid;
    float lumaOffset;
    /* LUT for conversion from layer space to blend or display space. */
    nvcomposer_lut_t blendLut;
    /* LUT for conversion to display space - display management LUT, if any.
     * If both blend and display LUT are given, the LUTs are combined into
     * a single LUT before applying to layer pixels.
     */
    nvcomposer_lut_t displayLut;
} nvcomposer_csc_luts_t;

#define HDMI_MAX_METADATA_BLOB_SIZE       (1920)

typedef enum {
    /**
     * No HDR metadata is present.
     *
     * Note, that the gralloc buffer may still have static mastering display
     * metadata. Fix is tracked in Jira MGTWOD-550.
     */
    NVCOMPOSER_HDR_METADATA_NONE = (0<<0),
    /**
     * Metadata provided as a binary blob.
     */
    NVCOMPOSER_HDR_METADATA_BLOB = (1<<0),
} nvcomposer_hdr_metadata_type_t;

typedef struct nvcomposer_hdr_metadata {
    nvcomposer_hdr_metadata_type_t type;
    size_t size;
    union {
        uint8_t blob[HDMI_MAX_METADATA_BLOB_SIZE];
    };
} nvcomposer_hdr_metadata_t;

/**
 * Information for layer color space conversion defined by client.
 */
typedef struct nvcomposer_csc {
    // TODO: We should add to this struct what is the input and output color
    //   space to LUT conversion instead of setting a 'magic' color space
    //   in the composer backend based on use case. This requires creating
    //   color space API for nvcomposer interface: Jira MGTWOD-568.

    nvcomposer_csc_luts_t luts;
} nvcomposer_csc_t;

/**
 * Information for target color space conversion defined by client.
 */
typedef struct nvcomposer_target_csc {

    /* Max display luminance in nits to be used in color conversions. */
    float maxLuminance;
    /* Matrix for conversion from XYZ to display gammut */
    float matrix[3][3];
} nvcomposer_target_csc_t;

/*
 * Defines the layer flags.
 */
typedef enum {
    /**
     * This flags is set by the client to indicate that this layer
     * should be skipped in composition.
     */
    NVCOMPOSER_LAYER_FLAG_SKIP = (1 << 0),

    /**
     * This flags is set by the client to indicate that this layer
     * is used as the cursor. Its position can be updated asynchronously by the
     * client.
     */
    NVCOMPOSER_LAYER_IS_CURSOR = (1 << 1),

    /**
     * This flags is set by the client to indicate that the layer is
     * filled with a solid color, so that the compositor can apply
     * potential optimizations.
     */
    NVCOMPOSER_LAYER_FLAG_IS_SOLID_COLOR = (1 << 2),
} nvcomposer_layer_flags_t;

/**
 * Defines the layer blending modes.
 */
typedef enum {
    NVCOMPOSER_BLENDING_NONE     = 0x100,
    NVCOMPOSER_BLENDING_PREMULT  = 0x105,
    NVCOMPOSER_BLENDING_COVERAGE = 0x405,
} nvcomposer_blending_t;

/**
 * Defines the upsampling types for YUV format image
 */
typedef enum {
    /**
     * Indicate the chroma location for sub-sampled YUV formats
     * Ref. (VIC4.1 IAS, Sec 2.5.17.1).
     * MPEG1: Chroma locations are mid-way between even and odd luma values.
     * BT601: Chroma locations are co-sited with even luma values.
     */
    NVCOMPOSER_YUV_UPSAMPLE_MPEG1 = 0,
    NVCOMPOSER_YUV_UPSAMPLE_BT601 = 1,
} nvcomposer_yuv_upsample_type_t;

/**
 * Defines the layer orientation/rotation transform.
 */
typedef enum {
    NVCOMPOSER_TRANSFORM_NONE          = 0,
    NVCOMPOSER_TRANSFORM_FLIP_H        = (1<<0),
    NVCOMPOSER_TRANSFORM_FLIP_V        = (1<<1),
    NVCOMPOSER_TRANSFORM_ROT_90        = (1<<2),
    NVCOMPOSER_TRANSFORM_ROT_180       = (NVCOMPOSER_TRANSFORM_FLIP_H |
                                          NVCOMPOSER_TRANSFORM_FLIP_V),
    NVCOMPOSER_TRANSFORM_ROT_270       = (NVCOMPOSER_TRANSFORM_FLIP_H |
                                          NVCOMPOSER_TRANSFORM_FLIP_V |
                                          NVCOMPOSER_TRANSFORM_ROT_90),
    NVCOMPOSER_TRANSFORM_TRANSPOSE     = (NVCOMPOSER_TRANSFORM_FLIP_V |
                                          NVCOMPOSER_TRANSFORM_ROT_90),
    NVCOMPOSER_TRANSFORM_INV_TRANSPOSE = (NVCOMPOSER_TRANSFORM_FLIP_H |
                                          NVCOMPOSER_TRANSFORM_ROT_90),
} nvcomposer_transform_t;

/**
 * SuperRes filter types.
 */
typedef enum {
    NVCOMPOSER_FILTER_NONE,
    NVCOMPOSER_FILTER_LANCZOS,
    NVCOMPOSER_FILTER_DEEPISP,
} nvcomposer_filter_t;

/**
 * SuperRes layer options
 */
typedef struct nvcomposer_superres_layer_opts {
    nvcomposer_filter_t filter;
    float detail;
} nvcomposer_superres_layer_opts_t;

/**
 * SuperRes split modes
 */
typedef enum {
    NVCOMPOSER_SPLIT_MODE_NONE,
    NVCOMPOSER_SPLIT_MODE_HORIZONTAL,
    NVCOMPOSER_SPLIT_MODE_VERTICAL,
} nvcomposer_split_mode_t;

/**
 * SuperRes options
 */
typedef struct nvcomposer_superres_opts {
    /* Split screen control */
    nvcomposer_split_mode_t split_mode;
    float  split_line;
    int    split_side;
} nvcomposer_superres_opts_t;

/**
 * SuperRes feedback
 */
typedef struct nvcomposer_superres_feedback {
    nvcomposer_filter_t active_filter;
    deepisp_reason_t fallback_reason;
    nvcomposer_split_mode_t split_mode;
} nvcomposer_superres_feedback_t;

/**
 * Defines the buffer type of the layer and target buffers.
 */
typedef enum {
    NVCOMPOSER_BUFFERTYPE_INVALID,

    /**
     * Used for android gralloc buffers.
     * When populating the nvcomposer_buffer_t variable, the buffer_handle_t
     * should be cast directly to an nvcomposer_buffer_t pointer.
     *
     * For example:
     *     buffer_handle_t handle = ...
     *     nvcomposer_target_t target;
     *     target.buffer = (nvcomposer_buffer_t *) handle;
     *     target.buffertype = NVCOMPOSER_BUFFERTYPE_GRALLOC;
     */
    NVCOMPOSER_BUFFERTYPE_GRALLOC,

    /**
     * Used for linux dma-buf buffers.
     * For this buffer type, the buffer needs to be attached to the composer
     * before usage, and detached once no longer needed. This can be done
     * using the nvcomposer_attach_dmabuf() and nvcomposer_detach_dmabuf()
     * helper functions.
     *
     * For example:
     *    # init
     *    nvcomposer_buffer_t *buf = nvcomposer_attach_dmabuf(composer, ...)
     *
     *    # compose
     *    nvcomposer_target_t target;
     *    target.buffer = buf;
     *    target.buffertype = NVCOMPOSER_BUFFERTYPE_DMABUF;
     *
     *    # term
     *    nvcomposer_detach_dmabuf(composer, buf);
     *    buf = NULL;
     */
    NVCOMPOSER_BUFFERTYPE_DMABUF,
} nvcomposer_buffertype_t;

/**
 * Struct for specifying rectangle information.
*/
typedef struct nvcomposer_rect {
    float left;
    float top;
    float right;
    float bottom;
} nvcomposer_rect_t;

typedef struct nvcomposer_color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} nvcomposer_color_t;

/**
 * Abstraction of native sync object.
 * This currently supports only sync FD.
 */
typedef int nvcomposer_fence_t;

/**
 * Opaque type for a buffer.
 * Along with the nvcomposer_buffertype_t enum, this is used to
 * specify buffers for layers and target.
 */
typedef struct nvcomposer_buffer nvcomposer_buffer_t;

/**
 * Layer information required by compositors.
 */
typedef struct nvcomposer_layer {
    uint32_t                         flags;
    uint32_t                         cscFlags;
    uint32_t                         dataSpace;
    uint8_t                          planeAlpha;
    bool                             useSolidColor;
    nvcomposer_color_t               solidColor;
    nvcomposer_transform_t           transform;
    nvcomposer_blending_t            blending;
    nvcomposer_rect_t                sourceCrop;
    nvcomposer_rect_t                displayFrame;
    nvcomposer_buffer_t             *buffer;
    nvcomposer_buffertype_t          buffertype;
    nvcomposer_fence_t               acquireFence;
    nvcomposer_yuv_upsample_type_t   yuv_upsample_type;

    /* Used only by backends with NVCOMPOSER_CAP_SUPERRES_YUV */
    nvcomposer_superres_layer_opts_t superres;

    /* Used only by backends with NVCOMPOSER_CAP_CLIENT_CSC */
    nvcomposer_csc_t                 clientCsc;
} nvcomposer_layer_t;

/**
 * Defines the layer configuration of the composition contents.
 */
typedef struct nvcomposer_contents {
    /**
     * Number of input layers to compose.
     */
    int numLayers;

    /**
     * Minimum and maximum scale factor used by the input layers. Although this
     * is not neccesary for carrying out composition, it is useful to store this
     * information for validation against a composer's capabilites.
     */
    float minScale;
    float maxScale;

    /**
     * List of input layers to compose.
     *
     * This provides the composer with information about the configuration of
     * each layer, its input buffer, and synchronisation details. The composer
     * must use the acquireFence of each layer to ensure composition does not
     * start reading from the buffer until that sync object has been signalled.
     *
     * However the composer does NOT take ownership of that sync object, so
     * the client is still responsible for closing it after calling
     * nvcomposer_t.set.
     */
    nvcomposer_layer_t layers[NVCOMPOSER_MAX_LAYERS];

    /**
     * Destination rectangle to clip the composited layers against.
     */
    nvcomposer_rect_t clip;

    /**
     * Bitmask of nv_composer_flags_t.
     */
    uint32_t flags;

    /**
     * Frame number to use in e.g. debug dump functions.
     */
    uint16_t frame_number;

    /**
     * Frame rate to use in deepisp.
     */
    float frame_rate;

    /**
     * SuperRes options
     */
    nvcomposer_superres_opts_t *superres;
} nvcomposer_contents_t;

/**
 * Defines the properties of the composition target.
 */
typedef struct nvcomposer_target {
    /**
     * Destination buffer and buffer type for composition.
     */
    nvcomposer_buffer_t     *buffer;
    nvcomposer_buffertype_t  buffertype;

    /**
     * Auxiliary destination buffer for intermediate composition result.
     */
    nvcomposer_buffer_t     *auxrgb_buffer;
    nvcomposer_buffertype_t  auxrgb_buffertype;

    /**
     * The acquireFence sync object is used to ensure composition does not
     * start writing to the target surface until this sync object has been
     * signalled. The composer takes ownership of the sync object and will close
     * it when no longer needed.
     */
    nvcomposer_fence_t acquireFence;

    /**
     * The releaseFence sync object is used to return a sync object to the
     * client. It is this sync object that will be signalled once composition is
     * complete. The client takes ownership of this sync object and so is
     * responsible for closing it.
     */
    nvcomposer_fence_t releaseFence;

    /**
     * Used only by backends with NVCOMPOSER_CAP_CLIENT_CSC
     */
    nvcomposer_target_csc_t clientCsc;

    /**
     * Used only by backends with NVCOMPOSER_CAP_METADATA
     */
    nvcomposer_hdr_metadata_t metadata;

    /**
     * Bitmask of nv_composer_flags_t.
     */
    uint32_t flags;

    /**
     * SuperRes status Feedback
     */
    nvcomposer_superres_feedback_t superres;
} nvcomposer_target_t;

typedef struct nvcomposer nvcomposer_t;

/**
 * Tells the composer to release all resources and close.
 *
 * After calling this function, the client must no longer attempt to access
 * the nvcomposer_t instance as it may have been freed.
 */
typedef void
(* nvcomposer_close_t)(nvcomposer_t *composer);

/**
 * Tells the composer that target flags have possibly changed. The composer
 * can start doing any related preparation work already at this stage.
 *
 * A return value of 0 indicates success.
 * Any other value indicates there was an error, and that the flags have not
 * been set.
 */
typedef int
(* nvcomposer_set_target_flags_t)(nvcomposer_t *composer,
                                  unsigned flags);

 /**
 * Tells the composer to prepare for compositing the specified contents to
 * the specified target.
 *
 * A return value of 0 indicates success.
 * Any other value indicates there was an error, and that its not possible to
 * composite the contents with this composer.
 */
typedef int
(* nvcomposer_prepare_t)(nvcomposer_t *composer,
                         const nvcomposer_contents_t *contents,
                         const nvcomposer_target_t *target);

/**
 * Tells the composer to carry out composition of the specified contents and
 * output the results to the specified target.
 *
 * A return value of 0 indicates success.
 * Any other value indicates there was an error, and that composition was not
 * done.
 */
typedef int
(* nvcomposer_set_t)(nvcomposer_t *composer,
                     const nvcomposer_contents_t *contents,
                     nvcomposer_target_t *target);

/**
 * Attaches a dma-buf buffer to the composer, which is needed before such
 * buffers can be used in prepare/set.
 *
 * A return value of NULL indicates there was an error.
 */
typedef nvcomposer_buffer_t *
(* nvcomposer_attach_dmabuf_t)(nvcomposer_t *composer,
                               const uint32_t width,
                               const uint32_t height,
                               const uint32_t format,
                               const uint32_t num_planes,
                               const int *fds,
                               const uint32_t *strides,
                               const uint32_t *offsets,
                               const uint64_t *modifiers);

/**
 * Detaches a previously attached a dma-buf buffer from the composer.
 */
typedef void
(* nvcomposer_detach_dmabuf_t)(nvcomposer_t *composer,
                               nvcomposer_buffer_t *buffer);

typedef bool
(* nvcomposer_buffer_is_protected_t)(nvcomposer_t *composer,
                                     const nvcomposer_buffer_t *buffer);

typedef bool
(*nvcomposer_read_pixels_t)(nvcomposer_t* composer,
                            uint32_t format,
                            void *pixels,
                            uint32_t x,
                            uint32_t y,
                            uint32_t width,
                            uint32_t height);
/**
 * Defines the interface to an internal NVIDIA composer module.
 * The client does not create instances of this structure, but instead retrieves
 * a instance by calling the appropriate composer library creation function, for
 * example nvviccomposer_create() or glc_get().
 */
struct nvcomposer {
    /**
     * Identity string of the composer.
     */
    const char *name;

    /**
     * Bitmask of nv_composer_caps_t.
     */
    uint32_t caps;

    /**
     * Maximum number of layer that can be composited.
     */
    int maxLayers;

    /**
     * Minimum and maximum scale factor that can be applied to an input layer.
     */
    float minScale;
    float maxScale;

    /**
     * The DRM formats supported by the composer.
     */
    uint32_t supportedDrmFormats[NVCOMPOSER_MAX_DRM_FORMATS];
    size_t   numSupportedDrmFormats;

    /**
     * The DRM format modifiers supported by the composer.
     */
    uint64_t supportedDrmFormatModifiers[NVCOMPOSER_MAX_DRM_MODIFIERS];
    size_t   numSupportedDrmFormatModifiers;

    /**
     * Functions to operate the composer.
     * Depending on its implementation, the close and set_target_flags
     * functions may be NULL in which case the client needs to check before
     * calling them.
     * If the composer does not support dma-buf buffers, then the attach_dmabuf
     * and attach_dmabuf functions may also be NULL.
     */
    nvcomposer_close_t               close;
    nvcomposer_set_target_flags_t    set_target_flags;
    nvcomposer_prepare_t             prepare;
    nvcomposer_set_t                 set;
    nvcomposer_attach_dmabuf_t       attach_dmabuf;
    nvcomposer_detach_dmabuf_t       detach_dmabuf;
    nvcomposer_buffer_is_protected_t buffer_is_protected;
    nvcomposer_read_pixels_t         read_pixels;
};

/**
 * Helper function for initialising nvcomposer_contents_t struct.
 */
static inline void
nvcomposer_contents_init(nvcomposer_contents_t *contents)
{
    contents->numLayers = 0;
    contents->minScale = 1.0f;
    contents->maxScale = 1.0f;
    contents->clip.left = 0.0f;
    contents->clip.top = 0.0f;
    contents->clip.right = 0.0f;
    contents->clip.bottom = 0.0f;
    contents->flags = 0;
    memset(contents->layers, 0, sizeof(contents->layers));
}

/**
 * Helper function for initialising nvcomposer_target_t struct.
 */
static inline void
nvcomposer_target_init(nvcomposer_target_t *target)
{
    target->buffer = NULL;
    target->buffertype = NVCOMPOSER_BUFFERTYPE_INVALID;
    target->acquireFence = -1;
    target->releaseFence = -1;
    target->flags = 0;
}

/**
 * Helper function for calling composer close() operation.
 */
static inline void
nvcomposer_close(nvcomposer_t *composer)
{
    if (composer != NULL && composer->close != NULL)
        composer->close(composer);
}

/**
 * Helper function for calling composer prepare() operation.
 */
static inline int
nvcomposer_prepare(nvcomposer_t *composer,
                   const nvcomposer_contents_t *contents,
                   const nvcomposer_target_t *target)
{
    if (composer == NULL || composer->prepare == NULL)
        return -1;
    return composer->prepare(composer, contents, target);
}

/**
 * Helper function for calling composer set() operation.
 */
static inline int
nvcomposer_set(nvcomposer_t *composer,
               const nvcomposer_contents_t *contents,
               nvcomposer_target_t *target)
{
    if (composer == NULL || composer->set == NULL)
        return -1;
    return composer->set(composer, contents, target);
}

/**
 * Helper function for attaching dma-buf buffer.
 */
static inline nvcomposer_buffer_t *
nvcomposer_attach_dmabuf(nvcomposer_t *composer,
                         const uint32_t width,
                         const uint32_t height,
                         const uint32_t format,
                         const uint32_t num_planes,
                         const int *fds,
                         const uint32_t *strides,
                         const uint32_t *offsets,
                         const uint64_t *modifiers)
{
    if (composer == NULL || composer->attach_dmabuf == NULL)
        return NULL;
    return composer->attach_dmabuf(composer,
                                   width,
                                   height,
                                   format,
                                   num_planes,
                                   fds,
                                   strides,
                                   offsets,
                                   modifiers);
}

/**
 * Helper function for detaching dma-buf buffer.
 */
static inline void
nvcomposer_detach_dmabuf(nvcomposer_t *composer,
                         nvcomposer_buffer_t *buffer)
{
    if (composer != NULL && composer->detach_dmabuf != NULL)
        composer->detach_dmabuf(composer, buffer);
}

/**
 * Helper function for querying whether the buffer is
 * allocated from VPR memory.
 */
static inline bool
nvcomposer_buffer_is_protected(nvcomposer_t *composer,
                               const nvcomposer_buffer_t *buffer)
{
    if (composer != NULL && buffer != NULL)
        return composer->buffer_is_protected(composer, buffer);
    return false;
}

static inline bool
nvcomposer_read_pixels(nvcomposer_t* composer,
                       uint32_t format,
                       void *pixels,
                       uint32_t x,
                       uint32_t y,
                       uint32_t width,
                       uint32_t height)
{
    if (composer != NULL && pixels != NULL)
        return composer->read_pixels(composer, format, pixels,
                                     x, y, width, height);
    return false;
}

#ifdef __cplusplus
}
#endif

#endif /* NVCOMPOSER_H */
