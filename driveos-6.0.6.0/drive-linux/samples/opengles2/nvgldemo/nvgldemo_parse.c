/*
 * nvgldemo_parse.c
 *
 * Copyright (c) 2007-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//
// This file contains utility functions to parse the command line for
//   settings shared by multiple demos.
//

#include "nvgldemo.h"

// Global parsed options structure
NvGlDemoOptions demoOptions;

// Static variable to detect failure during parsing
static int parseFailed   = 0;
static int parseComplete = 0;

// For some systems, the command line argument list is constant.
//   Therefore we provide a utility to duplicate as a non-const list.
char**
NvGlDemoArgDuplicate(
    int argc, const char *const *argv)
{
    char** newargv;
    int i;

    newargv = (char**)MALLOC((argc + 1) * sizeof(char*));
    for (i=0; i<argc; ++i) {
        int len = 1 + STRLEN(argv[i]);
        newargv[i] = (char*)MALLOC(len);
        STRNCPY(newargv[i], argv[i], len);
    }
    newargv[argc] = NULL;

    return newargv;
}

// Extract and return an argument from the list
const char*
NvGlDemoArgExtract(
    int* argc, char** argv, int index)
{
    char* val = argv[index];
    int   j;
    for (j=index+1; j<*argc; ++j) {
        argv[j-1] = argv[j];
    }
    (*argc)--;
    return val;
}

// Advance past first part of argument, extracting if nothing is left.
void
NvGlDemoArgAdvance(
    int* argc, char** argv, int index, int count)
{
    argv[index] += count;
    if (argv[index][0] == 0)
        (void)NvGlDemoArgExtract(argc, argv, index);
}

// Check whether argument matches specified string. If so, advance.
int
NvGlDemoArgMatch(
    int* argc, char** argv, int index, const char* test)
{
    int len;

    if (parseFailed) return 0;

    len = STRLEN(test);
    if (!STRNCMP(argv[index], test, len)) {
        NvGlDemoArgAdvance(argc, argv, index, len);
        return 1;
    }

    return 0;
}

// Check whether argument matches. If so, obtain string value.
int
NvGlDemoArgMatchStr(
    int* argc, char** argv, int index, const char* test,
    const char* usage,
    unsigned int maxlen, char* value)
{
    const char* v;

    if (!NvGlDemoArgMatch(argc, argv, index, test)) return 0;

    if (index > *argc - 1) {
        NvGlDemoLog("Missing parameters (%s) for %s.\n", usage, test);
        parseFailed = 1;
        return 0;
    }

    v = NvGlDemoArgExtract(argc, argv, index);
    if (STRLEN(v) >= maxlen) {
        NvGlDemoLog("Value for %s for %s exceeds allowed length (%d)\n",
                    v, test, maxlen);
        parseFailed = 1;
        return 0;
    }

    STRNCPY(value, v, maxlen);
    return 1;
}

// Check whether argument matches. If so, obtain integer value(s).
int
NvGlDemoArgMatchInt(
    int* argc, char** argv, int index, const char* test,
    const char* usage, int minval, int maxval,
    unsigned int count, int* value)
{
    const char* v;
    unsigned int i;

    if (!NvGlDemoArgMatch(argc, argv, index, test)) return 0;

    if (index > *argc - (int)count) {
        NvGlDemoLog("Missing parameter (%s) for %s.\n", usage, test);
        parseFailed = 1;
        return 0;
    }

    for (i=0; i<count; ++i) {
        v = NvGlDemoArgExtract(argc, argv, index);
        value[i] = STRTOL(v, NULL, 10);
        if ((value[i] < minval) || (maxval < value[i])) {
            NvGlDemoLog("Invalid parameter (%s) for %s.\n", v, test);
            parseFailed = 1;
            return 0;
        }
    }
    return 1;
}

// Check whether argument matches. If so, obtain float value(s).
int
NvGlDemoArgMatchFlt(
    int* argc, char** argv, int index, const char* test,
    const char* usage, float minval, float maxval,
    unsigned int count, float* value)
{
    const char* v;
    unsigned int i;

    if (!NvGlDemoArgMatch(argc, argv, index, test)) return 0;

    if (index > *argc - (int)count) {
        NvGlDemoLog("Missing parameter (%s) for %s.\n", usage, test);
        parseFailed = 1;
        return 0;
    }

    for (i=0; i<count; ++i) {
        v = NvGlDemoArgExtract(argc, argv, index);
        value[i] = STRTOF(v, NULL);
        if ((value[i] < minval) || (maxval < value[i])) {
            NvGlDemoLog("Invalid parameter (%s) for %s.\n", v, test);
            parseFailed = 1;
            return 0;
        }
    }

    return 1;
}

// Usage string for common command line arguments
const char*
NvGlDemoArgUsageString(void)
{
    return
        "  Window system options (not all available on all platforms):\n"
        "    [-windowsize] [<xres> <yres>]                  (window size)\n"
        "    [-windowoffset <xpos> <ypos>]                  (window position)\n"
        "                                   *Not supported on Wayland*\n"

#ifdef NVGLDEMO_HAS_DEVICE
        "    [-displaysize or -screensize] [<xres> <yres>]  (display size)\n"
        "    [-disp <name>                                  (vga|dvi|lvds|tftlcd|"
                                                            "clilcd|dsi|hdmi|tv)\n"
        "                                                   (display name)\n"
        "    [-colorkey                                     (<r0> <g0> <b0> <a0> <r1> "
                                                            "<g1> <b1> <a1>)\n"
        "                                                   (display layer color key)\n"
	//Refresh, current mode and allocator options are not applicable in QNX
	#if !defined(__QNX__)
        "    [-refresh <rate>]                              (display refresh rate)\n"
        "    [-currentmode]                                 (use currently available"
                                                            "display mode)\n"
        "    [-allocator <name>]                            (gbm|drm)\n"
        "                                                   (allocator to use for initial\n"
        "                                                    dumb buffer)\n"
	#endif

#endif
#if defined(__QNX__) || defined(NVGLDEMO_HAS_DEVICE)
        "    [-layer <depth>]                               (display layer)\n"
        "    [-dispno <connector index>]                    (display output number"
                                                            "selection)\n"
#endif
//blendmode only applicable to QNX screen
#if defined(__QNX__) && !defined(NVGLDEMO_HAS_DEVICE)
        "    [-blendmode {none|constant|perpixel}]          (display blend mode)\n"
#endif
        "\n"
        "  Rendering options (availability varies by platform):\n"

#if defined(__QNX__) || defined(NVGLDEMO_HAS_DEVICE)
        "    [-alpha <alpha value>]                         (0.0 min, 1.0 max)\n"
        "                                                   (constant blending alpha)\n"
#endif
        "    [-buffering <n>]                               (n-buffered swaps)\n"
#if defined(__QNX__)
        "                                                   (Max 8 for screen windows,"
        "                                                   4 otherwise)\n"
#else
        "                                                   (Max 4)\n"
#endif
        "    [-msaa <samples>]                              (multi sampling)\n"
#ifdef EGL_NV_coverage_sample
        "    [-csaa <samples>]                              (coverage sampling)\n"
        "    [-vcaa <samples>]                              (coverage sampling)\n"
#endif
#if defined(NVGLDEMO_HAS_DEVICE)
        "    [-vpr <0/1>]                                   (1 enable, 0 disable)\n"
        "                                                   (not available for all apps)\n"
#endif
        "    [-useprogbin <boolean>]                        (program binary loading)\n"
#if 0
#ifdef EGL_EXT_stream_consumer_qnxscreen_window
        "    [-eglqnxscreentest <disable|enable>]           (eglqnxscreentest selection)\n"
        "    [-postflags <int>]                             (0 immediate, 1 wait_idle)\n"
        "    [-surfacetype <int>]                           (0 Normal, 1 Bottom_Origin)\n"
#endif //EGL_EXT_stream_consumer_qnxscreen_window
#endif
        "    [-smart <boolean>]                             (detect termination in\n"
        "                                                   cross-partition stream)\n"
        "    [-eglstreamsocket <socket name>]               (socket to get EGLStream from)\n"
        "    [-fifo <length>]                               (0 min, 5 max)\n"
        "    [-renderahead <length>]                        (-1 min, 10 max)\n"
        "                                                   (Max number of in-flight GPU\n"
        "                                                   frames in mailbox mode to\n"
        "                                                   throttle mailbox mode.)\n"
        "    [-latency <usec>]                              (0 min, 2147483647 max)\n"
        "    [-timeout <usec>]                              (0 min, 2147483647 max)\n"
        "    [-frames <#>]                                  (max numnber of frames to run)\n"
        "    [-ip <IP address>]                             (IP address)\n"
        "    [-port <int>]                                  (port number for multiple "
                                                            "consumers)\n"
        "    [-proctype <producer or consumer>]             (Producer or Consumer)\n"
#if !defined(__QNX__)
        "    [-surfaceid <n>]                               (0 min, 9999 max)\n"
#endif
        "    [-sec <seconds>]                               (0 forever, 3153600 max)\n"
        "    [-inactivity <secs>]                           (time to render on/off)\n"
#ifdef NVGLDEMO_HAS_DEVICE
        "    [-drmpermissive <value>]                       (set drm-nvdc permissive.\n"
        "                                                   Set to -1 to get the value\n"
        "                                                   from drm, otherwise\n"
        "                                                   force setting the value\n"
        "                                                   to 0 or 1 here)\n"
#endif
        "\n"
        "  Note:\n"
        "    Use of parameters which modify the display configuration\n"
        "    is primarily intended for use when running the application\n"
        "    in a standalone fashion.  When there is a separate display\n"
        "    server running, their use is discouraged, as they will blindly\n"
        "    change the state without regard to any other applications\n"
        "    sharing the display.\n"
        ;
}

// Parse the common arguments
int
NvGlDemoArgParse(
    int* argc, char** argv)
{
    char tmp[32];
    int i;

    // If parsing already done, skip.
    if (parseComplete) return 1;

    // Initialize options
    MEMSET(&demoOptions, 0, sizeof(demoOptions));
#if defined(__QNX__) && !defined(NVGLDEMO_HAS_DEVICE)
    demoOptions.displayLayer = -1;
#endif
    demoOptions.displayBlend = -1;
    demoOptions.displayAlpha = -1.0;
    demoOptions.displayColorKey[0] = -1.0;
    demoOptions.renderahead = -1;
    demoOptions.timeout         = 16000;
    demoOptions.flags  = 0;
    demoOptions.duration = -0.5f;
    demoOptions.nFifo = 1;
#if defined(NVGLDEMO_HAS_DEVICE) && !defined (__QNX__)
    demoOptions.displayNumber = -1;
#ifdef NVGLDEMO_IS_EMBEDDED_LINUX
    demoOptions.allocator = NvGlDemoAllocator_GBM;
#endif // defined(NVGLDEMO_IS_EMBEDDED_LINUX)
#endif // defined(NVGLDEMO_HAS_DEVICE) && !defined (__QNX__)
    // Currently drm-nvdc permissive value is not set by the user.
    // This means that the value from drm is used during drm device
    // creation
    demoOptions.isDrmNvdcPermissive = -1;
#ifdef EGL_EXT_stream_consumer_qnxscreen_window
    //Default value of eglqnx screen window consumer
    demoOptions.postWindowFlags = 1;
    // By Default egldevice surface rendering type is
    // Bottom_Origin. Need to configure Flip display in DC
    demoOptions.surfaceType = 1;
#endif //EGL_EXT_stream_consumer_qnxscreen_window

    //Default value of port
    demoOptions.port = 8888;

    demoOptions.inactivityTime = -1.0f;

    // Parse all recognized arguments. Skip unrecognized.
    for (i=1; i<*argc ; i++) {

        // Check for help Options
        if (NvGlDemoArgMatch(argc, argv, i, "--help") ||
            NvGlDemoArgMatch(argc, argv, i, "-help") ||
            NvGlDemoArgMatch(argc, argv, i, "-h")) {
            // Just bail out and display usage for the app
            return 0;
        }
    }


    // Parse all recognized arguments. Skip unrecognized.
    for (i=1; i<*argc && !parseFailed; /*nop*/) {

        // Window size
        if (NvGlDemoArgMatchInt(argc, argv, i, "-windowsize",
                                     "<xres> <yres>", 1, 9999,
                                     2, demoOptions.windowSize)) {
            // No additional action needed
        }

        // Window offset
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-windowoffset",
                                     "<xpos> <ypos>", -9999, +9999,
                                     2, demoOptions.windowOffset)) {
            demoOptions.windowOffsetValid = 1;
        }

        // Display size
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-displaysize",
                                     "<xres> <yres>", 1, 9999,
                                     2, demoOptions.displaySize) ||
                 NvGlDemoArgMatchInt(argc, argv, i, "-screensize",
                                     "<xres> <yres>", 1, 9999,
                                     2, demoOptions.displaySize)) {
            // No additional action needed
        }

        // Use Current Display Mode
        else if(NvGlDemoArgMatch(argc, argv, 1, "-currentmode")) {
             demoOptions.useCurrentMode = 1;
        }

        // Allocator for initial dumb buffer
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-allocator",
                                 "{gbm|drm}",
                                 sizeof(tmp), tmp)) {
            if (!STRCMP(tmp, "gbm")) {
                demoOptions.allocator = NvGlDemoAllocator_GBM;
            } else if (!STRCMP(tmp, "drm")) {
                demoOptions.allocator = NvGlDemoAllocator_DRM;
            } else {
                NvGlDemoLog("Unrecognized value (%s) for -allocator.\n"
                            "  Use gbm or drm.\n",
                            tmp);
                parseFailed = 1;
            }
        }

        // Display rate
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-refresh",
                                     "<rate>", 1, 240,
                                     1, &demoOptions.displayRate)) {
            // No additional action needed
        }

        // Display output number selection
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-dispno",
                                 "<connector index>", 0, 99,
                                     1, &demoOptions.displayNumber)) {
            // No additional action needed
        }

        // Display selection
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-disp",
                                 "{vga|dvi|lvds|tftlcd|clilcd|dsi|hdmi|tv}[#]",
                                 NVGLDEMO_MAX_NAME,
                                 demoOptions.displayName)) {
            // No additional action needed
        }

        // Display layer
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-layer",
                                     "<depth>", 0, 99,
                                     1, &demoOptions.displayLayer)) {
            // No additional action needed
        }

        // TODO: This is app specific, it should go in eglcrosspart code
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-smart",
                                     "<smart>", 0, 1,
                                     1, &demoOptions.isSmart)) {
            // No additional action needed
        }

        // Display layer blending mode
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-blendmode",
                                 "{none|constant|perpixel}",
                                 sizeof(tmp), tmp)) {
            if (!STRCMP(tmp, "none")) {
                demoOptions.displayBlend = NvGlDemoDisplayBlend_None;
            } else if (!STRCMP(tmp, "constant")) {
                demoOptions.displayBlend = NvGlDemoDisplayBlend_ConstAlpha;
            } else if (!STRCMP(tmp, "perpixel")) {
                demoOptions.displayBlend = NvGlDemoDisplayBlend_PixelAlpha;
            } else if (!STRCMP(tmp, "colorkey")) {
                demoOptions.displayBlend = NvGlDemoDisplayBlend_ColorKey;
            } else {
                NvGlDemoLog("Unrecognized value (%s) for -blendmode.\n"
                            "  Use none, constant, perpixel, or colorkey.\n",
                            tmp);
                parseFailed = 1;
            }
        }

        // Display layer constant blending factor
        else if (NvGlDemoArgMatchFlt(argc, argv, i, "-alpha",
                                     "<alpha>", 0.0f, 1.0f,
                                     1, &demoOptions.displayAlpha)) {
            // No additional action needed
        }

        // Display layer colorkey range
        else if (NvGlDemoArgMatchFlt(argc, argv, i, "-colorkey",
                                     "<r0> <g0> <b0> <a0> <r1> <g1> <b1> <a1>",
                                     0.0f, 1.0f,
                                     8, &demoOptions.displayColorKey[0])) {
            // No additional action needed
        }

        // Multi/super sampling
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-msaa",
                                     "<samples>", 1, 64,
                                     1, &demoOptions.msaa)) {
            // No additional action needed
        }

        // Coverage sampling
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-csaa",
                                     "<samples>", 1, 64,
                                     1, &demoOptions.csaa)) {
            // No additional action needed
        }
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-vcaa",
                                     "<samples>", 1, 64,
                                     1, &demoOptions.csaa)) {
            // No additional action needed
        }

        // n-buffered swapping
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-buffering",
                                     "<n>", 1, 8,
                                     1, &demoOptions.buffering)) {
            // No additional action needed
        }

        // Socket to get EGLStream from
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-eglstreamsocket",
                                     "<socket name>",
                                     sizeof(demoOptions.eglstreamsock),
                                     demoOptions.eglstreamsock)) {
            // No additional action needed
        }

        // Program binary loading
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-useprogbin",
                                     "<boolean>", 0, 1,
                                     1, &demoOptions.useProgramBin)) {
            // No additional action needed
        }

// EGL_EXT_stream_consumer_qnxscreen_window extension is deprecated
#if 0
#ifdef EGL_EXT_stream_consumer_qnxscreen_window
        // To enable the eglQnxScreenTest
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-eglqnxscreentest",
                                     "<disable|enable>",
                                     sizeof(tmp), tmp)) {
            if (!STRCMP(tmp, "disable")) {
                demoOptions.eglQnxScreenTest = 0;
            } else if (!STRCMP(tmp, "enable")) {
                demoOptions.eglQnxScreenTest = 1;
            } else {
                NvGlDemoLog("Unrecognized value (%s) for -eglqnxscreentest.\n"
                            "  Use disable, or enable.\n",
                            tmp);
                parseFailed = 1;
            }
        }

        // QNX CAR2 Screen post window flags
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-postflags",
                                     "<int>", 0, 1,
                                     1, &demoOptions.postWindowFlags)) {
            // No additional action needed
        }

        // Rendering surface type
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-surfacetype",
                                     "<int>", 0, 1,
                                     1, &demoOptions.surfaceType)) {
            // No additional action needed
        }
#endif //EGL_EXT_stream_consumer_qnxscreen_window
#endif

        // FIFO mode for EGLStreams
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-fifo",
                                     "<length>", 0, 5,
                                     1, &demoOptions.nFifo)) {
            // No additional action needed
        }

        // Egl Stream Consumer latency
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-latency",
                                     "<usec>", 0, 2147483647,
                                     1, &demoOptions.latency)) {
            // No additional action needed
        }

        // Egl Stream Consumer acquire timeout
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-timeout",
                                     "<usec>", 0, 2147483647,
                                     1, &demoOptions.timeout)) {
            demoOptions.flags = demoOptions.flags | NVGL_DEMO_OPTION_TIMEOUT;
            // No additional action needed
        }

        // To set producer/consumer
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-proctype",
                                 "{producer|consumer}[#]",
                                 NVGLDEMO_MAX_NAME,
                                 demoOptions.proctype)) {
            // No additional action needed
        }

        // Set IP. This is needed only on producer part.
        else if (NvGlDemoArgMatchStr(argc, argv, i, "-ip",
                                 "[#]",
                                 NVGLDEMO_MAX_NAME,
                                 demoOptions.ipAddr)) {
            // No additional action needed
        }

        //Set port number. This is only needed to run
        //multiple instances of producer and consumer
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-port",
                                     "<int>", 0, 65535,
                                     1, &demoOptions.port)) {
            // No additional action needed
        }

        // Max number of frames to run
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-frames",
                                     "<int>", 0, 99999,
                                     1, &demoOptions.frames)) {
            // No additional action needed
        }

        // To Set Mailbox mode renderahead
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-renderahead",
                                     "<length>", -1, 10,
                                     1, &demoOptions.renderahead)) {
            // No additional action needed
        }

#if !defined(__QNX__)
        // Surface ID for weston ivi-shell
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-surfaceid",
                                    "<n>", 0, 9999,
                                    1, &demoOptions.surface_id)) {
           // No additional action needed
        }
#endif

        else if (NvGlDemoArgMatchFlt(argc, argv, i, "-sec",
                                    "<seconds>", -31536000.0f, 31536000.0f,
                                    1, &demoOptions.duration)) {
            // No additional action needed
        }

        else if (NvGlDemoArgMatchFlt(argc, argv, i, "-inactivity",
                                    "<n>", 0.0f, 3600.0f,
                                    1, &demoOptions.inactivityTime)) {
           // No additional action needed
        }

#ifdef NVGLDEMO_HAS_DEVICE
        else if (NvGlDemoArgMatchInt(argc, argv, i, "-drmpermissive",
                                    "<int>", -1, 1,
                                    1, &demoOptions.isDrmNvdcPermissive)) {
           // No additional action needed
        }
#endif

        else if (NvGlDemoArgMatchInt(argc, argv, i, "-vpr",
                                    "<int>", 0, 1,
                                    1, &demoOptions.isProtected)) {
          // No additional action needed
        }

        // Argument not recognized, so skip it
        // @todo: why skip? isn't this bad?
        else {
            i++;
            //parseFailed = 1;
        }
    }

    if (demoOptions.nFifo != 0 && demoOptions.renderahead != -1) {
        NvGlDemoLog("  set -fifo 0 for -renderahead\n");
        parseFailed = 1;
    }
    parseComplete = 1;
    return !parseFailed;
}

// Query whether a failure occurred during parsing
int
NvGlDemoArgFailed(void)
{
    return parseFailed;
}
