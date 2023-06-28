/*
 * nvgldemo_win_screen.h
 *
 * Copyright (c) 2014-2016, NVIDIA CORPORATION. All rights reserved.
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

// Exposes QNX CAR 2.1 objects created by nvgldemo to applications
// which demonstrate its specific features.

#ifndef __NVGLDEMO_WIN_SCREEN_H
#define __NVGLDEMO_WIN_SCREEN_H

#include <sys/keycodes.h>
#include <screen/screen.h>

#ifdef EGL_EXT_stream_consumer_qnxscreen_window
#define NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(option, func, retvalue, rettype, print)    \
    do {                                               \
        if (!strncmp(demoOptions.proctype, "producer", NVGLDEMO_MAX_NAME)) { \
            return (rettype)retvalue;                   \
        }                                               \
        rettype (*fnptr)(void) = func;                  \
        if (option) {                                   \
            if (print) {                                \
                NvGlDemoLog(print);                     \
            }                                           \
            if (fnptr) {                                \
                return (*fnptr)();             \
            } else {                                    \
                return (rettype)retvalue;               \
           }                                            \
       }                                                \
    } while (0)

#define NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(option, func, print)                     \
    do {                                                                     \
        if (!strncmp(demoOptions.proctype, "producer", NVGLDEMO_MAX_NAME)) { \
            return;                                     \
        }                                               \
        void (*fnptr)(void) = func;                     \
        if (option) {                                   \
            if (print) {                                \
                NvGlDemoLog(print);                     \
            }                                           \
            if (fnptr) {                                \
                (*fnptr)();                             \
            }                                           \
            return;                                     \
        }                                               \
    } while (0)

#else
#define NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_RET_API(option, func, retvalue, rettype, print)
#define NVGLDEMO_EGL_EXT_EXEC_QNX_SCREEN_CONSUMER_NORET_API(option, func, print)
#endif //EGL_EXT_stream_consumer_qnxscreen_window


enum
{
    // Function keys
    NvGlDemoKeyCode_F1 = 128,
    NvGlDemoKeyCode_F2,
    NvGlDemoKeyCode_F3,
    NvGlDemoKeyCode_F4,
    NvGlDemoKeyCode_F5,
    NvGlDemoKeyCode_F6,
    NvGlDemoKeyCode_F7,
    NvGlDemoKeyCode_F8,
    NvGlDemoKeyCode_F9,
    NvGlDemoKeyCode_F10,
    NvGlDemoKeyCode_F11,
    NvGlDemoKeyCode_F12,

    // Arrow keys and family
    NvGlDemoKeyCode_LeftArrow,
    NvGlDemoKeyCode_RightArrow,
    NvGlDemoKeyCode_UpArrow,
    NvGlDemoKeyCode_DownArrow,
    NvGlDemoKeyCode_Home,
    NvGlDemoKeyCode_End,
    NvGlDemoKeyCode_PageUp,
    NvGlDemoKeyCode_PageDown,

    // These other keys have real ASCII codes, but better to put them here in
    // a centralized place than to have magic numbers floating around.
    NvGlDemoKeyCode_Escape = 27,
    NvGlDemoKeyCode_Delete = 127,
};

// Platform-specific state info
struct NvGlDemoPlatformState {
    screen_context_t screen_ctx;
    screen_display_t screen_display;
    screen_window_t screen_win;
    screen_event_t screen_ev;
};

#ifdef EGL_EXT_stream_consumer_qnxscreen_window
/*
 * Thread state options
 */
typedef enum {
    THREAD_NONE      = 0,      // Thread has not been created
    THREAD_CREATED   = 1,      // Thread has been created
    THREAD_RUN       = 2,      // Thread should run
    THREAD_STOP      = 4,      // Thread should stop
    THREAD_DESTROY   = 8,      // Thread should destroy itself
    THREAD_EXIT      = 16       // Thread has been exit
} ThreadState;


// Platform-specific state info
typedef struct NvGlDemoEglQnxScreenInfostruct {
    void          *thread;
    void          *sem;
    ThreadState    state;
    void          *arg;
}NvGlDemoEglQnxScreenInfo;
#endif //EGL_EXT_stream_consumer_qnxscreen_window
#endif // __NVGLDEMO_WIN_SCREEN_H

