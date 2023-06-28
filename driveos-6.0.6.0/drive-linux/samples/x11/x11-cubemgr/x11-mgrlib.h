/*
 * x11-mgrlib.h
 *
 * Copyright (c) 2012-2012, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __X11MGRLIB_H
#define __X11MGRLIB_H

// Data structure for managed window/stream
typedef struct {
    Window window;
    EGLImageKHR eglImage;
    GLuint texID;
    int winSize[2];
    int bufSize[2];
} MgrObject;

// List of window objects
typedef struct windowObj {
    Window window;
    int ready;
    struct windowObj *next;
} WindowObj;

extern int          damageBaseEvent;
extern GLuint       logoTex;
extern MgrObject   *bufferList;

//============================================================================

int
MgrLibInit(void);

void
MgrLibTerm(void);

int
MgrObjectListInit(
    int             alloc);

void
MgrObjectListTerm(void);

void CreateEglImage(
    MgrObject       *obj);

void MapWindow(
    Window          window);

void UnmapWindow(
    Window          window);

void ConfigureWindow(
    Window          window);

void MarkReady(
    Window          window,
    Damage          damage);

#endif // __X11MGRLIB_H
