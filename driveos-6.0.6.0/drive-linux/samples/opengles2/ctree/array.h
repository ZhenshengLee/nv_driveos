/*
 * array.h
 *
 * Copyright (c) 2003-2012, NVIDIA CORPORATION. All rights reserved.
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
// Generic array abstraction
//

#ifndef __ARRAY_H
#define __ARRAY_H

// Data structure
typedef struct {
    int  elemCount;
    int  elemSize;
    int  buffSize;
    void *buffer;
} Array;

// Initialization and clean-up
//   (New/delete malloc and free the Array struct, where init/destroy don't.)
Array* Array_new(int elemsize);
void Array_delete(Array *o);
void Array_init(Array *o, int elemsize);
void Array_destroy(Array *o);
void Array_clear(Array *o);

// Access functions
//   (We can random read the array, but add or delete only the last item.)
void Array_push(Array *o, void *elem);
void *Array_get(Array *o, int i);
void Array_pop(Array *o);

#endif // ARRAY_H
