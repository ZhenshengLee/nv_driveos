/*
 * nvgldemo_cqueue.c
 *
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
// This file contains the logic to implement the circular queue of indexes
//

#include "nvgldemo.h"

//rear points to the index of last element that's inserted
//front points to the index of the first element that can be deleted
static int front, rear, queue_size;

void NvGlDemoCqInitIndex(int size)
{
    front = -1;
    rear = -1;
    queue_size = size;
}

int NvGlDemoCqFull(void)
{
    if ((front == rear + 1) ||
       (front == 0 && rear == (queue_size - 1))) {
        return 1;
    }
    return 0;
}

int NvGlDemoCqEmpty(void)
{
    if (front == -1) {
       return 1;
    }
    return 0;
}

int NvGlDemoCqInsertIndex()
{
    if (NvGlDemoCqFull()) {
        return -1;
    } else {

        if (front == -1) {
            front = 0;
        }
        rear = (rear + 1) % queue_size;
        return rear;
   }
}

int NvGlDemoCqDeleteIndex()
{
    int index;

    if (NvGlDemoCqEmpty()) {
        return -1;
    } else {
        index = front;

        if (front == rear) {
            front = -1;
            rear = -1;
        } else {
            front = (front + 1) % queue_size;
        }
        return index;
    }
}
