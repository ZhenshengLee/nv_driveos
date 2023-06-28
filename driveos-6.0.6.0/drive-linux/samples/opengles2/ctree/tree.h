/*
 * tree.h
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
// Complete tree (branches + leaves) setup/rendering
//

#ifndef __TREE_H
#define __TREE_H

#include <GLES2/gl2.h>

// The parameters used to control the characteristics of the tree generated
typedef enum {
    TREE_PARAM_DEPTH,
    TREE_PARAM_BALANCE,
    TREE_PARAM_TWIST,
    TREE_PARAM_SPREAD,
    TREE_PARAM_LEAF_SIZE,
    TREE_PARAM_BRANCH_SIZE,
    TREE_PARAM_FULLNESS,
    NUM_TREE_PARAMS,
} enumTreeParams;
extern float treeParams[NUM_TREE_PARAMS];
extern float treeParamsMin[NUM_TREE_PARAMS];
extern float treeParamsMax[NUM_TREE_PARAMS];

// Initialization and clean-up
void Tree_initialize(GLuint bark, GLuint leaf, GLuint leafb);
void Tree_deinitialize(void);

// Control
void Tree_toggleVBO(void);
void Tree_setParam(int param, float val);

// Geometry setup
void Tree_newCharacter(void);
void Tree_build(void);

// Rendering
void Tree_draw(void);

#endif // __TREE_H
