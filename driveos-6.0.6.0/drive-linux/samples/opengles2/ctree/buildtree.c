/*
 * buildtree.c
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
// Generate pseudorandom tree
//

#include "nvgldemo.h"

#include "vector.h"
#include "random.h"

#include "buildtree.h"
#include "tree.h"
#include "branches.h"
#include "leaves.h"

// Branch thickness threshhold
static float treebuildThreshhold;

//////////////////////////////////////////////////////////////////////////////
//
// The BranchNoise object tracks the random numbers generated for each branch.
//
typedef struct BranchNoise {
    float noise;
    struct BranchNoise* left;
    struct BranchNoise* right;
} BranchNoise;

// Root BrancNoise object
static BranchNoise *bn = NULL;

// Initialization and clean-up.
static BranchNoise*
BranchNoise_new(void)
{
    BranchNoise *o = (BranchNoise *)MALLOC(sizeof(BranchNoise));
    o->left = NULL;
    o->right = NULL;

    o->noise = ((float) GetRandom()) * 0.3f - 0.1f;

    return o;
}

static void
BranchNoise_delete(
    BranchNoise *o)
{
    ASSERT(o);

    // Recursively delete all the branches.
    if (o->left) { BranchNoise_delete(o->left); }
    if (o->right) { BranchNoise_delete(o->right); }

    FREE(o);
}

// Provide access to left and right branches of the noise tree.
// If the branch doesn't exist, it'll be created.
static BranchNoise*
BranchNoise_left(
    BranchNoise *o)
{
    ASSERT(o);
    if (!o->left) { o->left = BranchNoise_new(); }
    ASSERT(o->left);
    return o->left;
}

static BranchNoise*
BranchNoise_right(
    BranchNoise *o)
{
    ASSERT(o);
    if (!o->right) { o->right = BranchNoise_new(); }
    ASSERT(o->right);
    return o->right;
}

//////////////////////////////////////////////////////////////////////////////
//
// Recursive tree generation
//

// Forward declaration
static void
build(
    int         lower[BRANCHES_FACETS + 1],
    BranchNoise *noise,
    float4x4    mat,
    float       texcoordY,
    float       decay,
    int         level);

// This one is called once per branch.
static void
buildBranch(
    float       radius,
    float       angle,
    float       twist,
    float       decay,
    int         level,
    float       texcoordY,
    float4x4    translateMat,
    BranchNoise *noise,
    int         upper[])
{
    float4x4 mat, scaleMat, rotMat;
    float dec;
    int i;

    // Generate transformation matrix
    makeScale(scaleMat, radius, radius, radius),
    makeRotation(rotMat, SIN(twist), COS(twist), 0.0f, angle);
    mult_f4x4(mat, scaleMat, rotMat);
    multi_f4x4(mat, translateMat);

    // Apply tapering factor
    dec = radius * decay;

    // If we have exceeded maximum branching or thickness is below
    //   threshhold, add leaves to it.
    if ((level + 1) >= BRANCH_DEPTH || dec < treebuildThreshhold)
    {
        Leaves_add(mat);
    }

    // Otherwise create more branches
    else
    {
        int lower[BRANCHES_FACETS + 1];

        build(lower, noise, mat, texcoordY, dec, level + 1);

        for (i = 0; i < BRANCHES_FACETS + 1; ++i)
        {
            Branches_addIndex(lower[i]);
            Branches_addIndex(upper[i]);
        }
    }
}

static void
build(
    int         lower[BRANCHES_FACETS + 1],
    BranchNoise *noise,
    float4x4    mat,
    float       texcoordY,
    float       decay,
    int         level)
{
    int upper[BRANCHES_FACETS + 1];
    float btwist, leftBranchNoise, rightBranchNoise;
    float branchAngle, branchAngleBias;
    float leftRadius, leftAngle, rightRadius, rightAngle;
    float taper, branchRadius;
    BranchNoise *leftNoise, *rightNoise;
    float4x4 translateMat;
    int i;

    ASSERT(noise);

    Leaves_setRadius(treeParams[TREE_PARAM_LEAF_SIZE]);

    btwist = treeParams[TREE_PARAM_TWIST] * (level + 1);

    leftNoise = BranchNoise_left(noise);
    rightNoise = BranchNoise_right(noise);

    leftBranchNoise = leftNoise->noise * treeParams[TREE_PARAM_FULLNESS];
    rightBranchNoise = rightNoise->noise * treeParams[TREE_PARAM_FULLNESS];

    branchAngle = treeParams[TREE_PARAM_BALANCE];
    branchAngleBias = treeParams[TREE_PARAM_SPREAD];

    leftRadius = SQRT(1.0 - branchAngle) + leftBranchNoise;
    leftRadius = clamp(leftRadius, 0.0f, 1.0f);
    leftAngle = (float)(branchAngle * branchAngleBias * PI / 2.0f);

    rightRadius = SQRT(branchAngle) + rightBranchNoise;
    rightRadius = clamp(rightRadius, 0.0f, 1.0f);
    rightAngle = (float)((branchAngle - 1.0f) * branchAngleBias * PI / 2.0f);

    taper = (leftRadius > rightRadius) ? leftRadius : rightRadius;

    branchRadius = treeParams[TREE_PARAM_BRANCH_SIZE];

    Branches_buildCylinder(lower, mat, taper, texcoordY, GL_TRUE);
    texcoordY += 1.0f - 2 * branchRadius;
    Branches_buildCylinder(upper, mat, taper, texcoordY, GL_FALSE);
    texcoordY += 2 * branchRadius;

    makeTranslate(translateMat, 0.0f, 0.0f, 1.0f);
    multi_f4x4(translateMat, mat);

    for (i = 0; i < BRANCHES_FACETS + 1; ++i)
    {
        Branches_addIndex(upper[i]);
        Branches_addIndex(lower[i]);
    }
    buildBranch(leftRadius, leftAngle, btwist,
                decay, level, texcoordY, translateMat, leftNoise, upper);
    buildBranch(rightRadius, rightAngle, btwist,
                decay, level, texcoordY, translateMat, rightNoise, upper);
}

void
BuildTree_generate(void)
{
    int lower[BRANCHES_FACETS + 1];
    float u, max, min;

    // compute the threshhold.
    u = 1.0f - treeParams[TREE_PARAM_DEPTH];
    u = u * u * u * u;
    max = 0.5f;
    min = 0.03f;
    treebuildThreshhold = (max - min) * u + min;

    // Create the noise object if it's not there yet.
    if (!bn) { bn = BranchNoise_new(); }

    // Build the tree branches.
    build(lower, bn, ident_matrix_f, 0.0f, 1.0f, 0);

    // Build the tree stump.
    Branches_generateStump(lower);
}

void
BuildTree_newCharacter()
{
    if (bn) { BranchNoise_delete(bn); bn = NULL; }
}
