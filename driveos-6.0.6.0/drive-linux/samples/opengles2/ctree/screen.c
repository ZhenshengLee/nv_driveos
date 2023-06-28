/*
 * screen.c
 *
 * Copyright (c) 2003-2016, NVIDIA CORPORATION. All rights reserved.
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
// Pseudo-random number generation
//

#include <GLES2/gl2.h>

#include "nvgldemo.h"
#include "nvtexfont.h"
#include "array.h"
#include "random.h"
#include "vector.h"
#include "screen.h"
#include "shaders.h"
#include "tree.h"
#include "leaves.h"
#include "branches.h"
#include "firefly.h"
#include "ground.h"
#include "sky.h"
#include "picture.h"
#include "slider.h"

#include "ground_img.h"
#include "ground_s_img.h"
#include "bark_img.h"
#include "bark_s_img.h"
#include "sky_night_img.h"
#include "sky_night_s_img.h"
#include "leaf_img.h"
#include "leaf_s_img.h"
#include "leaf_back_img.h"
#include "leaf_back_s_img.h"
#include "NVidiaLogo_img.h"
#include "label_balance_img.h"
#include "label_spread_img.h"
#include "label_leaf_size_img.h"
#include "label_branch_size_img.h"
#include "label_depth_img.h"
#include "label_twist_img.h"
#include "label_fullness_img.h"

// Maximum number of lights (fireflies)
// NOTE: any changes to NUM_LIGHTS must also be made to lighting_vert.glslv
#define NUM_LIGHTS 8

// Texture font
NVTexfontRasterFont *nvtxf = NULL;

// Scene textures (full size and small versions)
static unsigned char* texBark[]         = {bark_img,
                                            bark_s_img};
static unsigned char* texLeafFront[]    = {leaf_img,
                                            leaf_s_img};
static unsigned char* texLeafBack[]     = {leaf_back_img,
                                            leaf_back_s_img};
static unsigned char* texSky[]          = {sky_night_img,
                                            sky_night_s_img};
static unsigned char* texGround[]       = {ground_img,
                                            ground_s_img};

// Array of data structures for overlay texture info
static struct pictInfoStruct {
    unsigned char *filename;
    float left, right, bottom, top;
} pictInfo[] = {
    {label_depth_img,       0, 150, 260, 275},
    {label_balance_img,     0, 150, 220, 235},
    {label_twist_img,       0, 150, 180, 195},
    {label_spread_img,      0, 150, 140, 155},
    {label_leaf_size_img,   0, 150, 100, 115},
    {label_branch_size_img, 0, 150,  60,  75},
    {label_fullness_img,    0, 150,  20,  35},
    {NVidiaLogo_img,        0, 150, 450, 480},
};
#define NUM_PICTS (sizeof(pictInfo) / sizeof(struct pictInfoStruct))
static int const LOGO_PICT = NUM_PICTS - 1;

// Object containing a tree and a set of fireflies
typedef struct
{
    float   x;
    float   y;
    float   angle;
    Firefly fireflies[NUM_LIGHTS];
} TreePos;

// Create a new tree/firefly set
static TreePos*
TreePos_new(
    float nx,
    float ny,
    float a)
{
    int     i;
    TreePos *o = (TreePos*)MALLOC(sizeof(TreePos));
    o->x = nx;
    o->y = ny;
    o->angle = a;
    for (i = 0; i < NUM_LIGHTS; i++)
    {
        Firefly_init(o->fireflies + i, i);
    }

    return o;
}

static void
TreePos_delete(
    TreePos *o)
{
    int i;
    for (i = 0; i < NUM_LIGHTS; i++)
    {
        Firefly_destroy(o->fireflies + i);
    }
    FREE(o);
}

//////////////////////////////////////////////////////////////////////
// variables used in this module.

// Properties of the window.
static int width, height;
static int swapInterval = 0;

// light_count is the current number of lights (bugs) where
//   NUM_LIGHTS is the max number of lights.
static int lightCount = 3;

// camera properties.
static float3 eye = {0.0f, 1.0f, 7.0f};
static float heading = 0.0f;
static float pitch = 17.0f;
static float dh = 0.0f;
static float dp = 0.0f;
static float velocity = 0.0f;

// keep track of all the trees in the scene.
static Array treePosList;

// 2D items in the window.
static Slider *sliders[NUM_TREE_PARAMS];
static int selectedSlider = TREE_PARAM_DEPTH;
static Picture *picts[8];
static GLboolean overlayFlag = GL_TRUE;

// FPS is visible on screen
static GLboolean fpsFlag = GL_TRUE;

// FPS is displayed/logged on the text console/terminal
static GLboolean fpsLogFlag = GL_FALSE;

// Time tracking
#define FPS_PERIOD 40 // How many frames to recompute fps
static double startTime;
static double currentTime;
static double duration;
static double dt = 1.0 / 20.0;
static double fps = 0.0;
static double fpsTime;
static int    fpsCount = 0;
#ifdef DUMP_VERTS_PER_SEC
int vertCount = 0;
static double vps = 0.0;
#endif

// Use low res textures
static GLboolean smalltex = GL_FALSE;

// Don't render sky
static GLboolean nosky = GL_FALSE;

// Don't render menus
static GLboolean nomenu = GL_FALSE;

//////////////////////////////////////////////////////////////////////
// utility functions.

static void
setSlider(
    int       selectedSlider,
    float     val,
    GLboolean mustSet)
{
    if (Slider_setValue(sliders[selectedSlider], val) || mustSet)
    {
        Tree_setParam(selectedSlider, val);
    }
}

static void
tick(void)
{
    double        t;
    double        delta;

    // Get current time and compute delta since last frame
    t      = (double)SYSTIME() / ((long long)1000000000LL);
    if (!currentTime) currentTime = fpsTime = t;
    delta = t - currentTime;
    currentTime = t;

    // If framerate is less than one frame per second, we clamp the time
    //   delta used for the animation to 1, in order to prevent things
    //   from appearing to pop around randomly and make the controls
    //   somewhat usable.
    if (delta >= 1.0f) delta = 1.0;

    // Now smooth out the delta a bit
    if (delta != 0.0f)
        dt = dt * 0.7f + delta * 0.3f;
    else
        dt = 0.0f;

    // In theory, 1 / dt is the fps, but that changes too often.
    // We find out fps only once in FPS_PERIOD frames.
    if (++fpsCount == FPS_PERIOD)
    {
        double fpsDelta = t - fpsTime;
        fpsCount = 0;
        fpsTime = t;
        fps = FPS_PERIOD / fpsDelta;

#       ifdef DUMP_VERTS_PER_SEC
        vps = vertCount / fpsDelta;
        vertCount = 0;
        if (fpsLogFlag)
        {
            NvGlDemoLog("fps: %f        vps: %f\n", fps, vps);
        }
#       else
        if (fpsLogFlag)
        {
            NvGlDemoLog("fps: %f\n", fps);
        }
#       endif
    }
}

static void
refreshLights(void)
{
    glUseProgram(prog_solids);
    glUniform1i(uloc_solidsLights, lightCount);
    glUseProgram(prog_leaves);
    glUniform1i(uloc_leavesLights, lightCount);
}

//////////////////////////////////////////////////////////////////////
// initialize and de-initialize the module.

int
Screen_initialize(
    float     d,
    GLsizei   w,
    GLsizei   h,
    GLboolean fpsFlagRequest)
{
    int i, ypos;
    TreePos *treeposPtr;

    fpsFlag = fpsLogFlag = fpsFlagRequest;

    // Load shaders
    if (!LoadShaders()) return 0;

    // Initialize the module parameters.
    width  = w;
    height = h;

    //
    // 3D components initialization.
    //

    // Initialize fireflies
    Firefly_global_init(NUM_LIGHTS);

    // Initialize trees
    Tree_initialize(NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1, &texBark[smalltex]),
                    NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1, &texLeafFront[smalltex]),
                    NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1, &texLeafBack[smalltex]));
    Array_init(&treePosList, sizeof(TreePos));
    treeposPtr = TreePos_new(0.0f, 0.0f, 0.0f);
    Array_push(&treePosList, treeposPtr);
    TreePos_delete (treeposPtr);

    // Initialize sky
    if (!nosky) {
        Sky_initialize(NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1, &texSky[smalltex]));
    }

    // Initialize ground
    Ground_initialize(NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1, &texGround[smalltex]));

    // Time tracking initialization.
    startTime = currentTime = (double)SYSTIME() / ((long long)1000*1000000);
    fpsTime   = startTime;
    duration  = d;
    tick();

    // Slider initialization.
    for (i = 0, ypos = 240; i < NUM_TREE_PARAMS; i++, ypos -= 40)
    {
        sliders[i] = Slider_new(treeParamsMin[i],
                                treeParamsMax[i],
                                treeParams[i]);
        Slider_setPos(sliders[i],
                      0.0f, 150.0f,
                      (float) ypos, (float) (ypos + 20));
    }
    Slider_select(sliders[selectedSlider], GL_TRUE);

    // Picture initialization.
    for (i = (nomenu ? LOGO_PICT : 0); i < (int)NUM_PICTS; i++)
    {
        picts[i] = Picture_new(NvGlDemoLoadTgaFromBuffer(GL_TEXTURE_2D, 1,
                                               &pictInfo[i].filename));
        Picture_setPos(picts[i],
                       pictInfo[i].left, pictInfo[i].right,
                       pictInfo[i].bottom, pictInfo[i].top);
    }

    // Some GL initializations here.
    NvGlDemoSwapInterval(demoState.display, swapInterval);
    glViewport(0, 0, width, height);
    glClearDepthf(1.0f);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDepthRangef(0.0f, 1.0f);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    refreshLights();

    // All shaders use texture unit 0
    glUseProgram(prog_solids);
    glUniform1i(uloc_solidsTexUnit, 0);
    glUseProgram(prog_leaves);
    glUniform1i(uloc_leavesTexUnit, 0);
    glUseProgram(prog_simpletex);
    glUniform1i(uloc_simpletexTexUnit, 0);
    glUseProgram(prog_overlaytex);
    glUniform1i(uloc_overlaytexTexUnit, 0);

    nvtxf = nvtexfontInitRasterFont(NV_TEXFONT_DEFAULT, 0, GL_TRUE,
                                    GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR);

    return 1;
}

void
Screen_resize(
    GLsizei w,
    GLsizei h)
{
    width  = w;
    height = h;
    glViewport(0, 0, width, height);
}

void
Screen_deinitialize(void)
{
    int i;
    for (i = (nomenu ? LOGO_PICT : 0); i < (int)NUM_PICTS; i++) {
        Picture_delete(picts[i]);
    }
    if (!nomenu) {
        for (i = 0; i < NUM_TREE_PARAMS; i++) {
            Slider_delete(sliders[i]);
        }
    }
    Tree_deinitialize();

    Ground_deinitialize();

    if (!nosky)
        Sky_deinitialize();
    Array_destroy(&treePosList);

    Firefly_global_destroy();

    if (nvtxf != NULL)
        nvtexfontUnloadRasterFont(nvtxf);
}

static void
addTree(void)
{
    float scale = SKY_RADIUS - GROUND_SIZE;
    static float cone = 60.0f;

    float r = ((float) GetRandom()) * ((float) GetRandom()) * scale;
    float h = degToRadF(heading + (((float)GetRandom()) * cone - cone / 2.0f));
    TreePos *treeposPtr;
    treeposPtr = TreePos_new(COS(h) * r, SIN(h) * r,
                             ((float) GetRandom()) * 360.0f);
    Array_push(&treePosList, treeposPtr);
    TreePos_delete(treeposPtr);
}

static GLboolean
sliderControl(
    int key)
{
    if (nomenu) {
        return GL_TRUE;
    }

    switch (key)
    {
    case 'i':
    case 'k':
        {
        int dir = (key == 'i' ? -1 : 1);
        Slider_select(sliders[selectedSlider], GL_FALSE);
        selectedSlider += dir;
        if (selectedSlider == -1) { selectedSlider = NUM_TREE_PARAMS - 1; }
        if (selectedSlider == NUM_TREE_PARAMS) { selectedSlider = 0; }
        Slider_select(sliders[selectedSlider], GL_TRUE);
        return GL_TRUE;
        }

    case 'j':
    case 'l':
        {
        float val;
        val = treeParamsMax[selectedSlider]-treeParamsMin[selectedSlider];
        val *= 0.02f * (key == 'j' ? -1.0f : +1.0f);
        val += Slider_getValue(sliders[selectedSlider]);
        val = clamp(val, treeParamsMin[selectedSlider],
                         treeParamsMax[selectedSlider]),
        setSlider(selectedSlider, val, GL_FALSE);
        return GL_TRUE;
        }
    }
    return GL_FALSE;
}


static GLboolean cameraControl(int key)
{
    switch(key)
    {
    case 'j':
    case 'l':
        {
        float h = degToRadF(heading);
        float ch = COS(h);
        float sh = SIN(h);
        float3 right_vec = {ch, 0.0f, sh};
        float3 tmp;
        mult_f3f(tmp, right_vec, 3.0f);
        multi_f3f(tmp, (float)(key == 'l' ? dt : -dt));
        addi_f3(eye, tmp);
        return GL_TRUE;
        }

    case 'i':
    case 'k':
        velocity += key == 'i' ? 0.2f : -0.2f;
        return GL_TRUE;

    case 'I':
    case 'K':
        eye[1] += (float)(key == 'I' ? dt : -dt);
        return GL_TRUE;
    }
    return GL_FALSE;
}

static GLboolean sceneControl(int key)
{
    switch (key)
    {
    case ' ':
        if (!nomenu) {
            overlayFlag = !overlayFlag;
        }
        return GL_TRUE;

    case 'q': case 'Q':
        // Artificially set the start time so we'll terminate.
        if (duration <= 0.0) { duration = 0.1; }
        startTime = currentTime - duration;
        return GL_TRUE;

    case 'h': case 'H':
        NvGlDemoLog("\n"
            "  SPACE: toggles the controls overlay\n"
            "\n"
            "  If the controls overlay is on:\n"
            "    i : select next higher parameter bar\n"
            "    j : decrease current parameter\n"
            "    k : select next lower parameter bar\n"
            "    l : increase current parameter\n"
            "\n"
            "  If the controls overlay is off\n"
            "    i : move forwards\n"
            "    j : move left\n"
            "    k : move backwards\n"
            "    l : move right\n"
            "    I : move up\n"
            "    K : move down\n"
            "\n"
            "  c    : change \"character\" of the tree\n"
            "  f    : toggle frames/sec display\n"
            "  S    : output polygon count to terminal window\n"
            "  v    : decrease swap interval\n"
            "  V    : increase swap interval\n"
            "  1-8  : number of fireflies (colored point lights)\n"
            "  r    : toggle use of VBO\n"
            "  q    : quit\n"
            "\n");
        return GL_TRUE;

    case '+':
        addTree();
        return GL_TRUE;

    case '-':
        if (treePosList.elemCount>1) { Array_pop(&treePosList); }
        return GL_TRUE;

    case 'c':
        Tree_newCharacter();
        return GL_TRUE;

    case 'f':
        if (!nomenu) {
            fpsFlag = !fpsFlag;
        }
        fpsLogFlag = !fpsLogFlag;
        return GL_TRUE;

    case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8':
        lightCount = key - '0';
        refreshLights();
        return GL_TRUE;

    case 'S':
        {
        int polygons = Leaves_polyCount() +
                       Branches_polyCount() +
                       Ground_polyCount();
        NvGlDemoLog("frames per second   : %d\n", (int)fps);
        NvGlDemoLog("leaves              : %d\n", Leaves_leafCount());
        NvGlDemoLog("branches            : %d\n", Branches_branchCount());
        NvGlDemoLog("polygons per frame  : %d\n", polygons);
        NvGlDemoLog("polygons per second : %d\n", (int)(polygons * fps));

        return GL_TRUE;
        }

    case 'V':
    case 'v':
        {
            swapInterval += (key == 'V' ? 1 : -1);
            if (swapInterval < 0) { swapInterval = 0; }
            NvGlDemoSwapInterval(demoState.display, swapInterval);
            NvGlDemoLog("swap interval set to: %d\n", swapInterval);
            return GL_TRUE;
        }
    case 'r':
        Tree_toggleVBO();
        return GL_TRUE;
    }
    return GL_FALSE;
}

void
Screen_callback(
    int key,
    int x,
    int y)
{
    if (overlayFlag && sliderControl(key)) {
        return;
    }
    if (!overlayFlag && cameraControl(key)) {
        return;
    }
    sceneControl(key);
}

void
Screen_draw(void)
{
    const float s = 0.05f;
    double aspect = ((double)width)/((double)height);
    float  to_h;
    int    i, j;
    float  h, p, sh, ch, sp, cp;
    float3 forward_vec, tmp;
    float  scenemvp[16];
    float  treemvp[16];

    // Update the clock
    tick();

    // Clear the buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //
    // Adjust the camera so view direction shifts towards origin
    //

    velocity = velocity * 0.95f;

    dp = dp * 0.8f;

    to_h = (float)(ATAN2(-eye[0], eye[2]) * 180.0f/PI);
    dh = (to_h - heading);
    while (dh > 180.0f) {
        dh -= 360.0f;
    }
    while (dh <= -180.0f) {
        dh += 360.0f;
    }
    dh *= 20.0f;

    heading += (float)(dh*dt*0.05f);
    pitch += (float)(dp*dt*0.05f);

    h = degToRadF(heading);
    p = degToRadF(pitch);
    sh = SIN(h);
    ch = COS(h);
    sp = SIN(p);
    cp = COS(p);

    forward_vec[0] = sh*cp;
    forward_vec[1] = sp;
    forward_vec[2] = -ch*cp;

    mult_f3f(tmp, forward_vec, velocity);
    multi_f3f(tmp, (float)dt);
    addi_f3(eye, tmp);

    /* clamps */
    if (eye[1] < 0.5f) {
        eye[1] = 0.5f;
    }
    if (pitch > 80.0f) {
        pitch = 80.0f;
    }
    if (pitch < -80.0f) {
        pitch = -80.0f;
    }

    // Set up modelview/projection matrix for current camera settings
    NvGlDemoMatrixIdentity(scenemvp);
    NvGlDemoMatrixFrustum(scenemvp, -s * ((float)aspect), s * ((float)aspect),
                          -s, s,
                          0.1f, 1000.0f);
    NvGlDemoMatrixRotate(scenemvp, pitch, -1.0f, 0.0f, 0.0f);
    NvGlDemoMatrixRotate(scenemvp, heading, 0.0f, 1.0f, 0.0f);
    NvGlDemoMatrixTranslate(scenemvp, -eye[0], -eye[1], -eye[2]);
    NvGlDemoMatrixRotate(scenemvp, -90, 1.0f, 0.0f, 0.0f);

    // Enable depth testing for the scene
    glEnable(GL_DEPTH_TEST);

    // Render the trees and the ground beneath them
    for (j = 0; j < treePosList.elemCount; j++) {
        TreePos *treePos = (TreePos*)Array_get(&treePosList, j);
        Firefly* fireflies = treePos->fireflies;

        if(dt != 0.0f) {
            // Update firefly positions
            for (i=0; i<lightCount; ++i) {
                Firefly_move(fireflies + i);
            }
        }
        // Adjust modelview/projection for tree position/orientation
        MEMCPY(treemvp, scenemvp, sizeof(treemvp));
        NvGlDemoMatrixTranslate(treemvp, treePos->x, treePos->y, 0.0f);
        NvGlDemoMatrixRotate(treemvp, treePos->angle, 0.0f, 0.0f, 1.0f);

        // Set up branch/ground shader
        glUseProgram(prog_solids);
        glUniformMatrix4fv(uloc_solidsMvpMat, 1, GL_FALSE, treemvp);
        glUniform3fv(uloc_solidsLightPos, lightCount, fPos);
        glUniform3fv(uloc_solidsLightCol, lightCount, fColor);

        // Set up leaf shader
        glUseProgram(prog_leaves);
        glUniformMatrix4fv(uloc_leavesMvpMat, 1, GL_FALSE, treemvp);
        glUniform3fv(uloc_leavesLightPos, lightCount, fPos);
        glUniform3fv(uloc_leavesLightCol, lightCount, fColor);

        // Render the tree
        Tree_draw();
    }

    // Draw the sky
    if (!nosky) {
        glUseProgram(prog_simplecol);
        glUniformMatrix4fv(uloc_simplecolMvpMat, 1, GL_FALSE, scenemvp);
        glUseProgram(prog_simpletex);
        glUniformMatrix4fv(uloc_simpletexMvpMat, 1, GL_FALSE, scenemvp);
        Sky_draw();
    }

    // Fireflies must be rendered after the rest of the scene because
    //   they disable depth mask so as not to self-occlude.
    glUseProgram(prog_simplecol);
    for (j = 0; j < treePosList.elemCount; j++) {
        TreePos *treePos = (TreePos*)Array_get(&treePosList, j);
        Firefly* fireflies = treePos->fireflies;

        // Adjust modelview/projection for tree position/orientation
        MEMCPY(treemvp, scenemvp, sizeof(treemvp));
        NvGlDemoMatrixTranslate(treemvp, treePos->x, treePos->y, 0.0f);
        NvGlDemoMatrixRotate(treemvp, treePos->angle, 0.0f, 0.0f, 1.0f);
        glUniformMatrix4fv(uloc_simplecolMvpMat, 1, GL_FALSE, treemvp);

        // Draw fireflies
        for (i=0; i<lightCount; ++i) {
            Firefly_draw(fireflies + i);
        }
    }

    // Disable depth testing and turn on blendin for the overlay
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // Do the sliders and logo
    if (overlayFlag) {
        int i;
        for (i = 0; i < NUM_TREE_PARAMS; i++) {
            Slider_draw(sliders[i]);
        }
        for (i = 0; i < (int)NUM_PICTS; i++) {
            Picture_draw(picts[i]);
        }
    } else {
        Picture_draw(picts[LOGO_PICT]);
    }

    if (nvtxf && fpsFlag) {
        char buf[30];
        SNPRINTF(buf, 30, "fps : %5.1f", fps);
        nvtexfontRenderString_All(nvtxf, buf,
                                  0.4f, -0.9f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f);
    }

    glDisable(GL_BLEND);
}

void
Screen_setDemoParams(void)
{
    setSlider(TREE_PARAM_DEPTH,       0.56f,   GL_TRUE);
    setSlider(TREE_PARAM_BRANCH_SIZE, 0.1588f, GL_TRUE);
    setSlider(TREE_PARAM_FULLNESS,    0.62f,   GL_TRUE);
    lightCount = 8;
    fpsFlag = GL_FALSE;
    overlayFlag = GL_FALSE;
    swapInterval = 1;
    NvGlDemoSwapInterval(demoState.display, swapInterval);
}

void
Screen_setSmallTex(void)
{
    smalltex = 1;
}

void
Screen_setNoSky(void)
{
    nosky = 1;
}

void
Screen_setNoMenu(void)
{
    nomenu = 1;
    overlayFlag = GL_FALSE;
    fpsFlag = GL_FALSE;
}

GLboolean
Screen_isFinished()
{
    return (duration > 0.0) && (currentTime - startTime >= duration);
}
