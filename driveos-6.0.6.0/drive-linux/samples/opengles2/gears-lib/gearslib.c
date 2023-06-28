/*
 * gearslib.c
 *
 * Copyright (c) 2003-2014, NVIDIA CORPORATION. All rights reserved.
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
// This file illustrates basic OpenGLES2 rendering with a simple set
//   of rotating gears. The functions defined here are expected to
//   be invoked by a wrapper application, such as gears, which renders
//   the gears directly to a window, or gearscube, which applies the
//   gears image to the faces of a spinning cube.
//

#include "nvgldemo.h"
#include "gearslib.h"

// Camera orientation
#define VIEW_ROTX 20.0f
#define VIEW_ROTY 30.0f
#define VIEW_ROTZ  0.0f

// Distance to gears and near/far planes
#define VIEW_ZNEAR  5.0f
#define VIEW_ZGEAR 40.0f
#define VIEW_ZFAR  50.0f

// Depending on compile options, we either build in the shader sources or
//   binaries or load them from external data files at runtime.
//   (Variables are initialized to the file contents or name, respectively).
#ifdef USE_EXTERN_SHADERS
static const char gearVertShader[] = { VERTFILE(gears_vert) };
static const char gearFragShader[] = { FRAGFILE(gears_frag) };
#else
static const char gearVertShader[] = {
#   include VERTFILE(gears_vert)
};
static const char gearFragShader[] = {
#   include FRAGFILE(gears_frag)
};
#endif

static const char gearPrgBin[] = { PROGFILE(gears_prog) };

// Vertex data describing the gears
typedef struct {
    int      teeth;
    GLfloat  *vertices;
    GLfloat  *normals;
    GLushort *frontbody;
    GLushort *frontteeth;
    GLushort *backbody;
    GLushort *backteeth;
    GLushort *outer;
    GLushort *inner;
} Gear;

// Gear structures and matrices
static Gear* gear1 = NULL;
static Gear* gear2 = NULL;
static Gear* gear3 = NULL;
static GLfloat gear1_mat[16];
static GLfloat gear2_mat[16];
static GLfloat gear3_mat[16];
//static GLfloat normal_mat[16];

// Shader program to use for gears and indices of attributes
static GLuint gearShaderProgram = 0;
static GLuint mview_mat_index;
static GLuint material_index;
static GLuint pos_index;
static GLuint nrm_index;

//  Make a gear wheel.
//
//  Input:  inner_radius - radius of hole at center
//          outer_radius - radius at center of teeth
//          width - width of gear
//          teeth - number of teeth
//          tooth_depth - depth of tooth
static Gear*
makegear(
    GLfloat  inner_radius,
    GLfloat  outer_radius,
    GLfloat  width,
    int      teeth,
    GLfloat  tooth_depth)
{
    Gear     *gear;
    GLfloat  r0, r1, r2, da, hw;
    GLfloat  *vert, *norm;
    GLushort *index, *indexF, *indexB;
    int      i;

    // Create gear structure and arrays of vertex/index data
    gear = (Gear*)MALLOC(sizeof(Gear));
    gear->teeth = teeth;
    gear->vertices   = (GLfloat*) MALLOC(20*teeth*3*sizeof(GLfloat));
    gear->normals    = (GLfloat*) MALLOC(20*teeth*3*sizeof(GLfloat));
    gear->frontbody  = (GLushort*)MALLOC((4*teeth+2)*sizeof(GLushort));
    gear->frontteeth = (GLushort*)MALLOC(4*teeth*sizeof(GLushort));
    gear->backbody   = (GLushort*)MALLOC((4*teeth+2)*sizeof(GLushort));
    gear->backteeth  = (GLushort*)MALLOC(4*teeth*sizeof(GLushort));
    gear->outer      = (GLushort*)MALLOC((16*teeth+2)*sizeof(GLushort));
    gear->inner      = (GLushort*)MALLOC((4*teeth+2)*sizeof(GLushort));

    // Set up vertices
    r0 = inner_radius;
    r1 = outer_radius - 0.5f * tooth_depth;
    r2 = outer_radius + 0.5f * tooth_depth;
    hw = 0.5f * width;
    da = (GLfloat)(0.5f * PI / teeth);
    vert = gear->vertices;
    norm = gear->normals;
    for (i=0; i<teeth; ++i) {
        GLfloat angA, angB, angC, angD;
        GLfloat cosA, cosB, cosC, cosD;
        GLfloat sinA, sinB, sinC, sinD;
        GLfloat posA0[2], posA1[2], posB2[2], posD0[2], posD1[2], posC2[2];
        GLfloat nrmA[2],  nrmAB[2], nrmBC[2], nrmCD[2], nrmD[2];

        // Compute angles used by this tooth
        angA = (GLfloat)(i * 2.0f * PI / teeth);
        cosA = COS(angA);
        sinA = SIN(angA);
        angB = angA + da;
        cosB = COS(angB);
        sinB = SIN(angB);
        angC = angB + da;
        cosC = COS(angC);
        sinC = SIN(angC);
        angD = angC + da;
        cosD = COS(angD);
        sinD = SIN(angD);

        // Compute x/y positions of vertices for this tooth
        posA0[0] = r0 * cosA;
        posA0[1] = r0 * sinA;
        posA1[0] = r1 * cosA;
        posA1[1] = r1 * sinA;
        posB2[0] = r2 * cosB;
        posB2[1] = r2 * sinB;
        posC2[0] = r2 * cosC;
        posC2[1] = r2 * sinC;
        posD1[0] = r1 * cosD;
        posD1[1] = r1 * sinD;
        posD0[0] = r0 * cosD;
        posD0[1] = r0 * sinD;

        // Compute normals used by this tooth
        nrmA[0]  = COS(angA - 0.5f*da);
        nrmA[1]  = SIN(angA - 0.5f*da);
        nrmAB[0] = +(posB2[1] - posA1[1]);
        nrmAB[1] = -(posB2[0] - posA1[0]);
        nrmBC[0] = +(posC2[1] - posB2[1]);
        nrmBC[1] = -(posC2[0] - posB2[0]);
        nrmCD[0] = +(posD1[1] - posC2[1]);
        nrmCD[1] = -(posD1[0] - posC2[0]);
        nrmD[0]  = COS(angD + 0.5f*da);
        nrmD[1]  = SIN(angD + 0.5f*da);

        // Produce the vertices
        //   Outer face gets doubled to handle flat shading, inner doesn't.
        vert[0] = vert[3] = posA0[0];
        vert[1] = vert[4] = posA0[1];
        vert[2] = +hw;
        vert[5] = -hw;
        norm[0] = norm[3] = -posA0[0];
        norm[1] = norm[4] = -posA0[1];
        norm[2] = norm[5] = 0.0f;
        vert += 6;
        norm += 6;

        vert[0] = vert[3]  = vert[6] = vert[9]  = posA1[0];
        vert[1] = vert[4]  = vert[7] = vert[10] = posA1[1];
        vert[2] = vert[8]  = +hw;
        vert[5] = vert[11] = -hw;
        norm[0] = norm[3]  = nrmA[0];
        norm[1] = norm[4]  = nrmA[1];
        norm[6] = norm[9]  = nrmAB[0];
        norm[7] = norm[10] = nrmAB[1];
        norm[2] = norm[5]  = norm[8] = norm[11] = 0.0f;
        vert += 12;
        norm += 12;

        vert[0] = vert[3]  = vert[6] = vert[9]  = posB2[0];
        vert[1] = vert[4]  = vert[7] = vert[10] = posB2[1];
        vert[2] = vert[8]  = +hw;
        vert[5] = vert[11] = -hw;
        norm[0] = norm[3]  = nrmAB[0];
        norm[1] = norm[4]  = nrmAB[1];
        norm[6] = norm[9]  = nrmBC[0];
        norm[7] = norm[10] = nrmBC[1];
        norm[2] = norm[5]  = norm[8] = norm[11] = 0.0f;
        vert += 12;
        norm += 12;

        vert[0] = vert[3]  = vert[6] = vert[9]  = posC2[0];
        vert[1] = vert[4]  = vert[7] = vert[10] = posC2[1];
        vert[2] = vert[8]  = +hw;
        vert[5] = vert[11] = -hw;
        norm[0] = norm[3]  = nrmBC[0];
        norm[1] = norm[4]  = nrmBC[1];
        norm[6] = norm[9]  = nrmCD[0];
        norm[7] = norm[10] = nrmCD[1];
        norm[2] = norm[5]  = norm[8] = norm[11] = 0.0f;
        vert += 12;
        norm += 12;

        vert[0] = vert[3]  = vert[6] = vert[9]  = posD1[0];
        vert[1] = vert[4]  = vert[7] = vert[10] = posD1[1];
        vert[2] = vert[8]  = +hw;
        vert[5] = vert[11] = -hw;
        norm[0] = norm[3]  = nrmCD[0];
        norm[1] = norm[4]  = nrmCD[1];
        norm[6] = norm[9]  = nrmD[0];
        norm[7] = norm[10] = nrmD[1];
        norm[2] = norm[5]  = norm[8] = norm[11] = 0.0f;
        vert += 12;
        norm += 12;

        vert[0] = vert[3] = posD0[0];
        vert[1] = vert[4] = posD0[1];
        vert[2] = +hw;
        vert[5] = -hw;
        norm[0] = norm[3] = -posD0[0];
        norm[1] = norm[4] = -posD0[1];
        norm[2] = norm[5] = 0.0f;
        vert += 6;
        norm += 6;
    }

    // Build index lists for circular parts of front and back faces
    indexF = gear->frontbody;
    indexB = gear->backbody;
    for (i=0; i<teeth; i++) {
        indexF[0] = 20 * i;
        indexF[1] = 20 * i + 2;
        indexF[2] = 20 * i + 18;
        indexF[3] = 20 * i + 16;
        indexF += 4;
        indexB[0] = 20 * i + 1;
        indexB[1] = 20 * i + 3;
        indexB[2] = 20 * i + 19;
        indexB[3] = 20 * i + 17;
        indexB += 4;
    }
    indexF[0] = 0;
    indexF[1] = 2;
    indexB[0] = 1;
    indexB[1] = 3;

    // Build index lists for front and back sides of teeth
    indexF = gear->frontteeth;
    indexB = gear->backteeth;
    for (i=0; i<teeth; ++i) {
        indexF[0] = 20 * i + 2;
        indexF[1] = 20 * i + 6;
        indexF[2] = 20 * i + 10;
        indexF[3] = 20 * i + 14;
        indexF += 4;
        indexB[0] = 20 * i + 3;
        indexB[1] = 20 * i + 7;
        indexB[2] = 20 * i + 11;
        indexB[3] = 20 * i + 15;
        indexB += 4;
    }

    // Build index list for inner core
    index = gear->inner;
    for (i=0; i<teeth; i++) {
        index[0] = 20 * i;
        index[1] = 20 * i + 1;
        index[2] = 20 * i + 18;
        index[3] = 20 * i + 19;
        index += 4;
    }
    index[0] = 0;
    index[1] = 1;

    // Build index list for outsides of teeth
    index = gear->outer;
    for (i=0; i<teeth; i++) {
        index[0]  = 20 * i + 2;
        index[1]  = 20 * i + 3;
        index[2]  = 20 * i + 4;
        index[3]  = 20 * i + 5;
        index[4]  = 20 * i + 6;
        index[5]  = 20 * i + 7;
        index[6]  = 20 * i + 8;
        index[7]  = 20 * i + 9;
        index[8]  = 20 * i + 10;
        index[9]  = 20 * i + 11;
        index[10] = 20 * i + 12;
        index[11] = 20 * i + 13;
        index[12] = 20 * i + 14;
        index[13] = 20 * i + 15;
        index[14] = 20 * i + 16;
        index[15] = 20 * i + 17;
        index += 16;
    }
    index[0] = 2;
    index[1] = 3;

    return gear;
}

// Draw a gear
static void
drawgear(
    Gear *gear)
{
    int i;

    // Set up constant normals for the front and back of the gear
    static float norm_front[3] = {0.0f, 0.0f, 1.0f};
    static float norm_back[3]  = {0.0f, 0.0f, -1.0f};

    // Enable vertex pointers, and initially disable normal pointers
    glVertexAttribPointer(pos_index, 3, GL_FLOAT, 0, 0, gear->vertices);
    glVertexAttribPointer(nrm_index, 3, GL_FLOAT, 0, 0, gear->normals);
    glEnableVertexAttribArray(pos_index);
    glDisableVertexAttribArray(nrm_index);

    // Set the constant normal for front side
    glVertexAttrib3fv(nrm_index, norm_front);

    // Draw circular part of front side
    glDrawElements(GL_TRIANGLE_STRIP, 4*gear->teeth + 2,
                   GL_UNSIGNED_SHORT, gear->frontbody);

    // Draw front side teeth
    for (i=0; i<gear->teeth; i++) {
        glDrawElements(GL_TRIANGLE_FAN, 4,
                       GL_UNSIGNED_SHORT, &gear->frontteeth[4*i]);
    }

    // Set the constant normal for back side
    glVertexAttrib3fv(nrm_index, norm_back);

    // Draw circular part of front side
    glDrawElements(GL_TRIANGLE_STRIP, 4*gear->teeth + 2,
                   GL_UNSIGNED_SHORT, gear->backbody);

    // Draw back side teeth
    for (i = 0; i < gear->teeth; i++) {
        glDrawElements(GL_TRIANGLE_FAN, 4,
                       GL_UNSIGNED_SHORT, &gear->backteeth[4*i]);
    }

    // Enable normal pointers for the inner and outer faces
    glEnableVertexAttribArray(nrm_index);

    // Draw outer faces of teeth
    glDrawElements(GL_TRIANGLE_STRIP, 16*gear->teeth + 2,
                   GL_UNSIGNED_SHORT, gear->outer);

    // Draw inside radius cylinder
    glDrawElements(GL_TRIANGLE_STRIP, 4*gear->teeth + 2,
                   GL_UNSIGNED_SHORT, gear->inner);
}

// Free a gear structure
static void
freegear(
    Gear *gear)
{
    FREE(gear->inner);
    FREE(gear->outer);
    FREE(gear->backteeth);
    FREE(gear->backbody);
    FREE(gear->frontteeth);
    FREE(gear->frontbody);
    FREE(gear->normals);
    FREE(gear->vertices);
    FREE(gear);
}

// Top level initialization of gears library
int
gearsInit(int width, int height)
{
    // Scene constants
    const GLfloat light_pos[4] = {1.0f, 3.0f, 5.0f, 0.0f};

    GLuint  index;
    GLfloat scene_mat[16];
    GLfloat light_norm, light_dir[4];

    glClearColor(0.10f, 0.20f, 0.15f, 1.0f);

    // Load the shaders (The macro handles the details of binary vs.
    //   source and external vs. internal)
    gearShaderProgram = LOADPROGSHADER(gearVertShader, gearFragShader,
                                   GL_TRUE, GL_FALSE,
                                   gearPrgBin);

    // Use the program we just loaded
    if (!gearShaderProgram) return 0;
    glUseProgram(gearShaderProgram);

    // Initialize projection matrix
    gearsResize(width, height);

    // Using a directional light, so find the normalized vector and load
    light_norm = (GLfloat)(ISQRT(light_pos[0]*light_pos[0]
                                +light_pos[1]*light_pos[1]
                                +light_pos[2]*light_pos[2]
                                +light_pos[3]*light_pos[3]));
    light_dir[0] = light_pos[0] * light_norm;
    light_dir[1] = light_pos[1] * light_norm;
    light_dir[2] = light_pos[2] * light_norm;
    light_dir[3] = light_pos[3] * light_norm;
    index = glGetUniformLocation(gearShaderProgram, "light_dir");
    glUniform3fv(index, 1, light_dir);

    // Get indices for uniforms and attributes updated each frame
    mview_mat_index = glGetUniformLocation(gearShaderProgram, "mview_mat");
    material_index  = glGetUniformLocation(gearShaderProgram, "material");
    pos_index       = glGetAttribLocation(gearShaderProgram, "pos_attr");
    nrm_index       = glGetAttribLocation(gearShaderProgram, "nrm_attr");

    // Create gear data
    gear1 = makegear(1.0f, 4.0f, 1.0f, 20, 0.7f);
    gear2 = makegear(0.5f, 2.0f, 2.0f, 10, 0.7f);
    gear3 = makegear(1.3f, 2.0f, 0.5f, 10, 0.7f);

    // Set up the global scene matrix
    NvGlDemoMatrixIdentity(scene_mat);
    NvGlDemoMatrixTranslate(scene_mat, 0.0f, 0.0f, -VIEW_ZGEAR);
    NvGlDemoMatrixRotate(scene_mat, VIEW_ROTX, 1.0f, 0.0f, 0.0f);
    NvGlDemoMatrixRotate(scene_mat, VIEW_ROTY, 0.0f, 1.0f, 0.0f);
    NvGlDemoMatrixRotate(scene_mat, VIEW_ROTZ, 0.0f, 0.0f, 1.0f);

    // Set up the individual gear matrices
    MEMCPY(gear1_mat, scene_mat, 16*sizeof(GLfloat));
    NvGlDemoMatrixTranslate(gear1_mat, -3.0f, -2.0f, 0.0f);
    MEMCPY(gear2_mat, scene_mat, 16*sizeof(GLfloat));
    NvGlDemoMatrixTranslate(gear2_mat,  3.1f, -2.0f, 0.0f);
    MEMCPY(gear3_mat, scene_mat, 16*sizeof(GLfloat));
    NvGlDemoMatrixTranslate(gear3_mat, -3.1f,  4.2f, 0.0f);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);

    return 1;
}

// Sets up the projection matrix for the surface size
void
gearsResize(
    int width,
    int height)
{
    GLfloat proj_mat[16];
    GLfloat aspect;
    GLuint  index;

    NvGlDemoMatrixIdentity(proj_mat);
    if (width >= height) {
        aspect = (GLfloat)width / (GLfloat)height;
        NvGlDemoMatrixFrustum(proj_mat,
                              -aspect, aspect,
                              -1.0f, 1.0f,
                              VIEW_ZNEAR, VIEW_ZFAR);
    } else {
        aspect = (GLfloat)height / (GLfloat)width;
        NvGlDemoMatrixFrustum(proj_mat,
                              -1.0f, 1.0f,
                              -aspect, aspect,
                              VIEW_ZNEAR, VIEW_ZFAR);
    }
    index = glGetUniformLocation(gearShaderProgram, "proj_mat");
    glUniformMatrix4fv(index, 1, 0, proj_mat);
}

// Draw a frame
void
gearsRender(
    GLfloat angle)
{
    static GLfloat red  [3] = {0.8f, 0.1f, 0.0f};
    static GLfloat green[3] = {0.0f, 0.8f, 0.2f};
    static GLfloat blue [3] = {0.2f, 0.2f, 1.0f};
    GLfloat mview_mat[16];

    // Clear the buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Make sure gear shader program is current
    glUseProgram(gearShaderProgram);

    // Transform, color, and draw gear 1
    MEMCPY(mview_mat, gear1_mat, 16*sizeof(float));
    NvGlDemoMatrixRotate(mview_mat, angle, 0.0f, 0.0f, 1.0f);
    glUniformMatrix4fv(mview_mat_index, 1, 0, mview_mat);
    glUniform3fv(material_index, 1, red);
    drawgear(gear1);

    // Transform, color, and draw gear 2
    MEMCPY(mview_mat, gear2_mat, 16*sizeof(float));
    NvGlDemoMatrixRotate(mview_mat, -2.0f * angle - 9.0f, 0.0f, 0.0f, 1.0f);
    glUniformMatrix4fv(mview_mat_index, 1, 0, mview_mat);
    glUniform3fv(material_index, 1, green);
    drawgear(gear2);

    // Transform, color, and draw gear 3
    MEMCPY(mview_mat, gear3_mat, 16*sizeof(float));
    NvGlDemoMatrixRotate(mview_mat, -2.0f * angle - 25.0f, 0.0f, 0.0f, 1.0f);
    glUniformMatrix4fv(mview_mat_index, 1, 0, mview_mat);
    glUniform3fv(material_index, 1, blue);
    drawgear(gear3);
}

// Clean up graphics objects
void
gearsTerm(void)
{
    if (gearShaderProgram) { glDeleteProgram(gearShaderProgram); }
    if (gear1) freegear(gear1);
    if (gear2) freegear(gear2);
    if (gear3) freegear(gear3);
}
