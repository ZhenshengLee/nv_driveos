/*
 * cubemgr_vert.glslv
 *
 * Copyright (c) 2009-2012, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

uniform mat4  cameramat;
uniform mat4  objectmat;
uniform vec2  texscale;

attribute vec3 vtxpos;
attribute vec2 vtxtex;

varying vec2 texcoord;

void main() {

    // Adjust texture coordinates to preserve window's aspect ratio
    texcoord = texscale * vtxtex;

    // Position is transformed input vertex
    gl_Position = cameramat * (objectmat*vec4(vtxpos,1));
}
