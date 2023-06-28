//----------------------------------------------------------------------------------
// File:        vk10-kepler\ThreadedRenderingVk\assets\src_shaders/groundplane.glsl
// SDK Version: v3.00 
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
#GLSL_VS
#version 440 core

precision mediump float;

// INPUT
layout(location = 0) in vec2 a_vPosition; // In BaseShader
layout(location = 1) in vec2 a_vTexCoord; // In SceneColorShader

// OUTPUT
layout(location = 0) out vec2 v_vTexcoord;
layout(location = 1) out vec4 v_vPosWorldSpace;
layout(location = 2) out vec4 v_vPosEyeSpace;

// UNIFORMS
layout(set = 0, binding=0) uniform ProjBlock {
	mat4 u_mProjectionMatrix;
	mat4 u_mInverseProjMatrix;
	mat4 u_mViewMatrix;
	mat4 u_mInverseViewMatrix;
};

layout(set = 0, binding=1) uniform LightingBlock {
	vec4 u_vLightPosition;  
	vec4 u_vLightAmbient;   
	vec4 u_vLightDiffuse;   
	float u_fCausticOffset;
	float u_fCausticTiling;
};

void main()
{
    float size = 100.0f;
    float y = 0.0f;
	float tiling = 0.1f;

    v_vPosWorldSpace = vec4(a_vPosition.x * size, 0.0, a_vPosition.y * size, 1.0f) + u_mInverseViewMatrix[3];
	v_vPosWorldSpace.y = y;
    v_vPosWorldSpace.w = 1.0f;

	v_vTexcoord = (v_vPosWorldSpace.zx + a_vTexCoord)  * tiling;

    v_vPosEyeSpace = u_mViewMatrix * vec4(v_vPosWorldSpace.xyz, 1.0);
    gl_Position = u_mProjectionMatrix * v_vPosEyeSpace;
}

#GLSL_FS
#version 440 core

precision mediump float;

// INPUT
layout(location = 0) in vec2 v_vTexcoord;
layout(location = 1) in vec4 v_vPosWorldSpace;
layout(location = 2) in vec4 v_vPosEyeSpace;

// OUTPUT
layout(location = 0) out vec4 o_vFragColor;

// UNIFORMS
layout(binding = 2) uniform sampler2D u_tCaustic1Tex;
layout(binding = 3) uniform sampler2D u_tCaustic2Tex;
layout(binding = 4) uniform sampler2D u_tSandTex;

layout(set = 0, binding = 1) uniform LightingBlock {
	vec4 u_vLightPosition;  
	vec4 u_vLightAmbient;   
	vec4 u_vLightDiffuse;   
	float u_fCausticOffset;
	float u_fCausticTiling;
};

float getCausticIntensity(vec4 worldPos)
{
	vec2 sharedUV = (worldPos.xz * u_fCausticTiling);
	float caustic1 = texture(u_tCaustic1Tex, sharedUV + vec2(u_fCausticOffset, 0.0f)).r;
	float caustic2 = texture(u_tCaustic2Tex, sharedUV * 1.3f + vec2(0.0f, u_fCausticOffset)).r;
	return (caustic1 * caustic2);
}

float g_maxFogDist = 75.0f;
vec3 g_fogColor = vec3(0.0f, 0.0f, 0.64f);

float fogIntensity(float dist)
{
	float normalizedDist = clamp((dist / g_maxFogDist), 0.0f, 1.0f);
	// Use a linear falloff to give the ground plane more of a distant look
	return normalizedDist;// * normalizedDist * normalizedDist;
}

void main()
{
    o_vFragColor = texture(u_tSandTex, v_vTexcoord) + getCausticIntensity(v_vPosWorldSpace);
    float fog = fogIntensity(length(v_vPosEyeSpace.xyz));
    o_vFragColor.rgb = mix(o_vFragColor.rgb, g_fogColor, fog);
}
