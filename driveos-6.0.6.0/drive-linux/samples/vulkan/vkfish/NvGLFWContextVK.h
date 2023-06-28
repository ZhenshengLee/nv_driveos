//----------------------------------------------------------------------------------
// File:        NvVkUtil/NvGLFWContextVK.h
// SDK Version: v3.00
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2019, NVIDIA CORPORATION. All rights reserved.
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
#ifndef NV_GLFW_CONTEXT_H
#define NV_GLFW_CONTEXT_H
//#define GLEW_STATIC
/*#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#else
#include <GL/glxew.h>
#endif*/
#include "NvPlatformVK.h"
/*#define GLFW_INCLUDE_VULKAN 1
#define VK_VERSION_1_0 1
#include <GLFW/glfw3.h>

#include "NvAppBase/gl/NvAppContextGL.h"*/
#include "NvAppNativeContextVK.h"
#include "NvLogs.h"

#ifdef ENABLE_IVI_SHELL
    #include "ivi-application-client-protocol.h"
    #include <ilm_client.h>
#endif

struct WaylandData {
	struct wl_display *display;
	struct wl_registry *registry;
	struct wl_compositor *compositor;
	struct wl_surface *window;
	struct wl_shell *shell;
	struct wl_shell_surface *shell_surface;
#ifdef ENABLE_IVI_SHELL
    struct ivi_surface *ivi_surface;
    unsigned int ivi_surfaceId;
    struct ivi_application *ivi_application;
    uint32_t has_ivi_controller;
#endif
};

class NvVkMultibufferedRenderTarget;

class NvGLFWContextVK : public NvAppNativeContextVK {
public:
	NvGLFWContextVK(NvVKConfiguration config, const NvPlatformCategory::Enum plat, const NvPlatformOS::Enum os) :
		NvAppNativeContextVK(config, "App", NvPlatformInfo(plat, os))
	{
		mUseFBOPair = false;
		mFBOWidth = 0;
		mFBOHeight = 0;
		//mWindow = NULL;
		// NO GL
		//glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	}

	bool createDirect2DisplaySurface(uint32_t width, uint32_t height, VkSurfaceKHR *surface);
	bool createWaylandSurface(VkSurfaceKHR *surface);
	bool createXlibSurface(uint32_t width, uint32_t height, VkSurfaceKHR *surface);

	/*void setWindow(GLFWwindow* window) {
		mWindow = window;
	}*/

	bool bindContext() {
		return true;
	}

	bool unbindContext() {
		return true;
	}

	bool swap() {
		return true;
	}

	bool setSwapInterval(int32_t interval) {
		return false;
	}

	int32_t width() {
		if (mUseFBOPair)
			return mFBOWidth;

		//int32_t w, h;
		//glfwGetFramebufferSize(mWindow, &w, &h);
		return 1920;
		//return w;
	}

	int32_t height() {
		if (mUseFBOPair)
			return mFBOHeight;

		//int32_t w, h;
		//glfwGetFramebufferSize(mWindow, &w, &h);
		return 1080;
		//return h;
	}

	virtual void* getCurrentPlatformContext() {
		return (void*)NULL;
	}

	virtual void* getCurrentPlatformDisplay() {
		return (void*)NULL;
	}

	virtual NvVkRenderTarget* mainRenderTarget();

	virtual VkSemaphore* getWaitSync() { return mUseFBOPair ? NULL : mainRenderTarget()->getWaitSemaphore(); }
	virtual VkSemaphore* getSignalSync() { return mUseFBOPair ? NULL : mainRenderTarget()->getSignalSemaphore(); }

	virtual bool useOffscreenRendering(int32_t w, int32_t h) {
		mUseFBOPair = true;
		mFBOWidth = w;
		mFBOHeight = h;
		return true;
	}

	virtual bool isRenderingToMainScreen() {
		return !mUseFBOPair;
	}

	virtual void platformReshape(int32_t& w, int32_t& h) {
		if (mUseFBOPair) {
			w = mFBOWidth;
			h = mFBOHeight;
		}

		if ((mCurrentWidth == (unsigned int)w) && (mCurrentHeight == (unsigned int)h))
			return;

		if (!reshape(w, h))
			return;

		mCurrentWidth = w;
		mCurrentHeight = h;
	}

	virtual bool readFramebufferRGBX32(uint8_t *dest, int32_t& w, int32_t& h);

protected:

	virtual bool initRenderTarget();

	NvVkMultibufferedRenderTarget* mSwapchainRenderTarget;
	//GLFWwindow* mWindow;
	uint32_t mFBOWidth;
	uint32_t mFBOHeight;
	bool mUseFBOPair;
	struct WaylandData mWaylandData;
};


#endif
