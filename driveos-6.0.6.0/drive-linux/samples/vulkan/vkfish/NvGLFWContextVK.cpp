//----------------------------------------------------------------------------------
// File:        NvVkUtil/NvGLFWContextVK.cpp
// SDK Version: v3.00
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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
#ifndef ANDROID
#include "NvGLFWContextVK.h"
#include "NvVkRenderTargetImpls.h"
#include <string.h>
#include <unistd.h>
#if defined(VK_USE_PLATFORM_XLIB_KHR)
#include <X11/Xutil.h>
#endif

NvVkRenderTarget* NvGLFWContextVK::mainRenderTarget() {
	return mSwapchainRenderTarget;
}

// These are hard-coded for now.
// TODO: Add a new wrapper library like nvgldemo or nvwinsys that handles these neatly.
static const int winWidth  = 1920;
static const int winHeight = 1080;

#ifdef ENABLE_IVI_SHELL
static const int DEFAULT_IVI_SURFACE_ID = 9000;
#endif

#if defined(VK_USE_PLATFORM_DISPLAY_KHR)
/**
* Create a direct to display surface
*/
bool NvGLFWContextVK::createDirect2DisplaySurface(uint32_t width, uint32_t height, VkSurfaceKHR *surface) {
	uint32_t displayPropertyCount;

	// Get display properties
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice(), &displayPropertyCount, NULL);
	VkDisplayPropertiesKHR* pDisplayProperties = new VkDisplayPropertiesKHR[displayPropertyCount];
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice(), &displayPropertyCount, pDisplayProperties);

	// Get display plane properties
	uint32_t planePropertyCount;
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice(), &planePropertyCount, NULL);
	VkDisplayPlanePropertiesKHR* pPlaneProperties = new VkDisplayPlanePropertiesKHR[planePropertyCount];
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice(), &planePropertyCount, pPlaneProperties);

	VkDisplayKHR display = VK_NULL_HANDLE;
	VkDisplayModeKHR displayMode;
	VkDisplayModePropertiesKHR* pModeProperties;
	bool foundMode = false;

	for (uint32_t i = 0; i < displayPropertyCount; ++i) {
		display = pDisplayProperties[i].display;
		uint32_t modeCount = 0;
		vkGetDisplayModePropertiesKHR(physicalDevice(), display, &modeCount, NULL);
		pModeProperties = new VkDisplayModePropertiesKHR[modeCount];
		vkGetDisplayModePropertiesKHR(physicalDevice(), display, &modeCount, pModeProperties);

		for (uint32_t j = 0; j < modeCount; ++j) {
			const VkDisplayModePropertiesKHR* mode = &pModeProperties[j];
			if (mode->parameters.visibleRegion.width == width && mode->parameters.visibleRegion.height == height) {
				displayMode = mode->displayMode;
				foundMode = true;
				break;
			}
		}
		if (foundMode) {
			break;
		}
		delete [] pModeProperties;
	}

	if (!foundMode) {
		LOGE("Failed to find a suitable mode!\n");
		return false;
	}

	// Search for a best plane we can use
	uint32_t bestPlaneIndex = UINT32_MAX;
	VkDisplayKHR* pDisplays = NULL;

	for (uint32_t i = 0; i < planePropertyCount; i++) {
		uint32_t planeIndex=i;
		uint32_t displayCount;
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice(), planeIndex, &displayCount, NULL);
		if (pDisplays) {
			delete [] pDisplays;
		}
		pDisplays = new VkDisplayKHR[displayCount];
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice(), planeIndex, &displayCount, pDisplays);

		// Find a display that matches the current plane
		bestPlaneIndex = UINT32_MAX;
		for (uint32_t j = 0; j < displayCount; j++) {
			if (display == pDisplays[j]) {
				bestPlaneIndex = i;
				break;
			}
		}
		if (bestPlaneIndex != UINT32_MAX) {
			break;
		}
	}

	if (bestPlaneIndex == UINT32_MAX) {
		LOGE("Can't find a suitable display plane!\n");
		return false;
	}

	VkDisplayPlaneCapabilitiesKHR planeCap;
	vkGetDisplayPlaneCapabilitiesKHR(physicalDevice(), displayMode, bestPlaneIndex, &planeCap);
	VkDisplayPlaneAlphaFlagBitsKHR alphaMode;

	if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR) {
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR) {
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR;
	}
	else {
		alphaMode = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR;
	}

	VkDisplaySurfaceCreateInfoKHR surfaceInfo = {
		.sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
		.pNext = NULL,
		.flags = 0,
		.displayMode = displayMode,
		.planeIndex = bestPlaneIndex,
		.planeStackIndex = pPlaneProperties[bestPlaneIndex].currentStackIndex,
		.transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.globalAlpha = 1.0,
		.alphaMode = alphaMode,
		.imageExtent = {
			width,
			height,
		}
	};

	VkResult result = vkCreateDisplayPlaneSurfaceKHR(instance(), &surfaceInfo, NULL, surface);
	if (result !=VK_SUCCESS) {
		LOGE("Failed to create display plane surface!\n");
		return false;
	}

	delete[] pDisplays;
	delete[] pModeProperties;
	delete[] pDisplayProperties;
	delete[] pPlaneProperties;

	return true;
}
#endif // defined(VK_USE_PLATFORM_DISPLAY_KHR)

#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
static void registry_handle_global(void *data,
								   struct wl_registry *registry,
								   uint32_t name,
								   const char *interface,
								   uint32_t version) {
	struct WaylandData *wayland_data = (struct WaylandData *) data;
	if (wayland_data) {
		if (strcmp(interface, "wl_compositor") == 0) {
			wayland_data->compositor = (wl_compositor*) wl_registry_bind(registry, name, &wl_compositor_interface, 3);
		} else if (strcmp(interface, "wl_shell") == 0) {
			wayland_data->shell = (wl_shell*) wl_registry_bind(registry, name, &wl_shell_interface, 1);
		}
#ifdef ENABLE_IVI_SHELL
		else if (strcmp(interface, "ivi_application") == 0) {
			wayland_data->ivi_application = (ivi_application *)wl_registry_bind(registry, name, &ivi_application_interface, 1);
			// By default use hmi controller
			wayland_data->has_ivi_controller = 0;
			// weston 2.0 has ivi_controller_interface and weston 3.0 has ivi_wm_interface
			// Embedded Linux is still pointing to weston 2.0, so we need to support both
		} else if ((strcmp(interface, "ivi_wm") == 0) || (strcmp(interface, "ivi_controller") == 0)) {
			wayland_data->has_ivi_controller = 1;
		}
#endif
	}
}

static void registry_handle_global_remove(void *data,
										  struct wl_registry *registry,
										  uint32_t name) {}

static const struct wl_registry_listener registry_listener = {
	registry_handle_global, registry_handle_global_remove
};

static void handle_ping(void *data,
						struct wl_shell_surface *shell_surface,
						uint32_t serial) {
	wl_shell_surface_pong(shell_surface, serial);
}

static void handle_configure(void *data,
							 struct wl_shell_surface *shell_surface,
							 uint32_t edges,
							 int32_t width,
							 int32_t height) {}

static void handle_popup_done(void *data,
							  struct wl_shell_surface *shell_surface) {}

static const struct wl_shell_surface_listener shell_surface_listener = {
	handle_ping, handle_configure, handle_popup_done};

#ifdef ENABLE_IVI_SHELL
// TODO:
// This code is the same as the code in nvgldemo_win_wayland.c
// We can probably refactor these into a util Windowing library
static void handle_ivi_surface_configure(void *data, struct ivi_surface *ivi_surface, int32_t width, int32_t height) {}

static const struct ivi_surface_listener ivi_surface_listener = {
        handle_ivi_surface_configure,
};

static int create_ivi_surface(struct WaylandData* waylandData)
{
    uint32_t id_ivisurf = waylandData->ivi_surfaceId + (uint32_t)getpid();
    waylandData->ivi_surface = ivi_application_surface_create(waylandData->ivi_application, id_ivisurf, waylandData->window);

    if (waylandData->ivi_surface == NULL) {
        printf("Failed to create ivi_client_surface\n");
        return 0;
    }

    ivi_surface_add_listener(waylandData->ivi_surface, &ivi_surface_listener, waylandData);
    return 1;
}
#endif

/**
* Create a wayland surface
*/
bool NvGLFWContextVK::createWaylandSurface(VkSurfaceKHR *surface) {
	mWaylandData.display = wl_display_connect(NULL);

	if (mWaylandData.display == NULL) {
		LOGE("Failed to get wayland display!\n");
		return false;
	}

	mWaylandData.registry = wl_display_get_registry(mWaylandData.display);
	wl_registry_add_listener(mWaylandData.registry, &registry_listener, &mWaylandData);
	wl_display_dispatch(mWaylandData.display);

	mWaylandData.window = wl_compositor_create_surface(mWaylandData.compositor);
	if (!mWaylandData.window) {
		LOGE("Can not create wayland surface from compositor!\n");
		return false;
	}

	if (mWaylandData.shell) {
		mWaylandData.shell_surface = wl_shell_get_shell_surface(mWaylandData.shell, mWaylandData.window);
		if (!mWaylandData.shell_surface) {
			LOGE("Can not get shell surface from wayland surface!\n");
			return false;
		}
		wl_shell_surface_add_listener(mWaylandData.shell_surface, &shell_surface_listener, NULL);
		wl_shell_surface_set_toplevel(mWaylandData.shell_surface);
		wl_shell_surface_set_title(mWaylandData.shell_surface, "vkfish");
	}

#ifdef ENABLE_IVI_SHELL
	else {
		if (mWaylandData.ivi_application ) {
			mWaylandData.ivi_surfaceId = DEFAULT_IVI_SURFACE_ID;
			if (!create_ivi_surface(&mWaylandData)) {
				printf("Can not get ivi shell surface from wayland surface!\n");
			}
		}
	}
#endif

	VkWaylandSurfaceCreateInfoKHR createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
	createInfo.pNext = NULL;
	createInfo.flags = 0;
	createInfo.display = mWaylandData.display;
	createInfo.surface = mWaylandData.window;

	VkResult err = vkCreateWaylandSurfaceKHR(instance(), &createInfo, NULL, surface);
	if (err != VK_SUCCESS) {
		LOGE("Failed to create wayland surface!\n");
		return false;
	}

	return true;
}
#endif // defined(VK_USE_PLATFORM_WAYLAND_KHR)
#if defined(VK_USE_PLATFORM_XLIB_KHR)
bool NvGLFWContextVK::createXlibSurface(uint32_t width, uint32_t height, VkSurfaceKHR *surface) {
	const char *display_envar = getenv("DISPLAY");
	if (display_envar == NULL || display_envar[0] == '\0') {
		printf("Environment variable DISPLAY requires a valid value.\nExiting ...\n");
		fflush(stdout);
		exit(1);
	}

	XInitThreads();
	Display *display = XOpenDisplay(NULL);
	long visualMask = VisualScreenMask;
	int numberOfVisuals;
	Window xlib_window;
	XVisualInfo vInfoTemplate = {};
	vInfoTemplate.screen = DefaultScreen(display);
	XVisualInfo *visualInfo = XGetVisualInfo(display, visualMask, &vInfoTemplate, &numberOfVisuals);

	Colormap colormap =
	XCreateColormap(display, RootWindow(display, vInfoTemplate.screen), visualInfo->visual, AllocNone);

	XSetWindowAttributes windowAttributes = {};
	windowAttributes.colormap = colormap;
	windowAttributes.background_pixel = 0xFFFFFFFF;
	windowAttributes.border_pixel = 0;

	xlib_window = XCreateWindow(display, RootWindow(display, vInfoTemplate.screen), 0, 0, width, height, 0, visualInfo->depth, InputOutput, visualInfo->visual, CWBackPixel | CWBorderPixel | CWEventMask | CWColormap, &windowAttributes);

	XMapWindow(display, xlib_window);
	XFlush(display);

	VkXlibSurfaceCreateInfoKHR createInfo;
	createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
	createInfo.pNext = NULL;
	createInfo.flags = 0;
    createInfo.dpy = display;
	createInfo.window = xlib_window;

	VkResult err = vkCreateXlibSurfaceKHR(instance(), &createInfo, NULL, surface);
	if (err != VK_SUCCESS) {
		LOGE("Failed to create Xlib surface!\n");
		return false;
	}

	return true;
}
#endif //defined(VK_USE_PLATFORM_XLIB_KHR)

bool NvGLFWContextVK::initRenderTarget() {
	mSwapchainRenderTarget = NvVkMultibufferedRenderTarget::create(*this, m_cbFormat, m_dsFormat, !mUseFBOPair, mFBOWidth, mFBOHeight);

	if (mSwapchainRenderTarget) {
		if (!mUseFBOPair) {
			//extern VkResult glfwCreateWindowSurface(VkInstance instance, GLFWwindow* window, const VkAllocationCallbacks* allocator, VkSurfaceKHR* surface);
			//glfwCreateWindowSurface(_instance, mWindow, NULL, &surface);
#if defined(VK_USE_PLATFORM_DISPLAY_KHR)
			VkSurfaceKHR surface;

			if (!createDirect2DisplaySurface(winWidth, winHeight, &surface)) {
				return false;
			}

			mSwapchainRenderTarget->setSurface(surface);

#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
			VkSurfaceKHR surface;

			if (!createWaylandSurface(&surface)) {
				return false;
			}

			mSwapchainRenderTarget->setSurface(surface);

#elif defined(VK_USE_PLATFORM_XLIB_KHR)
            VkSurfaceKHR surface;

            if (!createXlibSurface(winWidth, winHeight, &surface)) {
                return false;
            }

            mSwapchainRenderTarget->setSurface(surface);

#endif
		}
		return true;
	}
	else {
		return false;
	}
}

bool NvGLFWContextVK::readFramebufferRGBX32(uint8_t *dest, int32_t& w, int32_t& h) {
	NvVkRenderTarget& rt = *mainRenderTarget();

	w = rt.width();
	h = rt.height();

	if (!dest)
		return true;

	VkFormat format = rt.targetFormat();

	if (format == VK_FORMAT_R8G8B8A8_UNORM) {
		uint32_t size = 4 * w * h;

		VkBufferImageCopy region;
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageSubresource.mipLevel = 0;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { (uint32_t)w, (uint32_t)h, 1 };

		NvVkBuffer dstBuffer;
		createAndFillBuffer(size, VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, dstBuffer);

		VkCommandBuffer cmd = beginTempCmdBuffer();

		vkCmdCopyImageToBuffer(cmd,
			rt.image(),
			VK_IMAGE_LAYOUT_GENERAL,
			dstBuffer.buffer,
			1, &region);

		doneWithTempCmdBufferSubmit(cmd);

		vkDeviceWaitIdle(device());

		uint8_t* ptr = NULL;
		vkMapMemory(device(), dstBuffer.mem, 0, size, 0, (void**)&ptr);

		uint32_t rowSize = w * 4;
		ptr += rowSize * (h - 1);

		for (int32_t i = 0; i < h; i++) {
			memcpy(dest, ptr, rowSize);
			dest += rowSize;
			ptr -= rowSize;
		}

		return true;
	}

	return false;
}

#endif
