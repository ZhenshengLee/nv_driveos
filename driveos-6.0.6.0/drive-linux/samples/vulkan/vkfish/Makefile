
CFLAGS   = $(NV_PLATFORM_OPT)     $(NV_PLATFORM_CFLAGS)
CPPFLAGS = $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
LDFLAGS  = $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_LDFLAGS)

include ../../../make/nvdefs.mk

# By default we use the following options
#   - Use native functions for window and operating system interaction
#   - Use source shaders
#   - Build shaders into the application rather than using external data files
# Any of these can be overridden with environment variables or by
#   modifying this file. Note that demo executables must be build with the
#   same settings as the demo libraries they link against.
# If you choose external shader files, you will need to copy the files
#   (.bin for binary, .glsl[fv] for source) to the platform along with
#   the executable.
NV_USE_BINARY_SHADERS ?= 0
NV_USE_EXTERN_SHADERS ?= 0

ifeq ($(NV_USE_BINARY_SHADERS),1)
  CPPFLAGS += -DUSE_BINARY_SHADERS
endif
ifeq ($(NV_USE_EXTERN_SHADERS),1)
  CPPFLAGS += -DUSE_EXTERN_SHADERS
endif

CPPFLAGS += -DNVGLDEMO_IS_EMBEDDED_LINUX

TARGETS += $(NV_WINSYS)/vkfish

VKFISH_OBJS :=
VKFISH_OBJS += $(NV_WINSYS)/ThreadedRenderingVk.o
VKFISH_OBJS += $(NV_WINSYS)/School.o
VKFISH_OBJS += $(NV_WINSYS)/NvInstancedModelExtVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelExtVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvMeshExtVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvSkeleton.o
VKFISH_OBJS += $(NV_WINSYS)/NvVkContext.o
VKFISH_OBJS += $(NV_WINSYS)/NvImage.o
VKFISH_OBJS += $(NV_WINSYS)/NvImageDDS.o
VKFISH_OBJS += $(NV_WINSYS)/NvFilePtr.o
VKFISH_OBJS += $(NV_WINSYS)/MainLinux.o
VKFISH_OBJS += $(NV_WINSYS)/NvFoundationInit.o
VKFISH_OBJS += $(NV_WINSYS)/NsGlobals.o
VKFISH_OBJS += $(NV_WINSYS)/NsAllocator.o
VKFISH_OBJS += $(NV_WINSYS)/NsTempAllocator.o
VKFISH_OBJS += $(NV_WINSYS)/NsUnixMutex.o
VKFISH_OBJS += $(NV_WINSYS)/NsUnixAtomic.o
VKFISH_OBJS += $(NV_WINSYS)/NvAppBase.o
VKFISH_OBJS += $(NV_WINSYS)/NvAssetLoaderLinux.o
VKFISH_OBJS += $(NV_WINSYS)/NvThreadPosix.o
VKFISH_OBJS += $(NV_WINSYS)/BlockDXT.o
VKFISH_OBJS += $(NV_WINSYS)/ColorBlock.o
VKFISH_OBJS += $(NV_WINSYS)/NvMaterialVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvSharedVBOVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvSampleAppVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvSampleApp.o
VKFISH_OBJS += $(NV_WINSYS)/NsUnixThread.o
VKFISH_OBJS += $(NV_WINSYS)/NsUnixSync.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelExtObj.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelSubMeshObj.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelExt.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelExtBin.o
VKFISH_OBJS += $(NV_WINSYS)/NvModelMeshFace.o
VKFISH_OBJS += $(NV_WINSYS)/NvQuadVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvGLFWContextVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvVkRenderTargetImpls.o
VKFISH_OBJS += $(NV_WINSYS)/NvAppNativeContextVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvAppContextVK.o
VKFISH_OBJS += $(NV_WINSYS)/NvPlatformVK.o
INTERMEDIATES += $(VKFISH_OBJS)

VKFISH_LDLIBS :=
VKFISH_LDLIBS += -lpthread
VKFISH_LDLIBS += -lm
VKFISH_LDLIBS += -lc
VKFISH_LDLIBS += -lstdc++
VKFISH_LDLIBS += -lvulkan
VKFISH_LDLIBS += ${NV_PLATFORM_WINSYS_LIBS}

VKFISH_CXXFLAGS :=
VKFISH_CXXFLAGS += -std=c++11
VKFISH_CXXFLAGS += -DNDEBUG
CXXFLAGS += $(VKFISH_CXXFLAGS)

ifeq ($(findstring $(NV_WINSYS),direct-to-display wayland x11),)
all:
	@echo Sample not supported for NV_WINSYS=$(NV_WINSYS)
else
all: $(TARGETS)
endif

clean:
	rm -rf $(TARGETS) $(INTERMEDIATES)

.PHONY: FORCE
FORCE:

$(NV_WINSYS)/vkfish: $(VKFISH_OBJS) $(VKFISH_DEMOLIBS)
	$(LD) $(LDFLAGS) -o $@ $^ $(VKFISH_LDLIBS)

