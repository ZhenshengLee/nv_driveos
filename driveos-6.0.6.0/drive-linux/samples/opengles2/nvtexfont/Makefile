
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

TARGETS += $(NV_WINSYS)/libnvtexfont2.a

NVTEXFONT2_OBJS :=
NVTEXFONT2_OBJS += $(NV_WINSYS)/nvtexfont2.o
NVTEXFONT2_OBJS += $(NV_WINSYS)/nvtexfont2-rast.o
NVTEXFONT2_OBJS += $(NV_WINSYS)/nvtexfont2-vect.o
INTERMEDIATES += $(NVTEXFONT2_OBJS)

NVTEXFONT2_SHADER_STRS :=
NVTEXFONT2_SHADER_STRS += vtxraster.glslvh
NVTEXFONT2_SHADER_STRS += vtxvector.glslvh
NVTEXFONT2_SHADER_STRS += colraster.glslfh
NVTEXFONT2_SHADER_STRS += colvector.glslfh
INTERMEDIATES += $(NVTEXFONT2_SHADER_STRS)

NVTEXFONT2_SHADER_BINS :=
INTERMEDIATES += $(NVTEXFONT2_SHADER_BINS)
ifeq ($(NV_USE_EXTERN_SHADERS),1)
ifeq ($(NV_USE_BINARY_SHADERS),1)
TARGETS += $(NVTEXFONT2_SHADER_BINS)
endif
endif

ifeq ($(findstring $(NV_WINSYS),egldevice screen wayland x11 direct-to-display),)
all:
	@echo Sample not supported for NV_WINSYS=$(NV_WINSYS)
else
all: $(TARGETS)
endif

clean:
	rm -rf $(TARGETS) $(INTERMEDIATES)

.PHONY: FORCE
FORCE:

$(NV_WINSYS)/libnvtexfont2.a: $(NV_WINSYS)/libnvtexfont2.a($(NVTEXFONT2_OBJS))

ifeq ($(NV_USE_EXTERN_SHADERS),0)
ifeq ($(NV_USE_BINARY_SHADERS),0)
$(NVTEXFONT2_OBJS) : $(NVTEXFONT2_SHADER_STRS)
endif
endif

