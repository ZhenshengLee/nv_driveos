
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

TARGETS += $(NV_WINSYS)/libnvgldemo.a

NVGLDEMO_OBJS :=
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_main.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_parse.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_shader.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_math.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_texture.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_socket.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_os_posix.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_cqueue.o
NVGLDEMO_OBJS += $(NV_WINSYS)/nvgldemo_preswap.o
ifeq ($(NV_WINSYS),egldevice)
 NVGLDEMO_OBJS += egldevice/nvgldemo_win_egldevice.o
 NV_PLATFORM_CPPFLAGS += -DNVGLDEMO_HAS_DEVICE
endif
ifeq ($(NV_WINSYS),screen)
 NVGLDEMO_OBJS += screen/nvgldemo_win_screen.o
endif
ifeq ($(NV_WINSYS),wayland)
 NVGLDEMO_OBJS += wayland/nvgldemo_win_wayland.o
 ifneq ($(NV_PLATFORM_OS), QNX)
  NV_PLATFORM_CPPFLAGS += -DHAVE_LIBINPUT -DHAVE_XKBCOMMON
 endif
endif
ifeq ($(NV_WINSYS),x11)
 NVGLDEMO_OBJS += x11/nvgldemo_win_x11.o
endif
INTERMEDIATES += $(NVGLDEMO_OBJS)

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

$(NV_WINSYS)/libnvgldemo.a: $(NV_WINSYS)/libnvgldemo.a($(NVGLDEMO_OBJS))

