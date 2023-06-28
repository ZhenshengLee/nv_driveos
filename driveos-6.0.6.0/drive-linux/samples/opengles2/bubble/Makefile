
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

TARGETS += $(NV_WINSYS)/bubble

BUBBLE_OBJS :=
BUBBLE_OBJS += $(NV_WINSYS)/main.o
BUBBLE_OBJS += $(NV_WINSYS)/bubble.o
BUBBLE_OBJS += $(NV_WINSYS)/envcube.o
BUBBLE_OBJS += $(NV_WINSYS)/shape.o
BUBBLE_OBJS += $(NV_WINSYS)/shaders.o
BUBBLE_OBJS += $(NV_WINSYS)/algebra.o
INTERMEDIATES += $(BUBBLE_OBJS)

BUBBLE_SHADER_STRS :=
BUBBLE_SHADER_STRS += envcube_vert.glslvh
BUBBLE_SHADER_STRS += bubble_vert.glslvh
BUBBLE_SHADER_STRS += mesh_vert.glslvh
BUBBLE_SHADER_STRS += mouse_vert.glslvh
BUBBLE_SHADER_STRS += envcube_frag.glslfh
BUBBLE_SHADER_STRS += bubble_frag.glslfh
BUBBLE_SHADER_STRS += mesh_frag.glslfh
BUBBLE_SHADER_STRS += mouse_frag.glslfh
INTERMEDIATES += $(BUBBLE_SHADER_STRS)

BUBBLE_SHADER_BINS :=
BUBBLE_SHADER_BINS += gp10b/envcube_prog.bin
BUBBLE_SHADER_BINS += gv11b/envcube_prog.bin
BUBBLE_SHADER_BINS += gp10b/bubble_prog.bin
BUBBLE_SHADER_BINS += gv11b/bubble_prog.bin
BUBBLE_SHADER_BINS += gp10b/mesh_prog.bin
BUBBLE_SHADER_BINS += gv11b/mesh_prog.bin
BUBBLE_SHADER_BINS += gp10b/mouse_prog.bin
BUBBLE_SHADER_BINS += gv11b/mouse_prog.bin
INTERMEDIATES += $(BUBBLE_SHADER_BINS)
ifeq ($(NV_USE_EXTERN_SHADERS),1)
ifeq ($(NV_USE_BINARY_SHADERS),1)
TARGETS += $(BUBBLE_SHADER_BINS)
endif
endif

BUBBLE_DEMOLIBS :=
BUBBLE_DEMOLIBS += ../nvtexfont/$(NV_WINSYS)/libnvtexfont2.a
BUBBLE_DEMOLIBS += ../nvgldemo/$(NV_WINSYS)/libnvgldemo.a
INTERMEDIATES += $(BUBBLE_DEMOLIBS)

BUBBLE_LDLIBS :=
BUBBLE_LDLIBS += -lm
BUBBLE_LDLIBS += -lrt
BUBBLE_LDLIBS += -lpthread
BUBBLE_LDLIBS += -lEGL
BUBBLE_LDLIBS += -lGLESv2
BUBBLE_LDLIBS += ${NV_PLATFORM_WINSYS_LIBS}

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

$(NV_WINSYS)/bubble: $(BUBBLE_OBJS) $(BUBBLE_DEMOLIBS)
	$(LD) $(LDFLAGS) -o $@ $^ $(BUBBLE_LDLIBS)

ifeq ($(NV_USE_EXTERN_SHADERS),0)
ifeq ($(NV_USE_BINARY_SHADERS),0)
$(BUBBLE_OBJS) : $(BUBBLE_SHADER_STRS)
endif
endif

define demolib-rule
$(1): FORCE
	$(MAKE) -C $$(subst $$(NV_WINSYS)/,,$$(dir $$@))
endef
$(foreach demolib,$(BUBBLE_DEMOLIBS),$(eval $(call demolib-rule,$(demolib))))

define run-glslc
	mkdir -p $(4)
	$(NV_PLATFORM_DIR)/utils/openGL-tools/glslc -gles -binary $(4)/$(1) -vs  $(2) -fs  $(3) -chip $(4)
endef
gp10b/envcube_prog.bin:
	$(call run-glslc,envcube_prog.bin,envcube_vert.glslv,envcube_frag.glslf,gp10b)
gv11b/envcube_prog.bin:
	$(call run-glslc,envcube_prog.bin,envcube_vert.glslv,envcube_frag.glslf,gv11b)
gp10b/bubble_prog.bin:
	$(call run-glslc,bubble_prog.bin,bubble_vert.glslv,bubble_frag.glslf,gp10b)
gv11b/bubble_prog.bin:
	$(call run-glslc,bubble_prog.bin,bubble_vert.glslv,bubble_frag.glslf,gv11b)
gp10b/mesh_prog.bin:
	$(call run-glslc,mesh_prog.bin,mesh_vert.glslv,mesh_frag.glslf,gp10b)
gv11b/mesh_prog.bin:
	$(call run-glslc,mesh_prog.bin,mesh_vert.glslv,mesh_frag.glslf,gv11b)
gp10b/mouse_prog.bin:
	$(call run-glslc,mouse_prog.bin,mouse_vert.glslv,mouse_frag.glslf,gp10b)
gv11b/mouse_prog.bin:
	$(call run-glslc,mouse_prog.bin,mouse_vert.glslv,mouse_frag.glslf,gv11b)
