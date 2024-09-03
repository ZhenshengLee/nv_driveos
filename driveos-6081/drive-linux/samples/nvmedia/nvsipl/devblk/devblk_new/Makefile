# copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../../make/nvdefs.mk

# Build all drivers in the ./cameramodule directory
DRIVER_NAMES_FUSA = $(notdir $(shell find cameramodule/MAX96712cameramodule/* -maxdepth 0 -type d))
DRIVER_LIBS = $(foreach DRIVER, $(DRIVER_NAMES_FUSA), cameramodule/MAX96712cameramodule/$(DRIVER)/libnvsipl_devblk_drv_$(DRIVER).so)
DEVICES_NAMES_FUSA = $(notdir $(shell find devices/* -maxdepth 0 -type d))

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS)
CFLAGS   += -D_FILE_OFFSET_BITS=64 -fPIC
CFLAGS   += $(NV_PLATFORM_SDK_INC) -I. -I../ -I../../../utils
CFLAGS   += -I../common/utils
CFLAGS   += -I$(NV_PLATFORM_DIR)/include/nvmedia_6x
CFLAGS	 += -I./logging -I./include

CXXFLAGS += $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
CXXFLAGS += -I$(NV_PLATFORM_DIR)/include/nvmedia_6x
CXXFLAGS += -I. -I../common -I../common/include -I../common/utils -I../common/core -I../common/ddi -I../common/trace -I./cameramodule -I../cameramodule/common -I../cameramodule/common/TransportLinkIF -I../cameramodule/common/utils -I./cameramodule/MAX96712cameramodule/
CXXFLAGS += -I./logging -I./include
CXXFLAGS += -I../query/src -I./devices
CXXFLAGS += -I../../../utils/
CXXFLAGS += -std=c++14 -fexceptions -frtti -fPIC
CXXFLAGS += -DFUSA_CDD_NV

LDFLAGS  := -L$(NV_PLATFORM_SDK_LIB_DIR) -Wl,--no-undefined

LDLIBS  += -lnvos
LDLIBS  += -lnvsipl_devblk_cdi
LDLIBS  += -lnvsipl_devblk_ddi

ifeq ($(NV_PLATFORM_OS), Linux)
	LDLIBS  += -lstdc++ -ldl -lpthread
endif

ifeq ($(NV_PLATFORM_OS), QNX)
	LDLIBS   += -lc++
	LDLIBS   += -lnvdtcommon
	CFLAGS   += -DNVMEDIA_QNX
	CXXFLAGS += -DNVMEDIA_QNX
endif

CAMPWR_PATH         := devices/MAX20087Driver
CAMPWR_LIB_BASENAME := nvsipl_devblk_campwr_max20087_fusa
CAMPWR_LIB          := $(CAMPWR_PATH)/lib$(CAMPWR_LIB_BASENAME).so

CAMPWR_SRCS := $(shell find $(CAMPWR_PATH) -name '*.c' -or -name '*.cpp')
CAMPWR_SRCS += $(shell find ../cameramodule/common/CampwrIF -name '*.cpp')
CAMPWR_SRCS += $(shell find logging/ -name '*.c')
CAMPWR_OBJS := $(patsubst %, %.o, $(foreach CAMPWR_SRC, $(CAMPWR_SRCS), $(basename $(CAMPWR_SRC))))

CAMPWR_LDLIBS  := -l$(CAMPWR_LIB_BASENAME)
CAMPWR_LDFLAGS := -L$(CAMPWR_PATH)

.PHONY: default
default: $(DRIVER_LIBS) $(CAMPWR_LIB)

# Cameramodule common paths
CAMERAMODULE_COMMON := find ../cameramodule/common/TransportLinkIF ../cameramodule/common/EEPROMIF \
	../cameramodule/common/SerializerIF ../cameramodule/common/SensorIF ../cameramodule/common/PMICIF \
	../cameramodule/common/CampwrIF ../cameramodule/common/VCSELIF

# Common objects to build each driver
DRIVER_COMMON_OBJS := \
	$(patsubst %.cpp, %.o, $(shell $(CAMERAMODULE_COMMON) -name '*.cpp')) \
	../../../utils/thread_utils.o

# The camera power implementation ($(CAMPWR_LIB)) is built as its own shared
# library. Consequently, filter it out from the main driver build(s).
DRIVER_COMMON_OBJS := $(filter-out $(CAMPWR_OBJS), $(DRIVER_COMMON_OBJS))

DRIVER_COMMON_OBJS += \
	../cameramodule/common/utils/pwr_utils_stub.o

# Common objects to build each fusa driver
DRIVER_COMMON_FUSA_OBJS := \
	$(patsubst %.cpp, %.o, $(wildcard cameramodule/MAX96712cameramodule/*.cpp)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96712DeserializerDriver_nv/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96712DeserializerDriver_nv/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96717FSerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96717FSerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/M24C32EEPROMDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/M24C32EEPROMDriver/*.c)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX25614VCSELDriver/*.c)) \
	$(patsubst %.c, %.o, $(wildcard logging/*.c)) \
	$(patsubst %.c, %.o, $(wildcard ../../../utils/log_utils.c))


# Instantiate a rule to build each fusa driver

define DRIVER_RULE_FUSA
CFLAGS   += $(foreach DEVICE, $(DEVICES_NAMES_FUSA), -Idevices/$(DEVICE))
CXXFLAGS += $(foreach DEVICE, $(DEVICES_NAMES_FUSA), -Idevices/$(DEVICE))

OBJS_$(1) = $(patsubst %.cpp, %.o, $(wildcard cameramodule/MAX96712cameramodule/$(1)/*.cpp)) $(patsubst %.c, %.o, $(wildcard cameramodule/MAX96712cameramodule/$(1)/*.c))
OBJS_$(1) += $(DRIVER_COMMON_OBJS)
OBJS_$(1) += $(DRIVER_COMMON_FUSA_OBJS)

cameramodule/MAX96712cameramodule/$(1)/libnvsipl_devblk_drv_$(1).so: $$(OBJS_$(1)) $(CAMPWR_LIB)
	$(CXX) -shared $(LDFLAGS) -o $$@ $$(OBJS_$(1)) $(LDLIBS) $(LDFLAGS) $(CAMPWR_LDFLAGS) $(CAMPWR_LDLIBS)

endef

$(foreach DRIVER, $(DRIVER_NAMES_FUSA), $(eval $(call DRIVER_RULE_FUSA,$(DRIVER))))

$(CAMPWR_LIB): $(CAMPWR_OBJS)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS) $(LDFLAGS)

clean clobber:
	find ../../../utils/ -name 'log_utils.o' -exec echo rm {} + -exec rm {} +
	find . -name '*.so' -exec echo rm {} + -exec rm {} +
	find . -name '*.o' -exec echo rm {} + -exec rm {} +

