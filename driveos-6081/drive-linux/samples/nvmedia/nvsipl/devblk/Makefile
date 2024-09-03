# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../make/nvdefs.mk

# Build all drivers in the ./cameramodule directory
DRIVER_NAMES = $(notdir $(shell find cameramodule/MAX96712cameramodule/* -maxdepth 0 -type d))
DRIVER_NAMES_TI = $(notdir $(shell find cameramodule/DS90UB9724cameramodule/* -maxdepth 0 -type d))
DRIVER_LIBS = $(foreach DRIVER, $(DRIVER_NAMES), cameramodule/MAX96712cameramodule/$(DRIVER)/libnvsipl_devblk_drv_$(DRIVER).so)
DRIVER_LIBS += $(foreach DRIVER, $(DRIVER_NAMES_TI), cameramodule/DS90UB9724cameramodule/$(DRIVER)/libnvsipl_devblk_drv_$(DRIVER).so)
DEVICES_NAMES = $(notdir $(shell find devices/* -maxdepth 0 -type d))

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS)
CFLAGS   += -D_FILE_OFFSET_BITS=64 -fPIC
CFLAGS   += $(NV_PLATFORM_SDK_INC) -I. -I../ -I../../utils $(foreach DEVICE, $(DEVICES_NAMES), -Idevices/$(DEVICE))
CFLAGS   += -I./common/utils
CFLAGS   += -I./common/crypto/include
CFLAGS   += -I./cameramodule/common/utils
CFLAGS   += -I$(NV_PLATFORM_DIR)/include/nvmedia_6x

CXXFLAGS += $(NV_PLATFORM_SDK_INC) $(NV_PLATFORM_CPPFLAGS)
CXXFLAGS += -I$(NV_PLATFORM_DIR)/include/nvmedia_6x
CXXFLAGS += -I. -I./common -I./common/include -I./common/utils -I./common/core -I./common/ddi -I./common/trace  -I./cameramodule -I./cameramodule/common -I./cameramodule/common/utils
CXXFLAGS += -I../query/src -I./devices $(foreach DEVICE, $(DEVICES_NAMES), -Idevices/$(DEVICE))
CXXFLAGS += -std=c++14 -fexceptions -frtti -fPIC

LDFLAGS  := -L$(NV_PLATFORM_SDK_LIB_DIR) -Wl,--no-undefined

LDLIBS  += -lnvos
LDLIBS  += -lnvsipl_devblk_cdi
LDLIBS  += -lnvsipl_devblk_ddi
LDLIBS  += -lnvsipl_devblk_crypto

ifeq ($(NV_PLATFORM_OS), Linux)
	LDLIBS  += -lstdc++ -ldl -lpthread
endif

ifeq ($(NV_PLATFORM_OS), QNX)
	LDLIBS   += -lc++
	CFLAGS   += -DNVMEDIA_QNX
	CXXFLAGS += -DNVMEDIA_QNX
	CXXFLAGS += -DUSE_CDAC=2
endif

CAMPWR_PATH         := devices/MAX20087Driver
CAMPWR_LIB_BASENAME := nvsipl_devblk_campwr_max20087
CAMPWR_LIB          := $(CAMPWR_PATH)/lib$(CAMPWR_LIB_BASENAME).so

CAMPWR_SRCS := $(shell find $(CAMPWR_PATH) -name '*.c' -or -name '*.cpp')
CAMPWR_SRCS += $(shell find cameramodule/common/CampwrIF -name '*.cpp')
CAMPWR_OBJS := $(patsubst %, %.o, $(foreach CAMPWR_SRC, $(CAMPWR_SRCS), $(basename $(CAMPWR_SRC))))

CAMPWR_LDLIBS  := -l$(CAMPWR_LIB_BASENAME)
CAMPWR_LDFLAGS := -L$(CAMPWR_PATH)

.PHONY: default
default: $(DRIVER_LIBS) $(CAMPWR_LIB)

# Cameramodule common paths
CAMERAMODULE_COMMON := find cameramodule/common/TransportLinkIF cameramodule/common/EEPROMIF \
	cameramodule/common/SerializerIF cameramodule/common/SensorIF cameramodule/common/PMICIF \
	cameramodule/common/CampwrIF

# Common objects for each driver
DRIVER_COMMON_OBJS := \
	$(patsubst %.cpp, %.o, $(wildcard cameramodule/MAX96712cameramodule/*.cpp)) \
	$(patsubst %.cpp, %.o, $(shell $(CAMERAMODULE_COMMON) -name '*.cpp')) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX9295ASerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX9295ASerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96705SerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96705SerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96712DeserializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96712DeserializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96717FSerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96717FSerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/MAX96759SerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/MAX96759SerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/N24C64EEPROMDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/N24C64EEPROMDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/TPGSerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/TPGSerializerDriver/*.c)) \
	../../utils/log_utils.o \
	../../utils/thread_utils.o


DRIVER_COMMON_OBJS_TI := \
	$(patsubst %.cpp, %.o, $(wildcard cameramodule/DS90UB9724cameramodule/*.cpp)) \
	$(patsubst %.cpp, %.o, $(shell $(CAMERAMODULE_COMMON) -name '*.cpp')) \
	$(patsubst %.cpp, %.o, $(wildcard devices/DS90UB971SerializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/DS90UB971SerializerDriver/*.c)) \
	$(patsubst %.cpp, %.o, $(wildcard devices/DS90UB9724DeserializerDriver/*.cpp)) \
	$(patsubst %.c, %.o, $(wildcard devices/DS90UB9724DeserializerDriver/*.c)) \
	../../utils/log_utils.o \
	../../utils/thread_utils.o

# The camera power implementation ($(CAMPWR_LIB)) is built as its own shared
# library. Consequently, filter it out from the main driver build(s).
DRIVER_COMMON_OBJS := $(filter-out $(CAMPWR_OBJS), $(DRIVER_COMMON_OBJS))
DRIVER_COMMON_OBJS_TI := $(filter-out $(CAMPWR_OBJS), $(DRIVER_COMMON_OBJS_TI))

DRIVER_COMMON_OBJS += \
	cameramodule/common/utils/pwr_utils_stub.o
DRIVER_COMMON_OBJS_TI += \
	cameramodule/common/utils/pwr_utils_stub.o

#Instantiate a rule to build each driver
define DRIVER_RULE
OBJS_$(1) = $(patsubst %.cpp, %.o, $(wildcard cameramodule/MAX96712cameramodule/$(1)/*.cpp)) $(patsubst %.c, %.o, $(wildcard cameramodule/MAX96712cameramodule/$(1)/*.c))
OBJS_$(1) += $(DRIVER_COMMON_OBJS)
OBJS_TI_$(1) = $(patsubst %.cpp, %.o, $(wildcard cameramodule/DS90UB9724cameramodule/$(1)/*.cpp)) $(patsubst %.c, %.o, $(wildcard cameramodule/DS90UB9724cameramodule/$(1)/*.c))
OBJS_TI_$(1) += $(DRIVER_COMMON_OBJS_TI)

cameramodule/MAX96712cameramodule/$(1)/libnvsipl_devblk_drv_$(1).so: $$(OBJS_$(1)) $(CAMPWR_LIB)
	$(CXX) -shared $(LDFLAGS) -o $$@ $$(OBJS_$(1)) $(LDLIBS) $(LDFLAGS) $(CAMPWR_LDFLAGS) $(CAMPWR_LDLIBS)

cameramodule/DS90UB9724cameramodule/$(1)/libnvsipl_devblk_drv_$(1).so: $$(OBJS_TI_$(1)) $(CAMPWR_LIB)
	$(CXX) -shared $(LDFLAGS) -o $$@ $$(OBJS_TI_$(1)) $(LDLIBS) $(LDFLAGS) $(CAMPWR_LDFLAGS) $(CAMPWR_LDLIBS)

endef

$(foreach DRIVER, $(DRIVER_NAMES), $(eval $(call DRIVER_RULE,$(DRIVER))))
$(foreach DRIVER, $(DRIVER_NAMES_TI), $(eval $(call DRIVER_RULE,$(DRIVER))))


$(CAMPWR_LIB): $(CAMPWR_OBJS)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LDLIBS) $(LDFLAGS)

clean clobber:
	find . -name '*.so' -exec echo rm {} + -exec rm {} +
	find . -name '*.o' -exec echo rm {} + -exec rm {} +
