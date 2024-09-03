# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../make/nvdefs.mk

# Build all drivers in ./databse directory
DRIVER_NAMES = $(filter-out schema,$(filter-out example,$(notdir $(shell find database/* -maxdepth 0 -type d))))
DRIVER_LIBS = $(foreach DRIVER, $(DRIVER_NAMES), database/$(DRIVER)/libnvsipl_qry_$(DRIVER).so)

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I.
CFLAGS   += -D_FILE_OFFSET_BITS=64 -fPIC
CXXFLAGS := $(NV_PLATFORM_SDK_INC)

CXXFLAGS += $(NV_PLATFORM_CPPFLAGS)
CXXFLAGS += -I./include -I../include -I./src -I./database
CXXFLAGS += -std=c++14 -fexceptions -frtti -fPIC

LDFLAGS  := -L$(NV_PLATFORM_SDK_LIB_DIR) -Wl,--no-undefined

GENERATE_JSON_HEADER := $(wildcard database/gen_json_header.py)

LDLIBS  += -ljsoncpp
# Add platform specific libs
ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS  += -lstdc++ -ldl
endif

ifeq ($(NV_PLATFORM_OS), QNX)
  LDLIBS    += -lc++
  LDLIBS    += -lnvdtcommon
  CFLAGS    += -DNVMEDIA_QNX
  CXXFLAGS  += -DNVMEDIA_QNX
endif

.PHONY: default
default: $(DRIVER_LIBS)

# Intantiate a rule to build each driver
define DRIVER_RULE
OBJS_$(1) = database/$(1)/json_data.o
JSON_$(1) = $(wildcard database/$(1)/*.json)
database/$(1)/libnvsipl_qry_$(1).so: $$(OBJS_$(1))
	echo "OBJECTS: $$(OBJS_$(1))"
	echo "JSON: $$(JSON_$(1))"
	$(CXX) -shared $(LDFLAGS) -o $$@ $$^ $(LDLIBS) $(LDFLAGS)

database/$(1)/json_data.cpp: $$(JSON_$(1))
	python $(GENERATE_JSON_HEADER) -inputFiles $$(JSON_$(1)) -outputFile $$@

endef
$(foreach DRIVER, $(DRIVER_NAMES), $(eval $(call DRIVER_RULE,$(DRIVER))))

clean clobber:
	find . -name '*.so' -exec echo rm {} + -exec rm {} +
	find . -name '*.o' -exec echo rm {} + -exec rm {} +
	find . -name 'json_data.cpp' -exec echo rm {} + -exec rm {} +
