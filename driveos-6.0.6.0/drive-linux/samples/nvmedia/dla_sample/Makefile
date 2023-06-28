# SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvm_dlaSample

CFLAGS   := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS)
CPPFLAGS := $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC) -DWIN_INTERFACE_CUSTOM -D_POSIX_C_SOURCE=200112L -I. -I$(NV_PLATFORM_DIR)/include/nvmedia_6x -I./dla -I./logging -I./tensor -I./scisync -I./utils -I./cmdline
CPPFLAGS += -std=c++11 -fexceptions -frtti -fPIC

LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

OBJS   := cmdline/cmdParser.o
OBJS   += dla/dla.o
OBJS   += logging/cLogger.o
OBJS   += tensor/tensor.o
OBJS   += scisync/sciSync.o
OBJS   += utils/utils.o
OBJS   += main.o
OBJS   += testRuntime.o
OBJS   += testSciSync.o
OBJS   += testMT.o

LDLIBS  += -lnvscisync
LDLIBS  += -lnvscibuf
LDLIBS  += -lnvmedia_dla
LDLIBS  += -lnvmedia_tensor

ifeq ($(NV_PLATFORM_OS), Linux)
  LDLIBS += -lpthread
  LDLIBS += -ldl
  LDLIBS += -lrt
  LDLIBS += -lstdc++
else
  LDLIBS += -lsocket
  LDLIBS += -lc++
endif

.PHONY: default
default: $(TARGETS)

$(TARGETS): $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean clobber:
	rm -rf $(OBJS) $(TARGETS)
