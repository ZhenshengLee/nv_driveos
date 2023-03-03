# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

include ../../../../make/nvapedefs.mk

TARGETS = libnvfx.a

INCFILES := -I$(NV_PLATFORM_APE_COMMON_INC)
INCFILES += -I$(NV_PLATFORM_APE_NVFX_INC)
INCFILES += -I$(NV_PLATFORM_APE_LK_INC)

CFLAGS := $(NV_PLATFORM_APE_CFLAGS)

SOURCES := buffer.c
SOURCES += utils.c

OBJECTS := $(SOURCES:.c=.o)

all: $(SOURCES) $(TARGETS)

# generate static library
$(TARGETS): $(OBJECTS)
	$(AR) rcs $(NV_PLATFORM_APE_TARGET_LIBS)/$@ $^

.c.o:
	$(CC) $(CFLAGS) $(INCFILES) -c $< -o $@

# clean
clean clobber:
	rm -rf $(OBJECTS) $(TARGETS)
