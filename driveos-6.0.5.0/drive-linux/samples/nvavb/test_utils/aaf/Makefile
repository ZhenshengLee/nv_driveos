# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


include ../../../../make/nvdefs.mk

INCFILES += -I$(NV_PLATFORM_DIR)/include
INCFILES += -I../socket

LDFLAGS +=  $(NV_PLATFORM_SDK_LIB) -lpthread -ldl -lnvavtp

LDFLAGS += -Wl,--unresolved-symbols=ignore-in-shared-libs
LDFLAGS += $(NV_PLATFORM_LDFLAGS)

CFLAGS = $(NV_PLATFORM_CFLAGS)

SOURCES += nvavb_aaf_listener.c
SOURCES += ../socket/raw_socket.c

OBJECTS = $(SOURCES:.c=.o)

EXECUTABLE = aaf_listener

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS)  -o $@

.c.o:
	$(CC)  $(CFLAGS) $(INCFILES) -c $< -o $@

clean:
	rm -f nvavb_aaf_listener.o ../socket/raw_socket.o aaf_listener
