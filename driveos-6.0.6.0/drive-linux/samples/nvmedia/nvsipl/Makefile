# Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

SUBDIRS := query
SUBDIRS += devblk
SUBDIRS += test/camera
SUBDIRS += test/sample
SUBDIRS += test/plugin

.PHONY: all clean $(SUBDIRS)

all clean: $(SUBDIRS)

$(SUBDIRS):
	test ! -r $@/Makefile || make -C $@ ${MAKECMDGOALS}
