NvSciStream Safety Sample App - README

Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.

---
# nvscistream_safety_sample_app - NvSciStream Safety Sample App

## Description

This directory contains an NvSciStream safety sample application that
supports buffer sharing between ASIL and QM process.

To use this sample for writing your own applications:

* See proc_asil.c, proc_qm_proxy.c and proc_qm.c for examples of how to
  do top level application setup and how to create the blocks needed for
  your use case and connect them all together.
* See the appropriate block_*.c files for examples of creating the
  necessary blocks and handling the events that they encounter.
  See the block_producer.c and block_consumer.c files for examples of how
  to map the relevant engines to and from NvSci.

## Build the application

The NvSciStream sample includes source code and a Makefile.
Navigate to the sample application directory to build the application:

       make clean
       make

## Examples of how to run the sample application:

* NOTE:
* This test case must be run with sudo.

Multi-process CUDA/CUDA stream with two consumers, one in the same
process as the producer and one in another process on another OS running
in a peer SoC which interacts with producer via proxy application.

    ./nvscistream_safety_sample_app -a & ./nvscistream_safety_sample_app -q nvscic2c_pcie_s0_c5_1
    # Run below command on another OS running on peer SOC.
    ./nvscistream_safety_sample_app -p nvscic2c_pcie_s0_c6_1