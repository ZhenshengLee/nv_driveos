NvSciStream Event Loop Driven Sample App - README

Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.

NVIDIA Corporation and its licensors retain all intellectual property and
proprietary rights in and to this software, related documentation and any
modifications thereto. Any use, reproduction, disclosure or distribution
of this software and related documentation without an express license
agreement from NVIDIA Corporation is strictly prohibited.

---
# nvscistream_event_sample - NvSciStream Sample App

## Description

This directory contains an NvSciStream sample application that
supports a variety of use cases, using an event-loop driven model.
Once the stream is fully connected, all further setup and streaming
operations are triggered by events, processed either by a single
NvSciEvent-driven thread or separate threads which wait for events
on each block. The former is the preferred approach for implementing
NvSciStream applications. In addition to those events which NvSci
itself generates, any other event which can be bound to an NvSciEvent
can be added to the event loop. This allows for robust applications
which can handle events regardless of the order in which they occur.

To use this sample for writing your own applications:

* See main.c for examples of how to do top level application setup and
  how to select the blocks needed for your use case and connect them
  all together.
* See the descriptions in the usecase*.h files to determine which use cases
  involve the producer and consumer engines that you are interested in.
* See the appropriate block_*.c files for examples of creating the
  necessary blocks and handling the events that they encounter.
  See the block_producer_*.c and block_consumer_*.c files for examples of how
  to map the relevant engines to and from NvSci.
* See the appropriate event_loop_*.c file for your chosen event handling
  method.

## Build the application

The NvSciStream sample includes source code and a Makefile.
Navigate to the sample application directory to build the application:

       make clean
       make

## Examples of how to run the sample application:

* NOTE:
* Inter-process and inter-chip test cases must be run with sudo.
* NvMedia/CUDA stream (use case 2) of the sample application is not supported
  on x86 and Jetson Linux devices.
* Inter-chip use cases are not supported on Jetson Linux devices.

Single-process, single-consumer CUDA/CUDA stream that uses the default event
service:

    ./nvscistream_event_sample

Single-process, single-consumer stream that uses the threaded event handling:

    ./nvscistream_event_sample -e t

Single-process NvMedia/CUDA stream with three consumers, and the second uses
the mailbox mode:

    ./nvscistream_event_sample -u 2 -m 3 -q 1 m

Multi-process CUDA/CUDA stream with three consumers, one in the same
process as the producer, and the other two in separate processes. The
first and the third consumers use the mailbox mode:

    ./nvscistream_event_sample -m 3 -p -c 0 -q 0 m &
    ./nvscistream_event_sample -c 1 -c 2 -q 2 m

Multi-process CUDA/CUDA stream with three consumers, one in the same
process as the producer, and the other two in separate processes.
To simulate the case with a less trusted consumer, one of the consumer
processes is set with lower priority. A limiter block is used to restrict
this consumer to hold at most one packet. The total number of packets is
increased to five.

Linux example:

    ./nvscistream_event_sample -m 3 -f 5 -p -c 0 -l 2 1 &
    ./nvscistream_event_sample -c 1 &
    nice -n 19 ./nvscistream_event_sample -c 2 &
    # Makes the third process as nice as possible.

QNX example:

    ./nvscistream_event_sample -m 3 -f 5 -p -c 0 -l 2 1 &
    ./nvscistream_event_sample -c 1 &
    nice -n 1 ./nvscistream_event_sample -c 2 &
    # Reduces the priority level of the third process by 1.

Multi-process CUDA/CUDA stream with two consumers, one in the same
process as the producer, and the other in a separate processe. Both
processes enable the endpoint information option:

    ./nvscistream_event_sample -m 2 -p -c 0 -i &
    ./nvscistream_event_sample -c 1 -i

Multi-process CUDA/CUDA stream with one consumer on another SoC.
The consumer has the FIFO queue attached to the C2C IpcSrc block, and
a three-packet pool attached to the C2C IpcDst block. It uses IPC channel
nvscic2c_pcie_s0_c5_1 <-> nvscic2c_pcie_s0_c6_1 for C2C communication.

    ./nvscistream_event_sample -P 0 nvscic2c_pcie_s0_c5_1 -Q 0 f
    # Run below command on another OS running on peer SOC.
    ./nvscistream_event_sample -C 0 nvscic2c_pcie_s0_c6_1 -F 0 3

Multi-process CUDA/CUDA stream with four consumers, one in the same
process as the producer, one in another process but in the same OS as the
producer, and two in another process on another OS running in a peer SoC.
The third and fourth consumers have a mailbox queue attached to the C2C
IpcSrc block, and a five-packet pool attached to the C2C IpcDst block.
The third consumer uses nvscic2c_pcie_s0_c5_1 <-> nvscic2c_pcie_s0_c6_1 for
C2C communication. The 4th consumer uses nvscic2c_pcie_s0_c5_2 <->
nvscic2c_pcie_s0_c6_2 for C2C communication.

    ./nvscistream_event_sample -m 4 -c 0 -q 0 m -Q 2 m -Q 3 m -P 2 nvscic2c_pcie_s0_c5_1 -P 3 nvscic2c_pcie_s0_c5_2 &
    ./nvscistream_event_sample -c 1 -q 1 m
    # Run below command on another OS running on peer SOC.
    ./nvscistream_event_sample -C 2 nvscic2c_pcie_s0_c6_1 -q 2 f -F 2 5 -C 3 nvscic2c_pcie_s0_c6_2 -q 3 m -F 3 5
