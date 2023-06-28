/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _CMDPARSER_H_
#define _CMDPARSER_H_

/* STL Headers */
#include <unistd.h>
#include <cstdbool>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <vector>
#include <iomanip>

#include "utils.h"

//! Command line parser class
//! Parse command line options

class CmdParser
{
 public:
    uint8_t testRuntime = 0;
    uint8_t testSciSync = 0;
    uint8_t testPing = 0;
    uint8_t testMultiThread = 0;
    std::string loadableName = std::string();
    uint32_t dlaId = 100;
    uint32_t numTasks = 0;

    uint32_t verboseLevel;

    static void ShowUsage(void);

    int Parse(int argc, char* argv[]);

    void PrintArgs() const;
};

#endif // _CMDPARSER_H_
