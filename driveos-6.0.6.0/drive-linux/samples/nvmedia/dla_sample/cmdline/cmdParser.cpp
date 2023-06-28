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

#include "cmdParser.h"

void CmdParser::ShowUsage(void)
{
    LOG_MSG("\nExecute runtime test -\n");
    LOG_MSG("-h Usage of the test app\n");
    LOG_MSG("-v Enable verbose\n");
    LOG_MSG("--mode runtime/scisync/ping/multithread.\n");
    LOG_MSG("-d Specify the dla instance. Avaiable values are 0, 1\n");
    LOG_MSG("-n Specify the number of simultaneous tasks.\n");
    LOG_MSG("   Available values are in range of [1,31].\n");
    LOG_MSG("--loadable Specify path to the loadable file\n");
    LOG_MSG("Sample command line to run a ping test:\n");
    LOG_MSG("nvm_dlaSample --mode ping -d dlaId\n");
    LOG_MSG("Sample command line to run a runtime test:\n");
    LOG_MSG("nvm_dlaSample --mode runtime -d dlaId -n numTasks --loadable <PathToLoadableFile>\n");
    LOG_MSG("Sample command line to run a scisync test:\n");
    LOG_MSG("nvm_dlaSample --mode scisync -d dlaId -n numTasks --loadable <PathToLoadableFile>\n");
    LOG_MSG("Sample command line to run a multithread test:\n");
    LOG_MSG("nvm_dlaSample --mode multithread --loadable <PathToLoadableFile>\n");
    return;
}

int CmdParser::Parse(int argc, char* argv[])
{
    const char* const short_options = "hm:d:n:p:v";
    const struct option long_options[] =
    {
        { "help",                 no_argument,       0, 'h' },
        { "mode",                 required_argument, 0, 'm' },
        { "instance",             required_argument, 0, 'd' },
        { "numTasks",             required_argument, 0, 'n' },
        { "loadable",             required_argument, 0, 'l' },
        { "verbose",              no_argument,       0, 'v' },
        { 0,                      0,                 0,  0  }
    };

    int index = 0;
    auto bShowHelp = false;
    auto bInvalid = false;

    SET_LOG_LEVEL(CLogger::LogLevel::LEVEL_INFORMATION);

    while (1) {
        const auto getopt_ret = getopt_long(argc, argv, short_options , &long_options[0], &index);
        if (getopt_ret == -1) {
            // Done parsing all arguments.
            break;
        }

        switch (getopt_ret) {
        default: /* Unrecognized option */
        case '?': /* Unrecognized option */
            LOG_ERR("Invalid or Unrecognized command line option\n");
            bInvalid = true;
            break;
        case 'h': /* -h or --help */
            bShowHelp = true;
            break;
        case 'v':
            verboseLevel = CLogger::LogLevel::LEVEL_DEBUG;
            SET_LOG_LEVEL((CLogger::LogLevel)verboseLevel);
            break;
        case 'm':
            if (std::string(optarg) == "runtime") {
                testRuntime = 1;
            } else if (std::string(optarg) == "scisync") {
                testSciSync = 1;
            } else if (std::string(optarg) == "ping") {
                testPing = 1;
            } else if (std::string(optarg) == "multithread") {
                testMultiThread = 1;
            } else {
                LOG_ERR("Unknown mode\n");
                goto fail;
            }
            break;
        case 'd':
            dlaId = atoi(optarg);
            if (!((dlaId == 0) || (dlaId == 1))) {
                LOG_ERR("Invalid DLA ID\n");
                goto fail;
            }
            break;
        case 'n':
            numTasks = atoi(optarg);
            break;
        case 'l':
            loadableName = std::string(optarg);
            break;
        }
    }

    // Help Option
    if (bShowHelp) {
        return 1;
    }

    // Bad arguments
    if (bInvalid) {
        goto fail;
    }

    // Verify command line argument
    if ((testPing + testRuntime + testSciSync + testMultiThread) > 1) {
        LOG_ERR("Only one of Runtime / SciSync / Ping / MultiThread test should be enabled \n");
        goto fail;
    }

    if ((testRuntime + testSciSync) == 1) {
        if (!loadableName.size()) {
            LOG_ERR("loadableName need to be set\n");
            goto fail;
        }

        if (!numTasks) {
            LOG_ERR("numTasks need to be set and non-zero\n");
            goto fail;
        }
    }

    if ((testPing + testRuntime + testSciSync) == 1) {
        if (dlaId == 100) {
            LOG_WARN("dlaId not set \n\tdefault dlaId is set to 0\n\tuse -d [] option to set dlaId\n");
            dlaId=0;
        }
    }

    if (testMultiThread == 1) {
        if (!loadableName.size()) {
            LOG_ERR("loadableName need to be set\n");
            goto fail;
        }
    }

    return 0;
fail:
    return -1;
}

void CmdParser::PrintArgs() const
{
    LOG_INFO("Test runtime = %d \n", testRuntime);
    LOG_INFO("Test scisync = %d \n", testSciSync);
    LOG_INFO("Test ping = %d \n", testPing);
    LOG_INFO("Test multithread = %d \n", testMultiThread);
    LOG_INFO("loadable name = %s \n", loadableName.c_str());
    LOG_INFO("dlaId = %d \n", dlaId);
    LOG_INFO("numTasks = %d \n", numTasks);
    LOG_INFO("verbose level = %d \n", verboseLevel);
}
