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

#include "testRuntime.h"
#include "testSciSync.h"
#include "testMT.h"
#include "cmdParser.h"
#include "utils.h"
#include <chrono>
#include <thread>

int main(int argc, char *argv[])
{
    CmdParser cmdline {};
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    int ret = 0;
    ret = cmdline.Parse(argc, argv);
    if (ret != 0) {
        CmdParser::ShowUsage();
        ret = (ret == 1) ? 0 : ret; // If help option, return 0 else return 255.
        return ret;
    }

    LOG_INFO(">>>>>>>>>>>>>>>>>>>> Test Begin >>>>>>>>>>>>>>>>>>>> \n");
    LOG_INFO("List of input arguments: \n");
    cmdline.PrintArgs();

    if (cmdline.testRuntime || cmdline.testPing) {
        LOG_INFO("Test with dlaId = %d numTasks = %d profile name %s\n",\
                  cmdline.dlaId, cmdline.numTasks, cmdline.loadableName.c_str());

        // Runtime test
        TestRuntime test(cmdline.dlaId, cmdline.numTasks, cmdline.loadableName, cmdline.testPing);
        status = test.SetUp();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Setup failure \n");
            ret = -1;
            goto fail;
        }
        status = test.RunTest();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Runtime failure \n");
            ret = -1;
            goto fail;
        }
    } else if (cmdline.testSciSync) {
        LOG_INFO("Test with dlaId = %d numTasks = %d profile name %s\n",\
                  cmdline.dlaId, cmdline.numTasks, cmdline.loadableName.c_str());

        // SciSync Test
        TestSciSync test(cmdline.dlaId, cmdline.numTasks, cmdline.loadableName);
        status = test.SetUp();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Setup failure \n");
            ret = -1;
            goto fail;
        }

        status = test.RunTest();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Runtime failure \n");
            ret = -1;
            goto fail;
        }
    } else if (cmdline.testMultiThread) {
        LOG_INFO("Test with profile name %s\n", cmdline.loadableName.c_str());

        // Multi Thread Test
        TestMT test(cmdline.loadableName);
        status = test.SetUp();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Setup failure \n");
            ret = -1;
            goto fail;
        }

        status = test.RunTest();
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("Test: Runtime failure \n");
            ret = -1;
            goto fail;
        }
    }

fail:
    if (cmdline.testRuntime) {
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> Runtime Test >>>>>>>>>>>>>>>>>>>> \n");
    } else if (cmdline.testPing) {
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> Ping Test >>>>>>>>>>>>>>>>>>>> \n");
    } else if (cmdline.testSciSync){
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> SciSync Test >>>>>>>>>>>>>>>>>>>> \n");
    } else {
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> MultiThread Test >>>>>>>>>>>>>>>>>>>> \n");
    }
    if (ret != 0) {
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> Test Fail >>>>>>>>>>>>>>>>>>>> \n");
    } else {
        LOG_INFO(">>>>>>>>>>>>>>>>>>>> Test Pass >>>>>>>>>>>>>>>>>>>> \n");
    }

    LOG_INFO("<<<<<<<<<<<<<<<<<<<< Test End <<<<<<<<<<<<<<<<<<<<< \n");
    return ret;

}
