
/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <vector>
#include <iomanip>

#include "NvSIPLTrace.hpp" // NvSIPLTrace to set library trace level
#include "NvSIPLQuery.hpp" // NvSIPLQuery to display platform config

#ifndef CCMDPARSER_HPP
#define CCMDPARSER_HPP

using namespace std;
using namespace nvsipl;

class CCmdLineParser
{
 public:
    // Command line options
    string sConfigName = "";
    vector<uint32_t> vMasks;
    string sTestConfigFile = "";
    int32_t uRunDurationSec = -1;
    uint32_t verbosity = 1u;
    bool bDisableRaw = true;
    bool bDisableISP0 = false;
    bool bDisableISP1 = false;
    bool bDisableISP2 = false;
    bool bShowFPS = false;
    bool bShowMetadata = false;
    PluginType autoPlugin = NV_PLUGIN;
    bool bAutoRecovery = false;
    bool bNvSci = false;
    bool bEnableProfiling = false;
    bool bEnableInitProfiling = false;
    string sNitoFolderPath = "";
    bool bShowEEPROM = false;
    bool bIgnoreError = false;
    bool bEnableSc7Boot = false;
    bool bIspInputCropEnable = false;
    uint32_t uIspInputCropY = 0u;
    uint32_t uIspInputCropH = 0u;
    bool bEnableStatsOverrideTest = false;
    bool bIsParkingStream = false;
    bool bEnableSubframe = false;
    string sProfilePrefix = "";
    bool bEnablePassive = false;
    string sFiledumpPrefix = "";
    uint32_t uNumSkipFrames = 0u;
    uint64_t uNumWriteFrames = -1u;
    bool bTwoPassIsp = false;
    bool bTwoPassLowToHigh = true;
#if !NV_IS_SAFETY
    uint64_t uCamRecCfg = 0u;
    uint32_t uExpNo = -1;
    uint32_t uNumDisplays = 0u;
    NvSiplRect oDispRect = {0, 0, 0, 0};
    bool bAutoLEDControl = false;
    // Non-safety only option to fetch and display NITO Metadata (UUID, dataHash, schemaHash) to
    // console. Let user indicate number of parameter sets present in NITO file, default to 10.
    bool bShowNitoMetadata = false;
    bool bSetNitoMetadataNumParamSets = false;
    uint64_t uShowNitoMetadataNumParamSets = 10u;

#endif // !NV_IS_SAFETY
    vector<string> vInputRawFiles;

    // Other members
    bool bRectSet = false;

    static void ShowConfigs(string sConfigFile="")
    {
        auto pQuery = INvSIPLQuery::GetInstance();
        if (pQuery == nullptr) {
            cout << "INvSIPLQuery::GetInstance() failed\n";
        }

        auto status = pQuery->ParseDatabase();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLQuery::ParseDatabase failed\n");
        }

        if (sConfigFile != "") {
            status = pQuery->ParseJsonFile(sConfigFile);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("Failed to parse test config file\n");
            }
            LOG_INFO("Available platform configurations\n");
        }
        for (auto &cfg : pQuery->GetPlatformCfgList()) {
            cout << "\t" << std::setw(35) << std::left << cfg->platformConfig << ":" << cfg->description << endl;
        }
    }

    static void ShowUsage(void)
    {
#if !NV_IS_SAFETY
        int32_t numDisplays = 1;
        // TODO Support number output displays query
#endif // !NV_IS_SAFETY

        cout << "Usage:\n";
        cout << "-h or --help                               :Prints this help\n";
        cout << "-c or --platform-config 'name'             :Platform configuration. Supported values\n";
        ShowConfigs();
        cout << "--link-enable-masks 'masks'                :Enable masks for links on each deserializer connected to CSI\n";
        cout << "                                           :masks is a list of masks for each deserializer.\n";
        cout << "                                           :Eg: '0x0000 0x1101 0x0000 0x0000' disables all but links 0, 2 and 3 on CSI-CD intrface\n";
        cout << "-r or --runfor <seconds>                   :Exit application after n seconds\n";
        cout << "-v or --verbosity <level>                  :Set verbosity\n";
#if !NV_IS_SAFETY
        cout << "                                           :Supported values (default: 1)\n";
        cout << "                                           : " << INvSIPLTrace::LevelNone << " (None)\n";
        cout << "                                           : " << INvSIPLTrace::LevelError << " (Errors)\n";
        cout << "                                           : " << INvSIPLTrace::LevelWarning << " (Warnings and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelInfo << " (Infos and above)\n";
        cout << "                                           : " << INvSIPLTrace::LevelDebug << " (Debug and above)\n";
#endif // !NV_IS_SAFETY
        cout << "-t or --test-config-file <file>            :Set custom platform config json file\n";
        cout << "-l or --list-configs                       :List configs from file specified via -t or --test-config-file\n";
        cout << "--enableRawOutput                          :Enable the Raw output\n";
        cout << "--disableISP0Output                        :Disable the ISP0 output\n";
        cout << "--disableISP1Output                        :Disable the ISP1 output\n";
        cout << "--disableISP2Output                        :Disable the ISP2 output\n";
        cout << "--showfps                                  :Show FPS (frames per second) every 2 seconds\n";
        cout << "--showmetadata                             :Show Metadata when RAW output is enabled\n";
        cout << "--plugin <type>                            :Auto Control Plugin. Supported types (default: If nito available 0 else 1)\n";
        cout << "                                           : " << NV_PLUGIN << " Nvidia AE/AWB Plugin\n";
        cout << "                                           : " << CUSTOM_PLUGIN0 << " Custom Plugin\n";
        cout << "--autorecovery                             :Recover deserializer link failure automatically\n";
        cout << "--nvsci                                    :Use NvSci for communication and synchronization\n";
        cout << "--profile <file-prefix>                    :Dump profiling timestamps in file-prefix_cam_x_out_y.csv file\n";
        cout << "--initProfile                              :Enable profiling of the initialization sequence\n";
        cout << "--nito <folder>                            :Path to folder containing NITO files\n";
        cout << "--icrop 'y+h'                              :Specifies the cropping at the input in the format 'y+h'\n";
        cout << "--showEEPROM                               :Show EEPROM data\n";
        cout << "--ignoreError                              :Ignore the fatal error\n";
        cout << "--enableStatsOverrideTest                  :Enable ISP statistics settings override\n";
        cout << "--enableSubframe                           :Enable Subframe feature\n";
        cout << "-i or --input-raw-files <file1[,file2]...> :Set input RAW files for simulator mode testing.\n";
        cout << "                                           :Use comma separator for multiple files.\n";
        cout << "--enablePassive                            :Enable passive mode\n";
        cout << "-f or --filedump-prefix 'str'              :Dump RAW file with filename prefix 'str'  when RAW output is enabled.\n";
        cout << "--skipFrames <val>                         :Number of frames to skip before writing to file\n";
        cout << "--writeFrames <val>                        :Number of frames to write to file\n";
        cout << "--twoPassISP <val>                         :Process the same captured frames through two different ISP pipelines\n";
        cout << "                                           :Set value to zero to send frames from pipeline with higher ID to pipeline with lower ID\n";
        cout << "                                           :Use any nonzero value to send frames from pipeline with lower ID to pipeline with higher ID\n";
#if !NV_IS_SAFETY
        cout << "--camRecCfg <val>                          :Recorder configuration\n";
        cout << "                                           :0 - No recorder support(default)\n";
        cout << "                                           :1 - Recorder support with the cable version 1\n";
        cout << "                                           :2 - Recorder support with the cable version 2\n";
        cout << "--setSensorCharMode <expNo>                :Set sensor in characterization mode with exposure number.\n";
        cout << "-d or --num-disp <num>                     :Number of displays\n";
        cout << "                                           :Number of available displays: " << numDisplays << endl;
        cout << "-p or --disp-win-pos 'rect'                :Display position, where rect is x0, y0, width, height\n";
        cout << "--autoLED                                  :Enable automatic LED control, specifically for AR0234\n";
        cout << "--showNitoMetadata                         :Get and display NITO Metadata (ID, dataHash, schemaHash) from NITO file(s) to console.\n";
        cout << "--numParameterSetsInNITO <val>             :Number of parameter sets in NITO file. Defaults to 10, must be >=1. Use if expecting >10 parameter sets in NITO file(s).\n";

#endif // !NV_IS_SAFETY

        return;
    }

    int Parse(int argc, char* argv[])
    {
        const char *const short_options = "hc:m:nr:v:t:lR012sMP:aCk:FN:O:eIoLi:Sf:K:W:T:"
#ifdef NVMEDIA_QNX
                                          "7"
#endif //NVMEDIA_QNX
#if !NV_IS_SAFETY
                                          "E:d:p:AGg:"
#endif // !NV_IS_SAFETY
                                          ;
        const struct option long_options[] =
        {
            { "help",                     no_argument,       0, 'h' },
            { "platform-config",          required_argument, 0, 'c' },
            { "link-enable-masks",        required_argument, 0, 'm' },
            { "enable-notification",      no_argument,       0, 'n' },
            { "runfor",                   required_argument, 0, 'r' },
            { "verbosity",                required_argument, 0, 'v' },
            { "test-config-file",         required_argument, 0, 't' },
            { "list-configs",             no_argument,       0, 'l' },
            { "enableRawOutput",          no_argument,       0, 'R' },
            { "disableISP0Output",        no_argument,       0, '0' },
            { "disableISP1Output",        no_argument,       0, '1' },
            { "disableISP2Output",        no_argument,       0, '2' },
            { "showfps",                  no_argument,       0, 's' },
            { "showmetadata",             no_argument,       0, 'M' },
            { "plugin",                   required_argument, 0, 'P' },
            { "autorecovery",             no_argument,       0, 'a' },
            { "nvsci",                    no_argument,       0, 'C' },
            { "profile",                  required_argument, 0, 'k' },
            { "initProfile",              no_argument,       0, 'F' },
            { "nito",                     required_argument, 0, 'N' },
            { "icrop",                    required_argument, 0, 'O' },
            { "showEEPROM",               no_argument,       0, 'e' },
            { "ignoreError",              no_argument,       0, 'I' },
            { "enableStatsOverrideTest",  no_argument,       0, 'o' },
            { "enableSubframe",           no_argument,       0, 'u' },
            { "input-raw-files",          required_argument, 0, 'i' },
            { "enablePassive",            no_argument,       0, 'S' },
            { "filedump-prefix",          required_argument, 0, 'f' },
            { "skipFrames",               required_argument, 0, 'K' },
            { "writeFrames",              required_argument, 0, 'W' },
            { "twoPassISP",               required_argument, 0, 'T' },
#if !NV_IS_SAFETY
            { "camRecCfg",                required_argument, 0, 'w' },
#endif // !NV_IS_SAFETY
#ifdef NVMEDIA_QNX
            { "sc7-boot",                 no_argument,       0, '7' },
#endif //NVMEDIA_QNX
#if !NV_IS_SAFETY
            { "setSensorCharMode",        required_argument, 0, 'E' },
            { "num-disp",                 required_argument, 0, 'd' },
            { "disp-win-pos",             required_argument, 0, 'p' },
            { "autoLED",                  no_argument,       0, 'A' },
            { "showNitoMetadata",         no_argument,       0, 'G' },
            { "numParameterSetsInNITO",   required_argument, 0, 'g' },
#endif // !NV_IS_SAFETY
            { 0,                          0,                 0,  0 }
        };

        int index = 0;
        auto bShowHelp = false;
        auto bShowConfigs = false;

        while (1) {
            const auto getopt_ret = getopt_long(argc, argv, short_options , &long_options[0], &index);
            if (getopt_ret == -1) {
                // Done parsing all arguments.
                break;
            }

            switch (getopt_ret) {
            default: /* Unrecognized option */
            case '?': /* Unrecognized option */
                cout << "Invalid or Unrecognized command line option. Specify -h or --help for options\n";
                bShowHelp = true;
                break;
            case 'h': /* -h or --help */
                bShowHelp = true;
                break;
            case 'c':
                sConfigName = string(optarg);
                break;
            case 'm':
            {
                char* token = std::strtok(optarg, " ");
                while(token != NULL) {
                    vMasks.push_back(stoi(token, nullptr, 16));
                    token = std::strtok(NULL, " ");
                }
            }
                break;
            case 'r':
                uRunDurationSec = atoi(optarg);
                break;
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 't':
                sTestConfigFile = string(optarg);
                break;
            case 'l':
                bShowConfigs = true;
                break;
            case 'R':
                bDisableRaw = false;
                break;
            case '0':
                bDisableISP0 = true;
                break;
            case '1':
                bDisableISP1 = true;
                break;
            case '2':
                bDisableISP2 = true;
                break;
            case 's':
                bShowFPS = true;
                break;
            case 'M':
                bShowMetadata = true;
                break;
            case 'P':
                autoPlugin = (PluginType) atoi(optarg);
                break;
            case 'a':
                bAutoRecovery = true;
                break;
            case 'C':
                bNvSci = true;
                break;
            case 'k':
                bEnableProfiling = true;
                if (optarg[0] == '-') {
                    cout << "Invalid file prefix for profiling.\n";
                    bShowHelp = true;
                    break;
                }
                sProfilePrefix = string(optarg);
                break;
            case 'F':
                bEnableInitProfiling = true;
                break;
            case 'N':
                sNitoFolderPath = string(optarg);
                break;
            case 'O':
                sscanf(optarg, "%d+%d", &uIspInputCropY, &uIspInputCropH);
                bIspInputCropEnable = true;
                break;
            case 'e':
                bShowEEPROM = true;
                break;
            case 'I':
                bIgnoreError = true;
                break;
#ifdef NVMEDIA_QNX
            case '7':
                bEnableSc7Boot = true;
                break;
#endif //NVMEDIA_QNX
            case 'o':
                bEnableStatsOverrideTest = true;
                break;
            case 'u':
                bEnableSubframe = true;
                break;
            case 'i':
                char *rawFile;
                rawFile = strtok(optarg, ",");
                while (rawFile != NULL) {
                    vInputRawFiles.push_back(string(rawFile));
                    rawFile = strtok (NULL, ",");
                }
                break;
            case 'S':
                bEnablePassive = true;
                break;
            case 'f':
                sFiledumpPrefix = string(optarg);
                break;
            case 'K':
                uNumSkipFrames = atoi(optarg);
                break;
            case 'W':
                uNumWriteFrames = atoi(optarg);
                break;
            case 'T':
                bTwoPassIsp = true;
                bTwoPassLowToHigh = (atoi(optarg) != 0);
                break;
#if !NV_IS_SAFETY
            case 'w':
                uCamRecCfg = atoi(optarg);
                break;
            case 'E':
                uExpNo = atoi(optarg);
                break;
            case 'd':
                uNumDisplays = atoi(optarg);
                if (uNumDisplays == 0u) {
                    cout << "Warning: number of displays cannot be zero, setting to one" << endl;
                    uNumDisplays = 1u;
                }
                break;
            case 'p':
                int32_t x, y, w, h;
                sscanf(optarg, "%d %d %d %d", &x, &y, &w, &h);
                oDispRect.x0 = x;
                oDispRect.y0 = y;
                oDispRect.x1 = oDispRect.x0 + w;
                oDispRect.y1 = oDispRect.y0 + h;
                bRectSet = true;
                break;
            case 'A':
                bAutoLEDControl = true;
                break;
            case 'G':
                bShowNitoMetadata = true;
                break;
            case 'g':
                bSetNitoMetadataNumParamSets = true;
                uShowNitoMetadataNumParamSets = atoi(optarg);
                break;
#endif // !NV_IS_SAFETY
            }
        }

        if (bShowHelp) {
            ShowUsage();
            return -1;
        }

        if (bShowConfigs) {
            // User just wants to list available configs
            ShowConfigs(sTestConfigFile);
            return -1;
        }

        // Check for bad arguments
        if (sConfigName == "") {
            cout << "No platform configuration specified.\n";
            return -1;
        }

        if (bDisableRaw &&
            bDisableISP0 &&
            bDisableISP2 &&
            bDisableISP1) {
            cout << "At-least one output must be enabled\n";
            return -1;
        }

        if ((sFiledumpPrefix == "") && (uNumSkipFrames != 0u)) {
            cout << "skipFrames is only applicable when file dump is enabled\n";
            return -1;
        }

        if ((sFiledumpPrefix == "") && (uNumWriteFrames != -1u)) {
            cout << "writeFrames is only applicable when file dump is enabled\n";
            return -1;
        }

        if (bTwoPassIsp) {
            if (bDisableRaw) {
                cout << "Cannot perform two-pass ISP processing without capture output\n";
                return -1;
            }
            if (bDisableISP0 && bDisableISP1 && bDisableISP2) {
                cout << "Cannot perform two-pass ISP processing without any ISP outputs\n";
                return -1;
            }
            if (bNvSci || (!vInputRawFiles.empty())
#if !NV_IS_SAFETY
                || (uNumDisplays != 0U)
#endif // !NV_IS_SAFETY
                ) {
                cout << "Cannot perform two-pass ISP processing with the given consumer feature\n";
                return -1;
            }
        }

#if !NV_IS_SAFETY
        if (bAutoLEDControl) {
            string config_substr = "AR0234";
            if (!strstr(sConfigName.c_str(), config_substr.c_str())) {
                cout << "LED control is only supported for AR0234 camera modules\n";
                return -1;
            }
        }

        string park_substr = "PARKING_STREAM";
        if (strstr(sConfigName.c_str(), park_substr.c_str())) {
            // Display is not supported for platform configuration PARKING_STREAM
            if (uNumDisplays > 0u) {
                cout << "Display is not supported for the given platform configuration\n";
                return -1;
            }
            bIsParkingStream = true;
        }

        string config_substr = "OV2312";
        if (strstr(sConfigName.c_str(), config_substr.c_str())) {
            if (!bDisableISP2) {
                cout << "ISP2 output is not supported for RGB-IR sensors such as OV2312\n";
                return -1;
            }
            if ((uNumDisplays > 0u) && !bDisableISP1) {
                cout << "Displaying ISP1 output is not supported for RGB-IR sensors such as OV2312\n";
                return -1;
            }
        }

        if ((uNumDisplays > 0u) && !bDisableRaw) {
            // Throw warning if RAW output is being displayed
            cout << "WARNING: RAW output is being requested with Display\n";
            cout << "WARNING: Frame Drops and drops in FPS are likely to be observed\n";
        }

        // User must enable at least one ISP output to use NitoMetadata option
        if (bDisableISP0 &&
            bDisableISP1 &&
            bDisableISP2 &&
            bShowNitoMetadata) {
            cout << "showNitoMetadata only applicable when at least 1 ISP output is enabled\n";
            return -1;
        }

        // User must enable showNitoMetadata to specify numParameterSetsInNITO
        if (!bShowNitoMetadata && bSetNitoMetadataNumParamSets) {
            cout << "--numParameterSetsInNITO only applicable when --showNitoMetadata is enabled\n";
            return -1;
        }

        if (bShowNitoMetadata && bSetNitoMetadataNumParamSets) {
            if (uShowNitoMetadataNumParamSets < 1u) {
                cout << "--numParameterSetsInNITO must be >=1 \n";
                return -1;
            }
        }

        if (uExpNo != -1u) {
            // SetSensorCharMode to be used only for Custom Plugin
            if (autoPlugin == CUSTOM_PLUGIN0) {
                string config_substr = "F008A";
                /* SetSensorCharMode is not supported for platform configurations that do not match
                 * sub-string F008A which corresponds to AR0820 sensors supported
                 */
                if (!strstr(sConfigName.c_str(), config_substr.c_str())) {
                    cout << "SetSensorCharMode is not supported for the given platform configuration\n";
                    return -1;
                }
            } else {
                cout << "SetSensorCharMode is not supported for requested Plugin\n";
                return -1;
            }
        }
#endif // !NV_IS_SAFETY

        return 0;
    }

    void PrintArgs() const
    {
        if (sConfigName != "") {
            cout << "Platform configuration name: " << sConfigName << endl;
        }
        if (sTestConfigFile != "") {
            cout << "Platform configuration file: " << sTestConfigFile << endl;
        }
        if (uRunDurationSec != -1) {
            cout << "Running for " << uRunDurationSec << " seconds\n";
        }
        cout << "Verbosity level: " << verbosity << endl;
        if (bDisableRaw) {
            cout << "Raw output: disabled" << endl;
        } else {
            cout << "Raw output: enabled" << endl;
        }
        if (bDisableISP0) {
            cout << "ISP0 output: disabled" << endl;
        } else {
            cout << "ISP0 output: enabled" << endl;
        }
        if (bDisableISP1) {
            cout << "ISP1 output: disabled" << endl;
        } else {
            cout << "ISP1 output: enabled" << endl;
        }
        if (bDisableISP2) {
            cout << "ISP2 output: disabled" << endl;
        } else {
            cout << "ISP2 output: enabled" << endl;
        }
        if (bShowFPS) {
            cout << "Enabled FPS logging" << endl;
        } else {
            cout << "Disabled FPS logging" << endl;
        }
        if (bShowMetadata) {
            cout << "Enabled Metadata logging" << endl;
        } else {
            cout << "Disabled Metadata logging" << endl;
        }
        if (bAutoRecovery) {
            cout << "Enabled automatic recovery" << endl;
        } else {
            cout << "Disabled automatic recovery" << endl;
        }
        if (bNvSci) {
            cout << "Enabled NvSci" << endl;
        } else {
            cout << "Disabled NvSci" << endl;
        }
        if (bEnableProfiling) {
            cout << "Enabled profiling" << endl;
        } else {
            cout << "Disabled profiling" << endl;
        }
        if (bEnableInitProfiling) {
            cout << "Enabled initialization profiling" << endl;
        } else {
            cout << "Disabled initialization profiling" << endl;
        }
        if (sNitoFolderPath != "") {
            cout << "NITO folder path: " << sNitoFolderPath << endl;
        }
        if (bEnableStatsOverrideTest){
            cout << "Enabled ISP Statistics settings override" << endl;
        } else {
            cout << "Disabled ISP Statistics settings override" << endl;
        }
        if (bEnableSubframe) {
            cout << "Subframe : enabled" << endl;
        } else {
            cout << "Subframe : disabled" << endl;
        }
        if (bEnablePassive) {
            cout << "Enabled Passive mode" << endl;
        } else {
            cout << "Disabled Passive mode" << endl;
        }
        if (sFiledumpPrefix != "") {
            cout << "File dump prefix:            " << sFiledumpPrefix << endl;
        }
        if (!vInputRawFiles.empty()) {
            cout << "Input raw file(s):           ";
            for (auto s : vInputRawFiles) {
                cout << s << " ";
            }
            cout << endl;
        }
        if (uNumSkipFrames != 0u) {
            cout << "Number of frames to skip: " << uNumSkipFrames << endl;
        }
        if (uNumWriteFrames != -1u) {
            cout << "Number of frames to write: " << uNumWriteFrames << endl;
        }
#if !NV_IS_SAFETY
        if (uNumDisplays > 0u) {
            cout << "Number of displays:          " << uNumDisplays << endl;
            if (bRectSet) {
                uint32_t x = oDispRect.x0, y = oDispRect.y0;
                uint32_t w = oDispRect.x1 - x, h = oDispRect.y1 - y;
                cout << "Display window position:     " << x << " " << y << " " << w << " " << h << endl;
            }
        }
        if (bShowNitoMetadata) {
            cout << "Enabled retrieving/displaying NITO Metadata" << endl;
            if (bSetNitoMetadataNumParamSets) {
                cout << "User specified that all NITO files will have maximum of " << uShowNitoMetadataNumParamSets << " parameter sets\n" << endl;
            }
            else {
                cout << "Expecting all NITO files to have default maximum of " << uShowNitoMetadataNumParamSets << " parameter sets\n" << endl;
            }
        } else {
            cout << "Disabled retrieving/displaying NITO Metadata" << endl;
        }
#endif // !NV_IS_SAFETY
    }
};

#endif //CCMDPARSER_HPP
