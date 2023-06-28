#!/bin/bash

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# This script starts the weston with weston-launch utility.
# Returns 1 on failure, 0 on success.

drm_name=""
drm_driver=""
CHIP="$(tr -d '\0' < /proc/device-tree/compatible)"
AUTO_GROUP_ADD=1
#Color for any error messages.
RED='\033[0;31m'
#Color for any info messages.
LIGHTGRAY='\033[0;37m'
#Color for any success messages.
GREEN='\033[0;32m'
#No color or white, for any
#instructions to user.
WHITE='\033[0m'

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-add-group)
      AUTO_GROUP_ADD=0
      ;;
    *)
      echo -e >&2 "${RED}Unknown option $1"
      tput sgr0
      exit 1
      ;;
  esac
shift
done

trap '' HUP

if [[ ${CHIP} =~ "tegra210" ]] || [[ ${CHIP} =~ "tegra186" ]] || [[ ${CHIP} =~ "tegra194" ]]; then
    drm_name="tegra_udrm"
    drm_driver="tegra-udrm"
else
    drm_name="nvidia_drm"
    drm_driver="nvidia-drm"
fi

#load DRM module based on the chip-id
function load_drm_module {
    lsmod | grep -q "${drm_name}"
    local result="${?}"
    if [[ "${result}" -ne 0 ]]; then
        sudo modprobe ${drm_driver} modeset=1
    fi
}

#add user to weston-launch and render group
#if the user is not already part of.
function add_user_group() {
    group_update_done=0
    if [[ $AUTO_GROUP_ADD == 1 ]]; then
        for group in weston-launch render
        do
            #check the user is part of the group
            groups $USER | grep -q $group
            local result="${?}"
            if [[ "${result}" -ne 0 ]]; then
                sudo usermod -a -G $group $USER
                group_update_done=1
                echo -e "${LIGHTGRAY}$USER is not part of group:$group, successfully added."
            fi
        done
        if [ "${group_update_done}" -ne 0 ]; then
            echo -e ""
            echo -e "${WHITE}For all group changes to take effect, please reboot."
            echo -e "${WHITE}Alternatively, execute the following commands for the group changes to take effect immediately."
            echo -e "${WHITE}    newgrp render"
            echo -e "${WHITE}    newgrp weston-launch"
            echo -e ""
            #using nested sg commands as it accepts one group per command.
            exec sg weston-launch -c "sg render -c ${BASH_SOURCE[0]} --no-add-group"
            false
            return
        fi
    fi
    true
    return
}

#check whether x11 is running
is_x11_running() {
    pgrep Xorg > /dev/null  2>&1
}

is_weston_launch_running() {
    pgrep weston-launch  > /dev/null  2>&1
}

is_weston_running() {
    pgrep weston-desktop  > /dev/null  2>&1
}

#stop gdm, x11 if already running
stop_x11() {
    is_x11_running
    result="$?"
    if [ "${result}" -eq "0" ]; then
        #stop gdm
        sudo service gdm stop;
        pkill -15 Xorg
    fi
}

#start weston
#    1. check weston not already running
#    2. kill x11 if already running
#    3. prepare some prerequisites
#        i. load DRM module
#        ii. create libgbm.so.1 symlink
#    4. launch weston with weston-launch utility
start_weston() {
    #check weston is already running
    is_weston_running
    result="$?"

    if [ "${result}" -ne "0" ]; then
        if add_user_group; then
            #do pre-requisites
            stop_x11
            load_drm_module
            unset DISPLAY

            echo -e "${LIGHTGRAY}Launching weston..."
            nohup weston-launch &> /tmp/weston-startup.log &
            sleep 2

            #check weston-launch running
            is_weston_launch_running
            result="$?"

            #wait for weston to startup if weston-launch is running
            if [ "${result}" -eq "0" ]; then
                for _ in 1 2 3 4 5; do
                    is_weston_running
                    result="$?"
                    if [ "${result}" -eq "0" ]; then
                        echo -e ""
                        echo -e "${GREEN}!!! Weston launched successfully !!!"
                        echo -e ""
                        echo -e "${WHITE}Weston startup log at:/tmp/weston-startup.log"
                        echo -e ""
                        echo -e "${WHITE}To exit weston gracefully, execute:"
                        echo -e "${WHITE}    pkill -15 weston"
                        echo -e ""
                        echo -e "${WHITE}Execute the following steps before starting X11 again:"
                        echo -e "${WHITE}    pkill -15 weston"
                        echo -e "${WHITE}    sudo rmmod ${drm_driver}"
                        tput sgr0
                        exit 0
                    else
                        sleep 4
                    fi
                done
            fi

            #check the weston-launch fails due to iinvalid group permissions
            cat /tmp/weston-startup.log | grep -q "weston-launch: Permission denied"
            result="$?"
            if [ "${result}" -eq "0" ]; then
                echo -e "${RED}!!! Failed to start Weston !!!"
                echo -e "${WHITE}Possibly needs a reboot to ensure that user "ubuntu" is part of the required groups."
                echo -e "${WHITE}Alternatively, execute the following commands and relaunch this script."
                echo -e "${WHITE}    newgrp render"
                echo -e "${WHITE}    newgrp weston-launch"
                echo -e ""
            else
                echo -e "${RED}!!! Failed to start weston !!!"
                echo -e ""
                echo -e "${RED}Failure log at:/tmp/weston-startup.log"
            fi
            tput sgr0
            exit 1
        fi
    else
        echo -e "${GREEN}!!! Weston is already running !!!"
        tput sgr0
        exit 0
    fi
}

start_weston

