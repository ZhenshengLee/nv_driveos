#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys, getopt, re

DEBUG = False
INFO = False

PROG_NAME   = ""
config_name = ""
base_intra_vm = -1 #base offset for intra-vm
base_inter_vm = -1 #base offset for inter-vm
base_c2c = -1 #base offset for c2c

def debug_log(s):
    if DEBUG:
        print(s)

def info_log(s):
    if INFO:
        print(s)

def err_log(s):
    print(s)

def print_help():
    print(
     "Usage: python " + PROG_NAME + " [OPTION]...\n"
     "Option:"
     "\t -c <config file name>\n"
     "\t    NOTE:The same file and sub config file included in the file will be\
 updated with gid and should be writable\n"
     "\t -i GID base offset for intra-vm(inter-process or inter-thread)\n"
     "\t -v GID base offset for inter-vm\n"
     "\t    NOTE: This should be greater than base offset for intra-vm \n"
     "\t -x GID base offset for c2c(chip to chip)\n"
     "\t    NOTE: This should be greater than base offset for intra-vm and\
 inter-vm\n"
     "\t -h print this help screen")

def validate_base_offset(intra_vm, inter_vm, c2c):
    if inter_vm > -1:
        if inter_vm <= intra_vm:
            err_log(PROG_NAME + ": invalid inter-vm base offset")
            err_log("intra-vm: " + str(intra_vm) +
                        ", inter-vm: " + str(inter_vm))
            sys.exit()

    if c2c > -1:
        if inter_vm == -1 and c2c <= intra_vm:
            err_log(PROG_NAME + ": invalid c2c base offset")
            err_log("intra-vm: " + str(intra_vm) + ", c2c: " + str(c2c))
            sys.exit()
        elif inter_vm > -1 and c2c <= inter_vm:
                err_log(PROG_NAME + ": invalid c2c base offset")
                err_log("inter-vm: " + str(inter_vm) + ", c2c: " + str(c2c))
                sys.exit()

def parse_opt(argv):
    global PROG_NAME
    global config_name
    global base_intra_vm
    global base_inter_vm
    global base_c2c

    PROG_NAME = argv[0]

    #Get configure file and secpolicy file
    try:
        opts, etc_args = getopt.getopt(argv[1:], \
                                "c:i:h:v:x:", ["help"])
    except getopt.GetoptError:
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "help"):
            print_help()
            sys.exit()
        elif opt in ("-c", ""):
            config_name = arg
        elif opt in ("-i", ""):
            base_intra_vm = int(arg)
        elif opt in ("-v", ""):
            base_inter_vm = int(arg)
        elif opt in ("-x", ""):
            base_c2c = int(arg)
        else:
            err_log(PROG_NAME + ": unsupposed option " + opt)
            sys.exit()

    if len(config_name) == 0:
        print_help()
        sys.exit()

    debug_log("config_name: " + config_name)
    debug_log("base_intra_vm= " + str(base_intra_vm) + ", " +
              "base_inter_vm= " + str(base_inter_vm) + ", " +
              "base_c2c= " + str(base_c2c))

def read_cfile(config_file):
    global base_intra_vm
    global base_inter_vm
    global base_c2c

    fqueue = [] #FIFO storing config file
    fqueue.append(config_file)


    while len(fqueue) > 0:
        fname = fqueue.pop(0)
        line_num = 0

        #open config file and read all lines
        try:
            f = open(fname, "r")
        except:
            err_log(PROG_NAME + ": failed to open file " + fname)
            sys.exit()

        lines = f.readlines()
        f.close()

        #open config file and add or update gid at the end of line
        f = open(fname, "w")
        has_version = False

        for line in lines:
            line_num = line_num + 1
            line = line.strip()

            if line[0] == '#':
                #check if CFGVER is the comment
                if "CFGVER:" in line:
                    vers = line.split(":")
                    debug_log(vers)
                    if int(vers[1]) < 1: #version is less than '1'
                        vers[1] = '1'
                        line = vers[0] + ":" + vers[1]
                    has_version = True
                f.write(line + '\n')
                continue

            if not has_version:
                f.write("#\n#CFGVER:1\n#\n")
                has_version = True

            words = line.split() #split text by space
            if words[0] == "INTER_PROCESS" or words[0] == "INTER_THREAD":
                if base_intra_vm > -1:
                    #check if GID overwrites base inter-vm offset
                    if base_inter_vm > -1 and base_intra_vm >= base_inter_vm:
                        err_log(PROG_NAME +
                          ": intra-vm GID is going to overwrite base inter-vm")
                        err_log("skipping... " + words[1] + " GID: " +
                          str(base_intra_vm))
                    #check if GID overwrites base c2c offset
                    elif (base_inter_vm == -1 and base_c2c > -1) and \
                            base_intra_vm >= base_c2c:
                        err_log(PROG_NAME +
                          ": intra-vm GID is going to overwrite base c2c")
                        err_log("skipping... " + words[1] + " GID: " +
                          str(base_intra_vm))
                    elif len(words) == 6: #It has gid in the last word
                        #replace the existing gid with new one
                        lastword = words[-1]
                        line = line.replace(lastword, str(base_intra_vm))
                    else: #It has no gid and add new one
                        line = line + "    " + str(base_intra_vm)

                    base_intra_vm += 1
            elif words[0] == "INTER_VM":
                if base_inter_vm > -1:
                    #check if GID overwrites base c2c offset
                    if base_c2c > -1 and base_inter_vm >= base_c2c:
                        err_log(PROG_NAME +
                          ": inter-vm GID is going to overwrite base c2c")
                        err_log("skipping... " + words[1] + " GID: " +
                          str(base_inter_vm))
                    elif len(words) == 4: #It has gid in the last word
                        #replace the existing gid with new one
                        lastword = words[-1]
                        line = line.replace(lastword, str(base_inter_vm))
                    else: #It has no gid and add new one
                        line = line + "    " + str(base_inter_vm)

                    base_inter_vm += 1
            elif words[0] == "INTER_CHIP":
                if base_c2c > -1:
                    if len(words) == 4: #It has gid in the last word
                        #replace the existing gid with new one
                        lastword = words[-1]
                        line = line.replace(lastword, str(base_c2c))
                    else: #It has no gid and add new one
                        line = line + "    " + str(base_c2c)
                    base_c2c += 1
            elif words[0] == "include": #including another config file
                fqueue.append(words[1])
            else:
                err_log(PROG_NAME + ": unsupported backend type "
                        + words[0])

            f.write(line + '\n')

        f.close()

def main():
    #parse command line
    parse_opt(sys.argv)

    #validate base offset
    validate_base_offset(base_intra_vm, base_inter_vm, base_c2c)

    #open config file, add or update gid
    read_cfile(config_name)

if __name__ == "__main__":
    main()
