#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys, getopt

DEBUG = False
INFO = False

PROG_NAME   = ""
config_name = ""
secpolicy_name = ""
gen_gid = False #flag to generate gid
assign_gid = False #flag to assign gid
gids = {} #endpoint to gid dictionary
uids = {} #uid to endpoints dictionary
gid_nvsciipc = -1
nvsciipc_name = "@@NVSCIIPC@@" #reserved name not used in config file

def debug_log(s):
    if DEBUG:
        print(s)

def info_log(s):
    if INFO:
        print(s)

def err_log(s):
    print(s)

def print_help():
    print("Usage: python " + PROG_NAME + " [OPTION]...\n"
          "Option:"
          "\t -c <config file name, config file version >= 1, i.e. #CFGVER:1>\n"
          "\t -p <secpolicy file name>\n"
          "\t -g add gid to /etc/group\n"
          "\t -a assign the gid which is added with '-g' to uid in sepolicy \
file\n"
          "\t -u <id> add 'nvsciipc' gid with id and assign the nvsciipc gid \
to all uids in sepolicy file\n"
          "\t -h print this help screen")

def parse_opt(argv):
    global PROG_NAME
    global config_name
    global secpolicy_name
    global gen_gid
    global assign_gid
    global gids
    global gid_nvsciipc

    PROG_NAME = argv[0]

    # Get configure file and secpolicy file
    try:
        opts, etc_args = getopt.getopt(argv[1:], \
                                "ac:ghp:u:", ["help"])
    except getopt.GetoptError:
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "help"):
            print_help()
            sys.exit()
        elif opt in ("-c", ""):
            config_name = arg
        elif opt in ("-p", ""):
            secpolicy_name = arg
        elif opt in ("-g", ""):
            gen_gid = True
        elif opt in ("-a", ""):
            assign_gid = True
        elif opt in ("-u", ""):
            gid_nvsciipc = int(arg)
            if gid_nvsciipc > 0:
                gids[nvsciipc_name] = [arg, nvsciipc_name, True]
        else:
            err_log(PROG_NAME + ": unsupposed option " + opt)
            sys.exit()

    if len(config_name) == 0 or len(secpolicy_name) == 0:
        print_help()
        sys.exit()

    debug_log("config_name: " + config_name)
    debug_log("secpolicy_name: " + secpolicy_name)

def read_cfile(config_file):
    global gids

    fqueue = [] #FIFO storing config file
    fqueue.append(config_file)

    while len(fqueue) > 0:
        fname = fqueue.pop(0)
        line_num = 0

        try:
            f = open(fname, "r")
        except:
            err_log(PROG_NAME + ": failed to open file " + fname)
            sys.exit()

        lines = f.readlines()
        has_version = False
        for line in lines:
            line_num = line_num + 1
            line = line.strip()

            if line[0] == '#':
                debug_log("comment: " + line)
                #check if CFGVER is the comment
                if "CFGVER:" in line:
                    vers = line.split(":")
                    if int(vers[1]) >= 1: #version >= 1
                        has_version = True
                continue

            if not has_version:
                err_log(PROG_NAME + ": CFGVER is not valid")
                sys.exit()

            words = line.split() #split text by space
            try:
                if words[0] == "INTER_PROCESS" or words[0] == "INTER_THREAD":
                    debug_log("ep1: " + words[1] + ", gid: " + words[5])
                    debug_log("ep2: " + words[2] + ", gid: " + words[5])

                    #ep1 stores gid and backend
                    gids[words[1]] = [words[5], words[0], True]
                    #ep2 stores gid and backend
                    gids[words[2]] = [words[5], words[0], False]
                elif words[0] == "INTER_VM" or words[0] == "INTER_CHIP":
                    debug_log("ep: " + words[1] + ", gid: " + words[3])

                    #ep stores gid and backend
                    gids[words[1]] = [words[3], words[0], True]
                elif words[0] == "include": #including another config file
                    fqueue.append(words[1])
                else:
                    err_log(PROG_NAME + ": unsupported backend type "
                            + words[0])

            except:
                err_log(PROG_NAME + ": malformed config file " + fname)
                err_log(line_num + ": " + words)
                f.close()
                sys.exit()

        f.close()
    debug_log(gids)

def read_pfile(fname):
    global uids

    line_num = 0

    try:
        f = open(fname, "r")
    except:
        err_log(PROG_NAME + ": failed to open file " + fname)
        sys.exit()

    lines = f.readlines()
    for line in lines:
        line_num = line_num + 1
        line = line.strip()

        if line[0] == '#':
            debug_log("comment: " + line)
            #skip comment
            continue

        #split text by space
        words = line.split(":")
        try:
            endpoints = words[1].split(",")
            if gid_nvsciipc > 0: #add nvsciipc gid to endpoints
                endpoints.append(nvsciipc_name)
            uids[words[0]] = endpoints #uid stores list of endpoints

        except:
            err_log(PROG_NAME + ": malformed config file " + fname)
            err_log(line_num + ": " + words)
            f.close()
            sys.exit()

    f.close()
    debug_log(uids)

def add_gid(gids):
    for ep in gids:
        cmd_line = "groupadd -g "
        debug_log(ep + ": " + gids[ep][0] + ", " + gids[ep][1])
        if gids[ep][1] == "INTER_THREAD" or \
                gids[ep][1] == "INTER_PROCESS":
            # avoid duplicated gid
            if gids[ep][2] == True:
                cmd_line = cmd_line + gids[ep][0] + " ipc" + gids[ep][0]
            else:
                continue
        elif gids[ep][1] == "INTER_VM":
            cmd_line = cmd_line + gids[ep][0] + " ivc" + gids[ep][0]
        elif gids[ep][1] == "INTER_CHIP":
            cmd_line = cmd_line + gids[ep][0] + " c2c" + gids[ep][0]
        elif gids[ep][1] == nvsciipc_name:
            cmd_line = cmd_line + gids[ep][0] + " nvsciipc"

        info_log(cmd_line)
        #run cmd_line
        os.system(cmd_line)

def assign_gid_to_uid(uids, gids):
    for uid in uids:
        cmd_line = "usermod -a -G"
        endpoints = uids[uid]
        try:
            inx = 0
            for ep in endpoints:
                if gids[ep][2] == True:
                    if inx == 0: #first gid
                        cmd_line = cmd_line + " " + gids[ep][0]
                    else:
                        cmd_line = cmd_line + "," + gids[ep][0]
                inx = inx + 1
        except:
            print(PROG_NAME, "No gid is matched with the endpoint", ep)
            sys.exit()

        cmd_line = cmd_line + " " + uid
        info_log(cmd_line)
        #run cmd_line
        os.system(cmd_line)

def main():
    #parse command line
    parse_opt(sys.argv)

    #open config file and read endpoint name and gid
    read_cfile(config_name)

    #open secpolic file and read uid and list of endpoints
    read_pfile(secpolicy_name)

    #add gid to group file using "groupadd -g {GID#} {GIDname}"
    if gen_gid:
        add_gid(gids)

    #assign gid to each uid using "usermod -a -G {GIDs,...} {UID}"
    if assign_gid:
        assign_gid_to_uid(uids, gids)

if __name__ == "__main__":
    main()
