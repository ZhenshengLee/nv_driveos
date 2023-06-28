/*
 * Copyright (c) 2018-2022, NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <libgen.h>

#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>

#include <inttypes.h>
#include <nvsciipc.h>
#include <nvscievent.h>

#define MAX_ENDPOINT    1000 // limited by NvSciIpc library (500)
#define MAX_EVENTNOTIFIER MAX_ENDPOINT
#define DEFAULT_INIT_COUNT       1
#define DEFAULT_COUNT        128

#define EOK 0

struct endpoint
{
    char chname[NVSCIIPC_MAX_ENDPOINT_NAME];    /* endpoint name */
    NvSciIpcEndpoint h;  /* NvSciIpc handle */
    struct NvSciIpcEndpointInfo info; /* endpoint info */
    void *buf;  /* test buffer */
    uint32_t evt_cnt;
    NvSciEventNotifier *eventNotifier;
};

typedef struct {
    NvSciEventLoopService *eventLoopServiceP;
    struct endpoint ep;
    uint32_t epCnt; /* endpoint count */
    uint64_t iterations;
    int64_t timeout; /* msec timeout for WaitForEvent/NvSciIpcWaitEventQnx */
    char *prgname;
    bool initFlag;  /* init_resources() is executed successfully */
} test_params;

static bool s_verboseFlag = false;
static bool s_quietFlag = false;
static uint32_t s_Stop;
static test_params s_params;

#define mprintf(fmt, args...) \
    if (!s_quietFlag) { \
        printf(fmt, ## args); \
    }

#define dprintf(fmt, args...) \
    if (s_verboseFlag && !s_quietFlag) { \
        printf(fmt, ## args); \
    }

void print_usage(char *argv[]);
int32_t write_test(struct endpoint *ep);
NvSciError wait_event(struct endpoint *ep, int32_t value);
static void setup_termination_handlers(void);
static NvSciError init_resources(struct endpoint *ep);
static void release_resources(struct endpoint *ep);
static void *write_test_main(void *arg);

void print_usage(char *argv[])
{
    fprintf(stderr, "Usage: %s [OPTION]...\n", argv[0]);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "\t -h                 : "
            "Print this help screen\n");
    fprintf(stderr, "\t -c <endpoint_name> : "
            "name of NvSciIpc endpoint\n");
    fprintf(stderr, "\t -l                 : loop count\n");
    fprintf(stderr, "\t -v                 : verbose mode\n");
    fprintf(stderr, "\t -T {msec}          : "
            "use NvSciEventService with msec timeout\n"
            "\t\twait infinitely if msec is U (-E U)\n");
}

NvSciError wait_event(struct endpoint *ep, int32_t value)
{
    test_params *tp = &s_params;
    uint32_t event = 0;
    int64_t timeout;
    NvSciError err;

    while(!s_Stop) {
        err = NvSciIpcGetEventSafe(ep->h, &event);
        if (err != NvSciError_Success) {
            mprintf("[%20s] %s: get event: %d\n",
                tp->prgname, __func__, err);
            return err;
        }
        if (event & value) {
            break;
        }
        dprintf("[%20s] %s: event: 0x%x\n",
            tp->prgname, __func__, event);
        /* msec to usec */
        timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;
        /*2* WaitForEvent() - single eventNotifier */
        mprintf("[%20s] %s: Timeout : %ld\n",
                tp->prgname, __func__, timeout);
        err = tp->eventLoopServiceP->WaitForEvent(ep->eventNotifier,
            timeout);
        if (err != NvSciError_Success) {
            mprintf("[%20s] %s: WaitForEvent err: %d\n",
                tp->prgname, __func__, err);
            return err;
        }

        ep->evt_cnt++;
    }
    return NvSciError_Success;
}

static void release_resources(struct endpoint *ep)
{
    test_params *tp = &s_params;
    NvSciError err;

    if (ep->buf) {
        free(ep->buf);
        ep->buf = NULL;
    }

    /*2* NvSciEventNotifier::Delete() */
    ep->eventNotifier->Delete(ep->eventNotifier);
    dprintf("[%20s] closing NvSciIpc endpoint\n", tp->prgname);
    err = NvSciIpcCloseEndpointSafe(ep->h, true);
    if (err != NvSciError_Success) {
        printf("[%20s:%d] %s: NvSciIpcCloseEndpointSafe: fail (%d)\n",
            tp->prgname, getpid(), __func__, err);
    }

    NvSciIpcDeinit();

    /*2* NvSciEventService::Delete() */
    tp->eventLoopServiceP->EventService.Delete(&tp->eventLoopServiceP->EventService);
    if (s_Stop) {
        exit(1);
    }
}

static void sig_handler(int sig_num)
{
    test_params *tp = &s_params;

    s_Stop = 1;

    if (tp->initFlag) {
        release_resources(&tp->ep);
        tp->initFlag = false;
    }
}

static void setup_termination_handlers(void)
{
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGHUP, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGABRT, sig_handler);
}

static NvSciError init_resources(struct endpoint *ep)
{
    test_params *tp = &s_params;
    NvSciError err;

    /* NvSciEventLoopServiceCreateSafe() */
    dprintf("[%20s] %s: NvSciEventLoopServiceCreateSafe\n",
        tp->prgname, __func__);
    err = NvSciEventLoopServiceCreateSafe(1, NULL,
        &tp->eventLoopServiceP);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciEventLoopServiceCreateSafe: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    /* NvSciIpcInit() */
    err = NvSciIpcInit();
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcInit: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    mprintf("[%20s] opening NvSciIpc endpoint: %s\n", tp->prgname, ep->chname);
    /* NvSciIpcOpenEndpointWithEventService() */
    dprintf("[%20s] %s: NvSciIpcOpenEndpointWithEventService\n",
            tp->prgname, __func__);
    err = NvSciIpcOpenEndpointWithEventService(ep->chname, &ep->h,
        &tp->eventLoopServiceP->EventService);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcOpenEndpoint(%s): fail (%d)\n",
            tp->prgname, __func__, ep->chname, err);
        goto fail;
    }
    dprintf("[%20s] endpoint handle: 0x%lx\n", tp->prgname, ep->h);

    /* NvSciIpcGetEventNotifier() */
    dprintf("[%20s] %s: NvSciIpcGetEventNotifier\n",
            tp->prgname, __func__);
    err = NvSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcGetEventNotifier: fail (%d)\n",
                tp->prgname, __func__, err);
        goto fail;
    }

    /* NvSciIpcGetEndpointInfo() */
    err = NvSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcGetEndpointInfo: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }
    mprintf("[%20s] endpoint_info: nframes = %d, frame_size = %d\n",
        tp->prgname, ep->info.nframes, ep->info.frame_size);

    /* allocate frame buffer */
    ep->buf = calloc(1, ep->info.frame_size);
    if (ep->buf == NULL) {
        printf("[%20s] %s: Failed to allocate buffer of size %u\n",
            tp->prgname, __func__, ep->info.frame_size);
        goto fail;
    }

    err = NvSciIpcResetEndpointSafe(ep->h);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcResetEndpointSafe: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    err = NvSciError_Success;
    tp->initFlag = true;

fail:
    return err;
}

int32_t write_test(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t count = 0;
    uint32_t *ptr;
    uint32_t bytes;
    NvSciError err;
    uint32_t wr_count = 0;
    uint32_t wr_err_cnt = 0;

    mprintf("[%20s] Ping Test mode (loop: %ld)\n",
        tp->prgname, tp->iterations);

    ptr = (uint32_t *)ep->buf;

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, NV_SCI_IPC_EVENT_WRITE);
        if(err != NvSciError_Success) {
            mprintf("%s: error in waiting WR event: %d\n", __func__, err);
            wr_err_cnt++;
            break;
        }

        ptr[0] = count;
        ptr[1] = 0;
        dprintf("[%20s]%s: WR#%02d: %d %d\n", tp->prgname,
                ep->chname, count, ptr[0], ptr[1]);
        err = NvSciIpcWriteSafe(ep->h, ep->buf, ep->info.frame_size, &bytes);
        if(err != NvSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
            mprintf("%s: error in writing: %d\n", __func__, err);
            wr_err_cnt++;
            break;
        }
        wr_count++;
        count++;
    }

    mprintf("[%20s] %s write count: %d, wr err count: %d\n",
        tp->prgname, ep->chname, wr_count, wr_err_cnt);

    mprintf("[%20s] %s event count: %u\n", tp->prgname, ep->chname,
        ep->evt_cnt);

    if (wr_err_cnt > 0) {
        return EIO;
    }
    else {
        return 0;
    }
}

void *write_test_main(void *arg)
{
    test_params *tp = &s_params;
    int *result = (int *)arg;
    int retval = 0;
    NvSciError err;

    err = init_resources(&tp->ep);
    if (err != NvSciError_Success) {
        retval = -1;
        goto fail;
    }
    retval = write_test(&tp->ep);
fail:
    if (tp->initFlag) {
        release_resources(&tp->ep);
        tp->initFlag = false;
    }
    *result = retval;

    return NULL;
}

int main(int argc, char *argv[])
{
    test_params *tp = &s_params;
    struct endpoint *ep = &tp->ep;
    int opt;
    static int retval = -1;
    int exitcode;
    tp->epCnt = 0;
    tp->timeout = NV_SCI_EVENT_INFINITE_WAIT;
    tp->iterations = DEFAULT_COUNT;
    tp->prgname = basename(argv[0]);

    dprintf("[%20s] enter NvSciIpc test\n", argv[0]);

    while ((opt = getopt(argc, argv,
                         "c:hl:vT:"
                         )) != -1)
    {
        switch (opt)
        {
            case 'c':
                {
                    unsigned int slen = strlen(optarg);
                    if (slen > NVSCIIPC_MAX_ENDPOINT_NAME - 1) {
                        slen = NVSCIIPC_MAX_ENDPOINT_NAME - 1;
                    }
                    memcpy(ep->chname, optarg, slen);
                    ep->chname[slen] = '\0';
                    tp->epCnt = 1;
                    break;
                }
            case 'h': /* HELP */
                print_usage(argv);
                retval = 0;
                goto done;
            case 'l':
                tp->iterations = strtoul(optarg, NULL, 0);
                break;
            case 'v':
                s_verboseFlag = true;
                break;
            case 'T':   /* use NvSciEventService */
                {
                    uint32_t val;
                    errno = EOK;
                    val = strtoul(optarg, NULL, 0);
                    // decimal including 0
                    if ((optarg[0] != 'U') && (val >= 0) && (errno == 0)) {
                        tp->timeout = val;
                    }
                }
                break;
            case ':':
                fprintf(stderr, "Option `-%c` requires an argument.\n", optopt);
                goto done;
            case '?':
                if (isprint(optopt))
                {
                    fprintf(stderr, "Unknown option `-%c`.\n", optopt);
                }
                else
                {
                    fprintf(stderr, "Unknown option ``\\x%x`.\n", optopt);
                }
                goto done;
            default:
                print_usage(argv);
                goto done;
        }
    }

    /* check ABI compatibility */
    {
        bool compatible;
        uint32_t vererr = 0U;

        retval = NvSciIpcCheckVersionCompatibility(NvSciIpcMajorVersion,
                NvSciIpcMinorVersion, &compatible);
        mprintf("libnvsciipc:   %d.%d\n",
            NvSciIpcMajorVersion, NvSciIpcMinorVersion);
        if ((retval != NvSciError_Success) || (compatible != true)) {
            mprintf("nvsciipc library version is NOT compatible\n");
            vererr++;
        }
        retval = NvSciEventCheckVersionCompatibility(NvSciEventMajorVersion,
                NvSciEventMinorVersion, &compatible);
        mprintf("libnvscievent: %d.%d\n",
            NvSciEventMajorVersion, NvSciEventMinorVersion);
        if ((retval != NvSciError_Success) || (compatible != true)) {
            mprintf("nvscievent library version is NOT compatible\n");
            vererr++;
        }

        if (vererr > 0) {
            retval = -1;
            goto done;
        }
    }

    /* handle parameter errors */
    if (tp->epCnt == 0) {
        fprintf(stderr, "need one endpoint at least\n");
        print_usage(argv);
        goto done;
    }

    if (!ep->chname[0]) {
        fprintf(stderr, "need to give NvSciIpc endpoint name as input\n");
        print_usage(argv);
        goto done;
    }

    /* Display configuration */
    dprintf("[%20s] endpoint name is *%s*\n", tp->prgname, ep->chname);
    mprintf("[%20s] iteration for Native Event : %ld\n", tp->prgname,
        tp->iterations);
    mprintf("[%20s] Use NvSciEventService (timeout:%ldmsec)\n",
        tp->prgname, tp->timeout);

    /* setup sig handler */
    setup_termination_handlers();
    write_test_main(&retval);

done:
    if (retval != 0) {
        exitcode = EXIT_FAILURE;
        mprintf("[%20s] : test FAILED\n", tp->prgname);
    }
    else {
        exitcode = EXIT_SUCCESS;
        mprintf("[%20s] : test PASSED\n", tp->prgname);
    }

    return exitcode;
}

