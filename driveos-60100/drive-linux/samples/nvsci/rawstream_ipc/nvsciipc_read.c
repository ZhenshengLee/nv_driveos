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
#include <sys/stat.h>
#include <fcntl.h>

#include <math.h>
#include <inttypes.h>

#include <stdint.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>

#include <nvsciipc.h>
#include <nvscievent.h>

#define MAX_ENDPOINT    1000  // limited by NvSciIpc library (500)
#define MAX_EVENTNOTIFIER MAX_ENDPOINT
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
    /* it's not same with endpoint limitation of NvSciIpc library */
    struct endpoint ep;
    uint32_t epCnt; /* endpoint count */
    uint64_t iterations;    /* iterations for endpoint data transaction */
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
int32_t read_test(struct endpoint *ep);
NvSciError wait_event(struct endpoint *ep, int32_t value, uint32_t count);
static void setup_termination_handlers(void);
static NvSciError init_resources(struct endpoint *ep);
static void release_resources(struct endpoint *ep);
void *read_test_main(void *arg);

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
    fprintf(stderr, "\t -T <msec>          : "
            "use NvSciEventService with msec timeout\n"
            "\t wait infinitely if msec is U (-T U)\n");
}

NvSciError wait_event(struct endpoint *ep, int32_t value, uint32_t count)
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
            dprintf("[%20s] %s: evtcnt:%d (event:0x%x)\n",
                tp->prgname, __func__, ep->evt_cnt, event);
            break;
        }

        dprintf("[%20s] %s: event: 0x%x\n",
            tp->prgname, __func__, event);

        /* msec to usec */
        timeout = (tp->timeout >= 0)?(tp->timeout*1000):tp->timeout;
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
    NvSciError ret;

    ep = &tp->ep;

    if (ep->buf) {
        free(ep->buf);
        ep->buf = NULL;
    }

    ep->eventNotifier->Delete(ep->eventNotifier);

    dprintf("[%20s] closing NvSciIpc endpoint\n", tp->prgname);
    ret = NvSciIpcCloseEndpointSafe(ep->h, true);
    if (ret != NvSciError_Success) {
        printf("%s:NvSciIpcCloseEndpointSafe: fail (%d)\n", __func__, ret);
    }

    NvSciIpcDeinit();

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

    dprintf("[%20s] %s: NvSciEventLoopServiceCreateSafe\n",
        tp->prgname, __func__);
    err = NvSciEventLoopServiceCreateSafe(1, NULL,
        &tp->eventLoopServiceP);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciEventLoopServiceCreateSafe: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    err = NvSciIpcInit();
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcInit: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    mprintf("[%20s] opening NvSciIpc endpoint: %s\n", tp->prgname, ep->chname);
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

    err = NvSciIpcGetEventNotifier(ep->h, &ep->eventNotifier);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcGetEventNotifier: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }

    err = NvSciIpcGetEndpointInfo(ep->h, &ep->info);
    if (err != NvSciError_Success) {
        printf("[%20s] %s: NvSciIpcGetEndpointInfo: fail (%d)\n",
            tp->prgname, __func__, err);
        goto fail;
    }
    mprintf("[%20s] endpoint_info: nframes = %d, frame_size = %d\n",
        tp->prgname, ep->info.nframes, ep->info.frame_size);

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

int32_t read_test(struct endpoint *ep)
{
    test_params *tp = &s_params;
    uint32_t count = 0;
    uint32_t *ptr;
    uint32_t bytes;
    uint32_t rd_count = 0;
    uint32_t rd_err_cnt = 0;
    NvSciError err;

    mprintf("[%20s] Ping Test mode (loop: %ld)\n",
        tp->prgname, tp->iterations);

    ptr = (uint32_t *)ep->buf;

    while ((count < tp->iterations) && !s_Stop) {
        err = wait_event(ep, NV_SCI_IPC_EVENT_READ, count);
        if(err != NvSciError_Success) {
            mprintf("%s: error in waiting RD event: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }

        err = NvSciIpcReadSafe(ep->h, ep->buf, ep->info.frame_size, &bytes);
        if(err != NvSciError_Success || (uint32_t)bytes != ep->info.frame_size) {
            mprintf("%s: error in reading: %d\n", __func__, err);
            rd_err_cnt++;
            break;
        }
        dprintf("[%20s]%s: RD#%02d: %d %d\n",tp->prgname,
            ep->chname, count, ptr[0], ptr[1]);
        if (ptr[0] != count) {
            dprintf("%s: mismatch (rx: %d, expected: %d)\n", tp->prgname,
                ptr[0], count);
            rd_err_cnt++;
        }
        rd_count++;
        count++;
    }

    mprintf("[%20s] %s  read count: %d, rd err count: %d\n",
        tp->prgname, ep->chname, rd_count, rd_err_cnt);
    mprintf("[%20s] %s event count: %u\n", tp->prgname, ep->chname,
        ep->evt_cnt);

    if (rd_err_cnt > 0)
        return EIO;
    else
        return 0;
}

void *read_test_main(void *arg)
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

    retval = read_test(&tp->ep);

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
    struct endpoint *ep = &s_params.ep;
    int opt;
    static int retval = -1;
    int exitcode;

    tp->epCnt = 0;
    tp->timeout = NV_SCI_EVENT_INFINITE_WAIT; /* default timeout */

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
            case 'h':
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
                return -1;
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
    read_test_main(&retval);

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

