/*
 * Copyright (c) 2015-2017, NVIDIA Corporation.  All rights reserved.
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
#include "nvavtp.h"
#include <signal.h>

#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <unistd.h>
#include <errno.h>
#include <net/if.h>
#include<sys/types.h>
#include<sys/ioctl.h>
#include <sys/time.h>

#include "fcntl.h"
#include "raw_socket.h"

#include <sched.h>
#include <time.h>

#include <alsa/asoundlib.h>

#define ONE_MS 1000000LL
//#define DEBUG_CRF

#define LEVEL_BASIC                    (1<<0)
#define LEVEL_INACTIVE                 (1<<1)
#define LEVEL_ID                       (1<<2)
#define DEFAULT_TIME_INTERVAL          (150)
#define DEFAULT_ASRC_STREAM            (2)
#define DEFAULT_ARAD_LANE              (2)
#define MAX_ASRC_STREAM                (6)
#define MAX_ARAD_LANE                  (6)
#define DEFAULT_ETH_INTERFACE          "eth0.3"
#define HWDEVICE "/dev/eqos_ape_hw"
#define CALC_PARAMS_BASE               (1000000000000)
#define PERIOD_NS                      (1000000000)

static char card[] = "default";

volatile sig_atomic_t loop_status;
void breakLoop(int sig);

typedef struct
{
    U64 max;
    U64 min;
    U64 average;
    U64 count;
} NvDiagnostics;

typedef struct
{
    NvAvtpContextHandle pHandle;
    NvAvtpCRFParams  *pAvtpCRFParameters;
    char interface[64];
    U32 timeout;
    NvDiagnostics *pFrequencyData;
    U8 firstPacket;
    U8 asrcStream;
    U8 aradLane;
    int fd;
    bool set_asrc_ratio;
    bool get_gptp_ape_drift;
    bool get_arad_ratio;
} NvmAvbSinkData;

struct NvRatio{
    U32 integer;
    U32 fract;
};

struct eqos_ape_sync_cmd {
    U64 drift_num;
    U64 drift_den;
};

struct rate_to_time_period {
    U32 rate;
    U32 n_int;
    U32 n_fract;
    U32 n_modulo;
};

enum {
    EQOS_APE_AMISC_INIT = _IO(0xF9, 0x01),
    EQOS_APE_AMISC_DEINIT = _IO(0xF9, 0x02),
    EQOS_APE_AMISC_FREQ_SYNC = _IO(0xF9, 0x03),
    EQOS_APE_AMISC_PHASE_SYNC = _IO(0xF9, 0x04),
    EQOS_APE_TEST_FREQ_ADJ = _IOW(0xF9, 0x05, struct eqos_ape_sync_cmd),
    EQOS_APE_AMISC_GET_RATE = _IO(0xF9, 0x06)
};


void updateStats(NvDiagnostics *pData, U64 newData);
void dumpStats(NvmAvbSinkData *args);

static struct rate_to_time_period ape_amisc_calc_rate_info(struct rate_to_time_period rate_info);
static void initSoundProcessing(U8 asrcStream);
static void processCRFPacket(NvmAvbSinkData *args, U8* packet);
static void getInitialFrequency(NvmAvbSinkData *args, U8* packet);

U64 getTimestamp(NvmAvbSinkData *args, U8 *packet);
void initialize_AVTP(void *pParam);
void clean_exit(void *pParam);
static U64 time_of_day(void);

static void setASRCRatio(double frequencyRatio, U8 asrcStream);
static void set_asrcratio_snd_ctl(int argc, char *argv[]);

static void getARADRatio(U8 aradLane, struct NvRatio *arad_ratio);
static U64 get_aradratio_snd_ctl(int argc, char *argv[]);

int snd_ctl_ascii_elem_id_parse (snd_ctl_elem_id_t *dst, const char *str);
int snd_ctl_ascii_value_parse (snd_ctl_t *handle, snd_ctl_elem_value_t *dst, snd_ctl_elem_info_t *info, const char *value);

static U64 time_of_day(void)
{
    U64 temp_time = 0;
    struct timespec timeStruct;
    clock_gettime(CLOCK_MONOTONIC, &timeStruct);

    temp_time = ((timeStruct.tv_sec)*1000000000LL) + timeStruct.tv_nsec;
    return temp_time;
}

struct rate_to_time_period ape_amisc_calc_rate_info(struct rate_to_time_period rate_info)
{
    double period;
    S64 target, frac_target;
    S64 base, n;
    S64 est_target;
    S64 diff;
    S64 n_least, i_least;
    S64 i, inc;
    S32 N_int;

    n_least = 0;
    i_least = 0;
    diff = 0;

    base = CALC_PARAMS_BASE;
    inc = base / 2;

    period = (double) PERIOD_NS / (double)(rate_info.rate);

    N_int = (S32)period;
    target = (S64)((period - (double)N_int) * base);
    frac_target = (S64)((((period - (double)N_int) * base) - (double)target) * 100);

    for (i=1; i<65536; i++)
    {
        n = target * i  + inc;
        n = n / base;
        est_target = n * base * 100 / i;
        if(llabs(target*100 - est_target) == frac_target)
        {
            printf ("Found exact N (%d) n(%lld) i (%lld)\n", N_int, n, i);
            n_least = n;
            i_least = i;
            break;
        }
        else
        {
            if(i == 1)
            {
                diff = llabs(target*100 - est_target);
                n_least = n;
                i_least = i;
            }
            else
            {
                if (diff > llabs(target*100 - est_target))
                {
                    diff = llabs(target*100 - est_target);
                    n_least = n;
                    i_least = i;
                }
            }
        }
    }

    rate_info.n_int = (U32)N_int;
    rate_info.n_fract = (U32)n_least;
    rate_info.n_modulo = (U32)i_least;

    printf("rate: %d int: %d frac: %d modulo: %d\n", rate_info.rate, \
            rate_info.n_int, rate_info.n_fract, rate_info.n_modulo);

    return rate_info;
}

//! sets input parameters for AVTP
//! \param pParam - Pointer to NvAvbSinkData
void initialize_AVTP(void* pParam)
{
    NvAvtpInputParams *pAvtpInpPrms;
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *)pParam;

    pAvtpInpPrms = malloc(sizeof(NvAvtpInputParams));
    if (pAvtpInpPrms == NULL)
    {
        exit(1);
    }

    memset(pAvtpInpPrms, 0x0, sizeof(NvAvtpInputParams));
    strcpy(pAvtpInpPrms->interface,priv_data->interface);
    pAvtpInpPrms->bAvtpDepacketization = eNvAvtpTrue;
    NvAvtpInit (pAvtpInpPrms, &priv_data->pHandle);

    free(pAvtpInpPrms);
    priv_data->pAvtpCRFParameters = (NvAvtpCRFParams*) malloc(sizeof(NvAvtpCRFParams));
    priv_data->pFrequencyData = (NvDiagnostics*) malloc(sizeof(NvDiagnostics));

    memset(priv_data->pAvtpCRFParameters, 0, sizeof(NvAvtpCRFParams));
    memset(priv_data->pFrequencyData, 0, sizeof(NvDiagnostics));
}

//! clean up memory before exiting
//! \param pParam - Pointer to NvAvbSinkData
void clean_exit(void* pParam)
{
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *) pParam;
    NvAvtpDeinit(priv_data->pHandle);
    free(priv_data->pAvtpCRFParameters);
    free(priv_data->pFrequencyData);
    free(priv_data);
}

void dumpStats(NvmAvbSinkData *args)
{
    NvDiagnostics *pData = args->pFrequencyData;
    printf("Min frequency: %llu\n", pData->min);
    printf("Max frequency: %llu\n", pData->max);
    printf("Average frequency: %f\n", (double) pData->average / pData->count);
}

void updateStats(NvDiagnostics *pData, U64 newData)
{
    if(pData->count == 0)
    {
        pData->max = newData;
        pData->min = newData;
    }
    else
    {
        if (newData > pData-> max)
        {
            pData->max = newData;
        }
        if (newData < pData-> min)
        {
            pData->min = newData;
        }
    }
    pData->average += newData;
    pData->count++;
}

U64 getTimestamp(NvmAvbSinkData *args, U8 *packet)
{
    NvmAvbSinkData *priv_data = (NvmAvbSinkData *) args;
    NvAvtpContextHandle Handle = priv_data->pHandle;
    NvAvtpCRFParams *pAvtpCRFParameters = priv_data->pAvtpCRFParameters;
    ENvAvtpSubHeaderType eAvtpSubHeaderType;
    U8 *payload;

    NvAvtpParseAvtpPacket(Handle, (U8 *)packet, &eAvtpSubHeaderType);

    if (eAvtpSubHeaderType != eNvCRF)
    {
        return 0;
    }

    NvAvtpGetCRFParams(Handle, (U8 *)packet, pAvtpCRFParameters);
    payload = (U8 *) malloc(pAvtpCRFParameters->dataLength);
    NvAvtpExtractDataPayload(Handle, (U8 *)packet, payload);

    U64 timeStamp = *(U64*) payload;
    free(payload);
    return timeStamp;
}

void processCRFPacket(NvmAvbSinkData *args, U8* packet)
{
    static U32 count = 0;
    static U64 startTime = 0;
    static U64 initialTimestamp = 0;
    U64 endTime = 0;
    U64 finalTimestamp = 1;
    struct eqos_ape_sync_cmd cmd;

    count++;
    if (count == 1)
    {
        startTime = time_of_day();
        initialTimestamp = getTimestamp(args, packet);
        endTime = startTime;
        if (ioctl(args->fd, EQOS_APE_AMISC_FREQ_SYNC, &cmd) < 0)
        {
            printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_FREQ_SYNC);
            return;
        }

    }
    else
    {
        endTime = time_of_day();
        finalTimestamp = getTimestamp(args, packet);
    }

    if (initialTimestamp == 0 || finalTimestamp == 0)
    {
        count--;
        return;
    }

    U64 timePeriod = args->timeout * ONE_MS;
    if (endTime - startTime > timePeriod)
    {
        struct NvRatio arad_ratio;
        double ratio1 = 1.0, ratio2 = 1.0, ratio3 = 1.0;
        U64 timeDifference = finalTimestamp - initialTimestamp;

        NvAvtpCRFParams *pAvtpCRFParams = args->pAvtpCRFParameters;
        U32 samples = pAvtpCRFParams->timestampInterval;
        U32 frequency = (1000000000LL * --count * samples) / timeDifference;

        if(args->set_asrc_ratio)
        {
            //Calculate the CRF ratio and set the ratio on ASRC
            ratio1 = (double) frequency / 48000;
            if(args->get_gptp_ape_drift)
            {
                //Calculate the drift between the gPTP and APE clock
                //for both i2s master and slave case
                if (ioctl(args->fd, EQOS_APE_AMISC_FREQ_SYNC, &cmd) < 0)
                {
                    printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_FREQ_SYNC);
                    return;
                }
                ratio2 = ((double)(cmd.drift_num))/((double)(cmd.drift_den));

                if (args->get_arad_ratio)
                {
                    //Get ARAD ratio (between I2S1 (APE clock)
                    //and I2S3(SYNC clock)
                    getARADRatio(args->aradLane, &arad_ratio);
                    ratio3 = arad_ratio.integer + ((double)arad_ratio.fract/(0xFFFFFFFF));
                }
            }

            double frequencyRatio = ratio1 * ratio2 * ratio3;
            setASRCRatio(frequencyRatio, args->asrcStream);
            printf("ratio1 %1.9f ratio2 %1.9f ratio3 %1.9f ratio %1.9f\n", ratio1, ratio2, ratio3, frequencyRatio);
        }
        count = 0;
#ifdef DEBUG_CRF
        printf("Frequency %d\n", frequency);
#endif
        updateStats(args->pFrequencyData, frequency);
    }

}

void getInitialFrequency(NvmAvbSinkData *args, U8* packet)
{
    NvAvtpContextHandle Handle = args->pHandle;
    ENvAvtpSubHeaderType eAvtpSubHeaderType;

    NvAvtpParseAvtpPacket(Handle, (U8 *)packet, &eAvtpSubHeaderType);

    if (eAvtpSubHeaderType != eNvCRF)
    {
        return;
    }

    NvAvtpCRFParams *pAvtpCRFParams = args->pAvtpCRFParameters;
    NvAvtpGetCRFParams(Handle, (U8 *)packet, pAvtpCRFParams);
    printf("Frequency: %d\n", pAvtpCRFParams->frequency);
    args->firstPacket++;
}

static void setASRCRatio(double frequencyRatio, U8 asrcStream)
{
    unsigned int intPart = (unsigned int) frequencyRatio;
    double fractionalPart = frequencyRatio - (double) intPart;
    fractionalPart *= 0xFFFFFFFF;
    unsigned int fracPart = (unsigned int) fractionalPart;

    char* paramInt[2];
    char* paramFrac[2];
    char stringASRCInt[32];
    char stringASRCFrac[32];
    char stringA[16];
    char stringB[16];
    paramInt[0] = (char*)&stringASRCInt;
    paramFrac[0] = (char*)&stringASRCFrac;
    paramInt[1] = (char*)&stringA;
    paramFrac[1] = (char*)&stringB;


    sprintf(stringASRCInt, "name='ASRC1 Ratio%u Int'", asrcStream);
    sprintf(stringASRCFrac, "name='ASRC1 Ratio%u Frac'", asrcStream);


    snprintf(stringA, 16, "%u", intPart);
    snprintf(stringB, 16, "%u", fracPart);

#ifdef DEBUG_CRF
    printf("Int %u\n", intPart);
    printf("Frac %u\n", fracPart);
#endif

    set_asrcratio_snd_ctl(2, paramInt);
    set_asrcratio_snd_ctl(2, paramFrac);
}

static void getARADRatio(U8 aradLane, struct NvRatio *arad_ratio)
{
    U64 ratio;
    char* param[1];
    char stringARAD[32];
    param[0] = (char*)&stringARAD;

    sprintf(stringARAD, "name='Lane2 Ratio'");

    ratio = get_aradratio_snd_ctl(2, param);
    arad_ratio->integer = (U32)((ratio & 0xffffffff00000000ULL) >> 32);
    arad_ratio->fract = (U32)(ratio & 0x00000000ffffffff);
#ifdef DEBUG_CRF
    printf("ARAD Int %u\n", arad_ratio->integer);
    printf("ARAD Frac %u\n", arad_ratio->fract);
#endif
}

static U64 get_aradratio_snd_ctl(int argc, char *argv[])
{
    /*Declarations*/
    snd_ctl_t *handle = NULL;
    snd_ctl_elem_info_t *info;
    snd_ctl_elem_id_t *id;
    snd_ctl_elem_value_t *control;
    /*Initializations*/
    snd_ctl_elem_info_alloca(&info);
    snd_ctl_elem_id_alloca(&id);
    snd_ctl_elem_value_alloca(&control);
    /*Parse argument string*/
    if (snd_ctl_ascii_elem_id_parse(id, argv[0]))
    {
        fprintf(stderr, "Invalid string");
    }
    /*Opening card*/
    if (snd_ctl_open(&handle, card, 0) < 0)
    {
        fprintf(stderr, "Unable to open card");
    }
    /*Update card info*/
    snd_ctl_elem_info_set_id(info, id);
    if (snd_ctl_elem_info(handle, info) < 0)
    {
        goto ASRC_Error;
    }
    snd_ctl_elem_info_get_id(info, id);

    snd_ctl_elem_value_set_id(control, id);
    if (snd_ctl_elem_read(handle, control) < 0)
    {
        goto ASRC_Error;
    }

    snd_ctl_close(handle);
        return snd_ctl_elem_value_get_integer64(control, 0);

ASRC_Error:
    if (handle != NULL)
    {
        snd_ctl_close(handle);
    }
    fprintf(stderr, "Error in handling sound card\n");
    return -1;
}

//! signal handler
//! function to stop reading socket
void breakLoop(int sig)
{
    fprintf(stderr, "\n");
    loop_status = 0;
}

static void initSoundProcessing(U8 asrcStream)
{
    char* params[2];
    char stringASRC[32];
    params[0]=(char *)&stringASRC;
    sprintf(stringASRC,"name='ASRC1 Ratio%u SRC'", asrcStream);
    params[1] = "SW";
    set_asrcratio_snd_ctl(2, params);
}

static void set_asrcratio_snd_ctl(int argc, char *argv[])
{
    /*Declarations*/
    snd_ctl_t *handle = NULL;
    snd_ctl_elem_info_t *info;
    snd_ctl_elem_id_t *id;
    snd_ctl_elem_value_t *control;
    /*Initializations*/
    snd_ctl_elem_info_alloca(&info);
    snd_ctl_elem_id_alloca(&id);
    snd_ctl_elem_value_alloca(&control);
    /*Parse argument string*/
    if (snd_ctl_ascii_elem_id_parse(id, argv[0]))
    {
        fprintf(stderr, "Invalid string");
    }
    /*Opening card*/
    if (snd_ctl_open(&handle, card, 0) < 0)
    {
        fprintf(stderr, "Unable to open card");
    }
    /*Update card info*/
    snd_ctl_elem_info_set_id(info, id);
    if (snd_ctl_elem_info(handle, info) < 0)
    {
        goto ASRC_Error;
    }
    snd_ctl_elem_info_get_id(info, id);

    snd_ctl_elem_value_set_id(control, id);
    if (snd_ctl_elem_read(handle, control) < 0)
    {
        goto ASRC_Error;
    }

    if (snd_ctl_ascii_value_parse(handle, control, info, argv[1]) < 0)
    {
        goto ASRC_Error;
    }
    if (snd_ctl_elem_write(handle, control) < 0)
    {
        goto ASRC_Error;
    }

    snd_ctl_close(handle);
    return;

ASRC_Error:
    if (handle != NULL)
    {
        snd_ctl_close(handle);
    }
    fprintf(stderr, "Error in handling sound card\n");
}

int main(int argc, char *argv[])
{

    struct sched_param sParam;
    struct rate_to_time_period rate_info;
    NvmAvbSinkData *priv_data = NULL;
    S32 i;
    loop_status         = 1;

    /*setting thread-priority as real-time*/
    memset(&sParam, 0, sizeof(struct sched_param));
    sParam.sched_priority = sched_get_priority_max(SCHED_RR);
    int retval = sched_setscheduler(0, SCHED_RR, &sParam);
    if (retval != 0)
    {
        fprintf(stderr, "%s", "Scheduling error.\n");
        exit(1);
    }

    /*setting signal handler*/
    signal(SIGINT, breakLoop);

     /*initialization*/
    priv_data = (NvmAvbSinkData *)malloc(sizeof(NvmAvbSinkData));
    memset(priv_data, 0x00, sizeof(NvmAvbSinkData));

    priv_data->timeout            = DEFAULT_TIME_INTERVAL;
    priv_data->asrcStream         = DEFAULT_ASRC_STREAM;
    priv_data->aradLane           = DEFAULT_ARAD_LANE;
    priv_data->set_asrc_ratio     = false;
    priv_data->get_gptp_ape_drift = false;
    priv_data->get_arad_ratio     = false;
    strcpy(priv_data->interface, DEFAULT_ETH_INTERFACE);

    /*argument parsing*/
    if (argc > 1)
    {
        for(i=0;i<argc;i++)
        {
            if (!strcmp (argv[i], "-i"))
            {
                strcpy (priv_data->interface, argv[i + 1]);
            }
            if (!strcmp (argv[i], "-t"))
            {
                priv_data->timeout = atoi(argv[i + 1]);
            }
            if (!strcmp (argv[i], "-s"))
            {
                if((atoi(argv[i+1]) <= MAX_ASRC_STREAM) && (atoi(argv[i+1]) >= 1))
                    priv_data->asrcStream = atoi(argv[i+1]);
                else
                {
                    printf("Incorrect ASRC Stream %d\n", atoi(argv[i+1]));
                    printf("Allowed values 1 to 6\n");
                }
            }
            if (!strcmp (argv[i], "-a"))
            {
                if((atoi(argv[i+1]) <= MAX_ASRC_STREAM) && (atoi(argv[i+1]) >= 1))
                    priv_data->aradLane = atoi(argv[i+1]);
                else
                {
                    printf("Incorrect ARAD Lane %d\n", atoi(argv[i+1]));
                    printf("Allowed values 1 to 6\n");
                }
            }
            if (!strcmp (argv[i], "-crf_asrc"))
            {
                priv_data->set_asrc_ratio     = true;
            }
            if (!strcmp (argv[i], "-crf_i2s_master"))
            {
                priv_data->set_asrc_ratio     = true;
                priv_data->get_gptp_ape_drift = true;
            }
            if (!strcmp (argv[i], "-crf_i2s_slave"))
            {
                priv_data->set_asrc_ratio     = true;
                priv_data->get_gptp_ape_drift = true;
                priv_data->get_arad_ratio     = true;
            }
            if (!strcmp (argv[i], "-h"))
            {
                printf("crf_listener is a utility for capturing 1772 CRF stream\n");
                printf("and parsing the CRF packets to do media clock recovery\n\n");
                printf("Command line options:\n\n");
                printf("-i                 Select eth interface to listen to crf stream\n");
                printf("-t                 Select the time interval in milli-seconds to calculate the crf frequency\n");
                printf("-s                 Select the ASRC stream\n");
                printf("-a                 Select the ARAD lane\n");
                printf("-crf_asrc          Apply ASRC ratio without drift compensation\n");
                printf("-crf_i2s_master    Use drift compensation with i2s in master mode\n");
                printf("-crf_i2s_slave     Use drift compensation with i2s in slave mode\n");
                return 0;
            }
        }
    }

    initialize_AVTP(priv_data);

    /*open eqos_ape driver*/
    priv_data->fd = open(HWDEVICE, O_RDWR);
    if (!priv_data->fd)
    {
        printf("failed to open device\n");
        return 1;
    }
    sleep(10);
    printf("utiltiy opened the device\n");

    if (ioctl(priv_data->fd, EQOS_APE_AMISC_GET_RATE, &rate_info) < 0)
    {
        printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_GET_RATE);
        return 1;
    }

    /* Calculate N_int, N_fract and N_modulo */
    rate_info = ape_amisc_calc_rate_info(rate_info);

    if (ioctl(priv_data->fd, EQOS_APE_AMISC_INIT, &rate_info) < 0)
    {
        printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_INIT);
        return 1;
    }

    /*raw socket setup*/
    int fd_read_socket;
    S32 bytesRead = 0;
    U8 buffer[2048];

    fd_read_socket = set_socket((char*)priv_data->interface);

    printf("----------\n");
    while (loop_status)
    {

#ifdef DEBUG_CRF
        U64 start_time = time_of_day();
#endif
        bytesRead = recvfrom(fd_read_socket,buffer,2048,0,NULL,NULL);

        if (bytesRead < 0)
        {
            break;
        }
        /*packet is too short*/
        if (bytesRead < 14)
        {
            perror("recvfrom():");
            printf("Bytes read: %d\n", bytesRead);
            printf("Incomplete packet (errno is %d)\n", errno);
            if (ioctl(priv_data->fd, EQOS_APE_AMISC_DEINIT, NULL) < 0)
            {
                printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_DEINIT);
            }
            close(priv_data->fd);
            close(fd_read_socket);
            exit(1);
        }

        if (NvAvtpIs1722Packet((U8 *)buffer))
        {
            ENvAvtpSubHeaderType eAvtpSubHeaderType;
            NvAvtpParseAvtpPacket(priv_data->pHandle, (U8 *)buffer, &eAvtpSubHeaderType);
            if (eAvtpSubHeaderType == eNvCRF)
            {
                if (priv_data->firstPacket == 0)
                {
                    initSoundProcessing(priv_data->asrcStream);
                    getInitialFrequency(priv_data, (U8*) buffer);
                    processCRFPacket(priv_data, (U8*) buffer);
                }
                else
                {
                    processCRFPacket(priv_data, (U8*) buffer);
                }
            }
        }

#ifdef DEBUG_CRF
        U64 end_time = time_of_day();
        printf("Processing time: %llu\n", end_time - start_time);
        printf("Packet size: %d\n", bytesRead);
#endif
    }

    setASRCRatio(1.0, priv_data->asrcStream);

    if (ioctl(priv_data->fd, EQOS_APE_AMISC_DEINIT, NULL) < 0) {
        printf("eqos_ape_util: command failed %d\n", EQOS_APE_AMISC_DEINIT);
        return 1;
    }

    dumpStats(priv_data);
    close(priv_data->fd);
    close(fd_read_socket);
    clean_exit(priv_data);
    return 0;
}
