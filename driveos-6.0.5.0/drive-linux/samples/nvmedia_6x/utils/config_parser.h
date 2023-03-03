/* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _NVMEDIA_TEST_CONFIG_PARSER_H_
#define _NVMEDIA_TEST_CONFIG_PARSER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "nvmedia_core.h"

#define MAX_ITEMS_TO_PARSE 10000

typedef enum _ParamType {
    TYPE_UINT = 0,
    TYPE_UINT_HEX,
    TYPE_INT,
    TYPE_DOUBLE,
    TYPE_FLOAT,
    TYPE_UCHAR,
    TYPE_ULLONG,
    TYPE_USHORT,
    TYPE_CHAR_ARR,
    TYPE_UCHAR_ARR,
    TYPE_SHORT
} ParamType;

typedef enum {
    LIMITS_NONE = 0,
    LIMITS_MIN = 1,
    LIMITS_BOTH = 2
} LimitsType;

typedef enum {
    SECTION_NONE,
    SECTION_CAPTURE,
    SECTION_QP,
    SECTION_RC,
    SECTION_ENCODE_PIC,
    SECTION_ENCODE_PIC_H264,
    SECTION_ENCODE_PIC_H265,
    SECTION_MVC,
    SECTION_PAYLOAD,
    SECTION_2DPROCESSOR
} SectionType;

typedef struct {
    SectionType         secType;
    const char         *name;
    unsigned int        lastSectionIndex;
    size_t              sizeOfStruct;
} SectionMap;

typedef struct {
    const char         *paramName;
    void               *mappedLocation;
    ParamType           type;
    double              defaultValue;
    LimitsType          paramLimits;
    double              minLimit;
    double              maxLimit;
    unsigned int        stringLength;     // string param size
    unsigned int       *stringLengthAddr; // address of string param size
    SectionType         sectionType;
} ConfigParamsMap;

NvMediaStatus   ConfigParser_InitParamsMap(ConfigParamsMap *paramsMap);
NvMediaStatus   ConfigParser_ParseFile(ConfigParamsMap *paramsMap, unsigned int numParams, SectionMap *sectionsMap, char *file);
NvMediaStatus   ConfigParser_ValidateParams(ConfigParamsMap *paramsMap, SectionMap *sectionsMap);
NvMediaStatus   ConfigParser_DisplayParams(ConfigParamsMap *paramsMap, SectionMap *sectionsMap);
NvMediaStatus   ConfigParser_GetSectionIndexByName(SectionMap *sectionsMap, char *sectionName, unsigned int *index);
NvMediaStatus   ConfigParser_GetSectionIndexByType(SectionMap *sectionsMap, SectionType sectionType, unsigned int *index);

#ifdef __cplusplus
}
#endif

#endif /* _NVMEDIA_TEST_CONFIG_PARSER_H_ */
