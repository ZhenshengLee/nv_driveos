# Copyright (c) 2018-2022 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
from collections import OrderedDict

begining_string =  '''/* This file is generated with gen_json_header.py. Do not edit manually */

extern "C" {
#include "json_data.h"
}

const char *CNvMQuery_GetJsonData() {

    return R"nvsipl_query_esc(
'''

ending_string = '''
)nvsipl_query_esc";
}

'''

def add_json_string_variable(files):
    recognized_types = [
        "eeproms",
        "imageSensors",
        "serializers",
        "deserializers",
        "cameraModules",
        "platformConfigs"
    ]

    json_list = { x:[] for x in recognized_types}

    for json_file_name in files:
        json_contents = json.load(open(json_file_name))
        for json_type, values in json_contents.items():
            if json_type not in recognized_types:
                raise RuntimeError('Json key "{0}" in file "{1}" is not allowed!  Allowed keys are: {2}'.format(
                    json_type, json_file_name, ", ".join(recognized_types)))
            if isinstance(values, list):
                json_list[json_type] += values
            else:
                json_list[json_type].append(values)

    all_json_data = [ {json_type : json_list[json_type]} for json_type in recognized_types if json_list[json_type] ]
    return_string = json.dumps(all_json_data, indent = 4, separators=(',', ': '))
    return return_string



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputFiles', nargs = '*')
    parser.add_argument('-outputFile')
    args = vars(parser.parse_args())

    output_file_name = args['outputFile']
    input_file_names = args['inputFiles']

# Save to output file
    out_file = open(output_file_name, 'w')
    out_file.write(begining_string)
    out_file.write(add_json_string_variable(input_file_names))
    out_file.write(ending_string)
    out_file.close()


if __name__ == '__main__':
    main()
