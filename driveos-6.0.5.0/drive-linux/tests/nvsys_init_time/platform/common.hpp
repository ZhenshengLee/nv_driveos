/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#define SIPL_PIPELINE_ID (0U)
#define IMAGE_QUEUE_TIMEOUT_US (1000000U)
#define EVENT_QUEUE_TIMEOUT_US (1000000U)
#define INPUT_LINE_READ_SIZE (16U)
// Holds the I2C device bus number used to connect the deserializer with the SoC
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER (0U)
// Holds the I2C device bus number used to connect the deserializer with the SoC(Orin)
#define DESER_TO_SOC_I2C_DEVICE_BUS_NUMBER_CD (3U)
// Holds the deserializer I2C port number connected with the SoC
#define DESER_TO_SOC_I2C_PORT_NUMBER (0U)
// Holds the deserializer Tx port number connected with the SoC
// Set to UINT32_MAX since this is the "don't care" value
#define DESER_TO_SOC_TX_PORT_NUMBER (UINT32_MAX)

#endif // COMMON_HPP
