/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CNVMCAMERAMODULECOMMON_HPP
#define CNVMCAMERAMODULECOMMON_HPP

#include "utils/utils.hpp"

#include "NvSIPLCapStructs.h"

namespace nvsipl
{

/** enumerations of CSI lanes */
enum SensorCsiLaneMapping
{
    SENSOR_CSI_LANE_0, /*!< CSI Lane 0 */
    SENSOR_CSI_LANE_1, /*!< CSI Lane 1 */
    SENSOR_CSI_LANE_2, /*!< CSI Lane 2 */
    SENSOR_CSI_LANE_3  /*!< CSI Lane 3 */
};

/**
 * Class which encapsulates Camera Module information
 */
class CNvMCameraModuleCommon
{

public:

    /** lane polarity enumeration */
    enum LanePolarity : uint8_t
    {
        POLARITY_NORMAL, /*!< Lane polarity is same */
        POLARITY_INVERSE /*!< Lane polarity is inverse. active high becomes active low */
    };

    /**
     * All connection properties
     */
    typedef struct {
        typedef struct {
            typedef struct {
                bool isNeeded;                  /*!< Need release reset */
                uint8_t pinNum;                 /*!< Serializer pin number */
                bool releaseResetLevel;         /*!< Logic level to release reset */
                uint16_t assertResetDuration;   /*!< duration to assert reset in microsecond */
                uint16_t deassertResetWait;     /*!< wait time after deasserting reset in microsecond */
            } SensorReset;

            typedef struct {
                uint8_t pinNum;         /*!< Serializer pin number */
            } FrameSync;

            typedef struct {
                bool isNeeded;          /*!< Need release reset */
                uint8_t pinNum;         /*!< Serializer pin number */
                uint8_t sensorClock;    /*!< sensor input clock value */
            } RefClock;

            typedef struct {
                bool isLaneSwapNeeded;
                uint8_t lane0;          /*!< Input Sensor CSI data lane number on Serializer lane 0 */
                uint8_t lane1;          /*!< Input Sensor CSI data lane number on Serializer lane 1 */
                uint8_t lane2;          /*!< Input Sensor CSI data lane number on Serializer lane 2 */
                uint8_t lane3;          /*!< Input Sensor CSI data lane number on Serializer lane 3 */
                bool isLanePolarityConfigureNeeded;
                uint8_t lane0pol;       /*!< Input Sensor CSI data lane polarity on Serializer lane 0 */
                uint8_t lane1pol;       /*!< Input Sensor CSI data lane polarity on Serializer lane 1 */
                uint8_t lane2pol;       /*!< Input Sensor CSI data lane polarity on Serializer lane 2 */
                uint8_t lane3pol;       /*!< Input Sensor CSI data lane polarity on Serializer lane 3 */
                uint8_t clk1pol;        /*!< Input Sensor CSI clk polarity on Serializer phy1 clk */
                uint8_t clk2pol;        /*!< Input Sensor CSI clk polarity on Serializer phy2 clk */
                bool isTwoLane;         /*!< true if using two lanes */
             } PhyLanes;

            typedef struct {
                uint8_t sourceGpio;     /*!< Serializer GPIO number for the input signal */
                uint8_t destGpio;       /*!< Deserializer GPIO number for the output signal */
            } GpioMap;

            typedef struct {
                bool isNeeded;          /*!< Need write protect */
                uint8_t pinNum;         /*!< Serializer pin number */
                bool writeProtectLevel; /*!< Logic level to write protect */
            } EepromWriteProtect;

            typedef struct {
                bool isSupported = false;   /*!< Need to support PMIC device */
                uint8_t i2cAddress;         /*!< PMIC 7-bit i2c address */
            } PmicProperty;

            typedef struct {
                bool isSupported = false;   /*!< Need to support VCSEL device */
                uint8_t i2cAddress;         /*!< VCSEL 7-bit i2c address */
            } VcselProperty;

            typedef struct {
                bool isNeeded;          /*!< Need serializer ERRB error reporting */
                uint8_t pinNum;         /*!< deserializer pin number to receive
                                           ERR_TX_ID from serialzier */
            } SerializerErrbErrorReport;

            SensorReset sensorReset;

            FrameSync frameSync;

            RefClock refClock;

            PhyLanes phyLanes;

            EepromWriteProtect eepromWriteProtect;

            /**
             * Holds native address for sensor
             */
            std::uint8_t uBrdcstSensorAddrs;

            /**
             * Holds translation address for sensors
             */
            std::uint8_t uSensorAddrs;

            /**
             * Holds VCID
             */
            std::uint8_t uVCID;

            /**
             * Holds ICP input format type
             */
            NvSiplCapInputFormat inputFormat;

            /**
             * Holds embedded data type.
             * false means EMB coming with pixel data and true means EMB coming in CSI packet with different data type
             */
            bool bEmbeddedDataType;

            /**
             * Flag indicating whether trigger mode for the synchronization has been enabled or not in the sensor
             */
            bool bEnableTriggerModeSync;

            /**
             * Flag indicating whether the external synchronization or the internal synchronization enabled
             * true means the internal synchronization in the deserializer enabled
             * false means the external synchronization in the deserializer enabled
             * By default, the external synchronization is enabled
             */
            bool bEnableInternalSync;

            /**
             * Holds frame rate
             */
            float_t fFrameRate;

            /**
             * Holds sensor image width
             */
            uint32_t width;

            /**
             * Holds sensor image height
             */
            uint32_t height;

            /**
             * The number of top embedded data lines in the sensor if embDataType is 0
             * The number of top + bottom embedded data lines in the sensor if embDataType is 1
             */
            uint32_t embeddedTop;

            /**
             * The number of bottom embedded data lines in the sensor if embDataType is 0
             * 0 in the sensor if embDataType is 1
             */
            uint32_t embeddedBot;

/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#if !NV_IS_SAFETY
            /**
             * Holds whether sensor TPG is enabled
             */
            bool bEnableTPG;
/* coverity[misra_cpp_2008_rule_16_2_1_violation] : intentional TID-1391 */
#endif

            /**
             * Holds whether FSYNC should be enabled before or after sensor init
             */
            bool bPostSensorInitFsync;

            /**
             * Holds sensor pclk
             */
            uint32_t pclk;

            /**
             * Holds the serializer regen Vsync high in usec
             */
            uint32_t vsyncHigh;

            /**
             * Holds the serializer regen Vsync low in usec
             */
            uint32_t vsyncLow;

            /**
             * Holds the serializer regen Vsync delay in usec
             */
            uint32_t vsyncDelay;

            /**
             * Holds the serializer regen Vsync trigger edge
             * rising edge and falling edge values are dependent on particular module
             */
            uint32_t vsyncTrig;

            /**
             * Holds the GPIO mapping from the serializer to the deserializer
             */
            std::vector<GpioMap> vGpioMap;

            /**
             * Holds the serializer error report setting info
             */
            SerializerErrbErrorReport serializerErrbErrorReport;

            /**
             * Holds the sensor description
             */
            std::string sensorDescription = "";

            /**
             * Hold PMIC device info
             */
            PmicProperty pmicProperty;

            /**
             * Hold VCSEL device info
             */
            VcselProperty vcselProperty;
        } SensorConnectionProperty;

        SensorConnectionProperty sensorConnectionProperty;

        /**
         * Holds native address for EEPROMs
         */
        std::uint8_t brdcstEepromAddr;

        /**
         * Holds translation address for EEPROM. Same size as brdcstEepromAddr
         */
        std::uint8_t eepromAddr;

        std::uint16_t powerOnDelay;
    } ConnectionProperty;
};
}
#endif //CNVMCAMERAMODULECOMMON_HPP
