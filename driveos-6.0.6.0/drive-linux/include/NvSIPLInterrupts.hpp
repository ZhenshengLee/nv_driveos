/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVSIPLINTERRUPTS_HPP
#define NVSIPLINTERRUPTS_HPP

/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Interrupts </b>
 *
 */

namespace nvsipl
{

/**
 * @brief Interrupt Status Codes
 */
enum class InterruptCode : uint32_t {
    /**
     * Device Block-level event, indicates general deserializer failure.
     */
    INTR_STATUS_DES_FAILURE = 0U,
    /**
     * Device Block-level event, indicates deserializer response timeout.
     */
    INTR_STATUS_DES_TIMEOUT = 1U,
    /**
     * Device Block or Camera Link-level event, indicates deserializer
     * failure.
     */
    INTR_STATUS_DES_ERRB_ERR = 10U,
    /**
     * Camera Link-level event, indicates deserializer-serializer link
     * failure.
     */
    INTR_STATUS_DES_LOCK_ERR = 11U,

    /**
     * Device Block-level event, indicates general power load switch
     * failure.
     */
    INTR_STATUS_PWR_FAILURE = 100U,
    /**
     * Device Block-level event, indicates power load switch response
     * timeout.
     */
    INTR_STATUS_PWR_TIMEOUT = 101U,

    /**
     * Camera Link-level event, indicates general Camera Module serializer
     * failure.
     */
    INTR_STATUS_SER_FAILURE = 1000U,
    /**
     * Camera Link-level event, indicates Camera Module serializer response
     * timeout.
     */
    INTR_STATUS_SER_TIMEOUT = 1001U,

    /**
     * Camera Link-level event, indicates general Camera Module sensor
     * failure.
     */
    INTR_STATUS_SEN_FAILURE = 1100U,
    /**
     * Camera Link-level event, indicates Camera Module sensor reponse
     * timeout.
     */
    INTR_STATUS_SEN_TIMEOUT = 1101U,

    /**
     * Camera Link-level event, indicates general Camera Module PMIC
     * failure.
     */
    INTR_STATUS_PMIC_FAILURE = 1200U,
    /**
     * Camera Link-level event, indicates Camera Module PMIC response
     * timeout.
     */
    INTR_STATUS_PMIC_TIMEOUT = 1201U,

    /**
     * Camera Link-level event, indicates general Camera Module EEPROM
     * failure.
     */
    INTR_STATUS_EEPROM_FAILURE = 1300U,
    /**
     * Camera Link-level event, indicates Camera Module EEPROM response
     * timeout.
     */
    INTR_STATUS_EEPROM_TIMEOUT = 1301U,

    /**
     * Device Block or Camera Link-level event, indicates general failure.
     */
    INTR_STATUS_FAILURE = 10000U,
    /**
     * Device Block or Camera Link-level event, indicates general response
     * timeout.
     */
    INTR_STATUS_TIMEOUT = 10001U,
};

}  // namespace nvsipl


#endif // NVSIPLINTERRUPTS_HPP
