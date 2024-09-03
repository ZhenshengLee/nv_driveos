/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

/* Defining symbolic names for
AUTOSAR C++ Rule A5-1-1 (required, implementation, partially automated)
Literal values shall not be used apart from type initialization, otherwise
symbolic names shall be used instead.
Rationale
Avoid use of magic numbers and strings in expressions in preference to constant
variables with meaningful names. Literal values are supposed to be used only in type
initialization constructs, e.g. assignments and constructors.
The use of named constants improves both the readability and maintainability of the
code.*/

#ifndef CDD_DEFINES_HPP
#define CDD_DEFINES_HPP

namespace nvsipl {
static constexpr uint32_t GPIOIDX_0_UINT32 {0U};
static constexpr uint32_t GPIOIDX_1_UINT32 {1U};

static constexpr uint8_t MAX_LINK_INDEX {7U};

static constexpr uint8_t BIT_MASK_7F_UINT8 {0x7FU};

static constexpr uint16_t PWR_ON_DELAY_100_MSEC {100U};
static constexpr uint16_t PWR_ON_DELAY_20_MSEC {20U};
static constexpr uint16_t PWR_ON_DELAY_135_MSEC {135U};
static constexpr uint16_t PWR_OFF_DELAY_0_MSEC {0U};
static constexpr uint16_t PWR_OFF_DELAY_900_MSEC {900U};

static constexpr uint16_t ASSERT_RESET_DURATION_1000_USEC {1000U};
static constexpr uint16_t DEASSERT_RESET_WAIT_8880_USEC {8880U};

static constexpr uint8_t GPIO_IND_0 {0U};
static constexpr uint8_t GPIO_IND_1 {1U};
static constexpr uint8_t GPIO_IND_2 {2U};
static constexpr uint8_t GPIO_IND_3 {3U};
static constexpr uint8_t GPIO_IND_4 {4U};
static constexpr uint8_t GPIO_IND_5 {5U};
static constexpr uint8_t GPIO_IND_6 {6U};
static constexpr uint8_t GPIO_IND_7 {7U};
static constexpr uint8_t GPIO_IND_8 {8U};

static constexpr uint8_t VOLT_LEVEL_0 {0U};
static constexpr uint8_t VOLT_LEVEL_1 {1U};

static constexpr uint32_t I2C_PORT_0 {0U};
static constexpr uint32_t I2C_PORT_1 {1U};

static constexpr uint8_t PIN_NUM_0 {0U};
static constexpr uint8_t PIN_NUM_3 {3U};
static constexpr uint8_t PIN_NUM_4 {4U};
static constexpr uint8_t PIN_NUM_7 {7U};
static constexpr uint8_t PIN_NUM_8 {8U};

static constexpr uint8_t I2C_ADDRESS_0x60 {0x60U};
static constexpr uint8_t I2C_ADDRESS_0x0 {0U};

static constexpr uint8_t CAM_PWR_METHOD_0 {0U};

/** @brief Helper function to set requested
 * @param[in] bit number to be set.
 * @returns uint8_t with request bit set or 0U is bit number to set is >= uint8_t size */
static constexpr uint8_t bit8(uint8_t const i) noexcept
{
    uint8_t const result{
        (i < 8U) ?
            static_cast<uint8_t>((static_cast<uint32_t>(1) << i) & 0xFFU) :
            uint8_t()
    };
    return result;
}

/** @brief Helper function to set requested
 * @param[in] bit number to be set.
 * @returns uint32_t with request bit set or 0U is bit number to set is >= uint8_t size */
static constexpr uint32_t bit32(uint8_t const i) noexcept
{
    uint32_t const result{
        (i < 32U) ? (static_cast<uint32_t>(1) << i) : uint32_t()
    };
    return result;
}

/** @brief Helper function to convert bool to uint32_t
 * @param[in] bool (true or false)
 * @returns 1U or 0U in uint32_t */
constexpr uint32_t boolToU32 (bool const value) noexcept
{
        uint32_t const result{value ? static_cast<uint32_t>(1U) : uint32_t()};
        return result;
}
}
#endif /* CDD_DEFINES_HPP */
