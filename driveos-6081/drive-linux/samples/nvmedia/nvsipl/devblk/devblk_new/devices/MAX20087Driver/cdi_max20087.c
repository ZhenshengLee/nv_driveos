/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "devblk_cdi.h"
#include "devblk_cdi_i2c.h"
#include "cdi_max20087.h"
#include "os_common.h"
#include "sipl_error.h"
#include "cdd_nv_error.h"
#include "sipl_util.h"
#ifdef NVMEDIA_QNX
#include "cdi_max20087_qnx.h"
#else
#include "cdi_max20087_linux.h"
#endif // NVMEDIA_QNX

typedef struct {
    DevBlkCDII2CPgmr    i2cProgrammer;
} DriverHandleMAX20087;

static
DriverHandleMAX20087* getHandlePrivMAX20087(DevBlkCDIDevice const* handle)
{
    DriverHandleMAX20087* drvHandle;

    if (NULL != handle) {
        drvHandle = (DriverHandleMAX20087 *)handle->deviceDriverHandle;
    } else {
        drvHandle = NULL;
    }
    return drvHandle;
}

static NvMediaStatus
MAX20087DriverCreate(
    DevBlkCDIDevice *handle,
    void const* clientContext)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 *drvHandle = NULL;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("MAX20087DriverCreate : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else if (NULL != clientContext) {
        SIPL_LOG_ERR_STR("MAX20087DriverCreate : Context must not be supplied");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        drvHandle = calloc(1U, sizeof(*drvHandle));
        if (NULL == drvHandle) {
            SIPL_LOG_ERR_STR_UINT("MAX20087DriverCreate : memory allocation "
                "failed", (uint32_t)status);
            status = NVMEDIA_STATUS_OUT_OF_MEMORY;
        } else {
            handle->deviceDriverHandle = (void *)drvHandle;

            /* Create the I2C Programmer for register read/write */
            drvHandle->i2cProgrammer = DevBlkCDII2CPgmrCreate(handle,
                                                              REGISTER_ADDRESS_LENGTH,
                                                              REGISTER_DATA_LENGTH);
            if (NULL == drvHandle->i2cProgrammer) {
                free(drvHandle);
                handle->deviceDriverHandle = NULL;
                SIPL_LOG_ERR_STR("MAX20087DriverCreate : Failed initialize the "
                    "I2C Programmer");
                status = NVMEDIA_STATUS_ERROR;
            }
        }
    }
    return status;
}

static NvMediaStatus
MAX20087DriverDestroy(
    DevBlkCDIDevice *handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    if (NULL == drvHandle) {
        SIPL_LOG_ERR_STR("MAX20087DriverDestroy : Bad Parameter: Null ptr");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
    } else {
        /* Destroy the I2C Programmer */
        DevBlkCDII2CPgmrDestroy(drvHandle->i2cProgrammer);

        free(handle->deviceDriverHandle);
        handle->deviceDriverHandle = NULL;
    }
    return status;
}

/**
 * Get DT property values (uint32_t values) under the pwr_ctrl's max20087 node
 * with the given name
*/
static NvMediaStatus
MAX20087GetDTValue(
    const char* name,
    uint32_t* value,
    int32_t const csiPort)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    status = GetDTPropU32(name, value, csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_2STR("MAX20087GetDTValue : failed to read property: ",
             name);
    }
    return status;
}

static NvMediaStatus
MAX20087GetGpioIndex(
    uint32_t* gpio_index,
    int32_t const csiPort)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
#ifdef NVMEDIA_QNX
    uint32_t propValue = 0;

    status = GetDTPropFromPhandle("cam_pwr_int_gpio_index",
                                  "index",
                                  &propValue,
                                  csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR_INT("MAX20087GetGpioIndex : failed to get"
            " 'cam_pwr_int_gpio_index' pHandle : ", (int32_t)status);
    } else {
        *gpio_index = propValue;
    }

#else
    // TODO : Add support for linux
    SIPL_LOG_ERR_STR("MAX20087GetGpioIndex : Unsupported for Linux");
#endif

    return status;
}

static NvMediaStatus
MAX20087ReadRegisterImpl(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff,
    bool const verify)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("MAX20087ReadRegisterImpl : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto end;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        if (verify) {
            status = DevBlkCDII2CPgmrReadUint8Verify(drvHandle->i2cProgrammer,
                                                    toUint16FromUint32(registerNum + i),
                                                    &dataBuff[i]);
        } else {
            status = DevBlkCDII2CPgmrReadUint8(drvHandle->i2cProgrammer,
                                               toUint16FromUint32(registerNum + i),
                                               &dataBuff[i]);
        }
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087ReadRegisterImpl : Failed to read");
            goto end;
        }
    }
end:
    return status;
}

NvMediaStatus
MAX20087ReadRegister(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    return MAX20087ReadRegisterImpl(handle, deviceIndex, registerNum,
                                    dataLength, dataBuff, VERIFY_FALSE);
}

/**
 * Verifying I2C transactions for mask, config and ID registers.
 * Skipping I2C readback for Stat registers (due to clear on read property)
 * and ADC registers (as it may report false positive)
 */
static NvMediaStatus
MAX20087ReadRegisterVerify(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t *dataBuff)
{
    return MAX20087ReadRegisterImpl(handle, deviceIndex, registerNum,
                                    dataLength, dataBuff, VERIFY_TRUE);
}

static NvMediaStatus
MAX20087WriteRegisterImpl(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const *dataBuff,
    bool const verify)
{

    NvMediaStatus status = NVMEDIA_STATUS_OK;
    DriverHandleMAX20087 const* drvHandle = getHandlePrivMAX20087(handle);

    (void) deviceIndex; // unused parameter

    if ((NULL == drvHandle) || (NULL == dataBuff)) {
        SIPL_LOG_ERR_STR("MAX20087WriteRegisterImpl : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto end;
    }

    for (uint32_t i = 0U; i < dataLength; i++) {
        if (verify) {
            status = DevBlkCDII2CPgmrWriteUint8Verify(drvHandle->i2cProgrammer,
                                                    toUint16FromUint32(registerNum + i),
                                                    dataBuff[i]);
        } else {
            status = DevBlkCDII2CPgmrWriteUint8(drvHandle->i2cProgrammer,
                                                toUint16FromUint32(registerNum + i),
                                                dataBuff[i]);
        }
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087WriteRegisterImpl : Failed to write");
            goto end;
        }
    }
end:
    return status;
}

NvMediaStatus
MAX20087WriteRegister(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const *dataBuff)
{
    return MAX20087WriteRegisterImpl(handle, deviceIndex, registerNum,
                                     dataLength, dataBuff, VERIFY_FALSE);
}

/**
 * Verifying I2C transactions for mask, config and ID registers.
 * Skipping I2C readback for Stat registers (due to clear on read property)
 * and ADC registers (as it may report false positive)
 */
static NvMediaStatus
MAX20087WriteRegisterVerify(
    DevBlkCDIDevice const *handle,
    uint32_t deviceIndex,
    uint32_t registerNum,
    uint32_t dataLength,
    uint8_t const *dataBuff)
{
    return MAX20087WriteRegisterImpl(handle, deviceIndex, registerNum,
                                     dataLength, dataBuff, VERIFY_TRUE);
}

/**
 * SM4: OV comparator Diagnostics
 *
 * Part 1:
 * Set CONFIG.EN[4:1] = low
 * Set EN pin = high
 * Set MASK.OVTST = high, MASK.UVM = low
 * Verify STAT1.OVIN = high, STAT1.OVDD = high, STAT2.OV[4:1] = high
 * Verify STAT2.UV[4:1] = high
 *
 * Part 2:
 * Set MASK.UVM = high
 * Verify STAT2.UV[4:1] = high
 *
 * Part 3:
 * Set MASK.OVTST = low, MASK.UVM = low
 *
 * Deviation from Original steps implemented. Details in Bug 3884998
 */
static NvMediaStatus
MAX20087_SM_4(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U, stat_data[2] = {0U};
    bool verify_uv_bits = false;
    uint8_t i = 0U;

    if (NULL == handle) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /* Part 1 */
    /* Mask.OVTST = high */
    data |= (uint8_t)((1U << OVTST_BIT));
    /* MASK.UVM = low */
    /* coverity[cert_int31_c_violation] : intentional */
    data &= (uint8_t)(~(1U << UV_BIT));

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to write ADDR_REG_MASK");
        goto end;
    }

    // Sleep after write for 2 ms
    nvsleep(SLEEP_TIME);

    /**
    * Verifying I2C transactions for mask, config and ID registers.
    * Skipping I2C readback for Stat registers (due to clear on read property)
    * and ADC registers (as it may report false positive)
    */
    status = MAX20087ReadRegister(handle,
                                  0U,
                                  ADDR_REG_STAT1,
                                  REGISTER_DATA_LENGTH,
                                  &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to read ADDR_REG_STAT1");
        goto end;
    }
    if (((data & (1U << STAT1_OVDD_BIT)) != (1U << STAT1_OVDD_BIT)) ||
        ((data & (1U << STAT1_OVIN_BIT)) != (1U << STAT1_OVIN_BIT))) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Verify OVIN and OVDD bit set failed");
        status = NVMEDIA_STATUS_ERROR;
        goto end;
    }

    /**
     * Bug 3807625: Exiting the loop if time exceeds 450ms
     * 450 milliseconds is based on experimental analysis for sensor
     * capacitors to discharge and set the UV bits
    */
    for(i = 0U; i < (BUG_3807625_SLEEP_TIME/SLEEP_PERIOD_MS); i++) {
        status = MAX20087ReadRegister(handle,
                                      0U,
                                      ADDR_REG_STAT2_1,
                                      REGISTER_DATA_LENGTH * NUM_STAT2_REG,
                                      stat_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to read ADDR_REG_STAT2_1");
            goto end;
        }

        // Check STAT2(Upper byte) bit 0,4 = high
        // Check STAT2(Upper byte) bit 1,5 = high
        // Check STAT2(Lower byte) bit 0,4 = high
        // Check STAT2(Lower byte) bit 1,5 = high
        // Wait till Stat2 OV and UV bits are set
        if (((stat_data[0] & STAT2_UV_MASK) == STAT2_UV_MASK) &&
            ((stat_data[0] & STAT2_OV_MASK) == STAT2_OV_MASK) &&
            ((stat_data[1] & STAT2_UV_MASK) == STAT2_UV_MASK) &&
            ((stat_data[1] & STAT2_OV_MASK) == STAT2_OV_MASK)) {
            verify_uv_bits = true;
            break;
        }
        nvsleep(SLEEP_10MS_AS_US);
    }

    if (!verify_uv_bits) {
        status = NVMEDIA_STATUS_ERROR;
        SIPL_LOG_ERR_STR(
                    "MAX20087_SM_4 : UV bit not set after "
                    "BUG_3807625_SLEEP_TIME milliseconds");
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /* Part 2 */
    /* Set MASK.UVM = high */
    data |= (uint8_t)(1U << UV_BIT);

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to write ADDR_REG_MASK");
        goto end;
    }

    /**
     * Bug 4025592: Exiting the loop if time exceeds 450ms
     * 450 milliseconds is based on experimental analysis for sensor
     * capacitors to discharge and set the UV bits
    */
    verify_uv_bits = false;
    for(i = 0U; i < (BUG_3807625_SLEEP_TIME/SLEEP_PERIOD_MS); i++) {
        status = MAX20087ReadRegister(handle,
                                      0U,
                                      ADDR_REG_STAT2_1,
                                      REGISTER_DATA_LENGTH * NUM_STAT2_REG,
                                      stat_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to  read ADDR_REG_STAT2_1");
            goto end;
        }
        // Check STAT2(Upper byte) bit 0, 4 = high
        if(((stat_data[0] & STAT2_UV_MASK) == STAT2_UV_MASK) &&
           ((stat_data[1] & STAT2_UV_MASK) == STAT2_UV_MASK)) {
            verify_uv_bits = true;
            break;
        }
        nvsleep(SLEEP_10MS_AS_US);
    }
    if (!verify_uv_bits) {
         status = NVMEDIA_STATUS_ERROR;
         SIPL_LOG_ERR_STR("Verify UV bits set after setting UVM Mask failed");
         goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /* Part 3 */
    /* Mask.OVTST = low, MASK.UVM = low */
    /* coverity[cert_int31_c_violation] : intentional */
    data &= (uint8_t)(~((1U << OVTST_BIT) | (1U << UV_BIT)));

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_4 : Failed to write ADDR_REG_MASK");
        goto end;
    }
end:
    return status;
}

/*
 * Check if the current gpio pin state as expected.
 */
static NvMediaStatus CheckINTPinState(
                        const DevBlkCDIRootDevice* const cdiRootDev,
                        uint32_t gpioIndex,
                        uint32_t expectedPinLevel)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

#ifdef NVMEDIA_QNX
    uint32_t pinLevel = 0;

    status = DevBlkCDIRootDeviceGetGPIOPinLevel(cdiRootDev,
                                                gpioIndex,
                                                &pinLevel);
    if ((status != NVMEDIA_STATUS_OK) || (pinLevel != expectedPinLevel)) {
        const char *err_msg = NULL;
        uint32_t err_val = 0;

        if (status != NVMEDIA_STATUS_OK) {
            err_msg = "ERROR : Unable to get pin level, status ";
            err_val = (uint32_t)status;
        } else {
            err_msg = "ERROR : Incorrect pin level, level ";
            err_val = pinLevel;
            // If DevBlkCDIRootDeviceGetGPIOPinLevel() API return pass,
            // but pinLevel is not as expected, overwrite the status with
            // error.
            status = NVMEDIA_STATUS_ERROR;
        }

        SIPL_LOG_ERR_STR_UINT(err_msg, err_val);
    }
#else
    // TODO : Add support for linux
    SIPL_LOG_ERR_STR("CheckINTPinState() : Unsupported for Linux");
#endif

    return status;
}

/**
 * SM5: UV Comparator Diagnostics
 *
 * Set EN pin = high
 * Set CONFIG.EN[4:1] = low
 * Verify STAT2.UV[4:1] = high and INT pin is low
 * Set MASK.UVM = high
 * Verify STAT2.UV[4:1] = high and INT pin is high
 *
 * Deviation from Original steps implemented. Details in Bug 3884998
 */
static NvMediaStatus
MAX20087_SM_5(
    DevBlkCDIRootDevice* const cdiRootDev,
    DevBlkCDIDevice const* handle,
    int32_t const csiPort)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data_mask = 0U, stat_data[2] = {0U};
    uint32_t gpio_index = 0;
    bool verify_uv_bits = false;
    uint32_t i = 0U;

    status = MAX20087GetGpioIndex(&gpio_index, csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "MAX20087_SM_5 : Error in getting gpio-index value from dt\n");
        status = NVMEDIA_STATUS_ERROR;
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data_mask);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /* MASK.UVM = low */
    /* coverity[cert_int31_c_violation] : intentional */
    data_mask &= (uint8_t)(~(1U << UV_BIT));

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data_mask);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to write ADDR_REG_MASK");
        goto end;
    }

    /**
     * Bug 3807625: Exiting the loop if time exceeds 450ms
     * 450 milliseconds is based on experimental analysis for sensor
     * capacitors to discharge and set the UV bits
    */
    for(i = 0U; i < (BUG_3807625_SLEEP_TIME/SLEEP_PERIOD_MS); i++) {
        status = MAX20087ReadRegister(handle,
                                      0U,
                                      ADDR_REG_STAT2_1,
                                      REGISTER_DATA_LENGTH * NUM_STAT2_REG,
                                      stat_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to read ADDR_REG_STAT2_1");
            goto end;
        }

        /* Check STAT2(Upper byte) bit 0,4 = high and check STAT2(Lower byte)
        bit 0,4 = high */
        if (((stat_data[0] & STAT2_UV_MASK) == STAT2_UV_MASK) &&
            ((stat_data[1] & STAT2_UV_MASK) == STAT2_UV_MASK)) {
            verify_uv_bits = true;
            break;
        }
        nvsleep(SLEEP_10MS_AS_US);
    }

    if (!verify_uv_bits) {
        status = NVMEDIA_STATUS_ERROR;
        SIPL_LOG_ERR_STR(
            "MAX20087_SM_5 : Verify UV bits set before UV mask failed");
        goto end;
    }

    status = CheckINTPinState(cdiRootDev,
                              gpio_index,
                              DEVBLK_CDI_GPIO_LEVEL_LOW);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Unexpected pin level");
        goto end;
    }

    /* Set MASK.UVM = high */
    data_mask |= (uint8_t)(1U << UV_BIT);

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data_mask);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /**
     * Bug 3807625: Exiting the loop if time exceeds 450ms
     * 450 milliseconds is based on experimental analysis for sensor
     * capacitors to discharge and set the UV bits
    */
    verify_uv_bits = false;
    for(i = 0U; i < (BUG_3807625_SLEEP_TIME/SLEEP_PERIOD_MS); i++) {
        status = MAX20087ReadRegister(handle,
                                      0U,
                                      ADDR_REG_STAT2_1,
                                      REGISTER_DATA_LENGTH * NUM_STAT2_REG,
                                      stat_data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to read ADDR_REG_STAT2_1");
            goto end;
        }

        /* Check STAT2(Upper byte) bit 0,4 = high and check STAT2(Lower byte)
        bit 0,4 = high */
        if (((stat_data[0] & STAT2_UV_MASK) == STAT2_UV_MASK) &&
            ((stat_data[1] & STAT2_UV_MASK) == STAT2_UV_MASK)) {
            verify_uv_bits = true;
            break;
        }
        nvsleep(SLEEP_10MS_AS_US);
    }

    if (!verify_uv_bits) {
        status = NVMEDIA_STATUS_ERROR;
        SIPL_LOG_ERR_STR(
            "MAX20087_SM_5 : Verify UV bits set after UV mask failed");
        goto end;
    }


    status = CheckINTPinState(cdiRootDev,
                              gpio_index,
                              DEVBLK_CDI_GPIO_LEVEL_HIGH);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Unexpected pin level");
        goto end;
    }

    /* Set MASK.UVM = low */
    data_mask &= (uint8_t)(~(1U << UV_BIT));

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data_mask);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_5 : Failed to write ADDR_REG_MASK");
        goto end;
    }

end:
   return status;
}

/**
 * SM8: ADC Diagnostics
 *
 * Set CONFIG.EN[4:1] = low
 * Set CONFIG.MUX[1:0] = 2b10
 * Read ADC1 and confirm it matches with applied Vin
 * Read ADC2 and confirm it matches with applied Vdd
 * Read ADC3 and confirm it matches with applied Viset
 *
 */
static NvMediaStatus
MAX20087_SM_8(
    DevBlkCDIDevice const* handle,
    int32_t const csiPort)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U;
    uint32_t voltage_vin = 0U, voltage_vdd = 0U, voltage_viset = 0U;
    uint32_t expected_vin = 0U, expected_vdd = 0U, expected_viset = 0U;
    uint32_t viset_min = 0U, viset_max = 0U, vdd_margin = 0U, vin_margin = 0U;
    uint32_t r_iset = 0U, riset_tolerance = 0U;
    uint32_t iset_max = 0U, iset_min = 0U, riset_min = 0U, riset_max = 0U;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("MAX20087_SM_8 : Bad Parameter");
        goto end;
    }
    status = MAX20087GetDTValue("vin", &expected_vin, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting vin value",
             (uint32_t)status);
        goto end;
    }
    status = MAX20087GetDTValue("vin_margin", &vin_margin, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting vin_margin "
            "value", (uint32_t)status);
        goto end;
    }

    status = MAX20087GetDTValue("vdd", &expected_vdd, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting vdd value",
             (uint32_t)status);
        goto end;
    }
    status = MAX20087GetDTValue("vdd_margin", &vdd_margin, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting vdd_margin "
            "value", (uint32_t)status);
        goto end;
    }

    status = MAX20087GetDTValue("r_iset", &r_iset, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting r_iset value",
            (uint32_t)status);
        goto end;
    }
    status = MAX20087GetDTValue("r_iset_tolerance", &riset_tolerance, csiPort);
    if ((status != NVMEDIA_STATUS_OK)) {
        SIPL_LOG_ERR_STR_UINT("MAX20087_SM_8 : Error in getting r_iset "
            "tolerance value", (uint32_t)status);
        goto end;
    }
    PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: vin: %d mV, vdd: %d mV, "
        "r_iset: %d Ohms\n", expected_vin, expected_vdd, r_iset);
    PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: vin_margin: %d%%, "
        "vdd_margin: %d%%, riset_tolerance: %d%%\n", vin_margin, vdd_margin,
        riset_tolerance);

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_8 : Failed to read ADDR_REG_CONFIG");
        goto end;
    }

    /* Write CONFIG.MUX = 10, CONFIG.EN[4:1] = low, CONFIG.ENC = high */

    data &= 0xB0U;
    data |= 0xA0U;

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_8 : Failed to write ADDR_REG_CONFIG");
        goto end;
    }

    // Sleep after write for 2 ms
    nvsleep(SLEEP_TIME);
    // Skipping VIN check if vin_margin = 0xFFFFFFFF from dt file.
    if (vin_margin != UINT32_MAX) {
        status = MAX20087ReadRegister(handle,
                                    0U,
                                    ADDR_REG_ADC1,
                                    REGISTER_DATA_LENGTH,
                                    &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : Failed to read ADDR_REG_ADC1");
            goto end;
        }

        voltage_vin = (uint32_t)data * VIN_MULTIPLIER;

        if (CHECK_RANGE(voltage_vin, expected_vin, vin_margin, MARGIN_FACTOR)) {
            status = NVMEDIA_STATUS_ERROR;
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : VIN not in range")
            goto end;
        }
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: Skipped VIN Check\n");
    }

    // Skipping VDD check if vdd_margin = 0xFFFFFFFF from dt file.
    if (vdd_margin != UINT32_MAX) {
        status = MAX20087ReadRegister(handle,
                                    0U,
                                    ADDR_REG_ADC2,
                                    REGISTER_DATA_LENGTH,
                                    &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : Failed to read ADDR_REG_ADC2");
            goto end;
        }

        voltage_vdd = (uint32_t)data * VDD_MULTIPLIER;

        if (CHECK_RANGE(voltage_vdd, expected_vdd, vdd_margin, MARGIN_FACTOR)) {
            status = NVMEDIA_STATUS_ERROR;
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : VDD not in range")
            goto end;
        }
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: Skipped VDD Check\n");
    }

    // Skipping VISET check when r_iset is intentionally set to 0 in dt file.
    if (r_iset != 0U) {
        status = MAX20087ReadRegister(handle,
                                    0U,
                                    ADDR_REG_ADC3,
                                    REGISTER_DATA_LENGTH,
                                    &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : Failed to read ADDR_REG_ADC3");
            goto end;
        }

        // Voltage_viset value in micro-volt by multiplying by 1000
        voltage_viset = (uint32_t)data * VISET_MULTIPLIER * VISET_MARGIN_FACTOR;


        /**
         * viset_min/max = iset_min/max * riset_min/max
         * riset_tolerance has a factor of 100,
         * iset (NOMINAL_CURRENT) values are expressed in nanoAmpere.
        */
        iset_min = (NOMINAL_CURRENT - NOMINAL_CURRENT_VARIANCE);
        iset_max = (NOMINAL_CURRENT + NOMINAL_CURRENT_VARIANCE);
        riset_min = MIN_RANGE(r_iset, riset_tolerance, MARGIN_FACTOR);
        riset_max = MAX_RANGE(r_iset, riset_tolerance, MARGIN_FACTOR);
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: "
            "iset_min: %d nA, iset_max: %d nA, riset_min: %d Ohms, "
            "riset_max: %d Ohms\n", iset_min, iset_max, riset_min, riset_max);

        /**
         * Following viset calculation uses Ohms * nano-Ampere = nanoVolt.
         * Dividing by VISET_MARGIN_FACTOR to convert to microVolt.
         */
        expected_viset = (uint32_t)((NOMINAL_CURRENT * r_iset) /
                                    VISET_MARGIN_FACTOR);
        // Ignoring violation as it is checked in the following if.
        /* coverity[cert_int30_c_violation] : intentional */
        viset_min = (uint32_t)((iset_min * riset_min) / VISET_MARGIN_FACTOR);
        viset_max = (uint32_t)((iset_max * riset_max) / VISET_MARGIN_FACTOR);
        // Check for uint32_t wrap for cert violation for both viset min & max.
        if ((viset_max < expected_viset) || (viset_min > expected_viset)) {
            SIPL_LOG_ERR_STR("MAX20087_SM_8 : Unsigned Integer wrap");
            status = NVMEDIA_STATUS_ERROR;
            goto end;
        }
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: viset_min: %d uV, "
            "viset_max: %d uV, viset calculated: %d uV, viset actual: %d uV\n",
            viset_min, viset_max, expected_viset, voltage_viset);
        if ((voltage_viset < viset_min) || (voltage_viset > viset_max)) {
            SIPL_LOG_ERR_STR(
                "MAX20087_SM_8 : VISET does not fall in required range");
            status = NVMEDIA_STATUS_ERROR;
            goto end;
        }
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8: Skipped VISET Check\n");
    }
end:
    return status;
}

/**
 * SM17: OCP FET Switching Diagnostics
 * Set CONFIG.EN[4:1] = low
 * Set CONFIG.MUX[1:0] = 00
 * Read ADC1 and confirm its 0
 * Read ADC2 and confirm its 0
 * Read ADC3 and confirm its 0
 * Read ADC4 and confirm its 0
 */
static NvMediaStatus
MAX20087_SM_17(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0U, i = 0U;

    if (NULL == handle) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("MAX20087_SM_17 : Bad Parameter");
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_17 : Failed to read ADDR_REG_CONFIG");
        goto end;
    }

    /* Write CONFIG.MUX = 00, CONFIG.EN[4:1] = low */
    data &= 0x30U;

    /* Enable CONFIG.ENC*/
    data |= (1U << ENC_BIT);

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087_SM_17 : Failed to write ADDR_REG_CONFIG");
        goto end;
    }

    // Giving delay of 2ms for voltage change.
    nvsleep(2000);

    for(i = 0U; i < ADC_LENGTH; i++ ) {
        status = MAX20087ReadRegister(handle,
                                      0U,
                                      ADDR_REG_ADC1 + i,
                                      REGISTER_DATA_LENGTH,
                                      &data);

        if (status != NVMEDIA_STATUS_OK || data != 0x0U) {
            if (data != 0x0U) {
                status = NVMEDIA_STATUS_ERROR;
                SIPL_LOG_ERR_STR_UINT(
                    "MAX20087_SM_17 : Verification failed for ADC",
                    (i+1U));
            }
            goto end;
        }
    }
end:
    return status;
}

NvMediaStatus
MAX20087Init(
    DevBlkCDIRootDevice* const cdiRootDev,
    DevBlkCDIDevice const* handle,
    int32_t const csiPort)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data = 0;

    if ((NULL == handle) || (cdiRootDev == NULL)) {
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        SIPL_LOG_ERR_STR("MAX20087Init : Bad Parameter");
        goto end;
    }

    /***
     * TODO: Add GetDeserPower API to confirm that EN pin is set before running
     *       powerswitch SMs.
     */
    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to read ADDR_REG_MASK");
        goto end;
    }

    /* Mask ACCM and UV event */
    data |= (1U << UV_BIT);
    data |= (1U << ACCM_BIT);
    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to write ADDR_REG_MASK");
        goto end;
    }

    /* Disable power for all links */
    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to read ADDR_REG_CONFIG");
        goto end;
    }

    /* Disabling CONFIG.EN bits by resetting last 4 bits to 0 */
    data &= (MSB_4_MASK);
    /**
     * Setting ENC and CLR bit
     * CLR Bit is required to be set for SM5.  */
    data |= (1U << ENC_BIT);
    data |= (1U << CLR_BIT);

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to write ADDR_REG_CONFIG");
        goto end;
    }

    status = MAX20087_SM_4(handle);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : MAX20087_SM_4 failed");
        goto end;
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM4 is successfully executed\n");
    }

    status = MAX20087_SM_5(cdiRootDev, handle, csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : MAX20087_SM_5 failed");
        goto end;
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM5 is successfully executed\n");
    }

    status = MAX20087_SM_8(handle, csiPort);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : MAX20087_SM_8 failed");
        goto end;
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM8 is successfully executed\n");
    }

    status = MAX20087_SM_17(handle);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : MAX20087_SM_17 failed");
        goto end;
    } else {
        PrintLogMsg(LOG_LEVEL_INFO, "MAX20087: SM17 is successfully "
                        "executed\n");
    }

    /**
     * Mask UV event. WAR for Bug 3557250 with context from Bug 3453278.
     * This is resolved in MAX20087SetLinkPower.
     * The MAX20087 Power switch generates UV errors (in the STAT registers as
     * well as on INT pin if not masked) when there's an UnderVolt condition on
     * the output. Undervolt condition is even true when an output is disabled.
     * During init time, there's no awareness of how many outputs will be
     * enabled hence in SetLinkPower, the UV errors are unmasked only when all
     * 4 output links are enabled (else the disabled links generate UV faults)
     */
    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_MASK,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to read ADDR_REG_MASK");
        goto end;
    }
    data |= (1U << UV_BIT);
    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_MASK,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to write ADDR_REG_MASK");
        goto end;
    }

    /* Resetting CLR bit after end of SM. */
    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to read ADDR_REG_CONFIG");
        goto end;
    }
    data &= (uint8_t)(~(1U << CLR_BIT));
    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087Init : Failed to write ADDR_REG_CONFIG");
        goto end;
    }
end:
    return status;
}

NvMediaStatus
MAX20087CheckPresence(
    DevBlkCDIDevice const* handle)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t dev_rev_id = 0U;
    uint8_t data = 0;

    if (handle == NULL) {
        SIPL_LOG_ERR_STR("MAX20087CheckPresence : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_ID,
                                        REGISTER_DATA_LENGTH,
                                        &dev_rev_id);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087CheckPresence : Failed to read ADDR_REG_ID");
        goto end;
    }

    if (((dev_rev_id & 0x30) != MAX20087_DEV_ID) ||
        ((dev_rev_id & 0xF) != MAX20087_REV_ID)) {
        SIPL_LOG_ERR_STR("MAX20087CheckPresence : Dev/Rev ID not match");
        status = NVMEDIA_STATUS_ERROR;
        goto end;
    }

    /* Disable all individual EN bits */
    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "MAX20087SetLinkPower : Failed to read ADDR_REG_CONFIG");
        goto end;
    }

    /* Disabling CONFIG.EN bits by resetting last 4 bits to 0 */
    data &= (MSB_4_MASK);

    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "MAX20087SetLinkPower : Failed to write ADDR_REG_CONFIG");
        goto end;
    }
end:
    return status;
}

// Helper function to deal with UV mask when enabling/disabling output links
static NvMediaStatus
SetLinkPowerUpdateUVMask(
    DevBlkCDIDevice const * handle,
    bool const disable_uv_mask,
    uint8_t const stateInterruptMask,
    uint8_t *savedInterruptMask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data;

    if (stateInterruptMask != INTERRUPT_MASKED_STATE) {
        // Update UV mask
        status = MAX20087ReadRegisterVerify(handle,
                                            0U,
                                            ADDR_REG_MASK,
                                            REGISTER_DATA_LENGTH,
                                            &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR(
                "SetLinkPowerUpdateUVMask : Failed to read ADDR_REG_MASK");
            goto end;
        }
        if (disable_uv_mask) {
            data &= (uint8_t)(~bit8(UV_BIT));
        } else {
            data |= bit8(UV_BIT);
        }
        status = MAX20087WriteRegisterVerify(handle,
                                             0U,
                                             ADDR_REG_MASK,
                                             REGISTER_DATA_LENGTH,
                                             &data);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR(
                "SetLinkPowerUpdateUVMask : Failed to write ADDR_REG_MASK");
            goto end;
        }
    } else {
        if (disable_uv_mask) {
            *savedInterruptMask &= ~bit8(UV_BIT);
        } else {
            *savedInterruptMask |= bit8(UV_BIT);
        }
    }

end:
    return status;
}

NvMediaStatus
MAX20087SetLinkPower(
    DevBlkCDIDevice const* handle,
    uint8_t const linkIndex,
    bool const enable,
    uint8_t const stateInterruptMask,
    uint8_t *savedInterruptMask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    uint8_t data, prev_state, new_state, stat2_data[2];
    bool read_stat2_1 = true;

    if (((handle == NULL) || (savedInterruptMask == NULL)) ||
        (stateInterruptMask > INTERRUPT_RESTORED_STATE)) {
        SIPL_LOG_ERR_STR("MAX20087SetLinkPower : Bad Parameter");
        status = NVMEDIA_STATUS_BAD_PARAMETER;
        goto end;
    }

    status = MAX20087ReadRegisterVerify(handle,
                                        0U,
                                        ADDR_REG_CONFIG,
                                        REGISTER_DATA_LENGTH,
                                        &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR(
            "MAX20087SetLinkPower : Failed to read ADDR_REG_CONFIG");
        goto end;
    }
    prev_state = (data & ALL_OUT_LINKS_ENABLED);
    if (enable) {
        data |= bit8(linkIndex);
    } else {
        data &= ~bit8(linkIndex);
    }
    new_state = (data & ALL_OUT_LINKS_ENABLED);

    if ((prev_state == ALL_OUT_LINKS_ENABLED) && (new_state != ALL_OUT_LINKS_ENABLED)) {
        /**
         * When enabling UV mask, update the UV mask first, then disable the output link
         * and clear UV errors.
        **/
        status = SetLinkPowerUpdateUVMask(handle, !DISABLE_UV_MASK, stateInterruptMask,
                                          savedInterruptMask);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087SetLinkPower : Failure in enabling UV Mask");
            goto end;
        }
    }
    status = MAX20087WriteRegisterVerify(handle,
                                         0U,
                                         ADDR_REG_CONFIG,
                                         REGISTER_DATA_LENGTH,
                                         &data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087SetLinkPower : Failed to write ADDR_REG_CONFIG");
    }
    /**
     * Read status registers to clear the UV error on the modified link.
     * There's a possibility of missing out on a valid UV error on a link that was previously
     * enabled but a different link is now being enabled.
     * (Say Link 0 was enabled and it reported a UV error but now it is cleared when Link 1 is
     * being enabled. This fault will be missed due to clearing of STAT2 registers).
     * The scope is reduced to only faults missed in Links sharing a STAT2 register.
     * This is documented in Bug 3557250
     */

    /* Read STAT2 Reg for the link being enabled */
    read_stat2_1 = ((linkIndex / NUM_LINKS_PER_STAT2) == 0U);
    status = MAX20087ReadRegister(handle,
                                  0U,
                                  read_stat2_1 ? ADDR_REG_STAT2_1 : ADDR_REG_STAT2_2,
                                  REGISTER_DATA_LENGTH,
                                  stat2_data);
    if (status != NVMEDIA_STATUS_OK) {
        SIPL_LOG_ERR_STR("MAX20087SetLinkPower : Failed to read ADDR_REG_STAT2_1");
        goto end;
    }

    if ((prev_state != ALL_OUT_LINKS_ENABLED) && (new_state == ALL_OUT_LINKS_ENABLED)) {
        /**
         * When disabling UV mask, enable output link first (to change UV state)
         * and then clear the status register to update the INT pin status as already done above.
         * Finally, unmask UV errors.
        **/
        status = SetLinkPowerUpdateUVMask(handle, DISABLE_UV_MASK, stateInterruptMask,
                                          savedInterruptMask);
        if (status != NVMEDIA_STATUS_OK) {
            SIPL_LOG_ERR_STR("MAX20087SetLinkPower : Failure in disabling UV Mask");
            goto end;
        }
    }

end:
    return status;
}

NvMediaStatus
MAX20087MaskRestoreGlobalInterrupt(
    DevBlkCDIDevice const* handle,
    uint8_t * savedInterruptMask,
    const bool enableGlobalMask)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (enableGlobalMask) {
         /* Save the current interrupt mask */
        status = MAX20087ReadRegisterVerify(handle,
                                            0U,
                                            ADDR_REG_MASK,
                                            REGISTER_DATA_LENGTH,
                                            savedInterruptMask);
        if (status == NVMEDIA_STATUS_OK) {
            /* Mask all power switch interrupts */
            uint8_t mask = 0x7FU;
            status = MAX20087WriteRegisterVerify(handle,
                                                0U,
                                                ADDR_REG_MASK,
                                                REGISTER_DATA_LENGTH,
                                                &mask);
        }
    } else {
         /* Restore mask for interrupts */
        status = MAX20087WriteRegisterVerify(handle,
                                            0U,
                                            ADDR_REG_MASK,
                                            REGISTER_DATA_LENGTH,
                                            savedInterruptMask);
    }

    return status;
}

static DevBlkCDIDeviceDriver deviceDriver = {
    .deviceName = "MAX20087 Power Load Switch",
    .regLength = REGISTER_ADDRESS_LENGTH,
    .dataLength = REGISTER_DATA_LENGTH,
    .DriverCreate = MAX20087DriverCreate,
    .DriverDestroy = MAX20087DriverDestroy,
    .ReadRegister = MAX20087ReadRegister,
    .WriteRegister = MAX20087WriteRegister,
};

DevBlkCDIDeviceDriver *
GetMAX20087Driver(void)
{
    return &deviceDriver;
}
