/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file
 * @brief <b>NVIDIA NVCCP Interface: Camera Control Protocol Support</b> <br/>
 * @b Description: This file declares an interface for NVCCP support.
 */

#ifndef CCP_H
#define CCP_H
#include <stdint.h>

/**
 * @defgroup group_Camera_group Camera Control Protocol
 *
 * Describes an API that uses the NVIDIA<sup>&reg;</sup> Camera Control Protocol
 * (NVCCP) to send messages from NVIDIA<sup>&reg;</sup> Tegra<sup>&reg;</sup> to
 * AURIX MCU and receive acknowledgement back. For more information, search for
 * "CCP" in _NVIDIA DRIVE 5.0 Development Guide_.
 * @{
 */

/**
 * Defines values returned to the user library
 * on a nvccp function call depicting the status.
*/
typedef enum
{
  NVCCP_STATUS_OK = 0U,
  NVCCP_STATUS_DENIED,
  NVCCP_STATUS_NOT_REGISTERED,
  NVCCP_STATUS_ALREADY_ON,
  NVCCP_STATUS_ALREADY_OFF,
  NVCCP_STATUS_SLAVE_RUNNING,
  NVCCP_STATUS_I2C_OWNERSHIP_DENIED,
  NVCCP_REQ_FILE_OP_FAIL = 1000U,
  NVCCP_REQ_TIMEOUT,
  NVCCP_REQ_INVALID,
  NVCCP_REQ_DENIED,
  NVCCP_REQ_FAILED,
  NVCCP_REQ_CONNECT_ERR,
} nvccp_return_t;


/**
 * Defines the camera group ID.
 */
typedef enum {
  NVCCP_GROUP_A = 0x00U,
  NVCCP_GROUP_B = 0x01U,
  NVCCP_GROUP_C = 0x02U,
  NVCCP_GROUP_D = 0x03U
} nvccp_cam_group_id;

/**
 * Defines the camera ID.
 */
typedef enum{
  /* Camera group A */
  NVCCP_CAM_A0 = 0x01U,
  NVCCP_CAM_A1 = 0x02U,
  NVCCP_CAM_A2 = 0x04U,
  NVCCP_CAM_A3 = 0x08U,
  /* Camera group B */
  NVCCP_CAM_B0 = 0x10U,
  NVCCP_CAM_B1 = 0x20U,
  NVCCP_CAM_B2 = 0x40U,
  NVCCP_CAM_B3 = 0x80U,
  /* Camera group C */
  NVCCP_CAM_C0 = 0x0100U,
  NVCCP_CAM_C1 = 0x0200U,
  NVCCP_CAM_C2 = 0x0400U,
  NVCCP_CAM_C3 = 0x0800U,
  /* Camera group D */
  NVCCP_CAM_D0 = 0x1000U,
  NVCCP_CAM_D1 = 0x2000U,
  NVCCP_CAM_D2 = 0x4000U,
  NVCCP_CAM_D3 = 0x8000U
} nvccp_cam_id;

/**
 * Defines the tegra ID.
 */
typedef enum {
  NVCCP_TEGRA_A = 0x41U,
  NVCCP_TEGRA_B = 0x42U
} nvccp_tegra_id;

/**
 * Defines the mode in which to request ownership (nvccp_request_ownership())
 * or release ownership (nvccp_release_ownership()).
 */
typedef enum {
  NVCCP_CAM_MASTER = 0x01U,
  NVCCP_CAM_SLAVE
} nvccp_cam_mode;

/**
 * Requests ownership for selected camera group.
 * @param cam_group Aggregator ID (not a bitmask)
 * @param cam_master requested mode of operation from nvccp_cam_mode
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_request_ownership(nvccp_cam_group_id cam_group,nvccp_cam_mode cam_master);

/**
 * Releases ownership for selected camera group.
 * @param cam_group Aggregator ID (not a bitmask)
 * @param cam_master requested mode of operation from nvccp_cam_mode
 * @retval nvccp_return_t
 * */
nvccp_return_t nvccp_release_ownership(nvccp_cam_group_id cam_group,nvccp_cam_mode cam_master);

/**
 * Turns on power for all cameras in the aggregator.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_cam_pwr_on(nvccp_cam_group_id cam_group);

/**
 * Turns off power for all cameras in the aggregator.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_cam_pwr_off(nvccp_cam_group_id cam_group);

/**
 * Turns on power for camera unit.
 * @param cam_id camera ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_cam_unit_pwr_on(nvccp_cam_id cam_id);

/**
 * Turns off power for camera unit.
 * @param cam_id camera ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_cam_unit_pwr_off(nvccp_cam_id cam_id);

/**
 * Turns on power for camera aggregator.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_aggreg_pwr_on(nvccp_cam_group_id cam_group);

/**
 * Turns off power for camera aggregator.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_aggreg_pwr_off(nvccp_cam_group_id cam_group);

/**
 * Sets frame sync (frsync) owner.
 * @param tegra_id Tegra ID
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_frsync_owner(nvccp_tegra_id tegra_id);

/**
 * Sets frsync enable.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_frsync_enable(nvccp_cam_group_id cam_group);

/**
 * Sets frsync disable.
 * @param cam_group Aggregator ID (not a bitmask)
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_set_frsync_disable(nvccp_cam_group_id cam_group);

/**
 * Gets camera unit power status.
 * @param cam_id This will be filled with the camera power status
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_get_cam_unit_pwr_status(uint16_t *cam_id);

/**
 * Gets aggregator power status.
 * @param cam_group This will be filled with the aggregator power status
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_get_aggreg_pwr_status(uint8_t *cam_group);

/**
 * Gets frsync enable status.
 * @param cam_group This will be filled with frsync enable status
 * @retval nvccp_return_t
 */
nvccp_return_t nvccp_get_frsync_enable_status(uint8_t *cam_group);
/** @} */
#endif
