/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef NVSIPLDEVICEINTERFACEPROVIDERINTERFACE_HPP
#define NVSIPLDEVICEINTERFACEPROVIDERINTERFACE_HPP

#include <cstring>
#include <cstdint>
/**
 * @file
 *
 * @brief <b> NVIDIA SIPL: Device Interface Provider Interface</b>
 *
 */

namespace nvsipl
{

/** @brief A universally unique identifier. */
class UUID
{
public:
    /** Member variable to store the first opaque field composing the UUID */
    uint32_t time_low;
    /** Member variable to store the second opaque field composing the UUID */
    uint16_t time_mid;
    /** Member variable to store the third opaque field composing the UUID */
    uint16_t time_hi_and_version;
    /** Member variable to store the fourth opaque field composing the UUID */
    uint16_t clock_seq;
    /** Member variable to store the fifth-tenth opaque field composing the UUID */
    uint8_t  node[6];

    /**
     * @brief Constructs a new UUID object with parameter values.
     *
     * This method initializes the class variables with the parameter values passed in.
     * The variables are historically named and are used for opaque data values.
     *
     * @param[in] low The first opaque field.
     * @param[in] mid The second opaque field.
     * @param[in] high The third opaque field.
     * @param[in] seq The fourth opaque field.
     * @param[in] n0 The fifth opaque field.
     * @param[in] n1 The sixth opaque field.
     * @param[in] n2 The seventh opaque field.
     * @param[in] n3 The eighth opaque field.
     * @param[in] n4 The nineth opaque field.
     * @param[in] n5 The tenth opaque field.
     */
    constexpr UUID(uint32_t const low, uint16_t const mid,
         uint16_t const high, uint16_t const seq,
         uint8_t const n0, uint8_t const n1, uint8_t const n2,
         uint8_t const n3, uint8_t const n4, uint8_t const n5)
        : time_low(low)
        , time_mid(mid)
        , time_hi_and_version(high)
        , clock_seq(seq)
        , node{n0,n1,n2,n3,n4,n5}
    { }

    /** @brief Construct for a new UUID object with default intialization values. */
    UUID() : UUID(0U,0U,0U,0U,0U,0U,0U,0U,0U,0U)
    { }

    /** @brief comparison operator to compare contents of two SSID structures */
    friend bool operator==(const UUID &l, const UUID &r) noexcept
    {
        if (l.time_low != r.time_low){
            return false;
        } else if (l.time_mid != r.time_mid){
            return false;
        } else if (l.time_hi_and_version != r.time_hi_and_version){
            return false;
        } else if (l.clock_seq != r.clock_seq){
            return false;
        } else if (l.node[0] != r.node[0]){
            return false;
        } else if (l.node[1] != r.node[1]){
            return false;
        } else if (l.node[2] != r.node[2]){
            return false;
        } else if (l.node[3] != r.node[3]){
            return false;
        } else if (l.node[4] != r.node[4]){
            return false;
        } else if (l.node[5] != r.node[5]){
            return false;
        } else {
            return true;
        }
    };
};

static_assert(sizeof(UUID) ==
              sizeof(uint32_t) +
              sizeof(uint16_t) +
              sizeof(uint16_t) +
              sizeof(uint16_t) +
              sizeof(uint8_t) * 6,
              "Missing check in == operator as Structure shall not have padding");

/** @brief A class to prevent drivers being copied and duplicating state. */
class NonCopyable
{
public:
    /** default destructor */
    virtual ~NonCopyable() = default;

    /** copy constructor */
    NonCopyable(NonCopyable& other) = delete;

    /** copy assignment operator. */
    NonCopyable& operator=(NonCopyable& other) & = delete;

    /** move constructor */
    NonCopyable(NonCopyable&& other) = delete;

    /** move assignment operator. */
    NonCopyable& operator=(NonCopyable&& other) & = delete;

protected:
    /** default constructor */
    NonCopyable() = default;
};

/** @brief Top-level interface class implementable for a particular device.
 *
 * An interface .hpp file can be created with custom APIs intended to be accessable by the client.
 * The client may retrieve this Interface and compare the IDs to ensure they match
 * before casting and using the Interface.
 */
class Interface : public NonCopyable
{
public:
    /** @brief  A call to get the ID from the instance of the class inheriting this interface.
     *
     * @retval  Unique identifier for the interface instance
     */
    virtual const UUID &getInstanceInterfaceID() = 0;

protected:
    /** @brief Default Constructor. */
    Interface() = default;

    /** @brief Delete copy Constructor. */
    Interface(Interface &) = delete;

    /** @brief Delete move Constructor. */
    Interface(Interface &&) = delete;

    /** @brief Delete copy assigned Constructor. */
    Interface& operator=(Interface &) & = delete;

    /** @brief Delete move assigned Constructor. */
    Interface& operator=(Interface &&) & = delete;

    /** @brief Default destructor for the class */
    ~Interface() override = default;
};

/**
 * @brief Class providing access to device interfaces.
 */
class IInterfaceProvider
{
public:
    /**
     * @brief Get interface provided by this driver matching a provided ID
     *
     * This API is used to get access to device specific custom interfaces (if
     * any) defined by this Deserializer and for Camera Module,
     * which can be invoked directly by Device Block clients.
     *
     * The interface should be valid through the lifetime of the device object.
     *
     * @pre  None.
     *
     * @param[in] interfaceId   Unique identifier (of type @ref UUID) for the
     *                          interface to retrieve.
     *
     * @retval    Interface*    matching the provided ID;
     * @retval    nullptr       if no matching Interface found or not implemented.
     *
     * @usage
     * - Allowed context for the API call
     *   - Interrupt handler: No
     *   - Signal handler: No
     *   - Thread-safe: No
     *   - Re-entrant: No
     *   - Async/Sync: Sync
     * - Required privileges: Yes, with the following conditions:
     *   - Grants: nonroot, allow
     *   - Abilities: public_channel
     *   - Application needs to have access to the SGIDs that SIPL depends on as mentioned in the
     *     NVIDIA DRIVE OS Safety Developer Guide
     * - API group
     *   - Init: Yes
     *   - Runtime: No
     *   - De-Init: No
     */
    virtual Interface* GetInterface(const UUID &interfaceId) = 0;

protected:
    /** @brief Default Constructor. */
    IInterfaceProvider() = default;

    /** @brief Delete copy Constructor. */
    IInterfaceProvider(IInterfaceProvider &) = delete;

    /** @brief Delete move Constructor. */
    IInterfaceProvider(IInterfaceProvider &&) = delete;

    /** @brief Delete copy assigned Constructor. */
    IInterfaceProvider& operator=(IInterfaceProvider &) & = delete;

    /** @brief Delete move assigned Constructor. */
    IInterfaceProvider& operator=(IInterfaceProvider &&) & = delete;

    /** @brief Default destructor. */
    virtual ~IInterfaceProvider() = default;
};

} // end of namespace nvsipl
#endif //NVSIPLDEVICEINTERFACEPROVIDERINTERFACE_HPP
