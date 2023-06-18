/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
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

/** @brief A universally unique identifier.
 */
struct UUID
{
    uint32_t time_low;
    uint16_t time_mid;
    uint16_t time_hi_and_version;
    uint16_t clock_seq;
    uint8_t  node[6];

    UUID(uint32_t const low, uint16_t const mid,
         uint16_t const high, uint16_t const seq,
         uint8_t const n0, uint8_t const n1, uint8_t const n2,
         uint8_t const n3, uint8_t const n4, uint8_t const n5)
        : time_low(low)
        , time_mid(mid)
        , time_hi_and_version(high)
        , clock_seq(seq)
        , node{n0,n1,n2,n3,n4,n5}
    { }

    UUID()
        : time_low(0U)
        , time_mid(0U)
        , time_hi_and_version(0U)
        , clock_seq(0U)
        , node{0U,0U,0U,0U,0U,0U}
    { }

    bool operator==(const UUID &r) const
    {
        // memcmp relies on these structures not having padding
        // static assert was done for this struct; sanity check also that sizes match
        if (sizeof(r) != sizeof(UUID)) {
            return false;
        }
        return memcmp(this, &r, sizeof(UUID)) == 0;
    }
};

static_assert(sizeof(struct UUID) ==
              sizeof(uint32_t) +
              sizeof(uint16_t) +
              sizeof(uint16_t) +
              sizeof(uint16_t) +
              sizeof(uint8_t) * 6,
              "Structure shall not have padding");

/** @brief A class to prevent drivers being copied and duplicating state.
 */
class NonCopyable
{
protected:
   NonCopyable() {}

private:
    NonCopyable(NonCopyable& other);
    NonCopyable& operator=(NonCopyable& other);
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
    /** @brief Destructor
     */
    virtual ~Interface() = default;
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
     * @param[in] interfaceId   Unique identifier (of type \ref UUID) for the
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
    /** @brief Destructor
     */
    virtual ~IInterfaceProvider() = default;
};

} // end of namespace nvsipl
#endif //NVSIPLDEVICEINTERFACEPROVIDERINTERFACE_HPP

