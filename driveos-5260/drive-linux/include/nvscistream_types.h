/*
 * Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto. Any use, reproduction, disclosure or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 */
/**
 * @file
 *
 * @brief <b> NVIDIA Software Communications Interface (SCI) : NvSciStream </b>
 *
 * The NvSciStream library is a layer on top of NvSciBuf and NvSciSync libraries
 * to provide utilities for streaming sequences of data packets between
 * multiple application modules to support a wide variety of use cases.
 */
#ifndef NVSCISTREAM_TYPES_H
#define NVSCISTREAM_TYPES_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "nvscierror.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvsciipc.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nvsci_stream_data_types NvSciStream Data Types
 *
 * Contains a list of NvSciStream datatypes.
 *
 * @ingroup nvsci_stream
 * @{
 */

/*! \brief Handle to a block.*/
typedef uintptr_t NvSciStreamBlock;

/*! \brief NvSciStream assigned handle for a packet.*/
typedef uintptr_t NvSciStreamPacket;

/*! \brief Application assigned cookie for a
 *   NvSciStreamPacket.
 */
typedef uintptr_t NvSciStreamCookie;

/*! \brief Constant variable denoting an invalid
 *   NvSciStreamPacket.
 *
 *  \implements{19620996}
 */
static const NvSciStreamPacket NvSciStreamPacket_Invalid = 0U;

/*! \brief Constant variable denoting an invalid
 *   NvSciStreamCookie.
 *
 *  \implements{19620999}
 */
static const NvSciStreamCookie NvSciStreamCookie_Invalid = 0U;

/**
 * \cond Non-doxygen comment
 * page nvsci_stream_logical_data_types NvSciStream logical data types
 * section NvSciStream logical data types
 *  Block: A block is a modular portion of a stream which resides
 *  in a single process and manages one aspect of the stream's
 *  behavior. Blocks are connected together in a tree to form
 *  arbitrary streams.
 *
 *  NvSciStream supports the following types of blocks:
 *    - Producer: Producer is a type of block responsible for
 *      generating stream data. Each stream begins with a producer
 *      block, it is also referred to as the upstream endpoint of the
 *      stream.
 *
 *    - Consumer: Consumer is a type of block responsible for
 *      processing stream data. Each stream ends with one or
 *      more consumer blocks, it is also referred to as the
 *      downstream endpoint of the stream.
 *
 *    - Pool: Pool is a type of block containing the set of
 *      packets available for use by @a Producer. NvSciStream
 *      supports only a static pool in which the number of
 *      packets managed by the pool is fixed when the pool
 *      is created.
 *
 *    - Queue: Queue is a type of block containing set of
 *      packets available for use by @a Consumer.
 *      NvSciStream supports two types of queue blocks:
 *         - Fifo: A Fifo Queue block is used when all
 *           packets must be acquired by the consumer in
 *           the order received. Packets will wait in the FIFO
 *           until the consumer acquires them.
 *         - Mailbox: Mailbox Queue block is used when the consumer
 *           should acquire the most recent data. If a new
 *           packet is inserted in the mailbox when one
 *           is already waiting, the previous one will be skipped
 *           and its buffers will immediately be returned to the
 *           Producer for reuse.
 *
 *    - Multicast: Multicast is a type of block which is responsible
 *      for connecting separate pipelines when a stream has more than
 *      one Consumer.
 *
 *    - IpcSrc - IpcSrc is a type of block which is the upstream
 *      half of an IPC block pair which allows NvSciSyncObj waiter
 *      requirements, NvSciSyncObj(s), packet element
 *      information and packets to be transmitted to or received from
 *      the downstream half of the stream which resides in another
 *      process.
 *
 *    - IpcDst - IpcDst is a type of block which is the downstream
 *      half of an IPC block pair which allows NvSciSyncObj waiter
 *      requirements, NvSciSyncObj(s), packet element
 *      information and packets to be transmitted to or received from
 *      the upstream half of the stream which resides in another process.
 *
 *  Packet: A packet represents a set of NvSciBufObjs containing stream
 *  data, each NvSciBufObj it contains is also referred to as an element
 *  of the packet.
 * \endcond
 */

/*! \brief Defines NvSciStream attributes that are queryable.
 *
 *  \implements{19621074}
 */
typedef enum {
    /*! \brief Maximum number of elements allowed per packet. */
    NvSciStreamQueryableAttrib_MaxElements          = 0x000000,
    /*! \brief Maximum number of NvSciSyncObjs allowed. */
    NvSciStreamQueryableAttrib_MaxSyncObj           = 0x000001,
    /*! \brief Maximum number of multicast outputs allowed. */
    NvSciStreamQueryableAttrib_MaxMulticastOutputs  = 0x000002
} NvSciStreamQueryableAttrib;

/*! \brief Defines access modes for the elements of a packet.
 *
 *  \implements{19621080}
 */
typedef enum {
    /*! \brief Written asynchronously, typically by hardware engine.
     *   Requires waiting for associated NvSciSyncObj(s) before
     *   reading.
     */
    NvSciStreamElementMode_Asynchronous     = 0x000000,
    /*! \brief Written synchronously, typically by CPU.
     *   Available for reading immediately.
     */
    NvSciStreamElementMode_Immediate        = 0x000001
} NvSciStreamElementMode;

/*! \brief Defines event types for the blocks.
 *
 *  \implements{19621083}
 */
typedef enum {

    /*! \brief
     *  Indicates block has complete connection to producer and consumer
     *    endpoints. The user may now proceed to perform other operations
     *    on the block.
     *
     *  Received by all blocks.
     *
     *  No NvSciStreamEvent data fields are used.
     */
    NvSciStreamEventType_Connected                  = 0x004004,

    /*! \brief
     *  Indicates portions of the stream have disconnected such that no
     *    more useful work can be done with the block. Note that this
     *    event is not always triggered immediately when any disconnect
     *    occurs. For instance:
     *    - If a consumer still has packets waiting in its queue when
     *      a producer is destroyed, it will not be informed of the
     *      disconnection until all packets are acquired
     *      by calling NvSciStreamConsumerPacketAcquire().
     *
     *  Received by all blocks.
     *
     *  No NvSciStreamEvent data fields are used.
     */
    NvSciStreamEventType_Disconnected               = 0x004005,

    /*! \brief
     *  Specifies NvSciSyncObj waiter requirements.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - syncAttrList:
     *       Provides an NvSciSyncAttrList which the recipient can combine
     *       with its own requirements to create NvSciSyncObj(s) that will
     *       be used to signal the other endpoint.
     *  - synchronousOnly:
     *       If set, NvSciSyncObj(s) cannot be used by the other side. The
     *       recipient should not create NvSciSyncObj(s), and should instead
     *       perform CPU waits before presenting or releasing packets.
     *
     *  The values in the fields may not exactly match those sent from the
     *    other endpoint. The stream may transform them as they pass through.
     *    In particular, a multicast block combines the requirements of
     *    all consumers before passing them to the producer.
     */
    NvSciStreamEventType_SyncAttr                   = 0x004010,

    /*! \brief
     *  Specifies the number of NvSciSyncObj(s) sent from the opposite endpoint.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - count:
     *      Indicates the number of NvSciSyncObj(s) that will be provided by
     *      the other endpoint. The recipient can expect this many
     *      NvSciStreamEventType_SyncDesc events to follow.
     */
    NvSciStreamEventType_SyncCount                  = 0x004011,

    /*! \brief
     *  Specifies a NvSciSyncObj sent from the opposite endpoint.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - index:
     *      Specifies an index within the array of NvSciSyncObj(s).
     *  - syncObj:
     *      Provides a NvSciSyncObj which the recipient should
     *      map into the libraries which will operate on stream data.
     */
    NvSciStreamEventType_SyncDesc                   = 0x004012,

    /*! \brief
     *  Specifies supported packet element count from producer.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - count:
     *      Indicates the number of packet element types that the producer
     *      is capable of generating. The recipient can expect this many
     *      NvSciStreamEventType_PacketAttrProducer events.
     */
    NvSciStreamEventType_PacketElementCountProducer = 0x004020,

    /*! \brief
     *  Specifies supported packet element count from consumer.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - count:
     *      Indicates the number of packet element types that the consumer
     *      wishes to receive. The recipient can expect this many
     *      NvSciStreamEventType_PacketAttrConsumer events.
     */
    NvSciStreamEventType_PacketElementCountConsumer = 0x004021,

    /*! \brief
     *  Specifies the packet element count determined by pool.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - count:
     *      Indicates the number of packet elements that the pool
     *      will provide for each packet. The recipient can expect this
     *      many NvSciStreamEventType_PacketAttr events and this many
     *      NvSciStreamEventType_PacketElement events for each
     *      packet.
     */
    NvSciStreamEventType_PacketElementCount         = 0x004022,

    /*! \brief
     *  Specifies the packet element information from producer.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - index:
     *      Index within the list of packet elements provided by the producer.
     *  - userData:
     *      A user-defined type which applications use to identify the
     *      element and allow elements provided by the producer to be
     *      matched with those desired by the consumer. At most one
     *      element of any given type can be specified.
     *  - bufAttrList:
     *      Provides an NvSciBufAttrList which the recipient can combine
     *      with its own requirements and those of the consumer to allocate
     *      NvSciBufObj(s) which all parties can use.
     *  - syncMode:
     *      Indicates whether the buffer data will be available immediately
     *      when the producer provides a packet or if the user
     *      must wait for the producer's NvSciSyncObj(s) first.
     *
     *  The values in the fields may not exactly match those sent from the
     *    producer. The stream may transform them as they pass through.
     */
    NvSciStreamEventType_PacketAttrProducer         = 0x004023,

    /*! \brief
     *  Specifies the packet element information from consumer.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - index:
     *      Index within the list of packet elements requested by the consumer.
     *  - userData:
     *      A user-defined type which applications use to identify the
     *      element and allow elements provided by the producer to be
     *      matched with those desired by the consumer. At most one
     *      element of any given type can be specified.
     *  - bufAttrList:
     *      Provides an NvSciBufAttrList which the recipient can combine
     *      with its own requirements and those of the producer to allocate
     *      buffers which all parties can use.
     *  - syncMode:
     *      Indicates whether the consumer desires the buffer data to be
     *      available immediately when the packet is acquired
     *      or if it can wait until the producer's NvSciSyncObj(s) have
     *      triggered.
     *
     *  The values in the fields may not exactly match those sent from the
     *    consumer. The stream may transform them as they pass through.
     *    In particular, multi-cast components combine the requirements of
     *    all consumers before passing them to the pool.
     */
    NvSciStreamEventType_PacketAttrConsumer         = 0x004024,

    /*! \brief
     *  Specifies the finalized packet element information from pool.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - index:
     *      Index within the list of packet elements allocated by the pool.
     *  - userData:
     *      A user-defined type which applications use to identify the
     *      element and allow elements provided by the producer to be
     *      matched with those desired by the consumer. At most one
     *      element of any given type can be specified.
     *  - bufAttrList:
     *      Provides the combined NvSciBufAttrList used by the pool to
     *      allocate the element.
     *  - syncMode:
     *      Indicates the NvSciStreamElementMode that the producer should use
     *      and the consumer should expect.
     *
     *  The values in the fields may not exactly match those sent from the
     *    pool. The stream may transform them as they pass through.
     */
    NvSciStreamEventType_PacketAttr                 = 0x004025,

    /*! \brief
     *  Specifies the new packet created by pool.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetHandle:
     *      Contains NvSciStreamPacket for the new packet.
     *      This should be used whenever the component
     *      references the packet in the future.
     */
    NvSciStreamEventType_PacketCreate               = 0x004030,

    /*! \brief
     *  Specifies new packet element.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the recipient provided
     *      to identify its data structure for the NvSciStreamPacket
     *      upon accepting it.
     *  - index:
     *      Index within the list of the packet's elements.
     *  - bufObj:
     *      Provides a NvSciBufObj which the recipient should
     *      map into the libraries which will operate on stream data.
     */
    NvSciStreamEventType_PacketElement              = 0x004031,

    /*! \brief
     *  Specifies the deleted packet.
     *
     *  Received by producer and consumer blocks.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the recipient provided
     *      to identify its data structure for the NvSciStreamPacket
     *      upon accepting it.
     */
    NvSciStreamEventType_PacketDelete               = 0x004032,

    /*! \brief
     *  Specifies the producer's acceptance status of packet.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the pool provided to identify
     *      its data structure for the packet upon creating it.
     *  - error:
     *      An error code indicating whether the producer was able
     *      to add the new packet.
     */
    NvSciStreamEventType_PacketStatusProducer       = 0x004033,

    /*! \brief
     *  Specifies the consumer(s)' acceptance status of packet.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the pool provided to identify
     *      its data structure for the packet upon creating it.
     *  - error:
     *      An error code indicating whether the consumer was able
     *      to add the new packet.
     */
    NvSciStreamEventType_PacketStatusConsumer       = 0x004034,

    /*! \brief
     *  Specifies the producer's acceptance status of packet element.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the pool provided to identify
     *      its data structure for the packet upon creating it.
     *  - index:
     *      Index of packet element for which status is provided.
     *  - error:
     *      An error code indicating whether the producer was able
     *      to map in the NvSciBufObj of the packet element.
     */
    NvSciStreamEventType_ElementStatusProducer      = 0x004035,

    /*! \brief
     *  Specifies the consumer(s)' acceptance status of packet element.
     *
     *  Received by pool block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - packetCookie:
     *      Contains the NvSciStreamCookie which the pool provided to identify
     *      its data structure for the packet upon creating it.
     *  - index:
     *      Index of packet element for which status is provided.
     *  - error:
     *      An error code indicating whether the consumer was able
     *      to map in the NvSciBufObj of the packet element.
     */
    NvSciStreamEventType_ElementStatusConsumer      = 0x004036,

    /*! \brief
     *  Specifies a packet is available for reuse or acquire.
     *
     *  Received by producer and consumer blocks.
     *
     *  No NvSciStreamEvent data fields are used.
     */
    NvSciStreamEventType_PacketReady                = 0x004040,

    /*! \brief
     *  Indicates a failure not directly triggered by user action.
     *
     *  Received by any block.
     *
     *  The following NvSciStreamEvent fields will be set:
     *  - error:
     *      An error code providing information about what went wrong.
     */
    NvSciStreamEventType_Error                      = 0x0040FF

} NvSciStreamEventType;

/*! \brief Describes an event triggered by the blocks.
 *
 *  \implements{19621089}
 */
typedef struct {
    /*! \brief Holds the type of event. */
    NvSciStreamEventType type;

    /*! \brief
     *   Used with events that specify NvSciSyncObj waiter requirements:
     *     NvSciStreamEventType_SyncAttr
     *
     *   For other events, or if not needed, will be set to NULL.
     *
     *   If set, provides an NvSciSyncAttrList of which the recipient
     *   becomes the owner. It should free the NvSciSyncAttrList when
     *   it is no longer required.
     */
    NvSciSyncAttrList syncAttrList;

    /*! \brief
     *   Used with events that specify packet element information:
     *     NvSciStreamEventType_PacketAttr,
     *     NvSciStreamEventType_PacketAttrProducer,
     *     NvSciStreamEventType_PacketAttrConsumer
     *
     *   For other events, will be set to NULL.
     *
     *   If set, provides an NvSciBufAttrList of which the recipient
     *   becomes the owner. It should free the NvSciBufAttrList when
     *   it is no longer required.
     */
    NvSciBufAttrList bufAttrList;

    /*! \brief
     *   Used with events that provide a NvSciSyncObj:
     *     NvSciStreamEventType_SyncDesc
     *
     *   For other events, will be set to NULL.
     *
     *   If set, provides a NvSciSyncObj of which the recipient
     *   becomes the owner. It should free the NvSciSyncObj when it is no
     *   longer required.
     */
    NvSciSyncObj syncObj;

    /*! \brief
     *   Used with events that provide a NvSciBufObj:
     *     NvSciStreamEventType_PacketElement
     *
     *   For other events, will be set to NULL.
     *
     *   If set, provides a NvSciBufObj of which the recipient
     *   becomes the owner. It should free the NvSciBufObj when it is no
     *   longer required.
     */
    NvSciBufObj bufObj;

    /*! \brief
     *   Used with events that return a NvSciStreamPacket:
     *     NvSciStreamEventType_PacketCreate
     *
     *   For other events, will be set to NvSciStreamPacket_Invalid.
     */
    NvSciStreamPacket packetHandle;

    /*! \brief
     *   Used with events that indicate a packet operation:
     *     NvSciStreamEventType_PacketDelete,
     *     NvSciStreamEventType_PacketElement,
     *     NvSciStreamEventType_PacketStatusProducer,
     *     NvSciStreamEventType_PacketStatusConsumer,
     *     NvSciStreamEventType_ElementStatusProducer,
     *     NvSciStreamEventType_ElementStatusConsumer
     *
     *   For other events, will be set to NvSciStreamCookie_Invalid.
     *
     *   Provides the component's cookie identifying the packet.
     */
    NvSciStreamCookie packetCookie;

    /*! \brief
     *   Used with events that require a count:
     *     NvSciStreamEventType_SyncCount,
     *     NvSciStreamEventType_PacketElementCount,
     *     NvSciStreamEventType_PacketElementCountProducer,
     *     NvSciStreamEventType_PacketElementCountConsumer
     *
     *   For other events, will be set to 0.
     *
     *   Indicates a number of items for this or subsequent events.
     */
    uint32_t count;

    /*! \brief
     *   Used with events that require an index:
     *     NvSciStreamEventType_SyncDesc,
     *     NvSciStreamEventType_PacketAttr,
     *     NvSciStreamEventType_PacketElement,
     *     NvSciStreamEventType_ElementStatusProducer,
     *     NvSciStreamEventType_ElementStatusConsumer
     *
     *   For other events, will be set to 0.
     *
     *   Indicates position of item to which the event applies within a list.
     */
    uint32_t index;

    /*! \brief
     *   Used with events that return an error:
     *     NvSciStreamEventType_PacketStatusProducer,
     *     NvSciStreamEventType_PacketStatusConsumer,
     *     NvSciStreamEventType_ElementStatusProducer,
     *     NvSciStreamEventType_ElementStatusConsumer
     *
     *   For other events, will be set to NvSciError_Success.
     *
     *   Indicates the status of an operation.
     */
    NvSciError error;

    /*! \brief
     *   Used with events that require a user-defined data field:
     *     NvSciStreamEventType_PacketAttr,
     *     NvSciStreamEventType_PacketAttrProducer,
     *     NvSciStreamEventType_PacketAttrConsumer
     *
     *   For other events, will be set to 0.
     *
     *   A value provided by application and passed through the stream
     *     without interpretation.
     */
    uint32_t userData;

    /*! \brief
     *   Used with events that specify NvSciSynObj waiter requirements:
     *     NvSciStreamEventType_SyncAttr
     *
     *   For other events, will be set to false.
     *
     *   Indicates endpoint cannot process NvSciSynObj(s) of any kind and
     *   data must be published synchronously. (This case is not common.)
     */
    bool synchronousOnly;

    /*! \brief
      *  Used with events that specify a NvSciStreamElementMode:
      *    NvSciStreamEventType_PacketAttr,
      *    NvSciStreamEventType_PacketAttrProducer,
      *    NvSciStreamEventType_PacketAttrConsumer
      *
      *  For other events, defaults to NvSciStreamElementMode_Asynchronous,
      *  and can be ignored.
      *
      *  Indicates whether data requires synchronization before using.
      */
    NvSciStreamElementMode syncMode;

} NvSciStreamEvent;


/*!
 * The following data structures are no longer used by any interfaces
 *   and are deprecated. They will be removed if no applications depend
 *   on them.
 */

typedef struct {
    uint32_t index;
    uint32_t type;
    NvSciStreamElementMode mode;
    NvSciBufAttrList bufAttr;
} NvSciStreamElementAttr;

typedef struct {
    uint32_t index;
    NvSciStreamPacket handle;
    NvSciBufObj buffer;
} NvSciStreamElementDesc;

typedef struct {
    bool synchronousOnly;
    NvSciSyncAttrList waiterSyncAttr;
} NvSciStreamSyncAttr;

typedef struct {
    uint32_t index;
    NvSciSyncObj sync;
} NvSciStreamSyncDesc;

typedef struct {
    NvSciStreamCookie cookie;
    NvSciSyncFence *prefences;
} NvSciStreamPayload;

#ifdef __cplusplus
}
#endif
/** @} */
#endif /* NVSCISTREAM_TYPES_H */
