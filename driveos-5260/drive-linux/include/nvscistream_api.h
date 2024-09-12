/*
 * Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.
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
#ifndef NVSCISTREAM_API_H
#define NVSCISTREAM_API_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "nvscierror.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvsciipc.h"
#include "nvscievent.h"
#include "nvscistream_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup nvscistream_blanket_statements NvSciStream Blanket Statements
 * Generic statements applicable for NvSciStream interfaces.
 * @ingroup nvsci_stream
 * @{
 */

/**
 * \page nvscistream_page_blanket_statements NvSciStream blanket statements
 * \section nvscistream_input_parameters Input parameters
 * - NvSciStreamPacket passed as input parameter to NvSciStreamPoolPacketInsertBuffer()
 *   or NvSciStreamPoolPacketDelete() API is valid if it is returned from a successful call to
 *   NvSciStreamPoolPacketCreate() API and has not been deleted using
 *   NvSciStreamPoolPacketDelete().
 * - NvSciStreamPacket passed as input parameter to NvSciStreamBlockPacketAccept() or
 *   NvSciStreamBlockElementAccept() or NvSciStreamProducerPacketPresent() or
 *   NvSciStreamConsumerPacketRelease() API is valid if it was received earlier through a
 *   NvSciStreamEventType_PacketCreate event and NvSciStreamEventType_PacketDelete
 *   event for the same NvSciStreamPacket is not yet received.
 * - NvSciStreamBlock passed as input parameter to an API is valid if it is returned from
 *   a successful call to any one of the block create APIs and has not yet been destroyed
 *   using NvSciStreamBlockDelete() unless otherwise stated explicitly.
 * - An array of NvSciSyncFence(s) passed as an input parameter to an API is valid if it is not
 *   NULL and each entry in the array holds a valid NvSciSyncFence. The NvSciSyncFence(s)
 *   in the array should be in the order according to the indices specified for the NvSciSyncObj(s)
 *   during NvSciStreamBlockSyncObject() API. That is, if the NvSciSyncObj is set with index == 1,
 *   its NvSciSyncFence should be the 2nd element in the array. The array should be empty
 *   if the synchronousOnly flag received through NvSciStreamEventType_SyncAttr from
 *   other endpoint is true.
 * - NvSciIpcEndpoint passed as input parameter to an API is valid if it is obtained
 *   from successful call to NvSciIpcOpenEndpoint() and has not yet been freed using
 *   NvSciIpcCloseEndpoint().
 * - Pointer to NvSciEventService passed as input parameter to an API is valid if the
 *   NvSciEventService instance is obtained from successful call to
 *   NvSciEventLoopServiceCreate() and has not yet been freed using NvSciEventService::Delete().
 * - The input parameter for which the valid value is not stated in the interface specification and not
 *   covered by any of the blanket statements, it is considered that the entire range of the parameter
 *   type is valid.
 *
 * \section nvscistream_out_params Output parameters
 * - Output parameters are passed by reference through pointers. Also, since a
 *   null pointer cannot be used to convey an output parameter, API functions return
 *   an error code if a null pointer is supplied for a required output parameter unless
 *   otherwise stated explicitly. Output parameter is valid only if error code returned
 *   by an API function is NvSciError_Success unless otherwise stated explicitly.
 *
 * \section nvscistream_return_values Return values
 * - Any initialization API that allocates memory (for example, Block creation APIs) to
 *   store information provided by the caller, will return NvSciError_InsufficientMemory if the
 *   allocation fails unless otherwise stated explicitly.
 * \if (TIER3_SWAD || TIER4_SWAD || TIER4_SWUD)
 * - Any API function which takes a NvSciStreamBlock as input will panic if the NvSciStreamBlock
 *   is invalid.
 * - Any API function which takes a NvSciStreamPacket as input will panic if the NvSciStreamPacket
 *   is invalid unless otherwise stated explicitly.
 * \else
 * - Any API function which takes a NvSciStreamBlock as input will panic in safety build or
 *   return NvSciError_BadParameter in standard build if the NvSciStreamBlock is invalid.
 * - Any API function which takes a NvSciStreamPacket as input will panic in safety build or
 *   return NvSciError_BadParameter in standard build if the NvSciStreamPacket is invalid.
 * \endif
 * - Any element level interface which operates on a block other than the interfaces for
 *   block creation, NvSciStreamBlockConnect(), NvSciStreamBlockEventQuery(), and
 *   NvSciStreamBlockEventServiceSetup() will return NvSciError_StreamNotConnected error
 *   code if the producer block in the stream is not connected to every consumer block in
 *   the stream or the NvSciStreamEventType_Connected event from the block is not yet queried
 *   by the application by calling NvSciStreamBlockEventQuery() interface.
 * - If an API function is invoked with parameters such that two or more failure modes are
 *   possible, it is guaranteed that one of those failure modes will occur, but it is not
 *   guaranteed which.
 *
 * \section nvscistream_concurrency Concurrency
 * - Unless otherwise stated explicitly in the interface specifications,
 *   any two or more NvSciStream functions can be called concurrently
 *   without any side effects. The effect will be as if the functions were
 *   executed sequentially. The order of execution is not guaranteed.
 */

/**
 * @}
 */

/**
 * @defgroup nvsci_stream Streaming APIs
 *
 * The NvSciStream library is a layer on top of NvSciBuf and NvSciSync libraries
 * that provides utilities for streaming sequences of data packets between
 * multiple application modules to support a wide variety of use cases.
 *
 * @ingroup nvsci_group_stream
 * @{
 */

 /**
 * @defgroup nvsci_stream_apis NvSciStream APIs
 *
 * Methods to setup and stream sequences of data packets.
 *
 * @ingroup nvsci_stream
 * @{
 */

/*!
 * \brief Establishes connection between two blocks referenced
 *   by the given NvSciStreamBlock(s).
 *
 * - Connects an available output of one block with an available
 *   input of another block.
 *
 * - Each input and output can only have one connection. A stream is operational
 *   when all inputs and outputs of all blocks in the stream have a connection.
 *
 * <b>Preconditions</b>
 * - The upstream block has an available output connection.
 * - The downstream block has an available input connection.
 *
 * <b>Actions</b>
 * - Establish a connection between the two blocks.
 *
 * <b>Postconditions</b>
 * - When a block has a connection to producer as well as connection to
 *   every consumer(s) in the stream, it will receive a NvSciStreamEventType_Connected
 *   event.
 *
 * \param[in] upstream NvSciStreamBlock which references an upstream block.
 *   Valid value: A valid NvSciStreamBlock which does not reference
 *   a Pool or Queue block.
 * \param[in] downstream NvSciStreamBlock which references a downstream block.
 *   Valid value: A valid NvSciStreamBlock which does not reference
 *   a Pool or Queue block.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success The connection was made successfully.
 * - ::NvSciError_InsufficientResource block referenced by @a upstream argument has
 *     no available outputs or block referenced by @a downstream argument has
 *     no available inputs.
 * - ::NvSciError_AccessDenied: block referenced by @a upstream argument
 *     or block referenced by @a downstream argument(for example, Pool or Queue)
 *     does not allow explicit connection via NvSciStreamBlockConnect().
 */
NvSciError NvSciStreamBlockConnect(
    NvSciStreamBlock const upstream,
    NvSciStreamBlock const downstream
);

/*! \brief Creates an instance of producer block, associates
 *     the given pool referenced by the given NvSciStreamBlock
 *     with it and returns a NvSciStreamBlock referencing the
 *     created producer block.
 *
 *  - Creates a block for the producer end of a stream. All streams
 *  require one producer block. A producer block must have a pool
 *  associated with it for managing available packets. Producer blocks have one
 *  output connection and no input connections.
 *
 *  - Once the stream is operational, this block can be used to exchange
 *  NvSciSync information with the consumer and NvSciBuf information with the pool.
 *
 *  <b>Preconditions</b>
 *  - None.
 *
 *  <b>Actions</b>
 *  - Creates a new instance of producer block.
 *  - Associates the given pool block with the producer block.
 *
 *  <b>Postconditions</b>
 *  - The block is ready to be connected to other stream blocks.
 *  - The block can be queried for NvSciStreamEvent(s).
 *
 *  \param [in] pool NvSciStreamBlock which references a pool block
 *    to be associated with the producer block.
 *  \param [out] producer NvSciStreamBlock which references a
 *    new producer block.
 *
 *  \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new producer block was set up successfully.
 *  - ::NvSciError_BadParameter If the output parameter @a producer
 *      is a null pointer or the @a pool parameter does not reference a
 *      pool block.
 *  - ::NvSciError_InsufficientResource If the pool block is already
 *      associated with another producer.
 *  - ::NvSciError_StreamInternalError The producer block cannot be
 *      initialized properly.
 */
NvSciError NvSciStreamProducerCreate(
    NvSciStreamBlock const pool,
    NvSciStreamBlock *const producer
);

/*! \brief Creates an instance of consumer block, associates
 *     the given queue block referenced by the given NvSciStreamBlock
 *     with it and returns a NvSciStreamBlock referencing the created
 *     consumer block.
 *
 *  - Creates a block for the consumer end of a stream. All streams
 *  require at least one consumer block. A consumer block must have
 *  a queue associated with it for managing pending packets. Consumer blocks have
 *  one input connection and no output connections.
 *
 *  - Once the stream is operational, this block can be used to exchange
 *  NvSciSync information with the producer and NvSciBuf information with the pool.
 *
 *  <b>Preconditions</b>
 *  - None.
 *
 *  <b>Actions</b>
 *  - Creates a new instance of consumer block.
 *  - Associates the given queue block with the consumer block.
 *
 *  <b>Postconditions</b>
 *  - The block is ready to be connected to other stream blocks.
 *  - The block can be queried for NvSciStreamEvent(s).
 *
 *  \param [in] queue NvSciStreamBlock which references a queue
 *    block to be associated with the consumer block.
 *  \param [out] consumer NvSciStreamBlock which references a
 *    new consumer block.
 *
 *  \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new consumer block was set up successfully.
 *  - ::NvSciError_BadParameter If the output parameter @a consumer
 *      is a null pointer or the @a queue parameter does not reference a
 *      queue block.
 *  - ::NvSciError_InsufficientResource If the queue block is already bound to
 *      another consumer.
 *  - ::NvSciError_StreamInternalError The consumer block cannot be
 *      initialized properly.
 */
NvSciError NvSciStreamConsumerCreate(
    NvSciStreamBlock const queue,
    NvSciStreamBlock *const consumer
);

/*!
 * \brief Creates an instance of static pool block and returns a
 *   NvSciStreamBlock referencing the created pool block.
 *
 * - Creates a block for management of a stream's packet pool. Every producer
 * must be associated with a pool block, provided at the producer's creation.
 *
 * - A static pool has a fixed number of packets which must be created and accepted
 * before the first free packet is acquired by the producer block.
 *
 * - Once the stream is operational and the application has determined
 * the packet requirements(packet element count and packet element buffer attributes),
 * this block can be used to register NvSciBufObj(s) to each packet.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Allocates data structures to describe packets.
 * - Initializes queue of available packets.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 * - The block can be queried for NvSciStreamEvent(s).
 *
 * \param[in] numPackets Number of packets.
 * \param[out] pool NvSciStreamBlock which references a
 *   new pool block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new pool block was set up successfully.
 *  - ::NvSciError_BadParameter The output parameter @a pool is a null pointer.
 *  - ::NvSciError_StreamInternalError The pool block cannot be initialized properly.
 */
NvSciError NvSciStreamStaticPoolCreate(
    uint32_t const numPackets,
    NvSciStreamBlock *const pool
);

/*!
 * \brief Creates an instance of mailbox queue block and returns a
 *   NvSciStreamBlock referencing the created mailbox queue block.
 *
 * - Creates a block for managing the packets that are ready to be
 * acquired by the consumer. This is one of the two available types of queue
 * block. Every consumer must be associated with a queue block, provided at
 * the time the consumer is created.
 *
 * - A mailbox queue holds a single packet. If a new packet arrives, the packet
 * currently being held is replaced and returned to the pool for reuse.
 *
 * - This type of queue is intended for consumer applications
 * which don't need to process every packet and always wish to have the
 * latest input available.
 *
 * - Once connected, the application does not directly interact with this block.
 * The consumer block will communicate with the queue block to obtain new packets.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Initialize a queue to hold a single packet for acquire.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 *
 * \param[out] queue NvSciStreamBlock which references a
 *   new mailbox queue block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new mailbox queue block was set up successfully.
 *  - ::NvSciError_BadParameter The output parameter @a queue is a null pointer.
 *  - ::NvSciError_StreamInternalError The mailbox queue block cannot be initialized properly.
 */
NvSciError NvSciStreamMailboxQueueCreate(
    NvSciStreamBlock *const queue
);

/*!
 * \brief Creates an instance of FIFO queue block and returns a
 *   NvSciStreamBlock referencing the created FIFO queue block.
 *
 * - Creates a block for tracking the list of packets available to be
 * acquired by the consumer. This is one of the two available types of queue block.
 * Every consumer must be associated with a queue block, provided at the time the
 * consumer is created.
 *
 * - A FIFO queue can hold up to the complete set of packets created,
 * which will be acquired in the order they were presented.
 *
 * - This type of queue is intended for consumer applications which require
 *   processing of every packet that is presented.
 *
 * - Once connected, the application does not directly interact with this block.
 * The consumer block will communicate with it to obtain new packets.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Initialize a queue to manage waiting packets.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 *
 * \param[out] queue NvSciStreamBlock which references a
 *   new FIFO queue block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new FIFO queue block was set up successfully.
 *  - ::NvSciError_BadParameter The output parameter @a queue is a null pointer.
 *  - ::NvSciError_StreamInternalError The FIFO queue block cannot be initialized properly.
 */
NvSciError NvSciStreamFifoQueueCreate(
    NvSciStreamBlock *const queue
);

/*!
 * \brief Creates an instance of multicast block and returns a
 *   NvSciStreamBlock referencing the created multicast block.
 *
 * - Creates a block allowing for one input and one or more outputs.
 *
 * - Multicast block broadcasts messages sent from upstream to all of the
 * downstream blocks.
 *
 * - Mulitcast block aggregates messages of the same type from downstream into
 * a single message before sending it upstream.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Initializes a block that takes one input and one or more outputs.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 *
 * \param[in] outputCount Number of output blocks that will be connected.
 *   Valid value: 1 to NvSciStreamQueryableAttrib_MaxMulticastOutputs
 *   attribute value queried by successful call to NvSciStreamAttributeQuery()
 *   API.
 * \param[out] multicast NvSciStreamBlock which references a
 *   new multicast block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new multicast block was set up successfully.
 *  - ::NvSciError_BadParameter The output parameter @a multicast is a null
 *     pointer or @a outputCount is larger than the number allowed.
 *  - ::NvSciError_StreamInternalError The multicast block cannot be
 *    initialized properly.
 */
NvSciError NvSciStreamMulticastCreate(
    uint32_t const outputCount,
    NvSciStreamBlock *const multicast
);

/*!
 * \brief Creates an instance of IpcSrc block and returns a
 *   NvSciStreamBlock referencing the created IpcSrc block.
 *
 * - Creates the upstream half of an IPC block pair which allows
 * stream information to be transmitted between processes. If producer and
 * consumer(s) present in the same process, then there is no need for
 * IPC block pair.
 *
 * - IpcSrc block has one input connection and no output connection.
 *
 * - An IpcSrc block connects to downstream through the NvSciIpcEndpoint
 * used to create the block.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Establishes connection through NvSciIpcEndpoint.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 *
 * \param[in] ipcEndpoint NvSciIpcEndpoint handle.
 * \param[in] syncModule NvSciSyncModule that is used to import a
 *        NvSciSyncAttrList across an IPC boundary.  This must be same
 *        module that was used to create NvSciSyncAttrList
 *        when specifying the NvSciSyncObj waiter requirements.
 * \param[in] bufModule NvSciBufModule that is used to import a
 *        NvSciBufAttrList across an IPC boundary. This must be same
 *        module that was used to create NvSciBufAttrList
 *        when specifying the packet element information.
 * \param[out] ipc NvSciStreamBlock which references a
 *   new IpcSrc block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new IpcSrc block was set up successfully.
 *  - ::NvSciError_StreamInternalError: If connection establishment through
 *      @a ipcEndpoint fails or IpcSrc block cannot be initialized properly.
 *  - ::NvSciError_BadParameter The output parameter @a ipc is a null pointer.
 */
NvSciError NvSciStreamIpcSrcCreate(
    NvSciIpcEndpoint const ipcEndpoint,
    NvSciSyncModule const syncModule,
    NvSciBufModule const bufModule,
    NvSciStreamBlock *const ipc
);

/*!
 * \brief Creates an instance of IpcDst block and returns a
 *   NvSciStreamBlock referencing the created IpcDst block.
 *
 * - Creates the downstream half of an IPC block pair which allows
 * stream information to be transmitted between processes.If producer and
 * consumer(s) present in the same process, then there is no need for
 * IPC block pair.
 *
 * - IpcDst block has one output connection and no input connection.
 *
 * - An IpcDst block connects to upstream through the NvSciIpcEndpoint used
 * to create the block.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - Establishes connection through NvSciIpcEndpoint.
 *
 * <b>Postconditions</b>
 * - The block is ready to be connected to other stream blocks.
 *
 * \param[in] ipcEndpoint NvSciIpcEndpoint handle.
 * \param[in] syncModule NvSciSyncModule that is used to import a
 *        NvSciSyncAttrList across an IPC boundary. This must be same
 *        module that was used to create NvSciSyncAttrList
 *        when specifying the NvSciSyncObj waiter requirements.
 * \param[in] bufModule NvSciBufModule that is used to import a
 *        NvSciBufAttrList across an IPC boundary. This must be same
 *        module that was used to create NvSciBufAttrList
 *        when specifying the packet element information.
 * \param[out] ipc NvSciStreamBlock which references a
 *   new IpcDst block.
 *
 * \return ::NvSciError, the completion code of this operation.
 *  - ::NvSciError_Success A new IpcDst block was set up successfully.
 *  - ::NvSciError_StreamInternalError: If connection establishment through
 *      @a ipcEndpoint fails or IpcDst block can't be initialized properly.
 *  - ::NvSciError_BadParameter The output parameter @a ipc is a null pointer.
 */
NvSciError NvSciStreamIpcDstCreate(
    NvSciIpcEndpoint const ipcEndpoint,
    NvSciSyncModule const syncModule,
    NvSciBufModule const bufModule,
    NvSciStreamBlock *const ipc
);

/*!
* \brief Creates an instance of Limiter block and returns a
*   NvSciStreamBlock referencing the created Limiter block.
*
* - Creates a block to limit the number of packets allowed to be sent
*   downstream to a consumer block.
*
* - A limiter block can be inserted anywhere in the stream between the
*   Producer and Consumer Blocks, but its primary intent is to be inserted
*   between a Multicast block and a Consumer.
*
* <b>Preconditions</b>
* - None.
*
* <b>Actions</b>
* - Creates a new instance of Limiter block.
*
* <b>Postconditions</b>
* - The block is ready to be connected to other stream blocks.
*
* \param[in] maxPackets Number of packets allowed to be sent downstream
*   to a consumer block.
* \param[out] limiter NvSciStreamBlock which references a
*   new limiter block.
*
* \return ::NvSciError, the completion code of this operation.
*  - ::NvSciError_Success A new Limiter block was set up successfully.
*  - ::NvSciError_BadParameter The output parameter @a limiter is a null
*      pointer.
*  - ::NvSciError_StreamInternalError The Limiter block can't be initialized
*      properly.
*/
NvSciError NvSciStreamLimiterCreate(
    uint32_t const maxPackets,
    NvSciStreamBlock *const limiter
);

/*!
 * \brief Queries for the next NvSciStreamEvent from block referenced by the given
 *  NvSciStreamBlock, optionally waiting when the event information is not available,
 *  then removes the event from the queue and returns the event information to the caller.
 *
 * If the block is set up to use NvSciEventService, applications should call this
 * API with zero timeout after waking up from waiting on the NvSciEventNotifier
 * obtained from NvSciStreamBlockEventServiceSetup() interface, and applications
 * should query all the events in the block after waking up. Wake up due to spurious
 * events is possible, and in that case calling this function will return no event.
 *
 * The appropriate handling of each NvSciStreamEventType is described in the
 * section for the corresponding NvSciStreamEvent structure.
 *
 * <b>Preconditions</b>
 * - The block to query has been created.
 *
 * <b>Actions</b>
 * - If the block is not set up to use NvSciEventService, wait until
 *   an NvSciStreamEvent is pending on the block or the specified timeout
 *   period is reached, whichever comes first.
 * - Retrieves the next NvSciStreamEvent (if any) from the block, filling in the
 *   NvSciStreamEvent data structure.
 *
 * <b>Postconditions</b>
 * - As defined for each NvSciStreamEventType, dequeuing an NvSciStreamEvent
 *   may allow other operations on the block to proceed.
 *
 * \param[in] block NvSciStreamBlock which references a block.
 * \param[in] timeoutUsec Timeout in microseconds (-1 to wait forever).
 * \param[out] event NvSciStreamEvent filled with corresponding event data.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success The output argument @a event is filled with the
 *     queried event data.
 * - ::NvSciError_Timeout The @a timeoutUsec period was reached before an
 *     NvSciStreamEvent became available.
 * - ::NvSciError_BadParameter The output argument @a event is null, or
 *     @a timeoutUsec is not zero if the block has been set up to use
 *     NvSciEventService.
 */
NvSciError NvSciStreamBlockEventQuery(
    NvSciStreamBlock const block,
    int64_t const timeoutUsec,
    NvSciStreamEvent *const event
);

/*!
 * \brief Sets NvSciSyncObj waiter requirements to the block referenced
 *   by the given NvSciStreamBlock.
 *
 * Used with producer and consumer to establish their waiter requirements for
 * NvSciSyncObj(s) provided by the other endpoint(s).
 *
 * <b>Preconditions</b>
 * - Block must have received appropriate NvSciStreamEventType_Connected event
 *   from the other endpoint(s).
 *
 * <b>Actions</b>
 * - A NvSciStreamEventType_SyncAttr event is sent to the other endpoint(s).
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] block NvSciStreamBlock which references a producer
 *   or consumer block.
 * \param[in] synchronousOnly Flag to indicate whether the endpoint supports
 *   NvSciSyncObj(s) or not.
 * \param[in] waitSyncAttrList Requirements for endpoint to wait for NvSciSyncObj(s).
 *   This parameter should be NULL if synchronousOnly parameter is true.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success NvSciSyncObj waiter requirements are registered successfully.
 * - ::NvSciError_BadParameter @a waitSyncAttrList is NULL and synchronousOnly
 *   flag is false.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does not
 *   reference a producer or consumer block.
 * - ::NvSciError_InvalidOperation @a synchronousOnly flag is true but
 *   @a waitSyncAttrList is not NULL.
 * - ::NvSciError_InvalidState If NvSciSyncObj waiter requirements are already
 *   sent to other endpoint.
 * - Error/panic behavior of this API includes:
 *    - Any error/panic behavior that NvSciSyncAttrListClone() or
 *      NvSciSyncAttrListIpcExportUnreconciled() can generate when
 *      @a waitSyncAttrList argument is passed to it.
 *    - Any error/panic behavior that NvSciIpcWrite() can generate
 *      when NvSciSyncAttrList descriptor for the @a waitSyncAttrList is
 *      exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockSyncRequirements(
    NvSciStreamBlock const block,
    bool const synchronousOnly,
    NvSciSyncAttrList const waitSyncAttrList
);

/*!
 * \brief Sets NvSciSyncObj count to the block referenced by the
 *   given NvSciStreamBlock.
 *
 * - Used with producer and consumer to establish the information on the number of
 *   NvSciSyncObj(s) which will be created and shared with the other endpoint(s).
 *
 * - The NvSciSyncObj count is implicitly set to one.
 *
 * - If the block provides no NvSciSyncObj, the function has to be called,
 * with count equal to zero, to advance the state of the stream and inform the
 * other endpoint(s).
 *
 * <b>Preconditions</b>
 * - Block must have received NvSciStreamEventType_SyncAttr event from the
 *   other endpoint(s).
 * - Calling the function is not necessary if the block provides only one NvSciSyncObj.
 *
 * <b>Actions</b>
 * - A NvSciStreamEventType_SyncCount event is sent to the other endpoint(s).
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] block NvSciStreamBlock which references a producer
 *   or consumer block.
 * \param[in] count NvSciSyncObj count to be sent.
 *   Valid value: 0 to NvSciStreamQueryableAttrib_MaxSyncObj
 *   attribute value queried by successful call to
 *   NvSciStreamAttributeQuery() API.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success NvSciSyncObj count is sent successfully.
 * - ::NvSciError_InvalidOperation NvSciStreamEventType_SyncAttr event
 *   from the block is not yet queried by the application, or if stream is
 *   in synchronous mode(if the synchronousOnly flag received through
 *   NvSciStreamEventType_SyncAttr from other endpoint is true), but the
 *   count is greater than zero.
 * - ::NvSciError_BadParameter the @a count exceeds the maximum allowed.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does not
 *   reference a producer or consumer block.
 * - ::NvSciError_InvalidState: If the count is already sent.
 * - Error/panic behavior of this API includes any error/panic behavior that
 *   NvSciIpcWrite() can generate when the count is exported over NvSciIpc
 *   channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockSyncObjCount(
    NvSciStreamBlock const block,
    uint32_t const count
);

/*!
 * \brief Sets NvSciSyncObj to the block referenced by the given
 *   NvSciStreamBlock.
 *
 * Used with producer and consumer to establish the created NvSciSyncObj(s)
 * used by the endpoint for signaling.
 *
 * <b>Preconditions</b>
 * - Block must have received NvSciStreamEventType_SyncAttr event from the
 *   other endpoint(s).
 *
 * <b>Actions</b>
 * - A NvSciStreamEventType_SyncDesc event is sent to the other endpoint(s).
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] block NvSciStreamBlock which references a producer
 *   or consumer block.
 * \param[in] index Index in list of NvSciSyncObj(s).
 *   Valid value: 0 to count set earlier with NvSciStreamBlockSyncObjCount() - 1
 * (only when the count set with NvSciStreamBlockSyncObjCount() is greater than 0).
 *
 * \param[in] syncObj NvSciSyncObj.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success NvSciSyncObj is sent to the other endpoint(s) successfully.
 * - ::NvSciError_BadParameter @a index exceeds the NvSciSyncObj count count set
 *   earlier with NvSciStreamBlockSyncObjCount().
 * - ::NvSciError_InvalidOperation NvSciStreamEventType_SyncAttr event
 *   from the block is not yet queried by the application, or if stream is in synchronous mode
 *   (if the synchronousOnly flag received through NvSciStreamEventType_SyncAttr
 *   from other endpoint is true).
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does not
 *   reference a producer or consumer block.
 * - Error/panic behavior of this API includes:
 *    - Any error/panic behavior that NvSciSyncObjDup() or
 *      NvSciSyncObjIpcExport() can generate when @a syncObj argument
 *      is passed to it.
 *    - Any error/panic behavior that NvSciIpcWrite() can generate
 *      when NvSciSyncObjIpcExportDescriptor for the @a syncObj is
 *      exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockSyncObject(
    NvSciStreamBlock const block,
    uint32_t const index,
    NvSciSyncObj const syncObj
);

/*!
 * \brief Sets packet element count to the block referenced by the
 *   given NvSciStreamBlock.
 *
 * - Used with the consumer to establish the number of packet elements it desires,
 * the producer to establish the number of packet elements it can provide, and
 * the pool to determine the combined number of packet elements.
 *
 * - If the block provides only one element per packet, the function does not
 *   have to be called.
 *
 * - If the block needs or provides no element, the application must make this
 * call with count equal to zero.
 *
 * - The number of packet elements cannot be changed using this function during streaming.
 *
 * <b>Preconditions</b>
 * - For producer and consumer, must have received appropriate
 *   NvSciStreamEventType_Connected event from the other endpoint(s).
 * - For pool, must have received all NvSciStreamEventType_PacketAttrProducer
 *   and NvSciStreamEventType_PacketAttrConsumer events.
 *
 * <b>Actions</b>
 * - If the argument @a block references a producer or consumer block, a
 *   NvSciStreamEventType_PacketElementCountProducer or
 *   NvSciStreamEventType_PacketElementCountConsumer event
 *   will be sent to pool.
 * - If the argument  @a block references a pool block, a
 *   NvSciStreamEventType_PacketElementCount event will be sent to both
 *   producer and consumer endpoints.
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] block NvSciStreamBlock which references a producer
 *   or consumer or pool block.
 * \param[in] count Number of elements per packet.
 *   Valid value: 0 to NvSciStreamQueryableAttrib_MaxElements
 *   attribute value queried by successful call to
 *   NvSciStreamAttributeQuery() API.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet element count is sent successfully.
 * - ::NvSciError_BadParameter @a count exceeds the maximum allowed.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does
 *   not reference a producer or consumer or pool block.
 * - ::NvSciError_InvalidState: The argument @a block references a pool block but
 *   NvSciStreamEventType_PacketAttrProducer and
 *   NvSciStreamEventType_PacketAttrConsumer events
 *   are not yet queried from pool block, or if the count is already sent.
 * - Error/panic behavior of this API includes any error/panic behavior that
 *   NvSciIpcWrite() can generate when the count is exported over NvSciIpc
 *   channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockPacketElementCount(
    NvSciStreamBlock const block,
    uint32_t const count
);

/*!
 * \brief Sets packet element information to the block referenced by
 *   the given NvSciStreamBlock.
 *
 * - Used with the consumer to establish what packet elements it desires,
 * the producer to establish what packet elements it can provide, and
 * the pool to establish the combined packet element list.
 *
 * - The packet element information cannot be
 * changed using this function during streaming.
 *
 * <b>Preconditions</b>
 * - For producer and consumer, must have received appropriate
 *   NvSciStreamEventType_Connected event from the other endpoint(s).
 * - For pool, must have received all NvSciStreamEventType_PacketAttrProducer
 *   and NvSciStreamEventType_PacketAttrConsumer events.
 *
 * <b>Actions</b>
 * - If the argument @a block references a producer or consumer block, a
 *   NvSciStreamEventType_PacketAttrProducer or
 *   NvSciStreamEventType_PacketAttrConsumer event
 *   will be sent to pool.
 * - If the argument @a block references a pool block, a
 *   NvSciStreamEventType_PacketAttr event will be sent to
 *   both producer and consumer endpoints.
 *
 * <b>Postconditions</b>
 * - For the pool, packets may now be created.
 *
 * \param[in] block NvSciStreamBlock which references a producer
 *   or consumer or pool block.
 * \param[in] index Index of element within list of packet elements.
 *   Valid value: 0 to count set earlier with NvSciStreamBlockPacketElementCount() - 1.
 * \param[in] type User-defined type to identify element.
 * \param[in] syncMode Preferred NvSciStreamElementMode for element data.
 * \param[in] bufAttrList NvSciBufAttrList for packet element.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet element information is sent successfully.
 * - ::NvSciError_BadParameter @a bufAttrList is NULL or @a index exceeds
 *     the maximum allowed.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does not
 *   reference a producer or consumer or pool block.
 * - ::NvSciError_InvalidState Packet element information for the @a index
 *   is already sent or the argument @a block references a pool block but
 *   NvSciStreamEventType_PacketAttrProducer and
 *   NvSciStreamEventType_PacketAttrConsumer events
 *   are not yet queried from pool block.
 * - Error/panic behavior of this API includes:
 *    - Any error/panic behavior that NvSciBufAttrListClone() or
 *      NvSciBufAttrListIpcExportUnreconciled() or NvSciBufAttrListIpcExportReconciled()
 *      or NvSciBufAttrListAppendUnreconciled() can generate when @a bufAttrList
 *      argument is passed to it.
 *    - Any error/panic behavior that NvSciIpcWrite() can generate
 *      when NvSciBufAttrList descriptor for the @a bufAttrList is
 *      exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockPacketAttr(
    NvSciStreamBlock const block,
    uint32_t const index,
    uint32_t const type,
    NvSciStreamElementMode const syncMode,
    NvSciBufAttrList const bufAttrList
);

/*!
 * \brief Creates a new packet and adds it to the pool block
 *   referenced by the given NvSciStreamBlock, associates the
 *   given NvSciStreamCookie with the packet and returns a
 *   NvSciStreamPacket which references the created packet.
 *
 * <b>Preconditions</b>
 * - All packet element information must have been specified for the pool.
 * - For static pool, the number of packets already created has not reached the
 *   number of packets which was set when the static pool was created.
 *
 * <b>Actions</b>
 * - A new NvSciStreamPacket is assigned to the created packet
 *   and returned to the caller.
 * - A NvSciStreamEventType_PacketCreate event will be sent to
 *   the producer and the consumer endpoints.
 *
 * <b>Postconditions</b>
 * - The application can register NvSciBufObj(s) to the created packet.
 *
 * \param[in] pool NvSciStreamBlock which references a pool block.
 * \param[in] cookie Pool's NvSciStreamCookie for the packet.
 *   Valid value: cookie != NvSciStreamCookie_Invalid.
 * \param[out] handle NvSciStreamPacket which references the
 *   created packet.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet successfully created.
 * - ::NvSciError_BadParameter @a cookie argument is either invalid or
 *     already used for other packet or @a handle argument is NULL.
 * - ::NvSciError_NotImplemented The argument @a pool is valid but
 *   it does not reference a pool block.
 * - ::NvSciError_InsufficientMemory: If unable to create a new packet instance.
 * - ::NvSciError_InvalidOperation: Pool has reached its limit with
 *    the maximum number of packets that can be created from it.
 * - Error/panic behavior of this API includes any error/panic behavior
 *   that NvSciIpcWrite() can generate when the packet create information
 *   is exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamPoolPacketCreate(
    NvSciStreamBlock const pool,
    NvSciStreamCookie const cookie,
    NvSciStreamPacket *const handle
);

/*!
 * \brief Registers an NvSciBufObj to the packet element referenced by
 *   the given index of the packet referenced by the given NvSciStreamPacket,
 *   if the packet is associated with the pool block referenced by the
 *   given NvSciStreamBlock.
 *
 * <b>Preconditions</b>
 * - The number of NvSciBufObj(s) already registered to the packet hasn't reached
 *   the NvSciBufObj count set to pool by NvSciStreamBlockPacketElementCount().
 *
 * <b>Actions</b>
 * - The NvSciBufObj is registered to the given packet element.
 *
 * <b>Postconditions</b>
 * - A NvSciStreamEventType_PacketElement event (one per packet element) will be
 *   sent to producer and consumer endpoints.
 *
 * \param[in] pool NvSciStreamBlock which references a pool block.
 * \param[in] handle NvSciStreamPacket which references the actual packet.
 * \param[in] index Index of element within packet.
 *   Valid value: 0 to count set to pool earlier with NvSciStreamBlockPacketElementCount() - 1.
 * \param[in] bufObj NvSciBufObj to be registered.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success @a bufObj has been successfully registered.
 * - ::NvSciError_BadParameter @a bufObj is NULL or @a index exceeds
 *   the maximum allowed.
 * - ::NvSciError_NotImplemented The argument @a pool is valid but it does not
 *   reference a pool block.
 * - ::NvSciError_InvalidState: If NvSciBufObj for the same @a index is already registered.
 * - Error/panic behavior of this API includes:
 *    - Any error/panic behavior that NvSciBufObjDup() or
 *      NvSciBufObjIpcExport() can generate when @a bufObj argument
 *      is passed to it.
 *    - Any error/panic behavior that NvSciIpcWrite() can generate
 *      when NvSciBufObjIpcExportDescriptor for the @a bufObj is
 *      exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamPoolPacketInsertBuffer(
    NvSciStreamBlock const pool,
    NvSciStreamPacket const handle,
    uint32_t const index,
    NvSciBufObj const bufObj
);

/*!
 * \brief Removes a packet referenced by the given NvSciStreamPacket from
 *   the pool block referenced by the given NvSciStreamBlock.
 *
 * If the packet is currently in the pool, it is removed right away. Otherwise
 * this is deferred until the packet returns to the pool.
 *
 * <b>Preconditions</b>
 * - Specified packet must exist.
 *
 * <b>Actions</b>
 * - If the pool holds the specified packet or when it is returned to the
 *   pool, the pool releases resources associated with it and sends a
 *   NvSciStreamEventType_PacketDelete event to the producer and
 *   consumer endpoint(s).
 *
 * <b>Postconditions</b>
 * - The packet may no longer be used for pool operations.
 *
 * \param[in] pool NvSciStreamBlock which references a pool block.
 * \param[in] handle NvSciStreamPacket which references the actual
 *   packet.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet successfully removed.
 * - ::NvSciError_NotImplemented The argument @a pool is valid but it does not
 *   reference a pool block.
 * - ::NvSciError_BadParameter: If the Packet is already marked
 *   for removal.
 * - Error/panic behavior of this API includes any error/panic behavior that
 *   NvSciIpcWrite() can generate when the packet delete information is
 *   exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamPoolPacketDelete(
    NvSciStreamBlock const pool,
    NvSciStreamPacket const handle
);

/*!
 * \brief Accepts a packet referenced by the given NvSciStreamPacket.
 *
 * - Used with producer and consumer to establish their acceptance status
 * of the packet to pool block.
 *
 * - Upon receiving a NvSciStreamEventType_PacketCreate event, the
 * producer and consumer should set up their own internal data structures
 * for the packet and assign a NvSciStreamCookie which they will use to
 * look up the data structure. Afterwards, they should then call this function.
 *
 * - If the client setup is successful, the error value should be set to
 * NvSciError_Success, and the NvSciStreamCookie assigned to the packet
 * should be provided. NvSciStream will associate the NvSciStreamCookie with
 * the packet and use it for all subsequent events related to the packet.
 *
 * - If the client setup is not successful, an error value should be provided to
 * indicate what went wrong. The cookie is ignored.
 *
 * <b>Preconditions</b>
 * - The packet handle has been received in a previous
 *   NvSciStreamEventType_PacketCreate event, and
 *   has not yet been accepted.
 *
 * <b>Actions</b>
 * - Associates the NvSciStreamCookie with the packet if @a err argument
 *   is NvSciError_Success.
 * - A NvSciStreamEventType_PacketStatusProducer or
 *   NvSciStreamEventType_PacketStatusConsumer event will be sent to the pool,
 *   depending on whether the argument @a block references a producer or consumer block.
 *
 * <b>Postconditions</b>
 * - If successful, the NvSciStreamEventType_PacketElement event(s)
 *   may now be received.
 *
 * \param[in] block NvSciStreamBlock which references a producer or
 *   consumer block.
 * \param[in] handle NvSciStreamPacket which references the actual packet.
 * \param[in] cookie Block's NvSciStreamCookie to be associated with the
 *   NvSciStreamPacket. Valid value: cookie > NvSciStreamCookie_Invalid.
 * \param[in] err Status of packet setup.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet cookie/status is successfully reported.
 * - ::NvSciError_BadParameter @a cookie is NvSciStreamCookie_Invalid
 *   but the @a err is NvSciError_Success, or @a cookie is valid but the
 *   @a err is not NvSciError_Success.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does
 *   not reference a producer or consumer block.
 * - ::NvSciError_InvalidState: If status for the packet is sent already.
 * - Error/panic behavior of this API includes any error/panic behavior
 *   that NvSciIpcWrite() can generate when the packet cookie and
 *   acceptance status is exported over NvSciIpc channel if one is in
 *   use by the stream.
 */
NvSciError NvSciStreamBlockPacketAccept(
    NvSciStreamBlock const block,
    NvSciStreamPacket const handle,
    NvSciStreamCookie const cookie,
    NvSciError const err
);

/*!
 * \brief Accepts a packet element referenced by the given index of
 *   the packet referenced by the given NvSciStreamPacket.
 *
 * - Used with producer and consumer to send their acceptance status
 * of the NvSciStreamPacket element to pool block.
 *
 * - Upon receiving a NvSciStreamEventType_PacketElement event, the producer
 * and consumer should map the NvSciBufObj(s) into their space and report the
 * status by calling this function.
 *
 * - If successfully mapped, the error value should be NvSciError_Success.
 * Otherwise it should be an error code indicating what failed.
 *
 * <b>Preconditions</b>
 * - The NvSciStreamPacket handle has been received in a previous
 *   NvSciStreamEventType_PacketCreate event, and the NvSciBufObj
 *   for an indexed element has been received in a previous
 *   NvSciStreamEventType_PacketElement event.
 *
 * <b>Actions</b>
 * - A NvSciStreamEventType_ElementStatusProducer or
 *   NvSciStreamEventType_ElementStatusConsumer event will be sent to the pool,
 *   depending on whether the argument @a block references a producer or consumer block.
 *
 * <b>Postconditions</b>
 * - If successful, the NvSciBufObj(s) for the NvSciStreamPacket elements may now be used
 *   for producer and consumer processing.
 *
 * \param[in] block NvSciStreamBlock which references a producer or
 *   consumer block.
 * \param[in] handle NvSciStreamPacket which references the actual packet.
 * \param[in] index Index of the element within the packet.
 *   Valid value: 0 to packet element count received from pool
 *   through NvSciStreamEventType_PacketElementCount event.
 * \param[in] err Status of mapping operation.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success: Packet element status is successfully reported.
 * - ::NvSciError_BadParameter: @a index is invalid.
 * - ::NvSciError_NotImplemented The argument @a block is valid but it does not
 *   reference a producer or consumer block.
 * - ::NvSciError_InvalidState:  If status for the same packet element is sent already.
 * - Error/panic behavior of this API includes any error/panic behavior
 *   that NvSciIpcWrite() can generate when the element acceptance status
 *   is exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamBlockElementAccept(
    NvSciStreamBlock const block,
    NvSciStreamPacket const handle,
    uint32_t const index,
    NvSciError const err
);

/*!
 * \brief Instructs the producer referenced by the given
 *   NvSciStreamBlock to get a packet from the pool.
 *
 * - If a packet is available for producer processing, this function will retrieve
 * it from the pool and assign it to the producer.
 *
 * - The producer may hold multiple packets,
 * and is not required to present them in the order they were obtained.
 *
 * <b>Preconditions</b>
 * - The producer block must have received the NvSciStreamEventType_PacketReady
 * event from pool for processing the next available packet.
 * - All packets have been accepted by the producer and the consumer(s).
 *
 * <b>Actions</b>
 * - Retrieves an available packet for producer processing and returns it to the caller.
 * - Disassociates the packet from the pool block.
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] producer NvSciStreamBlock which references a producer block.
 * \param[out] cookie NvSciStreamCookie identifying the packet.
 * \param[out] prefences Pointer to an array of NvSciSyncFence(s) to wait for before
 *   using the packet. Valid value: Must be at least large enough to hold one
 *   NvSciSyncFence for each NvSciSyncObj created by the consumer. If the
 *   NvSciSyncObj count received through NvSciStreamEventType_SyncCount
 *   event from consumer is zero, it can be NULL.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet successfully retrieved.
 * - ::NvSciError_BadParameter If @a cookie is NULL
 * - ::NvSciError_BadParameter If @a prefences is NULL and NvSciSyncObj count
 *    received through NvSciStreamEventType_SyncCount event from consumer
 *    block(s) is greater than zero.
 * - ::NvSciError_NotImplemented The argument @a producer is valid but it does not
 *   reference a producer block.
 * - ::NvSciError_NoStreamPacket No packet available with the pool.
 */
NvSciError NvSciStreamProducerPacketGet(
    NvSciStreamBlock const producer,
    NvSciStreamCookie *const cookie,
    NvSciSyncFence *const prefences
);

/*!
 * \brief Instructs the producer referenced by the given NvSciStreamBlock to
 *   insert the packet referenced by the given NvSciStreamPacket and
 *   the associated NvSciSyncFence array to
 *   every queue in the stream configuration for consumer processing.
 *
 * <b>Preconditions</b>
 * - The NvSciStreamPacket must be associated with the producer block.
 *
 * <b>Actions</b>
 * - The NvSciStreamPacket is sent to the queue(s) where it will be held until
 * acquired by the consumer(s) or returned without acquisition.
 * - In case of FIFO queue, a NvSciStreamEventType_PacketReady will be sent
 *  to each consumer block.
 * - In case of Mailbox queue, If a NvSciStreamPacket is not already in the queue
 *  then a NvSciStreamEventType_PacketReady will be sent to each consumer block.
 * In both cases, the NvSciStreamPacket is disassociated from producer block.
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] producer NvSciStreamBlock which references a producer block.
 * \param[in] handle NvSciStreamPacket which references the actual packet.
 * \param[in] postfences A pointer to array of NvSciSyncFences associated
 *   with the packet.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet successfully presented.
 * - ::NvSciError_BadParameter @a postfences is NULL.
 * - ::NvSciError_NotImplemented The argument @a producer is valid but it does not
 *   reference a producer block.
 * - ::NvSciError_InvalidOperation If packet is not currently held by the application.
 * - Error/panic behavior of this API includes:
 *   - Any error/panic behavior that NvSciSyncFenceDup() or
 *     NvSciSyncIpcExportFence() can generate when an entry of the
 *     postfences array is passed to it.
 *   - Any error/panic behavior that NvSciIpcWrite() can generate
 *     when NvSciSyncFence descriptor for an entry of the postfences
 *     array is exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamProducerPacketPresent(
    NvSciStreamBlock const producer,
    NvSciStreamPacket const handle,
    NvSciSyncFence const *const postfences
);

/*!
 * \brief Instructs the consumer referenced by the given
 *   NvSciStreamBlock to get a ready packet from the queue.
 *
 * - If a packet is ready for consumer processing, this function will retrieve
 * it from the queue and assign it to the consumer.
 *
 * - The consumer may hold multiple packets, and is not required to
 * return them in the order they were obtained.
 *
 * <b>Preconditions</b>
 * - The consumer block must have received NvSciStreamEventType_PacketReady event from
 * queue block for processing the ready packet.
 * - All packets have been accepted by the producer and the consumer(s).
 * - Packet must be associated with queue block.
 *
 * <b>Actions</b>
 * - Retrieves a packet ready for consumer processing and returns it to the caller.
 * - Disassociates the packet from the queue.
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] consumer NvSciStreamBlock which references a consumer block.
 * \param[out] cookie NvSciStreamCookie identifying the packet.
 * \param[out] prefences Pointer to an array of NvSciSyncFence(s) to wait for before
 *   using the packet. Valid value: Must be at least large enough to hold one
 *   NvSciSyncFence for each NvSciSyncObj created by the producer. If the
 *   NvSciSyncObj count received through NvSciStreamEventType_SyncCount
 *   event from producer is zero, it can be NULL.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet was successfully acquired.
 * - ::NvSciError_BadParameter If @a cookie is NULL, or if @a prefences is
 *   NULL and NvSciSyncObj count received through NvSciStreamEventType_SyncCount
 *   event from producer block is greater than zero.
 * - ::NvSciError_NotImplemented The argument @a consumer is valid but it does not
 *   reference a consumer block.
 * - ::NvSciError_NoStreamPacket No packet is available with the queue.
 */
NvSciError NvSciStreamConsumerPacketAcquire(
    NvSciStreamBlock const consumer,
    NvSciStreamCookie *const cookie,
    NvSciSyncFence *const prefences
);

/*!
 * \brief Instructs the consumer referenced by the given NvSciStreamBlock to
 *   release the packet referenced by the given NvSciStreamPacket and
 *   the associated NvSciSyncFence array to pool block.
 *
 * <b>Preconditions</b>
 * - The NvSciStreamPacket must be associated with the consumer.
 *
 * <b>Actions</b>
 * - The NvSciStreamPacket is sent back upstream. Once released by all consumers:
 *     - It will be put in the pool where it can be obtained by the
 *       producer block.
 *     - A NvSciStreamEventType_PacketReady event will be sent to the producer block.
 * - The NvSciStreamPacket is disassociated with the consumer block.

 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] consumer NvSciStreamBlock which references a consumer block.
 * \param[in] handle NvSciStreamPacket which references the actual packet.
 * \param[in] postfences A pointer to array of NvSciSyncFences associated
 *   with the packet.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Packet was successfully released.
 * - ::NvSciError_BadParameter @a postfences is NULL.
 * - ::NvSciError_NotImplemented The argument @a consumer is valid but it does not
 *   reference a consumer block.
 * - ::NvSciError_InvalidOperation If packet is not currently held by the application.
 * - Error/panic behavior of this API includes:
 *   - Any error/panic behavior that NvSciSyncFenceDup() or
 *     NvSciSyncIpcExportFence() can generate when an entry of the
 *     postfences array is passed to it.
 *   - Any error/panic behavior that NvSciIpcWrite() can generate
 *     when NvSciSyncFence descriptor for an entry of the postfences
 *     array is exported over NvSciIpc channel if one is in use by the stream.
 */
NvSciError NvSciStreamConsumerPacketRelease(
    NvSciStreamBlock const consumer,
    NvSciStreamPacket const handle,
    NvSciSyncFence const *const postfences
);

/*!
 * \brief Schedules a block referenced by the given NvSciStreamBlock
 * for destruction, disconnecting the block if this
 * hasn't already occurred.
 *
 * - The block's handle may no longer be used for any function calls.
 *
 * - Resources associated with the block may not be freed immediately.
 *
 * - Any pending packets downstream of the destroyed block will
 * still be available for the consumer to acquire.
 *
 * - No new packets upstream of the destroyed block can be presented. Once packets
 * are released, they will be freed.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - The block is scheduled for destruction.
 * - A NvSciStreamEventType_Disconnected event is sent to all upstream
 *   and downstream blocks, if they haven't received one already.
 *
 * <b>Postconditions</b>
 * - The referenced NvSciStreamBlock is no longer valid.
 * - If there is an NvSciEventNotifier bound to the referenced
 *   NvSciStreamBlock, it is unbound.
 *
 * \param[in] block NvSciStreamBlock which references a block.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Block successfully destroyed.
 */
NvSciError NvSciStreamBlockDelete(
    NvSciStreamBlock const block
);

/*!
 * \brief Queries the value of one of the NvSciStreamQueryableAttrib.
 *
 * <b>Preconditions</b>
 * - None.
 *
 * <b>Actions</b>
 * - NvSciStream looks up the value for the given NvSciStreamQueryableAttrib
 *   and returns it.
 *
 * <b>Postconditions</b>
 * - None.
 *
 * \param[in] attr NvSciStreamQueryableAttrib to query.
 * \param[out] value The value queried.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success Query is successful.
 * - ::NvSciError_BadParameter @a attr is invalid or @a value is null.
 */
NvSciError NvSciStreamAttributeQuery(
    NvSciStreamQueryableAttrib const attr,
    int32_t *const value
);

/*!
 * Sets up the NvSciEventService on a block referenced by the given
 * NvSciStreamBlock by creating an NvSciEventNotifier to report the occurrence
 * of any events on that block. The NvSciEventNotifier is bound to the input
 * NvSciEventService and NvSciStreamBlock. Users can wait for events on the
 * block using the NvSciEventService API and then retrieve event details
 * using NvSciStreamBlockEventQuery(). Binding one or more blocks in a stream
 * to an NvSciEventService is optional. If not bound to an NvSciEventService,
 * users may instead wait for events on a block by specifying a non-zero
 * timeout in NvSciStreamBlockEventQuery(). If blocks in the same stream within
 * the same process are bound to different NvSciEventService, behavior is
 * undefined. The user is responsible for destroying the NvSciEventNotifier when
 * it's no longer needed.
 *
 * <b>Preconditions</b>
 * - After the input block is created, before calling this API, no NvSciStream
 *   API shall be called on the block.
 *
 * <b>Actions</b>
 * - Sets up the input block to use the input NvSciEventService for event
 * signaling.
 * - Creates an NvSciEventNotifier object and returns the pointer to the object
 * via @a eventNotifier.
 *
 * <b>Postconditions</b>
 * - NvSciStreamBlockEventQuery() calls with non-zero timeout on the block will
 * return error.
 *
 * \param[in] block NvSciStreamBlock which references a block.
 * \param[in] eventService Pointer to a NvSciEventService object.
 * \param[out] eventNotifier To be filled with the pointer to the created
 *  NvSciEventNotifier object.
 *
 * \return ::NvSciError, the completion code of this operation.
 * - ::NvSciError_Success The setup is successful.
 * - ::NvSciError_BadParameter @a eventService is null or @a eventNotifier
 *   is null.
 * - ::NvSciError_InvalidState An NvSciStream API has already been called on the
 *   block referenced by @a block.
 * - Error/panic behavior of this API includes
 *    - Any error/panic behavior that NvSciEventService::CreateLocalEvent()
 *      can generate when @a eventService and @a eventNotifier arguments
 *      are passed to it.
 */
NvSciError NvSciStreamBlockEventServiceSetup(
    NvSciStreamBlock const block,
    NvSciEventService  *const eventService,
    NvSciEventNotifier **const eventNotifier
);

#ifdef __cplusplus
}
#endif
/** @} */
/** @} */
#endif /* NVSCISTREAM_API_H */
