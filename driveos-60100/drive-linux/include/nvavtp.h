/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
/**
 * \file
 * \brief <b> NVIDIA Ethernet Audio/Video Bridge API </b>
 *
 * @b Description: This file contains the Ethernet Audio/Video Bridge API.
 */
#ifndef _NVAVTP_H_
#define _NVAVTP_H_



#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup mult_eth_avb_group Ethernet Audio/Video Bridge API
 * @ingroup mult_utils_grp
 *
 * Describes the available application controls for the Ethernet Audio/Video Bridge
 * (EAVB.)
 *
 * The Ethernet Audio Video Bridge (EAVB) carries different types of packets
 * (e.g., audio and video) on different virtual LANs (VLANs), using IEEE P802.1p
 * tagging to distinguish different types of packets. This saves applications from having to
 * monitor all of the packets delivered to the physical interface; an application can listen
 * instead to the VLAN interface that delivers the type of packets that interest it.
 *
 * The avtp_buffer data is arranged in the following ways:
 *   - Ethernet Header pointer: avtp_buffer
 *   - 1722 Header pointer: avtp_buffer + AVTP_ETHERNET_HEADER_OFFSET
 *   - 61883 Header pointer: avtp_buffer + AVTP_ETHERNET_HEADER_OFFSET + AVTP_1722_HEADER_OFFSET
 *   - AVTP Data Payload avtp_buffer + AVTP_ETHERNET_HEADER_OFFSET + AVTP_1722_HEADER_OFFSET +
 *     AVTP_61883_HEADER_V1
 *
 * @{
 */



typedef unsigned char         U8;                    // 8-bit
typedef unsigned short        U16;                   // 16-bit
typedef unsigned int          U32;                   // 32-bit
typedef unsigned long long    U64;                   // 64-bit
typedef signed char           S8;
typedef signed short          S16;
typedef signed int            S32;
typedef signed long long      S64;


/// Defines AVTP sub-header types.
typedef enum tagNvAvtpSubHeaderType {
    /// Specifies IEC 61883-6 audio and music data transmission protocol
    /// (uncompressed audio).
    eNvAudio,
    /// Specifies IEC 61883-4 MPEG2-TS data transmission.
    eNvMpegts,
    /// Specifies 1722 AVTP audio format.
    eNvAAF,
    /// Specifies 1722 AVTP clock reference frame.
    eNvCRF,
    /// Specifies IEC 61883-8 61883-8 transmission of ITU-R
    ///BT.601-style digital video data (raw video.)
    eNvRawVideo,
    /// Specifies 1722 CVF format
    eNvCvf
} ENvAvtpSubHeaderType;

/// Indicates results from an AVTP function.
typedef enum tagNvAvtpStatus {
    /// Indicates success.
    eNvAvtpSuccess,
    /// Indicates failure.
    eNvAvtpFail,
    /// Indicates a null pointer argument.
    eNvAvtpInvalidPointer,
    /// Indicates a memory allocation error.
    eNvAvtpAllocError,
    /// Indicates an unsupported setting.
    eNvAvtpUnsupported,
    /// Indicates an invalid entry.
    eNvAvtpInvalidParam
} ENvAvtpStatus;

/// Defines Boolean flags for true/false states.
typedef enum tagNvAvtpBool {
    /// Indicates false.
    eNvAvtpFalse,
    /// Indicates true.
    eNvAvtpTrue
}ENvAvtpBool;

/// Defines the AVTP 1722 header structure.
typedef struct __attribute__ ((packed)) tagNvAvtp1722Header {
    /// Holds the protocol being carried by AVTP, for example, IEC 61883.
    U64 subType:8;
    /// Holds a flag that indicates the validity of the AVTP timestamp field time value.
    U64 timestampValid:1;
    /// Reserved for use by AVTP gateways.
    U64 fsd:2;
    /// Holds the change in the source of the media clock.
    U64 mediaClockRestart:1;
    /// Holds the version of AVTP for which a particular AVTPDU is formatted.
    U64 version:3;
    /// Holds a flag that indicates whether the 64-bit \c stream_id contains a valid \c StreamID.
    U64 streamidValid:1;
    /// Holds the sequence of the AVTPDUs in a stream by the talker.
    U64 sequenceNum:8;
    /// Holds a flag that indicates the timestamps may not be globally synchronized with the
    /// network clock.
    U64 timestampUncertain:1;
    /// Reserved for future use in stream data of AVTPDU headers.
    U64 fsd1:7;
    /// Holds the ID of the stream.
    U64 streamId;
    /// Holds the AVTP presentation time in nanoseconds.
    U64 avtpTimestamp:32;
    /// Reserved for use by AVTP gateways.
    U64 fsd2:32;
    /// Holds the unsigned count of the AVTPDU of the stream's
    /// AVTPDU in octets.
    U64 streamDataLength:16;
} NvAvtp1722Header;

/// Defines the AVTP 61883-6 header structure.
typedef struct __attribute__ ((packed)) tagNvAvtp618836Fsd3Hdr {
    /// Holds the channel as specified in IEEE standard 1394-2008.
    U16 packetChannel:6;
    /// Holds the format specified by IEEE standard 1394-2008.
    U16 formatTag:2;
    /// Holds application-specific information.
    U16 syAppControl:4;
    /// Holds type code as specified in IEEE standard 1394-2008.
    /// For AVTP the only supported value is A.
    U16 packetTcode:4;
    /// Holds the source identifier as specified in IEC 61883-1.
    U16 sourceId:6;
    /// Holds the CIP header 1st quadlet as defined in IEC 61883-1.
    U16 reserved0:2;
    /// Holds data block size in quadlets as specified in IEC 61883-1:2003.
    U16 dataBlockSize:8;
    /// Reserved as defined in IEC 61883-1:2003.
    U16 rsv:2;
    /// Holds the source packet header as defined in IEC 61883-1:2003.
    /// Holds a value of 1 if AVTPDU contains IEC 61883-4:2004 and
    /// IEC 61883-7:2003, and holds a value of 0 if AVTPU.
    U16 sourcePacketHeader:1;
    /// Holds the quadlet padding count as defined in IEC 61883-1:2003.
    U16 quadletPaddingCount:3;
    /// Holds the fraction number as defined in IEC 61883-1. This is currently only
    /// used in IEC 61883-4:2004 and IEC 61883-7:2003.
    U16 fractionNumber:2;
    /// Holds DBC as defined in IEC 61883-1:2003, as the
    /// sequence number of the first data block in AVTPDU.
    U16 dataBlockContinuity:8;
    /// Holds the format ID as defined in IEC 61883-1:2003.
    U16 formatId:6;
    /// Holds the CIP header 2nd quadlet indicator defined in IEC 61883-1.
    U16 reserved1:2;
    /// Holds FDF as defined in in IEC 61883-1:2003. If SPH is 0, then FDF is 8.
    /// If SPH is 1, then FDF is 24.
    U16 formatDependentField:8;
    /// Holds synchronization timing as defined in IEC 61883-1.
    U16 syt;
} NvAvtp618836Fsd3Hdr;

//! Holds the AVTP 61883-4 header structure.
typedef struct __attribute__ ((packed)) tagNvAvtp618834Fsd3Hdr {
    /// Holds the channel as specified in IEEE Std 1394-2008.
    U16 packetChannel:6;
    /// Holds the format specified by IEEE Std 1394-2008.
    U16 formatTag:2;
    /// Holds the application-specific data.
    U16 syAppControl:4;
    /// Holds the type code as specified in IEEE Std 1394-2008.
    /// For AVTP, the only value supported is A.
    U16 packetTcode:4;
    /// Holds the source identifier as specified in IEC 61883-1.
    U16 sourceId:6;
    /// Holds the CIP header 1st quadlet indicator defined in IEC 61883-1.
    U16 reserved0:2;
    /// Holds the data block size in quadlets as specified in IEC 61883-1:2003.
    U16 dataBlockSize:8;
    /// Reserved as defined in IEC 61883-1:2003
    U16 rsv:2;
    /// Holds the source packet header as defined in IEC 61883-1:2003.
    /// Holds 1 if AVTPDU contains IEC 61883-4:2004 and IEC 61883-7:2003
    /// and holds 0 if AVTPU.
    U16 sourcePacketHeader:1;
    /// Holds the quadlet padding count as defined in IEC 61883-1:2003.
    U16 quadletPaddingCount:3;
    /// Holds the fraction number as defined in IEC 61883-1.
    /// Currently only used in IEC 61883-4:2004 & IEC 61883-7:2003.
    U16 fractionNumber:2;
    /// Holds the DBC as defined in IEC 61883-1:2003. Contains the
    /// the sequence number of the first data block in AVTPDU.
    U16 dataBlockContinuity:8;
    /// Holds the format ID as defined in IEC 61883-1:2003.
    U16 formatId:6;
    /// Holds the CIP header 2nd quadlet indicator defined in IEC 61883-1.
    U16 reserved1:2;
    /// Holds the FDF as defined in in IEC 61883-1:2003. If SPH is 0,
    /// then FDF is 8. If SPH is 1, then FDF is 20.
    U32 formatDependentField:24;
} NvAvtp618834Fsd3Hdr;

//! Holds the AVTP CVF header structure fsd2.
typedef struct __attribute__ ((packed)) tagNvAvtpCvfFsd2Hdr {
    /// Identifies compressed video payload format
    U64 format:8;
    /// Holds the subtype of cvf
    U64 format_subtype:8;
    /// Reserved for use by AVTP gateways.
    U64 reserved:16;
} NvAvtpCvfFsd2Hdr;

//! Holds the AVTP CVF header structure.
typedef struct __attribute__ ((packed)) tagNvAvtpCvfFsd3Hdr {
    /// Reserved as defined in IEC 1722 CVF
    U8  rsv1:3;
    /// The M field is defined on a format/format_subtype specific basis.
    /// If a format does not define a usage for M then the M field shall be set to zero 0
    U8  M:1;
    /// Reserved for upper level protocols to indicate events in the stream
    U8  evt:4;
    /// Reserved as defined in IEC 1722 CVF
    U8  reserved1:8;
    /// Reserved temporarily as the 1722 stream contains 4 extra bytes
    U16 reserved2:16;
    /// Reserved temporarily as the 1722 stream contains 4 extra bytes
    U16 reserved3:16;
} NvAvtpCvfFsd3Hdr;

typedef struct __attribute__ ((packed)) tagNvAvtpCvfH264PayloadHdr {
    /// Holds the h264 Data payload format type
    U8  payload_type:5;
    /// Holds the nal_ref_idc
    U8  nri:2;
    /// Holds forbidden_bit
    U8  forbidden_bit:1;
    /// Holds NALU type
    U8  nalu_type:5;
    /// Reserved as defined in IEC 1722 CVF
    U8  rsv2:1;
    /// Indicates last packet of a NALU
    U8  end_bit:1;
    /// Indicates first packet of a NALU
    U8  start_bit:1;
} NvAvtpH264PayloadHdr;

/// Defines the 1722 AVTP audio format fsd2.
typedef struct __attribute__ ((packed)) tagNvAvtp1722AAFFsd2 {
    U16 format:8;
    U16 channelsPerFrame_H2:2;
    U16 reserved:2;
    U16 nominalSampleRate:4;
    U16 channelsPerFrame_L8:8;
    U16 bitDepth:8;
} NvAvtp1722AAFFsd2;

/// Defines the 1722 AVTP audio format header.
typedef struct __attribute__ ((packed)) tagNvAvtp1722AAFFsd3Hdr {
    U16 evt:4;
    U16 sp:1;
    U16 reserved1:3;
    U16 reserved2:8;
} NvAvtp1722AAFFsd3Hdr;

//! Defines the Ethernet header structure.
typedef struct tagNvAvtpEthernetHeader {
    /// Holds the Ethernet address of the destination AVB endpoint.
    U8 destAddr[6];
    /// Holds the Ethernet address of the source AVB endpoint.
    U8 srcAddr[6];
    /// Holds the 802.1 Qtag field.
    U8 vtag[4];
    /// Holds the AVTP EtherType (0x22F0).
    U8 etherType[2];
} NvAvtpEthernetHeader;

typedef struct __attribute__ ((packed)) tagNvAvtpCRFHeader {
    U8  timingUncertain:1;
    U8  clockSync:1;
    U8  reservedBit:1;
    U8  mediaClockRestart:1;
    U8  version:3;
    U8  streamIDValid:1;
    U8  sequenceNum:8;
    U8  CRFType:8;
    U64 streamID:64;
    U8  baseFrequencyHighBits:5;
    U8  pull:3;
    U8  baseFrequencyMidBits;
    U16 baseFrequencyLowBits;
    U16 CRFDataLength:16;
    U16 timestampInterval:16;
} NvAvtpCRFHeader;

#define NVAVTP_ETH_INT_MAX 64

/// Specifies the AVTP input parameters structure.
typedef struct tagNvAvtpInputParams {
    /// Holds the name of the Ethernet interface in string format.
    char                         interface[NVAVTP_ETH_INT_MAX];
    /// Indicates whether the library is being used for packetization
    /// or depacketization.
    ENvAvtpBool                   bAvtpDepacketization;
    /// Indicates the header type.
    ENvAvtpSubHeaderType        eDataType;
} NvAvtpInputParams;

/// Defines the AVTP AAF input parameter structure.
typedef struct tagNvAvtp1722AAFParams {
    U8 format;
    U32 bitFormat;
    U16 channelsPerFrame;
    U32 samplingRate;
    U32 bitDepth;
    ENvAvtpBool sparseTimestamp;
    U8 bytesPerSample;
} NvAvtp1722AAFParams;

/// Defines the CRF input parameters structure.
typedef struct tagNvAvtpCRFParams {
    U8  type;
    U32 dataLength;
    U16 timestampInterval;
    U32 frequency;
    U64 streamID;
} NvAvtpCRFParams;

/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_ETHERNET_HEADER_SIZE sizeof(NvAvtpEthernetHeader)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_ETHERNET_HEADER_WITHOUT_QTAG_SIZE (sizeof(NvAvtpEthernetHeader) - 4)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_1722_HEADER_SIZE sizeof(NvAvtp1722Header)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_CRF_HEADER_SIZE sizeof(NvAvtpCRFHeader)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_61883_6_FSD3_HEADER_SIZE sizeof(NvAvtp618836Fsd3Hdr)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_61883_4_FSD3_HEADER_SIZE sizeof(NvAvtp618834Fsd3Hdr)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_1722_AAF_FSD3_HEADER_SIZE sizeof(NvAvtp1722AAFFsd3Hdr)
#define NV_AVTP_1722_61883_HEADER_SIZE 8
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_CVF_FSD3_HEADER_SIZE sizeof(NvAvtpCvfFsd3Hdr)
/// Macro for header offsets. See @ref mult_eth_avb_group.
#define NV_AVTP_H264_PAYLOAD_HEADER_SIZE sizeof(NvAvtpH264PayloadHdr)

#define NVAVTP_TSP_SIZE 188
#define NV_AVTP_MPEGTS_MAX_TS_PER_PKT 7
#define NV_AVTP_ETHER_TYPE_OFFSET 12
#define NVAVTP_ETH_INT_MAX 64

typedef void *NvAvtpContextHandle;




/// \brief Initializes prerequisites to start AVB communication,
/// This function performs the following tasks:
/// * Initializes for MRP static stream reservation.
/// * Initializes communication with gPTP daemon.
/// * Populates the default headers.
/// * Creates an AVB context structure instance.
///
/// \param[in] pNvAvtpInpParams A pointer to the input parameters.
/// \param[in] phContext A pointer to the context to create.
ENvAvtpStatus
NvAvtpInit(
    NvAvtpInputParams *pNvAvtpInpParams,
    NvAvtpContextHandle *phContext);


/// \brief Deinitializes AVB initializations and destroys the AVB
/// context structure instance.
/// \param[in] hContext A handle to the AVB context structure to destroy.
ENvAvtpStatus
NvAvtpDeinit(
    NvAvtpContextHandle hContext);


/// \brief Sets the static header part of the AVTP header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
ENvAvtpStatus
NvAvtpSetStaticAvtpHeader(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// \brief Sets the AVTP packet size (header + payload) based on
///  the number of samples to be sent per packet.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] numSampleUnits Specifies the number of samples per packet.
ENvAvtpStatus
NvAvtpSetPacketSize(
    NvAvtpContextHandle hContext,
    U16 numSampleUnits);

/// \brief Gets the AVTP packet size required for the header
/// and payload.
/// \param[in] hContext A handle to the AVB context structure.
U32
NvAvtpGetPacketSize(
    NvAvtpContextHandle hContext);

/// Sets the Destination MAC Address in the packet header.
/// \param[in] hContext  A handle to the AVB context structure.
/// \param[in] pPacket   A pointer to the AVTP packet.
/// \param[in] pDestAddr A pointer to the destination address.
ENvAvtpStatus
NvAvtpSetDestAdd(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 *pDestAddr);

/// Sets the Source MAC Address in the packet header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to the AVTP packet.
/// \param[in] pSrcAddr A pointer to the source address.
ENvAvtpStatus
NvAvtpSetSrcAdd(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 *pSrcAddr);

/// Sets the QTag fields of the AVTP header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to the AVTP packet.
/// \param[in] priority Specifies the priority of the stream.
/// \param[in] vid      Specifies the VLAN ID.
ENvAvtpStatus
NvAvtpSetQTagFields(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U32 priority,
    U32 vid);

/// Sets Stream ID validity.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to the AVTP packet.
/// \param[in] sidValid Specifies the validity of the stream ID.
ENvAvtpStatus
NvAvtpSetSIDValid(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 sidValid);

/// Sets the stream ID in the packet header.
/// \param[in] hContext   A handle to the AVB context structure.
/// \param[in] pPacket    A pointer to the AVTP packet.
/// \param[in] stream_id  A pointer to the stream ID.
ENvAvtpStatus
NvAvtpSetStreamID(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 *stream_id);

/// Gets the current time from PHC.
/// \param[in] hContext  A handle to the AVB context structure.
/// \param[out] pPtpTime A pointer to the PHC timestamp.
ENvAvtpStatus
NvAvtpGetCurrentGptpTimeStamp(
NvAvtpContextHandle hContext,
U32 *pPtpTime);

/// Sets the dynamic part of the packet header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to the AVTP packet.
ENvAvtpStatus
NvAvtpSetDynamicAvtpHeader(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Sets the data payload in the AVTP packet.
/// \param[in] hContext  A handle to the AVB context structure.
/// \param[in] pPacket   A pointer to the AVTP packet.
/// \param[in] pPayload  A pointer to the data payload.
/// \param[in] numSample Specifies the number of samples in the packet.
ENvAvtpStatus
NvAvtpFillDataPayload(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 *pPayload,
    U16 numSample);

/// Gets the timestamp in the 1722 header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to an AVTP packet.
/// \return The timestamp in the 1722 header.
U32
NvAvtpGetTimeStamp(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Gets the sequence number in the 1722 header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to an AVTP packet.
/// \return The sequence number in the 1722 header.
U8
NvAvtpGetSequenceNum(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Gets the stream id in the 1722 header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to an AVTP packet.
U64
NvAvtpGetStreamId(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Gets the length of the data payload in the packet.
/// \param[in] hContext     A handle to the AVB context structure.
/// \param[in] pPacket      A pointer to an AVTP packet.
/// \param[out] streamLength A pointer to the stream length.
ENvAvtpStatus
NvAvtpGetStreamLength(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U32 *streamLength);

/// Gets the data payload from the AVTP packet.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket  A pointer to an AVTP packet.
/// \param[out] pPayload A pointer to the data payload.
ENvAvtpStatus
NvAvtpExtractDataPayload(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U8 *pPayload);

/// Determines whether the Ethernet packet is an AVTP 1722 packet.
/// \param[in] pPacket A pointer to an AVTP packet.
/// \return A Boolean that, if \c true, indicates the packet is an AVTP 1722 packet.
ENvAvtpBool
NvAvtpIs1722Packet(
    U8 *pPacket);

/// Gets the recorded AAF parameters from context.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to an AVTP packet.
/// \param[out] pAvtp1722AAFParameters A pointer to the AAF parameters.
ENvAvtpStatus
NvAvtpGetAAFParams(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    NvAvtp1722AAFParams  *pAvtp1722AAFParameters);

/// Parses the AVTP AAF headers and populates the AAF
/// parameter structure specified with \a hContext.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to an AVTP packet.
ENvAvtpStatus
NvAvtpParseAAFHeaders(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Sets the AAF parameters into the context.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pAvtp1722AAFParameters A pointer to \ref NvAvtp1722AAFParams.
ENvAvtpStatus
NvAvtpSetAAFParams(
    NvAvtpContextHandle hContext,
    NvAvtp1722AAFParams  *pAvtp1722AAFParameters);

/// Sets the default AAF parameters into the context.
/// \param[in] hContext A handle to the AVB context structure.
ENvAvtpStatus
NvAvtpSetAAFDefaultParams(
    NvAvtpContextHandle hContext);

/// Sets the static header part of the AVTP header.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
ENvAvtpStatus
NvAvtpSetStaticAAFHeader(
    NvAvtpContextHandle hContext,
    U8 *pPacket);

/// Parses the AVTP packet and populates the header offsets
/// specified with \a hContext. Checks if there are any packet drops, and
/// checks which type of 61883 packet is present.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
/// \param[in] peDataType A pointer to the AVTP subheader type.
ENvAvtpStatus
NvAvtpParseAvtpPacket(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    ENvAvtpSubHeaderType *peDataType);

/// Gets the payload size of the AVTP MPEGTS packet.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
/// \param[out] dataPayloadSize A pointer to the payload size of the AVTP MPEGTS packet.
ENvAvtpStatus
NvAvtpGetMpegtsDataPayloadSize(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U32 *dataPayloadSize);

/// Gets the payload size of the AVTP H264 packet.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
/// \param[out] dataPayloadSize A pointer to the payload size of the AVTP H264 packet.
ENvAvtpStatus
NvAvtpGetCvfDataPayloadSize(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U32 *dataPayloadSize);

/// Gets the payload size of the AVTP audio packet.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
/// \param[out] dataPayloadSize A pointer to the payload size of the AVTP audio packet.
ENvAvtpStatus
NvAvtpGetAudioDataPayloadSize(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    U32 *dataPayloadSize);

/// Gets the recorded CRF parameters from the context.
/// \param[in] hContext A handle to the AVB context structure.
/// \param[in] pPacket A pointer to the AVTP packet.
/// \param[out] pAvtpCRFParameters A pointer to the recorded CRF parameters from the context.
ENvAvtpStatus
NvAvtpGetCRFParams(
    NvAvtpContextHandle hContext,
    U8 *pPacket,
    NvAvtpCRFParams  *pAvtpCRFParameters);

/** @} */

#endif

