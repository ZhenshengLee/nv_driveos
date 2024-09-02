/* ***************************************************************************** *
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 ******************************************************************************* */
/**
 * @file
 * @brief NVIDIA PKCS11 Public definitions header file
 * @details This header file contains the NVIDIA definitions for crypto operation
 * limitations with respect to hardware and software configuration
 */

#ifndef NVPKCS11_PUBLIC_DEFS_H_
#define NVPKCS11_PUBLIC_DEFS_H_

/**
 * @defgroup nvpkcs11_public_macros Macro Definitions
 *
 * Macro definitions based on NVIDIA H/W and S/W capabilities
 *
 * @ingroup grp_pkcs11_api
 * @{
 */

/** Block size required for AES Encryption, Input data for Encryption operations needs to be multiple of this value */
#define NVPKCS11_AES_CBC_BLOCK_SIZE 16UL
/** Defines the length of the AES Initial Vector as well as the AES counter */
#define NVPKCS11_AES_CBC_IV_LEN 16UL
/** Defines the AES_CTR counter size in bits */
#define NVPKCS11_AES_CTR_COUNTER_SIZE 32U
/** Maximum Key size for use in C_DeriveKey operations */
#define NVPKCS11_MAX_KEY_ID_SIZE 32U
/** Maximum CKA Label size */
#define NVPKCS11_MAX_CKA_LABEL_SIZE 32U
/** Maximum CKA_APPLICATION size for Data Objects */
#define NVPKCS11_MAX_GDO_CKA_APPLICATION_SIZE 32U
/** Maximum CKA_OBJECT_ID size for Data Objects */
#define NVPKCS11_MAX_GDO_CKA_OBJECT_ID_SIZE 64U
/** Maximum CKA_VALUE size for Data Objects */
#define NVPKCS11_MAX_GDO_CKA_VALUE_SIZE 3616U
/** Maximum Random Number length supported for C_GenerateRandom Operation*/
#define NVPKCS11_RANDOM_DATA_MAXLENGTH 1024U
/** Minimum Random Number length supported for C_GenerateRandom Operation*/
#define NVPKCS11_RANDOM_DATA_MINLENGTH 1U
/** Defines the maximum signature size parameter for AES_CMAC mechanism */
#define NVPKCS11_AES_CMAC_SIGNATURE_SIZE 16U
/* Defines the minimum and maximum Modulus and Exponent sizes for Public Key Objects */
/** Maximum size of Modulus for Public Key Objects */
#define NVPKCS11_MAX_KEY_MODULUS 512U
/** Minimum size of Modulus for Public Key Objects */
#define NVPKCS11_MIN_KEY_MODULUS 384U
/** Maximum size of Exponent for Public Key Objects */
#define NVPKCS11_MAX_KEY_EXPONENT 4U
/** Minimum size of Exponent for Public Key Objects */
#define NVPKCS11_MIN_KEY_EXPONENT 4U
/* Size of digest of each SHA mechanism */
/** SHA256 Digest Size */
#define NVPKCS11_SHA256_DIGEST_SIZE        32U
/** SHA384 Digest Size */
#define NVPKCS11_SHA384_DIGEST_SIZE        48U
/** SHA512 Digest Size */
#define NVPKCS11_SHA512_DIGEST_SIZE        64U
/** Max allowed SHA digest size */
#define NVPKCS11_MAX_SHA_DIGEST_SIZE NVPKCS11_SHA512_DIGEST_SIZE
/** ECDSA curve type identifier */
#define NVPKCS11_ECDSA_SECP256R1_STRING "secp256r1"
/** EDDSA curve type identifier */
#define NVPKCS11_EDDSA_ED25519_STRING "edwards25519"
/** EC MONTGOMERY curve type identifier */
#define NVPKCS11_EC_MONTGOMERY_25519_STRING "curve25519"
/** X9.62 Uncompressed point identifier */
#define NVPKCS11_ECDSA_X962_UNCOMP_ID 0x04U
/** ASN1 Identifier for PrintableString */
#define NVPKCS11_DER_PRINTABLE_IDENTIFIER 0x13U
/** ASN1 DER Identifier for uncompressed OCTET STRING */
#define NVPKCS11_DER_OCTET_IDENTIFIER 0x04U
 /** Maximum allowed size of ASN.1 DER format strings */
#define NVPKCS11_DER_MAX_SIZE 127U
/** Max allowed length for EC Params string */
#define NVPKCS11_MAX_EC_STRING_SIZE NVPKCS11_DER_MAX_SIZE
/** Size for EC point values */
#define NVPKCS11_EC_256_POINT_SIZE 32U
/** Max allowed Key size for ECC private key type */
#define NVPKCS11_ECC_PRIVATE_KEY_SIZE 32U
/** EDDSA signature size */
#define NVPKCS11_EDDSA_SIGNATURE_SIZE 64U
/** Max ECDSA signature size with curve SECP256R1 */
#define NVPKCS11_MAX_ECDSA_SECP256R1_SIGNATURE_SIZE 72U
/** Secret key size */
#define NVPKCS11_SECRET_KEY_LENGTH_IN_BYTES 16U
/** Double length secret key size */
#define NVPKCS11_LONG_SECRET_KEY_LENGTH_IN_BYTES 32U
/** TLS Master secret key size */
#define NVPKCS11_TLS_MASTER_SECRET_KEY_LENGTH_IN_BYTES 48U
/** Length of client and server random values for TLS handshake */
#define NVPKCS11_TLS_HANDSHAKE_RANDOM_LENGTH_IN_BYTES 32U
/** Length of label used to derive the TLS master key */
#define NVPKCS11_TLS12_MASTER_KEY_DERIVE_LABEL_LENGTH_IN_BYTES 13U
/** Length of label used to derive the TLS session keys */
#define NVPKCS11_TLS12_KEY_AND_MAC_DERIVE_LABEL_LENGTH_IN_BYTES 13U
/** Maximum length of the data field for CKM_NVIDIA_AES_CBC_KEY_DATA_WRAP */
#define NVPKCS11_AES_CBC_KEY_DATA_WRAP_MAX_DATA_LENGTH_IN_BYTES 32U
/** The length of the AES KEY WRAP Initial Vector */
#define NVPKCS11_AES_KEY_WRAP_IV_LENGTH 8U
/** Max data length for CKM_NVIDIA_PSC_AES_CMAC */
#define NVPKCS11_MAX_PSC_CMAC_DATA_LEN 1500U

/** GetInfo major PKCS11 API version */
#define NVPKCS11_GETINFO_CRYPTOKIVERSION_MAJOR 0x03U
/** GetInfo minor PKCS11 API version */
#define NVPKCS11_GETINFO_CRYPTOKIVERSION_MINOR 0x00U
/** GetInfo manufacturer name */
#define NVPKCS11_GETINFO_MANF_ID "NVIDIA"
/** GetInfo flags value */
#define NVPKCS11_GETINFO_FLAGS 0x00U
/** GetInfo library description */
#define NVPKCS11_GETINFO_LIBRARY_DESC "HPSE_Library"
/** GetInfo major library implementation version */
#define NVPKCS11_GETINFO_LIBRARYVERSION_MAJOR 0x3CU
/** GetInfo minor library implementation version */
#define NVPKCS11_GETINFO_LIBRARYVERSION_MINOR 0x09U

/** FSI token is for management purposes only, no safety token required */
#define NVPKCS11_FSI_DYNAMIC_1_MODEL_NAME     "FSI_DYN_1       "

/** CCPLEX_SAFETY_2 model name */
#define NVPKCS11_CCPLEX_SAFETY_2_MODEL_NAME   "CCPLEX_SAFE_2   "
/** CCPLEX_DYNAMIC_2 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_2_MODEL_NAME  "CCPLEX_DYN_2    "
/** TSEC_SAFETY_3 model name */
#define NVPKCS11_TSEC_SAFETY_3_MODEL_NAME     "TSEC_SAFE_3     "
/** TSEC_DYNAMIC_3 model name */
#define NVPKCS11_TSEC_DYNAMIC_3_MODEL_NAME    "TSEC_DYN_3      "
/** CCPLEX_SAFETY_4 model name */
#define NVPKCS11_CCPLEX_SAFETY_4_MODEL_NAME   "CCPLEX_SAFE_4   "
/** CCPLEX_DYNAMIC_4 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_4_MODEL_NAME  "CCPLEX_DYN_4    "

/** CCPLEX_SAFETY_5 model name */
#define NVPKCS11_CCPLEX_SAFETY_5_MODEL_NAME   "CCPLEX_SAFE_5   "
/** CCPLEX_DYNAMIC_5 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_5_MODEL_NAME  "CCPLEX_DYN_5    "

/** CCPLEX_SAFETY_6 model name */
#define NVPKCS11_CCPLEX_SAFETY_6_MODEL_NAME   "CCPLEX_SAFE_6   "
/** CCPLEX_DYNAMIC_6 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_6_MODEL_NAME  "CCPLEX_DYN_6    "

/** CCPLEX_SAFETY_7 model name */
#define NVPKCS11_CCPLEX_SAFETY_7_MODEL_NAME   "CCPLEX_SAFE_7   "
/** CCPLEX_DYNAMIC_7 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_7_MODEL_NAME  "CCPLEX_DYN_7    "

/** CCPLEX_SAFETY_8 model name */
#define NVPKCS11_CCPLEX_SAFETY_8_MODEL_NAME   "CCPLEX_SAFE_8   "
/** CCPLEX_DYNAMIC_8 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_8_MODEL_NAME  "CCPLEX_DYN_8    "

/** CCPLEX_SAFETY_9 model name */
#define NVPKCS11_CCPLEX_SAFETY_9_MODEL_NAME   "CCPLEX_SAFE_9   "
/** CCPLEX_DYNAMIC_9 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_9_MODEL_NAME  "CCPLEX_DYN_9    "

/** CCPLEX_SAFETY_10 model name */
#define NVPKCS11_CCPLEX_SAFETY_10_MODEL_NAME  "CCPLEX_SAFE_10  "
/** CCPLEX_DYNAMIC_10 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_10_MODEL_NAME "CCPLEX_DYN_10   "

/** CCPLEX_SAFETY_11 model name */
#define NVPKCS11_CCPLEX_SAFETY_11_MODEL_NAME  "CCPLEX_SAFE_11  "
/** CCPLEX_DYNAMIC_11 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_11_MODEL_NAME "CCPLEX_DYN_11   "

/** CCPLEX_SAFETY_12 model name */
#define NVPKCS11_CCPLEX_SAFETY_12_MODEL_NAME  "CCPLEX_SAFE_12  "
/** CCPLEX_DYNAMIC_12 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_12_MODEL_NAME "CCPLEX_DYN_12   "

/** CCPLEX_SAFETY_13 model name */
#define NVPKCS11_CCPLEX_SAFETY_13_MODEL_NAME  "CCPLEX_SAFE_13  "
/** CCPLEX_DYNAMIC_13 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_13_MODEL_NAME "CCPLEX_DYN_13   "

/** CCPLEX_SAFETY_14 model name */
#define NVPKCS11_CCPLEX_SAFETY_14_MODEL_NAME  "CCPLEX_SAFE_14  "
/** CCPLEX_DYNAMIC_14 model name */
#define NVPKCS11_CCPLEX_DYNAMIC_14_MODEL_NAME "CCPLEX_DYN_14   "

/**
 * Number of supported tokens.
 * Can be used to allocate an array of slots for use with C_GetSlotList */
#define NVPKCS11_TOKEN_COUNT 27U

/** The number of PKCS#11 sessions that can be opened */
#define NVPKCS11_MAX_SESSIONS 256U

/** @}*/

/**
 * @defgroup nvpkcs11_legacy_public_macros Legacy Macro Definitions
 *
 * Deprecated macro definitions
 *
 * @ingroup grp_pkcs11_api
 * @{
 */

/** Max allowed key size for ECDSA key type (Legacy) */
/** This macro definition is deprecated in favour of a mechanism-agnostic definition */
#define NVPKCS11_ECDSA_256_KEY_SIZE NVPKCS11_EC_256_POINT_SIZE

/** Max allowed key size for EDDSA key type (Legacy) */
/** This macro definition is deprecated in favour of a mechanism-agnostic definition */
#define NVPKCS11_EDDSA_256_KEY_SIZE NVPKCS11_EC_256_POINT_SIZE

/** Max input buffer size for SHA operation 256 x 1MB (Legacy) */
/** This macro definition is deprecated and will be removed in a future release */
/** This value exceeds the maximum buffer size on some channels which can be as low as 1KB */
/** Applicable max buffer size should be determined from knowledge of your device tree channel settings */
#define NVPKCS11_SHA_MAX_DATA_SIZE 256UL * 0x100000UL

/** Maximum input buffer size for AES encrypt/decrypt operation 1MB (Legacy) */
/** This macro definition is deprecated and will be removed in a future release */
/** This value exceeds the maximum buffer size on some channels which can be as low as 16KB */
/** Applicable max buffer size should be determined from knowledge of your device tree channel settings */
#define NVPKCS11_AES_BUFFER_LIMIT 1U * 0x100000UL

/** @}*/

/**
 * @defgroup nvpkcs11_public_struct Structure Definitions
 *
 * Structure definitions based on NVIDIA H/W and S/W capabilities.
 *
 * @ingroup grp_pkcs11_api
 * @{
 */

/**
* @brief **ecParameters_t** Holds EC parameters.
* @details Structure that contains the Elliptic curve name to be used.<br>
* This needs to be a printable non-null terminated string.
*/
typedef struct __attribute__((__packed__)) ecParameters_t
{
	CK_BYTE identifier; /**< Identifier, value must be set to #NVPKCS11_DER_PRINTABLE_IDENTIFIER */
	CK_BYTE size; /**< Size of the printable string #printableString */
	CK_UTF8CHAR printableString[NVPKCS11_MAX_EC_STRING_SIZE]; /**< String containing the name of the curve,<br> e.g #NVPKCS11_EDDSA_ED25519_STRING or #NVPKCS11_ECDSA_SECP256R1_STRING */
} ecParameters_t;

/**
* @brief **uncompressed32BytePoint_t** Holds an uncompressed value for point data.<br>
* ECDSA and ECDH use this format.
*/
typedef struct __attribute__((__packed__)) uncompressed32BytePoint_t
{
	CK_BYTE identifier; /**< Identifier value. <br>(Must be set to #NVPKCS11_DER_OCTET_IDENTIFIER) */
	CK_BYTE size; /**< The size of the fields #x962_id, #qX and #qY */
	CK_BYTE x962_id; /**< Type identifier. <br>(Must be set to #NVPKCS11_ECDSA_X962_UNCOMP_ID) */
	CK_BYTE qX[NVPKCS11_EC_256_POINT_SIZE]; /**< Elliptic Curve point x-coordinate <br>(size must be equal to #NVPKCS11_EC_256_POINT_SIZE) */
	CK_BYTE qY[NVPKCS11_EC_256_POINT_SIZE]; /**< Elliptic Curve point y-coordinate <br>(size must be equal to #NVPKCS11_EC_256_POINT_SIZE) */
} uncompressed32BytePoint_t;

/**
* @brief **compressed32BytePoint_t** Holds a compressed value for point data.<br>
* EDDSA and ECDH use this format.
*/
typedef struct __attribute__((__packed__)) compressed32BytePoint_t
{
	CK_BYTE identifier; /**< Identifier value.<br>(Must be set to #NVPKCS11_DER_OCTET_IDENTIFIER) */
	CK_BYTE size; /**< The size of the field <br>(Must be set to #NVPKCS11_EC_256_POINT_SIZE) */
	CK_BYTE point[NVPKCS11_EC_256_POINT_SIZE]; /**< Elliptic Curve point compressed value <br>(size must be equal to #NVPKCS11_EC_256_POINT_SIZE) */
} compressed32BytePoint_t;

/**
* @brief **eddsaPoint_t** is a legacy alias type that holds a generic compressed point value.<br>
* Uses a compressed 32 byte value for point data.
*/
typedef compressed32BytePoint_t eddsaPoint_t;

/**
* @brief **ecdsaPoint_t** is a legacy alias type that holds a generic uncompressed point value.<br>
* Uses an uncompressed 2 x 32 byte value for point data.
*/
typedef uncompressed32BytePoint_t ecdsaPoint_t;

/*
 * NVIDIA PKCS11 library supports 3 interfaces:
 * "PKCS 11": this interface name represents 2 interfaces, one which is associated
 * with the Oasis standards version 3.0 CK_FUNCTION_LIST_3_0 structure and the other,
 * with the Oasis standards version 2.40 CK_FUNCTION_LIST structure.
 * "Vendor NVIDIA", this interface name is associated with NV_CK_FUNCTION_LIST
 * structure that contains NVIDIA extension APIs.
 */

/** PKCS11 interface name */
#define PKCS11_INTERFACE_NAME "PKCS 11"
/** NVIDIA PKCS11 extensions interface name */
#define NVIDIA_INTERFACE_NAME "Vendor NVIDIA"
/** @}*/
#endif /* NVPKCS11_PUBLIC_DEFS_H_ */
