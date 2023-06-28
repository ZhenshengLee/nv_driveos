/*
 * Copyright (c) 2018, NVIDIA Corporation.  All Rights Reserved.
 *
 * BY INSTALLING THE SOFTWARE THE USER AGREES TO THE TERMS BELOW.

 *
 * User agrees to use the software under carefully controlled conditions
 * and to inform all employees and contractors who have access to the software
 * that the source code of the software is confidential and proprietary
 * information of NVIDIA and is licensed to user as such.  User acknowledges
 * and agrees that protection of the source code is essential and user shall
 * retain the source code in strict confidence.  User shall restrict access to
 * the source code of the software to those employees and contractors of user
 * who have agreed to be bound by a confidentiality obligation which
 * incorporates the protections and restrictions substantially set forth
 * herein, and who have a need to access the source code in order to carry out
 * the business purpose between NVIDIA and user.  The software provided
 * herewith to user may only be used so long as the software is used solely
 * with NVIDIA products and no other third party products (hardware or
 * software).   The software must carry the NVIDIA copyright notice shown
 * above.  User must not disclose, copy, duplicate, reproduce, modify,
 * publicly display, create derivative works of the software other than as
 * expressly authorized herein.  User must not under any circumstances,
 * distribute or in any way disseminate the information contained in the
 * source code and/or the source code itself to third parties except as
 * expressly agreed to by NVIDIA.  In the event that user discovers any bugs
 * in the software, such bugs must be reported to NVIDIA and any fixes may be
 * inserted into the source code of the software by NVIDIA only.  User shall
 * not modify the source code of the software in any way.  User shall be fully
 * responsible for the conduct of all of its employees, contractors and
 * representatives who may in any way violate these restrictions.
 *
 * NO WARRANTY
 * THE ACCOMPANYING SOFTWARE (INCLUDING OBJECT AND SOURCE CODE) PROVIDED BY
 * NVIDIA TO USER IS PROVIDED "AS IS."  NVIDIA DISCLAIMS ALL WARRANTIES,
 * EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.

 * LIMITATION OF LIABILITY
 * NVIDIA SHALL NOT BE LIABLE TO USER, USERS CUSTOMERS, OR ANY OTHER PERSON
 * OR ENTITY CLAIMING THROUGH OR UNDER USER FOR ANY LOSS OF PROFITS, INCOME,
 * SAVINGS, OR ANY OTHER CONSEQUENTIAL, INCIDENTAL, SPECIAL, PUNITIVE, DIRECT
 * OR INDIRECT DAMAGES (WHETHER IN AN ACTION IN CONTRACT, TORT OR BASED ON A
 * WARRANTY), EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGES.  THESE LIMITATIONS SHALL APPLY NOTWITHSTANDING ANY FAILURE OF THE
 * ESSENTIAL PURPOSE OF ANY LIMITED REMEDY.  IN NO EVENT SHALL NVIDIAS
 * AGGREGATE LIABILITY TO USER OR ANY OTHER PERSON OR ENTITY CLAIMING THROUGH
 * OR UNDER USER EXCEED THE AMOUNT OF MONEY ACTUALLY PAID BY USER TO NVIDIA
 * FOR THE SOFTWARE PROVIDED HEREWITH.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <assert.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_sdk_platform.h>

#define APP_NAME "exttex_gen"

#ifdef VK_KHR_external_memory
// Vulkan extension API function ptr
PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR = NULL;
#endif

static PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = NULL;

#define TEX_WIDTH  1024
#define TEX_HEIGHT 1024

#define FAIL_PRINT(err_msg) \
    do {                             \
        printf("%s\n", err_msg);     \
        fflush(stdout);              \
        goto fail;                     \
    } while (0)

// Vulkan global state object
struct vk_state_rec {
    struct tex_res_rec {
        VkImage image;
        VkMemoryRequirements mem_reqs;
        VkDeviceMemory exp_mem;
    } tex_res;
    VkInstance instance;
    VkPhysicalDevice gpu;
    VkDevice device;
    VkPhysicalDeviceMemoryProperties memory_properties;
} vk_state;

static char socket_name[16][64];
static int g_sock_count = 0;

static int BindSocket(int sock)
{
    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));

    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, socket_name[g_sock_count++]);
    return bind(sock, (struct sockaddr *)&server_addr,
                sizeof(server_addr));
}


static int CreateAcceptSocket(int sock)
{
    struct sockaddr_un client_addr;
    int client_addr_len = sizeof(client_addr);

    return accept(sock, (struct sockaddr *)&client_addr,
                  (socklen_t*)&client_addr_len);
}

static int Connect(void)
{
    int retval;
    int server_sock;
    int accept_sock;

    server_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock < 0) {
        FAIL_PRINT("[C] create socket failed\n");
    }

    retval = BindSocket(server_sock);
    if (retval < 0) {
        FAIL_PRINT("[C] BindSocket failed\n");
    }

    retval = listen(server_sock , 1);
    if (retval < 0) {
        FAIL_PRINT("[C] ListenSocket failed\n");
    }

    accept_sock = CreateAcceptSocket(server_sock);
    if (accept_sock < 0) {
        FAIL_PRINT("[C] CreateAcceptSocket failed\n");
    }
    close(server_sock);

    return accept_sock;
fail:
    return -1;
}

// send an FD over socket
static void sendFD(int socket, int fd)
{
    struct msghdr msg = { 0 };
    char buf[CMSG_SPACE(sizeof(fd))];
    memset(buf, '\0', sizeof(buf));
    struct iovec io = { .iov_base = "ABC", .iov_len = 3 };

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));

    int *data = (int *)CMSG_DATA(cmsg);
    *data = fd;

    msg.msg_controllen = cmsg->cmsg_len;

    if (sendmsg(socket, &msg, 0) < 0)
        printf("Failed to send message\n");
}

static int InitVulkanState(void) {
    VkResult err;
    uint32_t device_enabled_extension_count = 0;
    char * device_extension_names[64];
    uint32_t gpu_count;
    uint32_t device_extension_count = 0;
    uint32_t demo_queue_family_count;
    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    VkQueueFamilyProperties *demo_queue_props;
    float queue_priorities[1] = {0.0};
    VkDeviceQueueCreateInfo queues[2];
    uint32_t i;

    const VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = APP_NAME,
        .applicationVersion = 0,
        .pEngineName = APP_NAME,
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .pApplicationInfo = &app,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = NULL,
    };

    err = vkCreateInstance(&inst_info, NULL, &vk_state.instance);

    if (err == VK_ERROR_INCOMPATIBLE_DRIVER) {
        FAIL_PRINT("Failed to find a compatible Vulkan installable client driver (ICD).Exiting\n");
    } else if (err == VK_ERROR_EXTENSION_NOT_PRESENT) {
        FAIL_PRINT("Failed to find a specified extension library.Exiting\n");
    } else if (err) {
        FAIL_PRINT("vkCreateInstance failed.Exiting\n");
    }
    /* fetch the vkGetDeviceProcAddr function pointer */
    fpGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(vk_state.instance, "vkGetDeviceProcAddr");

    /* Make initial call to query gpu_count, then second call for gpu info */
    err = vkEnumeratePhysicalDevices(vk_state.instance, &gpu_count, NULL);
    assert(!err && gpu_count > 0);

    if (gpu_count > 0) {
        gpu_count = 1; // We blindly take the first physical device.
        err = vkEnumeratePhysicalDevices(vk_state.instance, &gpu_count, &vk_state.gpu);
        if (err != VK_SUCCESS || err != VK_INCOMPLETE) {
            assert(!err);
        }
    } else {
        FAIL_PRINT("vkEnumeratePhysicalDevices reported zero accessible devices. Exiting.\n");
    }

    /* Look for device extensions */
    memset(device_extension_names, 0, sizeof(device_extension_names));

    err = vkEnumerateDeviceExtensionProperties(vk_state.gpu, NULL,
                                               &device_extension_count, NULL);
    assert(!err);

    if (device_extension_count > 0) {
        VkExtensionProperties *device_extensions =
            malloc(sizeof(VkExtensionProperties) * device_extension_count);
        err = vkEnumerateDeviceExtensionProperties(
            vk_state.gpu, NULL, &device_extension_count, device_extensions);
        assert(!err);

        for (i = 0; i < device_extension_count; i++) {
            if (!strcmp(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
                        device_extensions[i].extensionName)) {
                device_extension_names[device_enabled_extension_count++] =
                    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
            }
            if (!strcmp(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                        device_extensions[i].extensionName)) {
                device_extension_names[device_enabled_extension_count++] =
                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
            }
        }
        if (device_enabled_extension_count != 2) {
            FAIL_PRINT("Atleast one of the required extensions 'VK_KHR_external_memory',"
                   "'VK_KHR_external_memory_fd' not found. Exiting.\n");
        }
        free(device_extensions);
    } else {
        FAIL_PRINT("Failed to query device extensions. Exiting\n");
    }

    /* Get device queue info. Call with NULL data to get count */
    vkGetPhysicalDeviceQueueFamilyProperties(vk_state.gpu,
                                             &demo_queue_family_count, NULL);
    assert(demo_queue_family_count >= 1);

    demo_queue_props = (VkQueueFamilyProperties *)malloc(
        demo_queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(
        vk_state.gpu, &demo_queue_family_count, demo_queue_props);

    for (i = 0; i < demo_queue_family_count; i++) {
        if ((demo_queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            graphicsQueueFamilyIndex = i;
            break;
        }
    }
    /* Generate error if could not find a graphics queue */
    if (graphicsQueueFamilyIndex == UINT32_MAX) {
        FAIL_PRINT("Could not find graphics queues.Exiting\n");
    }

    queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queues[0].pNext = NULL;
    queues[0].queueFamilyIndex = graphicsQueueFamilyIndex;
    queues[0].queueCount = 1;
    queues[0].pQueuePriorities = queue_priorities;
    queues[0].flags = 0;

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = queues,
        .enabledLayerCount = 0,      // deprecated and ignored
        .ppEnabledLayerNames = NULL, // deprecated and ignored
        .enabledExtensionCount = device_enabled_extension_count,
        .ppEnabledExtensionNames = (const char *const *)device_extension_names,
        .pEnabledFeatures =
            NULL, // If specific features are required, pass them in here
    };
    err = vkCreateDevice(vk_state.gpu, &device_info, NULL, &vk_state.device);
    if (err) {
        FAIL_PRINT("Failed to create Vulkan Device.Exiting\n");
    }
    return 1;
fail:
    return 0;
}

static bool getMemTypeFromProperties(uint32_t memTypeBits,
                                        VkFlags req_property_mask,
                                        uint32_t *memTypeIndex)
{
    // Search memtypes to find first index with those properties
    uint32_t ind;
    for (ind = 0; ind < VK_MAX_MEMORY_TYPES; ind++) {
        if ((memTypeBits & 1) == 1) {
            // Type is available, does it match user properties?
            if ((vk_state.memory_properties.memoryTypes[ind].propertyFlags &
                 req_property_mask) == req_property_mask) {
                *memTypeIndex = ind;
                return true;
            }
        }
        memTypeBits >>= 1;
    }
    // No matching memory types were found, return failure
    return false;
}

static int createVulkanTexImageFD(void)
{
    VkResult err;
    int exp_fd=-1;
    uint32_t memTypeIndex = 0;

    vkGetPhysicalDeviceMemoryProperties(vk_state.gpu, &vk_state.memory_properties);

    const VkImageCreateInfo image_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {TEX_WIDTH, TEX_HEIGHT, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .flags = 0,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    const VkExportMemoryAllocateInfoKHR exp_memalloc_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext = NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
    };

    err = vkCreateImage(vk_state.device, &image_create_info, NULL, &vk_state.tex_res.image);
    assert(!err);

    vkGetImageMemoryRequirements(vk_state.device, vk_state.tex_res.image, &vk_state.tex_res.mem_reqs);
    printf("Required texture memory size = %u\n", (unsigned int)vk_state.tex_res.mem_reqs.size);
    if (!getMemTypeFromProperties(vk_state.tex_res.mem_reqs.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                  &memTypeIndex)) {
        return -1;
    }

    const VkMemoryAllocateInfo memalloc_info = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
              .pNext = &exp_memalloc_info,
              .allocationSize = vk_state.tex_res.mem_reqs.size,
              .memoryTypeIndex = memTypeIndex
    };

    /* allocate memory */
    err = vkAllocateMemory(vk_state.device, &memalloc_info, NULL,
                           &vk_state.tex_res.exp_mem);
    if (err) {
        return -1;
    }

    const VkMemoryGetFdInfoKHR fd_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext = NULL,
        .memory = vk_state.tex_res.exp_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
    };

#ifdef VK_KHR_external_memory
    fpGetMemoryFdKHR =  (PFN_vkGetMemoryFdKHR)fpGetDeviceProcAddr(vk_state.device, "vkGetMemoryFdKHR");
    if (fpGetMemoryFdKHR == NULL) {
        printf("Failed to fetch vulkan extension function vkGetMemoryFdKHR");
    }
    // Get exportable FD from the vulkan memory's FD_info.
    err = fpGetMemoryFdKHR(vk_state.device, &fd_info, &exp_fd);
#else
    return -1;
#endif
    return (err ? -1 : exp_fd);
}

int main(int argc, char **argv)
{
    int sock[16], terminate = 0;
    int i, numClients = 1 /*default 1 client*/;
    char sock_id[4];
    int exp_fd = -1;
    int tex_dim[] = {TEX_WIDTH, TEX_HEIGHT};

    memset (sock, -1, 16 * sizeof(sock[0]));

    for (i=1; i<argc; i++) {
        if (!strcmp(argv[i], "--num-clients")) {
            if ((++i >= argc) || !isdigit(argv[i][0])) {
                printf("--num-clients requires an integer value [1-16] argument. Exiting.\n\n");
                return 0;
            }
            numClients = strtol(argv[i], NULL, 10);
            if (numClients < 1 || numClients > 16) {
                printf("'--num-clients must be given values from 1 to 16. Exiting.\n");
                return 0;
            }
            printf("Expecting %d clients to connect\n", numClients);
        } else {
            printf("Unknown argument - '%s'. Exiting.\n", argv[i]);
            return 0;
        }
    }

    // Generic initialization of Vulkan state.
    if (!InitVulkanState()) {
        goto fail;
    }

    // Create an exportable vulkan memory handle and get corresponding FD.
    exp_fd = createVulkanTexImageFD();
    if (exp_fd == -1) {
        FAIL_PRINT("Failed to get exportable Vulkan Memory Handle. Exiting.\n");
    } else {
        printf("Exportable FD of Vulkan memory = %d\n", exp_fd);
    }
    // Loop through the expected number of client connections, creating
    // socket conection with each of them one by one for sending texture FD and
    // texture sizes.
    for (i = 0; i < numClients; i++) {
        strcpy(socket_name[i], "/tmp/vksock");
        sprintf(sock_id, "%d", i+1);
        strcat(socket_name[i], sock_id);
        // Connect with a process that is expecting a vulkan memory handle FD.
        printf("Waiting for client process %d to connect over socket name %s ...\n", i+1, socket_name[i]);
        sock[i] = Connect();
        if (sock[i] > 0) {
            printf("Connected to client process %d\n", i+1);
        } else {
            FAIL_PRINT("Failed to connect with client process. Exiting\n");
        }
        sendFD(sock[i], exp_fd);
        printf("FD %d sent successfully\n", exp_fd);
        send(sock[i], tex_dim, 2*sizeof(int), 0);
        send(sock[i], &vk_state.tex_res.mem_reqs.size, sizeof(vk_state.tex_res.mem_reqs.size), 0);
    }
    // Wait for all clients to terminate before releasing the Vulkan resources
    while(!terminate) {
        sleep(1);
        terminate = 1;
        for (i = 0; i < numClients; i++) {
            int n = send(sock[i], 0, 0, MSG_NOSIGNAL);
            if(n >= 0) {
                terminate = 0;
                break;
            }
        }
    }
    printf("All clients finished. Terminating connections and Vulkan resources now.\n" );
    for (i = 0; i < numClients; i++) {
        close (sock[i]);
    }
fail:
    // Free Vulkan objects
    if (vk_state.tex_res.exp_mem) {
        vkFreeMemory(vk_state.device, vk_state.tex_res.exp_mem, NULL);
    }
    if (vk_state.tex_res.image) {
        vkDestroyImage(vk_state.device, vk_state.tex_res.image, NULL);
    }
    if (vk_state.device) {
        vkDestroyDevice(vk_state.device, NULL);
    }
    if (vk_state.instance) {
        vkDestroyInstance(vk_state.instance, NULL);
    }
    return 0;
}
