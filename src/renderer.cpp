#include "renderer.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

const auto app_name = "Spacevec";
const auto engine_name = "Spacevec Engine";

constexpr auto debug_validation_layers =
    std::array{"VK_LAYER_KHRONOS_validation"};
constexpr auto required_device_extensions = std::array{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

namespace engine {
GLFWwindow *init_window() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  return glfwCreateWindow(800, 600, app_name, nullptr, nullptr);
}

constexpr bool check_validation_layers(const auto &validation_layers) {
  return std::all_of(debug_validation_layers.begin(),
                     debug_validation_layers.end(), [&](auto requested) {
                       return std::any_of(
                           validation_layers.begin(), validation_layers.end(),
                           [&](auto layer) {
                             return std::strcmp(requested, layer.layerName) ==
                                    0;
                           });
                     });
}

vk::Instance init_instance(bool debug_mode) {
  auto app_info = vk::ApplicationInfo{.pApplicationName = app_name,
                                      .applicationVersion = 1,
                                      .pEngineName = engine_name,
                                      .engineVersion = 1,
                                      .apiVersion = VK_API_VERSION_1_0};

  uint32_t ext_count;
  auto extensions_ptr = glfwGetRequiredInstanceExtensions(&ext_count);
  auto create_info =
      vk::InstanceCreateInfo{.pApplicationInfo = &app_info,
                             .enabledExtensionCount = ext_count,
                             .ppEnabledExtensionNames = extensions_ptr};

  auto extensions = std::vector(extensions_ptr, extensions_ptr + ext_count);
  // TODO: Check extensions support

  if (debug_mode) {
    // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    auto validation_layers = vk::enumerateInstanceLayerProperties();

    if (check_validation_layers(validation_layers)) {
      create_info.enabledLayerCount = debug_validation_layers.size();
      create_info.ppEnabledLayerNames = debug_validation_layers.data();
    }
  }
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  return vk::createInstance(create_info, nullptr);
}

/*static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT message_type,
               const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
               void *user_data) {
  std::cerr << "validation layer: " << callback_data->pMessage << '\n';

  return VK_FALSE;
}*/

/*vk::DebugUtilsMessengerEXT init_debug_messenger() {
  using enum vk::DebugUtilsMessageSeverityFlagBitsEXT;
  using enum vk::DebugUtilsMessageTypeFlagBitsEXT;

  auto messenger_info = vk::DebugUtilsMessengerCreateInfoEXT{
      .messageSeverity = eVerbose | eWarning | eError,
      .messageType = eGeneral | eValidation | ePerformance,
      .pfnUserCallback = debug_callback};

  return vk::DebugUtilsMessengerEXT();
}*/

vk::SurfaceKHR init_surface(const vk::Instance &instance, GLFWwindow *window) {
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }

  return vk::SurfaceKHR(surface);
}

void filter_suitable_physical_devices(
    std::vector<vk::PhysicalDevice> &physical_devices,
    const vk::SurfaceKHR &surface) {
  std::erase_if(physical_devices, [&](const vk::PhysicalDevice &device) {
    const auto surface_formats = device.getSurfaceFormatsKHR(surface);
    const auto present_modes = device.getSurfacePresentModesKHR(surface);
    const auto supported_extensions =
        device.enumerateDeviceExtensionProperties();
    bool extensions_ok = std::all_of(
        required_device_extensions.begin(), required_device_extensions.end(),
        [&](const char *required) {
          return std::any_of(
              supported_extensions.begin(), supported_extensions.end(),
              [&](const vk::ExtensionProperties &extension) {
                return std::strcmp(extension.extensionName, required) == 0;
              });
        });

    bool deleted =
        surface_formats.empty() || present_modes.empty() || !extensions_ok;

    return deleted;
  });
}

struct DeviceData {
  vk::PhysicalDevice device;
  int weight;
  int graphics_queue;
};

void weight_physical_devices(
    std::vector<DeviceData> &weighted_devices,
    const std::vector<vk::PhysicalDevice> physical_devices,
    const vk::SurfaceKHR &surface) {
  std::ranges::transform(
      physical_devices, std::back_inserter(weighted_devices),
      [&](const vk::PhysicalDevice &device) {
        auto properties = device.getProperties();
        auto queue_families = device.getQueueFamilyProperties();
        int weight = 0;
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
          weight = 1;
        else if (properties.deviceType ==
                 vk::PhysicalDeviceType::eIntegratedGpu)
          weight = 2;
        else if (properties.deviceType == vk::PhysicalDeviceType::eVirtualGpu)
          weight = 3;
        else if (properties.deviceType == vk::PhysicalDeviceType::eCpu)
          weight = 4;

        int index = 0;
        int family_index = -1;
        for (const auto &queue_family : queue_families) {
          if ((queue_family.queueFlags & vk::QueueFlagBits::eGraphics) &&
              device.getSurfaceSupportKHR(index, surface))
            family_index = index;
          index++;
        }

        auto surface_formats = device.getSurfaceFormatsKHR(surface);
        auto present_modes = device.getSurfacePresentModesKHR(surface);

        return DeviceData{device, weight, family_index};
      });
}

DeviceData init_physical_device(const vk::Instance &instance,
                                const vk::SurfaceKHR &surface) {
  auto physical_devices = instance.enumeratePhysicalDevices();

  filter_suitable_physical_devices(physical_devices, surface);

  if (physical_devices.empty())
    throw std::runtime_error("No available supported Vulkan GPU.");

  auto weighted_devices = std::vector<DeviceData>{};
  weight_physical_devices(weighted_devices, physical_devices, surface);

  auto result = std::ranges::min_element(
      weighted_devices,
      [](const auto &a, const auto &b) { return a.weight < b.weight; });

  if (result->graphics_queue == -1) {
    throw std::runtime_error("No suitable graphics GPU.");
  }

  return *result;
}

vk::Device init_logical_device(const vk::PhysicalDevice &physical_device,
                               uint32_t family_index) {
  auto queue_priority = 1.0f;
  auto device_queue_create_info =
      vk::DeviceQueueCreateInfo{.queueFamilyIndex = family_index,
                                .queueCount = 1,
                                .pQueuePriorities = &queue_priority};

  auto physical_device_features = vk::PhysicalDeviceFeatures{};
  auto device_create_info = vk::DeviceCreateInfo{
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &device_queue_create_info,
      .enabledExtensionCount = required_device_extensions.size(),
      .ppEnabledExtensionNames = required_device_extensions.data(),
      .pEnabledFeatures = &physical_device_features,
  };

  return physical_device.createDevice(device_create_info);
}

struct SwapchainData {
  vk::SwapchainKHR swapchain;
  vk::SurfaceFormatKHR surface_format;
  vk::Extent2D extent;
};

auto init_swapchain(const vk::PhysicalDevice &physical_device,
                    const vk::SurfaceKHR &surface, GLFWwindow *window,
                    const vk::Device &device) -> SwapchainData {
  const auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
  const auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
  const auto surface_capabilities =
      physical_device.getSurfaceCapabilitiesKHR(surface);

  auto surface_format = surface_formats[0];
  for (const auto &format : surface_formats) {
    if (format.format == vk::Format::eB8G8R8A8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      surface_format = format;
      break;
    }
  }
  auto present_mode = vk::PresentModeKHR::eFifo;
  for (const auto &mode : present_modes) {
    if (mode == vk::PresentModeKHR::eMailbox) {
      present_mode = mode;
      break;
    }
  }

  vk::Extent2D extent;
  if (surface_capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    extent = surface_capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    extent.width = std::clamp(static_cast<uint32_t>(width),
                              surface_capabilities.minImageExtent.width,
                              surface_capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<uint32_t>(height),
                               surface_capabilities.minImageExtent.height,
                               surface_capabilities.maxImageExtent.height);
  }

  uint32_t image_count = surface_capabilities.minImageCount + 1;
  if (surface_capabilities.maxImageCount > 0 &&
      image_count > surface_capabilities.maxImageCount) {
    image_count = surface_capabilities.maxImageCount;
  }

  auto swapchain_create_info = vk::SwapchainCreateInfoKHR{
      .surface = surface,
      .minImageCount = image_count,
      .imageFormat = surface_format.format,
      .imageColorSpace = surface_format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = surface_capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = present_mode,
      .clipped = VK_TRUE,
  };

  return {device.createSwapchainKHR(swapchain_create_info), surface_formats[0],
          extent};
};

auto init_image_views(const vk::Device &device,
                      const std::vector<vk::Image> &images,
                      const vk::Format &image_format) {
  auto image_views = std::vector<vk::ImageView>{};
  for (const auto &image : images) {
    auto image_view_create_info = vk::ImageViewCreateInfo{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = image_format,
        .components = {vk::ComponentSwizzle::eIdentity},
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = 1}};
    image_views.push_back(device.createImageView(image_view_create_info));
  }

  return image_views;
};

auto load_shader(const char *file_name, const vk::Device &device) {
  auto file = std::ifstream{};
  file.open(file_name, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error(strerror(errno));
  }
  auto file_size = static_cast<size_t>(file.tellg());
  auto buffer = std::vector<char>(file_size);

  file.seekg(0);
  file.read(buffer.data(), file_size);

  auto shader_module_create_info = vk::ShaderModuleCreateInfo{
      .codeSize = file_size,
      .pCode = reinterpret_cast<const uint32_t *>(buffer.data())};
  return device.createShaderModule(shader_module_create_info);
}

auto init_render_pass(const vk::Device &device, const vk::Format format) {
  auto color_attachment = vk::AttachmentDescription{
      .format = format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR,
  };
  auto color_attachment_ref = vk::AttachmentReference{
      .layout = vk::ImageLayout::eColorAttachmentOptimal};
  auto subpass = vk::SubpassDescription{
      .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment_ref,
  };
  auto subpass_dependency = vk::SubpassDependency{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .srcAccessMask = {},
      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
  };

  auto render_pass_create_info = vk::RenderPassCreateInfo{
      .attachmentCount = 1,
      .pAttachments = &color_attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 1,
      .pDependencies = &subpass_dependency,
  };
  return device.createRenderPass(render_pass_create_info);
}

auto init_graphics_pipeline(const vk::Device &device,
                            const vk::ShaderModule &vertex_shader_module,
                            const vk::ShaderModule &fragment_shader_module,
                            const vk::Extent2D &extent,
                            const vk::RenderPass &render_pass) {
  auto vertex_stage_create_info = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = vertex_shader_module,
      .pName = "main"};
  auto fragment_stage_create_info = vk::PipelineShaderStageCreateInfo{
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = fragment_shader_module,
      .pName = "main"};

  auto shader_stages_create_info =
      std::array{vertex_stage_create_info, fragment_stage_create_info};

  auto dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo{};

  auto vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo{};
  auto assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::eTriangleList,
      .primitiveRestartEnable = VK_FALSE,
  };

  auto viewport = vk::Viewport{.x = 0.0f,
                               .y = 0.0f,
                               .width = static_cast<float>(extent.width),
                               .height = static_cast<float>(extent.height),
                               .minDepth = 0.0f,
                               .maxDepth = 1.0f};
  auto scissor = vk::Rect2D{.offset{0, 0}, .extent = extent};

  auto viewport_create_info = vk::PipelineViewportStateCreateInfo{
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
  };

  auto rasterization_create_info = vk::PipelineRasterizationStateCreateInfo{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eClockwise,
      .lineWidth = 1.0f,
  };

  auto multisample_create_info = vk::PipelineMultisampleStateCreateInfo{};

  auto color_blend_attachment = vk::PipelineColorBlendAttachmentState{
      .colorWriteMask =
          vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };
  auto color_blend_create_info = vk::PipelineColorBlendStateCreateInfo{
      .attachmentCount = 1, .pAttachments = &color_blend_attachment};

  auto pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{};
  auto pipeline_layout =
      device.createPipelineLayout(pipeline_layout_create_info);

  auto graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo{
      .stageCount = 2,
      .pStages = shader_stages_create_info.data(),
      .pVertexInputState = &vertex_input_create_info,
      .pInputAssemblyState = &assembly_create_info,
      .pViewportState = &viewport_create_info,
      .pRasterizationState = &rasterization_create_info,
      .pMultisampleState = &multisample_create_info,
      .pDepthStencilState = nullptr,
      .pColorBlendState = &color_blend_create_info,
      .pDynamicState = &dynamic_state_create_info,
      .layout = pipeline_layout,
      .renderPass = render_pass,
      .subpass = 0,
  };
  auto [result, graphics_pipeline] =
      device.createGraphicsPipeline(nullptr, graphics_pipeline_create_info);

  struct PipelineData {
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
  };

  return PipelineData{pipeline_layout, graphics_pipeline};
}

auto init_framebuffer(const vk::Device &device,
                      const std::vector<vk::ImageView> &image_views,
                      const vk::RenderPass &render_pass,
                      const vk::Extent2D &extent) {
  auto swapchain_framebuffers = std::vector<vk::Framebuffer>();
  for (const auto &image_view : image_views) {
    auto attachments = std::array{image_view};

    auto framebuffer_create_info = vk::FramebufferCreateInfo{
        .renderPass = render_pass,
        .attachmentCount = 1,
        .pAttachments = attachments.data(),
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };

    swapchain_framebuffers.push_back(
        device.createFramebuffer(framebuffer_create_info));
  }
  return swapchain_framebuffers;
}

auto init_command_pool(const vk::Device &device, uint32_t family_index) {
  auto command_pool_create_info = vk::CommandPoolCreateInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = family_index};

  return device.createCommandPool(command_pool_create_info);
}

auto init_command_buffer(const vk::Device &device,
                         const vk::CommandPool &command_pool) {
  auto command_buffer_allocate_info = vk::CommandBufferAllocateInfo{
      .commandPool = command_pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };

  return device.allocateCommandBuffers(command_buffer_allocate_info);
}

auto record_command_buffer(const vk::CommandBuffer &command_buffer,
                           const vk::RenderPass &render_pass,
                           const vk::Framebuffer &framebuffer,
                           const vk::Extent2D &extent,
                           const vk::Pipeline &pipeline) {
  auto command_buffer_begin_info = vk::CommandBufferBeginInfo{};
  command_buffer.begin(command_buffer_begin_info);

  auto clear_color =
      vk::ClearValue{.color = {std::array{0.0f, 0.0f, 0.0f, 1.0f}}};
  auto render_pass_begin_info = vk::RenderPassBeginInfo{
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = extent},
      .clearValueCount = 1,
      .pClearValues = &clear_color,
  };
  command_buffer.beginRenderPass(render_pass_begin_info,
                                 vk::SubpassContents::eInline);
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
  command_buffer.draw(3, 1, 0, 0);
  command_buffer.endRenderPass();
  command_buffer.end();
}

auto init_sync_objects(const vk::Device &device) {
  auto image_semaphore = device.createSemaphore({});
  auto render_semaphore = device.createSemaphore({});
  auto frame_fence =
      device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled});

  struct SyncObjects {
    vk::Semaphore image_semaphore;
    vk::Semaphore render_semaphore;
    vk::Fence frame_fence;
  };
  return SyncObjects{image_semaphore, render_semaphore, frame_fence};
}

renderer::renderer() {
  m_window = init_window();
  m_instance = init_instance(true);
  // m_debug_messenger = init_debug_messenger();
  m_surface = init_surface(m_instance, m_window);
  auto const &[physical_device, weight, index] =
      init_physical_device(m_instance, m_surface);
  m_physical_device = physical_device;
  m_graphics_queue_index = index;
  m_device = init_logical_device(m_physical_device, m_graphics_queue_index);
  const auto &[swapchain, surface_format, extent] =
      init_swapchain(m_physical_device, m_surface, m_window, m_device);
  m_swapchain = swapchain;
  m_swapchain_surface_format = surface_format;
  m_swapchain_extent = extent;
  m_graphics_queue = m_device.getQueue(m_graphics_queue_index, 0);
  m_swapchain_images = m_device.getSwapchainImagesKHR(m_swapchain);
  m_swapchain_image_views = init_image_views(m_device, m_swapchain_images,
                                             m_swapchain_surface_format.format);
  m_render_pass = init_render_pass(m_device, m_swapchain_surface_format.format);
  m_vertex_shader_module = load_shader("vertex.spv", m_device);
  m_fragment_shader_module = load_shader("fragment.spv", m_device);

  const auto &[pipeline_layout, pipeline] = init_graphics_pipeline(
      m_device, m_vertex_shader_module, m_fragment_shader_module,
      m_swapchain_extent, m_render_pass);
  m_pipeline_layout = pipeline_layout;
  m_pipeline = pipeline;

  m_swapchain_framebuffers = init_framebuffer(
      m_device, m_swapchain_image_views, m_render_pass, m_swapchain_extent);
  m_command_pool = init_command_pool(m_device, m_graphics_queue_index);
  m_command_buffers = init_command_buffer(m_device, m_command_pool);

  const auto &[image_semaphore, render_semaphore, frame_fence] =
      init_sync_objects(m_device);
  m_image_semaphore = image_semaphore;
  m_render_semaphore = render_semaphore;
  m_inflight_fence = frame_fence;
}

renderer::~renderer() {
  m_device.destroySemaphore(m_image_semaphore);
  m_device.destroySemaphore(m_render_semaphore);
  m_device.destroyFence(m_inflight_fence);

  m_device.destroyCommandPool(m_command_pool);
  std::ranges::for_each(m_swapchain_framebuffers, [&](auto &framebuffer) {
    m_device.destroyFramebuffer(framebuffer);
  });
  m_device.destroyPipeline(m_pipeline);
  m_device.destroyPipelineLayout(m_pipeline_layout);
  m_device.destroyRenderPass(m_render_pass);
  m_device.destroyShaderModule(m_fragment_shader_module);
  m_device.destroyShaderModule(m_vertex_shader_module);
  std::ranges::for_each(m_swapchain_image_views, [&](const auto &image_view) {
    m_device.destroyImageView(image_view);
  });
  m_device.destroySwapchainKHR(m_swapchain);
  m_device.destroy();
  m_instance.destroySurfaceKHR(m_surface);
  glfwDestroyWindow(m_window);
  m_instance.destroy();
}

void renderer::init() {
  // pass
}

void renderer::main_loop() {
  while (!glfwWindowShouldClose(m_window)) {
    glfwPollEvents();
    draw_frame();
  }

  m_device.waitIdle();
}

void renderer::draw_frame() {
  // Actually draw stuff
  [[maybe_unused]] auto result = m_device.waitForFences(
      1, &m_inflight_fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
  [[maybe_unused]] auto result2 = m_device.resetFences(1, &m_inflight_fence);

  auto [result3, image_index] = m_device.acquireNextImageKHR(
      m_swapchain, std::numeric_limits<uint64_t>::max(), m_image_semaphore);

  m_command_buffers[0].reset();
  record_command_buffer(m_command_buffers[0], m_render_pass,
                        m_swapchain_framebuffers[image_index],
                        m_swapchain_extent, m_pipeline);

  auto wait_stages =
      vk::PipelineStageFlags{vk::PipelineStageFlagBits::eColorAttachmentOutput};
  auto submit_info = vk::SubmitInfo{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &m_image_semaphore,
      .pWaitDstStageMask = &wait_stages,
      .commandBufferCount = static_cast<uint32_t>(m_command_buffers.size()),
      .pCommandBuffers = m_command_buffers.data(),
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &m_render_semaphore,
  };
  [[maybe_unused]] auto result4 =
      m_graphics_queue.submit(1, &submit_info, m_inflight_fence);

  auto present_info = vk::PresentInfoKHR{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &m_render_semaphore,
      .swapchainCount = 1,
      .pSwapchains = &m_swapchain,
      .pImageIndices = &image_index,
  };
  [[maybe_unused]] auto result5 = m_graphics_queue.presentKHR(present_info);
}

} // namespace engine
