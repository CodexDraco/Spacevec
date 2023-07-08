#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_handles.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace engine {
class renderer {
public:
  renderer();
  ~renderer();

  void init();
  void main_loop();
  void draw_frame();

private:
  GLFWwindow *m_window;
  vk::Instance m_instance;
  // vk::DebugUtilsMessengerEXT m_debug_messenger;
  vk::SurfaceKHR m_surface;
  vk::PhysicalDevice m_physical_device;
  uint32_t m_graphics_queue_index;
  vk::Device m_device;
  vk::SwapchainKHR m_swapchain;
  vk::Queue m_graphics_queue;
  std::vector<vk::Image> m_swapchain_images;
  vk::SurfaceFormatKHR m_swapchain_surface_format;
  vk::Extent2D m_swapchain_extent;
  std::vector<vk::ImageView> m_swapchain_image_views;
  vk::ShaderModule m_vertex_shader_module;
  vk::ShaderModule m_fragment_shader_module;
  vk::RenderPass m_render_pass;
  vk::PipelineLayout m_pipeline_layout;
  vk::Pipeline m_pipeline;
  std::vector<vk::Framebuffer> m_swapchain_framebuffers;
  vk::CommandPool m_command_pool;
  std::vector<vk::CommandBuffer> m_command_buffers;
  // Sync stuff
  vk::Semaphore m_image_semaphore;
  vk::Semaphore m_render_semaphore;
  vk::Fence m_inflight_fence;
};
} // namespace engine
