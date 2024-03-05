#include <glm/glm.hpp>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <VkBootstrap.h>
#include <vulkan/vulkan_core.h>

#include <string>

namespace spacevec {
struct VkData {
    GLFWwindow* window;
    vkb::Instance instance;
    vkb::InstanceDispatchTable inst_disp;
    VkSurfaceKHR surface;
    vkb::PhysicalDevice physical_device;
    vkb::Device device;
    vkb::DispatchTable disp;
    vkb::Swapchain swapchain;
};

struct RenderData {
    VkQueue graphics_queue;
    VkQueue present_queue;

    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkFramebuffer> framebuffers;

    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;

    std::vector<VkBuffer> vertex_buffers;
    std::vector<VkDeviceMemory> vertex_buffers_memory;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;

    std::vector<VkSemaphore> available_semaphores;
    std::vector<VkSemaphore> finished_semaphore;
    std::vector<VkFence> in_flight_fences;
    std::vector<VkFence> image_in_flight;
    size_t current_frame = 0;
};

struct init_result {
    std::string message;
    VkResult result;
    bool status;

    [[nodiscard]] constexpr static init_result ok() { return { .status = true }; };
    explicit operator bool() const { return status; }
  };

class Renderer {
public:
    [[nodiscard]] init_result init();
    [[nodiscard]] init_result main_loop();
    void destroy();

private:
    VkData vk_data;
    RenderData render_data;
};
} // namespace spacevec
