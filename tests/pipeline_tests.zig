const std = @import("std");
const testing = std.testing;
const gpu_zig = @import("gpu");
const ComputePipeline = gpu_zig.ComputePipeline;
const Buffer = gpu_zig.Buffer;
const GPU = gpu_zig.GPU;
const vk = gpu_zig.vk;
const zglfw = gpu_zig.zglfw;

fn initZglfwForVulkan() !void {
    try zglfw.init();
    if (!zglfw.isVulkanSupported()) {
        return error.NoVulkan;
    }
    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.visible, false);
}

fn createTestGPU() !GPU {
    const title = std.fmt.comptimePrint("Pipeline Test - {s}", .{gpu_zig.options.git_commit_hash});
    return GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        .build();
}
