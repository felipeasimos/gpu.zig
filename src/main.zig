const std = @import("std");
const gpu_zig = @import("gpu");
const zglfw = @import("zglfw");
const vk = gpu_zig.vk;
const Buffer = gpu_zig.Buffer;
const ComputePipeline = gpu_zig.ComputePipeline;
const Swapchain = gpu_zig.Swapchain;
const GPU = gpu_zig.GPU;

const square_spv align(@alignOf(u32)) = @embedFile("square").*;

// Image dimensions
const WIDTH = 512;
const HEIGHT = 512;

pub fn main() !void {
    var debug_allocator = std.heap.DebugAllocator(.{
        .safety = true,
        .retain_metadata = true,
        .backing_allocator_zeroes = false,
        .thread_safe = true,
    }).init;
    const allocator = debug_allocator.allocator();

    try zglfw.init();
    defer zglfw.terminate();

    if (!zglfw.isVulkanSupported()) {
        std.log.err("GLFW could not find libvulkan", .{});
        return error.NoVulkan;
    }
    zglfw.windowHint(.client_api, .no_api);

    const title = std.fmt.comptimePrint("gpu.zig - {s}", .{gpu_zig.options.git_commit_hash});
    const window = try zglfw.Window.create(WIDTH, HEIGHT, title, null);
    defer window.destroy();
    window.setSizeLimits(WIDTH, HEIGHT, -1, -1);

    // Enable mouse capture for FPS-style camera controls
    try window.setInputMode(.cursor, .disabled);

    var gpu = try GPU.Builder.init(allocator)
        .appName(title)
        .compute()
        // .graphics(window)
        .validation()
        .build();
    defer gpu.deinit();

    var in_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .hostToDevice()
        .build();
    defer in_buf.deinit(&gpu);
    try in_buf.allocate(&gpu);

    var out_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .deviceToHost()
        .build();
    defer out_buf.deinit(&gpu);
    try out_buf.allocate(&gpu);

    var pipeline = try ComputePipeline.Builder.init(&gpu)
        .code(&square_spv)
        .buffer(in_buf)
        .buffer(out_buf)
        .build();
    defer pipeline.deinit(&gpu);

    const cmd_pool = try gpu.dev.createCommandPool(&vk.CommandPoolCreateInfo{
        .queue_family_index = gpu.compute_queue.?.family,
        .flags = vk.CommandPoolCreateFlags{
            .transient_bit = true,
        },
    }, null);
    defer gpu.dev.destroyCommandPool(cmd_pool, null);

    var cmd_buffer: vk.CommandBuffer = undefined;
    try gpu.dev.allocateCommandBuffers(&vk.CommandBufferAllocateInfo{
        .command_pool = cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmd_buffer));
    defer gpu.dev.freeCommandBuffers(cmd_pool, 1, @ptrCast(&cmd_buffer));

    var data: [1024]f32 = undefined;
    for (0..1024) |i| {
        data[i] = @floatFromInt(i);
    }
    try in_buf.writeToBuffer(&gpu, std.mem.sliceAsBytes(&data));
    try gpu.dev.beginCommandBuffer(cmd_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{
            .one_time_submit_bit = true,
        },
    });
    gpu.dev.cmdBindPipeline(cmd_buffer, .compute, pipeline.pipeline);
    gpu.dev.cmdBindDescriptorSets(
        cmd_buffer,
        .compute,
        pipeline.pipeline_layout,
        0,
        1,
        @ptrCast(&pipeline.desc_set),
        0,
        null,
    );

    gpu.dev.cmdDispatch(cmd_buffer, 16, 1, 1);

    try gpu.dev.endCommandBuffer(cmd_buffer);
    try gpu.dev.queueSubmit(gpu.compute_queue.?.handle, 1, &[_]vk.SubmitInfo{.{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd_buffer),
    }}, .null_handle);

    try gpu.dev.queueWaitIdle(gpu.compute_queue.?.handle);
    try out_buf.readFromBuffer(&gpu, std.mem.sliceAsBytes(&data));

    var correct = true;
    for (0..1024) |i| {
        if (data[i] != @as(f32, @floatFromInt(i)) * 2.0) {
            correct = false;
            std.debug.print("wrong: data[{}] = {}\n", .{
                i,
                data[i],
            });
            return error.Wrong;
        }
    }
}
