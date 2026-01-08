const std = @import("std");
const gpu_zig = @import("gpu");
const zglfw = gpu_zig.zglfw;
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
        .usage(.{ .storage_buffer_bit = true, .transfer_src_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .hostToDevice()
        .build();
    try in_buf.allocate(&gpu);
    defer in_buf.deinit(&gpu);

    var dev_in_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_dst_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .device()
        .build();
    try dev_in_buf.allocate(&gpu);
    defer dev_in_buf.deinit(&gpu);

    var out_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_dst_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .deviceToHost()
        .build();
    defer out_buf.deinit(&gpu);
    try out_buf.allocate(&gpu);

    var dev_out_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_src_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(1024 * @sizeOf(f32))
        .device()
        .build();
    defer dev_out_buf.deinit(&gpu);
    try dev_out_buf.allocate(&gpu);

    std.debug.print("in_buf: {f}\n", .{in_buf});
    std.debug.print("dev_in_buf: {f}\n", .{dev_in_buf});
    std.debug.print("out_buf: {f}\n", .{out_buf});
    std.debug.print("dev_out_buf: {f}\n", .{dev_out_buf});

    var pipeline = try ComputePipeline.Builder.init(&gpu)
        .code(&square_spv)
        .buffer(&dev_in_buf)
        .buffer(&dev_out_buf)
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
    var result: [1024]f32 = undefined;
    for (0..1024) |i| {
        data[i] = @floatFromInt(i);
    }
    try in_buf.writeToBuffer(&gpu, std.mem.sliceAsBytes(&data));
    try gpu.dev.beginCommandBuffer(cmd_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{
            .one_time_submit_bit = true,
        },
    });
    try in_buf.copyBufferDataTo(&gpu, cmd_buffer, &dev_in_buf, 1024 * @sizeOf(f32));
    gpu.dev.cmdPipelineBarrier(cmd_buffer, vk.PipelineStageFlags{ .transfer_bit = true }, // Source: transfer operation
        vk.PipelineStageFlags{ .compute_shader_bit = true }, // Destination: compute shader
        .{}, 0, null, // No memory barriers
        1, &[_]vk.BufferMemoryBarrier{.{
            .src_access_mask = vk.AccessFlags{ .transfer_write_bit = true },
            .dst_access_mask = vk.AccessFlags{ .shader_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = dev_in_buf.handle, // Device buffer that was written to
            .offset = 0,
            .size = vk.WHOLE_SIZE,
        }}, 0, null // No image barriers
    );
    pipeline.bind(&gpu, cmd_buffer);

    gpu.dev.cmdDispatch(cmd_buffer, 16, 1, 1);
    gpu.dev.cmdPipelineBarrier(cmd_buffer, vk.PipelineStageFlags{ .compute_shader_bit = true }, // Source: compute shader
        vk.PipelineStageFlags{ .transfer_bit = true }, // Destination: transfer operation
        .{}, 0, null, 1, &[_]vk.BufferMemoryBarrier{.{
            .src_access_mask = vk.AccessFlags{ .shader_write_bit = true },
            .dst_access_mask = vk.AccessFlags{ .transfer_read_bit = true },
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = dev_out_buf.handle, // Device buffer that was written by compute
            .offset = 0,
            .size = vk.WHOLE_SIZE,
        }}, 0, null);

    try dev_out_buf.copyBufferDataTo(&gpu, cmd_buffer, &out_buf, 1024 * @sizeOf(f32));

    try gpu.dev.endCommandBuffer(cmd_buffer);
    try gpu.dev.queueSubmit(gpu.compute_queue.?.handle, 1, &[_]vk.SubmitInfo{.{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd_buffer),
    }}, .null_handle);

    try gpu.dev.queueWaitIdle(gpu.compute_queue.?.handle);
    try out_buf.readFromBuffer(&gpu, std.mem.sliceAsBytes(&result));

    var errors: u32 = 0;
    for (0..1024) |i| {
        if (result[i] != @as(f32, @floatFromInt(i)) * 2.0) {
            errors += 1;
            std.debug.print("wrong: result[{}] = {}\n", .{
                i,
                result[i],
            });
            if (errors > 5) {
                return error.Wrong;
            }
        }
    }
}
