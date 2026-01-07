const std = @import("std");
const testing = std.testing;
const gpu_zig = @import("gpu");
const Buffer = gpu_zig.Buffer;
const ComputePipeline = gpu_zig.ComputePipeline;
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
    const title = std.fmt.comptimePrint("Input Buffer Test - {s}", .{gpu_zig.options.git_commit_hash});
    return GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        .validation()
        .build();
}

// Simple compute shader embedded as bytes (compiled from GLSL)
// This shader reads from input buffer and adds 100.0: out_buf.values[gid] = in_buf.values[gid] + 100.0;
const simple_compute_spv align(@alignOf(u32)) = @embedFile("simple_test.spv").*;

test "minimal compute shader data access - 4 elements with staging buffers" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    var gpu = try createTestGPU();
    defer gpu.deinit();

    // Test data: 4 float values
    var test_data = [4]f32{ 1.0, 2.0, 3.0, 4.0 };
    const expected_result = [4]f32{ 101.0, 102.0, 103.0, 104.0 };

    // Create staging buffers
    var in_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .hostToDevice()
        .build();
    try in_buf.allocate(&gpu);
    defer in_buf.deinit(&gpu);

    var out_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .deviceToHost()
        .build();
    try out_buf.allocate(&gpu);
    defer out_buf.deinit(&gpu);

    // Create device buffers
    var dev_in_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_dst_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .device()
        .build();
    try dev_in_buf.allocate(&gpu);
    defer dev_in_buf.deinit(&gpu);

    var dev_out_buf: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_src_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .device()
        .build();
    try dev_out_buf.allocate(&gpu);
    defer dev_out_buf.deinit(&gpu);

    // Create compute pipeline with device buffers
    var pipeline = try ComputePipeline.Builder.init(&gpu)
        .code(&simple_compute_spv)
        .buffer(&dev_in_buf)
        .buffer(&dev_out_buf)
        .build();
    defer pipeline.deinit(&gpu);

    // Create command pool and buffer
    const cmd_pool = try gpu.dev.createCommandPool(&vk.CommandPoolCreateInfo{
        .queue_family_index = gpu.compute_queue.?.family,
        .flags = vk.CommandPoolCreateFlags{ .transient_bit = true },
    }, null);
    defer gpu.dev.destroyCommandPool(cmd_pool, null);

    var cmd_buffer: vk.CommandBuffer = undefined;
    try gpu.dev.allocateCommandBuffers(&vk.CommandBufferAllocateInfo{
        .command_pool = cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmd_buffer));
    defer gpu.dev.freeCommandBuffers(cmd_pool, 1, @ptrCast(&cmd_buffer));

    // Write test data to staging buffer
    try in_buf.writeToBuffer(&gpu, std.mem.sliceAsBytes(&test_data));

    // Record command buffer
    try gpu.dev.beginCommandBuffer(cmd_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    });

    // Copy staging → device
    try in_buf.copyBufferDataTo(&gpu, cmd_buffer, &dev_in_buf, 4 * @sizeOf(f32));

    // Transfer → compute barrier
    gpu.dev.cmdPipelineBarrier(cmd_buffer, vk.PipelineStageFlags{ .transfer_bit = true }, vk.PipelineStageFlags{ .compute_shader_bit = true }, .{}, 0, null, 1, &[_]vk.BufferMemoryBarrier{.{
        .src_access_mask = vk.AccessFlags{ .transfer_write_bit = true },
        .dst_access_mask = vk.AccessFlags{ .shader_read_bit = true },
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .buffer = dev_in_buf.handle,
        .offset = 0,
        .size = vk.WHOLE_SIZE,
    }}, 0, null);

    // Bind pipeline and dispatch compute
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

    gpu.dev.cmdDispatch(cmd_buffer, 1, 1, 1); // Only 1 workgroup for 4 elements

    // Compute → transfer barrier
    gpu.dev.cmdPipelineBarrier(cmd_buffer, vk.PipelineStageFlags{ .compute_shader_bit = true }, vk.PipelineStageFlags{ .transfer_bit = true }, .{}, 0, null, 1, &[_]vk.BufferMemoryBarrier{.{
        .src_access_mask = vk.AccessFlags{ .shader_write_bit = true },
        .dst_access_mask = vk.AccessFlags{ .transfer_read_bit = true },
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .buffer = dev_out_buf.handle,
        .offset = 0,
        .size = vk.WHOLE_SIZE,
    }}, 0, null);

    // Copy device → staging
    try dev_out_buf.copyBufferDataTo(&gpu, cmd_buffer, &out_buf, 4 * @sizeOf(f32));

    try gpu.dev.endCommandBuffer(cmd_buffer);

    // Submit and wait
    try gpu.dev.queueSubmit(gpu.compute_queue.?.handle, 1, &[_]vk.SubmitInfo{.{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd_buffer),
    }}, .null_handle);

    try gpu.dev.queueWaitIdle(gpu.compute_queue.?.handle);

    // Read results from staging buffer
    var result_data: [4]f32 = undefined;
    try out_buf.readFromBuffer(&gpu, std.mem.sliceAsBytes(&result_data));

    // Print debug information
    std.debug.print("\nTest Results:\n", .{});
    std.debug.print("Input:    [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ test_data[0], test_data[1], test_data[2], test_data[3] });
    std.debug.print("Expected: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ expected_result[0], expected_result[1], expected_result[2], expected_result[3] });
    std.debug.print("Actual:   [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ result_data[0], result_data[1], result_data[2], result_data[3] });

    for (0..4) |i| {
        try testing.expectApproxEqRel(expected_result[i], result_data[i], 0.001);
    }
}
