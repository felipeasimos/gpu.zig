const std = @import("std");
const testing = std.testing;
const gpu_zig = @import("gpu");
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
    const title = std.fmt.comptimePrint("Buffer Copy Test - {s}", .{gpu_zig.options.git_commit_hash});
    return GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        .validation()
        .build();
}

test "staging buffer copy operations - host to device to host" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    var gpu = try createTestGPU();
    defer gpu.deinit();

    // Test data: 4 float values
    var test_data = [4]f32{ 10.0, 20.0, 30.0, 40.0 };

    // Create staging buffers
    var input_staging: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .hostToDevice()
        .build();
    try input_staging.allocate(&gpu);
    defer input_staging.deinit(&gpu);

    var output_staging: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .deviceToHost()
        .build();
    try output_staging.allocate(&gpu);
    defer output_staging.deinit(&gpu);

    // Create device buffers
    var device_buffer: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true, .transfer_dst_bit = true, .transfer_src_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(4 * @sizeOf(f32))
        .device()
        .build();
    try device_buffer.allocate(&gpu);
    defer device_buffer.deinit(&gpu);

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

    std.debug.print("\n=== Testing Staging Buffer Copy Operations ===\n", .{});

    // Step 1: Write test data to input staging buffer
    std.debug.print("Step 1: Writing data to input staging buffer...\n", .{});
    try input_staging.writeToBuffer(&gpu, std.mem.sliceAsBytes(&test_data));
    std.debug.print("Written: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ test_data[0], test_data[1], test_data[2], test_data[3] });

    // Step 2: Since we can't read from hostToDevice buffers, skip verification
    // and proceed directly to copy operations
    std.debug.print("Step 2: Skipping read-back (hostToDevice buffers are write-only)\n", .{});

    // Step 3: Copy from input staging to device buffer
    std.debug.print("Step 3: Copying input staging → device buffer...\n", .{});

    try gpu.dev.beginCommandBuffer(cmd_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    });

    try input_staging.copyBufferDataTo(&gpu, cmd_buffer, &device_buffer, 4 * @sizeOf(f32));

    // Add barrier to ensure copy completes
    gpu.dev.cmdPipelineBarrier(cmd_buffer, vk.PipelineStageFlags{ .transfer_bit = true }, vk.PipelineStageFlags{ .transfer_bit = true }, .{}, 0, null, 1, &[_]vk.BufferMemoryBarrier{.{
        .src_access_mask = vk.AccessFlags{ .transfer_write_bit = true },
        .dst_access_mask = vk.AccessFlags{ .transfer_read_bit = true },
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .buffer = device_buffer.handle,
        .offset = 0,
        .size = vk.WHOLE_SIZE,
    }}, 0, null);

    // Step 4: Copy from device buffer to output staging
    std.debug.print("Step 4: Copying device buffer → output staging...\n", .{});

    try device_buffer.copyBufferDataTo(&gpu, cmd_buffer, &output_staging, 4 * @sizeOf(f32));

    try gpu.dev.endCommandBuffer(cmd_buffer);

    // Submit and wait
    try gpu.dev.queueSubmit(gpu.compute_queue.?.handle, 1, &[_]vk.SubmitInfo{.{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd_buffer),
    }}, .null_handle);

    try gpu.dev.queueWaitIdle(gpu.compute_queue.?.handle);

    // Step 5: Read results from output staging buffer
    std.debug.print("Step 5: Reading from output staging buffer...\n", .{});
    var result_data: [4]f32 = undefined;
    try output_staging.readFromBuffer(&gpu, std.mem.sliceAsBytes(&result_data));
    std.debug.print("Final result: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{ result_data[0], result_data[1], result_data[2], result_data[3] });

    // Step 6: Verify the complete chain worked
    std.debug.print("Step 6: Verifying complete data chain...\n", .{});
    for (0..4) |i| {
        std.debug.print("  Element {}: Original={d:.1}, Final={d:.1}\n", .{ i, test_data[i], result_data[i] });
        try testing.expectApproxEqRel(test_data[i], result_data[i], 0.001);
    }

    std.debug.print("✓ Complete staging buffer copy chain works correctly!\n", .{});
    std.debug.print("✓ Data successfully copied: Host → Staging → Device → Staging → Host\n", .{});
}

test "staging buffer copy operations - write verification with deviceToHost" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    var gpu = try createTestGPU();
    defer gpu.deinit();

    // Test different data pattern
    var test_data = [8]f32{ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 };

    // Create device-to-host buffer (can be read from)
    var device_to_host_buffer: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .size(8 * @sizeOf(f32))
        .deviceToHost()
        .build();
    try device_to_host_buffer.allocate(&gpu);
    defer device_to_host_buffer.deinit(&gpu);

    std.debug.print("\n=== Testing Device-to-Host Buffer Read (should fail write) ===\n", .{});

    // Try to write data (this should fail)
    std.debug.print("Attempting to write to deviceToHost buffer...\n", .{});
    if (device_to_host_buffer.writeToBuffer(&gpu, std.mem.sliceAsBytes(&test_data))) {
        std.debug.print("ERROR: Write succeeded when it should have failed!\n", .{});
        return error.UnexpectedSuccess;
    } else |err| {
        std.debug.print("✓ Write correctly failed with error: {}\n", .{err});
        try testing.expectEqual(error.BufferNotHostToDevice, err);
    }

    std.debug.print("✓ Buffer type restrictions working correctly!\n", .{});
}
