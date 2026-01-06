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
    const title = std.fmt.comptimePrint("Descriptor Set Test - {s}", .{gpu_zig.options.git_commit_hash});
    return GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        .validation()
        .build();
}

// Use the same simple shader as before
const simple_compute_spv align(@alignOf(u32)) = @embedFile("simple_test.spv").*;

test "descriptor set creation and binding verification" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    var gpu = try createTestGPU();
    defer gpu.deinit();

    std.debug.print("\n=== Testing Descriptor Set Creation and Binding ===\n", .{});

    // Create device buffers exactly as in the failing test
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

    // Print buffer information
    std.debug.print("Device Input Buffer:\n", .{});
    std.debug.print("  Handle: {}\n", .{dev_in_buf.handle});
    std.debug.print("  Size: {} bytes\n", .{dev_in_buf.size});
    std.debug.print("  Usage: storage_buffer={}, transfer_dst={}\n", .{ dev_in_buf.usage.storage_buffer_bit, dev_in_buf.usage.transfer_dst_bit });
    std.debug.print("  Stage: compute={}\n", .{dev_in_buf.stage_flags.compute_bit});
    std.debug.print("  Descriptor Type: {}\n", .{dev_in_buf.desc_type});

    std.debug.print("Device Output Buffer:\n", .{});
    std.debug.print("  Handle: {}\n", .{dev_out_buf.handle});
    std.debug.print("  Size: {} bytes\n", .{dev_out_buf.size});
    std.debug.print("  Usage: storage_buffer={}, transfer_src={}\n", .{ dev_out_buf.usage.storage_buffer_bit, dev_out_buf.usage.transfer_src_bit });
    std.debug.print("  Stage: compute={}\n", .{dev_out_buf.stage_flags.compute_bit});
    std.debug.print("  Descriptor Type: {}\n", .{dev_out_buf.desc_type});

    // Test individual descriptor bindings
    const input_binding = dev_in_buf.descriptorBinding(0, 1);
    const output_binding = dev_out_buf.descriptorBinding(1, 1);

    std.debug.print("Descriptor Bindings:\n", .{});
    std.debug.print("  Input (binding 0): binding={}, count={}, type={}, stage_flags.compute={}\n", .{ input_binding.binding, input_binding.descriptor_count, input_binding.descriptor_type, input_binding.stage_flags.compute_bit });
    std.debug.print("  Output (binding 1): binding={}, count={}, type={}, stage_flags.compute={}\n", .{ output_binding.binding, output_binding.descriptor_count, output_binding.descriptor_type, output_binding.stage_flags.compute_bit });

    // Verify buffer handles are valid
    try testing.expect(dev_in_buf.handle != vk.Buffer.null_handle);
    try testing.expect(dev_out_buf.handle != vk.Buffer.null_handle);
    try testing.expect(dev_in_buf.buffer_memory != vk.DeviceMemory.null_handle);
    try testing.expect(dev_out_buf.buffer_memory != vk.DeviceMemory.null_handle);

    std.debug.print("✓ Buffer handles and memory are valid\n", .{});

    // Create compute pipeline and examine its internals
    std.debug.print("Creating compute pipeline...\n", .{});
    var pipeline = try ComputePipeline.Builder.init(&gpu)
        .code(&simple_compute_spv)
        .buffer(dev_in_buf)
        .buffer(dev_out_buf)
        .build();
    defer pipeline.deinit(&gpu);

    std.debug.print("Pipeline created successfully:\n", .{});
    std.debug.print("  Pipeline handle: {}\n", .{pipeline.pipeline});
    std.debug.print("  Pipeline layout: {}\n", .{pipeline.pipeline_layout});
    std.debug.print("  Descriptor set: {}\n", .{pipeline.desc_set});
    std.debug.print("  Descriptor set layout: {}\n", .{pipeline.desc_set_layout});

    // Verify pipeline components are valid
    try testing.expect(pipeline.pipeline != vk.Pipeline.null_handle);
    try testing.expect(pipeline.pipeline_layout != vk.PipelineLayout.null_handle);
    try testing.expect(pipeline.desc_set != vk.DescriptorSet.null_handle);
    try testing.expect(pipeline.desc_set_layout != vk.DescriptorSetLayout.null_handle);

    std.debug.print("✓ Pipeline components are valid\n", .{});

    // Test command buffer binding (without execution)
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

    try gpu.dev.beginCommandBuffer(cmd_buffer, &vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    });

    // Test pipeline binding
    std.debug.print("Testing pipeline binding...\n", .{});
    gpu.dev.cmdBindPipeline(cmd_buffer, .compute, pipeline.pipeline);
    std.debug.print("✓ Pipeline bound successfully\n", .{});

    // Test descriptor set binding
    std.debug.print("Testing descriptor set binding...\n", .{});
    gpu.dev.cmdBindDescriptorSets(
        cmd_buffer,
        .compute,
        pipeline.pipeline_layout,
        0, // first set
        1, // descriptor set count
        @ptrCast(&pipeline.desc_set),
        0, // dynamic offset count
        null, // dynamic offsets
    );
    std.debug.print("✓ Descriptor set bound successfully\n", .{});

    try gpu.dev.endCommandBuffer(cmd_buffer);

    std.debug.print("✓ All descriptor set operations completed successfully\n", .{});
    std.debug.print("✓ Pipeline creation and binding work correctly\n", .{});
}

test "buffer descriptor binding compatibility" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    var gpu = try createTestGPU();
    defer gpu.deinit();

    std.debug.print("\n=== Testing Buffer-Shader Binding Compatibility ===\n", .{});

    // Test different buffer configurations
    var test_buffer: Buffer = Buffer.Builder.init()
        .usage(.{ .storage_buffer_bit = true })
        .descriptorType(.storage_buffer)
        .stage(.{ .compute_bit = true })
        .size(16)
        .device()
        .build();
    try test_buffer.allocate(&gpu);
    defer test_buffer.deinit(&gpu);

    // Test descriptor binding with different indices
    for (0..4) |binding_idx| {
        const binding = test_buffer.descriptorBinding(@intCast(binding_idx), 1);
        std.debug.print("Binding {}: type={}, stage_compute={}\n", .{ binding_idx, binding.descriptor_type, binding.stage_flags.compute_bit });

        // Verify binding properties match shader expectations
        try testing.expectEqual(vk.DescriptorType.storage_buffer, binding.descriptor_type);
        try testing.expect(binding.stage_flags.compute_bit == true);
        try testing.expectEqual(@as(u32, @intCast(binding_idx)), binding.binding);
    }

    std.debug.print("✓ Buffer descriptor bindings are compatible with shader requirements\n", .{});
}
