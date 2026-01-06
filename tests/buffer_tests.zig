const std = @import("std");
const testing = std.testing;
const gpu = @import("gpu");
const Buffer = gpu.Buffer;
const vk = gpu.vk;

// Test BufferMemoryType enum and its methods
test "BufferMemoryType.toMemoryPropertyFlags returns correct flags for DeviceLocal" {
    const buffer_mem_type = Buffer.BufferMemoryType.DeviceLocal;
    const flags = buffer_mem_type.toMemoryPropertyFlags();

    try testing.expect(flags.device_local_bit == true);
    try testing.expect(flags.host_visible_bit == false);
    try testing.expect(flags.host_coherent_bit == false);
    try testing.expect(flags.host_cached_bit == false);
}

test "BufferMemoryType.toMemoryPropertyFlags returns correct flags for HostToDevice" {
    const buffer_mem_type = Buffer.BufferMemoryType.HostToDevice;
    const flags = buffer_mem_type.toMemoryPropertyFlags();

    try testing.expect(flags.device_local_bit == false);
    try testing.expect(flags.host_visible_bit == true);
    try testing.expect(flags.host_coherent_bit == true);
    try testing.expect(flags.host_cached_bit == false);
}

test "BufferMemoryType.toMemoryPropertyFlags returns correct flags for DeviceToHost" {
    const buffer_mem_type = Buffer.BufferMemoryType.DeviceToHost;
    const flags = buffer_mem_type.toMemoryPropertyFlags();

    try testing.expect(flags.device_local_bit == false);
    try testing.expect(flags.host_visible_bit == true);
    try testing.expect(flags.host_coherent_bit == false);
    try testing.expect(flags.host_cached_bit == true);
}

// Test Buffer.Builder initialization
test "Buffer.Builder.init creates builder with default values" {
    const builder = Buffer.Builder.init();

    try testing.expectEqual(vk.SharingMode.exclusive, builder.b_sharing_mode);
    try testing.expectEqual(vk.BufferUsageFlags.fromInt(0), builder.b_usage);
    try testing.expectEqual(vk.ShaderStageFlags.fromInt(0), builder.b_stage_flags);
    try testing.expectEqual(Buffer.BufferMemoryType.DeviceLocal, builder.b_buffer_mem_type);
    try testing.expectEqual(@as(?vk.DescriptorType, null), builder.b_desc_type);
}

// Test Buffer.Builder size method
test "Buffer.Builder.size sets buffer size correctly" {
    const builder = Buffer.Builder.init().size(1024);

    try testing.expectEqual(@as(usize, 1024), builder.b_size);
}

// Test Buffer.Builder storage method
test "Buffer.Builder.storage sets storage buffer usage flag" {
    const builder = Buffer.Builder.init().storage();

    try testing.expect(builder.b_usage.storage_buffer_bit == true);
}

// Test Buffer.Builder uniform method
test "Buffer.Builder.uniform sets uniform buffer usage flag" {
    const builder = Buffer.Builder.init().uniform();

    try testing.expect(builder.b_usage.uniform_buffer_bit == true);
}

// Test Buffer.Builder usage method
test "Buffer.Builder.usage combines usage flags correctly" {
    const custom_usage = vk.BufferUsageFlags{ .transfer_src_bit = true, .transfer_dst_bit = true };
    const builder = Buffer.Builder.init().usage(custom_usage);

    try testing.expect(builder.b_usage.transfer_src_bit == true);
    try testing.expect(builder.b_usage.transfer_dst_bit == true);
}

// Test Buffer.Builder stage method
test "Buffer.Builder.stage sets shader stage flags" {
    const stage_flags = vk.ShaderStageFlags{ .compute_bit = true };
    const builder = Buffer.Builder.init().stage(stage_flags);

    try testing.expect(builder.b_stage_flags.compute_bit == true);
}

// Test Buffer.Builder descriptorType method
test "Buffer.Builder.descriptorType sets descriptor type" {
    const builder = Buffer.Builder.init().descriptorType(.uniform_buffer);

    try testing.expectEqual(vk.DescriptorType.uniform_buffer, builder.b_desc_type.?);
}

// Test Buffer.Builder memory type methods
test "Buffer.Builder.device sets DeviceLocal memory type" {
    const builder = Buffer.Builder.init().device();

    try testing.expectEqual(Buffer.BufferMemoryType.DeviceLocal, builder.b_buffer_mem_type);
}

test "Buffer.Builder.hostToDevice sets HostToDevice memory type" {
    const builder = Buffer.Builder.init().hostToDevice();

    try testing.expectEqual(Buffer.BufferMemoryType.HostToDevice, builder.b_buffer_mem_type);
}

test "Buffer.Builder.deviceToHost sets DeviceToHost memory type" {
    const builder = Buffer.Builder.init().deviceToHost();

    try testing.expectEqual(Buffer.BufferMemoryType.DeviceToHost, builder.b_buffer_mem_type);
}

// Test Buffer.Builder.build method
test "Buffer.Builder.build creates buffer with correct basic properties" {
    const buffer = Buffer.Builder.init()
        .size(2048)
        .storage()
        .device()
        .build();

    try testing.expectEqual(@as(usize, 2048), buffer.size);
    try testing.expectEqual(vk.SharingMode.exclusive, buffer.sharing_mode);
    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expectEqual(Buffer.BufferMemoryType.DeviceLocal, buffer.buffer_mem_type);
    try testing.expectEqual(vk.DescriptorType.storage_buffer, buffer.desc_type);
    try testing.expectEqual(vk.Buffer.null_handle, buffer.handle);
    try testing.expectEqual(vk.DeviceMemory.null_handle, buffer.buffer_memory);
}

test "Buffer.Builder.build adds transfer_dst_bit for DeviceToHost buffers" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .deviceToHost()
        .build();

    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_dst_bit == true);
    try testing.expectEqual(Buffer.BufferMemoryType.DeviceToHost, buffer.buffer_mem_type);
}

test "Buffer.Builder.build adds transfer_src_bit for HostToDevice buffers" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .hostToDevice()
        .build();

    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_src_bit == true);
    try testing.expectEqual(Buffer.BufferMemoryType.HostToDevice, buffer.buffer_mem_type);
}

// Test Buffer methods
test "Buffer.getHandle returns null when handle is null_handle" {
    var buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .build();

    try testing.expectEqual(@as(?vk.Buffer, null), buffer.getHandle());
}

test "Buffer.getHandle returns handle when set" {
    var buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .build();

    // Simulate setting a handle (normally done by allocate())
    buffer.handle = @enumFromInt(42); // Mock handle value

    try testing.expectEqual(@as(?vk.Buffer, @enumFromInt(42)), buffer.getHandle());
}

test "Buffer.descriptorBinding creates correct descriptor set layout binding" {
    var buffer = Buffer.Builder.init()
        .size(1024)
        .uniform()
        .stage(vk.ShaderStageFlags{ .vertex_bit = true, .fragment_bit = true })
        .build();

    const binding = buffer.descriptorBinding(2, 1);

    try testing.expectEqual(@as(u32, 2), binding.binding);
    try testing.expectEqual(@as(u32, 1), binding.descriptor_count);
    try testing.expectEqual(vk.DescriptorType.uniform_buffer, binding.descriptor_type);
    try testing.expect(binding.stage_flags.vertex_bit == true);
    try testing.expect(binding.stage_flags.fragment_bit == true);
}

// Test method chaining in Builder
test "Buffer.Builder method chaining works correctly" {
    const buffer = Buffer.Builder.init()
        .size(4096)
        .storage()
        .uniform()
        .hostToDevice()
        .stage(vk.ShaderStageFlags{ .compute_bit = true })
        .descriptorType(.storage_buffer)
        .build();

    try testing.expectEqual(@as(usize, 4096), buffer.size);
    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.uniform_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_src_bit == true);
    try testing.expectEqual(Buffer.BufferMemoryType.HostToDevice, buffer.buffer_mem_type);
    try testing.expect(buffer.stage_flags.compute_bit == true);
    try testing.expectEqual(vk.DescriptorType.storage_buffer, buffer.desc_type);
}

// Test edge cases
test "Buffer created with zero size" {
    const buffer = Buffer.Builder.init()
        .size(0)
        .storage()
        .build();

    try testing.expectEqual(@as(usize, 0), buffer.size);
}

test "Buffer with multiple usage flags combined" {
    const custom_usage = vk.BufferUsageFlags{ .transfer_src_bit = true, .transfer_dst_bit = true, .vertex_buffer_bit = true };

    const buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .usage(custom_usage)
        .build();

    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_src_bit == true);
    try testing.expect(buffer.usage.transfer_dst_bit == true);
    try testing.expect(buffer.usage.vertex_buffer_bit == true);
}

// Test descriptor type resolution through the build method
test "Buffer.Builder.build resolves descriptor type to uniform_buffer for uniform usage" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .uniform()
        .build();

    try testing.expectEqual(vk.DescriptorType.uniform_buffer, buffer.desc_type);
}

test "Buffer.Builder.build resolves descriptor type to uniform_texel_buffer for uniform texel usage" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .usage(vk.BufferUsageFlags{ .uniform_texel_buffer_bit = true })
        .build();

    try testing.expectEqual(vk.DescriptorType.uniform_texel_buffer, buffer.desc_type);
}

test "Buffer.Builder.build resolves descriptor type to storage_texel_buffer for storage texel usage" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .usage(vk.BufferUsageFlags{ .storage_texel_buffer_bit = true })
        .build();

    try testing.expectEqual(vk.DescriptorType.storage_texel_buffer, buffer.desc_type);
}

test "Buffer.Builder.build uses explicit descriptor type when provided" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .descriptorType(.uniform_buffer)
        .build();

    // Even though storage usage would normally result in storage_buffer descriptor type,
    // the explicit descriptor type should take precedence
    try testing.expectEqual(vk.DescriptorType.uniform_buffer, buffer.desc_type);
}

test "Buffer.Builder.build defaults to storage_buffer descriptor type" {
    const buffer = Buffer.Builder.init()
        .size(1024)
        .usage(vk.BufferUsageFlags{ .vertex_buffer_bit = true })
        .build();

    try testing.expectEqual(vk.DescriptorType.storage_buffer, buffer.desc_type);
}

// Test complex builder combinations
test "Buffer.Builder complex configuration with all options" {
    const buffer = Buffer.Builder.init()
        .size(8192)
        .storage()
        .uniform()
        .usage(vk.BufferUsageFlags{
            .transfer_src_bit = true,
            .transfer_dst_bit = true,
            .index_buffer_bit = true,
        })
        .stage(vk.ShaderStageFlags{
            .vertex_bit = true,
            .fragment_bit = true,
            .compute_bit = true,
        })
        .deviceToHost()
        .descriptorType(.storage_buffer)
        .build();

    try testing.expectEqual(@as(usize, 8192), buffer.size);
    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.uniform_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_src_bit == true);
    try testing.expect(buffer.usage.transfer_dst_bit == true);
    try testing.expect(buffer.usage.index_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_dst_bit == true); // Added by deviceToHost

    try testing.expect(buffer.stage_flags.vertex_bit == true);
    try testing.expect(buffer.stage_flags.fragment_bit == true);
    try testing.expect(buffer.stage_flags.compute_bit == true);

    try testing.expectEqual(Buffer.BufferMemoryType.DeviceToHost, buffer.buffer_mem_type);
    try testing.expectEqual(vk.DescriptorType.storage_buffer, buffer.desc_type);
}

// Test stage flag merging
test "Buffer.Builder.stage merges with existing stage flags" {
    const builder1 = Buffer.Builder.init()
        .stage(vk.ShaderStageFlags{ .vertex_bit = true });

    const builder2 = builder1.stage(vk.ShaderStageFlags{ .fragment_bit = true });

    try testing.expect(builder2.b_stage_flags.vertex_bit == true);
    try testing.expect(builder2.b_stage_flags.fragment_bit == true);
}

// Test usage flag merging
test "Buffer.Builder.usage merges with existing usage flags" {
    const builder1 = Buffer.Builder.init()
        .storage(); // sets storage_buffer_bit

    const builder2 = builder1.usage(vk.BufferUsageFlags{ .transfer_src_bit = true });

    try testing.expect(builder2.b_usage.storage_buffer_bit == true);
    try testing.expect(builder2.b_usage.transfer_src_bit == true);
}

// Test DeviceLocal memory type doesn't add extra usage flags
test "Buffer.Builder.build doesn't modify usage for DeviceLocal memory type" {
    const original_usage = vk.BufferUsageFlags{ .storage_buffer_bit = true };
    const buffer = Buffer.Builder.init()
        .size(1024)
        .usage(original_usage)
        .device() // DeviceLocal
        .build();

    try testing.expect(buffer.usage.storage_buffer_bit == true);
    try testing.expect(buffer.usage.transfer_src_bit == false);
    try testing.expect(buffer.usage.transfer_dst_bit == false);
}

// Test descriptorBinding with different parameters
test "Buffer.descriptorBinding with storage buffer configuration" {
    var buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .stage(vk.ShaderStageFlags{ .compute_bit = true })
        .build();

    const binding = buffer.descriptorBinding(5, 3);

    try testing.expectEqual(@as(u32, 5), binding.binding);
    try testing.expectEqual(@as(u32, 3), binding.descriptor_count);
    try testing.expectEqual(vk.DescriptorType.storage_buffer, binding.descriptor_type);
    try testing.expect(binding.stage_flags.compute_bit == true);
}

test "Buffer.descriptorBinding with uniform buffer configuration" {
    var buffer = Buffer.Builder.init()
        .size(1024)
        .uniform()
        .stage(vk.ShaderStageFlags{ .vertex_bit = true, .fragment_bit = true })
        .build();

    const binding = buffer.descriptorBinding(0, 1);

    try testing.expectEqual(@as(u32, 0), binding.binding);
    try testing.expectEqual(@as(u32, 1), binding.descriptor_count);
    try testing.expectEqual(vk.DescriptorType.uniform_buffer, binding.descriptor_type);
    try testing.expect(binding.stage_flags.vertex_bit == true);
    try testing.expect(binding.stage_flags.fragment_bit == true);
}

// Test immutability of Builder methods (each method returns a new instance)
test "Buffer.Builder methods don't modify original builder" {
    const original_builder = Buffer.Builder.init();

    // Call methods that should return new builders
    const size_builder = original_builder.size(2048);
    const storage_builder = original_builder.storage();
    const device_builder = original_builder.device();

    // Original builder should remain unchanged
    try testing.expectEqual(@as(usize, 0), original_builder.b_size);
    try testing.expect(original_builder.b_usage.storage_buffer_bit == false);
    try testing.expectEqual(Buffer.BufferMemoryType.DeviceLocal, original_builder.b_buffer_mem_type);

    // New builders should have the changes
    try testing.expectEqual(@as(usize, 2048), size_builder.b_size);
    try testing.expect(storage_builder.b_usage.storage_buffer_bit == true);
    try testing.expectEqual(Buffer.BufferMemoryType.DeviceLocal, device_builder.b_buffer_mem_type);
}

// Test large buffer sizes
test "Buffer.Builder supports large buffer sizes" {
    const large_size = 1024 * 1024 * 1024; // 1GB
    const buffer = Buffer.Builder.init()
        .size(large_size)
        .storage()
        .build();

    try testing.expectEqual(large_size, buffer.size);
}

// Test all memory type combinations with transfer flags
test "Buffer memory types correctly set transfer flags" {
    // DeviceLocal should not add any transfer flags
    const device_local_buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .device()
        .build();

    try testing.expect(device_local_buffer.usage.storage_buffer_bit == true);
    try testing.expect(device_local_buffer.usage.transfer_src_bit == false);
    try testing.expect(device_local_buffer.usage.transfer_dst_bit == false);

    // HostToDevice should add transfer_src_bit
    const host_to_device_buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .hostToDevice()
        .build();

    try testing.expect(host_to_device_buffer.usage.storage_buffer_bit == true);
    try testing.expect(host_to_device_buffer.usage.transfer_src_bit == true);
    try testing.expect(host_to_device_buffer.usage.transfer_dst_bit == false);

    // DeviceToHost should add transfer_dst_bit
    const device_to_host_buffer = Buffer.Builder.init()
        .size(1024)
        .storage()
        .deviceToHost()
        .build();

    try testing.expect(device_to_host_buffer.usage.storage_buffer_bit == true);
    try testing.expect(device_to_host_buffer.usage.transfer_src_bit == false);
    try testing.expect(device_to_host_buffer.usage.transfer_dst_bit == true);
}
