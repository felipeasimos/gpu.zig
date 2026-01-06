const GPU = @import("gpu.zig").GPU;
const vk = @import("vulkan");
const std = @import("std");

pub const Buffer = struct {
    size: usize,
    sharing_mode: vk.SharingMode = .exclusive,
    stage_flags: vk.ShaderStageFlags,
    usage: vk.BufferUsageFlags,
    handle: vk.Buffer = .null_handle,
    buffer_memory: vk.DeviceMemory = .null_handle,
    buffer_mem_type: BufferMemoryType = .DeviceLocal,
    desc_type: vk.DescriptorType = .storage_buffer,
    pub const BufferMemoryType = enum {
        DeviceLocal,
        HostToDevice,
        DeviceToHost,
        pub fn toMemoryPropertyFlags(self: BufferMemoryType) vk.MemoryPropertyFlags {
            return switch (self) {
                .DeviceLocal => vk.MemoryPropertyFlags{
                    .device_local_bit = true,
                },
                .HostToDevice => vk.MemoryPropertyFlags{
                    .host_visible_bit = true,
                    .host_coherent_bit = true,
                },
                .DeviceToHost => vk.MemoryPropertyFlags{
                    .host_visible_bit = true,
                    .host_cached_bit = true,
                },
            };
        }
    };
    pub const Builder = struct {
        // only exclusize is supported
        b_sharing_mode: vk.SharingMode = .exclusive,
        b_usage: vk.BufferUsageFlags = .{},
        b_size: usize = 0,
        b_stage_flags: vk.ShaderStageFlags = .{},
        b_buffer_mem_type: BufferMemoryType = .DeviceLocal,
        b_desc_type: ?vk.DescriptorType = null,
        pub fn init() Builder {
            return .{};
        }
        pub fn size(self: Builder, s: usize) Builder {
            var new = self;
            new.b_size = s;
            return new;
        }
        pub fn storage(self: Builder) Builder {
            var new = self;
            new.b_usage.storage_buffer_bit = true;
            return new;
        }
        pub fn uniform(self: Builder) Builder {
            var new = self;
            new.b_usage.uniform_buffer_bit = true;
            return new;
        }
        pub fn usage(self: Builder, arg_usage: vk.BufferUsageFlags) Builder {
            var new = self;
            new.b_usage = new.b_usage.merge(arg_usage);
            return new;
        }
        pub fn stage(self: Builder, arg_stage: vk.ShaderStageFlags) Builder {
            var new = self;
            new.b_stage_flags = new.b_stage_flags.merge(arg_stage);
            return new;
        }
        pub fn descriptorType(self: Builder, arg_desc_type: vk.DescriptorType) Builder {
            var new = self;
            new.b_desc_type = arg_desc_type;
            return new;
        }
        /// mark buffer as being device local buffer
        /// device_local
        pub fn device(self: Builder) Builder {
            var new = self;
            new.b_buffer_mem_type = .DeviceLocal;
            return new;
        }
        /// mark buffer as being a hostToDevice write buffer
        /// host_visible & host_coherent
        pub fn hostToDevice(self: Builder) Builder {
            var new = self;
            new.b_buffer_mem_type = .HostToDevice;
            return new;
        }
        /// mark buffer as being a deviceToHost write buffer
        /// host_visible & host_cached
        pub fn deviceToHost(self: Builder) Builder {
            var new = self;
            new.b_buffer_mem_type = .DeviceToHost;
            return new;
        }
        fn descTypeResolve(usg: vk.BufferUsageFlags, desc_type: ?vk.DescriptorType) vk.DescriptorType {
            if (desc_type) |t| {
                return t;
            }
            if (usg.uniform_buffer_bit) {
                return .uniform_buffer;
            }
            if (usg.uniform_texel_buffer_bit) {
                return .uniform_texel_buffer;
            }
            if (usg.storage_texel_buffer_bit) {
                return .storage_texel_buffer;
            }
            return .storage_buffer;
        }
        pub fn build(self: Builder) Buffer {
            const final_usage = switch (self.b_buffer_mem_type) {
                .DeviceToHost => self.b_usage.merge(vk.BufferUsageFlags{
                    .transfer_dst_bit = true,
                }),
                .HostToDevice => self.b_usage.merge(vk.BufferUsageFlags{
                    .transfer_src_bit = true,
                }),
                .DeviceLocal => self.b_usage,
            };
            return .{
                .sharing_mode = self.b_sharing_mode,
                .size = self.b_size,
                .stage_flags = self.b_stage_flags,
                .usage = final_usage,
                .buffer_mem_type = self.b_buffer_mem_type,
                .desc_type = descTypeResolve(self.b_usage, self.b_desc_type),
            };
        }
    };
    pub fn getHandle(self: *@This()) ?vk.Buffer {
        if (self.handle != .null_handle) {
            return self.handle;
        }
        return null;
    }
    /// create buffer handle and allocate gpu memory for it
    pub fn allocate(self: *@This(), gpu: *GPU) !void {
        self.handle = try gpu.dev.createBuffer(&vk.BufferCreateInfo{
            .size = self.size,
            .usage = self.usage,
            .sharing_mode = self.sharing_mode,
        }, null);
        const mem_reqs = gpu.dev.getBufferMemoryRequirements(self.handle);
        self.buffer_memory = try gpu.allocate(mem_reqs, self.buffer_mem_type.toMemoryPropertyFlags());
        try gpu.dev.bindBufferMemory(self.handle, self.buffer_memory, 0);
    }
    pub fn writeToBuffer(self: *@This(), gpu: *GPU, data: []u8) !void {
        if (self.buffer_mem_type != .HostToDevice) {
            return error.BufferNotHostToDevice;
        }
        if (self.buffer_memory == .null_handle) {
            return error.BufferMemoryNotAllocated;
        }
        if (data.len > self.size) {
            return error.BufferNotBigEnough;
        }
        const mapped_memory = try gpu.dev.mapMemory(self.buffer_memory, 0, data.len, .{});
        defer gpu.dev.unmapMemory(self.buffer_memory);

        const mapped_memory_data = @as([*]u8, @ptrCast(mapped_memory))[0..data.len];
        std.mem.copyForwards(u8, mapped_memory_data, data);
        for (0..mapped_memory_data.len) |i| {
            std.debug.print("mapped[{}] = {}\n", .{ i, mapped_memory_data[i] });
        }
        for (0..data.len) |i| {
            std.debug.print("data[{}] = {}\n", .{ i, data[i] });
        }
        try gpu.dev.flushMappedMemoryRanges(1, &[_]vk.MappedMemoryRange{.{
            .memory = self.buffer_memory,
            .offset = 0,
            .size = data.len,
        }});
    }
    pub fn readFromBuffer(self: *@This(), gpu: *GPU, data: []u8) !void {
        if (self.buffer_mem_type != .DeviceToHost) {
            return error.BufferNotHostToDevice;
        }
        if (self.buffer_memory == .null_handle) {
            return error.BufferMemoryNotAllocated;
        }
        const mapped_memory = try gpu.dev.mapMemory(self.buffer_memory, 0, data.len, .{});
        defer gpu.dev.unmapMemory(self.buffer_memory);

        std.mem.copyForwards(u8, data, @as([*]u8, @ptrCast(mapped_memory))[0..data.len]);
    }
    pub fn copyBufferDataTo(self: *@This(), gpu: *GPU, cmd_buffer: vk.CommandBuffer, to: *Buffer, size: u64) !void {
        if (self.handle == .null_handle) {
            return error.BufferHandleNotInitialized;
        }
        if (to.handle == .null_handle) {
            return error.BufferHandleNotInitialized;
        }
        gpu.dev.cmdCopyBuffer(cmd_buffer, to.handle, self.handle, &[_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        }});
    }
    pub fn deinit(self: *@This(), gpu: *GPU) void {
        if (self.handle != .null_handle) {
            gpu.dev.destroyBuffer(self.handle, null);
        }
        if (self.buffer_memory != .null_handle) {
            gpu.dev.freeMemory(self.buffer_memory, null);
        }
    }
    pub fn descriptorBinding(self: *@This(), binding_idx: u32, descriptor_count: u32) vk.DescriptorSetLayoutBinding {
        return .{
            .binding = binding_idx,
            .descriptor_count = descriptor_count,
            .descriptor_type = self.desc_type,
            .stage_flags = self.stage_flags,
        };
    }
};
