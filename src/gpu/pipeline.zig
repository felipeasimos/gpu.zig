const std = @import("std");
const GPU = @import("gpu.zig").GPU;
const vk = @import("vulkan");
const Buffer = @import("buffer.zig").Buffer;

/// orchestrate buffers, push constants and shader code
pub const ComputePipeline = struct {
    buffers: std.ArrayList(*Buffer) = .empty,
    desc_set: vk.DescriptorSet = .null_handle,
    desc_set_pool: vk.DescriptorPool = .null_handle,
    desc_set_layout: vk.DescriptorSetLayout = .null_handle,
    pipeline: vk.Pipeline = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    shader_module: vk.ShaderModule = .null_handle,
    pub const Builder = struct {
        b_code: []const u8,
        b_buffers: std.ArrayList(*Buffer) = .empty,
        b_push_constants_sizes: std.ArrayList(usize) = .empty,
        b_gpu: *GPU,
        pub fn init(gpu: *GPU) Builder {
            return .{
                .b_code = &.{},
                .b_gpu = gpu,
            };
        }
        pub fn code(self: Builder, arg_code: []const u8) Builder {
            var new = self;
            new.b_code = arg_code;
            return new;
        }
        pub fn buffer(self: Builder, buf: *Buffer) Builder {
            var new = self;
            new.b_buffers.append(self.b_gpu.allocator, buf) catch
                @panic("Couldn't append buffer to pipeline builder");
            return new;
        }
        pub fn pushConstant(self: Builder, comptime T: type) Builder {
            var new = self;
            new.b_push_constants_sizes.append(self.b_gpu.allocator, @sizeOf(T)) catch
                @panic("Couldn't append push constant type size to pipeline builder");
            return new;
        }
        pub fn build(builder: Builder) !ComputePipeline {
            var self = builder;
            const bindings = try self.b_gpu.allocator.alloc(vk.DescriptorSetLayoutBinding, self.b_buffers.items.len);
            defer self.b_gpu.allocator.free(bindings);
            for (self.b_buffers.items, 0..) |buf, i| {
                bindings[i] = buf.descriptorBinding(@intCast(i), 1);
            }
            const push_constant_ranges = try self.b_gpu.allocator.alloc(vk.PushConstantRange, self.b_push_constants_sizes.items.len);
            defer self.b_gpu.allocator.free(push_constant_ranges);
            for (self.b_push_constants_sizes.items, 0..) |pc_size, i| {
                push_constant_ranges[i] = vk.PushConstantRange{
                    .offset = 0,
                    .size = @intCast(pc_size),
                    .stage_flags = .{ .compute_bit = true },
                };
            }
            const desc_set_layout = try self.b_gpu.dev.createDescriptorSetLayout(&vk.DescriptorSetLayoutCreateInfo{
                .binding_count = @intCast(bindings.len),
                .p_bindings = bindings.ptr,
            }, null);
            var desc_pool_sizes = try self.b_gpu.allocator.alloc(vk.DescriptorPoolSize, self.b_buffers.items.len);
            defer self.b_gpu.allocator.free(desc_pool_sizes);
            for (self.b_buffers.items, 0..) |buf, i| {
                desc_pool_sizes[i] = .{
                    .type = buf.desc_type,
                    .descriptor_count = 1,
                };
            }

            const desc_set_pool = try self.b_gpu.dev.createDescriptorPool(&vk.DescriptorPoolCreateInfo{
                .max_sets = 1,
                .pool_size_count = @intCast(desc_pool_sizes.len),
                .p_pool_sizes = @ptrCast(desc_pool_sizes.ptr),
            }, null);
            var desc_set: vk.DescriptorSet = undefined;
            try self.b_gpu.dev.allocateDescriptorSets(&vk.DescriptorSetAllocateInfo{
                .descriptor_pool = desc_set_pool,
                .descriptor_set_count = 1,
                .p_set_layouts = &[_]vk.DescriptorSetLayout{desc_set_layout},
            }, @ptrCast(&desc_set));
            const layout = try self.b_gpu.dev.createPipelineLayout(&vk.PipelineLayoutCreateInfo{
                .set_layout_count = 1,
                .p_set_layouts = &[_]vk.DescriptorSetLayout{desc_set_layout},
                .push_constant_range_count = @intCast(push_constant_ranges.len),
                .p_push_constant_ranges = push_constant_ranges.ptr,
            }, null);
            const shader_module = try self.b_gpu.dev.createShaderModule(&.{
                .code_size = self.b_code.len,
                .p_code = @ptrCast(@alignCast(self.b_code.ptr)),
            }, null);
            var pipeline: vk.Pipeline = undefined;
            _ = try self.b_gpu.dev.createComputePipelines(
                .null_handle,
                1,
                &.{.{
                    .flags = .{},
                    .stage = .{
                        .stage = .{ .compute_bit = true },
                        .module = shader_module,
                        .p_name = "main",
                    },
                    .layout = layout,
                    .base_pipeline_handle = .null_handle,
                    .base_pipeline_index = 0,
                }},
                null,
                @ptrCast(&pipeline),
            );

            const desc_buffer_infos = try self.b_gpu.allocator.alloc(vk.DescriptorBufferInfo, self.b_buffers.items.len);
            defer self.b_gpu.allocator.free(desc_buffer_infos);
            const write_desc_sets = try self.b_gpu.allocator.alloc(vk.WriteDescriptorSet, self.b_buffers.items.len);
            defer self.b_gpu.allocator.free(write_desc_sets);
            for (self.b_buffers.items, 0..) |buf, i| {
                desc_buffer_infos[i] = .{
                    .buffer = buf.handle,
                    .offset = 0,
                    .range = vk.WHOLE_SIZE,
                };
                write_desc_sets[i] = vk.WriteDescriptorSet{
                    .dst_set = desc_set,
                    .dst_binding = @intCast(i),
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = buf.desc_type,
                    .p_buffer_info = @ptrCast(&desc_buffer_infos[i]),
                    .p_image_info = undefined,
                    .p_texel_buffer_view = undefined,
                };
            }
            self.b_gpu.dev.updateDescriptorSets(
                @intCast(write_desc_sets.len),
                @ptrCast(write_desc_sets.ptr),
                0,
                null,
            );
            return .{
                .buffers = self.b_buffers,
                .desc_set_layout = desc_set_layout,
                .pipeline = pipeline,
                .pipeline_layout = layout,
                .shader_module = shader_module,
                .desc_set = desc_set,
                .desc_set_pool = desc_set_pool,
            };
        }
    };

    pub fn bind(self: *@This(), gpu: *GPU, cmd_buffer: vk.CommandBuffer) void {
        gpu.dev.cmdBindPipeline(cmd_buffer, .compute, self.pipeline);
        gpu.dev.cmdBindDescriptorSets(
            cmd_buffer,
            .compute,
            self.pipeline_layout,
            0,
            1,
            @ptrCast(&self.desc_set),
            0,
            null,
        );
    }
    pub fn deinit(self: *@This(), gpu: *GPU) void {
        self.buffers.deinit(gpu.allocator);
        if (self.shader_module != .null_handle) {
            gpu.dev.destroyShaderModule(self.shader_module, null);
        }
        if (self.desc_set_layout != .null_handle) {
            gpu.dev.destroyDescriptorSetLayout(self.desc_set_layout, null);
        }
        if (self.desc_set_pool != .null_handle) {
            gpu.dev.destroyDescriptorPool(self.desc_set_pool, null);
        }
        if (self.pipeline_layout != .null_handle) {
            gpu.dev.destroyPipelineLayout(self.pipeline_layout, null);
        }
        if (self.pipeline != .null_handle) {
            gpu.dev.destroyPipeline(self.pipeline, null);
        }
    }
};
