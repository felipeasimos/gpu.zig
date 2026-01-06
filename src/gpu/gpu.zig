const std = @import("std");
const svdag = @import("svdag");
const vk = @import("vulkan");
const pipeline = @import("pipeline.zig");
const Queue = @import("queue.zig").Queue;
pub const Buffer = @import("buffer.zig").Buffer;
const utils = @import("utils.zig");
const zglfw = @import("zglfw");

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const required_graphics_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;

const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;
const CommandBuffer = vk.CommandBufferProxy;

pub const GPU = struct {
    pub const CommandBuffer = vk.CommandBufferProxy;

    allocator: std.mem.Allocator,
    vkb: BaseWrapper,
    instance: Instance,
    pdev: vk.PhysicalDevice,

    compute_queue: ?Queue = null,
    graphics_queue: ?Queue = null,
    present_queue: ?Queue = null,
    surface: ?vk.SurfaceKHR = null,

    dev: Device,

    mem_props: vk.PhysicalDeviceMemoryProperties,

    pub const Builder = struct {
        window: ?*zglfw.Window = null,
        app_name: [:0]const u8 = "default app name",
        instance_extensions: std.ArrayList([*:0]const u8) = .empty,
        device_extensions: std.ArrayList([*:0]const u8) = .empty,
        layers: std.ArrayList([*:0]const u8) = .empty,
        allocator: std.mem.Allocator,
        _compute: bool = false,

        pub fn init(alloc: std.mem.Allocator) @This() {
            var new: @This() = .{
                .allocator = alloc,
            };
            // these extensions are to support vulkan in mac os
            // see https://github.com/glfw/glfw/issues/2335
            new.instance_extensions.append(alloc, "VK_KHR_portability_enumeration") catch @panic("Out of Memory: Couldn't append extension to GPU Builder");
            new.instance_extensions.append(alloc, "VK_KHR_get_physical_device_properties2") catch @panic("Out of Memory: Couldn't append extension to GPU Builder");
            return new;
        }

        /// Look for a physical device with graphical capabilities
        pub fn graphics(self: @This(), window: *zglfw.Window) @This() {
            var new = self;
            new.window = window;

            new = new.deviceExtension(vk.extensions.khr_swapchain.name);
            const zglfw_extensions = zglfw.getRequiredInstanceExtensions() catch @panic("Couldn't get zglfw instance extensions");
            new.instance_extensions.appendSlice(self.allocator, zglfw_extensions) catch @panic("Couldn't append zglfw instance extensions");
            return new;
        }
        pub fn compute(self: @This()) @This() {
            var new = self;
            new._compute = true;
            return new;
        }
        /// Define an application name. Affects logs during debug (for validation layers)
        pub fn appName(self: @This(), name: [:0]const u8) @This() {
            var new = self;
            new.app_name = name;
            return new;
        }
        /// Add a required device extension.
        /// swapchain extension is already required when calling `graphics`
        pub fn deviceExtension(self: @This(), extension_name: [:0]const u8) @This() {
            var new = self;
            const contains_extension = utils.containString(
                new.device_extensions.items,
                extension_name,
            );
            if (!contains_extension) {
                new.device_extensions.append(new.allocator, extension_name.ptr) catch @panic("Out of Memory: Couldn't append device extension to GPU Builder");
            }
            return new;
        }
        /// Add a required instance extension.
        pub fn instanceExtension(self: @This(), extension_name: [:0]const u8) @This() {
            var new = self;
            const contains_extension = utils.containString(
                new.instance_extensions.items,
                extension_name,
            );
            if (!contains_extension) {
                new.instance_extensions.append(new.allocator, extension_name.ptr) catch @panic("Out of Memory: Couldn't append instance extension to GPU Builder");
            }
            return new;
        }
        pub fn validation(self: @This()) @This() {
            return self.layer("VK_LAYER_KHRONOS_validation");
        }
        pub fn layer(self: @This(), layer_name: [:0]const u8) @This() {
            var new = self;
            const contains_layer = utils.containString(
                new.layers.items,
                layer_name,
            );
            if (!contains_layer) {
                new.layers.append(new.allocator, layer_name.ptr) catch @panic("Out of Memory: Couldn't append layer to GPU Builder");
            }
            return new;
        }
        pub fn build(s: @This()) !GPU {
            var self = s;

            defer self.instance_extensions.deinit(self.allocator);
            defer self.device_extensions.deinit(self.allocator);
            defer self.layers.deinit(self.allocator);

            if (!self._compute and self.window == null) {
                return error.GPUMustBeAtLeastComputeOrGraphical;
            }
            var gpu: GPU = undefined;
            gpu.allocator = self.allocator;
            gpu.surface = null;
            gpu.compute_queue = null;
            gpu.graphics_queue = null;
            gpu.present_queue = null;

            // 1. create instance (load vulkan)
            try gpu.initInstance(self.app_name, self.instance_extensions.items, self.layers.items);
            errdefer gpu.instance.destroyInstance(null);

            // 2. create surface (if window is present)
            if (self.window) |window| {
                try gpu.initSurface(window);
            }

            // 3. find physical device (find GPU with required capabilities)
            // also get queue family indexes (queues are not created yet)
            const queue_family_indexes = try gpu.initPhysicalDevice(self._compute, self.device_extensions.items);

            // 4. initialize logical device
            try gpu.initDevice(queue_family_indexes, self.device_extensions.items);
            errdefer self.dev.destroyDevice(null);

            // 5. initialize queues
            try gpu.initQueues(queue_family_indexes);
            errdefer gpu.deinit();

            gpu.mem_props = gpu.instance.getPhysicalDeviceMemoryProperties(gpu.pdev);

            return gpu;
        }
    };

    fn initQueues(self: *GPU, family_indexes: QueueFamilyIndex) !void {
        if (family_indexes.compute) |c_idx| {
            self.compute_queue = Queue.init(self.dev, c_idx);
        }
        if (family_indexes.graphical) |g_idx| {
            self.graphics_queue = Queue.init(self.dev, g_idx);
        }
        if (family_indexes.present) |p_idx| {
            self.present_queue = Queue.init(self.dev, p_idx);
        }
    }

    fn checkSurfaceSupport(self: *GPU, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
        var format_count: u32 = undefined;
        _ = try self.instance.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

        var present_mode_count: u32 = undefined;
        _ = try self.instance.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

        return format_count > 0 and present_mode_count > 0;
    }

    fn checkExtensionSupport(self: *GPU, pdev: vk.PhysicalDevice, extensions: [][*:0]const u8) !bool {
        const propsv = try self.instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, self.allocator);
        defer self.allocator.free(propsv);

        for (extensions) |ext| {
            for (propsv) |props| {
                if (std.mem.eql(u8, std.mem.span(ext), std.mem.sliceTo(&props.extension_name, 0))) {
                    break;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    const QueueFamilyIndex = struct {
        graphical: ?u32 = null,
        present: ?u32 = null,
        compute: ?u32 = null,
    };

    fn requestQueueFamilies(
        self: *GPU,
        pdev: vk.PhysicalDevice,
        compute: bool,
    ) !?QueueFamilyIndex {
        const families = try self.instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, self.allocator);
        defer self.allocator.free(families);

        var family_indexes: QueueFamilyIndex = .{};

        for (families, 0..) |properties, i| {
            const family: u32 = @intCast(i);

            if (self.surface) |surface| {
                if (family_indexes.graphical == null and properties.queue_flags.graphics_bit) {
                    family_indexes.graphical = family;
                }

                if (family_indexes.present == null and (try self.instance.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == .true) {
                    family_indexes.present = family;
                }
            }
            if (compute) {
                if (family_indexes.compute == null and properties.queue_flags.compute_bit) {
                    family_indexes.compute = family;
                }
            }
        }
        if (compute and family_indexes.compute == null) {
            return null;
        }
        if (self.surface != null and (family_indexes.present == null or family_indexes.graphical == null)) {
            return null;
        }
        return family_indexes;
    }

    fn checkSuitablePhysicalDevice(self: *GPU, pdev: vk.PhysicalDevice, compute: bool, extensions: [][*:0]const u8) !?QueueFamilyIndex {
        if (!try self.checkExtensionSupport(pdev, extensions)) {
            return null;
        }

        if (self.surface) |surface| {
            if (!try self.checkSurfaceSupport(pdev, surface)) {
                return null;
            }
        }

        if (try self.requestQueueFamilies(pdev, compute)) |family_indexes| {
            return family_indexes;
        }
        return null;
    }

    fn initPhysicalDevice(self: *GPU, compute: bool, extensions: [][*:0]const u8) !QueueFamilyIndex {
        const pdevs = try self.instance.enumeratePhysicalDevicesAlloc(self.allocator);
        defer self.allocator.free(pdevs);

        for (pdevs) |pdev| {
            if (try self.checkSuitablePhysicalDevice(pdev, compute, extensions)) |family_indexes| {
                self.pdev = pdev;
                return family_indexes;
            }
        }

        return error.PhysicalDeviceAndQueueNotFound;
    }

    fn initDevice(self: *GPU, family_indexes: QueueFamilyIndex, extensions: [][*:0]const u8) !void {
        const vkd = try self.allocator.create(DeviceWrapper);
        errdefer self.allocator.destroy(vkd);

        var qci: std.ArrayList(vk.DeviceQueueCreateInfo) = .empty;
        defer qci.deinit(self.allocator);

        if (family_indexes.compute) |c_idx| {
            try qci.append(self.allocator, .{
                .queue_family_index = c_idx,
                .queue_count = 1,
                .p_queue_priorities = &.{1},
            });
        }

        if (family_indexes.graphical) |g_idx| {
            if (g_idx != family_indexes.compute) {
                try qci.append(self.allocator, .{
                    .queue_family_index = g_idx,
                    .queue_count = 1,
                    .p_queue_priorities = &.{1},
                });
            }
        }

        if (family_indexes.present) |p_idx| {
            if (p_idx != family_indexes.graphical and p_idx != family_indexes.compute) {
                try qci.append(self.allocator, .{
                    .queue_family_index = p_idx,
                    .queue_count = 1,
                    .p_queue_priorities = &.{1},
                });
            }
        }

        // var shader_float16_int8_features = vk.PhysicalDeviceShaderFloat16Int8Features{
        //     .shader_float_16 = vk.Bool32.true,
        //     .shader_int_8 = vk.Bool32.true,
        // };
        const device_features = vk.PhysicalDeviceFeatures{
            .shader_int_64 = vk.Bool32.true,
            // .shader_int_16 = vk.Bool32.true, // Enable 16-bit integers
        };

        const device = try self.instance.createDevice(self.pdev, &vk.DeviceCreateInfo{
            .queue_create_info_count = @intCast(qci.items.len),
            .enabled_extension_count = @intCast(extensions.len),
            .pp_enabled_extension_names = @ptrCast(extensions.ptr),
            .p_queue_create_infos = qci.items.ptr,
            .p_enabled_features = @ptrCast(&device_features),
        }, null);

        vkd.* = DeviceWrapper.load(device, self.instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
        self.dev = Device.init(device, vkd);
    }

    fn getVulkanLoader() !std.DynLib {
        const target = @import("builtin").target;
        return switch (target.os.tag) {
            .linux => std.DynLib.open("libvulkan.so.1"),
            .windows => std.DynLib.open("vulkan-1.dll"),
            .macos => blk: {
                if (std.DynLib.open("libvulkan.1.dylib")) |lib| break :blk lib;
                break :blk try std.DynLib.open("libMoltenVK.dylib");
            },
            else => std.DynLib.open("libvulkan.so.1"),
        };
    }

    fn initSurface(self: *GPU, window: *zglfw.Window) !void {
        var surface: vk.SurfaceKHR = undefined;
        try zglfw.createWindowSurface(self.instance.handle, window, null, &surface);
        self.surface = surface;
    }

    fn initInstance(self: *GPU, app_name: [:0]const u8, extensions: [][*:0]const u8, layers: [][*:0]const u8) !void {
        var lib = try getVulkanLoader();
        defer lib.close();

        const getInstanceProcAddr = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.MissingSymbol;

        self.vkb = BaseWrapper.load(getInstanceProcAddr);

        const instance = try self.vkb.createInstance(&vk.InstanceCreateInfo{
            .p_application_info = &.{
                .p_application_name = app_name,
                .application_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
                .p_engine_name = app_name,
                .engine_version = @bitCast(vk.makeApiVersion(0, 0, 0, 0)),
                .api_version = @bitCast(vk.API_VERSION_1_2),
            },
            .enabled_layer_count = @intCast(layers.len),
            .pp_enabled_layer_names = layers.ptr,
            .enabled_extension_count = @intCast(extensions.len),
            .pp_enabled_extension_names = extensions.ptr,
            // enumerate_portability_bit_khr to support vulkan in mac os
            // see https://github.com/glfw/glfw/issues/2335
            .flags = .{ .enumerate_portability_bit_khr = true },
        }, null);

        const vki = try self.allocator.create(InstanceWrapper);
        errdefer self.allocator.destroy(vki);
        vki.* = InstanceWrapper.load(instance, self.vkb.dispatch.vkGetInstanceProcAddr.?);
        self.instance = Instance.init(instance, vki);
    }

    pub fn deinit(self: GPU) void {
        self.dev.destroyDevice(null);
        if (self.surface) |surface| {
            self.instance.destroySurfaceKHR(surface, null);
        }
        self.instance.destroyInstance(null);

        // Don't forget to free the tables to prevent a memory leak.
        self.allocator.destroy(self.dev.wrapper);
        self.allocator.destroy(self.instance.wrapper);
    }

    pub fn findMemoryTypeIndex(self: *GPU, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    pub fn allocate(self: *GPU, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.dev.allocateMemory(&.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }
};
