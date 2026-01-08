const std = @import("std");
const gpu_zig = @import("gpu");
const GPU = gpu_zig.GPU;
const testing = std.testing;
const zglfw = gpu_zig.zglfw;

const WINDOW_WIDTH = 512;
const WINDOW_HEIGHT = 512;

/// needed for vulkan to load
fn initZglfwForVulkan() !void {
    try zglfw.init();

    if (!zglfw.isVulkanSupported()) {
        std.log.err("GLFW could not find libvulkan", .{});
        return error.NoVulkan;
    }

    zglfw.windowHint(.client_api, .no_api);
    zglfw.windowHint(.visible, false);
}

test "test failure to init" {
    const allocator = std.testing.allocator;
    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    const gpu_res = GPU.Builder.init(allocator)
        .appName(title)
        .build();
    try testing.expectError(error.GPUMustBeAtLeastComputeOrGraphical, gpu_res);
}

test "test compute queue init" {
    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        // .graphics(window)
        // .validation()
        .build();
    defer gpu.deinit();

    try testing.expect(gpu.compute_queue != null);
    try testing.expectEqual(null, gpu.graphics_queue);
    try testing.expectEqual(null, gpu.present_queue);
}

test "test compute queue with validation init" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .compute()
        // .graphics(window)
        .validation()
        .build();
    defer gpu.deinit();

    try testing.expect(gpu.compute_queue != null);
    try testing.expectEqual(null, gpu.graphics_queue);
    try testing.expectEqual(null, gpu.present_queue);
}

test "test graphics queues init" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    const window = try zglfw.Window.create(WINDOW_WIDTH, WINDOW_HEIGHT, title, null);
    defer window.destroy();

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .graphics(window)
        // .validation()
        .build();
    defer gpu.deinit();

    try testing.expectEqual(null, gpu.compute_queue);
    try testing.expect(gpu.graphics_queue != null);
    try testing.expect(gpu.present_queue != null);
}

test "test graphics queues with validation init" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    const window = try zglfw.Window.create(WINDOW_WIDTH, WINDOW_HEIGHT, title, null);
    defer window.destroy();

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .graphics(window)
        .validation()
        .build();
    defer gpu.deinit();

    try testing.expectEqual(null, gpu.compute_queue);
    try testing.expect(gpu.graphics_queue != null);
    try testing.expect(gpu.present_queue != null);
}

test "test graphics and compute queues with init" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    const window = try zglfw.Window.create(WINDOW_WIDTH, WINDOW_HEIGHT, title, null);
    defer window.destroy();

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .graphics(window)
        .compute()
        // .validation()
        .build();
    defer gpu.deinit();

    try testing.expect(gpu.compute_queue != null);
    try testing.expect(gpu.graphics_queue != null);
    try testing.expect(gpu.present_queue != null);
}

test "test graphics and compute queues with validation init" {
    try initZglfwForVulkan();
    defer zglfw.terminate();

    const title = std.fmt.comptimePrint("SVDAG - {s}", .{gpu_zig.options.git_commit_hash});

    const window = try zglfw.Window.create(WINDOW_WIDTH, WINDOW_HEIGHT, title, null);
    defer window.destroy();

    var gpu = try GPU.Builder.init(testing.allocator)
        .appName(title)
        .graphics(window)
        .compute()
        .validation()
        .build();
    defer gpu.deinit();

    try testing.expect(gpu.compute_queue != null);
    try testing.expect(gpu.graphics_queue != null);
    try testing.expect(gpu.present_queue != null);
}
