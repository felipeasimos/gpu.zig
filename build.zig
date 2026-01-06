const std = @import("std");

const GpuBackend = enum(u1) {
    wgpu,
    vulkan,
};

pub fn getGitHash(b: *std.Build) ![]const u8 {
    var process = std.process.Child.init(&[_][]const u8{ "git", "rev-parse", "HEAD" }, b.allocator);
    process.stdout_behavior = .Pipe;

    process.spawn() catch {
        return error.GitNotAvailable;
    };

    // Get the output
    const result: []u8 = process.stdout.?.readToEndAlloc(b.allocator, 1024) catch {
        _ = process.kill() catch @panic("Error getting git hash");
        return error.ReadFailed;
    };

    // Wait for process to finish
    const term = process.wait() catch {
        return error.WaitFailed;
    };

    // Check if process succeeded
    if (term.Exited != 0) {
        return error.GitCommandFailed;
    }

    // Trim trailing newline
    const trimmed = std.mem.trim(u8, result, "\r\n");
    if (trimmed.len != 40) {
        return error.InvalidResponse;
    }
    return trimmed;
}

pub fn get_spirv_shaders(b: *std.Build, shader_folder: []const u8, module: *std.Build.Module) !void {
    const spirv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv32,
        .os_tag = .vulkan,
        .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
        .ofmt = .spirv,
        .cpu_features_add = std.Target.spirv.featureSet(&.{
            // .arbitrary_precision_integers,
            // .float16,
            // .float64,
            // .generic_pointer,
            // .int64,
            // .storage_push_constant16,
            // .v1_0,
            // .v1_1,
            // .v1_2,
            // .v1_3,
            // .v1_4,
            // .v1_5,
            // .v1_6,
            // .variable_pointers,
            // .vector16,
        }),
    });

    var dir = try std.fs.cwd().openDir(shader_folder, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(b.allocator);
    defer walker.deinit();

    var shaders = std.ArrayList(*std.Build.Step.Compile).empty;

    while (try walker.next()) |entry| {
        if (!std.mem.endsWith(u8, entry.path, ".zig")) {
            continue;
        }
        const relative_path = try std.fs.path.join(b.allocator, &.{
            shader_folder,
            entry.path,
        });
        const shader_name = std.fs.path.stem(std.fs.path.basename(entry.path));

        const shader = b.addObject(.{
            .name = shader_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(relative_path),
                .target = spirv_target,
            }),
            .use_llvm = false, // llvm can't compile spirv
        });

        try shaders.append(b.allocator, shader);

        module.addAnonymousImport(shader_name, .{
            .root_source_file = shader.getEmittedBin(),
        });
    }
}

pub fn get_spirv_shaders_using_glslc(b: *std.Build, shader_folder: []const u8, module: *std.Build.Module) !void {
    var dir = try b.build_root.handle.openDir(shader_folder, .{ .iterate = true });
    defer dir.close();

    var walker = try dir.walk(b.allocator);
    defer walker.deinit();

    while (try walker.next()) |entry| {
        if (std.mem.endsWith(u8, entry.path, ".zig") or std.mem.endsWith(u8, entry.path, ".h")) {
            continue;
        }
        const relative_path = try std.fs.path.join(b.allocator, &.{
            shader_folder,
            entry.path,
        });
        const shader_name = std.fs.path.stem(std.fs.path.basename(entry.path));

        // command
        const shader_cmd = b.addSystemCommand(&.{
            "glslc",
            "--target-env=vulkan1.2",
            "-I",
            shader_folder,
            "-o",
        });
        const shader_spv = shader_cmd.addOutputFileArg(shader_name);
        shader_cmd.addFileArg(b.path(relative_path));

        module.addAnonymousImport(shader_name, .{
            .root_source_file = shader_spv,
        });
    }
}

pub fn build(b: *std.Build) void {
    const gpu_backend = b.option(GpuBackend, "gpu_backend", "Which GPU backend to use") orelse .vulkan;
    const use_llvm = b.option(bool, "use_llvm", "Use LLVM") orelse (gpu_backend == .wgpu);

    const git_commit_hash = "unknown git hash";
    // const git_commit_hash = getGitHash(b) catch "unknown git hash";

    const options = b.addOptions();
    options.addOption([]const u8, "git_commit_hash", git_commit_hash);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vulkan = b.dependency("vulkan", .{
        .registry = b.path("lib/vk.xml"),
        .video = b.path("lib/video.xml"),
    }).module("vulkan-zig");
    // const wgpu = b.dependency("wgpu_native_zig", .{
    //     .link_mode = std.builtin.LinkMode.static,
    // }).module("wgpu");
    const gpu_import: std.Build.Module.Import = gpu_import: {
        break :gpu_import .{
            .name = "vulkan",
            .module = vulkan,
        };
    };

    const contract = b.dependency("contract", .{});
    const contract_mod = contract.module("contract");

    const zglfw = b.dependency("zglfw", .{ .import_vulkan = true });
    const zglfw_mod = zglfw.module("root");
    zglfw_mod.addImport("vulkan", vulkan);

    // library
    const gpu = b.addModule("gpu", .{
        .root_source_file = b.path("src/gpu/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            gpu_import,
            .{ .name = "contract", .module = contract_mod },
            .{ .name = "zglfw", .module = zglfw_mod },
        },
    });
    gpu.addOptions("options", options);

    // get_spirv_shaders_using_glslc(b, "src/gpu/shaders/", lib) catch unreachable;
    // get_spirv_shaders(b, "src/gpu/shaders/", lib) catch unreachable;

    // library tests
    const lib_tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
        },
    });
    const lib_tests = b.addTest(.{
        .root_module = lib_tests_mod,
        .use_llvm = use_llvm,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_lib_tests.step);

    // executable
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
            .{ .name = "zglfw", .module = zglfw_mod },
        },
    });

    get_spirv_shaders_using_glslc(b, "src/shaders/", exe_mod) catch unreachable;
    // get_spirv_shaders(b, "src/shaders/", exe_mod) catch unreachable;

    if (target.result.os.tag != .emscripten) {
        gpu.linkLibrary(zglfw.artifact("glfw"));
    }

    const exe = b.addExecutable(.{
        .name = "gpu",
        .root_module = exe_mod,
        .use_llvm = use_llvm,
    });
    b.installArtifact(exe);
    const run_exe = b.addRunArtifact(exe);

    const exe_step = b.step("run", "Run executable");
    exe_step.dependOn(&run_exe.step);

    // add check step for fast ZLS diagnostics on tests and library
    const check_lib_tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "gpu", .module = gpu },
        },
    });
    const check_lib_tests = b.addTest(.{
        .root_module = check_lib_tests_mod,
    });
    const check_exe = b.addExecutable(.{
        .name = "check_gpu",
        .root_module = exe_mod,
        .use_llvm = use_llvm,
    });
    const check_step = b.step("check", "Check for compile errors");
    check_step.dependOn(&check_lib_tests.step);
    check_step.dependOn(&check_exe.step);
}
