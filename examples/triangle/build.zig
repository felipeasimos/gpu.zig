const std = @import("std");

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

pub const GpuBackend = enum {
    wgpu,
    vulkan,
};

pub fn build(b: *std.Build) void {
    const gpu_backend = b.option(GpuBackend, "gpu_backend", "Which GPU backend to use") orelse .vulkan;
    const use_llvm = b.option(bool, "use_llvm", "Use LLVM") orelse (gpu_backend == .wgpu);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const mod = b.dependency("gpu_zig", .{}).module("gpu");
    const exe = b.addExecutable(.{
        .name = "compute_square",
        .use_llvm = use_llvm,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "gpu", .module = mod },
            },
        }),
    });
    get_spirv_shaders_using_glslc(b, "src/shaders/", exe.root_module) catch unreachable;

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
}
