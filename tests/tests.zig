const std = @import("std");

test "test suite" {
    _ = @import("gpu_tests.zig");
    _ = @import("buffer_tests.zig");
    _ = @import("pipeline_tests.zig");
    _ = @import("input_buffer_tests.zig");
    _ = @import("staging_buffer_tests.zig");
    _ = @import("descriptor_set_tests.zig");
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
