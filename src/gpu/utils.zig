const std = @import("std");

fn ceql(a: anytype, b: anytype) bool {
    var i: usize = 0;
    while (true) : (i += 1) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

pub fn containString(slice_of_slices: anytype, item: anytype) bool {
    for (slice_of_slices) |name| {
        if (ceql(name, item)) {
            return true;
        }
    }
    return false;
}

pub fn printPositiveBitFields(value: anytype, writer: anytype) !void {
    const T = @TypeOf(value);
    comptime {
        const info = @typeInfo(T);
        if (info != .@"struct") {
            @compileError("Expected a struct");
        }
    }
    const fields = comptime @typeInfo(T).@"struct".fields;
    inline for (fields) |field| {
        const field_value = @field(value, field.name);
        const FieldType = field.type;
        const is_positive = is_positive: {
            const info = @typeInfo(FieldType);
            switch (info) {
                .int, .comptime_int => break :is_positive field_value > 0,
                .bool => break :is_positive field_value,
                inline else => {
                    @compileError(std.fmt.comptimePrint("Invalid field type: {s} ({})", .{ field.name, FieldType }));
                },
            }
        };
        // for different uN types
        if (is_positive) {
            try writer.print("{s}, ", .{field.name});
        }
    }
}
