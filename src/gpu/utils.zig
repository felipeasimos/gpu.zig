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
