const vk = @import("vulkan");

const Device = vk.DeviceProxy;

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    pub fn init(device: Device, family: u32) Queue {
        return .{
            .handle = device.getDeviceQueue(family, 0),
            .family = family,
        };
    }
};
