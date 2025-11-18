import ctypes
from OpenGL import EGL, error

# 定义 eglQueryDeviceBinaryEXT 的函数类型
PFNEGLQUERYDEVICEBINARYEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLDeviceEXT,     # 设备
    EGL.EGLint,           # 查询的属性
    EGL.EGLint,           # 最大大小
    ctypes.POINTER(ctypes.c_ubyte),  # 输出值（UUID）
    ctypes.POINTER(EGL.EGLint)        # 写入的大小
)

# 试图获取 eglQueryDeviceBinaryEXT 函数的地址
eglQueryDeviceBinaryEXT = PFNEGLQUERYDEVICEBINARYEXTPROC(
    EGL.eglGetProcAddress('eglQueryDeviceBinaryEXT')
)

# From the EGL_EXT_device_enumeration extension.
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLint,
    ctypes.POINTER(EGL.EGLDeviceEXT),
    ctypes.POINTER(EGL.EGLint),
)
try:
    _eglQueryDevicesEXT = PFNEGLQUERYDEVICESEXTPROC(  # pylint: disable=invalid-name
        EGL.eglGetProcAddress('eglQueryDevicesEXT'))
except TypeError as e:
    raise ImportError('eglQueryDevicesEXT is not available.') from e


# From the EGL_EXT_platform_device extension.
EGL_PLATFORM_DEVICE_EXT = 0x313F
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLDisplay, EGL.EGLenum, ctypes.c_void_p, ctypes.POINTER(EGL.EGLint))
try:
    eglGetPlatformDisplayEXT = PFNEGLGETPLATFORMDISPLAYEXTPROC(  # pylint: disable=invalid-name
        EGL.eglGetProcAddress('eglGetPlatformDisplayEXT'))
except TypeError as e:
    raise ImportError('eglGetPlatformDisplayEXT is not available.') from e


# Wrap raw _eglQueryDevicesEXT function into something more Pythonic.
def eglQueryDevicesEXT(max_devices=10):  # pylint: disable=invalid-name
    devices = (EGL.EGLDeviceEXT * max_devices)()
    num_devices = EGL.EGLint()
    success = _eglQueryDevicesEXT(max_devices, devices, num_devices)
    if success == EGL.EGL_TRUE:
        return [devices[i] for i in range(num_devices.value)]
    else:
        raise error.GLError(err=EGL.eglGetError(),
                            baseOperation=eglQueryDevicesEXT,
                            result=success)

# 获取可用设备的函数
def get_egl_devices(max_devices=10):
    devices = (EGL.EGLDeviceEXT * max_devices)()
    num_devices = EGL.EGLint()

    success = _eglQueryDevicesEXT(max_devices, devices, ctypes.byref(num_devices))
    if success != EGL.EGL_TRUE:
        raise error.GLError(err=EGL.eglGetError(),
                            baseOperation=get_egl_devices,
                            result=success)

    return devices[:num_devices.value]

# 查询设备 UUID
def get_device_uuids():
    devices = get_egl_devices()
    uuids = []

    for device in devices:
        uuid = (ctypes.c_ubyte * 16)()  # 创建一个16字节缓冲区
        size = EGL.EGLint()  # 存储写入的大小

        # 定义缺失的EGL常量
        EGL_DEVICE_UUID_EXT = 0x335C
        success = eglQueryDeviceBinaryEXT(device, EGL_DEVICE_UUID_EXT, ctypes.sizeof(uuid), uuid, ctypes.byref(size))

        if success == EGL.EGL_TRUE:
            uuids.append(bytes(uuid))  # 将UUID保存为字节格式
        else:
            print(f"Failed to query UUID for device: {EGL.eglGetError()}")

    return uuids

if __name__ == "__main__":
    try:
        device_uuids = get_device_uuids()
        print("Device UUIDs:")
        for uuid in device_uuids:
            print(uuid.hex())  # 打印 UUID 的十六进制表示
    except error.GLError as e:
        print(f"An error occurred: {e}")