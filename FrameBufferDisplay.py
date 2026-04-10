import fcntl
import mmap
import struct
import os

import cv2
import numpy as np

# ioctl constants from <linux/fb.h>

# get screen params (resolution, bpp, offsets of colors)
FBIOGET_VSCREENINFO = 0x4600
# set screen params (resolution, bpp, offsets of colors)
FBIOPUT_VSCREENINFO = 0x4601
# get fixed params
FBIOGET_FSCREENINFO = 0x4602
# xres, yres, xres_virtual, yres_virtual, xoffset, yoffset

# struct fb_var_screeninfo — 160 bytes
_VSCREENINFO_SIZE = 160

# ioctl for TTY graphics mode
KDSETMODE = 0x4B3A #set text/graphics mode from <linux/kd.h>
KD_TEXT = 0x00
KD_GRAPHICS = 0x01


class FrameBufferDisplay:
    def __init__(self, desired_width: int, desired_height: int,
                 fb_path: str = "/dev/fb0", tty_path: str = "/dev/tty1"):
        self.width = desired_width
        self.height = desired_height
        self._fb_path = fb_path
        self._tty_path = tty_path
        # file descriptors
        self._fb_fd = None
        self._tty_fd = None
        # frame buffer pointer - mmap.mmap object
        self._fbp = None
        self._screensize = 0
        # bits per pixel
        self._bpp = 0
        self._line_length = 0
        self._xres = 0
        self._yres = 0
        self._yres_virtual = 0
        self._xoffset = 0
        self._yoffset = 0
        self._red_offset = 0
        self._green_offset = 0
        self._blue_offset = 0
        self._is_initialized = False

    def initialize(self):
        self._fb_fd = os.open(self._fb_path, os.O_RDWR) 

        # Get variable screen info
        vinfo_buf = bytearray(_VSCREENINFO_SIZE)
        fcntl.ioctl(self._fb_fd, FBIOGET_VSCREENINFO, vinfo_buf)

        # Set desired resolution
        # xres(4), yres(4), xres_virtual(4), yres_virtual(4)
        struct.pack_into("IIII", vinfo_buf, 0,
                         self.width, self.height, self.width, self.height)
        fcntl.ioctl(self._fb_fd, FBIOPUT_VSCREENINFO, vinfo_buf)

        # Re-read after setting
        fcntl.ioctl(self._fb_fd, FBIOGET_VSCREENINFO, vinfo_buf)

        xres, yres = struct.unpack_from("II", vinfo_buf, 0)
        self._xres = xres
        self._yres = yres
        self._yres_virtual = struct.unpack_from("I", vinfo_buf, 12)[0]
        self._xoffset = struct.unpack_from("I", vinfo_buf, 16)[0]
        self._yoffset = struct.unpack_from("I", vinfo_buf, 20)[0]
        self._bpp = struct.unpack_from("I", vinfo_buf, 24)[0]  # bits_per_pixel at offset 24

        # Color offsets: red at offset 32, green at 44, blue at 56
        # Each bitfield is: offset(4), length(4), msb_right(4) = 12 bytes
        self._red_offset = struct.unpack_from("I", vinfo_buf, 32)[0]
        self._green_offset = struct.unpack_from("I", vinfo_buf, 44)[0]
        self._blue_offset = struct.unpack_from("I", vinfo_buf, 56)[0]

        # Get fixed screen info for line_length
        # line_length at offset 48, struct size 88
        finfo_buf = bytearray(88)
        fcntl.ioctl(self._fb_fd, FBIOGET_FSCREENINFO, finfo_buf)
        self._line_length = struct.unpack_from("I", finfo_buf, 48)[0]

        # Framebuffer memory is organized by line stride (line_length),
        # not strictly width * bytes_per_pixel.
        self._screensize = self._line_length * self._yres_virtual

        self._fbp = mmap.mmap(self._fb_fd, self._screensize,
                              mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

        try:
            self._tty_fd = os.open(self._tty_path, os.O_RDWR)
            fcntl.ioctl(self._tty_fd, KDSETMODE, KD_GRAPHICS)
        except OSError:
            self._tty_fd = None

        self._is_initialized = True

    def imshow(self, frame: np.ndarray):
        if not self._is_initialized:
            return

        # Always render to actual visible framebuffer resolution.
        # Driver may ignore requested mode and keep native xres/yres.
        target_w = self._xres if self._xres > 0 else self.width
        target_h = self._yres if self._yres > 0 else self.height

        if frame.shape[1] != target_w or frame.shape[0] != target_h:
            frame = cv2.resize(frame, (target_w, target_h))

        converted = self._convert_frame(frame)
        self._blit_frame(converted)

    def _blit_frame(self, converted: np.ndarray):
        rows = min(converted.shape[0], max(0, self._yres - self._yoffset))

        if not converted.flags["C_CONTIGUOUS"]:
            converted = np.ascontiguousarray(converted)

        src_stride = converted.strides[0]
        dst_stride = self._line_length
        bytes_per_pixel = max(1, self._bpp // 8)
        dst_x_bytes = self._xoffset * bytes_per_pixel
        dst_row_capacity = max(0, dst_stride - dst_x_bytes)

        if rows <= 0 or dst_row_capacity <= 0:
            return

        # Fast path: strides are equal for visible rows
        if src_stride == dst_stride and dst_x_bytes == 0:
            byte_count = rows * dst_stride
            self._fbp.seek(self._yoffset * dst_stride)
            self._fbp.write(converted.reshape(-1).view(np.uint8)[:byte_count].tobytes())
            return

        # Generic path: copy line-by-line, respecting framebuffer pitch
        src_bytes = converted.view(np.uint8).reshape(converted.shape[0], -1)
        row_bytes = min(src_bytes.shape[1], dst_row_capacity)

        for y in range(rows):
            dst_pos = (self._yoffset + y) * dst_stride + dst_x_bytes
            self._fbp.seek(dst_pos)
            self._fbp.write(src_bytes[y, :row_bytes].tobytes())


    def _convert_frame(self, frame):
        if self._bpp == 16:
            converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
        elif self._bpp == 24:
            converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self._bpp == 32:
            if self._blue_offset == 0:
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            else:
                converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            raise RuntimeError(f"Unsupported pixel format: {self._bpp} bpp")
        
        return converted_frame
            
    def close(self):
        if self._tty_fd is not None:
            try:
                fcntl.ioctl(self._tty_fd, KDSETMODE, KD_TEXT)
            except OSError:
                pass
            os.close(self._tty_fd)
            self._tty_fd = None

        if self._fbp is not None:
            self._fbp.close()
            self._fbp = None

        if self._fb_fd is not None:
            os.close(self._fb_fd)
            self._fb_fd = None

        self._is_initialized = False

    def __del__(self):
        self.close()
