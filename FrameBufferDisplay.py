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

        bytes_per_pixel = self._bpp // 8
        self._screensize = xres * yres * bytes_per_pixel

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

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        converted = self._convert_frame(frame)

        self._fbp.seek(0)
        self._fbp.write(converted.tobytes())


    def convert_frame(self, frame):
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
