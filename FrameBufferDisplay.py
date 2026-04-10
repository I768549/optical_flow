import fcntl
import mmap
import struct
import os

import cv2
import numpy as np

# ioctl constants from <linux/fb.h>
FBIOGET_VSCREENINFO = 0x4600
FBIOPUT_VSCREENINFO = 0x4601
FBIOGET_FSCREENINFO = 0x4602

_VSCREENINFO_SIZE = 160

# ioctl for TTY graphics mode
KDSETMODE = 0x4B3A
KD_TEXT = 0x00
KD_GRAPHICS = 0x01

class FrameBufferDisplay:
    def __init__(self, desired_width: int, desired_height: int,
                 fb_path: str = "/dev/fb0", tty_path: str = "/dev/tty1"):
        self.requested_width = desired_width
        self.requested_height = desired_height
        self._fb_path = fb_path
        self._tty_path = tty_path
        
        self._fb_fd = None
        self._tty_fd = None
        self._fbp = None
        self._fb_array = None  # <-- Указатель numpy на память экрана
        self._screensize = 0
        self._bpp = 0
        self._line_length = 0
        
        self._xres = 0
        self._yres = 0
        self._yres_virtual = 0
        self._red_offset = 0
        self._green_offset = 0
        self._blue_offset = 0
        
        self._is_initialized = False

    def initialize(self):
        self._fb_fd = os.open(self._fb_path, os.O_RDWR) 

        vinfo_buf = bytearray(_VSCREENINFO_SIZE)
        fcntl.ioctl(self._fb_fd, FBIOGET_VSCREENINFO, vinfo_buf)

        struct.pack_into("IIII", vinfo_buf, 0,
                         self.requested_width, self.requested_height, 
                         self.requested_width, self.requested_height)
        try:
            fcntl.ioctl(self._fb_fd, FBIOPUT_VSCREENINFO, vinfo_buf)
        except OSError:
            pass

        fcntl.ioctl(self._fb_fd, FBIOGET_VSCREENINFO, vinfo_buf)

        self._xres, self._yres = struct.unpack_from("II", vinfo_buf, 0)
        self._yres_virtual = struct.unpack_from("I", vinfo_buf, 12)[0]
        self._bpp = struct.unpack_from("I", vinfo_buf, 24)[0]

        self._red_offset = struct.unpack_from("I", vinfo_buf, 32)[0]
        self._green_offset = struct.unpack_from("I", vinfo_buf, 44)[0]
        self._blue_offset = struct.unpack_from("I", vinfo_buf, 56)[0]

        finfo_buf = bytearray(88)
        fcntl.ioctl(self._fb_fd, FBIOGET_FSCREENINFO, finfo_buf)
        fix_fmt = "@16s L I I I I H H H I"
        fields = struct.unpack_from(fix_fmt, finfo_buf, 0)
        self._line_length = fields[9]

        self._screensize = self._line_length * self._yres_virtual

        self._fbp = mmap.mmap(self._fb_fd, self._screensize,
                              mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

        # Создаем numpy массив, который смотрит прямо в /dev/fb0
        self._fb_array = np.ndarray(
            shape=(self._yres_virtual, self._line_length),
            dtype=np.uint8,
            buffer=self._fbp
        )

        try:
            self._tty_fd = os.open(self._tty_path, os.O_RDWR)
            fcntl.ioctl(self._tty_fd, KDSETMODE, KD_GRAPHICS)
        except OSError:
            self._tty_fd = None

        self._is_initialized = True

    def imshow(self, frame: np.ndarray):
        if not self._is_initialized:
            return

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Оптимизация 2: Ресайз это ОЧЕНЬ дорого. Делаем его только если реально нужно.
        if frame.shape[1] != self._xres or frame.shape[0] != self._yres:
            frame = cv2.resize(frame, (self._xres, self._yres), interpolation=cv2.INTER_NEAREST)

        converted = self._convert_frame(frame)
        self._blit_frame(converted)

    def _blit_frame(self, converted: np.ndarray):
        rows = min(converted.shape[0], self._yres_virtual)
        if rows <= 0:
            return

        # Превращаем картинку в плоский по X массив байтов (строка за строкой)
        src_bytes = converted.view(np.uint8).reshape(converted.shape[0], -1)
        dst_row_bytes = min(src_bytes.shape[1], self._line_length)

        # ВЕКТОРНАЯ ЗАПИСЬ: Копируем весь блок памяти за один вызов C-функции под капотом NumPy.
        self._fb_array[:rows, :dst_row_bytes] = src_bytes[:rows, :dst_row_bytes]

    def _convert_frame(self, frame):
        if self._bpp == 16:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
        elif self._bpp == 24:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self._bpp == 32:
            if self._blue_offset == 0:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            else:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        else:
            raise RuntimeError(f"Unsupported pixel format: {self._bpp} bpp")
            
    def close(self):
        if self._tty_fd is not None:
            try:
                fcntl.ioctl(self._tty_fd, KDSETMODE, KD_TEXT)
            except OSError:
                pass
            os.close(self._tty_fd)
            self._tty_fd = None

        # Убиваем ссылку numpy перед закрытием mmap
        self._fb_array = None

        if self._fbp is not None:
            self._fbp.close()
            self._fbp = None

        if self._fb_fd is not None:
            os.close(self._fb_fd)
            self._fb_fd = None

        self._is_initialized = False

    def __del__(self):
        self.close()