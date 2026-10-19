import cv2
import threading
import time
import numpy as np


class ThreadedCamera:
    """
    Asynchronous camera capture running in a dedicated background thread.
    
    Benefits:
    - Decouples fast local webcams from slower/high-latency Wi-Fi streams (e.g., DroidCam/IP Webcam).
    - Constantly drains the OpenCV buffer so that read() always returns the latest real-time frame
      without accumulating seconds of latency.
    - Automatically attempts reconnection if a network stream drops.
    """

    def __init__(self, source, name="Camera", reconnect_delay=2.0):
        self.source = source
        self.name = name
        self.reconnect_delay = reconnect_delay

        self.cap = None
        self.latest_frame = None
        self.has_new_frame = False
        self.is_connected = False
        self.is_running = False

        self.lock = threading.Lock()
        self.thread = None

        self.fps = 0.0
        self._fps_count = 0
        self._fps_timer = time.monotonic()

        self._open_capture()
        self.start()

    def _open_capture(self):
        try:
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.source)
            else:
                self.cap = cv2.VideoCapture(self.source)
                # For network streams, minimize internal buffer to prevent delay accumulation
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                self.is_connected = True
                print(f"[{self.name}] Successfully opened stream: {self.source}")
            else:
                self.is_connected = False
                print(f"[{self.name}] Warning: Unable to open stream: {self.source}")
        except Exception as e:
            self.is_connected = False
            print(f"[{self.name}] Error opening stream: {e}")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(
            target=self._worker_loop,
            name=f"ThreadedCam-{self.name}",
            daemon=True
        )
        self.thread.start()

    def _worker_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._open_capture()
                continue

            ret, frame = self.cap.read()

            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame
                    self.has_new_frame = True
                    self.is_connected = True

                    # Update measured FPS
                    self._fps_count += 1
                    now = time.monotonic()
                    elapsed = now - self._fps_timer
                    if elapsed >= 1.0:
                        self.fps = self._fps_count / elapsed
                        self._fps_count = 0
                        self._fps_timer = now
            else:
                with self.lock:
                    self.is_connected = False
                # If reading failed on a network stream, wait before retry
                time.sleep(0.05)

    def read(self):
        """
        Returns (success, frame).
        Does not block on network I/O; returns immediately with a copy of the latest frame.
        """
        with self.lock:
            if self.latest_frame is not None and self.is_connected:
                return True, self.latest_frame.copy()
            return False, None

    def is_opened(self):
        return self.is_connected

    def release(self):
        self.is_running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print(f"[{self.name}] Stream released.")
