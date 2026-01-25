#!/usr/bin/env python3
"""
Raspberry Pi Wildfire Detection Client
Captures RGB and thermal data, sends to computer for processing
"""

import time
import json
import socket
import struct
import numpy as np
from threading import Thread, Lock
import logging

# Camera imports with fallbacks
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    import cv2

try:
    import board
    import busio
    import adafruit_mlx90640
    THERMAL_AVAILABLE = True
except ImportError:
    THERMAL_AVAILABLE = False
    logging.warning("MLX90640 libraries not available - thermal camera disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RGBCamera:
    """Handles RGB image capture from Raspberry Pi Camera Module v2"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.camera = None
        self.lock = Lock()
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the appropriate camera interface"""
        try:
            if PICAMERA2_AVAILABLE:
                self.camera = Picamera2()
                config = self.camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": 60.0},
                )
                
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Allow camera to warm up
                logger.info("Initialized picamera2")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, 30)  # Request 30 fps
                logger.info("Initialized OpenCV camera")
        except Exception as e:
            logger.error(f"Failed to initialize RGB camera: {e}")
            raise
    
    def read(self):
        """Capture a single RGB frame"""
        with self.lock:
            try:
                if PICAMERA2_AVAILABLE:
                    frame = self.camera.capture_array()
                    return frame if frame is not None else None
                else:
                    ret, frame = self.camera.read()
                    if ret:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return None
            except Exception as e:
                logger.error(f"RGB capture failed: {e}")
                return None
    
    def close(self):
        """Release camera resources"""
        if self.camera:
            if PICAMERA2_AVAILABLE:
                self.camera.stop()
            else:
                self.camera.release()


class ThermalCamera:
    """Handles thermal image capture from MLX90640"""
    
    def __init__(self):
        self.camera = None
        self.lock = Lock()
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize MLX90640 thermal camera"""
        if not THERMAL_AVAILABLE:
            logger.warning("Thermal camera not available")
            return
        
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)  # Increased I2C frequency
            self.camera = adafruit_mlx90640.MLX90640(i2c)
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ  # Increased to 16Hz
            logger.info("Initialized MLX90640 thermal camera at 16Hz")
        except Exception as e:
            logger.error(f"Failed to initialize thermal camera: {e}")
            self.camera = None
    
    def read(self):
        """Capture a single thermal frame (32x24 temperature array)"""
        if not self.camera:
            return None
        
        with self.lock:
            try:
                frame = np.zeros((24 * 32,))
                self.camera.getFrame(frame)
                return frame.reshape((24, 32))
            except Exception as e:
                logger.error(f"Thermal capture failed: {e}")
                return None
    
    def close(self):
        """Release thermal camera resources"""
        pass  # MLX90640 doesn't require explicit cleanup


class DataTransmitter:
    """Handles network transmission of camera data to computer"""
    
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
    
    def connect(self):
        """Establish connection to computer"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            logger.info(f"Connected to {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.socket = None
            return False
    
    def send_data(self, rgb_frame, thermal_frame):
        """Send RGB and thermal data to computer"""
        if not self.socket:
            return False
        
        try:
            # Prepare data package
            data = {
                'timestamp': time.time(),
                'has_rgb': rgb_frame is not None,
                'has_thermal': thermal_frame is not None
            }
            
            # Serialize RGB frame
            if rgb_frame is not None:
                rgb_bytes = rgb_frame.tobytes()
                data['rgb_shape'] = rgb_frame.shape
                data['rgb_dtype'] = str(rgb_frame.dtype)
            else:
                rgb_bytes = b''
            
            # Serialize thermal frame
            if thermal_frame is not None:
                thermal_bytes = thermal_frame.tobytes()
                data['thermal_shape'] = thermal_frame.shape
                data['thermal_dtype'] = str(thermal_frame.dtype)
            else:
                thermal_bytes = b''
            
            # Create packet: [header_size][header_json][rgb_data][thermal_data]
            header = json.dumps(data).encode('utf-8')
            header_size = struct.pack('!I', len(header))
            
            packet = header_size + header + rgb_bytes + thermal_bytes
            
            # Send packet size first, then packet
            packet_size = struct.pack('!I', len(packet))
            self.socket.sendall(packet_size + packet)
            
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            self.socket = None
            return False
    
    def close(self):
        """Close network connection"""
        if self.socket:
            self.socket.close()
            self.socket = None


class WildfireClient:
    """Main client application for Raspberry Pi"""
    
    def __init__(self, server_host='192.168.1.100', server_port=5555):
        self.rgb_camera = RGBCamera()
        self.thermal_camera = ThermalCamera()
        self.transmitter = DataTransmitter(server_host, server_port)
        self.running = False
    
    def run(self):
        """Main capture and transmission loop"""
        self.running = True
        
        # Connect to server
        while self.running and not self.transmitter.connect():
            logger.info("Retrying connection in 5 seconds...")
            time.sleep(5)
        
        logger.info("Starting capture loop")
        
        while self.running:
            try:
                # Capture frames
                rgb_frame = self.rgb_camera.read()
                thermal_frame = self.thermal_camera.read()
                
                # Send to computer
                if not self.transmitter.send_data(rgb_frame, thermal_frame):
                    logger.warning("Transmission failed, reconnecting...")
                    if not self.transmitter.connect():
                        time.sleep(5)
                        continue
                
                # time.sleep(0.033)  # ~30 fps max transmission rate
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.rgb_camera.close()
        self.thermal_camera.close()
        self.transmitter.close()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    # Configure your computer's IP address and port here
    SERVER_HOST = '192.168.1.165'  # Change to your computer's IP
    SERVER_PORT = 5555
    
    client = WildfireClient(SERVER_HOST, SERVER_PORT)
    
    try:
        client.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        client.cleanup()