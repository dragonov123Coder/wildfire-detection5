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

# GPS modules
import serial
import pynmea2

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


# Load configuration from file
def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise


CONFIG = load_config()

class GPS:
    def __init__(self, config):
        try:
            self.port = config["client"]["gps"]["port"]
            self.baudrate = config["client"]["gps"]["baudrate"]
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        except Exception as e:
            print(f"An unexpected exeption occured while trying to setup GPS: {e}")
    
    def read(self):
        try:
            line = self.ser.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                # Ensure we have a valid fix before trying to access attributes
                if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                    return {
                        'lat': msg.latitude,
                        'lon': msg.longitude,
                        'alt': getattr(msg, 'altitude', None),
                        'timestamp': str(getattr(msg, 'timestamp', '')),
                        'satellites': getattr(msg, 'num_sats', None)
                    }
        except Exception as e:
            logger.error(f"GPS Parse Error: {e}")
        return

class RGBCamera:
    """Handles RGB image capture from Raspberry Pi Camera Module v2"""
    
    def __init__(self, config):
        cam_config = config['client']['rgb_camera']
        self.width = cam_config['width']
        self.height = cam_config['height']
        self.picamera2_framerate = cam_config['picamera2_framerate']
        self.opencv_framerate = cam_config['opencv_framerate']
        self.warmup_time = cam_config['warmup_time_seconds']
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
                    controls={"FrameRate": self.picamera2_framerate},
                )
                
                self.camera.configure(config)
                self.camera.start()
                time.sleep(self.warmup_time)  # Allow camera to warm up
                logger.info("Initialized picamera2")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.opencv_framerate)  # Request fps
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
    
    def __init__(self, config):
        thermal_config = config['client']['thermal_camera']
        self.i2c_frequency = thermal_config['i2c_frequency']
        self.initial_refresh_rate = thermal_config['refresh_rate_hz']
        self.current_refresh_rate = self.initial_refresh_rate
        self.camera = None
        self.lock = Lock()
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize MLX90640 thermal camera"""
        if not THERMAL_AVAILABLE:
            logger.warning("Thermal camera not available")
            return
        
        logger.info("Warming up thermal camera...")
        time.sleep(1.0)
        
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=self.i2c_frequency)
            self.camera = adafruit_mlx90640.MLX90640(i2c)
            self._set_refresh_rate(self.current_refresh_rate)
            logger.info(f"Initialized MLX90640 thermal camera at {self.current_refresh_rate}Hz")
        except Exception as e:
            logger.error(f"Failed to initialize thermal camera: {e}")
            self.camera = None
    
    def _set_refresh_rate(self, hz):
        """Set the thermal camera refresh rate based on Hz value"""
        if hz == 16:
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        elif hz == 8:
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        elif hz == 4:
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        elif hz == 2:
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        elif hz == 1:
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ
        else:
            logger.warning(f"Unsupported refresh rate {hz}Hz, using 16Hz")
            self.camera.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
            self.current_refresh_rate = 16
    
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
                error_msg = str(e)
                
                # Check for "Too many retries" error and reduce refresh rate
                if "Too many retries" in error_msg:
                    new_refresh_rate = max(1, self.current_refresh_rate // 2)
                    if new_refresh_rate != self.current_refresh_rate:
                        logger.warning(f"Thermal capture failed: Too many retries. Reducing refresh rate from {self.current_refresh_rate}Hz to {new_refresh_rate}Hz")
                        self.current_refresh_rate = new_refresh_rate
                        try:
                            self._set_refresh_rate(new_refresh_rate)
                            logger.info(f"Refresh rate updated to {new_refresh_rate}Hz")
                        except Exception as rate_error:
                            logger.error(f"Failed to update refresh rate: {rate_error}")
                    else:
                        logger.error(f"Thermal capture failed: {error_msg} (already at minimum refresh rate)")
                else:
                    logger.error(f"Thermal capture failed: {error_msg}")
                
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
    
    def send_data(self, rgb_frame, thermal_frame, gps_data):
        """Send RGB and thermal data to computer"""
        if not self.socket:
            return False
        
        try:
            # Prepare data package
            data = {
                'timestamp': time.time(),
                'has_rgb': rgb_frame is not None,
                'has_thermal': thermal_frame is not None,
                'gps_data': gps_data,
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
            
            logger.info("Data transferred sucsessfullly.")
            
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
    
    def __init__(self, config):
        self.config = config
        net_config = config['client']['network']
        self.server_host = net_config['server_host']
        self.server_port = net_config['server_port']
        self.retry_interval = net_config['retry_interval_seconds']
        
        self.rgb_camera = RGBCamera(config)
        self.thermal_camera = ThermalCamera(config)
        self.gps = GPS(config)
        self.transmitter = DataTransmitter(self.server_host, self.server_port)
        self.running = False
    
    def run(self):
        """Main capture and transmission loop"""
        self.running = True
        
        # Connect to server
        while self.running and not self.transmitter.connect():
            logger.info(f"Retrying connection in {self.retry_interval} seconds...")
            time.sleep(self.retry_interval)
        
        logger.info("Starting capture loop")
        
        while self.running:            
            try:
                # Capture frames
                rgb_frame = self.rgb_camera.read()
                thermal_frame = self.thermal_camera.read()
                
                try:
                    # Read GPS data
                    gps_data = self.gps.read()
                except Exception:
                    # logger.info(f"Unable to read GPS data: {e}")
                    pass
                    
                # Send to computer
                if not self.transmitter.send_data(rgb_frame, thermal_frame, gps_data):
                    logger.warning("Transmission failed, reconnecting...")
                    if not self.transmitter.connect():
                        time.sleep(self.retry_interval)
                        continue
                
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
    config = load_config()
    client = WildfireClient(config)
    
    try:
        client.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        client.cleanup()