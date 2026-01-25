#!/usr/bin/env python3
"""
Computer Wildfire Detection Server
Receives RGB and thermal data, performs detection, serves web interface
"""

import time
import json
import socket
import struct
import numpy as np
import cv2
from threading import Thread, Lock
from datetime import datetime
from pathlib import Path
import logging
from flask import Flask, render_template, jsonify, send_file
from ultralytics import YOLO
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalProcessor:
    """Processes thermal frames for fire detection"""
    
    # Fixed temperature range for visualization
    MIN_TEMP = 15.0
    MAX_TEMP = 40.0
    
    # Fire detection thresholds
    FIRE_TEMP_THRESHOLD = 180.0  # Temperature in Celsius
    FIRE_AREA_THRESHOLD = 0.005  # 2% of frame must be hot
    
    # Dead pixel threshold
    DEAD_PIXEL_THRESHOLD = -75.0
    
    def __init__(self):
        self.lock = Lock()
        self.latest_frame = None
        self.latest_visualization = None
    
    def process(self, thermal_frame):
        """Process thermal frame and detect fire"""
        if thermal_frame is None:
            return None, 0.0
        
        with self.lock:
            # Clean dead pixels
            cleaned = self._clean_dead_pixels(thermal_frame)
            
            # Detect fire based on temperature
            fire_confidence = self._detect_fire(cleaned)
            
            # Create visualization with fixed colormap
            visualization = self._create_visualization(cleaned)
            
            self.latest_frame = cleaned
            self.latest_visualization = visualization
            
            return visualization, fire_confidence
    
    def _clean_dead_pixels(self, frame):
        """Replace dead pixels with frame average"""
        dead_mask = frame < self.DEAD_PIXEL_THRESHOLD
        if np.any(dead_mask):
            avg_temp = np.mean(frame[~dead_mask]) if np.any(~dead_mask) else 0
            frame = frame.copy()
            frame[dead_mask] = avg_temp
        return frame
    
    def _detect_fire(self, frame):
        """Detect fire based on high temperature regions"""
        hot_pixels = frame > self.FIRE_TEMP_THRESHOLD
        hot_pixel_count = np.sum(hot_pixels)
        total_pixels = frame.size
        hot_area_ratio = hot_pixel_count / total_pixels
        
        if hot_area_ratio >= self.FIRE_AREA_THRESHOLD:
            # Calculate confidence based on how much area is hot
            confidence = min(1.0, hot_area_ratio / (self.FIRE_AREA_THRESHOLD * 5))
            # print(f"\n\n\n\n\n\n\n\nAHHHHHHHHHHHHHHHHHHHHHHHHHHHH - Confidence: {confidence}\nHot Area Ratio: {hot_area_ratio}\nHot pixel count: {hot_pixel_count}\n\n")
            return confidence
        
        return 0.0
    
    def _create_visualization(self, frame):
        """Create colormap visualization with fixed temperature range"""
        # Normalize to 0-255 using fixed range
        normalized = np.clip(frame, self.MIN_TEMP, self.MAX_TEMP)
        normalized = ((normalized - self.MIN_TEMP) / (self.MAX_TEMP - self.MIN_TEMP) * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Upscale to 640x480 for better viewing
        upscaled = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        return upscaled


class RGBProcessor:
    """Processes RGB frames using YOLO for fire and smoke detection"""
    
    FIRE_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, model_path='model.pt'):
        self.lock = Lock()
        self.model = None
        self.latest_frame = None
        self.latest_annotated = None
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load YOLO model"""
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def process(self, rgb_frame):
        """Process RGB frame and detect fire/smoke"""
        if rgb_frame is None or self.model is None:
            return None, 0.0, 0.0
        
        with self.lock:
            try:
                # Run YOLO detection
                results = self.model(rgb_frame, verbose=False)
                
                # Extract detections
                fire_conf, smoke_conf = self._extract_detections(results[0])
                
                # Create annotated visualization
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Upscale to match thermal camera output size
                upscaled = cv2.resize(annotated, (640, 480), interpolation=cv2.INTER_LINEAR)
                
                self.latest_frame = rgb_frame
                self.latest_annotated = upscaled
                
                return upscaled, fire_conf, smoke_conf
                
            except Exception as e:
                logger.error(f"RGB processing failed: {e}")
                return None, 0.0, 0.0
    
    def _extract_detections(self, result):
        """Extract fire and smoke confidences from YOLO results"""
        fire_conf = 0.0
        smoke_conf = 0.0
        
        if result.boxes is None or len(result.boxes) == 0:
            return fire_conf, smoke_conf
        
        # Iterate through detections
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls].lower()
            
            if 'fire' in name and conf >= self.FIRE_CONFIDENCE_THRESHOLD:
                fire_conf = max(fire_conf, conf)
            elif 'smoke' in name:
                smoke_conf = max(smoke_conf, conf)
        
        return fire_conf, smoke_conf


class ConfidenceFusion:
    """Fuses detection confidences from multiple sources"""
    
    # Weights for confidence fusion (must sum to 1.0)
    THERMAL_WEIGHT = 0.50
    RGB_FIRE_WEIGHT = 0.40
    RGB_SMOKE_WEIGHT = 0.10
    
    # Final threshold for fire detection
    FIRE_DETECTION_THRESHOLD = 0.4
    
    def __init__(self):
        self.lock = Lock()
        self.latest_confidence = 0.0
        self.latest_breakdown = {}
    
    def fuse(self, thermal_conf, rgb_fire_conf, rgb_smoke_conf):
        """Compute fused fire confidence score"""
        with self.lock:
            thermal_contrib = float(thermal_conf) * self.THERMAL_WEIGHT
            fire_contrib = float(rgb_fire_conf) * self.RGB_FIRE_WEIGHT
            smoke_contrib = float(rgb_smoke_conf) * self.RGB_SMOKE_WEIGHT
            
            final_confidence = thermal_contrib + fire_contrib + smoke_contrib
            
            self.latest_breakdown = {
                'thermal': thermal_contrib,
                'rgb_fire': fire_contrib,
                'rgb_smoke': smoke_contrib,
                'total': final_confidence
            }
            
            self.latest_confidence = final_confidence
            
            return final_confidence
    
    def is_fire_detected(self):
        """Check if fire is detected based on threshold"""
        return self.latest_confidence >= self.FIRE_DETECTION_THRESHOLD
    
    def get_breakdown(self):
        """Get confidence breakdown for monitoring"""
        with self.lock:
            return {
                'thermal': float(self.latest_breakdown.get('thermal', 0.0)),
                'rgb_fire': float(self.latest_breakdown.get('rgb_fire', 0.0)),
                'rgb_smoke': float(self.latest_breakdown.get('rgb_smoke', 0.0)),
                'total': float(self.latest_breakdown.get('total', 0.0))
            }


class DataReceiver:
    """Receives camera data from Raspberry Pi"""
    
    def __init__(self, port=5555):
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        self.lock = Lock()
        self.latest_rgb = None
        self.latest_thermal = None
    
    def start(self):
        """Start listening for connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(1)
        self.running = True
        
        logger.info(f"Listening on port {self.port}")
        
        thread = Thread(target=self._receive_loop, daemon=True)
        thread.start()
    
    def _receive_loop(self):
        """Main loop for receiving data"""
        while self.running:
            try:
                logger.info("Waiting for Raspberry Pi connection...")
                self.client_socket, addr = self.socket.accept()
                logger.info(f"Connected to {addr}")
                
                while self.running:
                    data = self._receive_packet()
                    if data is None:
                        break
                    
                    rgb_frame, thermal_frame = data
                    
                    with self.lock:
                        self.latest_rgb = rgb_frame
                        self.latest_thermal = thermal_frame
                
            except Exception as e:
                if self.running:
                    logger.error(f"Receive error: {e}")
                time.sleep(1)
    
    def _receive_packet(self):
        """Receive a single data packet"""
        try:
            # Receive packet size
            size_data = self._recv_exact(4)
            if not size_data:
                return None
            packet_size = struct.unpack('!I', size_data)[0]
            
            # Receive packet
            packet = self._recv_exact(packet_size)
            if not packet:
                return None
            
            # Parse header
            header_size = struct.unpack('!I', packet[:4])[0]
            header_json = packet[4:4+header_size].decode('utf-8')
            header = json.loads(header_json)
            
            offset = 4 + header_size
            
            # Parse RGB frame
            rgb_frame = None
            if header['has_rgb']:
                shape = tuple(header['rgb_shape'])
                dtype = np.dtype(header['rgb_dtype'])
                rgb_size = np.prod(shape) * dtype.itemsize
                rgb_bytes = packet[offset:offset+rgb_size]
                rgb_frame = np.frombuffer(rgb_bytes, dtype=dtype).reshape(shape)
                offset += rgb_size
            
            # Parse thermal frame
            thermal_frame = None
            if header['has_thermal']:
                shape = tuple(header['thermal_shape'])
                dtype = np.dtype(header['thermal_dtype'])
                thermal_size = np.prod(shape) * dtype.itemsize
                thermal_bytes = packet[offset:offset+thermal_size]
                thermal_frame = np.frombuffer(thermal_bytes, dtype=dtype).reshape(shape)
            
            return rgb_frame, thermal_frame
            
        except Exception as e:
            logger.error(f"Packet receive error: {e}")
            return None
    
    def _recv_exact(self, size):
        """Receive exact number of bytes"""
        data = b''
        while len(data) < size:
            chunk = self.client_socket.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def get_latest_frames(self):
        """Get the latest received frames"""
        with self.lock:
            return self.latest_rgb, self.latest_thermal
    
    def stop(self):
        """Stop receiving data"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()


class ImageSaver:
    """Saves images periodically and on fire detection"""
    
    def __init__(self, save_dir='saved_images'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.last_periodic_save = 0
        self.periodic_interval = 5.0  # Save every 5 seconds
    
    def save_if_needed(self, rgb_img, thermal_img, is_fire):
        """Save images if conditions are met"""
        current_time = time.time()
        
        # Save on fire detection
        if is_fire:
            self._save_images(rgb_img, thermal_img, 'fire')
        
        # Save periodically
        elif current_time - self.last_periodic_save >= self.periodic_interval:
            self._save_images(rgb_img, thermal_img, 'periodic')
            self.last_periodic_save = current_time
    
    def _save_images(self, rgb_img, thermal_img, prefix):
        """Save RGB and thermal images"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        if rgb_img is not None:
            rgb_path = self.save_dir / f"{prefix}_rgb_{timestamp}.jpg"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        if thermal_img is not None:
            thermal_path = self.save_dir / f"{prefix}_thermal_{timestamp}.jpg"
            cv2.imwrite(str(thermal_path), thermal_img)
    
    def save_with_metadata(self, rgb_img, thermal_img, thermal_frame, is_fire):
        """Save images with thermal metadata"""
        current_time = time.time()
        
        # Save on fire detection
        if is_fire:
            self._save_with_temp_metadata(rgb_img, thermal_img, thermal_frame, 'fire')
        
        # Save periodically
        elif current_time - self.last_periodic_save >= self.periodic_interval:
            self._save_with_temp_metadata(rgb_img, thermal_img, thermal_frame, 'periodic')
            self.last_periodic_save = current_time
    
    def _save_with_temp_metadata(self, rgb_img, thermal_img, thermal_frame, prefix):
        """Save images with temperature metadata in filename"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Calculate temperature statistics from raw thermal data
        if thermal_frame is not None:
            min_temp = np.min(thermal_frame)
            max_temp = np.max(thermal_frame)
            avg_temp = np.mean(thermal_frame)
            
            # Create metadata string
            temp_metadata = f"min{min_temp:.1f}_max{max_temp:.1f}_avg{avg_temp:.1f}"
        else:
            temp_metadata = "no_temp_data"
        
        # Save RGB image
        if rgb_img is not None:
            rgb_path = self.save_dir / f"{prefix}_rgb_{timestamp}.jpg"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # Save thermal image with temperature metadata in filename
        if thermal_img is not None:
            thermal_path = self.save_dir / f"{prefix}_thermal_{temp_metadata}_{timestamp}.jpg"
            cv2.imwrite(str(thermal_path), thermal_img)


class WildfireServer:
    """Main server application for wildfire detection"""
    
    def __init__(self):
        self.receiver = DataReceiver()
        self.thermal_processor = ThermalProcessor()
        self.rgb_processor = RGBProcessor()
        self.fusion = ConfidenceFusion()
        self.saver = ImageSaver()
        
        self.lock = Lock()
        self.latest_rgb_viz = None
        self.latest_thermal_viz = None
        self.latest_thermal_raw = None  # Store raw thermal data
        self.fire_detected = False
        self.fire_count = 0
        
        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/images')
        def get_images():
            """Return latest RGB and thermal images as base64"""
            with self.lock:
                rgb_b64 = self._encode_image(self.latest_rgb_viz, is_bgr=False)
                thermal_b64 = self._encode_image(self.latest_thermal_viz, is_bgr=True)
                
                # Calculate temperature statistics (excluding dead pixels)
                temp_stats = None
                if self.latest_thermal_raw is not None:
                    # Filter out dead pixels
                    valid_temps = self.latest_thermal_raw[
                        self.latest_thermal_raw > ThermalProcessor.DEAD_PIXEL_THRESHOLD
                    ]
                    
                    if len(valid_temps) > 0:
                        temp_stats = {
                            'min': float(np.min(valid_temps)),
                            'max': float(np.max(valid_temps)),
                            'avg': float(np.mean(valid_temps))
                        }
                
                return jsonify({
                    'rgb': rgb_b64,
                    'thermal': thermal_b64,
                    'fire_detected': self.fire_detected,
                    'confidence': self.fusion.latest_confidence,
                    'breakdown': self.fusion.get_breakdown(),
                    'fire_count': self.fire_count,
                    'temperature': temp_stats
                })
    
    def _encode_image(self, img, is_bgr=False):
        """Encode image to base64 string"""
        if img is None:
            return None
        # Only convert RGB to BGR if image is not already in BGR format
        if not is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def start(self):
        """Start the server"""
        # Start data receiver
        self.receiver.start()
        
        # Start processing loop
        process_thread = Thread(target=self._process_loop, daemon=True)
        process_thread.start()
        
        # Start Flask web server
        logger.info("Starting web server on http://0.0.0.0:8080")
        self.app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    
    def _process_loop(self):
        """Main processing loop"""
        while True:
            try:
                # Get latest frames
                rgb_frame, thermal_frame = self.receiver.get_latest_frames()
                
                # Process frames
                thermal_viz, thermal_conf = self.thermal_processor.process(thermal_frame)
                rgb_viz, fire_conf, smoke_conf = self.rgb_processor.process(rgb_frame)
                
                # Fuse confidences
                final_conf = self.fusion.fuse(thermal_conf, fire_conf, smoke_conf)
                is_fire = self.fusion.is_fire_detected()
                
                # Update state
                with self.lock:
                    self.latest_rgb_viz = rgb_viz
                    self.latest_thermal_viz = thermal_viz
                    self.latest_thermal_raw = thermal_frame  # Store raw data
                    if is_fire and not self.fire_detected:
                        self.fire_count += 1
                    self.fire_detected = is_fire
                
                # Save images with metadata
                self.saver.save_with_metadata(rgb_viz, thermal_viz, thermal_frame, is_fire)
                
                # Log fire detection
                if is_fire:
                    logger.warning(f"FIRE DETECTED! Confidence: {final_conf:.2f}")
                
                time.sleep(0.033)  # ~30 fps processing rate
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(1)


if __name__ == "__main__":
    server = WildfireServer()
    server.start()