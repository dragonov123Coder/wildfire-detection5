#!/usr/bin/env python3
"""
Computer Wildfire Detection Server
Receives RGB and thermal data from one or more Raspberry Pi units, performs
detection, serves the web interface (unit manager, per-unit dashboards, and
the fire-risk prediction map).
"""

import time
import json
import socket
import struct
import subprocess
import csv
import shutil
import numpy as np
import cv2
from threading import Thread, Lock
from datetime import datetime
from pathlib import Path
import logging
from flask import Flask, render_template, jsonify, send_file, request, redirect
from ultralytics import YOLO
import base64

import db
import scheduler as scheduler_mod

# Configure logging with file handler
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'wildfire_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file), # Into the log file
        logging.StreamHandler() # Console
    ]
)
logger = logging.getLogger(__name__)

UNIT_OFFLINE_TIMEOUT_SECONDS = 5


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
db.init_db()

# Paths for the fire-risk prediction sub-app (prediction-2/), resolved from
# this file's own location so they don't depend on the server's cwd.
BASE_DIR = Path(__file__).resolve().parent
PREDICTION_ROOT = BASE_DIR / 'prediction-2'
PREDICTION_PYTHON = PREDICTION_ROOT / 'venv' / 'Scripts' / 'python.exe'
PREDICTION_SCRIPT = PREDICTION_ROOT / 'scripts' / 'predict.py'
PREDICTION_OUTPUT_DIR = PREDICTION_ROOT / 'data' / 'output'
PREDICTION_RISK_CSV = PREDICTION_OUTPUT_DIR / 'risk_map.csv'
PREDICTION_HTML = PREDICTION_ROOT / 'output' / 'risk_map.html'


class UnitState:
    """Per-unit mutable state for a connected (or previously-connected)
    detection unit. Heavy processors (YOLO model, thermal colormap) stay
    shared singletons on WildfireServer; only the frames/results are
    namespaced per unit here."""

    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.lock = Lock()
        self.latest_rgb = None
        self.latest_thermal = None
        self.latest_gps_data = None
        self.latest_rgb_viz = None
        self.latest_thermal_viz = None
        self.latest_thermal_raw = None
        self.fire_detected = False
        self.fire_count = 0
        self.confidence = 0.0
        self.breakdown = {}
        self.connected = False
        self.last_seen_monotonic = 0.0
        self.last_periodic_save = 0.0


class ThermalProcessor:
    """Processes thermal frames for fire detection"""

    def __init__(self, config):
        self.config = config['server']['thermal_processor']

        # Fixed temperature range for visualization
        self.MIN_TEMP = self.config['min_temp_celsius']
        self.MAX_TEMP = self.config['max_temp_celsius']

        # Fire detection thresholds
        self.FIRE_TEMP_THRESHOLD = self.config['fire_temp_threshold_celsius']
        self.FIRE_AREA_THRESHOLD = self.config['fire_area_threshold']

        # Dead pixel threshold
        self.DEAD_PIXEL_THRESHOLD = self.config['dead_pixel_threshold']

        self.lock = Lock()

        logger.info(f"ThermalProcessor initialized - Temp range: {self.MIN_TEMP}-{self.MAX_TEMP}°C, Fire threshold: {self.FIRE_TEMP_THRESHOLD}°C, Area threshold: {self.FIRE_AREA_THRESHOLD}")

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
    """Processes RGB frames using YOLO for fire and smoke detection. One
    shared model instance across all units; self.lock serializes concurrent
    inference calls (a documented scaling ceiling at higher unit counts)."""

    def __init__(self, config):
        self.config = config['server']['rgb_processor']
        self.FIRE_CONFIDENCE_THRESHOLD = self.config['fire_confidence_threshold']

        self.lock = Lock()
        self.model = None
        self._load_model(self.config['model_path'])
        logger.info(f"RGBProcessor initialized - Fire confidence threshold: {self.FIRE_CONFIDENCE_THRESHOLD}")

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
    """Fuses detection confidences from multiple sources. Stateless: fuse()
    returns the result rather than mutating shared fields, since multiple
    units are processed concurrently through the same instance."""

    def __init__(self, config):
        self.config = config['server']['confidence_fusion']

        # Weights for confidence fusion (must sum to 1.0)
        self.THERMAL_WEIGHT = self.config['thermal_weight']
        self.RGB_FIRE_WEIGHT = self.config['rgb_fire_weight']
        self.RGB_SMOKE_WEIGHT = self.config['rgb_smoke_weight']

        # Final threshold for fire detection
        self.FIRE_DETECTION_THRESHOLD = self.config['fire_detection_threshold']

        # Component alert thresholds
        self.THERMAL_ALERT_THRESHOLD = self.config['thermal_alert_threshold']
        self.RGB_FIRE_ALERT_THRESHOLD = self.config['rgb_fire_alert_threshold']
        self.RGB_SMOKE_ALERT_THRESHOLD = self.config['rgb_smoke_alert_threshold']

        logger.info(f"ConfidenceFusion initialized - Weights: Thermal={self.THERMAL_WEIGHT}, Fire={self.RGB_FIRE_WEIGHT}, Smoke={self.RGB_SMOKE_WEIGHT}, Threshold={self.FIRE_DETECTION_THRESHOLD}")

    def fuse(self, thermal_conf, rgb_fire_conf, rgb_smoke_conf):
        """Compute fused fire confidence score. Returns (confidence, breakdown)."""
        thermal_contrib = float(thermal_conf) * self.THERMAL_WEIGHT
        fire_contrib = float(rgb_fire_conf) * self.RGB_FIRE_WEIGHT
        smoke_contrib = float(rgb_smoke_conf) * self.RGB_SMOKE_WEIGHT

        final_confidence = thermal_contrib + fire_contrib + smoke_contrib

        breakdown = {
            'thermal': thermal_contrib,
            'rgb_fire': fire_contrib,
            'rgb_smoke': smoke_contrib,
            'total': final_confidence
        }

        return final_confidence, breakdown


class DataReceiver:
    """Receives camera data from one or more Raspberry Pi units over TCP.
    Thread-per-connection: one accept loop spawns a handler thread per
    connected Pi, each tagged by the unit_id it sends in its packet header."""

    def __init__(self, config):
        self.port = config['server']['network']['port']
        self.socket = None
        self.running = False
        self.units_lock = Lock()
        self.units = {}  # unit_id -> UnitState

    def start(self):
        """Start listening for connections"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(5)
        self.running = True

        logger.info(f"Listening on port {self.port}")

        thread = Thread(target=self._accept_loop, daemon=True)
        thread.start()

    def _accept_loop(self):
        """Accept incoming connections and spawn a handler thread for each"""
        while self.running:
            try:
                logger.info("Waiting for Raspberry Pi connections...")
                client_socket, addr = self.socket.accept()
                Thread(target=self._handle_connection, args=(client_socket, addr), daemon=True).start()
            except OSError:
                if self.running:
                    logger.error("Accept loop socket error")
                break
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
                time.sleep(1)

    def _handle_connection(self, client_socket, addr):
        """Handle one Pi's connection for its lifetime"""
        unit_id = None
        try:
            first = self._receive_packet(client_socket)
            if first is None:
                logger.warning(f"Connection from {addr} closed before sending data")
                return
            unit_id, rgb_frame, thermal_frame, gps_data = first
            if not unit_id:
                logger.warning(f"Rejecting connection from {addr}: no unit_id in packet")
                return

            if db.get_unit(unit_id) is None:
                logger.warning(f"Unknown unit_id {unit_id!r} from {addr} -- auto-registering as an unnamed unit")
                db.create_unit(name=f"Unnamed ({unit_id})", lat=None, lon=None, unit_id=unit_id)

            db.set_unit_status(unit_id, 'online')
            db.touch_unit_last_seen(unit_id)
            if gps_data:
                db.set_unit_location_if_unset(unit_id, gps_data.get('lat'), gps_data.get('lon'))

            state = self._get_or_create_state(unit_id)
            with state.lock:
                state.connected = True
                state.latest_rgb = rgb_frame
                state.latest_thermal = thermal_frame
                state.latest_gps_data = gps_data
                state.last_seen_monotonic = time.monotonic()

            logger.info(f"Unit {unit_id} connected from {addr}")

            while self.running:
                packet = self._receive_packet(client_socket)
                if packet is None:
                    logger.warning(f"Unit {unit_id}: packet receive failed, disconnecting...")
                    break
                _uid, rgb_frame, thermal_frame, gps_data = packet
                with state.lock:
                    state.latest_rgb = rgb_frame
                    state.latest_thermal = thermal_frame
                    if gps_data:
                        state.latest_gps_data = gps_data
                    state.last_seen_monotonic = time.monotonic()
                if gps_data:
                    db.set_unit_location_if_unset(unit_id, gps_data.get('lat'), gps_data.get('lon'))
                db.touch_unit_last_seen(unit_id)

        except Exception as e:
            logger.error(f"Connection error for {unit_id or addr}: {e}")
        finally:
            client_socket.close()
            if unit_id:
                state = self.units.get(unit_id)
                if state:
                    with state.lock:
                        state.connected = False
                db.set_unit_status(unit_id, 'offline')
                logger.info(f"Unit {unit_id} disconnected")

    def _get_or_create_state(self, unit_id):
        with self.units_lock:
            if unit_id not in self.units:
                self.units[unit_id] = UnitState(unit_id)
            return self.units[unit_id]

    def get_unit_state(self, unit_id):
        with self.units_lock:
            return self.units.get(unit_id)

    def get_unit_ids(self):
        with self.units_lock:
            return list(self.units.keys())

    def _receive_packet(self, sock):
        """Receive a single data packet from the given socket.
        Returns (unit_id, rgb_frame, thermal_frame, gps_data) or None."""
        try:
            # Receive packet size
            size_data = self._recv_exact(sock, 4)
            if not size_data:
                return None
            packet_size = struct.unpack('!I', size_data)[0]

            # Receive packet
            packet = self._recv_exact(sock, packet_size)
            if not packet:
                return None

            # Parse header
            header_size = struct.unpack('!I', packet[:4])[0]
            header_json = packet[4:4+header_size].decode('utf-8')
            header = json.loads(header_json)

            offset = 4 + header_size

            # Parse RGB frame
            rgb_frame = None
            if header.get('has_rgb'):
                shape = tuple(header['rgb_shape'])
                dtype = np.dtype(header['rgb_dtype'])
                rgb_size = int(np.prod(shape) * dtype.itemsize)
                rgb_bytes = packet[offset:offset+rgb_size]
                rgb_frame = np.frombuffer(rgb_bytes, dtype=dtype).reshape(shape)
                offset += rgb_size

            # Parse thermal frame
            thermal_frame = None
            if header.get('has_thermal'):
                shape = tuple(header['thermal_shape'])
                dtype = np.dtype(header['thermal_dtype'])
                thermal_size = int(np.prod(shape) * dtype.itemsize)
                thermal_bytes = packet[offset:offset+thermal_size]
                thermal_frame = np.frombuffer(thermal_bytes, dtype=dtype).reshape(shape)

            gps_data = header.get('gps_data')
            unit_id = header.get('unit_id')

            return unit_id, rgb_frame, thermal_frame, gps_data

        except Exception as e:
            logger.error(f"Packet receive error: {e}")
            return None

    @staticmethod
    def _recv_exact(sock, size):
        """Receive exact number of bytes from a socket"""
        data = b''
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def stop(self):
        """Stop receiving data"""
        self.running = False
        if self.socket:
            self.socket.close()


class ImageSaver:
    """Saves images periodically and on fire detection, namespaced per unit."""

    def __init__(self, config):
        saver_config = config['server']['image_saver']
        self.rgb_dir = Path(saver_config['rgb_directory'])
        self.thermal_dir = Path(saver_config['thermal_directory'])
        self.periodic_interval = saver_config['periodic_save_interval_seconds']
        logger.info(f"ImageSaver initialized - RGB base: {self.rgb_dir}, Thermal base: {self.thermal_dir}")

    def save_with_metadata(self, unit_id, state, rgb_img, thermal_img, thermal_frame, is_fire, gps_data):
        """Save images with thermal metadata, if fire detected or on a periodic interval"""
        current_time = time.time()

        if is_fire:
            self._save_with_all_metadata(unit_id, rgb_img, thermal_img, thermal_frame, 'fire', gps_data)
        elif current_time - state.last_periodic_save >= self.periodic_interval:
            self._save_with_all_metadata(unit_id, rgb_img, thermal_img, thermal_frame, 'periodic', gps_data)
            state.last_periodic_save = current_time

    def _save_with_all_metadata(self, unit_id, rgb_img, thermal_img, thermal_frame, prefix, gps_data):
        """Save images with temperature/GPS metadata in filename, under this unit's own subdirectory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        rgb_dir = self.rgb_dir / unit_id
        thermal_dir = self.thermal_dir / unit_id
        rgb_dir.mkdir(parents=True, exist_ok=True)
        thermal_dir.mkdir(parents=True, exist_ok=True)

        if thermal_frame is not None:
            min_temp = np.min(thermal_frame)
            max_temp = np.max(thermal_frame)
            avg_temp = np.mean(thermal_frame)
            temp_metadata = f"min{min_temp:.1f}_max{max_temp:.1f}_avg{avg_temp:.1f}"
        else:
            temp_metadata = "no_temp_data"

        gps_data = gps_data or {}
        lat = gps_data.get('lat', '-')
        lon = gps_data.get('lon', '-')

        if rgb_img is not None:
            rgb_path = rgb_dir / f"{prefix}_rgb_({lat},{lon})_{timestamp}.jpg"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved RGB image with metadata: {rgb_path}")

        if thermal_img is not None:
            thermal_path = thermal_dir / f"{prefix}_thermal_{temp_metadata}_{timestamp}.jpg"
            cv2.imwrite(str(thermal_path), thermal_img)
            logger.info(f"Saved thermal image with metadata: {thermal_path}")


class WildfireServer:
    """Main server application for wildfire detection"""

    def __init__(self, config):
        self.config = config
        self.receiver = DataReceiver(config)
        self.thermal_processor = ThermalProcessor(config)
        self.rgb_processor = RGBProcessor(config)
        self.fusion = ConfidenceFusion(config)
        self.saver = ImageSaver(config)

        self._seed_initial_unit()

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _seed_initial_unit(self):
        """If the unit registry is empty, seed the id the existing physical
        Pi's config.json already sends, so it keeps working without a
        manual deploy step through the UI."""
        if not db.list_units():
            unit_id = self.config['client'].get('unit_id', 'unit-001')
            db.create_unit(name='Unit 1', lat=None, lon=None, unit_id=unit_id)
            logger.info(f"Seeded initial unit {unit_id}")

    def _build_config_snippet(self, unit_id):
        """A ready-to-copy client config for a newly deployed unit's Pi:
        the working client defaults, with only unit_id overridden."""
        client_config = json.loads(json.dumps(self.config['client']))
        client_config['unit_id'] = unit_id
        return {'client': client_config}

    def _thresholds(self):
        return {
            'fire_detection': self.fusion.FIRE_DETECTION_THRESHOLD,
            'thermal_weight': self.fusion.THERMAL_WEIGHT,
            'rgb_fire_weight': self.fusion.RGB_FIRE_WEIGHT,
            'rgb_smoke_weight': self.fusion.RGB_SMOKE_WEIGHT
        }

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/monitor')
        def monitor():
            return redirect('/units')

        @self.app.route('/units')
        def units_page():
            return render_template('units.html')

        @self.app.route('/units/<unit_id>')
        def unit_dashboard(unit_id):
            unit = db.get_unit(unit_id)
            if unit is None:
                return render_template('unit_not_found.html', unit_id=unit_id), 404
            return render_template('unit_dashboard.html', unit=unit)

        @self.app.route('/api/units', methods=['GET'])
        def api_units_list():
            units = db.list_units()
            for u in units:
                state = self.receiver.get_unit_state(u['unit_id'])
                u['connected'] = bool(state and state.connected)
            return jsonify(units)

        @self.app.route('/api/units', methods=['POST'])
        def api_units_create():
            body = request.get_json(silent=True) or {}
            lat = body.get('lat')
            lon = body.get('lon')
            if lat is None or lon is None:
                return jsonify({'error': 'lat and lon are required'}), 400
            try:
                lat = float(lat)
                lon = float(lon)
            except (TypeError, ValueError):
                return jsonify({'error': 'lat and lon must be numbers'}), 400
            name = (body.get('name') or '').strip() or f"Unit at {lat:.3f}, {lon:.3f}"
            unit = db.create_unit(name=name, lat=lat, lon=lon)
            unit['config_snippet'] = self._build_config_snippet(unit['unit_id'])
            return jsonify(unit), 201

        @self.app.route('/api/units/<unit_id>')
        def api_units_detail(unit_id):
            unit = db.get_unit(unit_id)
            if unit is None:
                return jsonify({'error': 'not found'}), 404
            state = self.receiver.get_unit_state(unit_id)
            unit['connected'] = bool(state and state.connected)
            unit['config_snippet'] = self._build_config_snippet(unit_id)
            return jsonify(unit)

        @self.app.route('/api/units/<unit_id>/images')
        def api_unit_images(unit_id):
            unit = db.get_unit(unit_id)
            if unit is None:
                return jsonify({'error': 'not found'}), 404

            state = self.receiver.get_unit_state(unit_id)
            if state is None:
                return jsonify({
                    'connected': False,
                    'status': unit['status'],
                    'rgb': None,
                    'thermal': None,
                    'fire_detected': False,
                    'confidence': 0.0,
                    'breakdown': {},
                    'fire_count': 0,
                    'temperature': None,
                    'thresholds': self._thresholds(),
                    'gps_data': None,
                })

            with state.lock:
                rgb_b64 = self._encode_image(state.latest_rgb_viz, is_bgr=False)
                thermal_b64 = self._encode_image(state.latest_thermal_viz, is_bgr=True)

                temp_stats = None
                if state.latest_thermal_raw is not None:
                    dead_pixel_threshold = self.config['server']['thermal_processor']['dead_pixel_threshold']
                    valid_temps = state.latest_thermal_raw[
                        state.latest_thermal_raw > dead_pixel_threshold
                    ]
                    if len(valid_temps) > 0:
                        temp_stats = {
                            'min': float(np.min(valid_temps)),
                            'max': float(np.max(valid_temps)),
                            'avg': float(np.mean(valid_temps))
                        }

                return jsonify({
                    'connected': state.connected,
                    'status': unit['status'],
                    'rgb': rgb_b64,
                    'thermal': thermal_b64,
                    'fire_detected': state.fire_detected,
                    'confidence': state.confidence,
                    'breakdown': state.breakdown,
                    'fire_count': state.fire_count,
                    'temperature': temp_stats,
                    'thresholds': self._thresholds(),
                    'gps_data': state.latest_gps_data
                })

        @self.app.route('/prediction')
        def prediction():
            return send_file(str(PREDICTION_HTML))

        @self.app.route('/api/prediction/dates')
        def prediction_dates():
            """List this morning's (or the bootstrap week's) available dated CSVs."""
            return jsonify(db.latest_prediction_runs())

        @self.app.route('/api/prediction/data')
        def prediction_data():
            """Serve a risk_map CSV. ?date=YYYY-MM-DD picks a specific
            pre-generated target date; omitted falls back to the legacy
            fixed path for backward compatibility."""
            date_str = request.args.get('date')
            if date_str:
                csv_path = db.csv_path_for_target(date_str)
                if not csv_path or not Path(csv_path).exists():
                    return jsonify({'error': f'no risk map generated for {date_str}'}), 404
                return send_file(str(Path(csv_path)), mimetype='text/csv')
            if not PREDICTION_RISK_CSV.exists():
                return jsonify({'error': 'no risk map generated yet'}), 404
            return send_file(str(PREDICTION_RISK_CSV), mimetype='text/csv')

        @self.app.route('/api/prediction/regenerate', methods=['POST'])
        def prediction_regenerate():
            """Rerun prediction-2/scripts/predict.py for a given date and
            report back once it finishes. Runs synchronously (Flask's dev
            server is threaded, so this only blocks the calling request).
            Manual on-demand fallback to the automated daily scheduler."""
            body = request.get_json(silent=True) or {}
            date_str = body.get('date', '')
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({'ok': False, 'error': 'date must be in YYYY-MM-DD format'}), 400

            logger.info(f"Regenerating fire risk prediction for {date_str}")
            run_date_str = datetime.now().strftime('%Y-%m-%d')
            out_path = PREDICTION_OUTPUT_DIR / f'risk_map_{date_str}_run{run_date_str}.csv'
            try:
                result = subprocess.run(
                    [str(PREDICTION_PYTHON), str(PREDICTION_SCRIPT), '--date', date_str, '--out', str(out_path)],
                    cwd=str(PREDICTION_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
            except subprocess.TimeoutExpired:
                return jsonify({'ok': False, 'error': 'predict.py timed out after 15 minutes'}), 504

            if result.returncode != 0:
                err = result.stderr.strip()[-1500:] or 'predict.py failed with no error output'
                logger.error(f"predict.py failed for {date_str}: {err}")
                return jsonify({'ok': False, 'error': err}), 500

            row_count, max_raw = scheduler_mod.summarize_risk_csv(out_path)

            try:
                shutil.copyfile(out_path, PREDICTION_RISK_CSV)
            except OSError:
                pass

            db.record_prediction_run(run_date_str, date_str, str(out_path), max_raw)

            return jsonify({'ok': True, 'date': date_str, 'rows': row_count})

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
        logger.info("=" * 80)
        logger.info("WILDFIRE DETECTION SERVER STARTING")
        logger.info("=" * 80)

        # Start data receiver
        self.receiver.start()

        # Start processing loop
        process_thread = Thread(target=self._process_loop, daemon=True)
        process_thread.start()

        # Start the daily prediction scheduler
        self.prediction_scheduler = scheduler_mod.init_scheduler()

        # Start Flask web server
        flask_host = self.config['server']['network']['flask_host']
        flask_port = self.config['server']['network']['flask_port']
        logger.info(f"Starting web server on http://{flask_host}:{flask_port}")
        self.app.run(host=flask_host, port=flask_port, debug=False, threaded=True)

    def _process_loop(self):
        """Main processing loop: runs shared detection processors over each
        connected unit's latest raw frames, once per tick."""
        loop_sleep = self.config['server']['processing']['loop_sleep_seconds']
        logger.info("Starting main processing loop")

        while True:
            try:
                for unit_id in self.receiver.get_unit_ids():
                    state = self.receiver.get_unit_state(unit_id)
                    if state is None:
                        continue

                    with state.lock:
                        rgb_frame = state.latest_rgb
                        thermal_frame = state.latest_thermal
                        gps_data = state.latest_gps_data
                        last_seen = state.last_seen_monotonic
                        was_connected = state.connected

                    # Staleness reaper: catches a dead connection that never
                    # closed cleanly (network drop, Pi crash).
                    if was_connected and time.monotonic() - last_seen > UNIT_OFFLINE_TIMEOUT_SECONDS:
                        with state.lock:
                            state.connected = False
                        db.set_unit_status(unit_id, 'offline')
                        logger.info(f"Unit {unit_id} timed out, marking offline")
                        continue

                    if rgb_frame is None and thermal_frame is None:
                        continue

                    # Flip images upside down (rotate 180 degrees vertically)
                    if rgb_frame is not None:
                        rgb_frame = cv2.flip(rgb_frame, 0)
                    if thermal_frame is not None:
                        thermal_frame = cv2.flip(thermal_frame, 0)

                    # Process frames through the shared processors
                    thermal_viz, thermal_conf = self.thermal_processor.process(thermal_frame)
                    rgb_viz, fire_conf, smoke_conf = self.rgb_processor.process(rgb_frame)

                    final_conf, breakdown = self.fusion.fuse(thermal_conf, fire_conf, smoke_conf)
                    is_fire = final_conf >= self.fusion.FIRE_DETECTION_THRESHOLD

                    if thermal_conf > 0:
                        status = "THERMAL ALERT" if thermal_conf >= self.fusion.THERMAL_ALERT_THRESHOLD else "thermal"
                        logger.info(f"[{unit_id}] {status}: {thermal_conf:.2f} (threshold: {self.fusion.THERMAL_ALERT_THRESHOLD})")
                    if fire_conf > 0:
                        status = "RGB FIRE ALERT" if fire_conf >= self.fusion.RGB_FIRE_ALERT_THRESHOLD else "rgb_fire"
                        logger.info(f"[{unit_id}] {status}: {fire_conf:.2f} (threshold: {self.fusion.RGB_FIRE_ALERT_THRESHOLD})")
                    if smoke_conf > 0:
                        status = "RGB SMOKE ALERT" if smoke_conf >= self.fusion.RGB_SMOKE_ALERT_THRESHOLD else "rgb_smoke"
                        logger.info(f"[{unit_id}] {status}: {smoke_conf:.2f} (threshold: {self.fusion.RGB_SMOKE_ALERT_THRESHOLD})")

                    if final_conf >= self.fusion.FIRE_DETECTION_THRESHOLD:
                        logger.warning(f"[{unit_id}] FIRE DETECTION THRESHOLD MET: Final confidence {final_conf:.2f} >= {self.fusion.FIRE_DETECTION_THRESHOLD}")

                    with state.lock:
                        state.latest_rgb_viz = rgb_viz
                        state.latest_thermal_viz = thermal_viz
                        state.latest_thermal_raw = thermal_frame
                        state.confidence = final_conf
                        state.breakdown = breakdown
                        if is_fire and not state.fire_detected:
                            state.fire_count += 1
                            logger.critical(f"[{unit_id}] FIRE ALERT #{state.fire_count} DETECTED! Final: {final_conf:.2f} | Thermal: {thermal_conf:.2f}, Fire: {fire_conf:.2f}, Smoke: {smoke_conf:.2f}")
                        state.fire_detected = is_fire

                    self.saver.save_with_metadata(unit_id, state, rgb_viz, thermal_viz, thermal_frame, is_fire, gps_data)

                    if is_fire:
                        logger.warning(f"[{unit_id}] FIRE DETECTED! Final: {final_conf:.2f} | Thermal: {thermal_conf:.2f}, Fire: {fire_conf:.2f}, Smoke: {smoke_conf:.2f}")

                time.sleep(loop_sleep)

            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
                time.sleep(1)


if __name__ == "__main__":
    config = load_config()
    server = WildfireServer(config)
    server.start()
