"""Mock Raspberry Pi client for testing the multi-unit server without real
hardware. Drives rpi_client.DataTransmitter directly (bypassing WildfireClient,
whose camera/GPS init would fail off-device), looping sample frames from
packet/rgb.jpg and packet/thermal.jpg at a fixed interval.

Usage:
    python scripts/mock_client.py --unit-id unit-001 --host localhost --port 5555
    python scripts/mock_client.py --unit-id unit-999-unknown --interval 0.5
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rpi_client import DataTransmitter  # noqa: E402

PACKET_DIR = ROOT / 'packet'


def load_frames():
    rgb = cv2.imread(str(PACKET_DIR / 'rgb.jpg'))
    if rgb is not None:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    thermal_img = cv2.imread(str(PACKET_DIR / 'thermal.jpg'), cv2.IMREAD_GRAYSCALE)
    if thermal_img is not None:
        # Fake a plausible 24x32 temperature array (MLX90640 shape) from the
        # sample grayscale image, scaled into a normal-looking Celsius range.
        small = cv2.resize(thermal_img, (32, 24), interpolation=cv2.INTER_AREA)
        thermal = 15.0 + (small.astype(np.float64) / 255.0) * 25.0
    else:
        thermal = None
    return rgb, thermal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit-id', required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--lat', type=float, default=55.0)
    parser.add_argument('--lon', type=float, default=-115.0)
    args = parser.parse_args()

    rgb, thermal = load_frames()
    if rgb is None and thermal is None:
        print(f"Could not load sample frames from {PACKET_DIR}")
        sys.exit(1)

    gps_data = {'lat': args.lat, 'lon': args.lon, 'alt': 780.0, 'timestamp': '', 'satellites': 8}

    transmitter = DataTransmitter(args.host, args.port, args.unit_id)
    print(f"[{args.unit_id}] connecting to {args.host}:{args.port}...")
    while not transmitter.connect():
        print(f"[{args.unit_id}] retrying...")
        time.sleep(2)
    print(f"[{args.unit_id}] connected, sending frames every {args.interval}s (Ctrl+C to stop)")

    try:
        while True:
            ok = transmitter.send_data(rgb, thermal, gps_data)
            if not ok:
                print(f"[{args.unit_id}] send failed, reconnecting...")
                if not transmitter.connect():
                    time.sleep(2)
                    continue
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"[{args.unit_id}] stopping")
    finally:
        transmitter.close()


if __name__ == '__main__':
    main()
