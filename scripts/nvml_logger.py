#!/usr/bin/env python3
"""
scripts/nvml_logger.py

Direct ctypes wrapper around nvml.dll on Windows (standard NVIDIA driver library).
Samples SM clock, memory clock, temperature, power, and clock throttle reasons at 10 Hz.
Writes samples to a CSV and computes summary statistics.
"""

import os
import time
import csv
import threading
import ctypes
from typing import Optional, Dict, Any, List

# NVML Constants
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0

THROTTLE_REASONS = {
    0x0000000000000001: "GPU_IDLE",
    0x0000000000000002: "APPLICATIONS_CLOCKS_SETTING",
    0x0000000000000004: "SW_POWER_CAP",
    0x0000000000000008: "HW_SLOWDOWN",
    0x0000000000000010: "SYNC_BOOST",
    0x0000000000000020: "SW_THERMAL_SLOWDOWN",
    0x0000000000000040: "HW_THERMAL_SLOWDOWN",
    0x0000000000000080: "HW_POWER_BRAKE_SLOWDOWN",
    0x0000000000000100: "DISPLAY_CLOCK_SETTING",
}

def decode_throttle_reasons(mask: int) -> str:
    reasons = []
    for bit, name in THROTTLE_REASONS.items():
        if mask & bit:
            reasons.append(name)
    return "|".join(reasons) if reasons else "NONE"

class NVMLLogger:
    def __init__(self, output_csv_path: str, interval_s: float = 0.10, device_index: int = 0):
        self.output_csv_path = output_csv_path
        self.interval_s = interval_s
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []

        # Load NVML
        try:
            self._nvml = ctypes.CDLL("nvml.dll")
        except Exception as e:
            raise RuntimeError(f"Could not load nvml.dll: {e}")

        res = self._nvml.nvmlInit_v2()
        if res != 0:
            raise RuntimeError(f"nvmlInit_v2 failed with error code: {res}")

        self._device = ctypes.c_void_p()
        res = self._nvml.nvmlDeviceGetHandleByIndex_v2(self.device_index, ctypes.byref(self._device))
        if res != 0:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 failed with error code: {res}")

    def _sample(self) -> Dict[str, Any]:
        sm_clock = ctypes.c_uint()
        mem_clock = ctypes.c_uint()
        temp = ctypes.c_uint()
        power = ctypes.c_uint()
        reasons = ctypes.c_ulonglong()

        self._nvml.nvmlDeviceGetClockInfo(self._device, NVML_CLOCK_SM, ctypes.byref(sm_clock))
        self._nvml.nvmlDeviceGetClockInfo(self._device, NVML_CLOCK_MEM, ctypes.byref(mem_clock))
        self._nvml.nvmlDeviceGetTemperature(self._device, NVML_TEMPERATURE_GPU, ctypes.byref(temp))
        self._nvml.nvmlDeviceGetPowerUsage(self._device, ctypes.byref(power))
        self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._device, ctypes.byref(reasons))

        return {
            "timestamp": time.time(),
            "sm_clock_mhz": sm_clock.value,
            "mem_clock_mhz": mem_clock.value,
            "temp_c": temp.value,
            "power_w": power.value / 1000.0,
            "throttle_mask": f"0x{reasons.value:x}",
            "throttle_reasons": decode_throttle_reasons(reasons.value),
        }

    def _run(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv_path)), exist_ok=True)
        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w", "throttle_mask", "throttle_reasons"])

            while not self._stop_event.is_set():
                sample = self._sample()
                self._samples.append(sample)
                writer.writerow([
                    f"{sample['timestamp']:.4f}",
                    sample["sm_clock_mhz"],
                    sample["mem_clock_mhz"],
                    sample["temp_c"],
                    f"{sample['power_w']:.2f}",
                    sample["throttle_mask"],
                    sample["throttle_reasons"]
                ])
                f.flush()
                self._stop_event.wait(self.interval_s)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

        if not self._samples:
            return {}

        import numpy as np
        sm_clocks = [s["sm_clock_mhz"] for s in self._samples]
        mem_clocks = [s["mem_clock_mhz"] for s in self._samples]
        temps = [s["temp_c"] for s in self._samples]
        powers = [s["power_w"] for s in self._samples]

        return {
            "num_samples": len(self._samples),
            "median_sm_clock_mhz": float(np.median(sm_clocks)),
            "min_sm_clock_mhz": int(np.min(sm_clocks)),
            "max_sm_clock_mhz": int(np.max(sm_clocks)),
            "median_mem_clock_mhz": float(np.median(mem_clocks)),
            "max_temp_c": int(np.max(temps)),
            "avg_power_w": float(np.mean(powers)),
            "csv_path": self.output_csv_path
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Log NVML metrics at 10 Hz.")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--out", type=str, default="data/nvml_test.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Logging NVML for {args.duration}s to {args.out}...")
    logger = NVMLLogger(args.out)
    logger.start()
    time.sleep(args.duration)
    summary = logger.stop()
    print("Summary:", summary)

