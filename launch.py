#!/usr/bin/env python3
"""Start the MuJoCo simulator and Pygame controller as separate processes."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Sequence

from ipc_protocol import (
    DEFAULT_COMMAND_PORT,
    DEFAULT_HOST,
    DEFAULT_TELEMETRY_PORT,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    # 先給子程序機會正常清理 socket/視窗；逾時才強制 kill。
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def run_processes(
    sim_python: str,
    headless: bool,
    host: str,
    command_port: int,
    telemetry_port: int,
    target_height: float,
    duration: float | None,
) -> int:
    # 以 argv list 啟動，不經 shell；路徑含空白時也不需要手動 quote。
    sim_command = shlex.split(sim_python) + [
        str(PROJECT_ROOT / "simulator.py"),
        "--host",
        host,
        "--command-port",
        str(command_port),
        "--telemetry-port",
        str(telemetry_port),
        "--target-height",
        str(target_height),
    ]
    if headless:
        sim_command.append("--headless")
    if duration is not None:
        sim_command.extend(("--duration", str(duration)))

    controller_command = [
        sys.executable,
        str(PROJECT_ROOT / "pygame_controller.py"),
        "--host",
        host,
        "--command-port",
        str(command_port),
        "--telemetry-port",
        str(telemetry_port),
    ]
    if duration is not None:
        controller_command.extend(("--duration", str(duration)))

    simulator: subprocess.Popen[bytes] | None = None
    controller: subprocess.Popen[bytes] | None = None
    try:
        # simulator 先綁定 command port；短暫等待可及早回報 XML/OpenGL 啟動失敗。
        print("Starting MuJoCo simulator:", shlex.join(sim_command), flush=True)
        simulator = subprocess.Popen(sim_command, cwd=PROJECT_ROOT)
        print(f"Simulator PID: {simulator.pid}", flush=True)
        time.sleep(0.35)
        if simulator.poll() is not None:
            return simulator.returncode or 1

        print("Starting Pygame controller:", shlex.join(controller_command), flush=True)
        controller = subprocess.Popen(controller_command, cwd=PROJECT_ROOT)
        print(f"Controller PID: {controller.pid}", flush=True)
        # 任一視窗結束就離開，finally 會回收另一個仍在執行的子程序。
        while True:
            simulator_code = simulator.poll()
            controller_code = controller.poll()
            if simulator_code is not None:
                return simulator_code
            if controller_code is not None:
                return controller_code
            time.sleep(0.10)
    except (KeyboardInterrupt, FileNotFoundError) as error:
        if isinstance(error, FileNotFoundError):
            print(f"failed to start process: {error}", file=sys.stderr)
            return 1
        return 0
    finally:
        stop_process(controller)
        stop_process(simulator)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch separate MuJoCo and Pygame controller processes."
    )
    parser.add_argument(
        "--sim-python",
        default=os.environ.get("MUJOCO_SIM_PYTHON", sys.executable),
        help="simulator Python command, for example mjpython on macOS",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--command-port", type=port_number, default=DEFAULT_COMMAND_PORT
    )
    parser.add_argument(
        "--telemetry-port", type=port_number, default=DEFAULT_TELEMETRY_PORT
    )
    parser.add_argument("--target-height", type=float, default=1.2)
    parser.add_argument("--duration", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be greater than zero")
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            parser.error("--host must be a loopback address")
    except ValueError as error:
        parser.error(str(error))
    return run_processes(
        args.sim_python,
        args.headless,
        args.host,
        args.command_port,
        args.telemetry_port,
        args.target_height,
        args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
