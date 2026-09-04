#!/usr/bin/env python3
"""Independent Pygame flight controller process using localhost UDP IPC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import math
import os
import socket
import sys
import time
from typing import Sequence

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from ipc_protocol import (  # noqa: E402
    CommandPacket,
    DEFAULT_COMMAND_PORT,
    DEFAULT_HOST,
    DEFAULT_TELEMETRY_PORT,
    MAX_PACKET_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    TelemetryPacket,
    decode_telemetry,
    encode_command,
)


WINDOW_SIZE = (1120, 800)
MAP_SCALE = 28.0
TELEMETRY_TIMEOUT = 0.5


@dataclass(frozen=True)
class ControlAxes:
    """高層 normalized 輸入；這裡沒有任何馬達推力或 MuJoCo state。"""

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0

    def any_active(self) -> bool:
        return any(
            abs(value) > 0.0
            for value in (self.forward, self.right, self.up, self.yaw)
        )


class FocusSafety:
    """Require focus and a full key release before enabling flight controls."""

    def __init__(self) -> None:
        self.controls_armed = False
        self.active = False

    def update(
        self, focused: bool, minimized: bool, movement_key_pressed: bool
    ) -> tuple[bool, bool]:
        # 失焦後即使重新點回視窗，也必須先放開所有移動鍵才能重新 armed；
        # 這能阻止「按住 W 切換視窗」留下卡住的前進命令。
        was_active = self.active
        if not focused or minimized:
            self.controls_armed = False
            self.active = False
        else:
            if not self.controls_armed and not movement_key_pressed:
                self.controls_armed = True
            self.active = self.controls_armed
        neutral_required = was_active and not self.active
        return self.active, neutral_required

    def force_lost(self) -> bool:
        was_active = self.active
        self.controls_armed = False
        self.active = False
        return was_active


# 尚未收到 simulator 資料時的安全顯示值，不會被送回或寫進模擬器。
EMPTY_TELEMETRY = TelemetryPacket(
    PROTOCOL_VERSION,
    0,
    0.0,
    "DISARMED",
    False,
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    0.0,
    (0.0, 0.0, 0.0, 0.0),
    999.0,
)


class ControllerLink:
    """Send high-level commands and receive telemetry without blocking UI."""

    def __init__(self, host: str, command_port: int, telemetry_port: int) -> None:
        address = ipaddress.ip_address(host)
        if not address.is_loopback:
            raise ValueError("IPC host must be a loopback address")
        self.command_address = (host, command_port)
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.telemetry_socket.bind((host, telemetry_port))
        self.telemetry_socket.setblocking(False)
        self.sequence = 0
        self.last_telemetry_sequence = -1
        self.last_telemetry_received: float | None = None
        self.telemetry = EMPTY_TELEMETRY
        self.malformed_packets = 0

    def send(self, axes: ControlAxes, action: str | None = None) -> None:
        packet = CommandPacket(
            PROTOCOL_VERSION,
            self.sequence,
            time.monotonic(),
            action,
            axes.forward,
            axes.right,
            axes.up,
            axes.yaw,
        )
        payload = encode_command(packet)
        # action 使用相同 sequence 重送三次以容忍 UDP 丟包；simulator 會依
        # sequence 去重，所以 TAKEOFF 等一次性操作仍只會執行一次。
        repetitions = 3 if action is not None else 1
        for _ in range(repetitions):
            self.command_socket.sendto(payload, self.command_address)
        self.sequence += 1

    def send_neutral(self) -> None:
        self.send(ControlAxes())

    def poll(self, now: float) -> TelemetryPacket:
        # 一次清空目前佇列中的新遙測，但設上限避免 UI frame 被網路資料拖住。
        for _ in range(64):
            try:
                payload, source = self.telemetry_socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                if not ipaddress.ip_address(source[0]).is_loopback:
                    continue
                if len(payload) > MAX_PACKET_BYTES:
                    raise ProtocolError("telemetry packet exceeds size limit")
                telemetry = decode_telemetry(payload)
                if telemetry.sequence <= self.last_telemetry_sequence:
                    continue
                self.last_telemetry_sequence = telemetry.sequence
                self.last_telemetry_received = now
                self.telemetry = telemetry
            except (ProtocolError, ValueError):
                self.malformed_packets += 1
        return self.telemetry

    def is_connected(self, now: float) -> bool:
        return (
            self.last_telemetry_received is not None
            and now - self.last_telemetry_received <= TELEMETRY_TIMEOUT
        )

    def close(self) -> None:
        self.command_socket.close()
        self.telemetry_socket.close()


@dataclass(frozen=True)
class ActionButton:
    action: str
    label: str
    rect: pygame.Rect
    color: tuple[int, int, int]
    enabled_states: frozenset[str]


class ControllerDashboard:
    """純顯示與 hit-testing；飛行合法性最後仍由 simulator 狀態機判定。"""

    BACKGROUND = (14, 20, 29)
    PANEL = (24, 34, 47)
    GRID = (48, 64, 80)
    TEXT = (225, 232, 240)
    MUTED = (145, 160, 177)
    CYAN = (62, 205, 230)
    YELLOW = (245, 190, 70)
    GREEN = (60, 215, 130)
    RED = (242, 78, 78)

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.title_font = pygame.font.Font(None, 34)
        self.body_font = pygame.font.Font(None, 23)
        self.small_font = pygame.font.Font(None, 19)
        self.map_rect = pygame.Rect(28, 112, 620, 650)
        self.buttons = (
            ActionButton(
                "arm",
                "ARM [M]",
                pygame.Rect(685, 555, 185, 42),
                (30, 132, 92),
                frozenset({"DISARMED", "LANDED"}),
            ),
            ActionButton(
                "disarm",
                "DISARM [N]",
                pygame.Rect(885, 555, 185, 42),
                (112, 95, 70),
                frozenset({"ARMED_IDLE", "LANDED"}),
            ),
            ActionButton(
                "takeoff",
                "TAKE OFF [T]",
                pygame.Rect(685, 607, 185, 42),
                (32, 125, 160),
                frozenset({"DISARMED", "ARMED_IDLE", "LANDED"}),
            ),
            ActionButton(
                "hover",
                "HOVER [H/SPACE]",
                pygame.Rect(885, 607, 185, 42),
                (55, 102, 154),
                frozenset({"HOVERING", "MANUAL", "LANDING"}),
            ),
            ActionButton(
                "land",
                "LAND [L]",
                pygame.Rect(685, 659, 185, 42),
                (177, 91, 42),
                frozenset({"TAKING_OFF", "HOVERING", "MANUAL"}),
            ),
            ActionButton(
                "reset",
                "RESET [BACKSPACE]",
                pygame.Rect(885, 659, 185, 42),
                (91, 89, 125),
                frozenset({"DISARMED", "LANDED", "EMERGENCY_STOP"}),
            ),
            ActionButton(
                "emergency_stop",
                "EMERGENCY STOP - CLICK TWICE",
                pygame.Rect(685, 711, 385, 45),
                (175, 42, 48),
                frozenset(
                    {
                        "ARMED_IDLE",
                        "TAKING_OFF",
                        "HOVERING",
                        "MANUAL",
                        "LANDING",
                    }
                ),
            ),
        )

    def action_at(
        self, position: tuple[int, int], state: str, connected: bool
    ) -> str | None:
        for button in self.buttons:
            if (
                connected
                and state in button.enabled_states
                and button.rect.collidepoint(position)
            ):
                return button.action
        return None

    def action_enabled(self, action: str, state: str, connected: bool) -> bool:
        return any(
            button.action == action
            and connected
            and state in button.enabled_states
            for button in self.buttons
        )

    def _text(
        self,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] | None = None,
        font: pygame.font.Font | None = None,
    ) -> None:
        surface = (font or self.body_font).render(text, True, color or self.TEXT)
        self.screen.blit(surface, position)

    def _world_to_map(self, x_position: float, y_position: float) -> tuple[int, int]:
        center_x, center_y = self.map_rect.center
        # 世界 +Y 朝上，但螢幕 pixel +Y 朝下，所以畫圖時要反轉 y。
        return (
            round(center_x + x_position * MAP_SCALE),
            round(center_y - y_position * MAP_SCALE),
        )

    def _draw_map(self, telemetry: TelemetryPacket) -> None:
        pygame.draw.rect(self.screen, self.PANEL, self.map_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.GRID, self.map_rect, 2, border_radius=10)
        center_x, center_y = self.map_rect.center
        for offset in range(-10, 11):
            pixel_x = round(center_x + offset * MAP_SCALE)
            pixel_y = round(center_y + offset * MAP_SCALE)
            pygame.draw.line(
                self.screen,
                self.GRID,
                (pixel_x, self.map_rect.top),
                (pixel_x, self.map_rect.bottom),
            )
            pygame.draw.line(
                self.screen,
                self.GRID,
                (self.map_rect.left, pixel_y),
                (self.map_rect.right, pixel_y),
            )
        pygame.draw.line(
            self.screen,
            (100, 120, 140),
            (self.map_rect.left, center_y),
            (self.map_rect.right, center_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            (100, 120, 140),
            (center_x, self.map_rect.top),
            (center_x, self.map_rect.bottom),
            2,
        )
        self._text("BODY-FORWARD INPUT", (42, 126), self.MUTED, self.small_font)

        target = self._world_to_map(
            telemetry.target_position[0], telemetry.target_position[1]
        )
        pygame.draw.circle(self.screen, self.YELLOW, target, 12, 2)
        pygame.draw.line(
            self.screen,
            self.YELLOW,
            (target[0] - 17, target[1]),
            (target[0] + 17, target[1]),
            2,
        )
        pygame.draw.line(
            self.screen,
            self.YELLOW,
            (target[0], target[1] - 17),
            (target[0], target[1] + 17),
            2,
        )

        vehicle = self._world_to_map(telemetry.position[0], telemetry.position[1])
        yaw = math.radians(telemetry.attitude_deg[2])
        nose = (
            vehicle[0] + round(24 * math.cos(yaw)),
            vehicle[1] - round(24 * math.sin(yaw)),
        )
        pygame.draw.circle(self.screen, self.CYAN, vehicle, 10)
        pygame.draw.line(self.screen, self.RED, vehicle, nose, 5)
        arm = 15
        pygame.draw.line(
            self.screen,
            self.CYAN,
            (vehicle[0] - arm, vehicle[1] - arm),
            (vehicle[0] + arm, vehicle[1] + arm),
            4,
        )
        pygame.draw.line(
            self.screen,
            self.CYAN,
            (vehicle[0] - arm, vehicle[1] + arm),
            (vehicle[0] + arm, vehicle[1] - arm),
            4,
        )

    def _draw_axis_bar(self, label: str, value: float, y_position: int) -> None:
        self._text(label, (685, y_position), self.MUTED, self.small_font)
        bar = pygame.Rect(805, y_position + 1, 265, 15)
        pygame.draw.rect(self.screen, (10, 16, 24), bar, border_radius=4)
        center = bar.centerx
        end = round(center + value * bar.width / 2)
        color = self.CYAN if value >= 0.0 else self.YELLOW
        pygame.draw.line(self.screen, color, (center, bar.centery), (end, bar.centery), 9)
        pygame.draw.line(
            self.screen, (125, 145, 165), (center, bar.top), (center, bar.bottom), 1
        )
        self._text(f"{value:+.2f}", (1077, y_position), self.TEXT, self.small_font)

    def _draw_buttons(
        self,
        state: str,
        connected: bool,
        emergency_pending: bool,
    ) -> None:
        mouse = pygame.mouse.get_pos()
        mouse_down = pygame.mouse.get_pressed()[0]
        for button in self.buttons:
            enabled = connected and state in button.enabled_states
            hovered = button.rect.collidepoint(mouse)
            color = button.color if enabled else (58, 63, 70)
            if enabled and hovered:
                color = tuple(min(value + 24, 255) for value in color)
            if enabled and hovered and mouse_down:
                color = tuple(max(value - 22, 0) for value in color)
            pygame.draw.rect(self.screen, color, button.rect, border_radius=7)
            border = self.TEXT if enabled else (90, 96, 104)
            pygame.draw.rect(self.screen, border, button.rect, 2, border_radius=7)
            label = button.label
            if button.action == "emergency_stop" and emergency_pending:
                label = "CONFIRM EMERGENCY STOP"
            text = self.small_font.render(label, True, self.TEXT)
            self.screen.blit(text, text.get_rect(center=button.rect.center))

    def draw(
        self,
        telemetry: TelemetryPacket,
        telemetry_connected: bool,
        focus_active: bool,
        axes: ControlAxes,
        emergency_pending: bool,
    ) -> None:
        self.screen.fill(self.BACKGROUND)
        self._text("Quadrotor Pygame Controller", (28, 22), font=self.title_font)
        self._text(
            "The MuJoCo Viewer is observation-only; flight input is accepted here.",
            (29, 57),
            self.MUTED,
            self.small_font,
        )

        connection_text = "CONNECTED" if telemetry_connected else "DISCONNECTED"
        connection_color = self.GREEN if telemetry_connected else self.RED
        focus_text = "CONTROL FOCUS ACTIVE" if focus_active else "NO CONTROL FOCUS"
        focus_color = self.GREEN if focus_active else self.RED
        self._text(connection_text, (685, 22), connection_color, self.title_font)
        self._text(focus_text, (885, 29), focus_color, self.body_font)
        self._draw_map(telemetry)

        self._text("FLIGHT STATE", (685, 92), self.MUTED, self.small_font)
        state_color = self.RED if telemetry.flight_state == "EMERGENCY_STOP" else self.CYAN
        self._text(telemetry.flight_state, (685, 113), state_color, self.title_font)
        rows = (
            "position      " + "  ".join(f"{v: .2f}" for v in telemetry.position),
            "velocity      " + "  ".join(f"{v: .2f}" for v in telemetry.velocity),
            "roll/pitch/yaw "
            + "  ".join(f"{v: .1f}" for v in telemetry.attitude_deg)
            + " deg",
            f"altitude / target  {telemetry.position[2]:.2f} / "
            f"{telemetry.target_position[2]:.2f} m",
            f"target yaw         {telemetry.target_yaw_deg:.1f} deg",
            f"command age        {telemetry.last_command_age:.3f} s",
            "motors        "
            + "  ".join(f"{v:.2f}" for v in telemetry.motor_thrusts)
            + " N",
        )
        for index, row in enumerate(rows):
            self._text(row, (685, 160 + index * 25), self.TEXT, self.small_font)

        self._text("ACTIVE HIGH-LEVEL COMMAND", (685, 347), self.MUTED, self.small_font)
        self._draw_axis_bar("Forward / Back", axes.forward, 374)
        self._draw_axis_bar("Right / Left", axes.right, 402)
        self._draw_axis_bar("Up / Down", axes.up, 430)
        self._draw_axis_bar("Yaw Left / Right", axes.yaw, 458)
        self._text(
            "W/S forward  A/D strafe  R/F altitude  Q/E yaw",
            (685, 493),
            self.TEXT,
            self.small_font,
        )
        self._text(
            "Focus loss -> neutral. Release keys to re-enable control.",
            (685, 518),
            self.MUTED,
            self.small_font,
        )
        self._draw_buttons(telemetry.flight_state, telemetry_connected, emergency_pending)
        pygame.display.flip()


def movement_keys_pressed(keys: Sequence[bool]) -> bool:
    return any(
        keys[key]
        for key in (
            pygame.K_w,
            pygame.K_s,
            pygame.K_a,
            pygame.K_d,
            pygame.K_r,
            pygame.K_f,
            pygame.K_q,
            pygame.K_e,
        )
    )


def pressed_axes(keys: Sequence[bool]) -> ControlAxes:
    # 成對按鍵相減，自然得到 -1/0/+1；同時按住相反方向會互相抵消。
    return ControlAxes(
        forward=float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
        right=float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
        up=float(keys[pygame.K_r]) - float(keys[pygame.K_f]),
        yaw=float(keys[pygame.K_q]) - float(keys[pygame.K_e]),
    )


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in [1, 65535]")
    return port


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def run_controller(
    host: str,
    command_port: int,
    telemetry_port: int,
    frames_per_second: int,
    command_rate: float,
    duration: float,
) -> int:
    link = ControllerLink(host, command_port, telemetry_port)
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Quadrotor Pygame Controller - keyboard input")
    dashboard = ControllerDashboard(screen)
    clock = pygame.time.Clock()
    focus = FocusSafety()
    minimized = False
    running = True
    axes = ControlAxes()
    next_command_time = 0.0
    start_time = time.monotonic()
    emergency_confirm_until = 0.0

    try:
        while running and time.monotonic() - start_time < duration:
            clock.tick(frames_per_second)
            now = time.monotonic()
            telemetry = link.poll(now)
            connected = link.is_connected(now)
            events = pygame.event.get()

            # 第一遍優先處理視窗生命週期與失焦，確保同一 frame 的按鍵事件
            # 不會在視窗已失焦後仍被當成飛行 action。
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWMINIMIZED:
                    minimized = True
                    if focus.force_lost():
                        axes = ControlAxes()
                        link.send_neutral()
                elif event.type == pygame.WINDOWRESTORED:
                    minimized = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    if focus.force_lost():
                        axes = ControlAxes()
                        link.send_neutral()

            # 連續軸每 frame 讀取實際 key state，不依賴 KEYDOWN repeat 頻率。
            keys = pygame.key.get_pressed()
            focus_active, neutral_required = focus.update(
                pygame.key.get_focused(), minimized, movement_keys_pressed(keys)
            )
            if neutral_required:
                axes = ControlAxes()
                link.send_neutral()
            axes = pressed_axes(keys) if focus_active else ControlAxes()

            # 第二遍才處理離散 action（起飛、降落等）與滑鼠按鈕。
            for event in events:
                if event.type == pygame.KEYDOWN:
                    shift = bool(event.mod & pygame.KMOD_SHIFT)
                    if event.key == pygame.K_ESCAPE and shift and focus_active:
                        if dashboard.action_enabled(
                            "emergency_stop", telemetry.flight_state, connected
                        ):
                            link.send(ControlAxes(), "emergency_stop")
                        continue
                    if event.key == pygame.K_ESCAPE:
                        axes = ControlAxes()
                        link.send_neutral()
                        running = False
                        continue
                    if not focus_active:
                        continue
                    key_actions = {
                        pygame.K_m: "arm",
                        pygame.K_n: "disarm",
                        pygame.K_t: "takeoff",
                        pygame.K_l: "land",
                        pygame.K_h: "hover",
                        pygame.K_SPACE: "hover",
                        pygame.K_BACKSPACE: "reset",
                    }
                    action = key_actions.get(event.key)
                    if action and dashboard.action_enabled(
                        action, telemetry.flight_state, connected
                    ):
                        link.send(ControlAxes(), action)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and focus_active
                ):
                    action = dashboard.action_at(
                        event.pos, telemetry.flight_state, connected
                    )
                    if action == "emergency_stop":
                        # 滑鼠急停需在兩秒內點兩次，降低單次誤觸風險。
                        if now <= emergency_confirm_until:
                            link.send(ControlAxes(), action)
                            emergency_confirm_until = 0.0
                        else:
                            emergency_confirm_until = now + 2.0
                    elif action is not None:
                        link.send(ControlAxes(), action)

            # UI 可用 60 FPS 繪圖，但 command 維持獨立的預設 50 Hz 發送率。
            if now >= next_command_time:
                link.send(axes)
                next_command_time = now + 1.0 / command_rate
            dashboard.draw(
                telemetry,
                connected,
                focus_active,
                axes,
                now <= emergency_confirm_until,
            )
    finally:
        try:
            # 正常關閉、例外或 Ctrl+C 都先送 neutral，縮短 timeout 接管前的空窗。
            link.send_neutral()
        finally:
            link.close()
            pygame.quit()
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Pygame controller for the MuJoCo quadrotor."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--command-port", type=port_number, default=DEFAULT_COMMAND_PORT
    )
    parser.add_argument(
        "--telemetry-port", type=port_number, default=DEFAULT_TELEMETRY_PORT
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--command-rate", type=positive_float, default=50.0)
    parser.add_argument("--duration", type=positive_float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            parser.error("--host must be a loopback address")
    except ValueError as error:
        parser.error(str(error))
    duration = args.duration if args.duration is not None else math.inf
    try:
        return run_controller(
            args.host,
            args.command_port,
            args.telemetry_port,
            args.fps,
            args.command_rate,
            duration,
        )
    except OSError as error:
        print(f"controller IPC error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
