"""Versioned localhost UDP protocol for quadrotor commands and telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Iterable


PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_COMMAND_PORT = 14560
DEFAULT_TELEMETRY_PORT = 14561
MAX_PACKET_BYTES = 4096

# 白名單同時用於 decode 驗證與 UI/state machine 對照；新增項目時要同步更新測試。
COMMAND_ACTIONS = frozenset(
    {
        "arm",
        "disarm",
        "takeoff",
        "hover",
        "land",
        "reset",
        "emergency_stop",
    }
)
FLIGHT_STATES = frozenset(
    {
        "DISARMED",
        "ARMED_IDLE",
        "TAKING_OFF",
        "HOVERING",
        "MANUAL",
        "LANDING",
        "LANDED",
        "EMERGENCY_STOP",
    }
)


class ProtocolError(ValueError):
    """Raised when an IPC packet is malformed or outside its legal range."""


@dataclass(frozen=True)
class CommandPacket:
    """Pygame → simulator 的高層命令；四個軸皆為 [-1, 1] normalized 值。"""

    version: int
    sequence: int
    timestamp: float
    action: str | None
    forward: float
    right: float
    up: float
    yaw: float

    @classmethod
    def neutral(
        cls, sequence: int, timestamp: float, action: str | None = None
    ) -> "CommandPacket":
        return cls(PROTOCOL_VERSION, sequence, timestamp, action, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TelemetryPacket:
    """simulator → Pygame 的唯讀狀態快照。"""

    version: int
    sequence: int
    timestamp: float
    flight_state: str
    connected: bool
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    attitude_deg: tuple[float, float, float]
    target_position: tuple[float, float, float]
    target_yaw_deg: float
    motor_thrusts: tuple[float, float, float, float]
    last_command_age: float


# 從 dataclass 自動取得完整 schema；decode 要求 exact keys，拼錯或多出的欄位不會被靜默忽略。
COMMAND_KEYS = frozenset(asdict(CommandPacket.neutral(0, 0.0)).keys())
TELEMETRY_KEYS = frozenset(
    asdict(
        TelemetryPacket(
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
            0.0,
        )
    ).keys()
)


def _reject_json_constant(value: str) -> None:
    # Python json 預設接受非標準 NaN/Infinity；控制與遙測不能讓它們流入數值運算。
    raise ProtocolError(f"JSON constant is not permitted: {value}")


def _decode_json(packet: bytes | str) -> dict[str, Any]:
    try:
        raw = packet.decode("utf-8") if isinstance(packet, bytes) else packet
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid JSON packet: {error}") from error
    if not isinstance(parsed, dict):
        raise ProtocolError("packet root must be a JSON object")
    return parsed


def _require_exact_keys(packet: dict[str, Any], expected: frozenset[str]) -> None:
    # 嚴格 schema 可讓兩個 process 的版本不一致時立刻失敗，而不是使用錯誤預設值。
    actual = frozenset(packet.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolError(f"packet keys mismatch; missing={missing}, extra={extra}")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    # bool 是 int 的子類別，所以必須先明確排除 True/False。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ProtocolError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ProtocolError(f"{name} must be <= {maximum}")
    return number


def _vector(
    value: Any,
    name: str,
    length: int,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{name} must be a {length}-element JSON array")
    return tuple(
        _finite_number(component, f"{name}[{index}]", minimum, maximum)
        for index, component in enumerate(value)
    )


def _encode(packet: object) -> bytes:
    try:
        # compact separators 減少 UDP payload；sort_keys 讓測試與除錯輸出可重現。
        return json.dumps(
            asdict(packet), allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"packet cannot be encoded: {error}") from error


def validate_command(packet: CommandPacket) -> CommandPacket:
    if packet.version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported command version: {packet.version}")
    _integer(packet.sequence, "sequence")
    timestamp = _finite_number(packet.timestamp, "timestamp", 0.0)
    if packet.action is not None and packet.action not in COMMAND_ACTIONS:
        raise ProtocolError(f"unknown action: {packet.action!r}")
    axes = tuple(
        _finite_number(value, name, -1.0, 1.0)
        for name, value in (
            ("forward", packet.forward),
            ("right", packet.right),
            ("up", packet.up),
            ("yaw", packet.yaw),
        )
    )
    return CommandPacket(
        PROTOCOL_VERSION,
        packet.sequence,
        timestamp,
        packet.action,
        *axes,
    )


def encode_command(packet: CommandPacket) -> bytes:
    return _encode(validate_command(packet))


def decode_command(packet: bytes | str) -> CommandPacket:
    # 解碼邊界依序做 JSON、schema、型別/範圍、跨欄位規則驗證。
    parsed = _decode_json(packet)
    _require_exact_keys(parsed, COMMAND_KEYS)
    version = _integer(parsed["version"], "version")
    sequence = _integer(parsed["sequence"], "sequence")
    action = parsed["action"]
    if action is not None and not isinstance(action, str):
        raise ProtocolError("action must be null or a string")
    decoded = CommandPacket(
        version,
        sequence,
        _finite_number(parsed["timestamp"], "timestamp", 0.0),
        action,
        _finite_number(parsed["forward"], "forward"),
        _finite_number(parsed["right"], "right"),
        _finite_number(parsed["up"], "up"),
        _finite_number(parsed["yaw"], "yaw"),
    )
    return validate_command(decoded)


def validate_telemetry(packet: TelemetryPacket) -> TelemetryPacket:
    if packet.version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported telemetry version: {packet.version}")
    _integer(packet.sequence, "sequence")
    timestamp = _finite_number(packet.timestamp, "timestamp", 0.0)
    if packet.flight_state not in FLIGHT_STATES:
        raise ProtocolError(f"unknown flight state: {packet.flight_state!r}")
    if not isinstance(packet.connected, bool):
        raise ProtocolError("connected must be a boolean")
    position = _tuple_vector(packet.position, "position", 3, -1000.0, 1000.0)
    velocity = _tuple_vector(packet.velocity, "velocity", 3, -1000.0, 1000.0)
    attitude = _tuple_vector(packet.attitude_deg, "attitude_deg", 3, -3600.0, 3600.0)
    target = _tuple_vector(
        packet.target_position, "target_position", 3, -1000.0, 1000.0
    )
    target_yaw = _finite_number(packet.target_yaw_deg, "target_yaw_deg", -180.0, 180.0)
    motors = _tuple_vector(packet.motor_thrusts, "motor_thrusts", 4, 0.0, 100.0)
    age = _finite_number(packet.last_command_age, "last_command_age", 0.0, 1e6)
    return TelemetryPacket(
        PROTOCOL_VERSION,
        packet.sequence,
        timestamp,
        packet.flight_state,
        packet.connected,
        position,
        velocity,
        attitude,
        target,
        target_yaw,
        motors,
        age,
    )


def _tuple_vector(
    value: Iterable[float],
    name: str,
    length: int,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise ProtocolError(f"{name} must be iterable") from error
    if len(values) != length:
        raise ProtocolError(f"{name} must contain {length} values")
    return tuple(
        _finite_number(component, f"{name}[{index}]", minimum, maximum)
        for index, component in enumerate(values)
    )


def encode_telemetry(packet: TelemetryPacket) -> bytes:
    return _encode(validate_telemetry(packet))


def decode_telemetry(packet: bytes | str) -> TelemetryPacket:
    # 與 command 相同，外部資料只有通過完整驗證後才建立可信任 dataclass。
    parsed = _decode_json(packet)
    _require_exact_keys(parsed, TELEMETRY_KEYS)
    decoded = TelemetryPacket(
        _integer(parsed["version"], "version"),
        _integer(parsed["sequence"], "sequence"),
        _finite_number(parsed["timestamp"], "timestamp", 0.0),
        parsed["flight_state"],
        parsed["connected"],
        _vector(parsed["position"], "position", 3, -1000.0, 1000.0),
        _vector(parsed["velocity"], "velocity", 3, -1000.0, 1000.0),
        _vector(parsed["attitude_deg"], "attitude_deg", 3, -3600.0, 3600.0),
        _vector(parsed["target_position"], "target_position", 3, -1000.0, 1000.0),
        _finite_number(parsed["target_yaw_deg"], "target_yaw_deg"),
        _vector(parsed["motor_thrusts"], "motor_thrusts", 4, 0.0, 100.0),
        _finite_number(parsed["last_command_age"], "last_command_age"),
    )
    if not isinstance(decoded.flight_state, str):
        raise ProtocolError("flight_state must be a string")
    return validate_telemetry(decoded)
