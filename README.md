# MuJoCo Quadrotor with Independent Pygame Controller

這是一個最小、可驗證的 MuJoCo 四旋翼 baseline。MuJoCo 物理模擬與 Pygame 飛行控制器是兩個獨立 process、兩個獨立視窗，透過 localhost UDP 傳送高層飛行命令與遙測。

Pygame 是唯一飛行輸入視窗。MuJoCo Viewer 只負責顯示模擬與相機操作，沒有註冊 WASD、Q/E、R/F、T、L 等飛行 callback。Pygame 不會修改 MuJoCo state，也不會發送四個馬達推力。

## 專案結構

```text
.
├── models/
│   └── quadrotor.xml          # 1 kg X 型四旋翼與四個 site actuators
├── fly.py                     # 核心位置/SO(3)姿態控制器與相容 hover CLI
├── simulator.py               # MuJoCo、flight state machine、IPC server
├── pygame_controller.py       # 獨立 Pygame UI、鍵盤焦點與 IPC client
├── pygame_fly.py              # pygame_controller.py 的相容入口
├── ipc_protocol.py            # 封包、JSON encode/decode 與驗證
├── launch.py                  # 啟動並清理兩個子程序
├── m0_benchmark.py            # M0 起飛/懸停 headless benchmark（輸出報告與 JSON）
├── m1_realism.py              # M1 plant perturbation（質量/慣量/馬達效率/drag）與量測雜訊層
├── m1_benchmark.py            # M1 robustness 矩陣 benchmark（輸出 m1_robustness_report.json）
├── requirements.txt
├── pytest.ini
└── tests/
    ├── test_protocol.py
    ├── test_flight_sequence.py
    ├── test_controller_safety.py
    ├── test_udp_integration.py
    ├── test_smoke.py
    ├── test_m0_physics.py     # 自由落體 / 懸停推力 / mixer 方向的物理 sanity tests
    ├── test_m0_benchmark.py   # M0 驗收：1.5 m 起飛 + 5 s 懸停
    ├── test_m1_motor_dynamics.py  # 一階馬達延遲步階響應
    ├── test_m1_measurements.py    # 量測雜訊：seed 重現性、零雜訊、合法旋轉
    ├── test_m1_robustness.py      # 質量/慣量/效率 perturbation、drag、不對稱暫態
    └── test_m1_benchmark.py       # M1 驗收：10 個 case 的 robustness 矩陣
```

## 建議閱讀順序

如果你想從「模型如何產生推力」一路讀到「鍵盤如何讓飛機移動」，建議依下列順序查閱：

1. `models/quadrotor.xml`：先認識機體座標、質量/慣量、旋翼位置與 actuator。
2. `fly.py`：理解位置控制、SO(3) 姿態控制與四旋翼 mixer。
3. `ipc_protocol.py`：查看 Pygame 與 simulator 之間允許傳送的資料格式。
4. `simulator.py`：追蹤飛行狀態機、起降軌跡、手動 target 積分與 MuJoCo stepping。
5. `pygame_controller.py`：理解鍵盤焦點保護、按鍵映射、命令發送與遙測畫面。
6. `launch.py`：查看兩個獨立 process 如何一起啟動與結束。
7. `tests/`：用測試案例確認封包邊界、狀態轉移、安全條件與完整飛行流程。

一個持續按住 `W` 的命令會沿著這條資料流前進：

```text
pygame.key.get_pressed()
    → pressed_axes() 產生 normalized forward=1
    → ControllerLink.send() 編碼 CommandPacket
    → localhost UDP
    → LocalUdpTransport.receive_commands() 解碼與驗證
    → QuadrotorSimulation.accept_command() 過濾舊 sequence
    → FlightStateMachine.integrate_manual() 更新世界座標 target
    → QuadrotorController.step_controller() 計算四槳推力
    → MuJoCo mj_step()
    → TelemetryPacket 經 UDP 回到 Pygame 畫面
```

程式內的中文註解集中解釋「為什麼這樣做」、座標/符號約定與安全邊界；函式名稱與型別則保留英文，方便對照 MuJoCo 與控制理論文件。

## 系統需求與安裝

- Python 3.10+
- Linux、macOS 或 Windows
- MuJoCo Viewer 模式需要可用的 OpenGL/display
- `pygame-ce` distribution；module 名稱仍是 `import pygame`

Linux/macOS：

```bash
cd ~/rm_uav_lab/mujoco_uav
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
cd path\to\rm_uav_lab\mujoco_uav
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若同一環境曾安裝傳統 `pygame`，先執行 `python -m pip uninstall pygame`，避免它與 `pygame-ce` 覆蓋同一個 module。

## 一次啟動兩個視窗

```bash
cd ~/rm_uav_lab/mujoco_uav
source .venv/bin/activate
python launch.py
```

啟動後會看到：

```text
MuJoCo Viewer
    只顯示 3D 模擬並提供相機操作。

Pygame Controller
    唯一接收飛行鍵盤與按鈕輸入的視窗。
```

先點擊 Pygame Controller，使畫面右上方顯示 `CONTROL FOCUS ACTIVE`，再使用 WASD。若顯示 `NO CONTROL FOCUS`，所有連續命令均為零。

macOS 若 passive viewer 要求主執行緒 wrapper：

```bash
python launch.py --sim-python mjpython
```

也可以設定：

```bash
export MUJOCO_SIM_PYTHON=mjpython
python launch.py
```

`launch.py` 在任一子程序結束或收到 Ctrl+C 時會終止另一個子程序，避免留下孤兒 process。

## 分別啟動

終端 1：

```bash
python simulator.py
```

終端 2：

```bash
python pygame_controller.py
```

控制器可以先開；沒有 simulator 時顯示 `DISCONNECTED`，但不會崩潰。Simulator 也可以在沒有控制器時持續運作。

## IPC 設計

只接受 loopback address：

```text
Pygame → simulator commands:  127.0.0.1:14560，約 50 Hz
simulator → Pygame telemetry:  127.0.0.1:14561，約 25 Hz
```

兩個 UDP socket 都不阻塞 simulation/render loop。封包包含 `version`、單調遞增 `sequence` 與 monotonic `timestamp`；malformed JSON、NaN/Inf、未知 action、非 localhost 來源與超出 ±1 的 command 都會被拒絕。舊 sequence 會被丟棄，因此重複 UDP action 不會重複觸發。

命令是 normalized 高層輸入：

```json
{
  "version": 1,
  "sequence": 123,
  "timestamp": 12.345,
  "action": null,
  "forward": 0.0,
  "right": 0.0,
  "up": 0.0,
  "yaw": 0.0
}
```

`action` 只允許 `arm`、`disarm`、`takeoff`、`hover`、`land`、`reset`、`emergency_stop`。IPC 不包含單槳命令。

遙測包含 connection、flight state、位置、速度、roll/pitch/yaw、target、四槳推力與 command age。Pygame 會全部顯示。

## Pygame 按鍵

| 按鍵 | 功能 |
|---|---|
| W / S | 沿目前 yaw 前進 / 後退 |
| A / D | 沿目前 yaw 左移 / 右移 |
| R / F | 上升 / 下降 |
| Q / E | 左偏航 / 右偏航 |
| T | 平滑起飛；會自動 ARM |
| L | 受控降落 |
| H / Space | HOVER/BRAKE，捕捉並保持目前 pose target |
| M | ARM |
| N | DISARM；只允許低速地面狀態 |
| Backspace | RESET；只允許地面或 EMERGENCY_STOP |
| Shift + Escape | EMERGENCY STOP |
| Escape | 送出 neutral command 並關閉 Pygame，不會直接切馬達 |

Pygame 也提供 ARM、DISARM、TAKE OFF、HOVER、LAND、RESET、EMERGENCY STOP 按鈕。按鈕會依目前 state disabled；EMERGENCY STOP 必須在兩秒內點擊兩次確認。

## 焦點與卡鍵保護

每個 Pygame frame 都檢查 `pygame.key.get_focused()`，連續控制使用 `pygame.key.get_pressed()`，不依賴 OS key repeat，也不使用全域 keyboard hook。

發生失焦、最小化、切換到 MuJoCo Viewer 或退出時：

1. 本地 axes 立即清零。
2. 立即發出 neutral UDP command。
3. UI 顯示 `NO CONTROL FOCUS`。
4. 回到 Pygame 後，必須先完全鬆開 W/S/A/D/R/F/Q/E，才會重新顯示 `CONTROL FOCUS ACTIVE`。

這避免按住 W 切換視窗後命令卡住。

## 手動飛行語義

輸入由 simulator 依無人機當前 yaw 轉成 heading-relative 世界座標 target velocity，再以 simulation timestep 積分成位置 target：

```text
forward max speed:    1.0 m/s
vertical max speed:   0.6 m/s
yaw max rate:         60 deg/s
horizontal target:    ±10 m
altitude target:      0.15–5.0 m
yaw target:           [-π, π)
```

放開按鍵後停止積分並保留最後 target，原位置與 SO(3) 姿態控制器會在該處懸停。

## 飛行狀態機

```text
DISARMED
    ↓ ARM / TAKEOFF
ARMED_IDLE
    ↓ TAKEOFF
TAKING_OFF
    ↓ target reached and stable for 0.4 s
HOVERING
    ↔ MANUAL
    ↓ LAND
LANDING
    ↓ ground stable for 0.4 s
    ↓ motor ramp-down
LANDED
    ↓ automatic safe disarm
DISARMED
```

任何 powered state 都可以進入 `EMERGENCY_STOP`。只有在地面、低速且姿態合理時才能 RESET 回到 `DISARMED`。

### 起飛

- 驗證地面高度、線速度、角速度與姿態。
- 自動 ARM，捕捉目前 x/y/yaw。
- 以 3 秒 smoothstep 將高度 target 升到預設 1.2 m。
- profile 峰值上升速度受 duration 限制。
- `|z error| < 0.08 m` 且 `|vz| < 0.12 m/s` 持續 0.4 秒後進入 HOVERING。
- TAKING_OFF 期間忽略普通 manual axes，只接受 LAND 與 EMERGENCY STOP。

### HOVER / BRAKE

清除 manual axes、捕捉目前 x/y/z/yaw 並設成新 target，完全透過 controller 減速；不修改 MuJoCo pose 或 velocity。

### 降落

- 捕捉開始時的 x/y/yaw 並固定。
- smoothstep 峰值下降速度限制為約 0.32 m/s，接近地面自動減速。
- 綜合高度、垂直速度與 roll/pitch，接地條件需連續成立 0.4 秒。
- 接地後用 0.5 秒把現有 motor thrust 線性降到 0。
- 經 LANDED 自動轉成 DISARMED。
- 高度仍足夠時可按 H/Space 中止降落並懸停。

### ARM、DISARM 與 EMERGENCY

- ARM 只允許安全地面狀態。
- 飛行中普通 DISARM 會被 simulator 拒絕。
- EMERGENCY STOP 是唯一立即將馬達歸零的飛行 action；空中觸發會掉落。
- EMERGENCY_STOP 只能在確認落地後 RESET。

## 失聯保護

Simulator 使用本地 monotonic receive time，而不是相信封包 timestamp。

- 有效 command 超過 0.5 秒未收到：axes 歸零、捕捉目前位置與 yaw、進入 HOVERING。
- DISARMED/LANDED 失聯：保持 motor 0。
- LANDING 失聯：繼續已開始的受控降落。
- 最小版本不做自動降落；長時間失聯維持懸停。
- Pygame 關閉不會讓 simulator 崩潰或立即切馬達。

## CLI

Simulator：

```bash
python simulator.py
python simulator.py --headless
python simulator.py --headless --scenario takeoff_hover --duration 8
python simulator.py --headless --scenario takeoff_land --duration 14
python simulator.py --command-port 14560 --telemetry-port 14561
```

Controller：

```bash
python pygame_controller.py
python pygame_controller.py --host 127.0.0.1
python pygame_controller.py --command-port 14560 --telemetry-port 14561
```

Launcher：

```bash
python launch.py
python launch.py --headless
python launch.py --sim-python mjpython
python launch.py --command-port 14560 --telemetry-port 14561
```

舊的 `fly.py` 保留 core smoke/hover 相容性，但它的 viewer 沒有飛行 keyboard mode：

```bash
python fly.py --headless --trajectory hover --duration 6
python fly.py --trajectory hover
```

## 自動化驗收

```bash
python -m pytest tests/test_protocol.py -q
python -m pytest -q
python simulator.py --headless --scenario takeoff_hover --duration 8
python simulator.py --headless --scenario takeoff_land --duration 14
python m0_benchmark.py
```

測試涵蓋：

- command/telemetry encode、decode 與 round-trip
- malformed JSON、NaN/Inf、out-of-range command
- 重複 action sequence 去重
- takeoff → hover 與 takeoff → land → disarm
- heading-relative forward command
- 飛行中 DISARM 拒絕與 EMERGENCY 零推力
- 0.5 秒 command timeout 回到 HOVERING
- Pygame focus loss、最小化與 key-release re-arm
- 真 localhost UDP command/telemetry round-trip
- 原始 mixer、位置、yaw 與 hover regression
- M0 物理 sanity tests：自由落體加速度、懸停推力 m·g/4、推力方向、
  roll/pitch/yaw mixer 產生的力矩符號
- M0 benchmark：起飛到 1.5 m 並懸停 5 秒的完整驗收門檻

## M1 非理想 robustness

M0 是理想剛體/控制 baseline；M1 問的是：「當 plant 與量測不再完美時，
同一套控制器與任務架構是否仍然穩定？」M1 只加入溫和、可理解的非理想
因素，**不是**最終 RM 微型無人機的驗證物理模型，而是 ideal simulation
與日後 system identification / Sim-to-Real 之間的 robustness bridge。

### 非理想因素

- **馬達一階延遲**（`models/quadrotor_m1.xml`）：`<motor>` shortcut 換成
  等價 `<general dyntype="filterexact" dynprm="0.040 ...">`，實際推力遵循
  Ḟ=(u−F)/τ，τ=0.040 s（測試另覆蓋 0.020/0.080）。名稱、site、gear、
  ctrlrange、推力單位（牛頓）與 M0 完全相同。
- **空氣阻力**（M1 XML 內建，`density=1.2`、`viscosity=1.8e-5`）：
  MuJoCo 內建流體交互以等效橢球近似對機體 geom 施加與速度反向的
  阻力/黏性項；不含槳葉尾流、地面效應或升力。`PlantConfig(drag=...)`
  可在任一模型上開關。
- **質量/慣量不確定性**（`PlantConfig(mass_scale=, inertia_scale=)`）：
  ±10%。物理模型改變，但 controller 永遠使用 nominal 1.0 kg 與
  (0.018, 0.018, 0.030)——這正是「不確定性」的意義。
- **馬達效率不一致**（`PlantConfig(motor_effectiveness=)`，順序為
  FL/FR/RR/RL）：實際推力 = effectiveness × 指令推力，以 actuator
  fixed gain 實現。M1 nominal 不對稱 case 為 (0.97, 1.00, 0.99, 1.02)，
  會在起飛時產生可量測的 pitch/yaw 擾動，由控制器回授吸收。
- **量測雜訊**（`NoiseConfig` + seeded `MeasurementModel`）：位置
  σ=0.010 m、速度 σ=0.020 m/s、姿態以小旋轉向量擾動 σ=0.2°
  （保證仍是合法旋轉，不直接對 quaternion 分量加噪）、角速度
  σ=0.01 rad/s。雜訊只進入控制器看到的 measured state；ground truth
  保留給狀態機、安全檢查與 metrics。相同 seed 完全可重現。
- **高度積分項**（M1 對控制器的唯一修改）：純 PD 在 ±10% 質量下有
  恆定誤差 e=Δm·g/kp_z=0.196 m（實測），導致永遠無法進入 HOVERING。
  因此在垂直通道加入小積分（ki_z=1.5，積分貢獻限幅 ±2.0 m/s²），
  消除恆定力偏差；M0 全部測試與 benchmark 維持通過。

### M1 benchmark

```bash
python m1_benchmark.py          # 10 個 case，輸出 m1_robustness_report.json
```

每個 case 固定一組 plant 參數跑完整 production path（action → 狀態機 →
控制器 → mixer → actuator → MuJoCo → measurement → 控制器），起飛到
1.5 m 後懸停 5 秒。Case 0 為 M0-like 參照；case 9 為全部非理想因素
疊加（seed 27，完全可重現）。

每個 case 的通過門檻：final altitude error < 0.15 m、hover RMS < 0.10 m、
水平漂移 < 0.15 m、起飛暫態後 roll/pitch < 7°、懸停水平速度 < 0.30 m/s、
懸停滿 5 s、無 NaN/Inf、無穿地、無持續 actuator 飽和（< 20%）。

## M0 驗證

M0 的目的是在沒有任何學習演算法之前，先證明「模型物理 + 控制器 +
狀態機」是可信的 baseline。驗證分兩層：

1. `tests/test_m0_physics.py`：直接對 MuJoCo plant 施力，驗證自由落體
   加速度等於 −g、總推力 m·g（每槳 m·g/4 = 2.4525 N）時加速度為零、
   增大總推力向上加速、以及 roll/pitch/yaw 差動推力產生的角加速度
   符號與大小符合 allocation matrix 的解析預測（τ = I·α）。
2. `python m0_benchmark.py`：只透過 TAKEOFF action 與控制器閉迴路到達
   1.5 m，狀態機宣告 HOVERING 後量測 5 秒懸停，輸出報告與
   `m0_hover_report.json`。驗收門檻：final altitude error ≤ 0.10 m、
   hover RMS error ≤ 0.10 m、水平漂移 ≤ 0.10 m、roll/pitch ≤ ±5°、
   無墜毀、無 NaN/Inf、懸停滿 5 s。pytest 中的
   `tests/test_m0_benchmark.py` 會重跑同一個 benchmark 作為 milestone gate。

## 高層動作介面（vx, vy, vz, yaw_rate）

IPC 上的四個 normalized 軸 `forward / right / up / yaw` 刻意設計成與
「期望速度」一一對應，而不是馬達命令。換算（ψ 為當前 yaw，世界座標
+Z 向上）：

```text
vx       = 1.0 m/s  * (forward * cos ψ + right * sin ψ)
vy       = 1.0 m/s  * (forward * sin ψ - right * cos ψ)
vz       = 0.6 m/s  * up
yaw_rate = 60 deg/s * yaw          # 正號為逆時針（向左轉）
```

反向換算（例如未來的 trajectory controller 或 policy 輸出速度命令）：

```text
forward = (vx * cos ψ + vy * sin ψ) / 1.0
right   = (vx * sin ψ - vy * cos ψ) / 1.0
up      = vz / 0.6
yaw     = yaw_rate / (60 deg/s)
```

語義上這是 position-mode：simulator 把 vx/vy/vz/yaw_rate 以 simulation
timestep 積分成位置/航向 target，再交給位置 + SO(3) 姿態控制器追蹤；
放開輸入（全零）時 target 停在原地懸停。未來的軌跡控制器、神經網路
policy 或 companion computer 只需要輸出 (vx, vy, vz, yaw_rate)，經過
上表正規化後即可沿用同一條 IPC 與控制管線，永遠不需要直接碰四個馬達。

## 目前限制

- 使用 MuJoCo ground-truth state，沒有感測器噪聲或 estimator。
- 沒有 motor lag、真實槳葉、ground effect、drag 或複雜氣動。
- UDP 僅供同一台機器的簡單控制，不是 MAVLink，也沒有加密或遠端網路支援。
- 失聯策略是懸停，不會自動返航或自動降落。
- 這是易懂、可擴充的 simulation baseline，不代表實機飛行安全性。
- 不依賴 ROS、ArduPilot、PX4 或其他大型飛控框架。
