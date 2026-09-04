# MuJoCo Quadrotor with Independent Pygame Controller

這是一個最小、可驗證的 MuJoCo 四旋翼 baseline。MuJoCo 物理模擬與 Pygame 飛行控制器是兩個獨立 process、兩個獨立視窗，透過 localhost UDP 傳送高層飛行命令與遙測。

Pygame 是唯一飛行輸入視窗。MuJoCo Viewer 只負責顯示模擬與相機操作，沒有註冊 WASD、Q/E、R/F、T、L 等飛行 callback。Pygame 不會修改 MuJoCo state，也不會發送四個馬達推力。

## 專案結構

```text
.
├── models/
│   ├── quadrotor.xml          # M0 理想模型：1 kg X 型四旋翼與四個 site actuators
│   └── quadrotor_m1.xml       # M1 模型：filterexact 馬達延遲 + MuJoCo 流體阻力
├── fly.py                     # 核心位置/SO(3)姿態控制器與相容 hover CLI
├── simulator.py               # MuJoCo、flight state machine、IPC server
├── pygame_controller.py       # 獨立 Pygame UI、鍵盤焦點與 IPC client
├── pygame_fly.py              # pygame_controller.py 的相容入口
├── ipc_protocol.py            # 封包、JSON encode/decode 與驗證
├── launch.py                  # 啟動並清理兩個子程序
├── m0_benchmark.py            # M0 起飛/懸停 headless benchmark（輸出報告與 JSON）
├── m1_realism.py              # M1 plant perturbation（質量/慣量/馬達效率/drag）與量測雜訊層
├── m1_benchmark.py            # M1 robustness 矩陣 benchmark（輸出 m1_robustness_report.json）
├── m2_mission.py              # M2 任務狀態機（READY→TAKEOFF→…→DONE/FAILED）與 heading-relative frame
├── m2_benchmark.py            # M2 自主任務 benchmark（輸出 m2_mission_report.json）
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
    ├── test_m1_benchmark.py       # M1 驗收：10 個 case 的 robustness 矩陣
    ├── test_m2_mission.py         # M2 任務狀態機：phase 轉移、timeout、measured/truth 分離
    ├── test_m2_frames.py          # M2 heading-relative frame 投影
    └── test_m2_benchmark.py       # M2 驗收：12 個 case 的任務矩陣 + heading 不變性
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
python m1_benchmark.py
python m2_benchmark.py
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
- M2 任務狀態機單元測試：phase 轉移、連續穩定計時、timeout、
  measured-state 決策與 ground-truth 評分結構性分離、heading 投影
- M2 benchmark：起飛 → 前進 5 m → 煞車 → 懸停 → 降落的 12 case 驗收

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
  fixed gain 實現。M1 nominal 不對稱 case 依馬達名稱明確定義為
  **FL=0.97、FR=1.00、RR=0.99、RL=1.02**（tuple `(0.97, 1.00, 0.99, 1.02)`
  對應 actuator 順序 FL/FR/RR/RL），會在起飛時產生可量測的 pitch/yaw
  擾動，由控制器回授吸收。tuple 順序由測試依 actuator 名稱鎖定，
  不可只依賴文字描述。
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

## M2 自主任務：起飛 → 前進 5 m → 煞車 → 懸停 → 降落

M0 驗證理想 plant/controller，M1 驗證非理想 robustness；M2 問的是：
「同一條 production path 能否完成一段有限距離的自主任務？」任務圖：

```text
READY → TAKEOFF → SETTLE → FORWARD → BRAKE → HOVER → LAND → DONE
                          （任何階段 timeout / 異常 → FAILED，附明確原因）
```

### Measured-state 決策 vs ground-truth 評分

M2 的核心規則：**mission 狀態機只看 measured state**（含 M1 量測雜訊），
ground truth 只用於 benchmark 評分、安全檢查與 NaN/runaway 偵測。
起飛完成、前進 5 m 到達、煞車完成、懸停穩定都由量測值判定——
MissionStateMachine 的 API 結構上不接受 truth state。因此報表刻意分開
記錄「飛機以為的位置」與「實際位置」（例如 combined case 在 LAND 前：
measured 5.122 m vs truth 5.113 m）。這個差距是日後 estimator 工作的
核心指標，不可合併成單一「距離」。

### Heading-relative 任務座標

「前進 5 m」定義為**任務起始時機頭方向的 5 m**，不是世界 +X。任務開始
時由 measured state 記錄 p0_xy 與 yaw0，定義：

```text
h = [cos yaw0, sin yaw0]   # 起始航向單位向量
l = [-sin yaw0, cos yaw0]  # 橫向單位向量
forward_progress = dot(p_xy - p0_xy, h)
lateral_error    = dot(p_xy - p0_xy, l)
```

yaw0 = 0° 時世界軌跡沿 +X；yaw0 = 90°（case10/11）時沿 +Y，
兩者的 projected forward displacement 都 ≈ 5 m。

### 速度命令剖面

FORWARD 依 measured 剩餘距離產生距離感知的速度命令：
remaining > slowdown_distance（2.0 m）時巡航 0.8 m/s；進入減速段後
v = max_speed × remaining / slowdown_distance，並以 minimum_approach_speed
（0.05 m/s）做下限。到達門檻後命令歸零進入 BRAKE，要求 measured
水平速度 < 0.20 m/s 連續 0.4 s 才算煞車完成，之後懸停 1.0 s 再降落。

注意：本專案的高層介面是 position-mode（速度命令被積分成位置 target
再由控制器追蹤），target 會「領先」機身約 v × τ_cascade。因此減速段
必須在機身到達前足夠早開始（slowdown_distance 需大於巡航領先量
≈ 1.1 m，實測 1.0 m 會衝到 5.46 m），minimum_approach_speed 也不能
太大（0.15 m/s 時 target 凍結後機身仍多走約 0.44 m，實測超出
5.3 m 門檻）。這兩個參數是任務層修正，控制器本身維持 M1 驗證值。

### M2 benchmark 與通過門檻

```bash
python m2_benchmark.py          # 12 個 case，輸出 m2_mission_report.json
```

Case 0 為 nominal 參照；case 1–9 逐一疊加 M1 非理想因素（馬達延遲、
質量/慣量 ±10%、drag、量測雜訊、馬達 mismatch，case 9 為全部疊加、
seed 27）；case 10/11 驗證 yaw0 = 90° 的 heading 不變性。每個 case
的通過門檻：mission DONE、truth 起飛高度誤差 < 0.15 m、LAND 前
truth 前進位移 4.7–5.3 m、|truth 橫向位移| < 0.20 m、進入 HOVER 時
truth 水平速度 < 0.30 m/s、最大傾角 < 10°（robust case < 15°）、
飽和比例 < 20%、無 NaN/Inf、無穿地。

M2 仍然不是最終 RM 微型無人機的物理模型；它是 estimator / 定位層
（M3）之前的最後一個 ground-truth-scored 任務里程碑。

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

- 控制器與 M2 任務已透過 M1 量測雜訊層取得 measured state，但仍沒有
  真正的 estimator（無 IMU 積分、optical flow、ToF 或濾波器）。
- 馬達以一階 lag 近似，沒有真實槳葉氣動、ground effect 或電池電壓
  衰退；drag 為 MuJoCo 等效橢球流體近似。
- UDP 僅供同一台機器的簡單控制，不是 MAVLink，也沒有加密或遠端網路支援。
- 失聯策略是懸停，不會自動返航或自動降落。
- 這是易懂、可擴充的 simulation baseline，不代表實機飛行安全性。
- 不依賴 ROS、ArduPilot、PX4 或其他大型飛控框架。
