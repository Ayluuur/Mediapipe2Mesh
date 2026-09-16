# MediaPipe2Mesh

Python 骨骼位移、旋转与坐标约定见 [骨骼数据接口文档](docs/python_skeleton_api.md)，包含异步调用示例和姿势恢复修正说明。

从摄像头获取 MediaPipe Hands 的 21 个三维关键点，通过IK恢复对应的
 MANO 手部网格，并在 Open3D 中显示和交互。在原项目基础上增加了
 左右手独立跟踪、相对深度估计和基于捏合的小球操作，用于手势交互原型。

## 功能

- 根据稳定后的 handedness 自动选择 `MANO_LEFT.npz` 或 `MANO_RIGHT.npz`。
- 使用 One Euro Filter、上一帧姿态先验和异步 IK 降低抖动与界面阻塞。
- 根据手掌屏幕尺寸估计相对深度。
- 使用捏合操作可对小球进行移动、旋转、缩放操作


## 安装

```powershell
conda create -n media2mano python=3.10 -y
conda activate media2mano
pip install -r requirements.txt
```


## 运行

### MANO 双手预览

默认读取 [configs/viewer.json](./configs/viewer.json)：

```powershell
python main.py
```

### 小球捏合交互

默认读取 [configs/interaction.json](./configs/interaction.json)：

```powershell
python interaction.py
```


## 配置

两个入口接收一个可选的 JSON 配置路径：

```powershell
python main.py path\to\viewer_override.json
python interaction.py path\to\interaction_override.json
```
主要配置分组如下：

| 分组 | 用途 |
| --- | --- |
| `camera` | 摄像头编号、分辨率和画面镜像 |
| `detector` | MediaPipe 模型复杂度、手数和检测阈值 |
| `mano` | IK 迭代次数与姿态平滑权重 |
| `tracking` | handedness 映射和切换确认帧数 |
| `depth_estimation` | 手掌尺寸标定、Z 范围、滤波和移动倍率 |
| `viewer` | Open3D 初始视角与窗口名称 |
| `pinch` | 捏合进入/退出阈值和抓取容差 |
| `ball` | 球半径、中心坐标、缩放范围和灵敏度 |



小球初始位置通过 `ball.center` 设置，格式为 `[x, y, z]`，单位为 mm。
默认值为 `[0.0, 0.0, 250.0]`，第三个值 Z 取代原来的 `ball.depth`。
自定义配置中的 `ball.depth` 也应改为 `ball.center`。

## 深度按钮交互（Python）

运行 `python interaction.py`，完成手掌深度标定后，将任意一只手的食指尖移向按钮。
左右食指尖保留触碰判定，与 MANO 网格使用同一帧关节数据和场景坐标，不显示判定球。
按钮被触碰时向深处压下并变绿；双手均离开后恢复。
摄像头画面中的 `Button` 显示按下状态、触发手和累计按下次数。按 Enter 可重新标定。

当前正面视角从负 Z 看向正 Z，因此按钮默认正面位于 `[110, 30, 370]` mm，
比初始小球中心（Z=250 mm）更深，横向错开以方便观察。按钮沿 +Z 压下，
切换观察视角不会改变按钮的物理位置和按压方向。

可在 `configs/interaction.json` 的 `button` 分组调整 `center`（正面中心坐标）、
`width`、`height`、`travel`（按压行程）和 `tip_radius`（判定球半径），单位均为 mm。
判定包含前向快速穿越检测及释放容差；手部检测丢失会释放对应接触。

## Python 跟踪抗抖

世界关键点、屏幕手腕位置、场景 XY 和深度使用三帧中值处理，抑制孤立检测尖峰。
位置滤波在计算自适应平滑系数前限制输入变化，避免把异常跳点当作快速运动。
短暂漏检保留滤波历史，超过 0.35 秒后重新建立跟踪。

新手部须经过 `tracking.handedness_confirm_frames` 帧连续确认才启用；跟踪期间
按手腕位置关联并锁定左右身份，标签抖动不会直接切换 MANO 模型。持续丢失
0.35 秒后关闭，重新进入时再次确认。初次确认的左右身份若不正确，需离开画面后重新进入。

两个 Python 入口的 `tracking.position_filter` 均支持：

| 配置 | 含义 |
| --- | --- |
| `min_cutoff` | 默认 1.5 Hz，降低后更平稳，但延迟增大 |
| `beta` | 默认 0.005，越大越容易跟随快速变化 |
| `median_window` | 默认 3，必须为正奇数；设为 1 关闭中值处理 |
| `max_speed` | 场景 XY 位置变化上限，默认 500 mm/s |

深度速度上限仍使用 `depth_estimation.max_speed`。三帧中值处理在稳定帧率下
约增加一帧延迟；持续误识别不能仅靠滤波消除。

## Credits

- [CalciferZh/Minimal-IK](https://github.com/CalciferZh/Minimal-IK)
- [MANO](https://mano.is.tue.mpg.de/)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [Open3D](https://www.open3d.org/)
