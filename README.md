# MediaPipe to MANO Mesh

从 MediaPipe 的 21 个手部世界关键点，经逆运动学恢复对应的 MANO mesh，
并用 Open3D 显示。程序会根据 `multi_handedness` 自动选择
`MANO_LEFT.npz` 或 `MANO_RIGHT.npz`；双手同时出现时会同时显示两个 mesh。

The application recovers matching left/right MANO meshes from MediaPipe hand
world landmarks and visualizes one or both hands with Open3D.

![demo](./demo.gif)

普通预览和交互入口分别由 `configs/viewer.json` 与
`configs/interaction.json` 控制镜像及 handedness 映射。

## Installation

If you don't have python3.8,

```
brew install python@3.8
```

```
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

\* If you are in MacOS M processors, you may need python3.8 or lower due to the issue in Open3D. It may require open3d version 0.14.1 and this is available python3.8 or lower.

## Usage

普通 MANO 预览使用 [configs/viewer.json](./configs/viewer.json)：

```bash
python main.py
```

小球捏合交互使用 [configs/interaction.json](./configs/interaction.json)：

```bash
python interaction.py
```

两个入口都不再使用 argparse。可将自定义 JSON 路径作为唯一位置参数传入；
自定义文件只需包含需要覆盖的字段。例如 `my_interaction.json`：

```json
{
  "camera": {"index": 1, "mirror": false},
  "mano": {"iterations": 5},
  "depth_estimation": {
    "enabled": true,
    "reference_depth": 100.0,
    "maximum_depth": 300.0,
    "motion_gain": 1.5
  },
  "pinch": {"enter_distance": 0.03, "exit_distance": 0.04},
  "ball": {"depth": 200.0, "scale_sensitivity": 0.5}
}
```

```bash
python interaction.py my_interaction.json
```

配置会递归覆盖默认 JSON，未填写字段继续使用默认值。当前交互默认值包括：捏合
进入/退出距离 `35/45 mm`、球半径 `35 mm`、球心 `Z=+50 mm`、缩放范围
`0.35～5.0`、缩放灵敏度 `0.8`、不额外镜像摄像头、handedness 直接映射。

相对深度默认启用。每只手出现后的前 30 帧用于记录参考掌部尺寸，手部默认深度
由 `depth_estimation.reference_depth` 独立设置，默认 `100 mm`；它不会读取或
跟随 `ball.depth`。之后按“当前尺寸/参考尺寸”估计正 Z：手掌画面尺寸变大时
Z 增大，尺寸变小时 Z 减小。
`motion_gain` 围绕参考深度放大前后位移，默认 `1.5` 倍。结果限制在
`10～300 mm`，并经过移动限速和 One Euro 滤波。标定期间画面会显示
`CAL 0～100%`，建议保持手掌张开、正对摄像头且距离稳定。

捏合判定使用 MediaPipe 拇指尖 `4` 与食指尖 `8`；Open3D 抓握点使用 MANO
拇指尖 `20` 与食指尖 `16` 的中点。单手控制平移和掌面旋转；双手以第一个
抓握点为枢轴，根据两点方向和距离控制旋转与缩放。

按 `1` 使用正视图，按 `3` 使用深度检查视图，按 `q`/`Esc` 退出。

## Project structure

```text
configs/                         默认 JSON 配置
mediapipe2mesh/
  config.py                      配置加载与局部覆盖
  ik/keypoints_to_mano.py        骨架重定向与 MANO IK
  tracking/filters.py            One Euro 时序滤波
  tracking/depth.py              基于手掌视觉尺寸的相对深度估计
  tracking/handedness.py         左右手映射与轨迹防抖
  tracking/hand_state.py         单手异步求解状态
  visualization/scene.py         Open3D 相机与坐标映射
  interaction/components.py      小球、捏合状态与单双手控制器
  apps/common.py                 摄像头与检测公共流程
  apps/viewer_app.py             普通预览应用
  apps/interaction_app.py        交互应用
main.py / interaction.py         兼容启动入口
```

## Credit

[CalciferZh/Minimal-IK](https://github.com/CalciferZh/Minimal-IK)

[SMPL MANO](https://mano.is.tue.mpg.de/)

[Mediapipe](https://pypi.org/project/mediapipe/)
