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

```powershell
python main.py --mode interaction
python main.py --mode multicore
python main.py --mode viewer
```

分别表示` 交互预览` 、`多核并行测试` 、`原双手预览` 

`--mode viewer` 保留原双手 MANO 预览并读取 `configs/viewer.json`。



## 配置

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


## 跟踪抗抖

| 配置 | 含义 |
| --- | --- |
| `min_cutoff` | 默认 3.0 Hz，降低后更平稳，但延迟增大 |
| `beta` | 默认 0.02，越大越容易跟随快速变化 |
| `derivative_cutoff` | 默认 2.0 Hz，控制速度估计的响应 |
| `median_window` | 默认 3，必须为正奇数；设为 1 关闭中值处理 |
| `max_speed` | 场景 XY 位置变化上限，默认 1500 mm/s |

### 并行推理

“手部交互预览”使用单批计算，左右手各自保留一个持久 IK 线程，独立维护姿态平滑和 warm start。

“多线程调用测试”左右手由持久线程调度，每手的 Jacobian 分片则由 8 个持久子进程计算。

QT版本中，增加了 XNNPACK 线程数，可在配置文件中设置测试。


## Credits

- [CalciferZh/Minimal-IK](https://github.com/CalciferZh/Minimal-IK)
- [MANO](https://mano.is.tue.mpg.de/)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [Open3D](https://www.open3d.org/)