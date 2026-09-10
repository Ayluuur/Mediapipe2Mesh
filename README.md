# MediaPipe2Mesh

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
| `ball` | 球半径、独立深度、缩放范围和灵敏度 |



## Credits

- [CalciferZh/Minimal-IK](https://github.com/CalciferZh/Minimal-IK)
- [MANO](https://mano.is.tue.mpg.de/)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [Open3D](https://www.open3d.org/)
