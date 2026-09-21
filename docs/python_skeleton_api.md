## 调用 IK

```python
import json
import numpy as np
from mediapipe2mesh.ik import Keypoints2Mano

converter = Keypoints2Mano(
    'MANO_RIGHT.npz', side='right',
    mirrored_input=True,  # 与送入 MediaPipe 的图像是否镜像一致
)

# world_hand 为 results.multi_hand_world_landmarks 中对应右手的结果。
landmarks = np.array([[p.x, p.y, p.z] for p in world_hand.landmark])
pose_pca = converter.get_mano_params(landmarks)
bones = converter.get_skeleton('camera')

positions_mm = bones.positions           # (16, 3)，关节位置
displacements_mm = bones.displacements   # (16, 3)，相对绑定姿势的位置差
local_rotations = bones.local_rotations  # (16, 3, 3)，相对父关节的旋转
local_quaternions = bones.local_quaternions  # (16, 4)，xyzw
transforms = bones.transforms            # (16, 4, 4)，关节局部 -> 输出空间
json_data = json.dumps(bones.to_dict(), ensure_ascii=False, allow_nan=False)
```

输入必须是 MediaPipe **world landmarks** 的 `(21, 3)` 有限数组，正常单位为米；不要使用归一化图像坐标。

IK 保留骨骼方向，将骨长适配到 MANO。

`get_mano_params()` 默认返回 45 个 PCA 系数，覆盖完整的 15 个手指关节旋转，用于驱动 MANO 运动。

#### 现有异步应用中调用

```python
# 在主循环 state.collect_result() 之后调用。
bones = state.get_skeleton('camera')
if bones is not None:
    joint_positions = bones.positions
    joint_rotations = bones.local_quaternions

# 场景平移已由深度/位置映射更新时，可读取 Open3D 空间数据。
scene_bones = state.get_skeleton('scene')
```

`HandState.get_skeleton()` 返回最近一次主线程已收集的结果，与 `state.vertices`、`state.keypoints` 属于同一次 IK 求解。

场景位置是应用最新的位置估计，姿势是最近完成的 IK 帧。

## 数据

| 字段 | 含义 |
| --- | --- |
| `side` | 实际左右手：`left` / `right` |
| `names` | `W, I0, I1, I2, M0, M1, M2, L0, L1, L2, R0, R1, R2, T0, T1, T2` |
| `parents` | 父关节索引，手腕为 `-1`；I/M/L/R/T 分别为食指/中指/小指/无名指/拇指 |
| `positions` | 输出空间中 16 个关节的位置，mm |
| `rotations`, `quaternions` | 关节局部到输出空间的旋转矩阵、xyzw 四元数 |
| `local_positions` | 父关节坐标系中的平移，mm；手腕为输出空间中的平移 |
| `local_rotations`, `local_quaternions` | 相对父关节的旋转；手腕相对输出空间 |
| `local_axis_angles` | 相同局部旋转的轴角向量，方向为旋转轴，模长为弧度 |
| `transforms`, `local_transforms` | 4×4 列向量齐次变换；前者到输出空间，后者到父关节空间 |
| `displacements` | `positions - rest_positions` |
| `fit_error_mm` | 输出姿势对适配骨长后的 IK 目标的 21 点欧氏 RMS 误差，含平滑影响 |
| `mirrored_local_axes` | 是否连同关节局部坐标轴一起进行了镜像转换 |

层级关系满足 `transforms[parent] @ local_transforms[joint] == transforms[joint]`。

需要帧间位移时，保存上一帧并计算 `current.positions - previous.positions`，两帧须使用相同坐标空间与镜像设置。

MANO 有 16 个旋转关节，另外 5 个指尖是表面关键点。21 点坐标继续通过 `get_camera_oriented_keypoints()` 获取，顺序是上述 16 关节后接食指、中指、小指、无名指、拇指指尖。

`camera` 空间以手腕为原点，x 向图像右、y 向下、z 向远处，单位 mm。`model` 空间保留原生 MANO 模型原点与局部轴。`scene` 对 camera 做 `[x, -y, -z]` 轴变换，再添加应用场景平移。

## MANO 与 PICO 手部映射
MANO 手部由 Wrist + 每根手指3个关节 + 5个指尖 = 21个关键点组成 [MANO 手部关节说明](https://hjj04.github.io/egocentric-paper/MANO_hand_model_explained.html)

PICO 手部由 Palm + Wrist + 大拇指3个关节 + 其他每根手指4个关节 + 5个指尖 = 26个关键点组成 [PICO 手部追踪说明](https://developer.picoxr.com/zh/document/unity/hand-tracking/)


| 部位 | MANO 关节及索引 | PICO 目标关节及编号 | 处理方式 |
| --- | --- | --- | --- |
| 手腕 | W：0 | Wrist：1 | 驱动手部根位置和朝向 |
| 拇指 | T0、T1、T2：13、14、15 | Thumb Metacarpal、Proximal、Distal：2、3、4 | 映射三个旋转关节 |
| 食指 | I0、I1、I2：1、2、3 | Index Proximal、Intermediate、Distal：7、8、9 | 映射三个旋转关节 |
| 中指 | M0、M1、M2：4、5、6 | Middle Proximal、Intermediate、Distal：12、13、14 | 映射三个旋转关节 |
| 无名指 | R0、R1、R2：10、11、12 | Ring Proximal、Intermediate、Distal：17、18、19 | 映射三个旋转关节 |
| 小指 | L0、L1、L2：7、8、9 | Little Proximal、Intermediate、Distal：22、23、24 | 映射三个旋转关节 |
| 指尖 | 拇指20、食指16、中指17、无名指19、小指18 | Thumb Tip 5、Index Tip 10、Middle Tip 15、Ring Tip 20、Little Tip 25 | 由目标骨架前向运动学得到位置；MANO 指尖用于误差检查或接触约束 |
| 掌骨 | 无独立对应节点 | Index、Middle、Ring、Little Metacarpal：6、11、16、21 | 保留绑定局部变换，随父节点运动 |
| 手掌中心 | 无独立对应节点 | Palm：0 | 使用目标模型中的掌心参考变换 |
