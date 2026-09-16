# Python 骨骼数据接口

Python 的预览和交互入口均使用此接口。Qt 移植目录没有改动。

## 直接调用 IK

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
displacements_mm = bones.displacements   # (16, 3)，相对当前朝向下绑定姿势的位置差
local_rotations = bones.local_rotations  # (16, 3, 3)，相对父关节的旋转
local_quaternions = bones.local_quaternions  # (16, 4)，xyzw
transforms = bones.transforms            # (16, 4, 4)，关节局部 -> 输出空间
json_data = json.dumps(bones.to_dict(), ensure_ascii=False, allow_nan=False)
```

输入必须是 MediaPipe **world landmarks** 的 `(21, 3)` 有限数组，正常单位为米；不要使用归一化图像坐标。IK 保留骨骼方向，将骨长适配到 MANO，因此不会恢复使用者的实际手掌尺寸。输入中的整体平移会被移除。缺失骨段、共线手掌和 NaN 会抛出 `ValueError`，保留上一份有效结果。

`get_mano_params()` 默认返回 45 个 PCA 系数，覆盖完整的 15 个手指关节旋转。需要旧输出维数时可显式传入 `n_pose=17`，但会限制手势拟合能力。PCA 系数不是关节角度；技术对接应使用骨骼接口。

## 现有异步应用中调用

```python
# 在主循环 state.collect_result() 之后调用。
bones = state.get_skeleton('camera')
if bones is not None:
    joint_positions = bones.positions
    joint_rotations = bones.local_quaternions

# 场景平移已由深度/位置映射更新时，可读取 Open3D 空间数据。
scene_bones = state.get_skeleton('scene')
```

`HandState.get_skeleton()` 返回最近一次主线程已收集的结果，与 `state.vertices`、`state.keypoints` 属于同一次 IK 求解。返回的数据是独立副本。尚未得到结果、跟踪重置或手部退场后返回 `None`。`scene` 还要求 `state.scene_translation` 已建立；未建立时返回 `None`。不要从主线程直接读取仍在后台求解的 `state.converter`。

场景位置是应用最新的位置估计，姿势是最近完成的 IK 帧，遵循现有渲染流程。独立调用方可通过 `bones.to_scene(wrist_translation_mm)` 指定场景中手腕的位置；该参数必须与绘制网格时使用的平移相同。

## 数据约定

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
| `displacements` | `positions - rest_positions`，不是相邻帧差分 |
| `fit_error_mm` | 输出姿势对适配骨长后的 IK 目标的 21 点欧氏 RMS 误差，含平滑影响 |
| `mirrored_local_axes` | 是否连同关节局部坐标轴一起进行了镜像转换 |

层级关系满足 `transforms[parent] @ local_transforms[joint] == transforms[joint]`。需要帧间位移时，保存上一帧并计算 `current.positions - previous.positions`，两帧须使用相同坐标空间与镜像设置。

MANO 有 16 个旋转关节，另外 5 个指尖是表面关键点，没有独立旋转自由度。21 点坐标继续通过 `get_camera_oriented_keypoints()` 获取，顺序是上述 16 关节后接食指、中指、小指、无名指、拇指指尖。

`camera` 空间以手腕为原点，x 向图像右、y 向下、z 向远处，单位 mm。`model` 空间保留原生 MANO 模型原点与局部轴。`scene` 对 camera 做 `[x, -y, -z]` 轴变换，再添加应用场景平移。MediaPipe world landmarks 的原点在手附近，不能据此获得相机中的绝对手腕位置。[MediaPipe 坐标说明](https://chuoling.github.io/mediapipe/solutions/hands.html#multi_hand_world_landmarks)

## 镜像与左右手

两个应用默认使用 `handedness_map="auto"`。交互配置的 `camera.mirror=false` 保持不变，此时必须交换 MediaPipe 的左右标签后再选 MANO 模型。此前的 `false + direct` 组合会选中相反侧模型，导致手掌与弯曲方向不匹配。自定义配置也应使用 `auto`；只有明确校准过输入源的标签约定时才使用 `direct` 或 `swapped`。

保留原有 `camera.mirror` 图像翻转、`handedness_map` 和左右 MANO 模型选择。两个 Python 应用把 `camera.mirror` 传入 IK 的 `mirrored_input`，默认仍为 `True`。MediaPipe 的 handedness 默认假设输入为镜像图像；非镜像时 `auto` 会交换标签。[MediaPipe handedness 说明](https://chuoling.github.io/mediapipe/solutions/hands.html#multi_handedness)

IK 在求解前解除输入 x 镜像，完成求解后恢复显示镜像。镜像输出的关节局部轴也相应镜像，使旋转矩阵保持行列式 +1，可以合法转换成四元数。要给原生 MANO 骨架直接赋局部旋转时使用 `get_skeleton('model')`；camera/scene 的数据用于相应空间中的骨架。

镜像网格会改变三角面绕序。自定义渲染时使用 `get_faces(camera_oriented=True)` 搭配 `get_camera_oriented_vertices()`；原生 `get_vertices()` 搭配默认 `get_faces()`。

## 姿势恢复修正及验证

关节位置现在来自前向运动学变换，不再由变形后的表面二次回归；蒙皮前应用模型的 `posedirs` 姿态修正，遵循 [MANO 参考实现](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py)。求解使用完整姿态空间、逐骨段最小旋转初值和固定的平直姿态正则，降低欠约束的轴向扭转。LM 只接受降低完整目标函数的更新，最终模型状态与返回参数一致。导数计算只蒙皮指尖，最终输出时再计算完整网格。

验证命令：

```powershell
conda run -n media2mano python -m unittest discover -s tests -v
```

新增测试覆盖真实 MANO 模型的 FK/蒙皮公式、左右手、镜像开关、手掌整体旋转、平直/弯曲姿势、骨骼层级重建、四元数、JSON、异步结果和非法输入。合成已知姿势测试中，镜像左手弯曲姿势的 21 点 RMS 误差从原流程约 38–39 mm 降到约 1.2–2.6 mm；右手相同测试从约 4.4–12.3 mm 降到约 1.2–2.0 mm。参数为 5 次迭代、关闭姿态平滑；这是算法回归数据，不是实拍精度测量。

21 个位置不能唯一确定每根骨骼的轴向扭转；接口导出的是 IK 估计的旋转。真实摄像头下仍需检查张手、握拳、捏合、侧掌和翻掌，重点观察遮挡时的 MediaPipe 深度误差与个体手型差异。
