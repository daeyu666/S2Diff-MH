# Single-step Direct / Generic Cold：相同骨干的递推范式对照

运行入口：`train_reconstruction_controls.py`。
只研究配准条件下 Innovation-1 的基础重建，不接入 CDRDI、HPGC-Refiner。
这两项需要从零训练，不能把物理扩散内部的 Raw-MSI Direct predictor
结果（如之前的 45.019 dB）当作 Single-step Direct 的结果。

## 固定项

三个模式都直接实例化原 `RawMSIDirectPredictor`，不增加或删除网络层：
`base_channels=64, spectral_hidden=8, time_dim=256, T=12, dropout=0`。
默认 seed=10；相同种子下参数初值逐元素一致，参数量完全一致。
Single-step 保留时间嵌入及其参数，但只输入固定 t=T，没有扩散状态采样。

观测固定为同一批 `Y_H=P_phys(X)`、`Y_M=R0(X)`，
并共用 `q=U_phys(Y_H)` 作为 HR 尺寸的初始输入。
`P_phys` 沿用当前物理 ×4、MTF Nyquist=0.2、PSF truncate=3.0；
`U_phys` 默认是已有归一化伴随回投。这里的 MTF 是模拟假设，不能写成逐传感器实测标定。
Generic **不会**用自己的 Gaussian+Bicubic 另生成一套输入 LR-HSI。

数据集拆分、归一化、SRF、增强均直接复用 `data_loader.py`。
PaviaU 仍为独立验证区域和正式中心 128×128 测试块；
CAVE 仍为 16/4/12 场景，测试 12×16=192 个非重叠 128×128 块。
训练块 64×64、stride=32；相同 seed 下逐 epoch 对齐训练 shuffle 和增强。
Cold 时间步使用独立随机流，不会改变数据增强；时间步采样规则和尺度阶段一致。

默认固定预算：200 epoch、batch=4、Adam、lr=1e-4、weight_decay=0、grad_clip=1，
没有学习率调度。每个训练 batch 均只调用一次网络、进行一次优化更新。
损失为相同的 `L1 + 0.1*SAMLoss`，SAM 使用已有弦长/反弦长形式（弧度）。
默认 `lambda_deg=0`。若显式启用 lambda_deg，三组都使用相同终端物理观测 L1，
不会让 Generic 和 Physical 分别使用不同的中间退化正则。
这项可选正则与旧 `main.py` 的中间状态 lambda_deg 语义不同，默认两者均关闭。

## A：Single-step Direct Reconstruction

`pred = F_theta(q, Y_M, T)`。
训练和推理均只调用一次骨干，不构造扩散前向训练状态，不执行反向递推，
不执行终端残差校正。物理观测生成及固定回投只是与其他组一致的输入预处理。
预测目标仍是干净 HR-HSI X。

## B：Generic Cold Diffusion（端点匹配版本）

这是受控实验定义，不声称复现某篇 Cold Diffusion 论文的原始完整配置。
为真正只改变中间轨迹，显式共享两端：

- t=0：`G_0(X)=X`；
- t=1,...,T-1：普通高斯模糊（sigma_t=2*t/T、固定 5×5 核）
  + 抗混叠 bicubic 下采样 + bilinear 上采样至 HR 网格；
- t=T：共享 `G_T(X)=L_T^physical(X)=U_phys(P_phys(X))`。

尺度阶段仍为 1→2→4、T=12，与物理组一致。
Generic 的中间状态没有 MTF→PSF 标定、探测器像元积分或物理伴随算子。
**公共终端边界仍使用物理端点适配器**，这是为了固定两组的终端观测和终端算子；
不能将此对照描述为“所有时间步都完全不接触物理算子”。
在 t=T，模型接触到的观测初始状态与物理组逐元素相同。

训练锚定规则在两种 Cold 中完全一致（只采样 t>=1）：

```
delta = q - G_T(X)
x_t_train = G_t(X) + delta
```

配准模拟数据中公共观测与公共算子闭合，delta=0；实现仍保留观测锚定，
并测试了非零残差的情况。t=T 训练输入严格为 q。
反向差分和正式候选输出也相同：

```
x_(t-1) = x_t + G_(t-1)(pred_t) - G_t(pred_t)
delta_hat = q - G_T(pred_1)
output = x_0 - delta_hat
```

默认 `--diffusion_output terminal_correction`，也支持 `state` 复核旧输出。
推理接口仅接收 LR-HSI、HR-MSI，不能使用 GT；上式没有 GT 泄漏。
Oracle 对每步预测 X 时，非零 delta 也满足校正后输出 X。
两组都使用同一公共 G_T 进行校正；不会让 Generic 独自承担终端算子失配。

## PaviaU 训练

在 S2Diff-MH 目录中运行。下面两条均从零训练，不需要预训练权重：

```bash
python train_reconstruction_controls.py --stage train --experiment single_step \
  --dataset PaviaU --data_root ./data/raw --base_channels 64 --spectral_hidden 8 \
  --diffusion_steps 12 --epochs 200 --batch_size 4 --lr 1e-4 \
  --lambda_l1 1.0 --lambda_sam 0.1 --seed 10 --device cuda

python train_reconstruction_controls.py --stage train --experiment generic_cold \
  --dataset PaviaU --data_root ./data/raw --base_channels 64 --spectral_hidden 8 \
  --diffusion_steps 12 --diffusion_output terminal_correction \
  --epochs 200 --batch_size 4 --lr 1e-4 \
  --lambda_l1 1.0 --lambda_sam 0.1 --seed 10 --device cuda
```

同入口还提供 `--experiment physical_cold`，便于严格成组重跑现有物理扩散基准。
它的中间轨迹调用现有物理过程，其他所有设置与 Generic 组共用。
不要把经过 CDRDI/HPGC 的完整模型指标直接当作这一组的基础扩散指标。
若旧基准训练预算不是上述配置，训练前须把各组预算统一为实际基准配置。
两种 Cold 的输出模式也应一致。

可选统一早停：`--early_stop_patience 10 --early_stop_min_delta 0.02`，
计数单位是验证次数而非 epoch；三组要一起设置。默认 0 表示固定预算不早停。
验证间隔默认 5，checkpoint 仅按验证 PSNR 选择；训练过程不评估测试集。

## 测试与汇总

```bash
python train_reconstruction_controls.py --stage test --experiment single_step \
  --dataset PaviaU --data_root ./data/raw --device cuda

python train_reconstruction_controls.py --stage test --experiment generic_cold \
  --dataset PaviaU --data_root ./data/raw \
  --diffusion_output terminal_correction --device cuda
```

自动加载相应实验的 best；默认路径：

```
checkpoints/reconstruction_controls/PaviaU/PaviaU_single_step_direct_seed10.pth
checkpoints/reconstruction_controls/PaviaU/PaviaU_generic_cold_terminal_correction_seed10.pth
```

非默认宽度、步数、seed、训练损失等设置在测试时应与训练一致，
协议不匹配会明确报错；`--checkpoint` 可指定同协议的其他检查点。
续训使用 `--resume ..._last.pth --epochs 新的总轮数`，保留原其他参数。
新实验目录独立，不会覆盖现有扩散或 Refiner 的 checkpoint。

每次测试写入 `outputs/reconstruction_controls/summary.csv`，保留其他已测试实验的行。
记录六项图像指标、参数量、网络推理调用次数、checkpoint epoch/更新次数、
权重路径和哈希、输入观测哈希。三组测试的 input_sha256 应相同。
详细 JSON 在 `outputs/reconstruction_controls/<数据集>/<实验名>/test_metrics.json`。
所有指标复用原 `metrics.py`，按测试块平均；不绘制或虚构尚未运行的实验结果。

CPU 验证：

```bash
python -m pytest -q tests/test_reconstruction_controls.py
```

覆盖骨干初值、参数量、公共观测/端点、中间算子隔离、单次调用、
非零残差 oracle、数据增强随机流对齐、三组真实训练/验证/测试、指标汇总及续训复现。
