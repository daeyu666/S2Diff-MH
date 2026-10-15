# 六档固定形变强度测试

两仓库均提供 `test_fixed_deformation.py`，共用逐字节相同的
`fixed_deformation_protocol.py`，协议编号为 `fixed-rigid-random-local-v1`。
这是新增的最终测试，不需要重新训练，也不改变原有随机测试和训练流程。

| 档位 | 固定 dx | 固定 dy | 固定旋转角 | 局部场幅度候选范围 |
|---|---:|---:|---:|---|
| 1 | +0.5 HR px | +0.5 HR px | +0.25° | U(0, 0.5) HR px |
| 2 | +1 HR px | +1 HR px | +0.5° | U(0, 1) HR px |
| 3 | +2 HR px | +2 HR px | +1° | U(0, 2) HR px |
| 4 | +3 HR px | +3 HR px | +1.5° | U(0, 3) HR px |
| 5 | +4 HR px | +4 HR px | +2° | U(0, 4) HR px |
| 6 | +6 HR px | +6 HR px | +3° | U(0, 4) HR px |

平移数值分别作用于x、y两轴，因此纯平移向量长度为表中数值的sqrt(2)倍。
平移、旋转不再随机，也不乘alpha。局部非刚性场由零均值5×5控制点生成
cubic B-spline场，按随机幅度归一化，并拒绝min local Jacobian<0.5的候选场。
局部上限是局部场最大位移范数的上限，实际幅度会写入JSON。
固定刚性参数不随拒绝采样改变。

观测生成沿用 `Y_H=P0(W_phi(X))`：先形变HR-HSI，再物理模糊/面积积分降采样；
HR-MSI及评价GT保持原坐标。固定x4、MTF=0.2、PSF truncate=3、test patch=128。
全图使用border padding，评价完整patch，不裁掉边界。
模型推理仅接收LR-HSI和HR-MSI，不接收GT-HSI或用于合成的真实形变。

第6档属于训练范围外的刚性扰动测试。S2Diff-MH模型自身的几何估计边界仍为
训练时4 px/2°/4 px，不会为测试偷偷扩大。第5、6档复用相同局部场，只改变刚性扰动。

## comparison_experiments

在该仓库根目录运行：

```bash
python test_fixed_deformation.py --model EMR-Diff --dataset CAVE --mode mixed
python test_fixed_deformation.py --model UAFL --dataset CAVE --mode mixed
```

`--mode mixed`默认使用该模型的stage-2混合训练best checkpoint；
`--mode registered`使用stage-1配准训练best checkpoint，但仍测试全部6档扰动，
可用来观察纯配准训练模型的退化。两种模式输出目录分开。
可以用 `--checkpoint /path/to/best.pth.tar` 指定对应阶段的权重。
新方法接入根目录`test_fixed_deformation.py`的`MODEL_BUILDERS`，返回
`test_set, predict(lr_hsi,hr_msi), degrade, device, metadata`；
不要为每个方法复制形变或指标逻辑。模型还应注册到`experiment_protocol.py`。

默认输出：

```text
comparison/<Model>/outputs/fixed_deformation/<Dataset>/mixed_seed10_cases10_eval1234/
```

## S2Diff-MH

在该仓库根目录运行完整CDRDI→扩散→GIGI流程：

```bash
python test_fixed_deformation.py --dataset CAVE --variant full \
  --refiner_checkpoint ./checkpoints/innovation3_gigi_cdrdi/CAVE_innovation3_gigi_cdrdi_full_k9.pth
```

请将refiner路径换成你实际训练完成的权重。脚本优先读取该权重记录的
geometry/diffusion checkpoint路径；可以用原测试参数显式覆盖：

```bash
python test_fixed_deformation.py --dataset CAVE --variant full \
  --geometry_checkpoint /path/to/CAVE_geometry.pth \
  --diffusion_checkpoint /path/to/CAVE_diffusion.pth \
  --refiner_checkpoint /path/to/CAVE_mixed_refiner.pth \
  --geometry_steps 9 --diffusion_steps 12 \
  --geometry_base_channels 32 --diffusion_base_channels 64 \
  --spectral_hidden 8 --refine_hidden 64 --heads 4
```

网络参数需与所加载权重一致；支持noalpha训练所得refiner。
这里没有训练过程，也不使用`--min_strength`对测试形变缩放。
`--eval_cases`是`--cases`的兼容别名。

默认输出：

```text
outputs/fixed_deformation/<Dataset>/full_seed10_cases10_eval1234/
```

## 两边相同的统计与复现设置

- 默认每档、每个test patch随机生成10个局部场（`--cases 10`）。
  `--cases 1`表示每张图每档只测试1个局部场。两边必须使用相同cases、seed和patch列表。
- 每档独立遍历全部test patch，不只测第1张；先平均每张图的cases，再平均全部图像。
  PSNR是逐图指标的算术平均，不是先汇总MSE再转PSNR。
  CAVE固定12个完整test scenes，每个512×512切为16个128×128 patch，共192个patch。
- 局部场使用私有CPU/float32 generator，seed取决于`seed、patch序号、case序号`，
  不受模型初始化、扩散随机数或GPU generator影响。不同patch使用不同局部场。
  相同patch/case跨档采用相同候选控制点和幅度比例；Jacobian拒绝采样可能使部分档位分岔。
- 每个预测使用独立`eval_seed`派生的随机种子，跨档保持一致；不代表不同算法会生成相同扩散噪声。
- 默认`--seed 10 --eval_seed 1234 --control_grid 5 --min_jacobian 0.5`。
  `--levels 1 5 6`可只跑指定档位；默认全部6档。
- `--dry_run`只打印档位及路径，不读取数据或权重；`--output_dir`可指定独立结果目录。
  对比不同权重/网络设置时应使用不同输出目录；重跑同一目录会更新该目录的结果。
- 两边使用相同的既有S2最终测试指标公式：PSNR、SSIM、ERGAS、SAM、CC、RMSE。
  RMSE保存[0,1]域raw值，不乘255。SSIM为原有全局统计简化版。
  保留既有PSNR/RMSE/SAM的clamp01和CC/ERGAS/SSIM的原始预测口径。
  原测试与新测试的形变分布不同，结果应分表标注。

每个输出目录包含：

```text
summary.csv      # 每档1行，包含6个指标、patch数和case数
summary.json     # 同样的6档均值及协议/权重信息
level_1.json     # 每张图/每个case的参数、实际局部幅度、指标
...
level_6.json
```

每完成1档就保存结果与汇总。每个case包含`local_field_sha256`，
每张test图包含GT/MSI的`sha256`及顺序，便于核对两仓库是否使用同一组输入和局部形变。
严格复现应使用相同PyTorch版本及数据文件；版本号记录在JSON中。
