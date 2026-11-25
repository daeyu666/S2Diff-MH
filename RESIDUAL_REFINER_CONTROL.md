# 参数量匹配的普通 Residual Refiner 对照

新增 `--variant residual`，支持独立 Registered 和 CDRDI mixed 两个训练/测试入口。
原 `conv` 变体不作为这一项参数量匹配对照；原有 HPGC 变体的网络结构保持不变。

## 对照定义

普通 CNN 直接拼接同样四路输入：扩散重建 X_hat、HR-MSI、R_phy、H_M。
由 3×3 stem、三个 3×3 普通卷积残差块、一个 1×1 普通卷积残差块、3×3 输出头组成。
卷积块使用 GroupNorm/SiLU，包含普通特征残差连接。
输出为 `X_hat + update`；两组共享原来的 tangent 输出投影选项。
输出头与 HPGC 一样零初始化，两者初始输出都严格等于冻结扩散结果。
没有 GCI、自引导/交叉注意力、异质性 query 构造或异质性乘性门控。
H_M、R_phy 都作为普通卷积输入，所有参数真实参与前向/反向，没有占位或冻结参数凑数。

固定网络拓扑，只按 Full HPGC 的参数量自动选择普通卷积通道数，不用验证/测试性能选宽度。
匹配包括所有可训练卷积权重、bias、GroupNorm 参数。默认 PaviaU 配置：

| Refiner | 可训练参数 |
| --- | ---: |
| Full HPGC，hidden=64、heads=4，103 HSI / 4 MSI 通道 | 544,431 |
| Ordinary Residual Refiner，自动匹配 | 544,431 |

对应 CNN 主宽度 69，三个空间块 bottleneck 为 89/88/88，点卷积块 bottleneck 为 136。
`--refine_hidden 64 --heads 4` 对 residual 表示匹配的 **Full 参考配置**，不表示 CNN 必须宽 64。
默认其他五个数据集配置也可精确匹配；非默认小宽度若无法整数精确匹配，
允许最多 0.1% 差异，打印并保存实际参数和误差，超过则报错。
匹配参数量不表示 FLOPs 或有效感受野相同。

## 推荐：补入原 Registered HPGC 消融表

更新仓库后，在 S2Diff-MH 目录运行：

```bash
python run_paviau_hpgc_ablation.py \
  --stage all \
  --variants residual \
  --full_checkpoint ./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full_terminal_correction.pth \
  --diffusion_output terminal_correction \
  --device cuda:0
```

仅训练新增普通 Residual Refiner，再测试 Full 和 Residual，
同时从 Full 测试中取得 w/o Refiner 基础重建指标。不重训 Full，不需要其他消融权重。
默认训练预算、学习率、batch、loss 权重、seed、划分与监控规则均继承 Full 检查点元数据。
普通 refiner 从零训练；冻结相同的扩散权重；两个模型使用相同输出模式和四路输入。
默认四项 loss 与原训练入口一致：L1、SAM、HSI 物理观测损失、MSI 投影损失。
若 Full 还使用过额外初始化/预训练/续训预算，应为普通对照匹配相同训练阶段，
不能把预训练 Full 和只训练一个阶段的普通网络称为相同预算。

冻结扩散路径默认读取 Full 元数据；若权重已移动，可附加：

```bash
--baseline_checkpoint /path/to/the/same/registered_diffusion.pth
```

检查点输出：
`checkpoints/PaviaU/ablation_hpgc/PaviaU_HPGC_Residual_Matched_terminal_correction.pth`。
结果：`results/ablation_hpgc/PaviaU_HPGC_Registered_Ablation_terminal_correction.csv`。
表中加入 `Params_refiner`，包含六项图像质量指标；保留同版本已有的其他消融行。
旧 `state` Full 也支持，需使用该版本 Full 权重并省略模式或传 `--diffusion_output state`。
两种输出模式结果分文件保存，不允许用旧版 Full 作为校正版普通 Refiner 的对照。

训练与测试也可以分开：上述命令将 `--stage all` 分别替换为 `train`、`test`。
参数匹配方案保存在普通检查点的 `refiner_parameter_match` 以及结果 JSON 中。

## 直接调用独立 Registered 训练入口

```bash
python train_innovation3_gigi.py \
  --stage train --variant residual \
  --config_from_checkpoint ./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full_terminal_correction.pth \
  --diffusion_output terminal_correction --device cuda
```

这只继承配置，不加载 Full 的精修权重。默认普通检查点命名为
`checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_residual_terminal_correction.pth`。
测试时使用同样配置和 `--stage test`，或通过 `--refiner_checkpoint` 指定路径。

## Mixed / Warp 场景

`train_innovation3_gigi_cdrdi.py --variant residual` 和共享的 fixed/multiseed 测试入口均支持新模型。
沿用原 mixed 命令中的冻结几何、扩散、训练预算、形变分布和 loss 设置。

如果原 Full 是先训练 Registered，再做 mixed 适配，普通对照也应先训练上面的
Registered Residual，再用它初始化 mixed；不能加载 Full HPGC 权重初始化普通网络。
mixed 默认初始化会寻找同名 `PaviaU_innovation3_gigi_residual_terminal_correction.pth`。
如果使用批量脚本生成的普通检查点，应显式传：

```bash
--variant residual \
--init_refiner_checkpoint ./checkpoints/PaviaU/ablation_hpgc/PaviaU_HPGC_Residual_Matched_terminal_correction.pth \
--diffusion_output terminal_correction
```

也可以 Full/Residual 两组均使用 `--from_scratch`，冻结同一几何/扩散前端重新成组训练。
mixed 的 test 仍同时报告 Registered、Warp，保持同一套 mixed 权重。
正式采用校正输出时，重新从该输出计算 R_phy，代码复用原有物理残差路径。

实验用于检验完整 HPGC 的结构收益；只有测得对应结果后才能陈述性能优势。
不能预先用参数量匹配本身推断 PSNR/SAM 一定更优。
