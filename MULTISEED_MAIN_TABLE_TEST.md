# 主表：固定权重下的10个形变种子 × 每种子10个case

两个仓库的 `multiseed_deformation_protocol.py` 保持逐字相同，
协议号为 `main-table-random-geometry-10x10-v1`。
这是测试协议升级，不需要重新训练，不改变训练、验证或模型选择。

| 主表 | 权重 | 测试 | 报告 |
|---|---|---|---|
| 表1 | 配准训练权重 | 配准观测 | 单组指标，沿用原配准测试 |
| 表2 Registered | 混合训练权重 | 配准观测 | 单组指标，每个patch测1次 |
| 表2 Warp | 同一份混合训练权重 | 随机形变观测 | 10个形变测试种子的mean±std |

## comparison_experiments

原有按模型名/数据集切换的根入口保留：

```bash
# 表1：原配准训练、配准测试。
python test.py --model EMR-Diff --dataset CAVE --mode registered

# 表2：同一混合权重的Registered单值 + Warp的10×10 mean±std。
python test.py --model EMR-Diff --dataset CAVE --mode mixed
python test.py --model UAFL --dataset CAVE --mode mixed
python test.py --model CLSNet --dataset CAVE --mode mixed
```

模型沿用既有架构和checkpoint检查；CLSNet保持当前数据集专属宽度/注意力头。
数据集支持PaviaU、Houston13、Chikusei、CAVE、Botswana、Augsburg（合成x4）。
真实Augsburg的Wald/native入口保持原状，不应用此合成形变协议。

mixed默认从`comparison/<Model>/checkpoints/hsi_warp_final/<Dataset>/best.pth.tar`加载。
仍可显式指定权重和数据目录：

```bash
python test.py --model EMR-Diff --dataset CAVE --mode mixed \
  --checkpoint /path/to/mixed_best.pth.tar --data_root ./data/raw
```

也可直接调用新入口：

```bash
python test_multiseed_deformation.py --model CLSNet --dataset CAVE --mode mixed
```

旧单形变种子结果可通过`test.py ... --mode mixed --legacy_single_seed`复现。
原各模型的`test_hsi_deformed.py`和固定强度`test_fixed_deformation.py`仍保留。
新方法只需在已有`test_fixed_deformation.MODEL_BUILDERS`中添加模型适配器，
新主表测试会复用该适配器；根入口还需注册`experiment_protocol.ModelAdapter`。
无需为每个模型复制形变或统计代码。

## S2Diff-MH

在仓库根目录使用当前最终测试的同一组geometry/diffusion/refiner权重：

```bash
python test_multiseed_deformation.py --dataset CAVE --variant full \
  --refiner_checkpoint ./checkpoints/innovation3_gigi_cdrdi/CAVE_innovation3_gigi_cdrdi_full_k9.pth
```

refiner文件名以实际混合训练权重为准。geometry/diffusion路径优先从其`extra`
读取，也支持显式覆盖，网络尺寸参数应与当前最终测试一致：

```bash
python test_multiseed_deformation.py --dataset CAVE --variant full \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/CAVE_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_checkpoint ./checkpoints/cdrdi_stage2/CAVE_estimated_deform_diffusion_k9_60ep.pth \
  --refiner_checkpoint /path/to/CAVE_mixed_refiner.pth \
  --geometry_steps 9 --diffusion_steps 12 \
  --geometry_base_channels 32 --diffusion_base_channels 64 \
  --spectral_hidden 8 --refine_hidden 64 --heads 4
```

Registered沿用原`FINAL_REGISTERED`：同一份适配扩散/GIGI权重，绕过形变估计器。
Warp使用CDRDI估计形变→形变感知扩散→终端GIGI。
本入口不加载GT形变到网络，也不加载另一份配准骨干。
表1仍使用原配准模型测试入口。

## 固定的新测试分布

- patch为128×128，scale=4，物理MTF=0.2，PSF truncate=3。
- `dx,dy`独立候选`U(-4,+4)` HR px；旋转`U(-2,+2)`度。
- 零均值5×5控制点生成cubic B-spline局部场，最大局部位移范数候选`U(0,4)` HR px。
- 拒绝局部Jacobian<0.5的候选场；拒绝局部场时保持已抽取的刚性参数不变。
- 不混入10%的Identity，不乘训练alpha；Registered另行评估。
- 全部方法均先warp HR-HSI，再物理退化：`Y_H=P0(W_phi(X))`。
  HR-MSI与GT保持原坐标，评估完整patch，不裁边。

**新协议统一使用0～4局部幅度候选。旧S2随机测试使用0.65～1倍上限，
旧对比测试使用0～1倍上限。新旧均值不能直接混用。**
旧入口不变，训练/验证的原抽样分布也不变。

## 种子、输入配对和统计

默认形变种子为`10001 10002 10003 10004 10005 10006 10007 10008 10009 10010`，
每个种子每个patch有10个case。形变仅依赖协议号、形变种子、patch序号、case序号，
由私有CPU/float32 generator生成，不依赖模型初始化或推理噪声。
不同patch不再复用相同的10套形变。

`--seed`保留为模型/数据初始化种子；**不作为10个形变种子**。
`--eval_seed 1234`控制重建随机数，按patch/case派生，跨形变种子保持不变。
随机模型因此使用固定的重建噪声案例表；不会把每个形变种子的变化与另一组噪声变化混在一起。
Registered对每个patch使用一次固定重建种子，不报告std。

每个形变种子先平均patch内cases，再平均全部patch，得到完整测试集指标。
最终对10个完整测试集指标求均值与样本标准差（`ddof=1`）。
不是对100个单case计算std，不是对所有像素计算std，也不是10次训练。
std仅反映固定测试集和固定权重下形变采样的波动。
单种子冒烟检查的std写为null/NA，不能假装估计出0波动。

CAVE默认192个test patch，Warp共192×10×10=19,200个观测实例。
它们包含相同场景的不同形变，不是19,200个独立场景。

两边必须使用同样的形变种子列表、cases、eval_seed、PyTorch版本和数据文件。
`manifest.json`记录GT/MSI顺序与SHA256；逐case记录局部场和LR-HSI的SHA256，
用来核对输入是否真正一致。相同seed不能替代数据/输入核对。

## 输出与续跑

默认输出到各仓库/模型的`outputs/multiseed_deformation/<Dataset>/.../`。
目录名包含种子列表指纹、种子数、case数和eval_seed；不覆盖旧测试结果。

| 文件 | 内容 |
|---|---|
| summary.csv | Registered单值；Warp各指标mean/std列，Registered的std留空 |
| summary.json | 完成标记、Registered、Warp的mean/std、每个种子的结果 |
| per_seed.csv | 每个种子的完整测试集均值 |
| manifest.json | 完整协议、权重SHA256、数据指纹、噪声与形变种子规则 |
| registered.json | 配准结果及逐patch指标 |
| seed_10001.json | 已完成种子的均值和逐patch case均值 |
| seed_10001.jsonl | 每个patch/case的刚性参数、局部场幅度/Jacobian、观测指纹、指标 |

每完成1个种子就保存汇总；`summary.json.complete=true`表示全部完成。
中断后使用相同命令追加`--resume_output`，会验证权重/数据/协议，跳过已完成种子；
未完成种子从头运行。`--overwrite_output`明确丢弃旧结果并重跑，不能与resume合用。

两边均支持以下选项：

```bash
--geometry_seeds 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010
--cases 10
--eval_seed 1234
--output_dir /path/to/results
--resume_output
```

先用`--max_samples 2 --geometry_seeds 10001 10002 --cases 1`做小规模检查。
该结果明确标记SMOKE，不能作为主表；正式测试删掉这些限制。
`--dry_run`只打印参数，不加载数据和checkpoint。
固定强度鲁棒性图继续使用原固定测试入口，不受这次改动影响。
