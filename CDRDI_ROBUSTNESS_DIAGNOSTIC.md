# CDRDI非配准诊断：配准骨干、估计形变、GT形变

入口为 `diagnose_cdrdi_robustness.py`，仅加载现有权重测试，不训练。
每个test patch/case只生成一次LR-HSI，4条路径复用完全相同的观测。
覆盖整个测试集；CAVE默认192个128×128 test patch，而非仅第一张。

| 输出path | 扩散权重 | 重建时采用的几何 | 回答的问题 |
|---|---|---|---|
| registered_backbone | 配准条件训练的Innovation-1权重 | 忽略形变 | 自身配准骨干遇到错位会下降多少 |
| adapted_naive | 非配准适配后的扩散权重 | 忽略形变 | 仅训练适配的贡献 |
| estimated | 与adapted_naive完全相同 | CDRDI估计形变 | 使用形变机制的贡献 |
| oracle | 与estimated完全相同 | 合成观测所用GT形变 | 形变估计是否构成主要瓶颈 |

只有oracle接收GT形变。其他路径只接收LR-HSI和HR-MSI。
不会执行LR-HSI inverse warp。oracle是诊断参考，不保证PSNR一定最高：
现有扩散网络可能适应了估计形变的误差分布，且形变采样和空间退化具有信息损失。

## 推荐运行顺序

使用与你原测试相同的geometry/diffusion/refiner权重。
以下3个大写变量需设置为你的实际checkpoint路径；
`REG_DIFF`必须指向非配准适配之前的Innovation-1权重，不是Stage-2权重。

```bash
REG_DIFF=/path/to/registered_innovation1.pth
GEOM=/path/to/CAVE_geometry.pth
ADAPTED_DIFF=/path/to/CAVE_estimated_deform_diffusion.pth

# 1. 原随机WARP协议；关闭GIGI，先隔离Innovation-2。
python diagnose_cdrdi_robustness.py --dataset CAVE \
  --registered_checkpoint "$REG_DIFF" \
  --geometry_checkpoint "$GEOM" \
  --diffusion_checkpoint "$ADAPTED_DIFF" \
  --geometry_steps 9 --diffusion_steps 12 \
  --protocol random --cases 10 --refiner_mode off

# 2. 同一组固定强度；默认仅1~5档，不要求OOD第6档。
python diagnose_cdrdi_robustness.py --dataset CAVE \
  --registered_checkpoint "$REG_DIFF" \
  --geometry_checkpoint "$GEOM" \
  --diffusion_checkpoint "$ADAPTED_DIFF" \
  --protocol fixed --levels 1 2 3 4 5 --cases 10 --refiner_mode off
```

`--protocol both`一次运行registered + random + 指定fixed档位。
默认每个变形条件每张patch运行10个case，4条路径都会推理；
registered仅运行1次，因为本仓库的物理逆扩散在eval模式下是确定性的。

可先追加 `--max_samples 2 --cases 1`检查权重和接口。
该结果会明确标记为SMOKE，不应作为全测试集指标。
`--dry_run`不导入torch，不读取数据或权重，仅检查CLI并打印计划参数。

## 与最终完整模型指标对照

```bash
FULL_GIGI=/path/to/CAVE_innovation3_gigi_cdrdi_full_k9.pth

python diagnose_cdrdi_robustness.py --dataset CAVE \
  --registered_checkpoint "$REG_DIFF" \
  --geometry_checkpoint "$GEOM" \
  --diffusion_checkpoint "$ADAPTED_DIFF" \
  --refiner_checkpoint "$FULL_GIGI" \
  --protocol random --cases 10 --refiner_mode shared
```

`shared`会额外输出refined指标，base指标仍保存。
所有路径使用同一份冻结GIGI权重、相同的异质性计算和相同残差构造流程，
残差中的前向/伴随算子与该路径的几何假设一致。
registered_backbone/refined是共享校正器的受控诊断，
不代表“配准骨干+其独立训练GIGI”的最佳配准模型。

指定full refiner时，会按现有逻辑从其`extra`读取geometry/diffusion路径；
命令行显式路径优先。未提供registered_checkpoint时，会尝试读取
`init_refiner_checkpoint`所指向的配准GIGI文件中的`extra.baseline_checkpoint`。
无法读取就明确报错，不会偷偷用适配后的扩散权重替代配准骨干。
默认自动兼容standard/legacy Raw-Direct格式，转换后严格检查全部模型参数。

## 两套测试协议

**random**逐字复用原`train_innovation3_gigi_cdrdi.evaluate`的观测生成逻辑：

- `dx,dy ~ U(-max_translation,+max_translation)`；旋转为对应有符号均匀采样。
- 局部B-spline场最大位移在`[0.65,1.0] * max_local_px`中采样，拒绝局部Jacobian不合格的场。
- 每个patch重置设备私有generator到`seed+70000`，复现原测试的case序列。
- **不乘训练期alpha，不混入训练期10%的Identity。** registered单独报告。

**fixed**直接复用`fixed_deformation_protocol.sample_fixed_geometry`，
对应原`test_fixed_deformation.py`同一档位、seed、patch序号与case序号。
dx=dy为固定正值，旋转为固定正角度，局部幅度为该档上限内随机值。
局部场使用CPU/float32合成，保持模型估计边界4/2/4；第6档需显式选择。

两者均采用`Y_H=P0(W_phi(X))`，HR-MSI和GT坐标不动。
全部路径评估完整patch，不裁边，使用仓库`metrics.py`既有6指标，
先平均patch内cases，再对全部patch等权平均。
默认不修改训练和既有测试入口。

## 输出与判断

默认目录为`outputs/cdrdi_robustness/CAVE/<protocol>_<refiner_mode>_seed10_cases10/`。
已有非空目录默认拒绝覆盖；换目录，或明确使用`--overwrite_output`。

| 文件 | 内容 |
|---|---|
| summary.csv | 条件×路径×base/refined的6指标，相对该路径自身registered的PSNR下降/SAM增加 |
| comparisons.csv | 训练适配收益、几何机制收益、总收益、Oracle差距 |
| geometry.csv | EPE、真实位移mean/P95/max、局部最大幅度、naive/estimated/oracle物理闭环MSE |
| summary.json | 权重路径和SHA256、模型参数、协议、完整/部分测试标记、汇总 |
| registered.jsonl / random.jsonl / fixed_N.jsonl | 每个patch/case的观测与局部场SHA256、刚性参数、几何诊断和所有路径指标 |
| test_inputs.json | 全部测试GT/MSI的顺序、shape与SHA256 |

`comparisons.csv`的正数均表示对应方向有利：

- adaptation_gain：adapted_naive相对registered_backbone。
- geometry_gain：estimated相对adapted_naive。
- total_gain：estimated相对registered_backbone。
- oracle_gap：oracle相对estimated；正值表示估计形变仍有改进空间。

PSNR_gain为前者PSNR减后者PSNR；SAM_reduction为后者SAM减前者SAM。
geometry.csv的P95/max先在单case内计算，再对case和patch平均；
不是整个数据集所有像素合并后的全局P95或最坏最大位移。
物理闭环MSE是额外诊断，下降本身不保证重建质量改善。

注意与历史FINAL_REGISTERED的对应关系：

- 历史registered_base = 本脚本registered/adapted_naive/base。
- 历史FINAL_REGISTERED = 本脚本registered/adapted_naive/refined。
- 历史FINAL_WARP = 本脚本random/estimated/refined。

历史REGISTERED绕过几何估计器；本脚本还额外运行registered/estimated，
用于检查几何估计器是否会损伤本已对齐的输入。
判断“相对自身配准下降”时看summary.csv；
判断“相同权重下形变机制的贡献”时看geometry_gain。
不要把二者混为同一个结论。

每完成一个条件就更新汇总，`summary.json.complete=true`表示全部条件完成。
中断运行时已完成条件仍有汇总，当前条件JSONL保留已完成case，不自动续跑。

## 小模型验证

```bash
python -m pytest -q tests/test_cdrdi_robustness_diagnostic.py
```

测试覆盖真实小模型加载、精确估计=Oracle、Identity下Oracle=忽略几何、
原随机输入复现、固定协议输入SHA256一致、全patch遍历和指标差值方向。
