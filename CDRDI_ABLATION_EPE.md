# CDRDI deformation estimation 消融：EPE / EPE95

对应脚本 `run_paviau_cdrdi_iteration_ablation.py`：w/o CDRDI、独立训练的
One-shot、递归 CDRDI 及其推理深度诊断。

每个 HR 像素 p 的误差为完整前向采样坐标之差：

`e(p) = sqrt((phi_hat_x(p)-phi_gt_x(p))^2 + (phi_hat_y(p)-phi_gt_y(p))^2)`。

- **EPE ↓**：每个形变 case 内所有像素 e(p) 的均值。
- **EPE95 ↓**：每个形变 case 内 e(p) 的第 95 百分位，使用线性插值分位数。
- 单位统一为 **HR pixels**，不是归一化 grid 坐标，也不是 LR 像素。
- 比较完整的刚性+局部组合形变，包括平移、绕图中心旋转和局部位移；
  使用 `cdrdi_geometry.sampling_coordinates` 的相同组合顺序。
- 在完整 HR 测试块上计算，采样坐标不截断到图像边界。
- 先在每个 case 内统计，再对同一块的 cases 求均值，最后对测试块等权平均。
  表里的 EPE95 是逐 case 的 95 分位数平均，**不是所有 case 像素合并后的 95 分位数**。

旧版本已经将相应数值保存为 `EPE_mean_px`、`EPE_p95_px`，但主表只显示 PSNR/SAM。
现在主表和终端输出都包含 `Variant, PSNR, SAM, EPE, EPE95`。
推理、形变抽样、原 PSNR/SAM 的计算与汇总不变。
GT 形变仅供评价，不输入估计网络。

## 已有实验直接补表

将 output_dir 指向以前运行该消融实验生成的目录，其中应有完整 `manifest.json`：

```bash
python run_paviau_cdrdi_iteration_ablation.py \
  --summarize_only \
  --output_dir ./results/ablation_cdrdi_iteration/PaviaU_independent_oneshot
```

该命令只读取原 manifest 中已计算的指标，重新写入 `summary.csv`，
附加 `epe_metric_definition.json` 说明指标口径。不加载权重、数据，不执行推理，
不修改原 manifest、per_patch.csv 或 cases.jsonl。
本地仍需安装脚本正常导入所需的 Python/PyTorch 依赖，不需要 CUDA。
缺失 EPE 原始记录或 manifest 不完整时会报错，不会生成替代数值。

重新执行完整实验时沿用原命令，主表会自动包含新增两列，无需额外参数。
该更新仅补充几何评价，不改变已有消融所采用的扩散输出模式。
