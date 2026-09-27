# Augsburg/MDAS 实验协议建议与文献核对

日期：2026-09-27。以下“实测”来自本地数据和代码；“建议”尚未启动训练。

## 结论

建议保留六数据集合成 ×4 主实验，另加 Augsburg 官方模拟 ×3 和真实 MSI 半真实 ×3 两条实验线。现有 MDAS 文件中的 EnMAP 是 HySpex 派生产品，不能写成真实星载 EnMAP 与 Sentinel-2 双星观测实验。也不要将当前 4 个 128×128 test tile 的结果直接对比官方整块 sub_area_1 的数值。

## 文献依据

- **MDAS，Hu et al., ESSD 2023**：同日采集的 HySpex 与 Sentinel-2；SR 基准以派生的 30m HSI、模拟 10m MSI 为输入，以派生的 10m HSI 为参考，划分独立 train/validation/test 地区，比较 CNMF、HySure、SSR-NET、ResTFNet，指标为 PSNR、SAM、ERGAS、Q2ⁿ。该 SR 实验使用模拟 MSI，并不因为数据包含真实 Sentinel-2 就成为真实双传感器实验。[原文 §2.3.4、§3.1](https://essd.copernicus.org/articles/15/113/2023/)
- **Acito et al., JSTARS 2022，PRISMA-SR**：先处理 Sentinel-2 各原生分辨率，再配准、融合；真实案例以相邻日期 L2 观测配对，地理坐标匹配后仍需处理残余配准误差。为几何鲁棒性实验提供相关参考，但并非 Augsburg 实验。[原文](https://arpi.unipi.it/retrieve/e0d6c931-d1a0-fcf8-e053-d805fe0aa794/PRISMA_Spatial_Resolution_Enhancement_by_Fusion_With_Sentinel-2_Data.pdf)
- **Alparone et al., JSTARS 2024**：真实 EnMAP/Sentinel-2B 的 Groningen 案例，同日采集、约一小时差；从 30m 融合到 10m。报告原始 HSI 与回降尺度结果的一致性，以及空间、跨传感器一致性；文中还给出适用于 hypersharpening 的 QNR*。没有可直接充当真实 10m 星载 HSI GT 的图像。[原文 §IV](https://usiena-air.unisi.it/retrieve/2f0099c6-ecae-4dac-b721-21137aaa0c73/Alparone%20et%20al.%20-%202024%20-%20Spatial%20Resolution%20Enhancement%20of%20Satellite%20Hypers.pdf)
- **Cristille et al., 2026，Physics-guided supervision…**：真实 S2/EnMAP 评估使用 Wald 协议、观测一致性及 QNR；其物理模拟监督与本项目 Innovation 1 有直接参考价值。期刊网页直接访问受限，本次依据搜索服务提供的期刊原文索引核对，不声称下载了全文。[期刊原文 §4.3](https://www.tandfonline.com/doi/full/10.1080/19479832.2026.2613383)
- **2024 PRISMA/S2 实验数据发布**：提供配准后真实 PRISMA/S2，以及重采样到相应网格的同期机载 HS 参考。说明严格真实验证可引入独立航空参考，但仍需光谱、空间匹配，不能把不同仪器原始像素直接当同一 GT。[作者数据与说明](https://zenodo.org/records/11547257)

本次没有找到证据支持“Augsburg 上普遍统一采用真实 Sentinel-2 + 真实 EnMAP + 真值”的说法。最直接、可复现的 Augsburg 依据仍是 MDAS 官方基准，其他真实星载案例应注明不同地点和传感器。

## 本地数据已确认的事实

形状以下均为 H×W×C，来自实际读取而非论文标称值。

| 地区 | 10m HSI | 30m HSI | 模拟 10m MSI |
|---|---|---|---|
| deep_train | 540×1371×242 | 180×457×242 | 540×1371×4 |
| deep_valid | 300×639×242 | 100×213×242 | 300×639×4 |
| sub_area_1 | 300×360×242 | 100×120×242 | 300×360×4 |

九个官方 SR TIFF 均可完整读取，同地区三种图像地理边界相同；三个地区互不相交，CRS 为 EPSG:32632。整城真实 Sentinel-2 及 sub_area_1 的真实 Sentinel-2 均可读取，后者为 300×360×12。TIFF 波段说明给出顺序 B1、B2、B3、B4、B5、B6、B7、B8、B8A、B9、B11、B12，因此本项目四通道应读 **1-based `[2,3,4,8]`**，而不是前四个波段。

真实 MSI 的 deep_train/deep_valid 不能按不存在的同名文件读取；应从 `entire_city/Sentinel-2.tif` 按官方 HSI 的 CRS、transform 和 bounds 裁出对应窗口。测试使用 `sub_area_1/Sentinel_2_sub_area1.tif`，并再次核对网格。地理坐标对齐不代表亚像素完全配准。

原始 HySpex 另有 368 波段元数据，不能直接套用 EnMAP 的 242 个波长。现有 SR pipeline 使用 EnMAP-grid 派生 HSI。242 波段的 985→905nm 接缝属于 VNIR/SWIR 重叠，需保留波段对应关系。当前修复的是合成 MSI 积分权重，未将任何真实观测替换成合成数据。

## 三条实验线（本项目建议）

| 实验线 | LR-HSI | HR-MSI | 参考/倍率 | 用途 |
|---|---|---|---|---|
| A：统一合成 | 从 EnMAP-grid 10m 参考按 Innovation 1 生成 | 固定 S2A SRF 合成四通道 | 派生 10m HSI，×4，相当于 40m→10m | 与其余五数据集同一退化设定、受控消融 |
| B：官方模拟 | 官方 `EeteS_EnMAP_30m_*` | 官方 `EeteS_Sentinel_2_10m_*` | 官方 `EeteS_EnMAP_10m_*`，×3 | 可对应 MDAS 基准；不要重新由 10m GT 生成输入 |
| C：真实 MSI 半真实 | 官方 30m 派生 HSI | 实测 Sentinel-2 L2A B2/B3/B4/B8 | 10m 派生 HSI 为 surrogate reference，×3 | 验证跨传感器域差、天然错位和真实 MSI 引导 |

**GT 的准确说法**：A/B 有模拟或派生参考；C 可算相对于 HySpex 派生参考的指标，但不是独立真实星载 HR-HSI 真值。若以后另找真正的 EnMAP/S2 星载观测，通常没有 10m HSI GT，需要无参考/一致性评估，以及额外降尺度验证。

B/C 保留官方地理划分。训练建议 HR patch=96、stride=48，对应 LR 32×32；三个创新和各比较方法共享同一实际观测及有效区。验证测试覆盖完整地区，采用带上下文的滑窗、重叠融合，最终只在原图有效像素评分，padding 不进入指标。不要为了网络要求把官方 30m HSI 插值成“40m 输入”冒充 ×4。

数据以 `/10000` 转到反射率单位，固定预处理。当前合成 loader 为兼容旧主实验会 clip 到 [0,1]，而本地官方文件存在大于 10000 的值；B/C 的专用 paired loader 应保留原始数值并显式记录无效/饱和 mask，不能对 HSI、MSI 各自做逐图 min-max 来消除域差。PSNR 的 data_range（建议 1.0 对应反射率单位）、边界处理和异常值政策须预先固定。

真实 Sentinel-2 的 A/B 平台身份和产品处理基线尚未从现有 TIFF 中确证。当前 S2A V4 是**合成实验的固定定义**；C 线应追溯原始产品元数据，确认具体平台后选相应 SRF。若无法追溯，应报告这一限制，并把响应标定作为 train/validation 上预先固定的敏感性分析。

## Innovation 1–3 的验证安排

以当前代码为准：Innovation 1 是退化一致的渐进扩散；Innovation 2 为 CDRDI 物理残差几何求解；Innovation 3 当前终端方案为 GIGI，结合 MSI 异质性与物理残差。

1. **Innovation 1**：B/C 使用真实提供的 30m HSI 作为终端观测，不在测试时由参考 GT 重新生成。训练阶段在训练区估计/设定 PSF、采样相位和噪声，验证区固定参数。官方 10m/30m 图像分别来自更高分辨率源，不能先验假定“对 10m 图像再应用一次默认退化”可严格复现 30m 观测。先检查算子闭合误差，再比较普通扩散与退化一致扩散。×3 渐进状态需单独验证，不能复用 ×4 checkpoint 冒充新模型。
2. **Innovation 2**：统一输出在 MSI 坐标系，先做所有方法共享的粗地理配准，再让 CDRDI 估计残余错位。天然错位没有已知密集形变 GT，不能报真实 EPE。另设受控错位实验：在高分辨率参考上先形变、后退化形成 LR-HSI，保持 acquisition order；这条线才可报告 EPE、Jacobian folding、不同错位强度曲线。应同时保留“现成配准+fusion”强基线。测试参考图不参与配准、PSF 标定或 checkpoint 选择。
3. **Innovation 3**：在固定 I1+I2 上比较无 refiner、参数量相近卷积、仅异质性、仅物理残差、完整 GIGI；按训练/验证区确定的 MSI 异质性阈值分层统计 SAM/RMSE，展示边界与混合像元误差图、谱形和修正前后观测残差。要同时报告 VNIR 与 SWIR：四个原生 10m S2 通道对 SWIR 没有直接观测约束，不能把视觉锐化当作 SWIR 谱形恢复证据。

建议三个训练种子，例如 10/20/30，固定验证选模规则；冻结超参后执行最终测试。独立地理 test 区只有一个，patch 不能当作独立场景扩大统计显著性。若使用 sub_area_2/3 扩展测试，必须先从训练区剔除它们；本地 bounds 显示它们处在当前 southern training region 内，不能直接追加为独立测试集。

## 评价与 Wald 协议

- **有参考/半真实参考**：PSNR、SAM（角度制）、ERGAS（比例明确为 3 或 4）、Q2ⁿ；可补 SSIM、RMSE。C 线标题须标注 surrogate-reference。Q2ⁿ 与 QNR 是不同指标，不能互换名称。
- **原尺度观测一致性**：报告 `A_phi(X_hat)` 与实际 LR-HSI 的 band-wise RMSE/SAM；报告 `R(X_hat)` 与真实 MSI 的残差/相关性；固定有效区，并明确使用相同还是估计的算子、SRF和几何。天然跨传感器辐射差不应全部归因于融合误差；一致性高也不能证明唯一正确恢复。
- **无参考质量**：可采用与 HSI–MSI/hypersharpening 对应的 QNR*、分项光谱/空间失真及跨传感器一致性。必须披露实现、滤波器、窗口和 MSI 到 sharpened-band 的对应规则；不要随意取某个 MSI 波段充当 PAN 后声称标准 QNR。无参考分数作为辅助，结合误差谱与局部结构图。
- **额外 Wald 降尺度验证（建议）**：将实际/半真实的 30m HSI 再降到 90m，将 10m MSI 降到 30m，执行 ×3 的 90m+30m→30m 重建，以原始 30m HSI 为降尺度参考。两种观测保持共同 footprint 和物理采样相位，PSF 使用组合退化设计；已有传感器模糊不能重复计入。此结果验证尺度一致性，不替代原尺度 10m 真值。实际不可获得高分辨率 HSI 时，报告“原尺度无参考 + 降尺度有参考”两套结果。

当前数据检查没有实现或启动 B/C 的 paired loader、训练、Q2ⁿ/QNR* 计算或最终模型评测；这些是下一阶段工作。
