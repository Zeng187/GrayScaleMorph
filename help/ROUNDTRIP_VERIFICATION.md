# Round-Trip Verification Report

**Date**: 2026-03-05
**Branch**: `verify/roundtrip-v2`
**Model**: hemisphere (Loop subdivided to ~500 faces, Platewidth=40mm)

---

## 1. 测试目标

验证 SGN (Sparse Gauss-Newton) 逆向设计框架的正确性：
1. 生成已知可行的材料参数分布 (lambda, kappa)
2. 用这些参数正向模拟，得到变形形状
3. 从扰动的初始参数出发，逆向设计恢复原始参数
4. 比较恢复的参数与真实参数的误差

---

## 2. Round-Trip v1: 均匀参数测试

**方法**: 使用全局均匀的可行参数 (lambda=1.059, kappa=0.100)

### 结果

| 测试 | 描述 | 初始距离 | 最终距离 | Lambda 误差 | Kappa 误差 | 结论 |
|------|------|---------|---------|------------|------------|------|
| A | 固定λ, 恢复κ | 26.57 | 0.055 | - | 14% | ✓ |
| B | 固定κ, 恢复λ | 0.610 | 1.6e-8 | 0.0001% | - | ✓ 完美恢复 |
| C | 交替优化 | 26.8 | 0.007 | 0.05% | 10% | ✓ 单调下降 |

### 发现
- SGN 框架可以成功恢复均匀参数
- Lambda 恢复极其精确 (近乎完美)
- Kappa 恢复较慢 (14% → 10% 误差)
- 距离单调下降，确认框架正确性

---

## 3. Round-Trip v2: 空间变化参数测试 (Per-Vertex Kappa)

**方法**:
- 为每个面根据参数化域中的质心位置 (x, y) 分配离散可行的 (t1, t2) 值
- t1 沿 x 方向变化, t2 沿 (1-y) 方向变化, 形成对角线分布
- 使用材料曲线 compute_lamb_d / compute_curv_d 计算对应的 lambda, kappa
- 所有参数都是严格可行的 (属于 7×7 离散可行集)
- **Kappa 为 per-vertex**: FaceData kappa 通过面积加权平均到 VertexData

**真实参数分布统计**:
- Lambda (per-face): min=1.027, max=1.076, mean=1.056
- Kappa (per-vertex, 面平均): min=-0.095, max=0.096, mean=0.004

### 3.1 Config 1: 默认正则化 (wM_kap=0.1, wL_kap=0.1, MaxIter=20)

| 测试 | 描述 | Lambda relErr | Kappa relErr | Final Distance | 备注 |
|------|------|:------------:|:------------:|:--------------:|------|
| T1 | 无 penalty, 近扰动 (λ+3%, κ×50%) | 0.76% | **48.8%** | 0.0019 | κ 恢复差 |
| T2 | 有 penalty, 近扰动 | 0.73% | **59.0%** | 0.0075 | penalty 使 κ 更差 |
| T3 | 无 penalty, 远扰动 (可行集均值) | 0.56% | **42.9%** | 0.0014 | |

### 3.2 Config 2: 低正则化 (wM_kap=0.001, wL_kap=0.001, MaxIter=50)

| 测试 | 描述 | Lambda relErr | Kappa relErr | Final Distance | 备注 |
|------|------|:------------:|:------------:|:--------------:|------|
| T1 | 无 penalty, 近扰动 | 0.30% | **17.2%** | 7.21e-5 | **26× 距离改善** |
| T2 | 有 penalty, 近扰动 | 0.35% | **43.2%** | 5.37e-4 | penalty 仍然恶化 κ |
| T3 | 无 penalty, 远扰动 | 0.38% | **176.5%** | 2.27e-4 | 远起点 κ 收敛困难 |

---

## 4. Round-Trip v3: Per-Face Kappa 测试

### 4.0 改进动机

Per-vertex kappa 的问题:
- 每个面的能量只"感知" kappa 的三顶点平均值, 无法区分各顶点的贡献
- 梯度在顶点间被平均分配, 降低空间分辨率
- 弯曲能量对 kappa 的灵敏度本身就较低 (缩放 h²/3)

**改进**: 将 kappa 从 VertexData 改为 FaceData, 消除 vertex-to-face 平均。

新增:
- `adjointFunction_FixLam_OptKapPF`: 变量 [x(3|V|), kap(|F|)], 弯曲元素用 `add_elements<19>` (非 21)
- `adjointFunction_FixKap_OptLam2(FD kappa)`: kappa 直接从 FaceData 读取, 无需平均
- 4 个新 SGN solver: `sparse_gauss_newton_{FixLam_OptKap,FixKap_OptLam}[_Penalty](FD, FD)`
- 正则化: face mass matrix + dual graph Laplacian (同 lambda 的处理方式)

### 4.1 Config A: 低正则化 (wM_kap=0.001, wL_kap=0.001)

⚠ **结果极差 — line search 失败, kappa 未收敛**

| 测试 | Lambda relErr | Kappa relErr | Final Distance | 备注 |
|------|:------------:|:------------:|:--------------:|------|
| T1 | 0.88% | **96.93%** | 0.000983 | Line search 失败, 距离暴增后恢复 |
| T2 | 0.30% | **38.62%** | 0.000595 | Penalty 反而有帮助 |
| T3 | 0.42% | **92.37%** | 0.000168 | 距离好但 kappa 几乎没恢复 |

**根因分析**: Per-face kappa 的 Hessian 是对角的!
- 每个面的弯曲元素只包含 **1 个 kappa 变量** (该面自己的)
- 相邻面的 kappa 变量不出现在同一个元素中
- → H_{kap_f, kap_f'} = 0 (无自然耦合)
- 只有 Laplacian 正则化提供面间耦合
- wL_kap=0.001 远远不够

对比 per-vertex: 每个弯曲元素包含 3 个顶点 kappa 变量 (因为平均), 自然耦合通过 Hessian 提供。

### 4.2 Config B: 高正则化 (wM_kap=0.01, wL_kap=0.1) ← 推荐

| 测试 | Lambda relErr | Kappa relErr | Final Distance | 备注 |
|------|:------------:|:------------:|:--------------:|------|
| T1 | 0.22% | **23.26%** | 0.000144 | 稳定收敛, 无 line search 失败 |
| T2 | 0.46% | **42.81%** | 0.002791 | 只跑 1 stage (penalty < threshold) |
| T3 | 0.39% | **32.37%** | 0.000281 | 比 per-vertex 改善! |

### 4.3 Config C: 更高正则化 (wM_kap=0.01, wL_kap=0.5)

| 测试 | Lambda relErr | Kappa relErr | Final Distance | 备注 |
|------|:------------:|:------------:|:--------------:|------|
| T1 | 0.23% | **28.45%** | 0.000300 | 过度平滑, 略差于 Config B |
| T2 | 0.47% | **40.55%** | 0.002736 | |
| T3 | 0.39% | **28.62%** | 0.000430 | 与 T1 相当 |

---

## 5. 方法对比汇总

### Per-Vertex vs Per-Face Kappa (最佳配置对比)

| 方法 | 配置 | T1 κ误差 | T3 κ误差 | T1 距离 | T3 距离 |
|------|------|:--------:|:--------:|:-------:|:-------:|
| Per-vertex | wM=0.001, wL=0.001 | **17.2%** | 176.5% | 7.2e-5 | 2.3e-4 |
| Per-vertex | wM=0.1, wL=0.1 | 48.8% | 42.9% | 1.9e-3 | 1.4e-3 |
| **Per-face** | **wM=0.01, wL=0.1** | 23.3% | **32.4%** | **1.4e-4** | 2.8e-4 |
| Per-face | wM=0.01, wL=0.5 | 28.4% | **28.6%** | 3.0e-4 | 4.3e-4 |

**关键对比**:
- Per-vertex 低正则化: T1 最好 (17.2%), 但 T3 严重发散 (176.5%)
- Per-face 中正则化: T1 略差 (23.3%), 但 T3 明显更好 (32.4%), 且 **无 line search 失败**
- Per-face 方案在不同初始点间表现更一致 (23-32% vs 17-176%)

---

## 6. 关键发现

### 6.1 Lambda 恢复优秀, Kappa 恢复困难

**一致性发现**: 在所有测试中:
- Lambda 相对误差: **0.22% ~ 0.88%** → 优秀
- Kappa 相对误差: **17.2% ~ 96.9%** → 困难

### 6.2 弯曲能量地形平坦

距离很小但 kappa 误差大 (如 T3-PF: 距离 2.8e-4, κ误差 32%), 说明:
- 弯曲能量占比小 (h²/3 ≈ 1/3 缩放)
- 不同的 kappa 分布可以产生相似的形状
- 目标函数对 kappa 不敏感 → 优化难度大

### 6.3 Per-face kappa 需要显著更高的正则化

Per-face kappa 的 Hessian 是块对角的 (无面间耦合):
- wL_kap=0.001: 完全不收敛 (92-97% 误差)
- wL_kap=0.1: 较好收敛 (23-32% 误差)
- wL_kap=0.5: 过度平滑 (28% 误差, 距离变差)

最佳: **wL_kap = 0.1** (100x per-vertex 的值)

### 6.4 Per-face kappa 更稳定但精度稍低

| 特性 | Per-vertex | Per-face |
|------|-----------|---------|
| 最佳 T1 κ误差 | **17.2%** | 23.3% |
| 最佳 T3 κ误差 | 42.9% | **32.4%** |
| T1-T3 一致性 | 差 (17-177%) | **好 (23-32%)** |
| 数值稳定性 | 有 line search 失败 | **无失败** |
| 所需正则化 | wL=0.001 | wL=0.1 |

### 6.5 Penalty 的影响

加入 penalty 后 kappa 误差普遍上升, 因为 penalty 和距离目标竞争。
但 penalty 确保材料可行性, 是实际应用中必需的。
建议: penalty 权重从小开始, 逐步增大。

---

## 7. 可行性诊断 (Hemisphere 真实目标)

| 参数 | 目标范围 | 可行集范围 | 目标在可行集内的比例 | 最大间隙 |
|------|---------|-----------|:------------------:|---------|
| Lambda (per-face) | [0.967, 1.185] | [1.026, 1.093] | **18.5%** | 0.092 |
| Kappa (per-vertex) | [-0.086, -0.026] | [-0.100, 0.100] | **100%** | 0.0 |

---

## 8. 结论与建议

### 确认的事实
1. **SGN 框架正确** — Lambda 恢复近乎完美证明了这一点
2. **Kappa 恢复是核心优化问题** — 弯曲能量地形平坦
3. **Per-face kappa 需要更高正则化** (wL_kap ≈ 0.1) 来补偿 Hessian 的块对角结构

### 当前推荐配置

```json
{
  "wM_kap": 0.01,    // kappa mass matrix 正则化权重
  "wL_kap": 0.1,     // kappa Laplacian 正则化权重 (per-face 需要较高值)
  "wM_lam": 0.00,    // lambda mass matrix 正则化权重
  "wL_lam": 0.1,     // lambda Laplacian 正则化权重
  "wP_kap": 0.01,    // kappa penalty 权重
  "wP_lam": 0.01,    // lambda penalty 权重
  "MaxIter": 50,     // 每阶段 SGN 最大迭代次数
  "betaP": 50        // penalty 锐度参数
}
```

### 下一步
1. 在真实目标 (hemisphere, saddle, hat) 上测试完整逆向设计管线
2. 探索更好的 kappa 优化策略 (如增大 w_b 提高弯曲能量权重)
3. 与 EvolutionCut 集成, 进行分片逆向设计
