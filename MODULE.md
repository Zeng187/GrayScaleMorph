# GrayScaleMorph 模块文档

## 概述

材料-形状编程内核，提供 4 个可执行入口 + 1 个共享静态库（`grayscale_core`）。给定目标 3D 形状，通过灰度复合材料参数化、逆设计、正向验证、可行性评估等独立步骤完成从平板到目标形状的变形设计。

**参考论文**：
- **Shrink & Morph (TOG 2023)**：主要框架——非欧薄壳能量、参数化、逆设计目标函数
- **SGN (TOG 2021)**：稀疏 Gauss-Newton 求解器——bilevel 优化加速
- **Physical Simulation of Induced Thin Shell (TOG 2018)**：离散薄壳理论基础

## 可执行入口

| 可执行文件 | 源文件 | 功能 | CLI |
|-----------|--------|------|-----|
| **Parameterize** | `parameterize/main.cpp` | 3D 网格 → 2D 参数化 + 目标形状 | `./Parameterize [--cfg cfg.json]` |
| **Inverse** | `inverse/main.cpp` | 单 patch 逆设计（调试用） | `./Inverse`（读 `inverse_cfg.json`，需 `patch.id ≥ 0`） |
| **InverseWhole** | `inverse_whole/main.cpp` | 全 patch 逆设计 | `./InverseWhole`（读 `inverse_whole_cfg.json`） |
| **Forward** | `forward/main.cpp` | 参数化网格 + 材料 → Newton 正向求解 → 平衡形状 | `./Forward --cfg cfg.json` 或显式文件模式 |
| **Evaluate** | `eval/main.cpp` | 可行性评估或逆设计误差 | `./Evaluate --cfg cfg.json --feas` 或 `--inverse` |

> **兼容性说明**：CMake 仍生成 `GrayScaleMorph` 可执行文件作为 `Inverse` 的向后兼容别名。

### Inverse / InverseWhole（三阶段 Pipeline）

直接读取 `segmentDir()/patches/patch_*.obj`，不依赖 `seg_id.txt`。

**Phase 1**：读所有 patches → linearSubdivide（按需）→ parameterizeMesh（仅 gauge shift P，V 不变）
**Phase 2**：PCA 旋转每个 P（最长轴 → x 轴）→ `scale_i = platewidth / P_extent_i` → `globalScale = min(scale_i)`
**Phase 3**：V × globalScale, P × globalScale → SGN 逆设计 → 材料投影 → forward 验证（rigid-aligned）→ 输出

- **Inverse**：仅处理 `patch.id` 指定的 patch（调试单个 patch），但读取所有 patches 以计算一致的 globalScale
- **InverseWhole**：处理所有 patches

> **缩放逻辑**：globalScale 基于 P（物理板材尺寸）而非 V（目标形状尺寸），确保 platewidth 约束的是实际板材。所有 patch 共享同一 globalScale，h·κ 物理意义全局一致。

### Parameterize

独立参数化工具：Tutte → ASAP → gauge shift（仅缩放 P）。

- **Whole-mesh 模式**（无 seg_id.txt）：输出 `{model}_param.obj` + `{model}_targ.obj`
- **Per-patch 模式**（有 seg_id.txt）：输出 `patch_{pid}_param.obj` + `{model}_targ.obj`

> **注意**：Parameterize 是独立工具，不含逆设计的三阶段缩放逻辑。Inverse/InverseWhole 内部已集成参数化步骤。

### Forward

从平板初始状态执行 Newton 正向求解，给定材料分配 (t1, t2) 恢复 3D 平衡形状。

- **显式文件模式**：`--param`, `--material`, `--curves`, `--output` 全部指定
- **Config 模式**：从 `forward_cfg.json` 派生路径

### Evaluate

两种评估模式：

- `--feas`：参数化 + 目标场计算 + StVK 残差评估，报告材料可行性超限
- `--inverse`：完整逆设计 + 形状误差度量（`dist_inv`, `dist_proj`）

输出为 JSON（stdout）。

## 输入/输出

### 路径约定

```
{modelDir} = {model.name}[_{segment.method}][_{segment.plan}]
例: name=cup, method=original, plan=planA_jigsaw → modelDir = cup_original_planA_jigsaw
```

所有 `paths.*` 字段仅存储基础目录（如 `../Resources/design/`），代码通过 `modelDir()` 自动拼接 `{base_path}/{modelDir}/`。

### Inverse / InverseWhole 输入输出

| 方向 | 文件 | 路径 |
|------|------|------|
| **输入** | patch 网格 | `Resources/segment/{modelDir}/patches/patch_{pid}.obj` |
| **输入** | 材料曲线 | `Resources/materials/poly-curves.json` |
| **输出** | 2D 参数化 | `Resources/param/{modelDir}/patch_{pid}_param.obj` |
| **输出** | 3D 目标（globalScale） | `Resources/morph/{modelDir}/patch_{pid}_target.obj` |
| **输出** | forward 验证形状（rigid-aligned） | `Resources/morph/{modelDir}/patch_{pid}_proj.obj` |
| **输出** | per-face λ/κ excess | `Resources/morph/{modelDir}/patch_{pid}_metrics.txt` |
| **输出** | per-face 材料 (t1, t2) | `Resources/design/{modelDir}/patch_{pid}_material.txt` |
| **输出** | 固定顶点索引 | `Resources/cond/{modelDir}/patch_{pid}_bound_center.txt` |

### Forward 输入输出

| 方向 | 文件 | 路径 |
|------|------|------|
| **输入** | 参数化网格 | `Resources/param/{modelDir}/{model}_param.obj` |
| **输入** | 材料分配 | `Resources/design/{modelDir}/{design_name}.txt` |
| **输入** | 材料曲线 | `Resources/materials/poly-curves.json` |
| **输入** | 边界条件（可选） | `Resources/cond/{modelDir}/{cond_name}` |
| **输出** | 平衡形状 | `Resources/forward/{modelDir}/{design_name}_forward.obj` |
| **输出** | per-face 诊断 | `Resources/forward/{modelDir}/{design_name}_forward.vtk` |

### Parameterize 输入输出

| 方向 | 文件 | 路径 |
|------|------|------|
| **输入** | 目标网格 | `Resources/meshes/{model}.obj` |
| **输入** | 分割（可选） | `Resources/segment/{modelDir}/seg_id.txt` |
| **输入** | 材料曲线 | `Resources/materials/poly-curves.json` |
| **输出** | 参数化网格 | `Resources/param/{modelDir}/{model}_param.obj` 或 `patch_{pid}_param.obj` |
| **输出** | 目标参考 | `Resources/param/{modelDir}/{model}_targ.obj` |
| **输出** | distortion 诊断 | `Resources/param/{modelDir}/*_distortion.txt` + `*_colored.ply` |

### Evaluate 输入输出

| 方向 | 文件 | 路径 |
|------|------|------|
| **输入** | 目标网格 | `Resources/meshes/{model}.obj` |
| **输入** | 分割（可选） | `Resources/segment/{modelDir}/seg_id.txt` |
| **输入** | 材料曲线 | `Resources/materials/poly-curves.json` |
| **输出** | JSON | stdout |

## cfg.json 格式

```json
{
  "model": {
    "name": "cup",
    "mesh_path": "../Resources/meshes/cup.obj"
  },
  "material": {
    "curves_path": "../Resources/materials/grayscale-material-poly-curves.json"
  },
  "segment": {
    "path": "../Resources/segment/",
    "method": "original",
    "plan": "planA_jigsaw"
  },
  "paths": {
    "param_path": "../Resources/param/",
    "morph_path": "../Resources/morph/",
    "design_path": "../Resources/design/",
    "cond_path": "../Resources/cond/"
  },
  "solver": {
    "platewidth": 80.0,
    "max_iter": 20,
    "nf_min": 200,
    "epsilon": 1e-6,
    "w_s": 1.0,
    "w_b": 1.0,
    "wM_kap": 0.1,
    "wL_kap": 0.1,
    "wM_lam": 0.0,
    "wL_lam": 0.1,
    "wP_kap": 0.01,
    "wP_lam": 0.01,
    "penalty_threshold": 0.01,
    "betaP": 50.0
  }
}
```

| Section | Key | 类型 | 说明 |
|---------|-----|------|------|
| `model` | `name` | string | 模型名称 |
| `model` | `mesh_path` | string | 目标网格 OBJ 路径（Parameterize/Evaluate 使用；Inverse/InverseWhole 直接读 patches，但 Config 仍要求此字段存在） |
| `material` | `curves_path` | string | 材料多项式曲线 JSON 路径 |
| `segment` | `path` | string | 分割输出根目录 |
| `segment` | `method` | string | EvolutionCut 方法标签（空则不拼接） |
| `segment` | `plan` | string | EvolutionCut 方案标签（空则不拼接） |
| `paths` | `param_path` | string | 参数化结果基础目录 |
| `paths` | `morph_path` | string | 逆设计产物基础目录 |
| `paths` | `design_path` | string | 材料分配基础目录 |
| `paths` | `cond_path` | string | 边界条件基础目录 |
| `solver` | `platewidth` | double | 物理板材尺寸（决定 globalScale，影响 h·κ 比） |
| `solver` | `max_iter` | int | SGN 外层最大迭代数（内层 Newton 固定 100 次） |
| `solver` | `nf_min` | int | 最小面数（不足时线性细分） |
| `solver` | `epsilon` | double | 收敛阈值 |
| `solver` | `w_s`, `w_b` | double | 拉伸/弯曲能量权重 |
| `solver` | `wM_kap`, `wL_kap` | double | kappa L2/Laplacian 正则权重 |
| `solver` | `wM_lam`, `wL_lam` | double | lambda L2/Laplacian 正则权重 |
| `solver` | `wP_kap`, `wP_lam` | double | kappa/lambda 材料 penalty 权重 |
| `solver` | `penalty_threshold` | double | penalty 倍增阈值 |
| `solver` | `betaP` | double | soft-min penalty 形状参数 |

> **路径构建规则**：`modelDir() = {name}[_{method}][_{plan}]`。所有 `paths.*` 仅存储基础目录，代码自动拼接 `{base_path}/{modelDir}/`。JSON 中**不应**包含模型名。

## 代码架构

```
GrayScaleMorph/
├── core/                            # 共享静态库 libgrayscale_core
│   ├── functions.h/cpp              # 物理能量 + 伴随函数 + penalty
│   ├── newton.h/cpp                 # Newton + SGN 求解器
│   ├── morphmesh.hpp/cpp            # 目标 (lambda, kappa) 从变形梯度反算
│   ├── morph_functions.hpp/cpp      # MrInv 预计算工具
│   ├── simulation_utils.h/cpp       # 矩阵装配 + line search
│   ├── parameterization.h/cpp       # 底层 Tutte + ASAP 参数化
│   ├── LocalGlobalSolver.h/cpp      # Local-global ASAP 求解器
│   ├── parameterize_pipeline.h/cpp  # ParameterizeResult + parameterizeMesh()
│   ├── config.hpp/cpp               # cfg.json 解析器（含向后兼容）
│   ├── material.hpp/cpp             # 灰度材料模型：dose → strain/modulus + feasible set
│   ├── output.hpp/cpp               # OBJ/VTK 导出
│   ├── patch_utils.h/cpp            # linearSubdivide + seg_id 读取 + patch 子网格抽取
│   ├── rigid_align.h/cpp            # 闭式 Procrustes 刚体对齐（SVD，旋转+平移）
│   └── common.hpp, solvers.h, timer.h
├── inverse/                         # Inverse 单 patch 调试入口
│   ├── main.cpp                     # 三阶段 pipeline，仅处理 patch.id
│   └── inverse_design.h/cpp         # InverseDesignResult + runInverseDesign()
├── inverse_whole/main.cpp           # InverseWhole 全 patch 入口
├── forward/main.cpp                 # Forward 入口
├── parameterize/main.cpp            # Parameterize 入口
├── eval/main.cpp                    # Evaluate 入口
└── CMakeLists.txt
```

### CMake 构建结构

- **grayscale_core**（STATIC 库）：编译 `core/` + `inverse/inverse_design.cpp`，链接 libigl、geometry-central、TinyAD
- **Parameterize**、**Forward**、**Inverse**、**Evaluate**：各自只有一个 `main.cpp`，链接 `grayscale_core`
- **GrayScaleMorph**：`Inverse` 的向后兼容别名（编译自同一 `inverse/main.cpp`）
- 所有可执行文件输出到 `build/bin/`

## 执行流程（InverseWhole / Inverse）

```
main()
  ├── Phase 1: 读取所有 patches (segment/{modelDir}/patches/patch_*.obj)
  │   ├── linearSubdivide（按需，保几何）
  │   └── parameterizeMesh(V_original) → V(不变) + P(gauge-shifted)
  │
  ├── Phase 2: 计算 globalScale
  │   ├── PCA 旋转每个 P（最长轴 → x）
  │   ├── scale_i = platewidth / P_extent_i
  │   └── globalScale = min(scale_i)
  │
  └── Phase 3: 对每个 patch（或 Inverse 仅 target patch）
      ├── V *= globalScale, P *= globalScale
      ├── 构建 mesh/geometry/MrInv/fixedIdx
      ├── runInverseDesign():
      │   ├── ComputeMorphophing() → 目标 (λ_target, κ_target)
      │   ├── SGN 主循环（最多 5 stage）:
      │   │   ├── OptKap: 固定 λ, 优化 κ (per-face)
      │   │   ├── OptLam: 固定 κ, 优化 λ (per-face)
      │   │   ├── 每 stage：投影 (λ,κ) → forward 验证 → rigid-align → projected distance
      │   │   └── penalty 超阈值 → wP 翻倍, 正则减半
      │   ├── 最终投影到最近可制造 (t1, t2)
      │   ├── 从平板 (P.x, P.y, 0) 重新 forward → V_proj
      │   └── rigidAlign(V_proj, targetV) → 对齐后计算 dist_proj
      └── 输出: param + target + proj + material + metrics + cond
```

## 物理模型

### 能量泛函
离散非欧薄壳，目标第一/第二基本形式：

```
E = Sigma_faces [ W_stretch(f) + W_bend(f) ]

W_stretch = w_s * dA/lambda^2 * [ alpha/2 * tr(eps_s)^2 + beta * tr(eps_s^2) ]
W_bend   = w_b * dA * h^2/(3*lambda^2) * [ alpha/2 * tr(eps_b)^2 + beta * tr(eps_b^2) ]

其中：
  F = M * MrInv           (2D → 3D 变形梯度)
  a = F^T F               (当前第一基本形式)
  b = F^T L F             (当前第二基本形式，L = 二面角曲率算子)
  a_bar = lambda^2 I      (目标第一基本形式)
  b_bar = lambda^2 kappa I (目标第二基本形式)
  eps_s = a - a_bar,  eps_b = b - b_bar
  alpha = E*nu/(1-nu^2),  beta = E/(2*(1+nu))
```

### 设计变量
| 变量 | 维度 | 含义 |
|------|------|------|
| `lambda_pf` | per-face 标量 | 面内各向同性伸缩 |
| `kappa_pf` | per-face 标量 | 弯曲曲率 |

**交替优化**：OptKap（固定 lambda 优化 kappa）与 OptLam（固定 kappa 优化 lambda）交替进行。两者均为 per-face。

### 材料模型
灰度双层复合材料，dose → (strain, modulus) 多项式映射：
```
strain(t) = poly_strain(t)
modulus(t) = poly_modulus(t)
lambda(t1,t2) = 1 + 0.5 * (strain(t1) + strain(t2))
kappa(t1,t2) = 1.5 * (strain(t1) - strain(t2)) / thickness
```
**注意**：当前主优化在连续 (lambda, kappa) 空间进行，优化结束后再投影到最近的离散可制造 (t1, t2)。模量 Eeff 已读入但未参与主能量计算（即当前只用"剂量影响本征应变"，未用"剂量影响弹性模量"）。

## 逆设计 Pipeline

### 目标函数
```
J(theta) = ||x*(theta) - x_target||^2_M + wM * theta^T M_theta theta
           + wL * theta^T L_theta theta + wP * q(theta)

其中：
  x*(theta) = 内层 Newton 求解的平衡形状
  M = 面积加权质量矩阵（数据项）
  M_theta = 设计变量 L2 正则
  L_theta = 设计变量 Laplacian 平滑
  q(theta) = 向离散 feasible set 吸附的 soft-min penalty
```

### 梯度计算
adjoint / implicit differentiation（非有限差分）：
1. `adjointFunction_*` 构建联合 Hessian H（状态 x + 设计 theta）
2. 解伴随系统得 `nabla_theta J`
3. `buildHGN()` 装配 SGN 稀疏块系统更新设计变量

### 边界条件
- 无夹持边界，仅固定参数域中心面 3 个顶点（9 DOF）去除刚体运动
- 边界边不贡献二面角弯曲项

## 参数化

1. **Tutte embedding**：边界映射到圆 → 内部 harmonic
2. **ASAP (LocalGlobalSolver)**：local-global 迭代，投影奇异值到平均值
3. **Gauge shift**：全局缩放使 lambda 分布落入材料可行窗口

## 与参考论文的关键差异

| 方面 | Shrink & Morph (原论文) | GrayScaleMorph (本实现) |
|------|------------------------|------------------------|
| 材料 | 单材料 PLA（各向异性收缩） | 灰度双层复合材料（各向同性 lambda, kappa） |
| 设计变量 | 连续轨迹场 (theta_1, theta_2) | 连续 (lambda, kappa) → 离散投影 (t1, t2) |
| 参数化 | Local-Global 直接截断到材料窗口 | ASAP + gauge shift 全局缩放 |
| Patch 支持 | 无（整板） | seg_id.txt → per-patch 独立逆设计 |
| 材料 penalty | 无 | soft-min 吸附到 feasible set |

| 方面 | SGN (原论文) | GrayScaleMorph (本实现) |
|------|-------------|------------------------|
| 设计变量 | 统一 vertex-based | lambda per-face + kappa per-face 交替优化 |
| 正则 | 通用 | lambda 使用 face-based 正则图 |
| 目标 | 纯参数场 | 形状误差 + 可制造 penalty + 场正则 |

## Bug 修复历史
详见 `PLAN.md`，记录了 10 个关键问题的诊断与修复（fixedIdx 为空、能量不一致、adjoint 错误等），所有 P0 bug 已修复。
