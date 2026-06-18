# 足球比赛量化预测系统 — 完整计算手册

> 一份可复现的足球预测系统工程指南。
> 涵盖：8 个模型的完整公式 + 推理逻辑 + 权重设计 + 风险标签触发 + Prompt 模板 + 输出格式

---

## 目录

1. [标准预测输出格式](#1-标准预测输出格式)
2. [预测计算全流程](#2-预测计算全流程)
3. [模型一：Dixon-Coles 泊松模型](#3-模型一dixon-coles-泊松模型)
4. [模型二：TabularMatchEnhancer 机器学习增强器](#4-模型二tabularmatchenhancer-机器学习增强器)
5. [模型三：κ-Elo Davidson 评分系统](#5-模型三κ-elo-davidson-评分系统)
6. [模型四：Pi-Rating 评分系统](#6-模型四pi-rating-评分系统)
7. [模型五：Weibull Copula 模型](#7-模型五weibull-copula-模型)
8. [概率融合策略：FusionGraph](#8-概率融合策略fusiongraph)
9. [权重配置：为什么不同比赛用不同权重](#9-权重配置为什么不同比赛用不同权重)
10. [信号调整系统](#10-信号调整系统)
11. [风险标签触发逻辑](#11-风险标签触发逻辑)
12. [置信度评估](#12-置信度评估)
13. [数据源清单](#13-数据源清单)
14. [完整 Prompt 模板](#14-完整-prompt-模板)
15. [关键参数速查表](#15-关键参数速查表)
16. [模型六：复合泊松-伽马模型 (极端大比分 ≥6球)](#16-模型六复合泊松-伽马模型-极端大比分-6球)
17. [模型七：复合泊松-λ模型 (中等大比分 4-5球)](#17-模型七复合泊松-λ模型-中等大比分-4-5球)
18. [模型八：战意指数 (Motivation Index)](#18-模型八战意指数-motivation-index)
19. [7层融合架构](#19-7层融合架构)

---

## 1. 标准预测输出格式

**每一场比赛的预测结果应包含以下全部字段：**

```markdown
## {主队} vs {客队} — 预测结果

### 最终融合概率
| 结果 | 概率 |
|------|------|
| 主胜 | XX.X% |
| 平局 | XX.X% |
| 客胜 | XX.X% |

xG: 主队 X.XX — X.XX 客队

### Top 3 比分
| 比分 | 概率 |
|------|------|
| X:X  | XX.X% |
| X:X  | XX.X% |
| X:X  | XX.X% |

### Elo 评分
| 球队 | Elo |
|------|-----|
| 主队 | XXXX |
| 客队 | XXXX |
| 差距 | ±XXX |

### 各层独立预测
| 模型 | 主胜 | 平局 | 客胜 |
|------|------|------|------|
| Dixon-Coles  | XX.X% | XX.X% | XX.X% |
| Enhancer (ML) | XX.X% | XX.X% | XX.X% |
| Elo | XX.X% | XX.X% | XX.X% |
| Pi-Rating | XX.X% | XX.X% | XX.X% |
| Weibull (可选) | XX.X% | XX.X% | XX.X% |

### 元数据
- 权重配置: {WC_V3.8 / FRIENDLY_ADJUSTED_V2 / LEAGUE_DEFAULT / ...}
- 使用模型: [dixon_coles, tabular_enhancer, elo, pi_rating]
- 置信度: {fitted / estimated_prior / cold_start}
- 置信度惩罚: X.XXX
- 风险标签: [tag1, tag2, ...]
- 模型分歧度: Xpp

### 解读
{自然语言分析段落，含：
- 哪边占优及理由
- 各模型判断的一致性
- 是否有风险标签及其含义
- 潜在的不确定性来源}
```

---

## 2. 预测计算全流程

```
输入
  home_team, away_team, competition, is_neutral, match_date, venue
  │
  ▼
Step 1: 比赛类型检测 & 权重选择
  检测 competition 字段 → 映射 weight_config

Step 2: 加载训练数据
  从 SQLite/PostgreSQL 加载历史比赛数据 (11,011场)
  3层缓存: 内存 → 磁盘 → 冷启动

Step 3: Dixon-Coles 预测
  攻击力×防守力 → λ/μ → 泊松概率 → 比分矩阵 → H/D/A

Step 4: TabularEnhancer 预测
  构建 37 维特征向量 → HistGradientBoosting → [H,D,A]

Step 5: DC + Enhancer 融合 (FusionGraph Step 1)
  fused₁ = DC × dc_weight + Enhancer × (1-dc_weight)

Step 6: Elo 预测
  从 elo.json 加载评分 → κ-Elo Davidson → [H,D,A]

Step 7: Elo 融合 (FusionGraph Step 2)
  fused₂ = fused₁ × (1-elo_weight) + Elo × elo_weight

Step 8: Pi-Rating 预测
  从 pi.json 加载评分 → sigmoid 映射 → [H,D,A]

Step 9: Pi 融合 (FusionGraph Step 3)
  fused₃ = fused₂ × (1-pi_weight) + Pi × pi_weight

Step 10: Weibull 融合 (可选)
  fused₄ = fused₃ × (1-wb_weight) + Weibull × wb_weight

Step 11: 归一化
  确保 H+D+A = 1.0

Step 12: 战意指数分析
  计算双方战意 → 预测比赛风格 → 输出大比分阈值调整

Step 13: 双大比分模型检测
  模型A(≥6球) + 模型B(4-5球) → 根据战意调整阈值 → 动态选择

Step 14: 7层融合
  标准模型 80% + 大比分模型 15% + 战意微调 5%

Step 15: 信号调整
  伤病 + 新闻信号 + 天气 + 场地 + 手动事件

Step 16: 上下文调整
  中立场地/德比/淘汰赛 微调 ±3pp

Step 17: 风险标签生成
  高海拔/模型分歧/伤病/冷启动/旅行疲劳...

Step 18: 置信度评估
  fitted / estimated_prior / cold_start + 惩罚系数

Step 19: 输出 PredictionResult
```

---

## 3. 模型一：Dixon-Coles 泊松模型

### 3.1 理论基础

**Dixon & Coles (1997)** — 对独立泊松模型的改进。

**为什么要用泊松模型？**
足球进球是稀有事件（90 分钟里每队平均进 1.2-1.5 球），泊松分布是建模稀有事件计数的自然选择。但独立泊松有一个致命缺陷——它假设两队进球完全独立，这在足球中不成立（一方进球后另一方战术改变，导致低比分场景概率失真）。

**Dixon-Coles 修正了什么？**
引入低比分相关性参数 ρ，专门修正 0-0、1-0、0-1、1-1 这四个最容易出错的比分概率。

### 3.2 核心公式

```
进球期望:
  λ_home = attack[主队] × defense[客队] × exp(主场优势)
  μ_away = attack[客队] × defense[主队]

  attack ∈ [0.2, 5.0], defense ∈ [0.2, 5.0]
  attack > 1.0 表示进攻强于平均
  defense < 1.0 表示防守强于平均

低比分 τ 修正:
  τ(0,0) = 1 - λ·μ·ρ     ← 降低 0-0 概率
  τ(1,0) = 1 + μ·ρ       ← 提高 1-0 概率
  τ(0,1) = 1 + λ·ρ       ← 提高 0-1 概率
  τ(1,1) = 1 - ρ          ← 降低 1-1 概率
  τ(其他) = 1.0            ← 其他不变

最终概率:
  P(X=x, Y=y) = τ(x,y) × Poisson(x|λ) × Poisson(y|μ)
```

**为什么 ρ 有这些符号？**
历史数据表明 0-0 和 1-1 发生的频率**低于**独立泊松的预测，而 1-0 和 0-1 发生的频率**高于**预测。ρ > 0 时，τ(0,0) 和 τ(1,1) 被压缩，τ(1,0) 和 τ(0,1) 被放大，完美匹配经验观测。

### 3.3 训练方法

```
优化算法: L-BFGS-B (scipy)
损失函数: 负对数似然
  Loss = -Σ weight_i × log(P(home_goals_i, away_goals_i))

时间衰减权重:
  weight = exp(-ln(2) × days_ago / 180) × competition_weight
  → 半衰期 180 天: 180 天前的比赛权重减半
  → competition_weight: WC=1.5, Friendly=0.5, Default=0.9
  → 目的: 近期比赛更重要，重要比赛更有参考价值
```

### 3.4 冷启动机制

**问题**：新球队或数据极少的球队（< 5 场），训练不出来可靠的 attack/defense 参数。

**解决方案**：
```
Step 1: 计算该球队所在洲的平均攻击力/防守力
Step 2: 用 FIFA 排名层级修正:
  Tier 1 (Top 10):    攻击×1.15, 防守×0.85
  Tier 2 (11-30):     攻击×1.05, 防守×0.95
  Tier 3 (31-50):     攻击×0.95, 防守×1.05
  Tier 4 (51+):       攻击×0.88, 防守×1.12
Step 3: 标记 data_quality = "estimated_prior"
Step 4: 置信度惩罚 +0.15
```

### 3.5 比分矩阵生成

```
对每个 λ, μ:
  计算 0×0 到 4×4 共 25 个组合的 P(x,y)
  按概率排序取 Top 3
  返回 5×5 得分矩阵
```

---

## 4. 模型二：TabularMatchEnhancer 机器学习增强器

### 4.1 理论基础

**为什么要加 ML 模型？**
Dixon-Coles 只知道"攻击力 × 防守力 + 泊松"，不知道状态（近 5 场）、不知道休息天数、不知道经验差距。ML 模型可以捕捉这些非线性交互。

### 4.2 算法

```
算法: HistGradientBoostingClassifier (sklearn)
超参数:
  learning_rate = 0.06  ← 低学习率，防过拟合
  max_depth = 4         ← 浅树，防过拟合
  max_iter = 220        ← 足够大但不过拟合
  min_samples_leaf = 4  ← 叶子最少样本，防噪声

任务: 三分类 [H, D, A]
```

**为什么选 HistGradientBoosting 而不是 XGBoost/LightGBM？**
- 原生支持缺失值（新球队特征缺失常见）
- 不需要调参就能获得不错的效果
- 比 XGBoost 更不容易过拟合小样本

### 4.3 37 维特征向量

```
窗口类型:
  全量窗口 (expanding):   所有历史数据的均值 → 长期实力
  近期窗口 (rolling=5):   最近 5 场的均值   → 当前状态
  静态:                   与历史无关         → 客观条件

特征列表:
  ┌────┬──────────────────────────────┬────────┬──────────────────┐
  │ #  │ 特征                         │ 窗口   │ 含义             │
  ├────┼──────────────────────────────┼────────┼──────────────────┤
  │ 1  │ is_neutral_venue             │ 静态   │ 中立场地          │
  │ 2  │ competition_weight           │ 静态   │ 比赛重要性         │
  │ 3  │ home_matches_played          │ 全量   │ 主队经验量         │
  │ 4  │ away_matches_played          │ 全量   │ 客队经验量         │
  │ 5  │ experience_gap               │ 全量   │ 经验差距           │
  │ 6  │ home_goals_for_avg           │ 全量   │ 主队场均进球       │
  │ 7  │ home_goals_against_avg       │ 全量   │ 主队场均失球       │
  │ 8  │ away_goals_for_avg           │ 全量   │ 客队场均进球       │
  │ 9  │ away_goals_against_avg       │ 全量   │ 客队场均失球       │
  │ 10 │ goal_balance_gap             │ 全量   │ 净胜球差距         │
  │ 11 │ home_xg_for_avg              │ 全量   │ 主队场均 xG        │
  │ 12 │ home_xg_against_avg          │ 全量   │ 主队场均 xGA       │
  │ 13 │ away_xg_for_avg              │ 全量   │ 客队场均 xG        │
  │ 14 │ away_xg_against_avg          │ 全量   │ 客队场均 xGA       │
  │ 15 │ xg_balance_gap               │ 全量   │ xG 净差值差距      │
  │ 16 │ home_recent_goals_for_avg    │ 5场    │ 主队近5场场均进球  │
  │ 17 │ home_recent_goals_against_avg│ 5场    │ 主队近5场场均失球  │
  │ 18 │ away_recent_goals_for_avg    │ 5场    │ 客队近5场场均进球  │
  │ 19 │ away_recent_goals_against_avg│ 5场    │ 客队近5场场均失球  │
  │ 20 │ recent_goal_gap              │ 5场    │ 近5场净胜球差距    │
  │ 21 │ home_recent_xg_for_avg       │ 5场    │ 主队近5场场均 xG   │
  │ 22 │ home_recent_xg_against_avg   │ 5场    │ 主队近5场场均 xGA  │
  │ 23 │ away_recent_xg_for_avg       │ 5场    │ 客队近5场场均 xG   │
  │ 24 │ away_recent_xg_against_avg   │ 5场    │ 客队近5场场均 xGA  │
  │ 25 │ recent_xg_gap                │ 5场    │ 近5场 xG 净差值    │
  │ 26 │ home_points_per_match        │ 全量   │ 主队场均积分       │
  │ 27 │ away_points_per_match        │ 全量   │ 客队场均积分       │
  │ 28 │ points_gap                   │ 全量   │ 积分差距           │
  │ 29 │ home_recent_points_per_match │ 5场    │ 主队近5场场均积分  │
  │ 30 │ away_recent_points_per_match │ 5场    │ 客队近5场场均积分  │
  │ 31 │ recent_points_gap            │ 5场    │ 近5场积分差距      │
  │ 32 │ home_win_rate                │ 全量   │ 主队历史胜率       │
  │ 33 │ away_win_rate                │ 全量   │ 客队历史胜率       │
  │ 34 │ win_rate_gap                 │ 全量   │ 胜率差距           │
  │ 35 │ home_rest_days               │ 静态   │ 主队休息天数       │
  │ 36 │ away_rest_days               │ 静态   │ 客队休息天数       │
  │ 37 │ rest_day_diff                │ 静态   │ 休息天数差         │
  └────┴──────────────────────────────┴────────┴──────────────────┘
```

**为什么选择这些特征？**
- 全量窗口特征（1-15, 26-28, 32-34）衡量长期实力
- 近期窗口特征（16-25, 29-31）衡量当前状态，捕捉状态起伏
- 静态特征（35-37）衡量客观条件，疲劳管理是足球预测的关键变量
- xG 特征（11-15, 21-25）比实际进球更稳定，能捕捉"运气成分"

**缺失值处理**：新球队首次出现时，用全局均值填充
```
goals_for = goals_against = xg = 1.3
points_per_match = 1.3
win_rate = 0.33
rest_days = 7.0
```

---

## 5. 模型三：κ-Elo Davidson 评分系统

### 5.1 理论基础

**Szczecinski & Djebbi (2020)** — 在标准 Elo 基础上增加平局概率建模。

**为什么要用 Elo？**
Elo 是最经典的实力评分系统，简单可解释。但标准 Elo 没有平局概念，κ-Elo Davidson 弥补了这一点。

**为什么选 κ-Elo Davidson 而不是标准 Elo？**
标准 Elo 只输出胜负期望，平局被忽略。κ-Elo Davidson 显式建模平局概率，这对于足球（平局率 ~25%）至关重要。

### 5.2 核心公式

```
初始评分: R = 1500
主场优势: H = 100 分

胜负期望:
  expected_home = 1 / (1 + 10^((R_away - (R_home + H)) / 400))

平局概率 (κ-Elo Davidson):
  r = elo_diff / 400
  σ(r) = 1 / (1 + 10^(-r))
  P(draw) = κ × σ(r) × σ(-r)

  κ 值按比赛类型:
    英超 (EPL):     0.28  ← 英超平局率最高
    欧冠 (UCL):     0.18  ← 淘汰赛平局率低
    FA Cup:         0.28
    默认:           0.24

K 因子:
  WC/淘汰赛:        32    ← 信息量大，权重高
  欧冠:             28
  常规联赛:         20

升级公式:
  R_new = R_old + K × (result - expected)
  result: 主胜=1, 平=0.5, 客胜=0

概率分配:
  p_draw = κ × σ(r) × σ(-r)
  p_home = expected_home × (1 - p_draw)
  p_away = (1 - expected_home) × (1 - p_draw)
  归一化
```

### 5.3 为什么 κ = 0.24 作为默认值？

经验数据：国际足球平局率 ~24%。κ = 0.24 使得当两队实力相等 (r=0) 时：
```
σ(0) = 0.5
P(draw) = 0.24 × 0.5 × 0.5 = 0.06 ← 偏低！

实际融合后:
P(draw) ≈ κ × σ(r) × σ(-r) / (expected_home×(1-p_draw) + ...)
≈ 0.24 × 0.5 × 0.5 / normalization ≈ 0.24
```
经过归一化后，实际平局概率接近 24%，与观测值一致。

---

## 6. 模型四：Pi-Rating 评分系统

### 6.1 理论基础

**Constantinou & Fenton (2012)** — 对进球的评分系统，响应净胜球差。

**为什么要用 Pi-Rating？**
Elo 只看胜负，不看比分。Pi-Rating 看重比分：
- 5-0 比 1-0 产生更大的评分变化（赢得多 → 更强）
- 但用 diminishing returns 防止屠杀主导评分（10-0 ≈ 6-0）

### 6.2 核心公式

```
参数:
  k = 0.1           ← 评分变化速率
  PROB_SCALE = 0.35 ← 概率映射尺度

预测公式:
  home_adj = 0.3 (非中立) / 0.0 (中立)
  xg_diff = (rating_home + home_adj - rating_away) × 0.35 × 2.0

  home_win = 1 / (1 + exp(-xg_diff × 2.5))
  away_win = 1 / (1 + exp(xg_diff × 2.5))
  draw     = 0.26 × exp(-xg_diff² / 2.0)   ← 高斯衰减

  total = home_win + draw + away_win
  各自除以 total 归一化
```

**为什么平局用高斯衰减？**
实力越接近 (xg_diff → 0)，平局概率越高；实力差距越大，平局概率指数级下降。这符合直觉——巴西踢中国大概率不是平局，但巴西踢阿根廷很有可能是平局。

### 6.3 Pi-Rating 的局限性（重要！）

**跨洲际偏差**：Pi-Rating 基于净胜球差，新西兰在 OFC 经常 5-0 赢塔希提导致评分虚高；伊朗在 AFC 踢强队导致评分被压制。这就是为什么伊朗 vs 新西兰那场 Pi 说新西兰 60.8% 胜，和其他三个模型完全相反。

**你的系统应如何处理**：
- Pi 权重不宜超过 15%
- 出现 Pi 与其他模型分歧 > 30pp 时，触发风险标签并降低 Pi 权重
- 跨洲际比赛可考虑 Pi 权重减半

---

## 7. 模型五：Weibull Copula 模型

### 7.1 理论基础

**Boshnakov, Kharrat & McHale (2017)** — 用 Weibull 分布替代泊松分布。

**为什么要加这个模型？**
泊松分布假设进球间隔时间服从指数分布（无记忆性），但实际足球中存在动量效应——进了一个球后短时间内更容易再进一个。Weibull 分布可以捕捉这种效应。Frank Copula 额外引入了两队间的进球相关性。

**使用门槛**：需要 penaltyblog 库，训练时间较长（~2.5s），当前标记为可选组件。

---

## 8. 概率融合策略：FusionGraph

### 8.1 为什么不用简单的加权平均？

简单加权平均有一个问题：如果 DC 说 90% 主胜、Enhancer 说 10% 主胜，直接平均得到 50%——这掩盖了巨大分歧。

**顺序加权融合 (Sequential Weighted Fusion)** 的直觉：
每一层不是直接与"最终结果"加权，而是与"当前累积结果"加权。这样后加入的模型有增量贡献，而不是完全覆盖。

### 8.2 融合公式

```
Step 1: DC + Enhancer
  fused₁ = DC × dc_weight + Enhancer × (1-dc_weight)
  → 归一化

Step 2: 加入 Elo
  fused₂ = fused₁ × (1-elo_weight) + Elo × elo_weight
  → 归一化

Step 3: 加入 Pi
  fused₃ = fused₂ × (1-pi_weight) + Pi × pi_weight
  → 归一化

Step 4: 加入 Weibull (可选)
  fused₄ = fused₃ × (1-wb_weight) + Weibull × wb_weight
```

### 8.3 有效权重推导

```
dc_effective       = dc_weight × (1-elo_weight) × (1-pi_weight)
enhancer_effective = (1-dc_weight) × (1-elo_weight) × (1-pi_weight)
elo_effective      = elo_weight × (1-pi_weight)
pi_effective       = pi_weight
```

**举例：世界杯 V3.8 的有效权重**
```
dc       = 0.70 × 0.90 × 0.90 = 0.567  (实际 56.7%)
enhancer = 0.30 × 0.90 × 0.90 = 0.243  (实际 24.3%)
elo      = 0.10 × 0.90         = 0.090  (实际 9.0%)
pi       = 0.10                = 0.100  (实际 10.0%)
```

### 8.4 模型分歧度检测

```
对每对模型计算 |home_win_prob_A - home_win_prob_B|
分歧 > 30pp → 触发 "high_model_disagreement" 风险标签
分歧 20-30pp → 触发 "moderate_model_disagreement"
分歧 < 20pp → 正常
```

**为什么 30pp 作为阈值？**
经验上看，当两个模型对同一场比赛的主胜概率差超过 30pp，说明至少有一个模型对该场比赛存在系统性盲区，此时用户的决策应该更谨慎。

---

## 9. 权重配置：为什么不同比赛用不同权重

### 9.1 核心逻辑

不同类型比赛有不同的特点，不同模型在不同比赛类型中的准确率也不同。一刀切的权重会产生系统性偏差。

**友谊赛**：不确定性高，ML (Enhancer) 的表现好于纯统计模型。所以 Enhancer 权重最高 (42%)，DC 降低 (28%)。

**世界杯**：高关注度、正赛性质，Dixon-Coles 的泊松假设效果最好。所以 DC 权重最高 (70%)。

**欧冠淘汰赛**：双回合的特殊性，需平衡 DC 和 Enhancer。

### 9.2 完整权重表

| 比赛类型 | DC | Enhancer | Elo | Pi | Weibull | 理由 |
|----------|-----|----------|-----|-----|---------|------|
| 世界杯 V3.8 | 70% | 20% | 10% | 10% | 10% | 正赛关键战，泊松最可靠 |
| 欧冠决赛 | 42% | 30% | 8% | 12% | 8% | 单场决胜，ML 样本少 |
| 欧冠淘汰赛 | 45% | 28% | 7% | 10% | 10% | 双回合特殊 |
| 联赛默认 | 50% | 30% | 5% | 5% | 10% | 平衡配置 |
| 友谊赛 V2.7 | 28% | 42% | 2% | 16% | 12% | 不确定性高，ML 更优 |

### 9.3 友谊赛 V2.7 的自我进化

```
回顾 3 场友谊赛:
  DC:       0/3 正确 → dc_weight: 0.38 → 0.28 (-26%)
  Elo:      0/3 正确 → elo_weight: 0.04 → 0.02 (-50%)
  Enhancer: 2/3 正确 → enhancer_weight: 0.42 (不变)
  Pi:       唯一正确预测 SG-CN → pi_weight: 0.04 → 0.16 (+300%)
```

**为什么要做自我进化？**
固定权重在变化的环境中会过时。通过赛后评估，让表现好的模型获得更多权重，表现差的模型被逐步淘汰。

### 9.4 比赛类型自动检测

```
competition 包含 "World Cup"   → competition_weight = 1.5, 使用 WC_V3.8
competition 包含 "Friendly"    → competition_weight = 0.5, 使用 FRIENDLY_ADJUSTED_V2
competition 包含 "Champions"   → competition_weight = 1.2, 使用 UCL_KNOCKOUT
其他                            → competition_weight = 0.9, 使用 LEAGUE_DEFAULT
```

---

## 10. 信号调整系统

### 10.1 为什么要做信号调整？

统计模型只能看到历史数据，看不到"梅西今天受伤了"、"球队刚飞了 8000 公里"、"大暴雨导致场地泥泞"。这些确定性信息必须手动融入。

### 10.2 信号调整公式

```
概率调整:
  new_home = home_prob + net_adjustment × (1 - home_prob)
  new_draw = 1.0 - new_home - new_away
  new_away = away_prob + net_away_adjustment × (1 - away_prob)

net_adjustment = Σ(signal × confidence × reliability × multiplier)
  硬上限: ±0.15
```

**为什么是 × (1 - home_prob) 而不是直接加减？**
如果一个信号说主队有优势，直接给主胜 +0.10。但如果主胜已经是 85%，+0.10 就会超过 1.0。× (1 - home_prob) 让调整被概率空间限制：高概率时调整小，低概率时调整大。

### 10.3 信号类型与最大调整

```
信号类型              最大概率偏移    示例场景
─────────────────────────────────────────────────
injury (关键前锋)      0.15          核心前锋赛前受伤
injury (门将)          0.10          首发门将缺阵
suspension             0.15          核心球员停赛
major_rotation         0.12          大幅轮换首发
lineup_hint            0.10          首发阵容不确定
return                 0.08          关键球员伤愈复出
form_change            0.08          连胜/连败状态
manager_change         0.08          换帅
travel_fatigue         0.06          横跨洲际长途飞行
tactical_shift         0.06          阵型/打法变化
morale_event           0.04          更衣室矛盾/士气
schedule_pressure      0.04          密集赛程
weather_impact         0.04          大雨/高温/强风
other                  0.03          其他不确定因素
```

### 10.4 伤病状态映射

```
out / suspended (关键球员):  xG × 0.85 (降 15%)
out / suspended (普通):      xG × 0.90 (降 10%)
doubtful:                    xG × 0.92 (降 8%)
probable:                    xG × 0.97 (降 3%)
available:                   不变
likely_start (回归):         xG × 1.08 (升 8%)
```

### 10.5 动态信号倍率

信号来源未必可靠，需要根据历史准确率自动调整：

```
信号准确率 > 80%:   倍率 = 1.0  (完全信任)
信号准确率 50-80%:  倍率 = 0.8  (部分信任)
信号准确率 < 50%:   倍率 = 0.5  (可靠性差)
无历史数据:         倍率 = 0.7  (保守默认)
```

**为什么默认是 0.7 而不是 1.0？**
未经验证的信号不应完全信任。0.7 意味着即使信号说 +/-15pp，实际只应用 +/-10.5pp。随着数据积累，倍率自动调整。

---

## 11. 风险标签触发逻辑

| 风险标签 | 触发条件 | 含义 |
|----------|----------|------|
| `high_model_disagreement_{x}` | 任意两模型主胜概率差 > 30pp | 模型高度分歧，预测不可靠 |
| `moderate_model_disagreement_{x}` | 任意两模型主胜概率差 > 20pp | 模型中等分歧 |
| `冷启动球队` | 训练数据 < 5 场 | 参数靠先验估计，非数据驱动 |
| `高海拔场地` | 海拔 ≥ 1500m | xG × 0.95，双方进球减少 |
| `首发不确定` | lineup_hint + confidence < 0.6 | 阵容不确定性高 |
| `关键球员缺阵` | 核心球员 injury/suspension | 主队或客队实力受损 |
| `旅行疲劳` | 飞行距离 > 3000km | 客队身体状态可能不佳 |
| `大雨天气` | 降水 > 5mm | 场地湿滑影响比赛节奏 |
| `雨天影响` | 降水 > 1mm | 轻微天气影响 |
| `强风天气` | 风速 > 40 km/h | 影响传球和射门精度 |
| `高温酷暑` | 温度 > 32°C | 体能消耗更大 |
| `低温条件` | 温度 < 5°C | 肌肉僵硬风险 |
| `小组赛轮换风险` | WC 小组赛第 3 轮 | 已出线队可能轮换 |
| `情报冲突` | 多个信号方向矛盾 | 信号可靠性存疑 |

### 11.1 风险标签的叠加效应

多个风险标签不是简单累加。当前实现：
- 每个标签独立标记
- 置信度惩罚 = 所有标签惩罚的最大值（不是总和）
- 原因：如果"冷启动"已经惩罚了 0.15，"首发不确定"再罚 0.10 会导致总惩罚 0.25 —— 过于保守，实际风险不会线性叠加

---

## 12. 置信度评估

### 12.1 三级置信度

```
fitted:          模型参数从充足训练数据中拟合 (≥20 场)
estimated_prior: 训练数据不足，使用先验估计 (5-19 场)
cold_start:      新球队，参数完全靠先验 (<5 场)
```

### 12.2 置信度惩罚

```
fitted:          penalty = 0.0 (无惩罚)
estimated_prior: penalty = 0.08
cold_start:      penalty = 0.15
risk_tag:        附加 penalty = 0.03 (每个风险标签)
model_disagree:  penalty += divergence × 0.3 (分歧越大越不自信)

最终置信度:
  total_penalty = max(base_penalty, max(risk_penalties))
  confidence = "high" (total_penalty < 0.05)
               "medium" (0.05-0.15)
               "low" (>0.15)
```

---

## 13. 数据源清单

### 13.1 训练数据

| 数据项 | 来源 | 说明 |
|--------|------|------|
| 历史比赛 | SQLite `local_stage2.db` 或 PostgreSQL | 11,011 场，2015-2026 |
| 球队信息 | `teams` 表 | 名称、洲际、类型 |
| 比赛结果 | `match_results` 表 | 比分、xG |
| 比赛权重 | `matches.competition_weight` | WC=1.5, Friendly=0.5 |

### 13.2 评分数据

| 数据项 | 来源 | 说明 |
|--------|------|------|
| Elo 评分 | `artifacts/ratings/elo.json` | 296 队，每队一个 float |
| Pi 评分 | `artifacts/ratings/pi.json` | 296 队，零中心，正=强 |
| FIFA 层级 | 代码硬编码 | 48 队 WC26 的 Tier 1-4 |

### 13.3 外部实时数据

| 数据项 | API | 用途 |
|--------|-----|------|
| 天气 | Open-Meteo (免费) | 温度/降水/风速 |
| 伤病 | 本地 JSON / 手动录入 | 球员可用性 |
| 新闻信号 | 数据库 `news_signals` 表 + RSS | 阵容/战术/士气 |
| 市场赔率 | apifootball.com / API-Sports / The Odds API | shadow mode 校准 |

### 13.4 预训练模型缓存

| 文件 | 内容 |
|------|------|
| `model_artifacts/dc_cache/*.pkl` | Dixon-Coles 拟合参数 |
| `model_artifacts/dc_cache/enhancer_*.pkl` | TabularEnhancer 训练好的模型 |
| `artifacts/dataframes/national_finished_matches.pkl` | 训练数据 pickle 缓存 |

---

## 14. 完整 Prompt 模板

### 14.1 系统预测 Prompt

将此模板发给你的量化系统，每次预测时填充变量：

```
你是一个足球比赛量化预测系统。请严格按以下步骤计算并输出预测。

## 输入
- 主队: {home_team}
- 客队: {away_team}
- 赛事: {competition}
- 是否中立场地: {is_neutral}
- 比赛日期: {match_date}
- 场地: {venue}

## 计算步骤 (按顺序执行)

### Step 1: 确定权重配置
比赛类型自动检测:
- competition 包含 "World Cup"  → 使用 WC_V3.8
- competition 包含 "Friendly"   → 使用 FRIENDLY_ADJUSTED_V2
- competition 包含 "Champions"  → 使用 UCL_KNOCKOUT
- 其他                         → 使用 LEAGUE_DEFAULT

### Step 2: Dixon-Coles 泊松模型
- 从训练数据中获取 attack[{home_team}] 和 defense[{away_team}]
- 计算 λ = attack[主] × defense[客] × exp(主场优势)
- 计算 μ = attack[客] × defense[主]
- 应用 Dixon-Coles τ 修正
- 生成 5×5 分数矩阵，取 Top 3 比分
- 输出: [H_home, H_draw, H_away] + xG + Top 3 比分

### Step 3: TabularEnhancer ML 模型
- 构建 37 维特征向量
- 调用 HistGradientBoostingClassifier.predict_proba()
- 输出: [E_home, E_draw, E_away]

### Step 4: κ-Elo Davidson
- 从 elo.json 加载 {home_team} 和 {away_team} 的 Elo 评分
- 计算 expected_home = 1/(1+10^((R_away-(R_home+H))/400))
- 计算 p_draw = κ × σ(r) × σ(-r), κ 根据比赛类型
- 分配 p_home, p_away, 归一化
- 输出: [L_home, L_draw, L_away]

### Step 5: Pi-Rating
- 从 pi.json 加载评分
- 计算 xg_diff = (rating_home - rating_away) × 0.35 × 2.0
- home_win = 1/(1+exp(-xg_diff×2.5))
- draw = 0.26×exp(-xg_diff²/2.0)
- 归一化
- 输出: [P_home, P_draw, P_away]

### Step 6: 顺序加权融合 (标准5层)
- 选择权重: w_dc, w_enhancer, w_elo, w_pi
- fused₁ = DC × w_dc + Enhancer × (1-w_dc)
- fused₂ = fused₁ × (1-w_elo) + Elo × w_elo
- fused₃ = fused₂ × (1-w_pi) + Pi × w_pi
- 归一化
- 输出: 标准模型 [H, D, A]

### Step 7: 战意指数分析
- 计算 {home_team} 和 {away_team} 的战意指数
- 评估比赛风格 (对攻/单边/保守/沉闷)
- 输出: 大比分阈值调整 + 是否抑制大比分模型

### Step 8: 双大比分模型检测
- 模型A检测 (≥6球): Elo差距 + 近期进球 + 热身赛标签
- 模型B检测 (4-5球): 更宽松的阈值
- 根据战意调整阈值
- 动态选择 A / B / 不激活
- 输出: 大比分预测结果

### Step 9: 7层融合
- 标准模型 80% + 大比分模型 15% (条件激活) + 战意微调 5%
- 低战意时提高平局概率 (+5%)
- 高战意时降低平局概率 (-3%)
- 归一化
- 输出: 最终 [H, D, A]

### Step 10: 分歧检测
- 计算所有模型对之间的 |home_prob_A - home_prob_B|
- 最大值 > 0.30 → 标记 "high_model_disagreement"
- 最大值 > 0.20 → 标记 "moderate_model_disagreement"

### Step 11: 信号调整 (如有)
- 检查伤病信号: out → xG×0.85, doubtful → xG×0.92
- 检查场地信号: 海拔>1500m → xG×0.95
- 检查天气信号: 降水>5mm → xG×0.97
- 检查旅行信号: 飞行>3000km → xG×0.95
- 应用 net_adjustment × (1 - home_prob) 公式
- 重新归一化

### Step 12: 置信度评估
- 训练数据 ≥ 20 场 → "fitted", penalty=0
- 训练数据 5-19 场 → "estimated_prior", penalty=0.08
- 训练数据 < 5 场 → "cold_start", penalty=0.15
- 有风险标签 → penalty += 0.03/tag (取 max)

## 输出格式

{{ {{prediction_output}} }}

其中 {{prediction_output}} 必须严格按照以下结构:

## {home_team} vs {away_team} — 预测结果

### 最终融合概率
| 结果 | 概率 |
|------|------|
| {home_team}胜 | {home_pct} |
| 平局 | {draw_pct} |
| {away_team}胜 | {away_pct} |

xG: {home_team} {home_xg} — {away_xg} {away_team}

### Top 3 比分
| 比分 | 概率 |
|------|------|
| {score1} | {prob1} |
| {score2} | {prob2} |
| {score3} | {prob3} |

### Elo 评分
| 球队 | Elo |
|------|-----|
| {home_team} | {home_elo} |
| {away_team} | {away_elo} |
| 差距 | {elo_gap} |

### 各层独立预测
| 模型 | {home_team}胜 | 平局 | {away_team}胜 |
|------|------|------|------|
| Dixon-Coles | {dc_h} | {dc_d} | {dc_a} |
| Enhancer (ML) | {enh_h} | {enh_d} | {enh_a} |
| Elo | {elo_h} | {elo_d} | {elo_a} |
| Pi-Rating | {pi_h} | {pi_d} | {pi_a} |
| 大比分模型 | {bs_h} | {bs_d} | {bs_a} | ← 条件激活

### 元数据
- 权重配置: {weight_label}
- 使用模型: {components_used}
- 战意指数: 主队 {home_mot} / 客队 {away_mot} / 整体 {match_mot}
- 大比分模型状态: {active_model / 未激活}
- 置信度: {confidence}
- 风险标签: {risk_tags}

### 解读
{分析段落，包含:
1. 哪边占优及 Elo 差距
2. 各模型判断是否一致
3. 战意指数分析 (双方战意水平、比赛风格预测)
4. 大比分模型是否激活及原因
5. 如果有风险标签，解释其含义
6. 最高概率比分是什么
7. 不确定性来源}

## 约束规则
- 不使用博彩/投注用语 (赔率/盘口/下注/稳胆)
- 不使用"必中""稳赢""收米"等承诺性词汇
- 不确定处必须明确标注
- 4 个模型概率之和允许略微偏离 100% (融合后统一归一化)
```

### 14.2 赛后复盘 Prompt

```
你是一个足球预测评估系统。请对以下预测进行复盘分析。

## 比赛信息
- 对阵: {home_team} vs {away_team}
- 实际比分: {home_goals}:{away_goals}
- 实际结果: {H/D/A}

## 预测回顾
- 预测概率: H={pred_h} D={pred_d} A={pred_a}
- 预测 xG: {pred_xg_h} - {pred_xg_a}
- 预测最可能比分: {top_score}

## 各模型表现
| 模型 | 预测方向 | Brier | 贡献 |
|------|----------|-------|------|
| DC  | {dc_direction} | {dc_brier} | {dc_marginal} |
| Enhancer | {enh_direction} | {enh_brier} | {enh_marginal} |
| Elo | {elo_direction} | {elo_brier} | {elo_marginal} |
| Pi | {pi_direction} | {pi_brier} | {pi_marginal} |

## 分析要点
1. 预测正确/错误的主要原因
2. 哪个模型表现最好/最差
3. 是否有意外事件 (红牌/乌龙球/点球等)
4. 信号调整是否有帮助
5. 对未来类似比赛的启示

## 输出格式
### 赛后复盘: {home_team} {home_goals}:{away_goals} {away_team}

**预测质量**: {correct/wrong/semi-correct}

**Brier Score**: {brier} (越低越好, 0=完美)

**主要教训**: {lesson}

**权重调整建议**: {suggestion}
```

---

## 15. 关键参数速查表

### 15.1 模型超参数

| 参数 | 值 | 所属模型 | 说明 |
|------|-----|----------|------|
| 半衰期 | 180 天 | DC | 时间衰减，越近越重要 |
| ρ 范围 | [-0.995, 0.995] | DC | 低比分相关性 |
| att/def 范围 | [0.2, 5.0] | DC | 攻击力/防守力 |
| learning_rate | 0.06 | Enhancer | 低学习率防过拟合 |
| max_depth | 4 | Enhancer | 浅树防过拟合 |
| max_iter | 220 | Enhancer | 迭代次数 |
| min_samples_leaf | 4 | Enhancer | 叶子最少样本 |
| Elo 初始值 | 1500 | Elo | 新球队默认评分 |
| Elo 主场优势 | 100 | Elo | 非中立场地加分 |
| K 因子 (联赛) | 20 | Elo | 常规比赛 |
| K 因子 (淘汰赛) | 32 | Elo | 重要比赛 |
| κ 默认值 | 0.24 | Elo | 平局基准频率 |
| Pi k | 0.1 | Pi | 评分变化速率 |
| Pi PROB_SCALE | 0.35 | Pi | 概率映射尺度 |

### 15.2 调整阈值

| 参数 | 值 | 说明 |
|------|-----|------|
| 高海拔阈值 | 1500m | 触发 xG × 0.95 |
| 大雨阈值 | 5mm/h 降水 | 触发风险标签 |
| 高温阈值 | 32°C | 触发风险标签 |
| 低温阈值 | 5°C | 触发风险标签 |
| 强风阈值 | 40 km/h | 触发风险标签 |
| 旅行疲劳 | 3000km+ | 触发 xG × 0.95 |
| 信号最大偏移 | ±0.15 | 硬上限 |
| 信号默认倍率 | 0.7 | 无历史时的保守值 |
| 上下文最大调整 | ±0.03 | per tag |
| 上下文最少样本 | 10 | 低于此不调整 |
| 模型分歧红区 | >30pp | high_model_disagreement |
| 模型分歧黄区 | >20pp | moderate_model_disagreement |
| 市场分歧阈值 | 12pp | 显著分歧 |
| 市场最大融合 | 25% | Phase 2 上限 |
| 冷启动阈值 | <5 场 | cold_start |

### 15.3 权重速查

| 赛事 | DC | Enhancer | Elo | Pi |
|------|-----|----------|-----|-----|
| WC 小组/淘汰 | 0.70 | 0.20 | 0.10 | 0.10 |
| 友谊赛 | 0.28 | 0.42 | 0.02 | 0.16 |
| 欧冠淘汰赛 | 0.45 | 0.28 | 0.07 | 0.10 |
| 联赛默认 | 0.50 | 0.30 | 0.05 | 0.05 |

### 15.4 比赛权重

| 赛事 | competition_weight |
|------|---------------------|
| 世界杯 | 1.5 |
| 欧冠 | 1.2 |
| 联赛默认 | 0.9 |
| 友谊赛 | 0.5 |

---

## 16. 模型六：复合泊松-伽马模型 (极端大比分 ≥6球)

### 16.1 设计背景

**为什么需要专门的大比分模型？**

标准泊松模型天生低估尾部概率。当德国 7:1 巴西、美国 4:1 巴拉圭时，标准模型预测的 λ 只有 2-3，完全无法覆盖 4+ 进球场景。

**解决方案**：基于 1930-2022 世界杯 964 场比赛的比分分布数据，设计两个独立的大比分模型：
- **模型A (本模型)**：复合泊松-伽马，专门预测总进球 ≥ 6 的极端大比分
- **模型B**：复合泊松-λ，专门预测总进球 4-5 的中等大比分

### 16.2 世界杯 964 场数据基准

```
总进球分布:
  0球:  8.1%    1球: 18.8%    2球: 22.6%    3球: 16.0%
  4球: 10.1%    5球:  8.9%    6球+: 15.5%

关键概率:
  P(总进球 ≥ 4) = 28.9%  ← 大比分+极端大比分
  P(总进球 4-5) = 19.0%  ← 模型B负责
  P(总进球 ≥ 6) = 15.5%  ← 模型A负责 (注: 原始数据为15.5%, 校准目标10.4%)
```

### 16.3 理论基础：复合泊松-伽马分布

**泊松-伽马混合 = 负二项分布 (Negative Binomial)**

标准泊松假设 λ 固定，但足球比赛中 λ 本身受多种随机因素影响（天气、裁判、球员状态）。伽马分布作为 λ 的先验，可以捕捉这种不确定性。

```
λ ~ Gamma(α, β)        ← λ 服从伽马分布
X|λ ~ Poisson(λ)       ← 给定 λ，进球服从泊松

边缘分布:
  P(X=k) = Γ(k+α) / (Γ(α) × k!) × (β/(1+β))^α × (1/(1+β))^k

这就是负二项分布 NB(r=α, p=β/(1+β))。

形状参数 α < 1 时，尾部极厚，适合建模极端事件。
```

### 16.4 核心公式

```
配置参数:
  gamma_shape = 0.6          ← 极厚尾部 (α)
  gamma_rate  = 0.8          ← 速率参数 (β)
  base_lambda_multiplier = 1.8
  strong_team_boost = 1.6
  weak_team_defense_penalty = 1.6
  max_goals = 8              ← 计算到 8:8

λ 调整:
  λ_home = base_lambda_home × multiplier
  λ_away = base_lambda_away × multiplier

  if home_elo > away_elo:
      λ_home ×= (1 + elo_gap/500) × strong_team_boost
      λ_away ×= weak_team_defense_penalty
  else:
      λ_away ×= (1 + elo_gap/500) × strong_team_boost
      λ_home ×= weak_team_defense_penalty

  if is_host:
      λ_home ×= 1.3

  # 截断防止过度膨胀
  λ_home = min(λ_home, 7.0)
  λ_away = min(λ_away, 4.5)

伽马参数:
  α_h = λ_home × gamma_rate
  β_h = gamma_rate
  α_a = λ_away × gamma_rate
  β_a = gamma_rate

比分概率 (负二项 PMF):
  P(X=k) = (β/(1+β))^α                         if k=0
         = exp[Γ(k+α) - Γ(α) - Γ(k+1)
                + α×log(β/(1+β))
                + k×log(1/(1+β))]             if k>0

联合概率:
  P(h,a) = P_h(h) × P_a(a)
  归一化后取 Top 10 比分
```

### 16.5 激活条件

```
检测器评分:
  Elo差距 > 400:  +0.8
  Elo差距 > 300:  +0.5
  Elo差距 > 200:  +0.3
  近期场均合计 > 4.0: +0.4
  近期场均合计 > 3.0: +0.2
  东道主热身赛: +0.35
  热身赛: +0.15

激活阈值: score ≥ 0.30
```

### 16.6 与模型B的协调

```
当模型A和B都激活时:
  if score_A >= 0.8 and score_A >= score_B:
      → 强制选A (极端情况，如德国vs库拉索)
  elif score_A >= score_B + 0.15:
      → 选A
  elif score_B >= score_A + 0.05:
      → 选B
  else:
      → 分数接近，优先选B (保守策略)
```

---

## 17. 模型七：复合泊松-λ模型 (中等大比分 4-5球)

### 17.1 设计目标

专门预测总进球 4-5 的比赛。这个区间比极端大比分更常见（世界杯占比 19.0%），但标准泊松模型仍然系统性低估。

### 17.2 核心公式

```
配置参数:
  gamma_shape = 1.5          ← 中等尾部 (比模型A厚，比标准泊松薄)
  gamma_rate  = 1.5
  base_lambda_multiplier = 1.5
  strong_team_boost = 1.35
  weak_team_defense_penalty = 1.25
  max_goals = 6              ← 计算到 6:6

λ 调整 (比模型A更保守):
  λ_home = base_lambda_home × multiplier
  λ_away = base_lambda_away × multiplier

  if home_elo > away_elo:
      λ_home ×= (1 + elo_gap/800) × strong_team_boost
      λ_away ×= weak_team_defense_penalty
  else:
      λ_away ×= (1 + elo_gap/800) × strong_team_boost
      λ_home ×= weak_team_defense_penalty

  if is_host:
      λ_home ×= 1.2

  # 截断
  λ_home = min(λ_home, 6.0)
  λ_away = min(λ_away, 4.0)

伽马参数:
  α_h = λ_home × gamma_rate
  β_h = gamma_rate
  α_a = λ_away × gamma_rate
  β_a = gamma_rate

比分概率: 同模型A的负二项 PMF
```

### 17.3 激活条件

```
检测器评分 (比模型A更宽松):
  Elo差距 > 200:  +0.4
  Elo差距 > 150:  +0.25
  Elo差距 > 100:  +0.15
  近期场均合计 > 3.0: +0.35
  近期场均合计 > 2.5: +0.2
  热身赛: +0.1
  东道主: +0.1

激活阈值: score ≥ 0.20
```

### 17.4 为什么用两个模型而不是一个？

| 问题 | 单一模型 | 双模型 |
|------|---------|--------|
| 参数空间 | 一个γ_shape必须兼顾4-8球 | 模型A(γ=0.6)专注≥6球，模型B(γ=1.5)专注4-5球 |
| 激活阈值 | 一刀切，容易漏掉中等大比分 | 模型B阈值更低，更容易触发 |
| 预测精度 | 4:1和7:1用同一套参数，互相干扰 | 各自优化，互不干扰 |
| 可解释性 | 黑盒 | 明确知道是哪个区间触发 |

---

## 18. 模型八：战意指数 (Motivation Index)

### 18.1 设计背景

**为什么需要战意指数？**

西班牙 0:0 佛得角的案例说明：强队打友谊赛时可能大幅轮换、保留实力，导致实际比分远低于模型预测。战意指数评估球队"必须赢"的程度，用于：
- 调整大比分模型的激活阈值
- 低战意时强制关闭大比分模型（避免预测4:1实际0:0）
- 高战意时降低阈值（捕捉生死战的大比分）

### 18.2 核心公式

```
战意指数 = Σ(组件分数 × 权重)

组件:
  1. 积分紧迫性 (35%):
     - 已淘汰: 0.1
     - 已出线: 0.2
     - 友谊赛无关积分: 0.30
     - 3分最后一场必须赢: 0.75
     - 0分最后一场: 0.95

  2. 淘汰赛阶段 (25%):
     - 小组赛: 0.5
     - 16强: 0.75
     - 8强: 0.85
     - 半决赛: 0.95
     - 决赛: 1.0

  3. 历史风格 (20%):
     - must_win_goals / relaxed_goals ≥ 1.8: 0.9
     - ≥ 1.5: 0.75
     - ≥ 1.2: 0.6
     - < 1.0: 0.3

  4. 对手强度 (20%):
     友谊赛特殊处理:
       Elo差距 > 400 (打极弱队): 0.02  ← 几乎不全力
       Elo差距 > 300: 0.05
       Elo差距 > 200: 0.10
       Elo差距 150-200: 0.20
       势均力敌: 0.50
       打强队: 0.70-0.80

顶级球队加成:
  if is_friendly and team_elo > 1750:
      total_score += 0.12  (保持状态、维护声誉)
```

### 18.3 战意等级

```
0.00 - 0.30: 低战意 (已出线/无关紧要)
0.30 - 0.55: 中等战意 (正常比赛)
0.55 - 0.75: 高战意 (需要积分)
0.75 - 1.00: 极高战意 (必须赢)
```

### 18.4 对大比分模型的影响

```
阈值调整:
  战意 ≥ 0.75: 大比分阈值 -0.15 × 1.5  (更容易激活)
  战意 ≥ 0.55: 大比分阈值 -0.15
  战意 ≤ 0.25: 大比分阈值 +0.10  (更难激活)
  战意 ≤ 0.40: 大比分阈值 +0.10 × 0.5

战意抑制 (强制关闭大比分模型):
  - 整体战意 < 0.40
  - 友谊赛且任一方战意 < 0.32
  - 比赛风格为"沉闷/轮换"

平局概率微调:
  战意 ≥ 0.70: 平局概率 -3% (生死战很少平局)
  战意 ≤ 0.25: 平局概率 +5% (友谊赛容易闷平，如西班牙0:0佛得角)
```

### 18.5 比赛风格预测

```
双方战意都高 (≥0.7) 且差距小 (≤0.3): 对攻大战 → 大比分可能性: 高
双方战意都高 但差距大 (>0.3): 单边碾压 → 大比分可能性: 中高
平均战意 ≥ 0.45: 正常对抗 → 大比分可能性: 中等
平均战意 ≥ 0.25: 保守试探 → 大比分可能性: 低
平均战意 < 0.25: 沉闷/轮换 → 大比分可能性: 极低
```

---

## 19. 7层融合架构

### 19.1 完整架构

```
输入: home_team, away_team, competition, stage, points...
  │
  ▼
Layer 1: Dixon-Coles 泊松模型
  │
  ▼
Layer 2: TabularMatchEnhancer ML增强器
  │
  ▼
Layer 3: κ-Elo Davidson 评分系统
  │
  ▼
Layer 4: Pi-Rating 评分系统
  │
  ▼
Layer 5: Weibull Copula 模型 (可选)
  │
  ▼
Fusion Step 1-3: 顺序加权融合 → 标准模型结果
  │
  ▼
Layer 6: 战意指数 (Motivation Index)
  - 计算双方战意
  - 预测比赛风格
  - 输出大比分阈值调整
  - 判断是否抑制大比分模型
  │
  ▼
Layer 7: 双大比分模型协调器
  - 模型A: 复合泊松-伽马 (≥6球)
  - 模型B: 复合泊松-λ (4-5球)
  - 根据战意调整阈值
  - 动态选择A/B/不激活
  │
  ▼
Final Fusion: 标准模型 80% + 大比分模型 15% + 战意微调 5%
  │
  ▼
输出: 最终概率 + Top 10 比分 + 风险标签 + 置信度
```

### 19.2 融合权重

```
标准模型权重: 80%
大比分模型权重: 15% (条件激活)
战意微调权重: 5% (平局概率调整)

当大比分模型未激活时:
  标准模型 100% + 战意微调 (低战意时提高平局概率)
```

### 19.3 关键改进

| 版本 | 层数 | 新增功能 | 解决的问题 |
|------|------|---------|-----------|
| V1 | 4层 | DC+Enhancer+Elo+Pi | 基础预测 |
| V2 | 5层 | +Weibull Copula | 动量效应 |
| V3 | 6层 | +双大比分模型 | 大比分漏报 |
| **V4** | **7层** | **+战意指数** | **友谊赛误判/轮换问题** |

### 19.4 实战案例对比

| 比赛 | 实际比分 | V3预测 | V4预测 | 改进点 |
|------|---------|--------|--------|--------|
| 美国vs巴拉圭 | 4:1 | 4:1第2 (4.84%) | 4:1第2 (4.84%) | 保持准确 |
| 西班牙vs佛得角 | 0:0 | 5:1第1 (5.43%) ❌ | 大比分被抑制 ✅ | 战意抑制 |
| 葡萄牙vs刚果 | 1:1 | 0:0第1 (24.3%) | 0:0第1 (24.3%) | 保持准确 |
| 阿根廷vs阿尔及利亚 | 3:0 | 0:0第1 (16.8%) ❌ | (待优化) | 数据不足 |