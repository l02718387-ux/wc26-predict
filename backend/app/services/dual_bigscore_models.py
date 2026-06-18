"""
双大比分预测模型 — 基于世界杯964场比分分布数据

设计思路:
根据用户提供的1930-2022世界杯964场比赛比分热力图，将大比分预测拆分为两个独立模型:

模型A: 复合泊松-伽马 (Poisson-Gamma) — 专门预测总进球 ≥ 6 的极端大比分
模型B: 复合泊松-λ (Poisson-Lambda) — 专门预测总进球 4-5 的中等大比分

这样两个模型互不干扰，各自优化自己的参数空间。

世界杯964场比分分布数据 (从热力图提取):

总进球分布:
  0球:  0:0=78 (8.1%)                    → 总计 8.1%
  1球:  1:0=118(12.2%) + 0:1=64(6.6%)   → 总计 18.8%
  2球:  2:0=71(7.4%) + 1:1=107(11.1%) + 0:2=40(4.1%)  → 总计 22.6%
  3球:  3:0=36(3.7%) + 2:1=52(5.4%) + 1:2=45(4.7%) + 0:3=21(2.2%) → 总计 16.0%
  4球:  4:0=20(2.1%) + 3:1=25(2.6%) + 2:2=32(3.3%) + 1:3=16(1.7%) + 0:4=4(0.4%) → 总计 10.1%
  5球:  5:0=7(0.7%) + 4:1=25(2.6%) + 3:2=35(3.6%) + 2:3=11(1.1%) + 1:4=6(0.6%) + 0:5=3(0.3%) → 总计 8.9%
  6球+: 6:0=0 + 5:1=4(0.4%) + 4:2=15(1.6%) + 3:3=7(0.7%) + ... → 总计 ~15.5%

关键发现:
- 总进球 0-3 球: 65.5% (大多数比赛)
- 总进球 4-5 球: 19.0% (中等大比分，模型B负责)
- 总进球 ≥6 球: 15.5% (极端大比分，模型A负责)
- 1:1 是最常见比分 (11.1%)，其次是 1:0 (12.2%) 和 2:1 (5.4%)

作者: WC26 Predict Enhancement V3
"""

import numpy as np
from scipy.special import gammaln, gamma
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# ==================== 世界杯964场比分分布数据 ====================

WORLD_CUP_SCORE_DISTRIBUTION = {
    # 总进球数 → 出现次数和百分比
    'total_goals': {
        0: {'count': 78, 'pct': 8.1},
        1: {'count': 182, 'pct': 18.8},
        2: {'count': 218, 'pct': 22.6},
        3: {'count': 154, 'pct': 16.0},
        4: {'count': 97, 'pct': 10.1},
        5: {'count': 86, 'pct': 8.9},
        6: {'count': 49, 'pct': 5.1},
        7: {'count': 28, 'pct': 2.9},
        8: {'count': 12, 'pct': 1.2},
        9: {'count': 5, 'pct': 0.5},
        10: {'count': 2, 'pct': 0.2},
    },
    # 常见比分 → 百分比 (用于校准)
    'common_scores': {
        '1:0': 12.2, '0:1': 6.6,
        '2:0': 7.4, '1:1': 11.1, '0:2': 4.1,
        '3:0': 3.7, '2:1': 5.4, '1:2': 4.7, '0:3': 2.2,
        '4:0': 2.1, '3:1': 2.6, '2:2': 3.3, '1:3': 1.7,
        '5:0': 0.7, '4:1': 2.6, '3:2': 3.6, '2:3': 1.1,
    }
}

# 从分布数据计算关键概率
TOTAL_MATCHES = 964

# 总进球 ≥ 4 的概率 (大比分+极端大比分)
P_TOTAL_4_PLUS = sum(v['pct'] for k, v in WORLD_CUP_SCORE_DISTRIBUTION['total_goals'].items() if k >= 4) / 100
# 总进球 ≥ 6 的概率 (极端大比分)
P_TOTAL_6_PLUS = sum(v['pct'] for k, v in WORLD_CUP_SCORE_DISTRIBUTION['total_goals'].items() if k >= 6) / 100
# 总进球 4-5 的概率 (中等大比分)
P_TOTAL_4_5 = sum(v['pct'] for k, v in WORLD_CUP_SCORE_DISTRIBUTION['total_goals'].items() if 4 <= k <= 5) / 100

print(f"世界杯964场数据基准:")
print(f"  P(总进球 ≥ 4) = {P_TOTAL_4_PLUS:.1%}")
print(f"  P(总进球 4-5) = {P_TOTAL_4_5:.1%}  ← 模型B")
print(f"  P(总进球 ≥ 6) = {P_TOTAL_6_PLUS:.1%}  ← 模型A")


# ==================== 模型A: 复合泊松-伽马 (极端大比分 ≥6球) ====================

@dataclass
class ExtremeBigScoreConfig:
    """极端大比分模型配置 — 专门预测总进球 ≥ 6"""
    # 触发条件
    min_total_goals: int = 6           # 预测范围: 总进球 ≥ 6
    activation_threshold: float = 0.30  # 激活阈值 (比通用模型更低，因为极端事件更稀有)
    
    # 复合泊松-伽马参数 — 极厚尾部
    # 形状参数 < 1 产生极厚尾部，适合极端事件
    gamma_shape: float = 0.6           # 极厚尾部 (0.6 比 1.0 更厚)
    gamma_rate: float = 0.8            # 速率参数
    
    # λ 调整 — 针对极端大比分
    base_lambda_multiplier: float = 1.8  # 基础 λ 乘数 (控制膨胀)
    strong_team_boost: float = 1.6      # 强队提升 60%
    weak_team_defense_penalty: float = 1.6  # 弱队防守差 → 对方 λ 提升 60%
    
    # 比分矩阵
    max_goals: int = 8                 # 计算到 8:8
    
    # 校准目标: P(总进球 ≥ 6) ≈ 10.4% (世界杯数据)
    target_p_6plus: float = 0.104


class ExtremeBigScoreModel:
    """
    极端大比分预测模型 (Model A)
    
    专门预测总进球 ≥ 6 的比赛。
    使用极厚尾部的复合泊松-伽马分布。
    """
    
    def __init__(self, config: Optional[ExtremeBigScoreConfig] = None):
        self.config = config or ExtremeBigScoreConfig()
    
    def should_activate(self, home_elo: float, away_elo: float,
                        home_recent_goals_avg: float = 1.5,
                        away_recent_goals_avg: float = 1.0,
                        is_warmup: bool = False,
                        is_host: bool = False) -> Dict:
        """
        判断是否可能产生极端大比分 (≥6球)
        
        触发条件 (满足任意一条):
        1. Elo 差距 > 300 分 (极强 vs 极弱)
        2. 双方近期场均进球合计 > 4.0
        3. 世界杯东道主热身赛
        4. 一方近期连续大比分
        """
        score = 0.0
        triggers = []
        
        elo_gap = abs(home_elo - away_elo)
        if elo_gap > 400:
            score += 0.8
            triggers.append(f'Elo差距{elo_gap:.0f}>400')
        elif elo_gap > 300:
            score += 0.5
            triggers.append(f'Elo差距{elo_gap:.0f}>300')
        elif elo_gap > 200:
            score += 0.3
            triggers.append(f'Elo差距{elo_gap:.0f}>200')
        
        combined_recent_goals = home_recent_goals_avg + away_recent_goals_avg
        if combined_recent_goals > 4.0:
            score += 0.4
            triggers.append(f'近期场均合计{combined_recent_goals:.1f}>4.0')
        elif combined_recent_goals > 3.0:
            score += 0.2
            triggers.append(f'近期场均合计{combined_recent_goals:.1f}>3.0')
        
        if is_host and is_warmup:
            score += 0.35
            triggers.append('东道主热身赛')
        elif is_warmup:
            score += 0.15
            triggers.append('热身赛')
        
        is_likely = score >= self.config.activation_threshold
        
        return {
            'is_likely': is_likely,
            'score': round(score, 3),
            'triggers': triggers,
            'target_range': '总进球 ≥ 6'
        }
    
    def predict(self, home_team: str, away_team: str,
                home_elo: float, away_elo: float,
                base_lambda_home: float = 1.5,
                base_lambda_away: float = 1.0,
                is_host: bool = False) -> Optional[Dict]:
        """预测极端大比分"""
        
        # 调整 λ — 极端大比分需要极高的 λ
        lambda_h = base_lambda_home * self.config.base_lambda_multiplier
        lambda_a = base_lambda_away * self.config.base_lambda_multiplier
        
        elo_gap = abs(home_elo - away_elo)
        
        if home_elo > away_elo:
            lambda_h *= (1 + elo_gap / 500) * self.config.strong_team_boost
            lambda_a *= self.config.weak_team_defense_penalty
        else:
            lambda_a *= (1 + elo_gap / 500) * self.config.strong_team_boost
            lambda_h *= self.config.weak_team_defense_penalty
        
        if is_host:
            lambda_h *= 1.3
        
        # 截断防止过度膨胀
        lambda_h = min(lambda_h, 7.0)
        lambda_a = min(lambda_a, 4.5)
        
        # 复合泊松-伽马参数
        alpha_h = lambda_h * self.config.gamma_rate
        beta_h = self.config.gamma_rate
        alpha_a = lambda_a * self.config.gamma_rate
        beta_a = self.config.gamma_rate
        
        # 生成比分矩阵
        matrix, hda, scores = self._compute_matrix(alpha_h, beta_h, alpha_a, beta_a)
        
        # 计算 P(总进球 ≥ 6)
        p_6plus = sum(matrix[h, a] for h in range(self.config.max_goals + 1)
                      for a in range(self.config.max_goals + 1) if h + a >= 6)
        
        return {
            'model': 'ExtremeBigScore (Poisson-Gamma)',
            'target': '总进球 ≥ 6',
            'xg': {'home': round(alpha_h / beta_h, 2), 'away': round(alpha_a / beta_a, 2)},
            'lambda': {'home': round(lambda_h, 2), 'away': round(lambda_a, 2)},
            'hda': {'home': round(hda[0], 4), 'draw': round(hda[1], 4), 'away': round(hda[2], 4)},
            'p_total_6plus': round(p_6plus, 4),
            'top_scores': [{'score': s[0], 'prob': round(s[1] * 100, 2)} for s in scores[:10]],
            'params': {'alpha_h': round(alpha_h, 2), 'beta_h': beta_h,
                       'alpha_a': round(alpha_a, 2), 'beta_a': beta_a}
        }
    
    def _compute_matrix(self, alpha_h, beta_h, alpha_a, beta_a):
        size = self.config.max_goals + 1
        matrix = np.zeros((size, size))
        
        for h in range(size):
            for a in range(size):
                p_h = self._nb_pmf(h, alpha_h, beta_h)
                p_a = self._nb_pmf(a, alpha_a, beta_a)
                matrix[h, a] = p_h * p_a
        
        matrix /= matrix.sum()
        
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        scores = []
        for h in range(size):
            for a in range(size):
                scores.append((f"{h}:{a}", matrix[h, a]))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return matrix, np.array([home_win, draw, away_win]), scores
    
    def _nb_pmf(self, k, alpha, beta):
        if k == 0:
            return (beta / (1 + beta)) ** alpha
        log_p = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1) +
                 alpha * np.log(beta / (1 + beta)) +
                 k * np.log(1 / (1 + beta)))
        return float(np.exp(log_p))


# ==================== 模型B: 复合泊松-λ (中等大比分 4-5球) ====================

@dataclass
class MediumBigScoreConfig:
    """中等大比分模型配置 — 专门预测总进球 4-5"""
    # 触发条件
    target_total_goals_min: int = 4     # 预测范围: 总进球 4-5
    target_total_goals_max: int = 5
    activation_threshold: float = 0.20  # 激活阈值 (更宽松，热身赛单独可触发)
    
    # 复合泊松-λ参数 — 中等厚度尾部
    gamma_shape: float = 1.5            # 中等尾部 (1.5 比 0.6 薄，比 2.0 厚)
    gamma_rate: float = 1.5
    
    # λ 调整 — 针对中等大比分
    base_lambda_multiplier: float = 1.5  # 中等提升
    strong_team_boost: float = 1.35     # 强队提升 35%
    weak_team_defense_penalty: float = 1.25  # 弱队防守差 → 对方 λ 提升 25%
    
    # 比分矩阵
    max_goals: int = 6                 # 计算到 6:6
    
    # 校准目标: P(总进球 4-5) ≈ 19.0% (世界杯数据)
    target_p_4_5: float = 0.190


class MediumBigScoreModel:
    """
    中等大比分预测模型 (Model B)
    
    专门预测总进球 4-5 的比赛。
    使用中等厚度的复合泊松-λ分布。
    """
    
    def __init__(self, config: Optional[MediumBigScoreConfig] = None):
        self.config = config or MediumBigScoreConfig()
    
    def should_activate(self, home_elo: float, away_elo: float,
                        home_recent_goals_avg: float = 1.5,
                        away_recent_goals_avg: float = 1.0,
                        is_warmup: bool = False,
                        is_host: bool = False) -> Dict:
        """
        判断是否可能产生中等大比分 (4-5球)
        
        触发条件 (比极端模型更宽松):
        1. Elo 差距 > 150 分
        2. 双方近期场均进球合计 > 2.5
        3. 热身赛
        4. 一方近期连续有进球
        """
        score = 0.0
        triggers = []
        
        elo_gap = abs(home_elo - away_elo)
        if elo_gap > 200:
            score += 0.4
            triggers.append(f'Elo差距{elo_gap:.0f}>200')
        elif elo_gap > 150:
            score += 0.25
            triggers.append(f'Elo差距{elo_gap:.0f}>150')
        elif elo_gap > 100:
            score += 0.15
            triggers.append(f'Elo差距{elo_gap:.0f}>100')
        
        combined_recent_goals = home_recent_goals_avg + away_recent_goals_avg
        if combined_recent_goals > 3.0:
            score += 0.35
            triggers.append(f'近期场均合计{combined_recent_goals:.1f}>3.0')
        elif combined_recent_goals > 2.5:
            score += 0.2
            triggers.append(f'近期场均合计{combined_recent_goals:.1f}>2.5')
        
        if is_warmup:
            score += 0.1
            triggers.append('热身赛')
        
        if is_host:
            score += 0.1
            triggers.append('东道主')
        
        is_likely = score >= self.config.activation_threshold
        
        return {
            'is_likely': is_likely,
            'score': round(score, 3),
            'triggers': triggers,
            'target_range': '总进球 4-5'
        }
    
    def predict(self, home_team: str, away_team: str,
                home_elo: float, away_elo: float,
                base_lambda_home: float = 1.5,
                base_lambda_away: float = 1.0,
                is_host: bool = False) -> Optional[Dict]:
        """预测中等大比分"""
        
        lambda_h = base_lambda_home * self.config.base_lambda_multiplier
        lambda_a = base_lambda_away * self.config.base_lambda_multiplier
        
        elo_gap = abs(home_elo - away_elo)
        
        if home_elo > away_elo:
            lambda_h *= (1 + elo_gap / 800) * self.config.strong_team_boost
            lambda_a *= self.config.weak_team_defense_penalty
        else:
            lambda_a *= (1 + elo_gap / 800) * self.config.strong_team_boost
            lambda_h *= self.config.weak_team_defense_penalty
        
        if is_host:
            lambda_h *= 1.2
        
        # 截断
        lambda_h = min(lambda_h, 6.0)
        lambda_a = min(lambda_a, 4.0)
        
        # 复合泊松参数
        alpha_h = lambda_h * self.config.gamma_rate
        beta_h = self.config.gamma_rate
        alpha_a = lambda_a * self.config.gamma_rate
        beta_a = self.config.gamma_rate
        
        matrix, hda, scores = self._compute_matrix(alpha_h, beta_h, alpha_a, beta_a)
        
        # 计算 P(总进球 4-5)
        p_4_5 = sum(matrix[h, a] for h in range(self.config.max_goals + 1)
                    for a in range(self.config.max_goals + 1) if 4 <= h + a <= 5)
        
        return {
            'model': 'MediumBigScore (Poisson-Lambda)',
            'target': '总进球 4-5',
            'xg': {'home': round(alpha_h / beta_h, 2), 'away': round(alpha_a / beta_a, 2)},
            'lambda': {'home': round(lambda_h, 2), 'away': round(lambda_a, 2)},
            'hda': {'home': round(hda[0], 4), 'draw': round(hda[1], 4), 'away': round(hda[2], 4)},
            'p_total_4_5': round(p_4_5, 4),
            'top_scores': [{'score': s[0], 'prob': round(s[1] * 100, 2)} for s in scores[:10]],
            'params': {'alpha_h': round(alpha_h, 2), 'beta_h': beta_h,
                       'alpha_a': round(alpha_a, 2), 'beta_a': beta_a}
        }
    
    def _compute_matrix(self, alpha_h, beta_h, alpha_a, beta_a):
        size = self.config.max_goals + 1
        matrix = np.zeros((size, size))
        
        for h in range(size):
            for a in range(size):
                p_h = self._nb_pmf(h, alpha_h, beta_h)
                p_a = self._nb_pmf(a, alpha_a, beta_a)
                matrix[h, a] = p_h * p_a
        
        matrix /= matrix.sum()
        
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        scores = []
        for h in range(size):
            for a in range(size):
                scores.append((f"{h}:{a}", matrix[h, a]))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return matrix, np.array([home_win, draw, away_win]), scores
    
    def _nb_pmf(self, k, alpha, beta):
        if k == 0:
            return (beta / (1 + beta)) ** alpha
        log_p = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1) +
                 alpha * np.log(beta / (1 + beta)) +
                 k * np.log(1 / (1 + beta)))
        return float(np.exp(log_p))


# ==================== 双模型协调器 ====================

class DualBigScoreCoordinator:
    """
    双大比分模型协调器
    
    同时运行模型A和模型B，根据检测结果决定使用哪个。
    两个模型互不干扰，各自独立判断。
    """
    
    def __init__(self):
        self.model_a = ExtremeBigScoreModel()
        self.model_b = MediumBigScoreModel()
    
    def predict(self, home_team: str, away_team: str,
                home_elo: float, away_elo: float,
                base_lambda_home: float = 1.5,
                base_lambda_away: float = 1.0,
                home_recent_goals_avg: float = 1.5,
                away_recent_goals_avg: float = 1.0,
                is_warmup: bool = False,
                is_host: bool = False) -> Dict:
        """
        双模型预测流程:
        1. 同时检测模型A和模型B
        2. 如果都激活，优先使用模型A (极端大比分)
        3. 如果只激活一个，使用激活的
        4. 如果都没激活，返回空
        """
        
        # 同时检测
        detect_a = self.model_a.should_activate(
            home_elo, away_elo, home_recent_goals_avg,
            away_recent_goals_avg, is_warmup, is_host
        )
        detect_b = self.model_b.should_activate(
            home_elo, away_elo, home_recent_goals_avg,
            away_recent_goals_avg, is_warmup, is_host
        )
        
        result_a = None
        result_b = None
        
        # 模型A预测 (如果激活)
        if detect_a['is_likely']:
            result_a = self.model_a.predict(
                home_team, away_team, home_elo, away_elo,
                base_lambda_home, base_lambda_away, is_host
            )
        
        # 模型B预测 (如果激活)
        if detect_b['is_likely']:
            result_b = self.model_b.predict(
                home_team, away_team, home_elo, away_elo,
                base_lambda_home, base_lambda_away, is_host
            )
        
        # 决定使用哪个模型 — 根据检测分数动态选择
        selected_model = None
        selection_reason = ""
        
        if result_a and result_b:
            # 都激活 — 比较检测分数，选分数更高的
            score_a = detect_a['score']
            score_b = detect_b['score']
            # 极端情况：A检测分很高(>=0.8)且A>=B，强制选A (如德国vs库拉索Elo差距400+)
            if score_a >= 0.8 and score_a >= score_b:
                selected_model = result_a
                selection_reason = f"模型A和B都激活，A检测分{score_a}达到极端阈值且>=B{score_b}，强制选A"
            elif score_a >= score_b + 0.15:
                selected_model = result_a
                selection_reason = f"模型A和B都激活，A检测分{score_a}显著高于B{score_b}，选A"
            elif score_b >= score_a + 0.05:
                selected_model = result_b
                selection_reason = f"模型A和B都激活，B检测分{score_b}高于A{score_a}，选B (中等大比分)"
            else:
                # 分数接近，优先B (中等大比分更常见，避免过度预测极端)
                selected_model = result_b
                selection_reason = f"模型A和B都激活且分数接近(A={score_a}, B={score_b})，优先选B (保守策略)"
        elif result_a:
            selected_model = result_a
            selection_reason = "仅模型A激活 (极端大比分)"
        elif result_b:
            selected_model = result_b
            selection_reason = "仅模型B激活 (中等大比分)"
        else:
            selection_reason = "两个模型都未激活"
        
        return {
            'detection_a': detect_a,
            'detection_b': detect_b,
            'result_a': result_a,
            'result_b': result_b,
            'selected_model': selected_model,
            'selection_reason': selection_reason
        }


# ==================== 测试 ====================

def test_usa_vs_paraguay():
    """测试美国 vs 巴拉圭 — 实际 4:1 (总进球 5)"""
    print("\n" + "=" * 70)
    print("【测试】美国 vs 巴拉圭 — 实际 4:1 (总进球 5)")
    print("=" * 70)
    
    coordinator = DualBigScoreCoordinator()
    
    result = coordinator.predict(
        home_team='United States',
        away_team='Paraguay',
        home_elo=1689,
        away_elo=1650,
        base_lambda_home=2.0,
        base_lambda_away=1.0,
        home_recent_goals_avg=2.5,  # 美国近期进攻不错
        away_recent_goals_avg=1.0,
        is_warmup=True,
        is_host=True  # 世界杯东道主
    )
    
    print(f"\n模型A检测 (极端大比分 ≥6球):")
    print(f"  激活: {result['detection_a']['is_likely']} (分数: {result['detection_a']['score']})")
    print(f"  触发: {result['detection_a']['triggers']}")
    
    print(f"\n模型B检测 (中等大比分 4-5球):")
    print(f"  激活: {result['detection_b']['is_likely']} (分数: {result['detection_b']['score']})")
    print(f"  触发: {result['detection_b']['triggers']}")
    
    print(f"\n选择结果: {result['selection_reason']}")
    
    if result['selected_model']:
        m = result['selected_model']
        print(f"\n选中模型: {m['model']}")
        print(f"目标范围: {m['target']}")
        print(f"xG: {m['xg']['home']} - {m['xg']['away']}")
        print(f"λ: {m['lambda']['home']} - {m['lambda']['away']}")
        print(f"\nTop 5 比分:")
        for i, s in enumerate(m['top_scores'][:5], 1):
            print(f"  {i}. {s['score']:>5s}  ({s['prob']:5.2f}%)")
        
        # 检查 4:1 是否在 Top 10
        score_4_1 = next((s for s in m['top_scores'] if s['score'] == '4:1'), None)
        if score_4_1:
            print(f"\n✅ 4:1 在预测中! 概率: {score_4_1['prob']:.2f}%")
        else:
            print(f"\n❌ 4:1 不在 Top 10")
    
    return result


def test_germany_vs_curacao():
    """测试德国 vs 库拉索 — 预期大比分"""
    print("\n" + "=" * 70)
    print("【测试】德国 vs 库拉索 — 强弱悬殊")
    print("=" * 70)
    
    coordinator = DualBigScoreCoordinator()
    
    result = coordinator.predict(
        home_team='Germany',
        away_team='Curaçao',
        home_elo=1722,
        away_elo=1300,
        base_lambda_home=2.5,
        base_lambda_away=0.4,
        home_recent_goals_avg=3.0,
        away_recent_goals_avg=0.5,
        is_warmup=True,
        is_host=False
    )
    
    print(f"\n模型A检测 (极端大比分 ≥6球):")
    print(f"  激活: {result['detection_a']['is_likely']} (分数: {result['detection_a']['score']})")
    
    print(f"\n模型B检测 (中等大比分 4-5球):")
    print(f"  激活: {result['detection_b']['is_likely']} (分数: {result['detection_b']['score']})")
    
    print(f"\n选择结果: {result['selection_reason']}")
    
    if result['selected_model']:
        m = result['selected_model']
        print(f"\n选中模型: {m['model']}")
        print(f"Top 5 比分:")
        for i, s in enumerate(m['top_scores'][:5], 1):
            print(f"  {i}. {s['score']:>5s}  ({s['prob']:5.2f}%)")
    
    return result


def test_england_vs_croatia():
    """测试英格兰 vs 克罗地亚 — 实际 4:2 (总进球 6)"""
    print("\n" + "=" * 70)
    print("【测试】英格兰 vs 克罗地亚 — 实际 4:2 (总进球 6)")
    print("=" * 70)
    
    coordinator = DualBigScoreCoordinator()
    
    result = coordinator.predict(
        home_team='England',
        away_team='Croatia',
        home_elo=1743,
        away_elo=1715,
        base_lambda_home=2.2,
        base_lambda_away=1.4,
        home_recent_goals_avg=2.8,
        away_recent_goals_avg=1.5,
        is_warmup=True,
        is_host=False
    )
    
    print(f"\n模型A检测 (极端大比分 ≥6球):")
    print(f"  激活: {result['detection_a']['is_likely']} (分数: {result['detection_a']['score']})")
    print(f"  触发: {result['detection_a']['triggers']}")
    
    print(f"\n模型B检测 (中等大比分 4-5球):")
    print(f"  激活: {result['detection_b']['is_likely']} (分数: {result['detection_b']['score']})")
    print(f"  触发: {result['detection_b']['triggers']}")
    
    print(f"\n选择结果: {result['selection_reason']}")
    
    if result['selected_model']:
        m = result['selected_model']
        print(f"\n选中模型: {m['model']}")
        print(f"Top 5 比分:")
        for i, s in enumerate(m['top_scores'][:5], 1):
            print(f"  {i}. {s['score']:>5s}  ({s['prob']:5.2f}%)")
        
        # 检查 4:2
        score_4_2 = next((s for s in m['top_scores'] if s['score'] == '4:2'), None)
        if score_4_2:
            print(f"\n✅ 4:2 在预测中! 概率: {score_4_2['prob']:.2f}%")
    
    return result


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + "  双大比分预测模型 — 基于世界杯964场数据".center(66) + "█")
    print("█" * 70)
    
    print(f"\n世界杯964场数据基准:")
    print(f"  P(总进球 0-3)  = {1 - P_TOTAL_4_PLUS:.1%} (标准模型)")
    print(f"  P(总进球 4-5)  = {P_TOTAL_4_5:.1%} (模型B: 中等大比分)")
    print(f"  P(总进球 ≥ 6)  = {P_TOTAL_6_PLUS:.1%} (模型A: 极端大比分)")
    
    r1 = test_usa_vs_paraguay()
    r2 = test_germany_vs_curacao()
    r3 = test_england_vs_croatia()
    
    print("\n" + "=" * 70)
    print("【测试总结】")
    print("=" * 70)
    for name, r in [('美国vs巴拉圭', r1), ('德国vs库拉索', r2), ('英格兰vs克罗地亚', r3)]:
        a_active = r['detection_a']['is_likely']
        b_active = r['detection_b']['is_likely']
        sel = r['selected_model']
        selected = 'A' if sel and 'Extreme' in sel['model'] else ('B' if sel else '无')
        print(f"  {name}: A={'✅' if a_active else '❌'} B={'✅' if b_active else '❌'} → 选中模型{selected}")
    print("=" * 70)
