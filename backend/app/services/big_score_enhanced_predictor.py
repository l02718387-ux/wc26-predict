"""
大比分预测增强模块 — 针对泊松模型无法预测大比分（4+ 球）的系统性缺陷

核心改进:
1. 复合泊松 (Poisson-Gamma) — 允许方差 > 均值，增加尾部厚度
2. 比赛强度检测 — 区分"认真踢的热身赛"和"轮换友谊赛"
3. 比分矩阵扩展到 7×7 — 覆盖 6:6 以内所有比分
4. 大比分历史衰减 — 近期大比分比赛提升 λ
5. 负二项分布备选 — 当泊松尾部过薄时自动切换

作者: WC26 Predict Enhancement
"""

import numpy as np
from scipy import stats
from scipy.special import gamma, gammaln, factorial
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json


@dataclass
class BigScoreConfig:
    """大比分预测配置参数"""
    # 复合泊松参数 — 关键: 小 shape 产生厚尾
    gamma_shape: float = 1.5          # Gamma 形状参数 (越小尾部越厚, 1.0=指数分布)
    gamma_rate: float = 1.5           # Gamma 速率参数
    
    # 比赛强度检测
    warmup_boost_threshold: int = 30  # 距离大赛 <30 天视为热身赛
    warmup_lambda_boost: float = 1.6  # 热身赛 λ 提升 60% (原40%不够)
    
    # 大比分历史衰减
    big_score_window: int = 5         # 近 N 场检测大比分
    big_score_threshold: int = 4      # 单场总进球 ≥4 视为大比分
    big_score_lambda_boost: float = 1.4  # 有大比分历史时 λ 提升 40%
    
    # 比分矩阵扩展
    max_goals_matrix: int = 6         # 比分矩阵计算到 6:6
    
    # 负二项分布切换阈值
    nb_switch_threshold: float = 1.5  # 当观测方差/均值 > 1.5 时切换
    
    #  Dixon-Coles ρ 参数调整
    rho_big_score_adjustment: float = -0.15  # 大比分场景下降低 ρ


class CompoundPoissonGoalsModel:
    """
    复合泊松进球模型 (Poisson-Gamma Mixture)
    
    标准泊松: P(X=k|λ) = λ^k × e^(-λ) / k!
    复合泊松: λ ~ Gamma(α, β), 然后 X ~ Poisson(λ)
    
    边缘分布是负二项分布:
    P(X=k) = Γ(k+α) / (k! × Γ(α)) × (β/(1+β))^α × (1/(1+β))^k
    
    为什么更好:
    - 泊松: E[X] = Var[X] = λ (方差=均值)
    - 复合泊松: E[X] = α/β, Var[X] = α/β + α/β² (方差 > 均值)
    - 允许"过度分散" — 足球比赛中常见现象
    """
    
    def __init__(self, config: Optional[BigScoreConfig] = None):
        self.config = config or BigScoreConfig()
        self.alpha_home = self.config.gamma_shape
        self.beta_home = self.config.gamma_rate
        self.alpha_away = self.config.gamma_shape
        self.beta_away = self.config.gamma_rate
        self.rho = 0.0  # Dixon-Coles 低比分相关性
    
    def fit(self, home_goals: np.ndarray, away_goals: np.ndarray,
            weights: Optional[np.ndarray] = None) -> 'CompoundPoissonGoalsModel':
        """
        拟合复合泊松参数
        
        使用矩估计法:
        E[X] = α/β, Var[X] = α/β + α/β²
        → α = E[X]² / (Var[X] - E[X])
        → β = E[X] / (Var[X] - E[X])
        """
        if weights is None:
            weights = np.ones_like(home_goals, dtype=float)
        
        # 加权均值
        mean_h = np.average(home_goals, weights=weights)
        mean_a = np.average(away_goals, weights=weights)
        
        # 加权方差 (无偏估计)
        var_h = np.average((home_goals - mean_h)**2, weights=weights)
        var_a = np.average((away_goals - mean_a)**2, weights=weights)
        
        # 矩估计 α, β
        # 负二项分布: mean = α/β, var = α/β + α/β²
        # → var - mean = α/β²
        # → mean / (var - mean) = β
        # → mean × β = α
        
        if var_h > mean_h * 1.1:  # 确保过度分散
            self.beta_home = mean_h / (var_h - mean_h)
            self.alpha_home = mean_h * self.beta_home
        else:
            # 方差不够大，退化为泊松 (α→∞, β→∞, α/β=λ)
            self.alpha_home = 1000.0
            self.beta_home = 1000.0 / mean_h if mean_h > 0 else 1000.0
        
        if var_a > mean_a * 1.1:
            self.beta_away = mean_a / (var_a - mean_a)
            self.alpha_away = mean_a * self.beta_away
        else:
            self.alpha_away = 1000.0
            self.beta_away = 1000.0 / mean_a if mean_a > 0 else 1000.0
        
        # 拟合 ρ (Dixon-Coles 低比分相关性)
        self.rho = self._fit_rho(home_goals, away_goals, weights)
        
        return self
    
    def _fit_rho(self, home_goals: np.ndarray, away_goals: np.ndarray,
                 weights: np.ndarray) -> float:
        """拟合 Dixon-Coles τ 修正参数 ρ"""
        # 简化的 ρ 估计: 基于 0-0, 1-0, 0-1, 1-1 的观测频率 vs 期望频率
        n = len(home_goals)
        
        # 观测频率
        obs_00 = np.sum(weights[(home_goals == 0) & (away_goals == 0)]) / np.sum(weights)
        obs_10 = np.sum(weights[(home_goals == 1) & (away_goals == 0)]) / np.sum(weights)
        obs_01 = np.sum(weights[(home_goals == 0) & (away_goals == 1)]) / np.sum(weights)
        obs_11 = np.sum(weights[(home_goals == 1) & (away_goals == 1)]) / np.sum(weights)
        
        # 使用复合泊松的期望 (近似用均值代替)
        lambda_h = self.alpha_home / self.beta_home
        lambda_a = self.alpha_away / self.beta_away
        
        # 泊松期望
        exp_00_poisson = np.exp(-lambda_h - lambda_a)
        exp_10_poisson = lambda_h * np.exp(-lambda_h - lambda_a)
        exp_01_poisson = lambda_a * np.exp(-lambda_h - lambda_a)
        exp_11_poisson = lambda_h * lambda_a * np.exp(-lambda_h - lambda_a)
        
        # 估计 ρ (简化版)
        if exp_00_poisson > 0:
            rho_est = (exp_00_poisson - obs_00) / (lambda_h * lambda_a * exp_00_poisson)
            return float(np.clip(rho_est, -0.3, 0.3))
        return -0.05  # 默认值
    
    def predict_score_probability(self, home_goals: int, away_goals: int) -> float:
        """
        预测特定比分的概率 (复合泊松 + Dixon-Coles τ 修正)
        
        P(X=x, Y=y) = τ(x,y) × NB(x|α_h, β_h) × NB(y|α_a, β_a)
        """
        # 负二项分布概率质量函数
        def nb_pmf(k: int, alpha: float, beta: float) -> float:
            """负二项分布 PMF: P(X=k)"""
            if k == 0:
                return (beta / (1 + beta)) ** alpha
            # 使用对数计算防溢出
            log_p = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1) +
                     alpha * np.log(beta / (1 + beta)) +
                     k * np.log(1 / (1 + beta)))
            return np.exp(log_p)
        
        # Dixon-Coles τ 修正
        lambda_h = self.alpha_home / self.beta_home
        lambda_a = self.alpha_away / self.beta_away
        
        if home_goals == 0 and away_goals == 0:
            tau = 1.0 - lambda_h * lambda_a * self.rho
        elif home_goals == 1 and away_goals == 0:
            tau = 1.0 + lambda_a * self.rho
        elif home_goals == 0 and away_goals == 1:
            tau = 1.0 + lambda_h * self.rho
        elif home_goals == 1 and away_goals == 1:
            tau = 1.0 - self.rho
        else:
            tau = 1.0
        
        tau = max(0.1, min(2.0, tau))  # 约束
        
        p_h = nb_pmf(home_goals, self.alpha_home, self.beta_home)
        p_a = nb_pmf(away_goals, self.alpha_away, self.beta_away)
        
        return tau * p_h * p_a
    
    def predict_proba(self, max_goals: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成完整的比分概率矩阵
        
        Returns:
            score_matrix: (max_goals+1) × (max_goals+1) 概率矩阵
            home_probs: [P(主胜), P(平), P(客胜)]
            top_scores: [(score, prob), ...] 排序后的比分列表
        """
        size = max_goals + 1
        matrix = np.zeros((size, size))
        
        for h in range(size):
            for a in range(size):
                matrix[h, a] = self.predict_score_probability(h, a)
        
        # 归一化
        matrix /= matrix.sum()
        
        # 计算胜负平概率
        home_win = np.sum(np.tril(matrix, -1))  # 下三角 (主队进球 > 客队)
        draw = np.sum(np.diag(matrix))            # 对角线
        away_win = np.sum(np.triu(matrix, 1))     # 上三角 (客队进球 > 主队)
        
        # 收集所有比分并排序
        scores = []
        for h in range(size):
            for a in range(size):
                scores.append((f"{h}:{a}", matrix[h, a]))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return matrix, np.array([home_win, draw, away_win]), scores
    
    def get_expected_goals(self) -> Tuple[float, float]:
        """返回期望进球 xG"""
        return self.alpha_home / self.beta_home, self.alpha_away / self.beta_away


class MatchIntensityDetector:
    """
    比赛强度检测器 — 区分"认真踢的热身赛"和"轮换友谊赛"
    """
    
    def __init__(self, config: Optional[BigScoreConfig] = None):
        self.config = config or BigScoreConfig()
    
    def detect(self, match_date: str, tournament_start_date: Optional[str] = None,
               team_lineup_strength: Optional[float] = None,
               historical_intensity: Optional[float] = None) -> dict:
        """
        检测比赛强度并返回调整系数
        
        Args:
            match_date: 比赛日期 (YYYY-MM-DD)
            tournament_start_date: 大赛开始日期 (如世界杯 2026-06-11)
            team_lineup_strength: 预期首发强度 (0-1, 1=全主力)
            historical_intensity: 该队历史友谊赛平均进球数
        
        Returns:
            {
                'intensity_level': 'high' | 'medium' | 'low',
                'lambda_boost': float,  # λ 调整系数
                'is_warmup': bool,      # 是否热身赛
                'reason': str           # 判断理由
            }
        """
        from datetime import datetime
        
        result = {
            'intensity_level': 'medium',
            'lambda_boost': 1.0,
            'is_warmup': False,
            'reason': '默认中等强度'
        }
        
        # 检测 1: 距离大赛时间
        if tournament_start_date:
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            tour_dt = datetime.strptime(tournament_start_date, "%Y-%m-%d")
            days_until = (tour_dt - match_dt).days
            
            if 0 < days_until <= self.config.warmup_boost_threshold:
                result['is_warmup'] = True
                result['intensity_level'] = 'high'
                result['lambda_boost'] = self.config.warmup_lambda_boost
                result['reason'] = f'距离大赛 {days_until} 天，世界杯前热身赛'
                return result
        
        # 检测 2: 首发强度
        if team_lineup_strength is not None:
            if team_lineup_strength >= 0.8:
                result['intensity_level'] = 'high'
                result['lambda_boost'] = max(result['lambda_boost'], 1.25)
                result['reason'] += '; 预计全主力首发'
            elif team_lineup_strength <= 0.4:
                result['intensity_level'] = 'low'
                result['lambda_boost'] = min(result['lambda_boost'], 0.85)
                result['reason'] += '; 预计大幅轮换'
        
        # 检测 3: 历史强度
        if historical_intensity is not None:
            if historical_intensity >= 3.0:  # 历史场均总进球 > 3
                result['intensity_level'] = 'high'
                result['lambda_boost'] = max(result['lambda_boost'], 1.2)
                result['reason'] += f'; 历史友谊赛场均 {historical_intensity:.1f} 球'
        
        return result


class BigScoreEnhancedPredictor:
    """
    大比分增强预测器 — 整合所有改进
    """
    
    def __init__(self, config: Optional[BigScoreConfig] = None):
        self.config = config or BigScoreConfig()
        self.model = CompoundPoissonGoalsModel(config)
        self.intensity_detector = MatchIntensityDetector(config)
    
    def predict(self, home_team: str, away_team: str,
                match_date: str,
                tournament_start_date: Optional[str] = None,
                team_lineup_strength: Optional[float] = None,
                historical_intensity: Optional[float] = None,
                recent_big_scores: Optional[List[Tuple[int, int]]] = None,
                base_lambda_home: float = 1.2,
                base_lambda_away: float = 0.9) -> dict:
        """
        完整的大比分增强预测流程
        
        Args:
            home_team: 主队名称
            away_team: 客队名称
            match_date: 比赛日期
            tournament_start_date: 大赛开始日期 (可选)
            team_lineup_strength: 首发强度 0-1 (可选)
            historical_intensity: 历史友谊赛场均进球 (可选)
            recent_big_scores: 近期大比分比赛列表 [(h,a), ...] (可选)
            base_lambda_home: 基础主队期望进球
            base_lambda_away: 基础客队期望进球
        
        Returns:
            完整的预测结果字典
        """
        # Step 1: 检测比赛强度
        intensity = self.intensity_detector.detect(
            match_date, tournament_start_date,
            team_lineup_strength, historical_intensity
        )
        
        # Step 2: 调整 λ
        lambda_h = base_lambda_home * intensity['lambda_boost']
        lambda_a = base_lambda_away * intensity['lambda_boost']
        
        # Step 3: 检测近期大比分历史
        if recent_big_scores:
            big_score_count = sum(1 for h, a in recent_big_scores 
                                  if h + a >= self.config.big_score_threshold)
            if big_score_count >= 2:  # 近 N 场中有 2+ 场大比分
                lambda_h *= self.config.big_score_lambda_boost
                lambda_a *= self.config.big_score_lambda_boost
                intensity['reason'] += f'; 近期 {big_score_count} 场大比分'
        
        # Step 4: 设置复合泊松参数
        # 从 λ 反推 α, β: λ = α/β, 设 β=2 → α=2λ
        self.model.alpha_home = lambda_h * self.config.gamma_rate
        self.model.beta_home = self.config.gamma_rate
        self.model.alpha_away = lambda_a * self.config.gamma_rate
        self.model.beta_away = self.config.gamma_rate
        
        # Step 5: 调整 ρ (大比分场景降低低比分相关性)
        if intensity['intensity_level'] == 'high':
            self.model.rho = self.config.rho_big_score_adjustment
        
        # Step 6: 生成预测
        matrix, hda_probs, top_scores = self.model.predict_proba(
            self.config.max_goals_matrix
        )
        
        xg_h, xg_a = self.model.get_expected_goals()
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date,
            'intensity': intensity,
            'lambda_adjusted': {'home': lambda_h, 'away': lambda_a},
            'xg': {'home': xg_h, 'away': xg_a},
            'hda_probabilities': {
                'home_win': float(hda_probs[0]),
                'draw': float(hda_probs[1]),
                'away_win': float(hda_probs[2])
            },
            'top_10_scores': [
                {'score': s[0], 'probability': round(s[1] * 100, 2)}
                for s in top_scores[:10]
            ],
            'score_matrix': matrix.tolist(),
            'model_params': {
                'alpha_home': round(self.model.alpha_home, 3),
                'beta_home': round(self.model.beta_home, 3),
                'alpha_away': round(self.model.alpha_away, 3),
                'beta_away': round(self.model.beta_away, 3),
                'rho': round(self.model.rho, 3)
            }
        }


# ==================== 使用示例 ====================

def demo_england_vs_croatia():
    """
    用增强模型重新预测英格兰 vs 克罗地亚
    假设这是世界杯前热身赛，双方全主力
    """
    predictor = BigScoreEnhancedPredictor()
    
    result = predictor.predict(
        home_team='England',
        away_team='Croatia',
        match_date='2026-06-01',  # 假设日期
        tournament_start_date='2026-06-11',  # 世界杯前 10 天
        team_lineup_strength=0.9,  # 预计 90% 主力
        historical_intensity=3.2,  # 历史友谊赛场均 3.2 球
        recent_big_scores=[(4, 2), (3, 1), (2, 2), (1, 0), (2, 1)],  # 近期有大比分
        base_lambda_home=1.8,  # 英格兰基础 xG (提升, 英格兰进攻强)
        base_lambda_away=1.1   # 克罗地亚基础 xG
    )
    
    print("=" * 60)
    print(f"  大比分增强预测: {result['home_team']} vs {result['away_team']}")
    print(f"  比赛日期: {result['match_date']}")
    print(f"  强度检测: {result['intensity']['intensity_level'].upper()}")
    print(f"  判断理由: {result['intensity']['reason']}")
    print(f"  λ 调整: 主队 {result['lambda_adjusted']['home']:.2f}, 客队 {result['lambda_adjusted']['away']:.2f}")
    print()
    print(f"  xG: {result['xg']['home']:.2f} - {result['xg']['away']:.2f}")
    print()
    print("  === 最终融合概率 ===")
    hda = result['hda_probabilities']
    print(f"  主胜: {hda['home_win']*100:.1f}%")
    print(f"  平局: {hda['draw']*100:.1f}%")
    print(f"  客胜: {hda['away_win']*100:.1f}%")
    print()
    print("  === Top 10 比分预测 ===")
    for i, s in enumerate(result['top_10_scores'], 1):
        print(f"  {i:2d}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
    print()
    print("  === 模型参数 ===")
    params = result['model_params']
    print(f"  α_home={params['alpha_home']}, β_home={params['beta_home']}")
    print(f"  α_away={params['alpha_away']}, β_away={params['beta_away']}")
    print(f"  ρ={params['rho']}")
    print("=" * 60)
    
    return result


def demo_standard_friendly():
    """
    普通友谊赛预测 — 无大赛临近，轮换阵容
    """
    predictor = BigScoreEnhancedPredictor()
    
    result = predictor.predict(
        home_team='Team A',
        away_team='Team B',
        match_date='2026-03-15',
        tournament_start_date=None,  # 无大赛
        team_lineup_strength=0.3,    # 大幅轮换
        historical_intensity=1.8,    # 历史场均 1.8 球
        recent_big_scores=[(1, 0), (0, 0), (1, 1), (0, 1), (2, 0)],
        base_lambda_home=1.0,
        base_lambda_away=0.8
    )
    
    print("\n" + "=" * 60)
    print(f"  普通友谊赛预测: {result['home_team']} vs {result['away_team']}")
    print(f"  强度检测: {result['intensity']['intensity_level'].upper()}")
    print(f"  判断理由: {result['intensity']['reason']}")
    print(f"  λ 调整: 主队 {result['lambda_adjusted']['home']:.2f}, 客队 {result['lambda_adjusted']['away']:.2f}")
    print()
    print("  === Top 5 比分预测 ===")
    for i, s in enumerate(result['top_10_scores'][:5], 1):
        print(f"  {i}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    # 运行示例
    print("\n【示例 1: 世界杯前热身赛 — 英格兰 vs 克罗地亚】")
    demo_england_vs_croatia()
    
    print("\n【示例 2: 普通轮换友谊赛】")
    demo_standard_friendly()
    
    # 对比: 标准泊松 vs 复合泊松的大比分概率
    print("\n【对比: 泊松 vs 复合泊松的大比分概率】")
    print("假设 λ_home=1.96, λ_away=1.26 (热身赛调整后)")
    
    # 标准泊松
    lambda_h, lambda_a = 1.96, 1.26
    poisson_4_2 = (lambda_h**4 * np.exp(-lambda_h) / factorial(4)) * \
                  (lambda_a**2 * np.exp(-lambda_a) / factorial(2))
    print(f"  标准泊松 P(4:2) = {poisson_4_2*100:.3f}%")
    
    # 复合泊松 (α=3.92, β=2)
    alpha_h, beta_h = 3.92, 2.0
    alpha_a, beta_a = 2.52, 2.0
    
    def nb_pmf(k, alpha, beta):
        if k == 0:
            return (beta / (1 + beta)) ** alpha
        log_p = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1) +
                 alpha * np.log(beta / (1 + beta)) +
                 k * np.log(1 / (1 + beta)))
        return np.exp(log_p)
    
    compound_4_2 = nb_pmf(4, alpha_h, beta_h) * nb_pmf(2, alpha_a, beta_a)
    print(f"  复合泊松 P(4:2) = {compound_4_2*100:.3f}%")
    print(f"  提升倍数: {compound_4_2 / poisson_4_2:.1f}x")
    
    # P(总进球 ≥ 4)
    poisson_total_4plus = 1 - stats.poisson.cdf(3, lambda_h + lambda_a)
    compound_total_4plus = 1 - sum(
        nb_pmf(h, alpha_h, beta_h) * nb_pmf(a, alpha_a, beta_a)
        for h in range(4) for a in range(4)
        if h + a <= 3
    )
    print(f"\n  标准泊松 P(总进球≥4) = {poisson_total_4plus*100:.1f}%")
    print(f"  复合泊松 P(总进球≥4) = {compound_total_4plus*100:.1f}%")