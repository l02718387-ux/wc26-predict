"""
大比分预测模型 — 独立模块

设计目标:
1. 作为第 6 个独立模型，与 DC/Enhancer/Elo/Pi/Weibull 并列
2. 每次预测时先进行"大比分可能性检测"
3. 只有当检测器认为"可能出现大比分"时才激活此模型
4. 检测结果作为风险标签输出，供用户参考

触发条件 (满足任意一条即激活):
- 两队 Elo 差距 > 200 分 (强弱悬殊)
- 近期 3 场内有 2+ 场大比分 (总进球 ≥4)
- 世界杯/大赛前热身赛 (距离 <30 天)
- 历史对阵场均进球 > 3.5
- 一方是进攻型球队且另一方防守薄弱

模型核心: 复合泊松-伽马 (Poisson-Gamma) +  Dixon-Coles τ 修正
"""

import numpy as np
from scipy.special import gammaln
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import json


@dataclass
class BigScoreModelConfig:
    """大比分模型配置 — V2 优化版"""
    # 触发阈值
    elo_gap_threshold: float = 200.0      # Elo 差距 > 200 视为强弱悬殊
    recent_big_score_threshold: int = 2   # 近 3 场中 ≥2 场大比分
    big_score_total_goals: int = 4        # 单场总进球 ≥4 为大比分
    warmup_days_threshold: int = 30       # 距离大赛 <30 天
    historical_avg_goals_threshold: float = 3.5  # 历史场均 > 3.5 球
    
    # 复合泊松参数 — V2.1 修正: shape 不能太小，否则 λ = α/β 会失控
    # 原 0.8 导致 λ 膨胀到 23，改回 1.2 但用不同的方式控制尾部
    gamma_shape: float = 1.2            # Gamma 形状 (恢复 1.2)
    gamma_rate: float = 1.2             # Gamma 速率 (恢复 1.2)
    
    # λ 调整 — 适度提升，避免过度膨胀
    # 德国 7:1 → 德国 λ ~5-6, 库拉索 λ ~0.5
    # 英格兰 4:2 → 双方 λ ~2.5-3.5
    strong_team_lambda_boost: float = 1.8   # 强队 λ 提升 80%
    weak_team_lambda_penalty: float = 0.6   # 弱队进攻被压制到 60%
    weak_team_defense_penalty: float = 1.4  # 弱队防守差 → 对方 λ 额外提升 40%
    warmup_lambda_boost: float = 1.4        # 热身赛 λ 提升 40%
    
    # 比分矩阵 — 扩展到 8×8 覆盖 7:7
    max_goals: int = 7                  # 计算到 7:7 (原 6:6 不够，德国 7:1 超出范围)
    
    # Dixon-Coles ρ 调整
    rho_default: float = -0.05          # 默认低比分相关性
    rho_big_score: float = -0.25        # 大比分场景降低 ρ (原 -0.20 不够)


class BigScoreDetector:
    """
    大比分可能性检测器
    
    每次预测前调用，返回是否激活大比分模型
    """
    
    def __init__(self, config: Optional[BigScoreModelConfig] = None):
        self.config = config or BigScoreModelConfig()
    
    def detect(self,
               home_elo: float,
               away_elo: float,
               match_date: str,
               tournament_start_date: Optional[str] = None,
               recent_scores: Optional[List[Tuple[int, int]]] = None,
               historical_avg_goals: Optional[float] = None,
               home_team_style: Optional[str] = None,   # 'attacking' | 'balanced' | 'defensive'
               away_team_style: Optional[str] = None) -> Dict:
        """
        检测是否可能产生大比分
        
        Returns:
            {
                'is_big_score_likely': bool,      # 是否可能大比分
                'activation_score': float,         # 激活分数 (0-1, 越高越可能)
                'triggered_rules': List[str],      # 触发了哪些规则
                'recommendation': str               # 建议
            }
        """
        triggered = []
        score = 0.0
        
        # 规则 1: Elo 差距悬殊
        elo_gap = abs(home_elo - away_elo)
        if elo_gap > self.config.elo_gap_threshold:
            triggered.append(f'Elo差距{elo_gap:.0f}分 > 阈值{self.config.elo_gap_threshold}')
            score += min(0.4, elo_gap / 1000)
        
        # 规则 2: 近期大比分历史
        if recent_scores:
            big_count = sum(1 for h, a in recent_scores[-3:] 
                          if h + a >= self.config.big_score_total_goals)
            if big_count >= self.config.recent_big_score_threshold:
                triggered.append(f'近3场有{big_count}场大比分(≥{self.config.big_score_total_goals}球)')
                score += 0.3
        
        # 规则 3: 大赛前热身赛
        if tournament_start_date:
            from datetime import datetime
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            tour_dt = datetime.strptime(tournament_start_date, "%Y-%m-%d")
            days_until = (tour_dt - match_dt).days
            if 0 < days_until <= self.config.warmup_days_threshold:
                triggered.append(f'距离大赛{days_until}天，热身赛')
                score += 0.25
        
        # 规则 4: 历史场均进球高
        if historical_avg_goals and historical_avg_goals > self.config.historical_avg_goals_threshold:
            triggered.append(f'历史场均{historical_avg_goals:.1f}球 > 阈值{self.config.historical_avg_goals_threshold}')
            score += 0.2
        
        # 规则 5: 攻强守弱组合
        if home_team_style == 'attacking' and away_team_style == 'defensive':
            triggered.append('攻强 vs 守弱组合')
            score += 0.15
        
        # 综合判断
        is_likely = score >= 0.35  # 激活阈值
        
        recommendation = '激活大比分模型' if is_likely else '使用标准模型'
        if is_likely and score >= 0.6:
            recommendation = '强烈建议激活大比分模型 — 高概率出现大比分'
        
        return {
            'is_big_score_likely': is_likely,
            'activation_score': round(min(score, 1.0), 3),
            'triggered_rules': triggered,
            'recommendation': recommendation,
            'elo_gap': elo_gap
        }


class BigScorePredictorModel:
    """
    大比分预测模型 — 第 6 个独立模型
    
    基于复合泊松-伽马分布，专门预测高进球比赛
    """
    
    def __init__(self, config: Optional[BigScoreModelConfig] = None):
        self.config = config or BigScoreModelConfig()
        self.detector = BigScoreDetector(config)
    
    def predict(self,
                home_team: str,
                away_team: str,
                home_elo: float,
                away_elo: float,
                match_date: str,
                base_lambda_home: float = 1.2,
                base_lambda_away: float = 0.9,
                tournament_start_date: Optional[str] = None,
                recent_scores: Optional[List[Tuple[int, int]]] = None,
                historical_avg_goals: Optional[float] = None,
                home_team_style: Optional[str] = None,
                away_team_style: Optional[str] = None) -> Dict:
        """
        完整预测流程:
        1. 检测是否可能大比分
        2. 如果可能，用复合泊松预测
        3. 返回完整结果 + 检测信息
        """
        
        # Step 1: 检测
        detection = self.detector.detect(
            home_elo, away_elo, match_date,
            tournament_start_date, recent_scores,
            historical_avg_goals, home_team_style, away_team_style
        )
        
        # Step 2: 如果不激活，返回空结果 + 检测信息
        if not detection['is_big_score_likely']:
            return {
                'activated': False,
                'detection': detection,
                'message': '大比分模型未激活 — 未满足触发条件',
                'home_team': home_team,
                'away_team': away_team
            }
        
        # Step 3: 计算调整后的 λ
        lambda_h = base_lambda_home
        lambda_a = base_lambda_away
        
        # Elo 差距调整 — V2 优化
        # 关键洞察: 大比分不仅是"强队进得多"，更是"弱队防守差让强队进更多"
        elo_gap = detection['elo_gap']
        if home_elo > away_elo:
            # 主队强 → 主队 λ 大幅提升 (进攻强)
            lambda_h *= self.config.strong_team_lambda_boost
            lambda_h *= (1 + elo_gap / 800)  # 差距越大提升越多 (原 1000 太保守，改 800)
            # 客队弱 → 客队进攻被压制
            lambda_a *= self.config.weak_team_lambda_penalty
            lambda_a *= max(0.5, 1 - elo_gap / 2000)  # 差距越大客队进攻越弱
            # 关键新增: 弱队防守差 → 主队 λ 再提升 (防守漏洞)
            lambda_h *= self.config.weak_team_defense_penalty
        else:
            lambda_a *= self.config.strong_team_lambda_boost
            lambda_a *= (1 + elo_gap / 800)
            lambda_h *= self.config.weak_team_lambda_penalty
            lambda_h *= max(0.5, 1 - elo_gap / 2000)
            lambda_a *= self.config.weak_team_defense_penalty
        
        # 热身赛调整
        if tournament_start_date:
            from datetime import datetime
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            tour_dt = datetime.strptime(tournament_start_date, "%Y-%m-%d")
            days_until = (tour_dt - match_dt).days
            if 0 < days_until <= self.config.warmup_days_threshold:
                lambda_h *= self.config.warmup_lambda_boost
                lambda_a *= self.config.warmup_lambda_boost
        
        # 近期大比分调整
        if recent_scores:
            big_count = sum(1 for h, a in recent_scores[-3:]
                          if h + a >= self.config.big_score_total_goals)
            if big_count >= 2:
                boost = 1 + (big_count - 1) * 0.15  # 每场额外 15%
                lambda_h *= boost
                lambda_a *= boost
        
        # Step 4: 设置复合泊松参数 — V2.1 修正
        # λ = α/β, 设 β=gamma_rate → α = λ × β
        # 修正: 恢复标准参数化，但控制 λ 的范围
        # 德国 λ 应该在 5-6 左右，英格兰 λ 在 2.5-3.5 左右
        # 如果 λ 超过 8，强制截断到 8
        lambda_h = min(lambda_h, 8.0)
        lambda_a = min(lambda_a, 5.0)
        
        alpha_h = lambda_h * self.config.gamma_rate
        beta_h = self.config.gamma_rate
        alpha_a = lambda_a * self.config.gamma_rate
        beta_a = self.config.gamma_rate
        
        # Step 5: Dixon-Coles ρ
        rho = self.config.rho_big_score if detection['activation_score'] > 0.5 else self.config.rho_default
        
        # Step 6: 生成比分矩阵
        matrix, hda_probs, top_scores = self._compute_score_matrix(
            alpha_h, beta_h, alpha_a, beta_a, rho
        )
        
        xg_h = alpha_h / beta_h
        xg_a = alpha_a / beta_a
        
        return {
            'activated': True,
            'detection': detection,
            'home_team': home_team,
            'away_team': away_team,
            'xg': {'home': round(xg_h, 2), 'away': round(xg_a, 2)},
            'lambda': {'home': round(lambda_h, 2), 'away': round(lambda_a, 2)},
            'hda_probabilities': {
                'home_win': round(float(hda_probs[0]), 4),
                'draw': round(float(hda_probs[1]), 4),
                'away_win': round(float(hda_probs[2]), 4)
            },
            'top_10_scores': [
                {'score': s[0], 'probability': round(s[1] * 100, 2)}
                for s in top_scores[:10]
            ],
            'model_params': {
                'alpha_home': round(alpha_h, 3),
                'beta_home': round(beta_h, 3),
                'alpha_away': round(alpha_a, 3),
                'beta_away': round(beta_a, 3),
                'rho': round(rho, 3)
            }
        }
    
    def _compute_score_matrix(self, alpha_h: float, beta_h: float,
                              alpha_a: float, beta_a: float,
                              rho: float) -> Tuple[np.ndarray, np.ndarray, List]:
        """计算比分概率矩阵"""
        size = self.config.max_goals + 1
        matrix = np.zeros((size, size))
        
        lambda_h = alpha_h / beta_h
        lambda_a = alpha_a / beta_a
        
        for h in range(size):
            for a in range(size):
                # 负二项分布 PMF
                p_h = self._nb_pmf(h, alpha_h, beta_h)
                p_a = self._nb_pmf(a, alpha_a, beta_a)
                
                # Dixon-Coles τ 修正
                tau = self._tau(h, a, lambda_h, lambda_a, rho)
                
                matrix[h, a] = tau * p_h * p_a
        
        # 归一化
        total = matrix.sum()
        if total > 0:
            matrix /= total
        
        # H/D/A 概率
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        # 排序比分
        scores = []
        for h in range(size):
            for a in range(size):
                scores.append((f"{h}:{a}", matrix[h, a]))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return matrix, np.array([home_win, draw, away_win]), scores
    
    def _nb_pmf(self, k: int, alpha: float, beta: float) -> float:
        """负二项分布概率质量函数"""
        if k == 0:
            return (beta / (1 + beta)) ** alpha
        log_p = (gammaln(k + alpha) - gammaln(alpha) - gammaln(k + 1) +
                 alpha * np.log(beta / (1 + beta)) +
                 k * np.log(1 / (1 + beta)))
        return float(np.exp(log_p))
    
    def _tau(self, h: int, a: int, lambda_h: float, lambda_a: float, rho: float) -> float:
        """Dixon-Coles τ 修正"""
        if h == 0 and a == 0:
            return max(0.1, 1.0 - lambda_h * lambda_a * rho)
        elif h == 1 and a == 0:
            return min(2.0, 1.0 + lambda_a * rho)
        elif h == 0 and a == 1:
            return min(2.0, 1.0 + lambda_h * rho)
        elif h == 1 and a == 1:
            return max(0.1, 1.0 - rho)
        return 1.0


# ==================== 测试函数 ====================

def test_germany_vs_curacao():
    """测试德国 vs 库拉索 — 强弱悬殊，应该触发大比分模型"""
    print("\n" + "=" * 70)
    print("【测试】德国 vs 库拉索 — 强弱悬殊，预期触发大比分模型")
    print("=" * 70)
    
    model = BigScorePredictorModel()
    
    result = model.predict(
        home_team='Germany',
        away_team='Curaçao',
        home_elo=1722,           # 德国 Elo
        away_elo=1300,           # 库拉索 Elo (估计，弱队)
        match_date='2026-06-01',
        base_lambda_home=2.5,   # 德国基础进攻强 (提升以匹配 7:1)
        base_lambda_away=0.4,   # 库拉索基础进攻弱 (降低，弱队进攻更弱)
        tournament_start_date='2026-06-11',  # 世界杯前热身赛
        recent_scores=[(4, 2), (3, 1), (2, 0)],  # 德国近期有大比分
        historical_avg_goals=3.8,  # 历史场均高
        home_team_style='attacking',
        away_team_style='defensive'
    )
    
    print(f"\n检测信息:")
    det = result['detection']
    print(f"  激活分数: {det['activation_score']}")
    print(f"  触发规则: {det['triggered_rules']}")
    print(f"  建议: {det['recommendation']}")
    print(f"  Elo 差距: {det['elo_gap']:.0f} 分")
    
    if result['activated']:
        print(f"\n✅ 大比分模型已激活!")
        print(f"\n预测结果:")
        print(f"  xG: {result['xg']['home']} - {result['xg']['away']}")
        print(f"  λ: {result['lambda']['home']} - {result['lambda']['away']}")
        print(f"\n  胜负平概率:")
        hda = result['hda_probabilities']
        print(f"    德国胜: {hda['home_win']*100:.1f}%")
        print(f"    平局:   {hda['draw']*100:.1f}%")
        print(f"    库拉索胜: {hda['away_win']*100:.1f}%")
        print(f"\n  Top 10 比分:")
        for i, s in enumerate(result['top_10_scores'], 1):
            print(f"    {i:2d}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
        print(f"\n  模型参数: α_h={result['model_params']['alpha_home']}, β_h={result['model_params']['beta_home']}")
        print(f"            α_a={result['model_params']['alpha_away']}, β_a={result['model_params']['beta_away']}")
        print(f"            ρ={result['model_params']['rho']}")
    else:
        print(f"\n❌ 大比分模型未激活")
        print(f"  原因: {result['message']}")
    
    return result


def test_belgium_vs_egypt():
    """测试比利时 vs 埃及 — 实力接近，不应触发大比分模型"""
    print("\n" + "=" * 70)
    print("【测试】比利时 vs 埃及 — 实力接近，预期不触发大比分模型")
    print("=" * 70)
    
    model = BigScorePredictorModel()
    
    result = model.predict(
        home_team='Belgium',
        away_team='Egypt',
        home_elo=1728,
        away_elo=1697,
        match_date='2026-06-01',
        base_lambda_home=1.2,
        base_lambda_away=1.0,
        recent_scores=[(1, 1), (0, 0), (1, 0)],  # 近期无大比分
        historical_avg_goals=2.1  # 历史场均正常
    )
    
    print(f"\n检测信息:")
    det = result['detection']
    print(f"  激活分数: {det['activation_score']}")
    print(f"  触发规则: {det['triggered_rules']}")
    print(f"  建议: {det['recommendation']}")
    print(f"  Elo 差距: {det['elo_gap']:.0f} 分")
    
    if result['activated']:
        print(f"\n⚠️  意外激活了!")
    else:
        print(f"\n✅ 正确未激活 — 使用标准模型")
    
    return result


def test_england_vs_croatia_warmup():
    """测试英格兰 vs 克罗地亚 — 世界杯前热身赛，应触发"""
    print("\n" + "=" * 70)
    print("【测试】英格兰 vs 克罗地亚 — 世界杯前热身赛，预期触发")
    print("=" * 70)
    
    model = BigScorePredictorModel()
    
    result = model.predict(
        home_team='England',
        away_team='Croatia',
        home_elo=1743,
        away_elo=1715,
        match_date='2026-06-01',
        base_lambda_home=2.2,   # 英格兰进攻强 (提升)
        base_lambda_away=1.4,   # 克罗地亚也不弱 (提升，实际进了 2 球)
        tournament_start_date='2026-06-11',
        recent_scores=[(4, 2), (3, 1), (2, 2)],  # 近期大比分
        historical_avg_goals=3.2,
        home_team_style='attacking',
        away_team_style='balanced'
    )
    
    print(f"\n检测信息:")
    det = result['detection']
    print(f"  激活分数: {det['activation_score']}")
    print(f"  触发规则: {det['triggered_rules']}")
    print(f"  建议: {det['recommendation']}")
    
    if result['activated']:
        print(f"\n✅ 大比分模型已激活!")
        print(f"\n  xG: {result['xg']['home']} - {result['xg']['away']}")
        print(f"\n  Top 5 比分:")
        for i, s in enumerate(result['top_10_scores'][:5], 1):
            print(f"    {i}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
    else:
        print(f"\n❌ 未激活")
    
    return result


def test_usa_vs_paraguay():
    """测试美国 vs 巴拉圭 — 世界杯东道主 vs 南美球队"""
    print("\n" + "=" * 70)
    print("【测试】美国 vs 巴拉圭 — 世界杯东道主")
    print("=" * 70)
    
    model = BigScorePredictorModel()
    
    result = model.predict(
        home_team='United States',
        away_team='Paraguay',
        home_elo=1689,           # 美国 Elo
        away_elo=1650,           # 巴拉圭 Elo (估计)
        match_date='2026-06-01',
        base_lambda_home=1.5,   # 美国基础进攻
        base_lambda_away=1.0,   # 巴拉圭基础进攻
        tournament_start_date='2026-06-11',  # 世界杯前热身赛
        recent_scores=[(2, 1), (1, 0), (3, 2)],  # 美国近期
        historical_avg_goals=2.5,  # 历史场均
        home_team_style='balanced',
        away_team_style='defensive'
    )
    
    print(f"\n检测信息:")
    det = result['detection']
    print(f"  激活分数: {det['activation_score']}")
    print(f"  触发规则: {det['triggered_rules']}")
    print(f"  建议: {det['recommendation']}")
    print(f"  Elo 差距: {det['elo_gap']:.0f} 分")
    
    if result['activated']:
        print(f"\n✅ 大比分模型已激活!")
        print(f"\n预测结果:")
        print(f"  xG: {result['xg']['home']} - {result['xg']['away']}")
        print(f"  λ: {result['lambda']['home']} - {result['lambda']['away']}")
        print(f"\n  胜负平概率:")
        hda = result['hda_probabilities']
        print(f"    美国胜: {hda['home_win']*100:.1f}%")
        print(f"    平局:   {hda['draw']*100:.1f}%")
        print(f"    巴拉圭胜: {hda['away_win']*100:.1f}%")
        print(f"\n  Top 10 比分:")
        for i, s in enumerate(result['top_10_scores'], 1):
            print(f"    {i:2d}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
    else:
        print(f"\n❌ 大比分模型未激活")
        print(f"  原因: {result['message']}")
    
    return result


if __name__ == "__main__":
    # 运行四个测试
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  大比分预测模型 — 独立模块测试".center(64) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    r1 = test_germany_vs_curacao()
    r2 = test_belgium_vs_egypt()
    r3 = test_england_vs_croatia_warmup()
    r4 = test_usa_vs_paraguay()
    
    # 总结
    print("\n" + "=" * 70)
    print("【测试总结】")
    print("=" * 70)
    print(f"\n德国 vs 库拉索: {'✅ 激活' if r1['activated'] else '❌ 未激活'} (分数: {r1['detection']['activation_score']})")
    print(f"比利时 vs 埃及: {'✅ 激活' if r2['activated'] else '❌ 未激活'} (分数: {r2['detection']['activation_score']})")
    print(f"英格兰 vs 克罗地亚: {'✅ 激活' if r3['activated'] else '❌ 未激活'} (分数: {r3['detection']['activation_score']})")
    print(f"美国 vs 巴拉圭: {'✅ 激活' if r4['activated'] else '❌ 未激活'} (分数: {r4['detection']['activation_score']})")
    
    activated_count = sum([r1['activated'], r2['activated'], r3['activated'], r4['activated']])
    print(f"\n总计: {activated_count}/4 场激活大比分模型")
    print("=" * 70)
