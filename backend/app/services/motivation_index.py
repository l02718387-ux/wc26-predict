"""
战意指数 (Motivation Index) — 第7层预测模块

根据球队在锦标赛中的处境，评估其"必须赢"的程度，
从而调整大比分模型的激活阈值和预测偏向。

核心逻辑:
- 必须赢 → 战意高 → 进攻激进 → 大比分概率上升
- 已出线 → 战意低 → 轮换保留 → 小比分概率上升
- 生死战 → 战意极高 → 双方保守 → 可能反而小比分
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class TournamentStage(Enum):
    GROUP_STAGE = "小组赛"
    ROUND_OF_16 = "16强"
    QUARTER_FINAL = "8强"
    SEMI_FINAL = "半决赛"
    FINAL = "决赛"


class GroupPosition(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4


@dataclass
class MotivationConfig:
    """战意指数配置"""
    # 基础权重
    points_urgency_weight: float = 0.35      # 积分紧迫性权重
    knockout_weight: float = 0.25            # 淘汰赛阶段权重
    historical_style_weight: float = 0.20    # 历史风格权重
    opponent_strength_weight: float = 0.20   # 对手强度权重

    # 积分紧迫性评分标准
    must_win_threshold: float = 0.75         # "必须赢"阈值
    high_motivation_threshold: float = 0.55  # "高战意"阈值
    low_motivation_threshold: float = 0.30   # "低战意"阈值

    # 大比分模型阈值调整系数
    bigscore_boost_when_high: float = 0.15   # 战意高时降低阈值
    bigscore_penalty_when_low: float = 0.10  # 战意低时提高阈值


class MotivationIndexCalculator:
    """
    战意指数计算器

    输出: 0.0 ~ 1.0 的战意指数
    - 0.0-0.3: 低战意 (已出线/无关紧要)
    - 0.3-0.55: 中等战意 (正常比赛)
    - 0.55-0.75: 高战意 (需要积分/晋级)
    - 0.75-1.0: 极高战意 (生死战/必须赢)
    """

    def __init__(self, config: Optional[MotivationConfig] = None):
        self.config = config or MotivationConfig()

    def calculate(
        self,
        team_name: str,
        current_points: int = 0,
        matches_played: int = 0,
        group_size: int = 4,
        group_position: Optional[GroupPosition] = None,
        stage: TournamentStage = TournamentStage.GROUP_STAGE,
        opponent_elo: float = 1500,
        team_elo: float = 1500,
        historical_avg_goals_when_must_win: float = 2.0,
        historical_avg_goals_when_relaxed: float = 1.2,
        is_already_qualified: bool = False,
        is_already_eliminated: bool = False,
        must_win_by_goals: int = 0,  # 需要净胜球
        is_friendly: bool = False,
    ) -> Dict:
        """
        计算战意指数

        Parameters:
            current_points: 当前积分
            matches_played: 已赛场次
            group_size: 小组球队数 (通常4)
            group_position: 当前排名
            stage: 比赛阶段
            opponent_elo: 对手Elo
            team_elo: 本队Elo
            historical_avg_goals_when_must_win: 历史必须赢时的场均进球
            historical_avg_goals_when_relaxed: 历史无关紧要时的场均进球
            is_already_qualified: 是否已经确定出线
            is_already_eliminated: 是否已经确定淘汰
            must_win_by_goals: 需要净胜几个球
            is_friendly: 是否为友谊赛/热身赛
        """

        scores = {}

        # 1. 积分紧迫性评分 (0-1)
        scores['points_urgency'] = self._calc_points_urgency(
            current_points, matches_played, group_size,
            is_already_qualified, is_already_eliminated, must_win_by_goals,
            is_friendly
        )

        # 2. 淘汰赛阶段评分 (0-1)
        scores['knockout'] = self._calc_knockout_urgency(stage)

        # 3. 历史风格评分 (0-1)
        scores['historical_style'] = self._calc_historical_style(
            historical_avg_goals_when_must_win,
            historical_avg_goals_when_relaxed
        )

        # 4. 对手强度评分 (0-1)
        scores['opponent_strength'] = self._calc_opponent_strength(
            team_elo, opponent_elo, is_friendly
        )

        # 加权计算总战意指数
        total_score = (
            scores['points_urgency'] * self.config.points_urgency_weight +
            scores['knockout'] * self.config.knockout_weight +
            scores['historical_style'] * self.config.historical_style_weight +
            scores['opponent_strength'] * self.config.opponent_strength_weight
        )

        # 顶级球队加成: Elo>1750的顶级球队在友谊赛中战意更高 (保持状态、维护声誉)
        if is_friendly and team_elo > 1750:
            total_score = min(1.0, total_score + 0.12)

        # 确定战意等级
        level = self._determine_level(total_score)

        # 计算对大比分模型的影响
        bigscore_adjustment = self._calc_bigscore_adjustment(total_score)

        return {
            'team': team_name,
            'motivation_index': round(total_score, 3),
            'level': level,
            'components': {k: round(v, 3) for k, v in scores.items()},
            'bigscore_adjustment': round(bigscore_adjustment, 3),
            'description': self._get_description(level, total_score)
        }

    def _calc_points_urgency(
        self, points: int, played: int, group_size: int,
        is_qualified: bool, is_eliminated: bool, must_win_by: int,
        is_friendly: bool = False,
    ) -> float:
        """积分紧迫性: 评估球队对积分的渴求程度"""

        if is_friendly and played == 0:
            return 0.30  # 友谊赛基础战意 (保持状态、磨合阵容)

        if is_eliminated:
            return 0.1  # 已淘汰，战意极低

        if is_qualified:
            return 0.2  # 已出线，战意低

        if played == 0:
            return 0.5  # 第一场，正常战意

        # 小组赛积分分析
        max_possible = played * 3
        points_ratio = points / max_possible if max_possible > 0 else 0

        # 剩余场次
        remaining = (group_size - 1) * 3 // 3 - played  # 简化计算
        if group_size == 4:
            total_matches = 3
        else:
            total_matches = group_size - 1
        remaining = total_matches - played

        if remaining == 0:
            return 0.5  # 最后一场已踢完

        # 计算"安全积分"
        # 世界杯历史: 小组出线通常需要 4-6 分
        typical_qualifying_points = 4.5

        if points >= 6:
            return 0.25  # 基本已出线，但理论上可能还没确定
        elif points >= 4 and remaining == 1:
            return 0.4   # 4分最后一场，比较安全
        elif points >= 4 and remaining >= 2:
            return 0.6   # 4分还有多场，需要继续抢分
        elif points == 3:
            if remaining == 2:
                return 0.65  # 3分还剩2场，需要努力
            else:
                return 0.75  # 3分最后一场，必须赢
        elif points == 1 or points == 2:
            if remaining >= 2:
                return 0.70
            else:
                return 0.90  # 1-2分最后一场，生死战
        elif points == 0:
            if remaining >= 2:
                return 0.60
            else:
                return 0.95  # 0分最后一场，必须赢

        # 净胜球需求加成
        if must_win_by >= 3:
            return min(1.0, 0.85 + 0.05 * must_win_by)
        elif must_win_by >= 2:
            return min(1.0, 0.75 + 0.05 * must_win_by)

        return 0.5

    def _calc_knockout_urgency(self, stage: TournamentStage) -> float:
        """淘汰赛阶段: 越往后战意越高"""
        mapping = {
            TournamentStage.GROUP_STAGE: 0.5,
            TournamentStage.ROUND_OF_16: 0.75,
            TournamentStage.QUARTER_FINAL: 0.85,
            TournamentStage.SEMI_FINAL: 0.95,
            TournamentStage.FINAL: 1.0,
        }
        return mapping.get(stage, 0.5)

    def _calc_historical_style(
        self, must_win_goals: float, relaxed_goals: float
    ) -> float:
        """历史风格: 必须赢时的进攻倾向"""
        if relaxed_goals <= 0:
            return 0.5
        ratio = must_win_goals / relaxed_goals
        # ratio > 1.5 表示必须赢时明显更激进
        if ratio >= 1.8:
            return 0.9
        elif ratio >= 1.5:
            return 0.75
        elif ratio >= 1.2:
            return 0.6
        elif ratio >= 1.0:
            return 0.5
        else:
            return 0.3  # 必须赢时反而更保守

    def _calc_opponent_strength(self, team_elo: float, opponent_elo: float, is_friendly: bool = False) -> float:
        """对手强度: 打弱队时可能保留实力，打强队时必须全力"""
        gap = team_elo - opponent_elo
        # 友谊赛中，强队打弱队更容易轮换/保留
        if is_friendly:
            if gap > 400:
                return 0.02  # 友谊赛打极弱队，几乎不全力
            elif gap > 300:
                return 0.05  # 友谊赛打弱队，大幅轮换
            elif gap > 200:
                return 0.10  # 友谊赛打明显弱队，保留实力
            elif gap > 150:
                return 0.20
            elif gap > -50:
                return 0.50  # 势均力敌
            elif gap > -200:
                return 0.70
            else:
                return 0.80
        else:
            if gap > 400:
                return 0.15  # 打极弱队，大幅轮换/保留实力
            elif gap > 300:
                return 0.25  # 打弱队，可能轻敌/轮换
            elif gap > 150:
                return 0.45
            elif gap > -50:
                return 0.6  # 势均力敌
            elif gap > -200:
                return 0.75  # 打稍强对手
            else:
                return 0.85  # 打明显强队，必须全力

    def _determine_level(self, score: float) -> str:
        if score >= self.config.must_win_threshold:
            return "极高战意 (必须赢)"
        elif score >= self.config.high_motivation_threshold:
            return "高战意 (需要积分)"
        elif score >= self.config.low_motivation_threshold:
            return "中等战意 (正常比赛)"
        else:
            return "低战意 (已出线/无关紧要)"

    def _calc_bigscore_adjustment(self, score: float) -> float:
        """
        计算对大比分模型阈值的调整量
        正值 = 降低阈值 (更容易激活大比分模型)
        负值 = 提高阈值 (更难激活)
        """
        if score >= 0.75:
            return self.config.bigscore_boost_when_high * 1.5
        elif score >= 0.55:
            return self.config.bigscore_boost_when_high
        elif score <= 0.25:
            return -self.config.bigscore_penalty_when_low
        elif score <= 0.40:
            return -self.config.bigscore_penalty_when_low * 0.5
        else:
            return 0.0

    def _get_description(self, level: str, score: float) -> str:
        descriptions = {
            "极高战意 (必须赢)": "球队处于生死边缘，必须全力争胜，进攻会非常激进",
            "高战意 (需要积分)": "球队需要积分确保出线或排名，会积极进攻",
            "中等战意 (正常比赛)": "正常比赛节奏，按常规战术执行",
            "低战意 (已出线/无关紧要)": "球队可能轮换阵容、保留实力，进球欲望低",
        }
        return descriptions.get(level, "未知")


class MatchMotivationAnalyzer:
    """
    比赛双方战意分析器

    综合分析主客双方的战意，输出比赛整体的"激进指数"
    """

    def __init__(self):
        self.calculator = MotivationIndexCalculator()

    def analyze_match(
        self,
        home_team: str,
        away_team: str,
        home_elo: float,
        away_elo: float,
        home_points: int = 0,
        away_points: int = 0,
        matches_played: int = 0,
        stage: TournamentStage = TournamentStage.GROUP_STAGE,
        is_neutral: bool = True,
        is_friendly: bool = False,
        is_host: bool = False,
    ) -> Dict:
        """
        分析一场比赛的双方战意
        """

        home_mot = self.calculator.calculate(
            team_name=home_team,
            current_points=home_points,
            matches_played=matches_played,
            stage=stage,
            opponent_elo=away_elo,
            team_elo=home_elo,
            is_friendly=is_friendly,
        )

        away_mot = self.calculator.calculate(
            team_name=away_team,
            current_points=away_points,
            matches_played=matches_played,
            stage=stage,
            opponent_elo=home_elo,
            team_elo=away_elo,
            is_friendly=is_friendly,
        )

        # 东道主加成: 东道主在友谊赛中战意更高 (要在家门口表现)
        if is_host and is_friendly:
            home_mot['motivation_index'] = min(1.0, home_mot['motivation_index'] + 0.15)
            home_mot['components']['host_bonus'] = 0.15

        # 比赛整体激进指数
        # 双方战意都高 → 对攻大战 (大比分)
        # 一方高一方低 → 单边进攻
        # 双方都低 → 沉闷比赛
        home_score = home_mot['motivation_index']
        away_score = away_mot['motivation_index']

        # 综合激进指数
        avg_motivation = (home_score + away_score) / 2
        motivation_gap = abs(home_score - away_score)

        if avg_motivation >= 0.7 and motivation_gap <= 0.3:
            match_style = "对攻大战"
            bigscore_likelihood = "高"
        elif avg_motivation >= 0.7 and motivation_gap > 0.3:
            match_style = "单边碾压"
            bigscore_likelihood = "中高"
        elif avg_motivation >= 0.45:
            match_style = "正常对抗"
            bigscore_likelihood = "中等"
        elif avg_motivation >= 0.25:
            match_style = "保守试探"
            bigscore_likelihood = "低"
        else:
            match_style = "沉闷/轮换"
            bigscore_likelihood = "极低"

        # 对大比分模型阈值的综合调整
        combined_adjustment = (
            home_mot['bigscore_adjustment'] +
            away_mot['bigscore_adjustment']
        ) / 2

        return {
            'home_motivation': home_mot,
            'away_motivation': away_mot,
            'match_motivation_index': round(avg_motivation, 3),
            'motivation_gap': round(motivation_gap, 3),
            'match_style': match_style,
            'bigscore_likelihood': bigscore_likelihood,
            'bigscore_threshold_adjustment': round(combined_adjustment, 3),
        }


# ==================== 测试 ====================

def test_motivation_index():
    print("\n" + "=" * 70)
    print("【测试】战意指数计算器")
    print("=" * 70)

    calc = MotivationIndexCalculator()
    analyzer = MatchMotivationAnalyzer()

    # 场景1: 西班牙 vs 佛得角 (友谊赛，西班牙已出线/无关紧要)
    print("\n场景1: 西班牙 vs 佛得角 (友谊赛)")
    result = analyzer.analyze_match(
        home_team="Spain", away_team="Cape Verde",
        home_elo=1730, away_elo=1495,
        home_points=0, away_points=0,
        matches_played=0,
        stage=TournamentStage.GROUP_STAGE,
    )
    print(f"  主队战意: {result['home_motivation']['motivation_index']} ({result['home_motivation']['level']})")
    print(f"  客队战意: {result['away_motivation']['motivation_index']} ({result['away_motivation']['level']})")
    print(f"  比赛风格: {result['match_style']}")
    print(f"  大比分可能性: {result['bigscore_likelihood']}")
    print(f"  阈值调整: {result['bigscore_threshold_adjustment']}")

    # 场景2: 阿根廷生死战 (小组赛最后一场，必须赢)
    print("\n场景2: 阿根廷 (3分) vs 波兰 (4分) — 阿根廷必须赢")
    result = analyzer.analyze_match(
        home_team="Argentina", away_team="Poland",
        home_elo=1780, away_elo=1720,
        home_points=3, away_points=4,
        matches_played=2,
        stage=TournamentStage.GROUP_STAGE,
    )
    print(f"  主队战意: {result['home_motivation']['motivation_index']} ({result['home_motivation']['level']})")
    print(f"  客队战意: {result['away_motivation']['motivation_index']} ({result['away_motivation']['level']})")
    print(f"  比赛风格: {result['match_style']}")
    print(f"  大比分可能性: {result['bigscore_likelihood']}")
    print(f"  阈值调整: {result['bigscore_threshold_adjustment']}")

    # 场景3: 巴西已出线，最后一场轮换
    print("\n场景3: 巴西 (6分) vs 喀麦隆 (1分) — 巴西已出线")
    result = analyzer.analyze_match(
        home_team="Brazil", away_team="Cameroon",
        home_elo=1800, away_elo=1650,
        home_points=6, away_points=1,
        matches_played=2,
        stage=TournamentStage.GROUP_STAGE,
    )
    print(f"  主队战意: {result['home_motivation']['motivation_index']} ({result['home_motivation']['level']})")
    print(f"  客队战意: {result['away_motivation']['motivation_index']} ({result['away_motivation']['level']})")
    print(f"  比赛风格: {result['match_style']}")
    print(f"  大比分可能性: {result['bigscore_likelihood']}")
    print(f"  阈值调整: {result['bigscore_threshold_adjustment']}")

    # 场景4: 世界杯决赛
    print("\n场景4: 法国 vs 阿根廷 — 世界杯决赛")
    result = analyzer.analyze_match(
        home_team="France", away_team="Argentina",
        home_elo=1820, away_elo=1780,
        home_points=0, away_points=0,
        matches_played=0,
        stage=TournamentStage.FINAL,
    )
    print(f"  主队战意: {result['home_motivation']['motivation_index']} ({result['home_motivation']['level']})")
    print(f"  客队战意: {result['away_motivation']['motivation_index']} ({result['away_motivation']['level']})")
    print(f"  比赛风格: {result['match_style']}")
    print(f"  大比分可能性: {result['bigscore_likelihood']}")
    print(f"  阈值调整: {result['bigscore_threshold_adjustment']}")


if __name__ == "__main__":
    test_motivation_index()
