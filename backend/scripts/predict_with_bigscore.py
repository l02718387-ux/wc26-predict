"""
7层模型融合预测脚本 — 标准5层 + 双大比分模型 + 战意指数 (V4)

使用方式:
    python scripts/predict_with_bigscore.py --home "Spain" --away "Cape Verde"
    python scripts/predict_with_bigscore.py --home "Argentina" --away "Poland" --home-points 3 --away-points 4 --played 2

输出:
    - 5层标准模型预测 (DC/Enhancer/Elo/Pi/Weibull)
    - 战意指数分析 (第6层)
    - 双大比分检测器判断 (第7层: 模型A ≥6球 / 模型B 4-5球)
    - 条件激活的大比分模型预测
    - 7层融合最终结果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.prediction_pipeline import PredictionPipeline
from app.services.dual_bigscore_models import DualBigScoreCoordinator
from app.services.motivation_index import MatchMotivationAnalyzer, TournamentStage


def predict_with_7_layers(
    home_team: str, away_team: str,
    competition: str = "International Friendly",
    home_points: int = 0, away_points: int = 0,
    matches_played: int = 0,
    stage: str = "GROUP_STAGE",
    is_neutral: bool = True,
):
    """
    7层模型融合预测 (V4 — 战意指数 + 双大比分模型)

    Layer 1-5: 标准模型 (DC/Enhancer/Elo/Pi/Weibull)
    Layer 6: 战意指数 (Motivation Index)
    Layer 7: 双大比分模型协调器 (模型A: 极端≥6球 / 模型B: 中等4-5球)
    """

    # 解析比赛阶段
    stage_map = {
        'GROUP_STAGE': TournamentStage.GROUP_STAGE,
        'ROUND_OF_16': TournamentStage.ROUND_OF_16,
        'QUARTER_FINAL': TournamentStage.QUARTER_FINAL,
        'SEMI_FINAL': TournamentStage.SEMI_FINAL,
        'FINAL': TournamentStage.FINAL,
    }
    tournament_stage = stage_map.get(stage.upper(), TournamentStage.GROUP_STAGE)

    print("=" * 70)
    print(f"  7层融合预测 (V4 战意指数+双大比分): {home_team} vs {away_team}")
    print(f"  赛事: {competition} | 阶段: {tournament_stage.value}")
    if matches_played > 0:
        print(f"  积分: {home_team} {home_points}分 vs {away_team} {away_points}分 (已赛{matches_played}轮)")
    print("=" * 70)

    # ── Step 1: 标准5层模型 ──
    print("\n【Step 1】标准5层模型预测...")
    pipeline = PredictionPipeline.from_artifacts(mode='full')
    standard_result = pipeline.predict_sync(home_team, away_team, competition, is_neutral=is_neutral)

    print(f"\n  标准模型融合结果:")
    print(f"    主胜: {standard_result.home_win_prob*100:.1f}%")
    print(f"    平局: {standard_result.draw_prob*100:.1f}%")
    print(f"    客胜: {standard_result.away_win_prob*100:.1f}%")
    print(f"    xG: {standard_result.home_xg:.2f} - {standard_result.away_xg:.2f}")

    print(f"\n  各层独立预测:")
    print(f"    Dixon-Coles:  H={standard_result.dc_probs['home']*100:.1f}% D={standard_result.dc_probs['draw']*100:.1f}% A={standard_result.dc_probs['away']*100:.1f}%")
    if standard_result.enhancer_probs:
        print(f"    Enhancer:     H={standard_result.enhancer_probs['home']*100:.1f}% D={standard_result.enhancer_probs['draw']*100:.1f}% A={standard_result.enhancer_probs['away']*100:.1f}%")
    if standard_result.elo_probs:
        print(f"    Elo:          H={standard_result.elo_probs['home']*100:.1f}% D={standard_result.elo_probs['draw']*100:.1f}% A={standard_result.elo_probs['away']*100:.1f}%")
    if standard_result.pi_probs:
        print(f"    Pi-Rating:    H={standard_result.pi_probs['home']*100:.1f}% D={standard_result.pi_probs['draw']*100:.1f}% A={standard_result.pi_probs['away']*100:.1f}%")

    print(f"\n  Top 3 比分 (标准模型):")
    for s in standard_result.top_scores[:3]:
        print(f"    {s['score']}  ({s['prob']*100:.1f}%)")

    # ── Step 2: 战意指数分析 (第6层) ──
    print("\n【Step 2】战意指数分析 (第6层)...")
    mot_analyzer = MatchMotivationAnalyzer()

    home_elo = standard_result.home_elo
    away_elo = standard_result.away_elo

    is_friendly = 'Friendly' in competition or 'Warmup' in competition
    is_host = not is_neutral  # 简化判断，非中立场地即主队有主场优势
    mot_result = mot_analyzer.analyze_match(
        home_team=home_team, away_team=away_team,
        home_elo=home_elo, away_elo=away_elo,
        home_points=home_points, away_points=away_points,
        matches_played=matches_played,
        stage=tournament_stage,
        is_neutral=is_neutral,
        is_friendly=is_friendly,
        is_host=is_host,
    )

    print(f"\n  {home_team} 战意: {mot_result['home_motivation']['motivation_index']} "
          f"({mot_result['home_motivation']['level']})")
    print(f"  {away_team} 战意: {mot_result['away_motivation']['motivation_index']} "
          f"({mot_result['away_motivation']['level']})")
    print(f"\n  比赛整体战意指数: {mot_result['match_motivation_index']}")
    print(f"  比赛风格预测: {mot_result['match_style']}")
    print(f"  大比分可能性(基于战意): {mot_result['bigscore_likelihood']}")
    print(f"  大比分阈值调整: {mot_result['bigscore_threshold_adjustment']:+.3f}")

    # ── Step 3: 双大比分检测器 (第7层，受战意指数调整) ──
    print("\n【Step 3】双大比分检测器 (第7层，已根据战意调整阈值)...")

    # 战意抑制逻辑: 低战意时强制关闭大比分模型
    match_motivation = mot_result['match_motivation_index']
    home_mot = mot_result['home_motivation']['motivation_index']
    away_mot = mot_result['away_motivation']['motivation_index']
    motivation_suppress = False

    # 抑制条件1: 整体战意低
    if match_motivation < 0.40:
        motivation_suppress = True
        print(f"\n  ⚠️ 战意抑制激活: 整体战意指数 {match_motivation:.3f} < 0.40")
        print(f"     比赛风格: {mot_result['match_style']} → 大比分模型强制关闭")
    # 抑制条件2: 任意一方战意极低 (友谊赛中一方轮换会导致沉闷)
    elif is_friendly and (home_mot < 0.35 or away_mot < 0.35):
        motivation_suppress = True
        low_team = home_team if home_mot < away_mot else away_team
        print(f"\n  ⚠️ 战意抑制激活: {low_team}战意过低 ({min(home_mot, away_mot):.3f})")
        print(f"     友谊赛一方保留实力 → 大比分模型强制关闭")
    elif mot_result['match_style'] == '沉闷/轮换':
        motivation_suppress = True
        print(f"\n  ⚠️ 战意抑制激活: 比赛风格为'{mot_result['match_style']}'")
        print(f"     整体战意: {match_motivation:.3f} → 大比分模型强制关闭")

    coordinator = DualBigScoreCoordinator()

    if not motivation_suppress:
        # 应用战意指数调整到大比分模型阈值
        threshold_adjustment = mot_result['bigscore_threshold_adjustment']
        original_a_threshold = coordinator.model_a.config.activation_threshold
        original_b_threshold = coordinator.model_b.config.activation_threshold

        # 动态调整阈值
        adjusted_a_threshold = max(0.15, original_a_threshold - threshold_adjustment)
        adjusted_b_threshold = max(0.10, original_b_threshold - threshold_adjustment)

        # 临时修改阈值进行检测
        coordinator.model_a.config.activation_threshold = adjusted_a_threshold
        coordinator.model_b.config.activation_threshold = adjusted_b_threshold

        detect_a = coordinator.model_a.should_activate(
            home_elo=home_elo, away_elo=away_elo,
            home_recent_goals_avg=standard_result.home_xg,
            away_recent_goals_avg=standard_result.away_xg,
            is_warmup=is_friendly,
            is_host=False
        )
        detect_b = coordinator.model_b.should_activate(
            home_elo=home_elo, away_elo=away_elo,
            home_recent_goals_avg=standard_result.home_xg,
            away_recent_goals_avg=standard_result.away_xg,
            is_warmup=is_friendly,
            is_host=False
        )

        print(f"\n  模型A检测 (极端大比分 ≥6球):")
        print(f"    原始阈值: {original_a_threshold} → 调整后: {adjusted_a_threshold:.3f}")
        print(f"    激活: {'✅ 是' if detect_a['is_likely'] else '❌ 否'} (分数: {detect_a['score']:.3f})")
        print(f"    触发: {detect_a['triggers']}")

        print(f"\n  模型B检测 (中等大比分 4-5球):")
        print(f"    原始阈值: {original_b_threshold} → 调整后: {adjusted_b_threshold:.3f}")
        print(f"    激活: {'✅ 是' if detect_b['is_likely'] else '❌ 否'} (分数: {detect_b['score']:.3f})")
        print(f"    触发: {detect_b['triggers']}")
    else:
        # 战意抑制，直接返回未激活
        detect_a = {'is_likely': False, 'score': 0.0, 'triggers': ['战意抑制'], 'target_range': '总进球 ≥ 6'}
        detect_b = {'is_likely': False, 'score': 0.0, 'triggers': ['战意抑制'], 'target_range': '总进球 4-5'}

    # ── Step 4: 双大比分模型预测 (条件激活) ──
    bigscore_result = None
    any_activated = detect_a['is_likely'] or detect_b['is_likely']

    if any_activated:
        print("\n【Step 4】大比分模型已激活! 运行双模型协调器预测...")

        base_lambda_h = max(standard_result.home_xg * 1.5, 1.8)
        base_lambda_a = max(standard_result.away_xg * 1.5, 1.2)

        dual_result = coordinator.predict(
            home_team=home_team, away_team=away_team,
            home_elo=home_elo, away_elo=away_elo,
            base_lambda_home=base_lambda_h, base_lambda_away=base_lambda_a,
            home_recent_goals_avg=base_lambda_h, away_recent_goals_avg=base_lambda_a,
            is_warmup='Friendly' in competition or 'Warmup' in competition,
            is_host=False
        )

        bigscore_result = dual_result['selected_model']
        selection_reason = dual_result['selection_reason']

        print(f"\n  选择逻辑: {selection_reason}")

        if bigscore_result:
            print(f"\n  选中模型: {bigscore_result['model']}")
            print(f"  目标范围: {bigscore_result['target']}")
            print(f"  xG: {bigscore_result['xg']['home']} - {bigscore_result['xg']['away']}")
            print(f"  λ: {bigscore_result['lambda']['home']} - {bigscore_result['lambda']['away']}")
            print(f"\n  胜负平概率:")
            hda = bigscore_result['hda']
            print(f"    主胜: {hda['home']*100:.1f}%")
            print(f"    平局: {hda['draw']*100:.1f}%")
            print(f"    客胜: {hda['away']*100:.1f}%")
            print(f"\n  Top 5 比分 (大比分模型):")
            for i, s in enumerate(bigscore_result['top_scores'][:5], 1):
                print(f"    {i}. {s['score']:>5s}  ({s['prob']:5.2f}%)")
        else:
            print("\n  ⚠️ 检测器激活但协调器未选中任何模型")
    else:
        print("\n【Step 4】大比分模型未激活 — 使用标准模型结果")

    # ── Step 5: 7层融合 ──
    print("\n【Step 5】7层融合最终结果...")

    standard_weight = 0.80
    bigscore_weight = 0.15
    motivation_weight = 0.05  # 战意指数微调权重

    if bigscore_result:
        h_standard = standard_result.home_win_prob
        d_standard = standard_result.draw_prob
        a_standard = standard_result.away_win_prob

        h_big = bigscore_result['hda']['home']
        d_big = bigscore_result['hda']['draw']
        a_big = bigscore_result['hda']['away']

        # 基础融合: 标准 + 大比分
        h_fused = h_standard * standard_weight + h_big * bigscore_weight
        d_fused = d_standard * standard_weight + d_big * bigscore_weight
        a_fused = a_standard * standard_weight + a_big * bigscore_weight

        # 战意指数微调
        # 战意高 → 降低平局概率 (生死战很少平局)
        # 战意低 → 提高平局概率 (友谊赛容易闷平)
        mot_index = mot_result['match_motivation_index']
        if mot_index >= 0.7:
            draw_adjustment = -0.03  # 生死战平局概率-3%
        elif mot_index <= 0.3:
            draw_adjustment = +0.03  # 低战意平局概率+3%
        else:
            draw_adjustment = 0.0

        d_fused = max(0.05, d_fused + draw_adjustment)
        # 重新分配平局调整量到胜负
        redistribute = draw_adjustment / 2 if draw_adjustment != 0 else 0
        h_fused -= redistribute
        a_fused -= redistribute

        # 归一化
        total = h_fused + d_fused + a_fused
        h_fused /= total
        d_fused /= total
        a_fused /= total

        xg_h_fused = standard_result.home_xg * standard_weight + bigscore_result['xg']['home'] * bigscore_weight
        xg_a_fused = standard_result.away_xg * standard_weight + bigscore_result['xg']['away'] * bigscore_weight

        print(f"\n  ✅ 7层融合完成")
        print(f"     标准模型: {standard_weight*100:.0f}%")
        print(f"     大比分模型: {bigscore_weight*100:.0f}%")
        print(f"     战意微调: {motivation_weight*100:.0f}%")
        if draw_adjustment != 0:
            print(f"     战意平局调整: {draw_adjustment:+.1%}")

        print(f"\n  === 最终融合结果 ===")
        print(f"    主胜: {h_fused*100:.1f}%")
        print(f"    平局: {d_fused*100:.1f}%")
        print(f"    客胜: {a_fused*100:.1f}%")
        print(f"    xG: {xg_h_fused:.2f} - {xg_a_fused:.2f}")

        print(f"\n  === 综合比分推荐 ===")
        print(f"    标准模型推荐: {standard_result.top_scores[0]['score']} ({standard_result.top_scores[0]['prob']*100:.1f}%)")
        print(f"    大比分模型推荐: {bigscore_result['top_scores'][0]['score']} ({bigscore_result['top_scores'][0]['prob']:.1f}%)")

    else:
        # 大比分模型未激活，但仍有战意微调
        h_fused = standard_result.home_win_prob
        d_fused = standard_result.draw_prob
        a_fused = standard_result.away_win_prob

        mot_index = mot_result['match_motivation_index']
        if mot_index <= 0.25:
            # 低战意 → 提高平局概率 (友谊赛容易闷平，如西班牙vs佛得角0:0)
            draw_adjustment = +0.05
            d_fused = min(0.5, d_fused + draw_adjustment)
            redistribute = draw_adjustment / 2
            h_fused = max(0.1, h_fused - redistribute)
            a_fused = max(0.1, a_fused - redistribute)

            total = h_fused + d_fused + a_fused
            h_fused /= total
            d_fused /= total
            a_fused /= total

            print(f"\n  ✅ 5层融合 + 战意微调 (低战意提高平局概率)")
            print(f"     战意平局调整: +5.0%")
        else:
            print(f"\n  ✅ 5层融合结果 (大比分模型未激活)")

        print(f"\n  === 最终融合结果 ===")
        print(f"    主胜: {h_fused*100:.1f}%")
        print(f"    平局: {d_fused*100:.1f}%")
        print(f"    客胜: {a_fused*100:.1f}%")
        print(f"    xG: {standard_result.home_xg:.2f} - {standard_result.away_xg:.2f}")

    print("\n" + "=" * 70)

    return {
        'standard': standard_result,
        'motivation': mot_result,
        'detection_a': detect_a,
        'detection_b': detect_b,
        'bigscore': bigscore_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="7层模型融合预测 (V4 战意指数+双大比分)")
    parser.add_argument("--home", required=True, help="主队名称")
    parser.add_argument("--away", required=True, help="客队名称")
    parser.add_argument("--competition", default="International Friendly", help="赛事名称")
    parser.add_argument("--home-points", type=int, default=0, help="主队当前积分")
    parser.add_argument("--away-points", type=int, default=0, help="客队当前积分")
    parser.add_argument("--played", type=int, default=0, help="已赛场次")
    parser.add_argument("--stage", default="GROUP_STAGE", help="比赛阶段")

    args = parser.parse_args()

    predict_with_7_layers(
        args.home, args.away,
        competition=args.competition,
        home_points=args.home_points,
        away_points=args.away_points,
        matches_played=args.played,
        stage=args.stage,
    )
