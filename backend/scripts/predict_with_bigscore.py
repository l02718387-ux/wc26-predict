"""
6层模型融合预测脚本 — 标准5层 + 双大比分模型 (V3)

使用方式:
    python scripts/predict_with_bigscore.py --home "United States" --away "Paraguay"

输出:
    - 5层标准模型预测 (DC/Enhancer/Elo/Pi/Weibull)
    - 双大比分检测器判断 (模型A: ≥6球 / 模型B: 4-5球)
    - 条件激活的大比分模型预测
    - 6层融合最终结果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.prediction_pipeline import PredictionPipeline
from app.services.dual_bigscore_models import DualBigScoreCoordinator


def predict_with_6_layers(home_team: str, away_team: str, competition: str = "International Friendly"):
    """
    6层模型融合预测 (V3 — 双大比分模型)

    Layer 1-5: 标准模型 (DC/Enhancer/Elo/Pi/Weibull)
    Layer 6: 双大比分模型协调器 (模型A: 极端≥6球 / 模型B: 中等4-5球)
    """

    print("=" * 70)
    print(f"  6层融合预测 (V3 双大比分模型): {home_team} vs {away_team}")
    print(f"  赛事: {competition}")
    print("=" * 70)

    # ── Step 1: 标准5层模型 ──
    print("\n【Step 1】标准5层模型预测...")
    pipeline = PredictionPipeline.from_artifacts(mode='full')
    standard_result = pipeline.predict_sync(home_team, away_team, competition, is_neutral=True)

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

    # ── Step 2: 双大比分检测器 ──
    print("\n【Step 2】双大比分检测器 (模型A: 极端≥6球 / 模型B: 中等4-5球)...")
    coordinator = DualBigScoreCoordinator()

    home_elo = standard_result.home_elo
    away_elo = standard_result.away_elo

    # 使用协调器进行检测 (不运行完整预测，只看检测)
    detect_a = coordinator.model_a.should_activate(
        home_elo=home_elo,
        away_elo=away_elo,
        home_recent_goals_avg=standard_result.home_xg,
        away_recent_goals_avg=standard_result.away_xg,
        is_warmup='Friendly' in competition or 'Warmup' in competition,
        is_host=False
    )
    detect_b = coordinator.model_b.should_activate(
        home_elo=home_elo,
        away_elo=away_elo,
        home_recent_goals_avg=standard_result.home_xg,
        away_recent_goals_avg=standard_result.away_xg,
        is_warmup='Friendly' in competition or 'Warmup' in competition,
        is_host=False
    )

    print(f"\n  模型A检测 (极端大比分 ≥6球):")
    print(f"    激活: {'✅ 是' if detect_a['is_likely'] else '❌ 否'} (分数: {detect_a['score']}, 阈值: {coordinator.model_a.config.activation_threshold})")
    print(f"    触发: {detect_a['triggers']}")

    print(f"\n  模型B检测 (中等大比分 4-5球):")
    print(f"    激活: {'✅ 是' if detect_b['is_likely'] else '❌ 否'} (分数: {detect_b['score']}, 阈值: {coordinator.model_b.config.activation_threshold})")
    print(f"    触发: {detect_b['triggers']}")

    # ── Step 3: 双大比分模型预测 (条件激活) ──
    bigscore_result = None
    any_activated = detect_a['is_likely'] or detect_b['is_likely']

    if any_activated:
        print("\n【Step 3】大比分模型已激活! 运行双模型协调器预测...")

        # 大比分模型需要更高的基础 λ，如果标准模型 xG 偏低则提升
        base_lambda_h = max(standard_result.home_xg * 1.5, 1.8)
        base_lambda_a = max(standard_result.away_xg * 1.5, 1.2)

        dual_result = coordinator.predict(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            base_lambda_home=base_lambda_h,
            base_lambda_away=base_lambda_a,
            home_recent_goals_avg=base_lambda_h,
            away_recent_goals_avg=base_lambda_a,
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
        print("\n【Step 3】大比分模型未激活 — 使用标准模型结果")

    # ── Step 4: 6层融合 ──
    print("\n【Step 4】6层融合最终结果...")

    standard_weight = 0.85
    bigscore_weight = 0.15

    if bigscore_result:
        h_standard = standard_result.home_win_prob
        d_standard = standard_result.draw_prob
        a_standard = standard_result.away_win_prob

        h_big = bigscore_result['hda']['home']
        d_big = bigscore_result['hda']['draw']
        a_big = bigscore_result['hda']['away']

        h_fused = h_standard * standard_weight + h_big * bigscore_weight
        d_fused = d_standard * standard_weight + d_big * bigscore_weight
        a_fused = a_standard * standard_weight + a_big * bigscore_weight

        total = h_fused + d_fused + a_fused
        h_fused /= total
        d_fused /= total
        a_fused /= total

        xg_h_fused = standard_result.home_xg * standard_weight + bigscore_result['xg']['home'] * bigscore_weight
        xg_a_fused = standard_result.away_xg * standard_weight + bigscore_result['xg']['away'] * bigscore_weight

        print(f"\n  ✅ 6层融合完成 (标准模型 85% + 大比分模型 15%)")
        print(f"\n  === 最终融合结果 ===")
        print(f"    主胜: {h_fused*100:.1f}%")
        print(f"    平局: {d_fused*100:.1f}%")
        print(f"    客胜: {a_fused*100:.1f}%")
        print(f"    xG: {xg_h_fused:.2f} - {xg_a_fused:.2f}")

        print(f"\n  === 综合比分推荐 ===")
        print(f"    标准模型推荐: {standard_result.top_scores[0]['score']} ({standard_result.top_scores[0]['prob']*100:.1f}%)")
        print(f"    大比分模型推荐: {bigscore_result['top_scores'][0]['score']} ({bigscore_result['top_scores'][0]['prob']:.1f}%)")

    else:
        print(f"\n  ✅ 5层融合结果 (大比分模型未激活)")
        print(f"\n  === 最终融合结果 ===")
        print(f"    主胜: {standard_result.home_win_prob*100:.1f}%")
        print(f"    平局: {standard_result.draw_prob*100:.1f}%")
        print(f"    客胜: {standard_result.away_win_prob*100:.1f}%")
        print(f"    xG: {standard_result.home_xg:.2f} - {standard_result.away_xg:.2f}")

    print("\n" + "=" * 70)

    return {
        'standard': standard_result,
        'detection_a': detect_a,
        'detection_b': detect_b,
        'bigscore': bigscore_result,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="6层模型融合预测 (V3 双大比分模型)")
    parser.add_argument("--home", required=True, help="主队名称")
    parser.add_argument("--away", required=True, help="客队名称")
    parser.add_argument("--competition", default="International Friendly", help="赛事名称")

    args = parser.parse_args()

    predict_with_6_layers(args.home, args.away, args.competition)
