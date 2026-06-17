"""
6层模型融合预测脚本 — 标准5层 + 大比分模型

使用方式:
    python scripts/predict_with_bigscore.py --home "United States" --away "Paraguay"

输出:
    - 5层标准模型预测 (DC/Enhancer/Elo/Pi/Weibull)
    - 大比分检测器判断
    - 大比分模型预测 (如激活)
    - 6层融合最终结果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.prediction_pipeline import PredictionPipeline
from app.services.big_score_model import BigScorePredictorModel


def predict_with_6_layers(home_team: str, away_team: str, competition: str = "International Friendly"):
    """
    6层模型融合预测
    
    Layer 1-5: 标准模型 (DC/Enhancer/Elo/Pi/Weibull)
    Layer 6: 大比分模型 (复合泊松-伽马，条件激活)
    """
    
    print("=" * 70)
    print(f"  6层融合预测: {home_team} vs {away_team}")
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
    
    # ── Step 2: 大比分检测器 ──
    print("\n【Step 2】大比分检测器...")
    bigscore = BigScorePredictorModel()
    
    # 获取 Elo 评分用于检测
    home_elo = standard_result.home_elo
    away_elo = standard_result.away_elo
    
    detection_result = bigscore.detector.detect(
        home_elo=home_elo,
        away_elo=away_elo,
        match_date="2026-06-01",
        tournament_start_date="2026-06-11",  # 世界杯前
        recent_scores=[(2, 1), (1, 0), (3, 2)],
        historical_avg_goals=2.5,
        home_team_style='balanced',
        away_team_style='defensive'
    )
    
    print(f"\n  检测分数: {detection_result['activation_score']:.2f} (阈值: 0.35)")
    print(f"  触发规则: {detection_result['triggered_rules']}")
    print(f"  建议: {detection_result['recommendation']}")
    
    # ── Step 3: 大比分模型 (条件激活) ──
    bigscore_result = None
    if detection_result['is_big_score_likely']:
        print("\n【Step 3】大比分模型已激活! 运行复合泊松-伽马预测...")
        
        bigscore_result = bigscore.predict(
            home_team=home_team,
            away_team=away_team,
            home_elo=home_elo,
            away_elo=away_elo,
            match_date="2026-06-01",
            base_lambda_home=standard_result.home_xg * 1.2,  # 基于标准模型 xG 微调
            base_lambda_away=standard_result.away_xg * 1.2,
            tournament_start_date="2026-06-11",
            recent_scores=[(2, 1), (1, 0), (3, 2)],
            historical_avg_goals=2.5,
            home_team_style='balanced',
            away_team_style='defensive'
        )
        
        print(f"\n  大比分模型预测:")
        print(f"    xG: {bigscore_result['xg']['home']} - {bigscore_result['xg']['away']}")
        print(f"    λ: {bigscore_result['lambda']['home']} - {bigscore_result['lambda']['away']}")
        print(f"\n    胜负平概率:")
        hda = bigscore_result['hda_probabilities']
        print(f"      主胜: {hda['home_win']*100:.1f}%")
        print(f"      平局: {hda['draw']*100:.1f}%")
        print(f"      客胜: {hda['away_win']*100:.1f}%")
        print(f"\n    Top 5 比分 (大比分模型):")
        for i, s in enumerate(bigscore_result['top_10_scores'][:5], 1):
            print(f"      {i}. {s['score']:>5s}  ({s['probability']:5.2f}%)")
    else:
        print("\n【Step 3】大比分模型未激活 — 使用标准模型结果")
    
    # ── Step 4: 6层融合 ──
    print("\n【Step 4】6层融合最终结果...")
    
    # 基础权重
    standard_weight = 0.85  # 标准模型权重 85%
    bigscore_weight = 0.15  # 大比分模型权重 15% (仅当激活时)
    
    if bigscore_result and bigscore_result['activated']:
        # 融合: 标准模型 × 0.85 + 大比分模型 × 0.15
        h_standard = standard_result.home_win_prob
        d_standard = standard_result.draw_prob
        a_standard = standard_result.away_win_prob
        
        h_big = bigscore_result['hda_probabilities']['home_win']
        d_big = bigscore_result['hda_probabilities']['draw']
        a_big = bigscore_result['hda_probabilities']['away_win']
        
        h_fused = h_standard * standard_weight + h_big * bigscore_weight
        d_fused = d_standard * standard_weight + d_big * bigscore_weight
        a_fused = a_standard * standard_weight + a_big * bigscore_weight
        
        # 归一化
        total = h_fused + d_fused + a_fused
        h_fused /= total
        d_fused /= total
        a_fused /= total
        
        # xG 融合
        xg_h_fused = standard_result.home_xg * standard_weight + bigscore_result['xg']['home'] * bigscore_weight
        xg_a_fused = standard_result.away_xg * standard_weight + bigscore_result['xg']['away'] * bigscore_weight
        
        print(f"\n  ✅ 6层融合完成 (标准模型 85% + 大比分模型 15%)")
        print(f"\n  === 最终融合结果 ===")
        print(f"    主胜: {h_fused*100:.1f}%")
        print(f"    平局: {d_fused*100:.1f}%")
        print(f"    客胜: {a_fused*100:.1f}%")
        print(f"    xG: {xg_h_fused:.2f} - {xg_a_fused:.2f}")
        
        # 综合比分推荐
        print(f"\n  === 综合比分推荐 ===")
        print(f"    标准模型推荐: {standard_result.top_scores[0]['score']} ({standard_result.top_scores[0]['prob']*100:.1f}%)")
        print(f"    大比分模型推荐: {bigscore_result['top_10_scores'][0]['score']} ({bigscore_result['top_10_scores'][0]['probability']:.1f}%)")
        
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
        'bigscore_detection': detection_result,
        'bigscore': bigscore_result,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="6层模型融合预测")
    parser.add_argument("--home", required=True, help="主队名称")
    parser.add_argument("--away", required=True, help="客队名称")
    parser.add_argument("--competition", default="International Friendly", help="赛事名称")
    
    args = parser.parse_args()
    
    predict_with_6_layers(args.home, args.away, args.competition)
