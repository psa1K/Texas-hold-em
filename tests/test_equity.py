"""Equity 计算测试 —— 手牌强度、牌型概率、多玩家 MC。"""

import pytest

from src.engine.card import Card, Cards
from src.analysis.equity import (
    EquityCalculator,
    calculate_hand_strength,
    calculate_hand_type_probs,
    _preflop_strength_vs_random,
    _postflop_strength_vs_random,
    _enumerate_hand_type_probs,
    _canonical_type,
    _to_equity_str,
    _preflop_strength_cache,
)
from src.utils.constants import HandRank


def cards(s: str) -> list[Card]:
    return Card.from_str_multi(s)


# ================================================================
# 工具函数
# ================================================================

class TestConversionHelpers:
    """py-poker-equity 格式转换测试。"""

    def test_to_equity_str_ten(self) -> None:
        """T → 10 转换。"""
        card = Card.from_str("Th")
        assert _to_equity_str(card) == "10h"

    def test_to_equity_str_normal(self) -> None:
        """无 T 时保持不变。"""
        card = Card.from_str("Ah")
        assert _to_equity_str(card) == "Ah"
        card2 = Card.from_str("2c")
        assert _to_equity_str(card2) == "2c"

    def test_canonical_type_pair(self) -> None:
        """对子。"""
        assert _canonical_type(cards("Ah As")) == "AA"
        assert _canonical_type(cards("2h 2s")) == "22"

    def test_canonical_type_suited(self) -> None:
        """同花。"""
        assert _canonical_type(cards("Ah Kh")) == "AKs"
        assert _canonical_type(cards("Th Jh")) == "JTs"

    def test_canonical_type_offsuit(self) -> None:
        """非同花。"""
        assert _canonical_type(cards("Ah Kd")) == "AKo"
        assert _canonical_type(cards("7h 2c")) == "72o"


# ================================================================
# calculate_hand_strength
# ================================================================

class TestHandStrength:
    """手牌强度计算测试。"""

    def test_preflop_aa_strong(self) -> None:
        """AA 翻牌前应有高强度。"""
        strength = calculate_hand_strength(cards("Ah As"))
        assert strength > 80, f"AA strength={strength}, expected > 80"

    def test_preflop_72o_weak(self) -> None:
        """72o 翻牌前应有低强度。"""
        strength = calculate_hand_strength(cards("7h 2c"))
        assert strength < 50, f"72o strength={strength}, expected < 50"

    def test_preflop_ak_suited(self) -> None:
        """AKs 翻牌前约 67% vs 随机。"""
        strength = calculate_hand_strength(cards("Ah Kh"))
        assert 55 < strength < 72, f"AKs strength={strength}"

    def test_preflop_cache_hit(self) -> None:
        """同 canonical type 命中缓存。"""
        _preflop_strength_cache.clear()
        s1 = calculate_hand_strength(cards("Ah As"))
        assert len(_preflop_strength_cache) == 1
        s2 = calculate_hand_strength(cards("Ad Ac"))  # 同为 "AA"
        assert s1 == s2
        assert len(_preflop_strength_cache) == 1  # 缓存未增长
        _preflop_strength_cache.clear()

    def test_preflop_returns_float_range(self) -> None:
        """翻牌前强度在合理范围。"""
        for hand_str in ["Ah As", "Kh Kd", "Qh Js", "Th 9s", "7h 2c", "2h 3c"]:
            strength = calculate_hand_strength(cards(hand_str))
            assert 0 <= strength <= 100, f"{hand_str}: {strength}"

    def test_postflop_made_nuts(self) -> None:
        """翻牌后已成绝对坚果 → 高强度。"""
        strength = calculate_hand_strength(
            cards("Ah Kh"),
            cards("Qh Jh Th"),
        )
        assert strength > 85, f"nut straight+flush draw: {strength}"

    def test_postflop_air(self) -> None:
        """翻牌后完全未中牌 → 低强度。"""
        strength = calculate_hand_strength(
            cards("2h 7c"),
            cards("Ad Kd Qd"),
        )
        assert strength < 25, f"air on AKQ: {strength}"

    def test_postflop_top_pair(self) -> None:
        """翻牌后顶对顶踢脚 → 中高强度。"""
        strength = calculate_hand_strength(
            cards("Ah Kh"),
            cards("Kd 8c 2h"),
        )
        assert 60 < strength < 95, f"TPTK: {strength}"

    def test_turn_made_hand(self) -> None:
        """转牌已成葫芦 → 强度接近 100。"""
        strength = calculate_hand_strength(
            cards("Ah Ad"),
            cards("As Kh Kd 2c"),
        )
        assert strength > 90, f"full house: {strength}"


# ================================================================
# calculate_hand_type_probs
# ================================================================

class TestHandTypeProbs:
    """牌型概率计算测试。"""

    def test_river_known(self) -> None:
        """河牌时返回 100% 当前牌型。"""
        # 8d 9d 构成顺子 T-J-Q-K-9 → 9TQK（不是同花）
        probs = calculate_hand_type_probs(
            cards("Td Jd"),
            cards("Qd Ks 8s 9c 2h"),
        )
        # 顺子：T-J-Q-K-9（使用 Td-Jd-Qd-Ks-9c 或类似）
        assert probs["顺子"] == 100.0
        for key in probs:
            if key != "顺子":
                assert probs[key] == 0.0

    def test_turn_exact_enumeration(self) -> None:
        """转牌时精确枚举（46 种河牌）。"""
        # Ah As + Ad 2h 3c 4s = 三条 A
        # 河牌改善牌型：A→四条(1), 2/3/4→葫芦(9), 5→顺子(4)
        # 保持三条: 46-14=32, 32/46≈69.6%
        probs = calculate_hand_type_probs(
            cards("Ah As"),
            cards("Ad 2h 3c 4s"),
        )
        assert probs["三条"] == pytest.approx(69.6, abs=1.0)
        assert probs["葫芦"] == pytest.approx(9.0 / 46 * 100, abs=1.0)
        assert probs["四条"] == pytest.approx(1.0 / 46 * 100, abs=0.5)
        assert probs["顺子"] == pytest.approx(4.0 / 46 * 100, abs=1.0)
        total = sum(probs.values())
        assert abs(total - 100.0) < 0.5

    def test_flop_exact_enumeration(self) -> None:
        """翻牌时精确枚举 C(47,2)=1081 种组合。"""
        probs = calculate_hand_type_probs(
            cards("Ah As"),
            cards("Ad Kd Qd"),
        )
        total = sum(probs.values())
        assert abs(total - 100.0) < 1.0, f"total={total}"

    def test_preflop_mc(self) -> None:
        """翻牌前蒙特卡洛。"""
        probs = calculate_hand_type_probs(cards("Ah Kh"))
        total = sum(probs.values())
        assert abs(total - 100.0) < 1.0, f"total={total}"
        # 所有牌型都应存在概率
        assert set(probs.keys()) == {
            "皇家同花顺", "同花顺", "四条", "葫芦", "同花",
            "顺子", "三条", "两对", "一对", "高牌",
        }

    def test_postflop_sum_to_100(self) -> None:
        """翻牌后概率总和为 100%。"""
        for holes, board in [
            ("Ah Kh", "Qd Js Th"),  # flop
            ("Ah Kh", "Qd Js Th 2c"),  # turn
        ]:
            probs = calculate_hand_type_probs(cards(holes), cards(board))
            total = sum(probs.values())
            assert abs(total - 100.0) < 0.5, f"{holes} | {board}: {total}"


# ================================================================
# Exact enumeration
# ================================================================

class TestExactEnumeration:
    """精确枚举内核测试。"""

    def test_turn_46_combos(self) -> None:
        """转牌精确枚举 46 种组合。"""
        hole = cards("Ah Kh")
        board = cards("Qh Jh Th 2c")
        all_known = hole + board
        from src.utils.constants import Rank, Suit
        import itertools
        known_set = set(all_known)
        remaining = [
            Card(rank=r, suit=s)
            for r, s in itertools.product(Rank, Suit)
            if Card(rank=r, suit=s) not in known_set
        ]
        assert len(remaining) == 46
        result = _enumerate_hand_type_probs(hole, board, remaining, 1)
        total = sum(result.values())
        assert abs(total - 100.0) < 0.5

    def test_flop_1081_combos(self) -> None:
        """翻牌精确枚举 C(47,2)=1081 种组合。"""
        hole = cards("Ah Kh")
        board = cards("Qh Jh Th")
        all_known = hole + board
        from src.utils.constants import Rank, Suit
        import itertools
        known_set = set(all_known)
        remaining = [
            Card(rank=r, suit=s)
            for r, s in itertools.product(Rank, Suit)
            if Card(rank=r, suit=s) not in known_set
        ]
        assert len(remaining) == 47
        result = _enumerate_hand_type_probs(hole, board, remaining, 2)
        total = sum(result.values())
        assert abs(total - 100.0) < 0.5


# ================================================================
# EquityCalculator (保留的多玩家 MC)
# ================================================================

class TestEquityCalculator:
    """胜率计算器测试。"""

    def test_heads_up_preflop_aa_vs_kk(self) -> None:
        """AA vs KK 翻牌前胜率。"""
        calc = EquityCalculator(num_simulations=2000, seed=42)
        hand_a = cards("Ah As")
        hand_b = cards("Kh Ks")
        win_a, win_b, tie = calc.heads_up_equity(hand_a, hand_b)

        # AA 应显著领先 (约 81% vs 19%)
        assert win_a > win_b
        assert win_a > 0.75
        assert win_b < 0.25

    def test_heads_up_preflop_pair_vs_overcards(self) -> None:
        """22 vs AK — 小对子轻微领先。"""
        calc = EquityCalculator(num_simulations=2000, seed=42)
        hand_a = cards("2h 2s")
        hand_b = cards("Ad Kh")
        win_a, win_b, tie = calc.heads_up_equity(hand_a, hand_b)

        # 22 vs AK 约 52% vs 47%, close race
        assert 0.45 < win_a < 0.58
        assert 0.40 < win_b < 0.55

    def test_heads_up_dominated_hands(self) -> None:
        """AK vs AQ — 统治局面。"""
        calc = EquityCalculator(num_simulations=2000, seed=42)
        hand_a = cards("Ah Kh")
        hand_b = cards("As Qd")
        win_a, win_b, tie = calc.heads_up_equity(hand_a, hand_b)

        # AK 应显著领先 AQ
        assert win_a > win_b
        assert win_a > 0.65

    def test_postflop_made_hand_vs_draw(self) -> None:
        """翻牌后：成牌 vs 听牌。"""
        calc = EquityCalculator(num_simulations=2000, seed=42)
        # A 有顶对顶踢脚, B 有同花听牌
        hand_a = cards("Ah Kh")   # top pair top kicker on K-high board
        hand_b = cards("Qc Jc")   # flush draw
        community = cards("Kc 8c 2d")  # K-high, club draw

        result = calc.calculate([hand_a, hand_b], community)
        keys = list(result.keys())
        # 顶对 vs 同花听牌：顶对轻微领先
        win_a = result[keys[0]][0]
        win_b = result[keys[1]][0]
        assert 0.50 < win_a < 0.70
        assert 0.25 < win_b < 0.50

    def test_result_sum_to_one(self) -> None:
        """胜率 + 平率 + 负率 = 1.0。"""
        calc = EquityCalculator(num_simulations=500, seed=42)
        hand_a = cards("Ah Kh")
        hand_b = cards("2s 2d")
        result = calc.calculate([hand_a, hand_b])
        for desc, stats in result.items():
            total = sum(stats)
            assert abs(total - 1.0) < 0.01, f"{desc}: sum={total}"

    def test_multi_player(self) -> None:
        """多人底池胜率计算。"""
        calc = EquityCalculator(num_simulations=1000, seed=42)
        hands = [
            cards("Ah As"),   # AA
            cards("Kh Ks"),   # KK
            cards("Qh Qs"),   # QQ
        ]
        result = calc.calculate(hands)
        keys = list(result.keys())
        # AA 应有最高胜率
        win_rates = [result[k][0] for k in keys]
        assert win_rates[0] > win_rates[1]
        assert win_rates[1] > win_rates[2]

    def test_preflop_matchup(self) -> None:
        """翻牌前对决快捷方法。"""
        calc = EquityCalculator(num_simulations=500, seed=42)
        result = calc.preflop_matchup("Ah Kh", "2s 2d")
        assert "win_a" in result
        assert "win_b" in result
        assert "tie" in result

    def test_known_dead_cards(self) -> None:
        """已知死牌影响胜率。"""
        calc = EquityCalculator(num_simulations=1000, seed=42)
        hand_a = cards("Ah Kh")
        hand_b = cards("Qs Qd")
        # 有一张 Q 已死
        dead = cards("Qh")
        result = calc.calculate([hand_a, hand_b], dead_cards=dead)
        keys = list(result.keys())
        # B 的胜率应低于没有死牌的情况
        win_b = result[keys[1]][0]
        assert win_b < 0.55
