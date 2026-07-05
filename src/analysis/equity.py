"""手牌强度与牌型概率 —— 精确 equity 计算。

核心变更：
- 翻牌前：使用 py-poker-equity 预计算查表（精确、瞬时）
- 翻牌后：使用精确枚举替代蒙特卡洛模拟（手牌类型概率）
- 手牌强度（equity vs 随机对手）：py-poker-equity（翻牌前） + Monte Carlo（翻牌后）
"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.engine.card import Card, Cards
from src.engine.deck import Deck
from src.engine.hand import HandEvaluator
from src.utils.constants import HandRank, Rank, Suit


# ================================================================
# py-poker-equity 适配层
# ================================================================

# py-poker-equity 使用 '10' 表示 10，而非我们的 'T'
_RANK_TO_EQUITY: dict[str, str] = {
    "T": "10",
    "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9",
    "J": "J", "Q": "Q", "K": "K", "A": "A",
}

# 169 种起手牌的缓存：{canonical_type: strength_pct}
_preflop_strength_cache: dict[str, float] = {}


def _to_equity_str(card: Card) -> str:
    """将 Card.short_str（如 'Th'）转为 py-poker-equity 格式（'10h'）。"""
    s = card.short_str  # e.g. "Th"
    rank_char = s[0]
    suit_char = s[1]
    equity_rank = _RANK_TO_EQUITY.get(rank_char, rank_char)
    return equity_rank + suit_char


def _canonical_type(hole_cards: Cards) -> str:
    """将底牌映射为 169 种标准起手牌类型字符串。

    例：Ah Kh → "AKs", 2h 2c → "22", Ad Kh → "AKo"
    """
    r1 = hole_cards[0].rank.short
    r2 = hole_cards[1].rank.short
    suited = hole_cards[0].suit == hole_cards[1].suit
    if r1 == r2:
        return r1 + r2
    # 按 rank value 排序，高牌在前
    v1 = hole_cards[0].rank.value
    v2 = hole_cards[1].rank.value
    high, low = (r1, r2) if v1 >= v2 else (r2, r1)
    suffix = "s" if suited else "o"
    return high + low + suffix


# ================================================================
# 手牌强度（equity vs 随机对手）
# ================================================================

def calculate_hand_strength(
    hole_cards: Cards,
    community_cards: Optional[Cards] = None,
) -> float:
    """计算手牌强度 —— 与一个随机对手的 Heads-up equity。

    翻牌前：枚举 C(50,2)=1225 种对手手牌，用 py-poker-equity 查表精确计算。
    翻牌后：采样随机对手手牌 + 蒙特卡洛模拟。

    Returns:
        手牌强度百分比（0.0–100.0）。
    """
    community = list(community_cards) if community_cards else []

    if not community:
        return _preflop_strength_vs_random(hole_cards)

    return _postflop_strength_vs_random(hole_cards, community)


def _preflop_strength_vs_random(hole_cards: Cards) -> float:
    """翻牌前手牌强度：py-poker-equity 精确查表。

    从剩余 50 张牌中枚举所有 C(50,2)=1225 种对手手牌，
    用 py-poker-equity 的预计算查表获取精确 equity，加权平均。
    每种 canonical type 只计算一次并缓存。
    """
    canonical = _canonical_type(hole_cards)
    if canonical in _preflop_strength_cache:
        return _preflop_strength_cache[canonical]

    from py_poker_equity import get_equity

    my_hand_strs = [_to_equity_str(c) for c in hole_cards]
    known_strs = set(my_hand_strs)

    # 构建剩余牌池
    all_remaining: list[str] = []
    for rank_char in "23456789TJQKA":
        for suit_char in "cdhs":
            card_str = _RANK_TO_EQUITY.get(rank_char, rank_char) + suit_char
            if card_str not in known_strs:
                all_remaining.append(card_str)

    total_equity = 0.0
    count = 0
    n = len(all_remaining)
    for i in range(n):
        for j in range(i + 1, n):
            opp_hand = [all_remaining[i], all_remaining[j]]
            result = get_equity(my_hand_strs, opp_hand)
            total_equity += result["a_win"] + result["tie"] * 0.5
            count += 1

    strength = round(total_equity / count, 1)
    _preflop_strength_cache[canonical] = strength
    return strength


def _postflop_strength_vs_random(
    hole_cards: Cards,
    community_cards: list[Card],
    num_opponent_samples: int = 15,
    mc_samples: int = 500,
) -> float:
    """翻牌后手牌强度：蒙特卡洛 vs 随机对手。

    随机采样 N 个对手手牌，对每个用 heads-up 蒙特卡洛计算 equity，
    加权平均后返回。每手 MC 仅 500 次模拟，15 个对手 × 500 ≈ 7500 次评估，<30ms。
    """
    known: set[Card] = set(hole_cards) | set(community_cards)
    remaining = [
        Card(rank=r, suit=s)
        for r, s in itertools.product(Rank, Suit)
        if Card(rank=r, suit=s) not in known
    ]

    calc = EquityCalculator(num_simulations=mc_samples, seed=hash(tuple(hole_cards)) % 10000)
    rng = random.Random(42)

    total_equity = 0.0
    actual_samples = 0
    for _ in range(num_opponent_samples):
        if len(remaining) < 2:
            break
        opp_cards = rng.sample(remaining, 2)
        win, _, tie = calc.heads_up_equity(hole_cards, opp_cards, community_cards)
        total_equity += win + tie * 0.5
        actual_samples += 1

    if actual_samples == 0:
        return 50.0
    return round(total_equity / actual_samples * 100, 1)


# ================================================================
# EquityCalculator —— 多玩家蒙特卡洛（LLM Bot 使用）
# ================================================================

class EquityCalculator:
    """蒙特卡洛胜率计算器。

    通过大量随机模拟，估算一手或多手牌的胜率。
    保留此类供 LLM Bot 多玩家场景使用。
    """

    def __init__(self, num_simulations: int = 1000, seed: int = 42) -> None:
        self.num_simulations = num_simulations
        self.rng = random.Random(seed)

    def calculate(
        self,
        hole_cards_list: List[Cards],
        community_cards: Optional[Cards] = None,
        dead_cards: Optional[Cards] = None,
    ) -> Dict[str, List[float]]:
        """计算每手牌的胜率。

        Args:
            hole_cards_list: 各玩家的底牌列表。
            community_cards: 已知的公共牌（0–5 张）。
            dead_cards: 已知的已死牌（已被弃/烧）。

        Returns:
            {描述: [胜率, 平率, 负率]} 的字典。
        """
        community = list(community_cards or [])
        dead = list(dead_cards or [])

        # 收集所有已知牌
        known_cards: Set[Card] = set()
        for hand in hole_cards_list:
            for c in hand:
                known_cards.add(c)
        for c in community:
            known_cards.add(c)
        for c in dead:
            known_cards.add(c)

        # 从牌堆中移除已知牌
        remaining = [
            Card(rank=r, suit=s)
            for r, s in itertools.product(Rank, Suit)
            if Card(rank=r, suit=s) not in known_cards
        ]

        needed = 5 - len(community)
        descriptions = [
            " ".join(c.short_str for c in hand)
            for hand in hole_cards_list
        ]

        wins = [0] * len(hole_cards_list)
        ties = [0] * len(hole_cards_list)
        losses = [0] * len(hole_cards_list)

        for _ in range(self.num_simulations):
            # 随机抽取剩余公共牌
            sim_community = community + self.rng.sample(remaining, needed)

            # 评估每手牌
            results = []
            for hand in hole_cards_list:
                all_cards = list(hand) + sim_community
                results.append(HandEvaluator.evaluate(all_cards))

            # 找最佳手牌
            best_score = max(r.score for r in results)
            best_indices = [
                i for i, r in enumerate(results)
                if r.score == best_score
            ]

            if len(best_indices) == 1:
                wins[best_indices[0]] += 1
                for i in range(len(hole_cards_list)):
                    if i != best_indices[0]:
                        losses[i] += 1
            else:
                for i in best_indices:
                    ties[i] += 1
                for i in range(len(hole_cards_list)):
                    if i not in best_indices:
                        losses[i] += 1

        total = self.num_simulations
        return {
            desc: [
                round(w / total, 4),
                round(t / total, 4),
                round(l / total, 4),
            ]
            for desc, w, t, l in zip(descriptions, wins, ties, losses)
        }

    def heads_up_equity(
        self,
        hand_a: Cards,
        hand_b: Cards,
        community_cards: Optional[Cards] = None,
    ) -> Tuple[float, float, float]:
        """双人胜率计算。

        Returns:
            (A胜率, B胜率, 平率)。
        """
        result = self.calculate([hand_a, hand_b], community_cards)
        keys = list(result.keys())
        return (
            result[keys[0]][0],
            result[keys[1]][0],
            result[keys[0]][1],
        )

    def preflop_matchup(self, hand_a_str: str, hand_b_str: str) -> Dict[str, float]:
        """两个起手牌的翻牌前胜率对决。

        Args:
            hand_a_str: 如 "Ah Kh"
            hand_b_str: 如 "2s 2d"

        Returns:
            A胜率, B胜率, 平率。
        """
        hand_a = Card.from_str_multi(hand_a_str)
        hand_b = Card.from_str_multi(hand_b_str)
        win_a, win_b, tie = self.heads_up_equity(hand_a, hand_b)
        return {"win_a": win_a, "win_b": win_b, "tie": tie}


# ================================================================
# 牌型概率计算
# ================================================================

def calculate_hand_type_probs(
    hole_cards: Cards,
    community_cards: Optional[Cards] = None,
    num_simulations: int = 2000,
) -> Dict[str, float]:
    """计算凑到各种牌型的条件概率。

    翻牌后：精确枚举所有剩余公共牌组合（无需随机抽样）。
    翻牌前：蒙特卡洛模拟（组合数太大 C(50,5)≈2.1M）。

    Args:
        hole_cards: 底牌（2 张）。
        community_cards: 已知的公共牌（0–5 张）。
        num_simulations: 翻牌前蒙特卡洛模拟次数（默认 2000）。

    Returns:
        {牌型中文名: 概率百分比} 的字典，按牌型等级降序排列。
    """
    community = list(community_cards or [])
    all_known = list(hole_cards) + community

    # 河牌圈：所有牌已知，直接评估
    if len(community) == 5:
        result = HandEvaluator.evaluate(all_known)
        target_rank = result.hand_rank
        probs: Dict[str, float] = {}
        for rank in HandRank:
            probs[rank.display_name] = 100.0 if rank == target_rank else 0.0
        return probs

    # 排除已知牌
    known_set = set(all_known)
    remaining = [
        Card(rank=r, suit=s)
        for r, s in itertools.product(Rank, Suit)
        if Card(rank=r, suit=s) not in known_set
    ]

    needed = 5 - len(community)

    # 翻牌/转牌：精确枚举（flop: C(47,2)=1081, turn: C(46,1)=46 种组合）
    if needed <= 2:
        return _enumerate_hand_type_probs(hole_cards, community, remaining, needed)

    # 翻牌前：蒙特卡洛
    counts = {rank: 0 for rank in HandRank}
    rng = random.Random()
    for _ in range(num_simulations):
        sim_community = community + rng.sample(remaining, needed)
        result = HandEvaluator.evaluate(list(hole_cards) + sim_community)
        counts[result.hand_rank] += 1

    return {
        rank.display_name: round(counts[rank] / num_simulations * 100, 1)
        for rank in reversed(HandRank)
    }


def _enumerate_hand_type_probs(
    hole_cards: Cards,
    community: list[Card],
    remaining: list[Card],
    needed: int,
) -> Dict[str, float]:
    """精确枚举所有剩余公共牌组合，统计牌型分布。"""
    counts = {rank: 0 for rank in HandRank}

    if needed == 1:
        for card in remaining:
            result = HandEvaluator.evaluate(
                list(hole_cards) + community + [card]
            )
            counts[result.hand_rank] += 1
    elif needed == 2:
        n = len(remaining)
        for i in range(n):
            for j in range(i + 1, n):
                result = HandEvaluator.evaluate(
                    list(hole_cards) + community + [remaining[i], remaining[j]]
                )
                counts[result.hand_rank] += 1

    total = sum(counts.values())
    if total == 0:
        # 兜底：不应该到达这里
        rank = HandRank.HIGH_CARD
        return {rank.display_name: 100.0 for rank in HandRank}

    return {
        rank.display_name: round(counts[rank] / total * 100, 1)
        for rank in reversed(HandRank)
    }
