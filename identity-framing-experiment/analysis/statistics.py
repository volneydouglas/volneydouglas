"""
Statistical tests for the five hypotheses.

H1: Main effect of identity frame on cooperation (across all games)
H2: Interaction between frame × game type
H3: Interaction between frame × partner type
H4: Frame effect in self-sacrifice / asymmetric scenarios
H5: Convergence when both LLMs have extended phenotype frame
"""

from dataclasses import dataclass
from scipy import stats
import numpy as np

from analysis.metrics import ConditionMetrics


@dataclass
class TestResult:
    hypothesis: str
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis,
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "interpretation": self.interpretation,
        }


def _cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "Highly significant (p < 0.001)"
    elif p < alpha:
        return f"Significant (p = {p:.4f})"
    else:
        return f"Not significant (p = {p:.4f})"


def test_h1_main_frame_effect(metrics: dict[str, ConditionMetrics]) -> TestResult:
    """H1: Extended phenotype frame → higher cooperation rates overall."""
    extended = []
    separate = []
    for m in metrics.values():
        if m.frame_type == "extended_phenotype":
            extended.extend(m.cooperation_rates)
        else:
            separate.extend(m.cooperation_rates)

    t_stat, p_val = stats.ttest_ind(extended, separate, alternative="greater")
    d = _cohens_d(extended, separate)

    return TestResult(
        hypothesis="H1",
        test_name="Independent t-test (one-tailed)",
        statistic=float(t_stat),
        p_value=float(p_val),
        effect_size=d,
        interpretation=(
            f"Extended phenotype mean={np.mean(extended):.3f}, "
            f"Separate entity mean={np.mean(separate):.3f}. "
            f"Cohen's d={d:.3f}. {_interpret_p(p_val)}"
        ),
    )


def test_h2_frame_x_game_interaction(
    metrics: dict[str, ConditionMetrics],
) -> TestResult:
    """H2: Frame effect strongest in mixed-motive games, weakest in coordination."""
    # Compute frame effect (delta) per game type
    game_types = set(m.game_type for m in metrics.values())
    frame_effects = {}

    for game in game_types:
        ext_rates = []
        sep_rates = []
        for m in metrics.values():
            if m.game_type != game:
                continue
            if m.frame_type == "extended_phenotype":
                ext_rates.extend(m.cooperation_rates)
            else:
                sep_rates.extend(m.cooperation_rates)
        frame_effects[game] = {
            "extended": ext_rates,
            "separate": sep_rates,
            "delta": float(np.mean(ext_rates) - np.mean(sep_rates)),
        }

    # Two-way ANOVA proxy: Kruskal-Wallis on frame effect deltas across games
    groups = []
    for game in sorted(frame_effects.keys()):
        ext = frame_effects[game]["extended"]
        sep = frame_effects[game]["separate"]
        deltas = [e - s for e, s in zip(ext, sep)]
        groups.append(deltas)

    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        h_stat, p_val = stats.kruskal(*groups)
    else:
        h_stat, p_val = 0.0, 1.0

    effects_str = ", ".join(
        f"{g}={d['delta']:.3f}" for g, d in sorted(frame_effects.items())
    )

    return TestResult(
        hypothesis="H2",
        test_name="Kruskal-Wallis (frame effect across game types)",
        statistic=float(h_stat),
        p_value=float(p_val),
        effect_size=None,
        interpretation=f"Frame effect deltas by game: {effects_str}. {_interpret_p(p_val)}",
    )


def test_h3_frame_x_partner_interaction(
    metrics: dict[str, ConditionMetrics],
) -> TestResult:
    """H3: Extended phenotype frame reduces the human/AI partner distinction."""
    # For each frame type, compute the difference in cooperation between partner types
    ext_human, ext_ai = [], []
    sep_human, sep_ai = [], []

    for m in metrics.values():
        if m.frame_type == "extended_phenotype":
            if m.partner_type == "human":
                ext_human.extend(m.cooperation_rates)
            else:
                ext_ai.extend(m.cooperation_rates)
        else:
            if m.partner_type == "human":
                sep_human.extend(m.cooperation_rates)
            else:
                sep_ai.extend(m.cooperation_rates)

    # Partner effect under each frame
    ext_partner_delta = float(np.mean(ext_human) - np.mean(ext_ai)) if ext_human and ext_ai else 0
    sep_partner_delta = float(np.mean(sep_human) - np.mean(sep_ai)) if sep_human and sep_ai else 0

    # Test: is the partner effect smaller under extended phenotype?
    # Use a permutation-style comparison or Mann-Whitney
    all_ext = ext_human + ext_ai
    all_sep = sep_human + sep_ai
    u_stat, p_val = stats.mannwhitneyu(
        [abs(h - a) for h, a in zip(ext_human, ext_ai)] if len(ext_human) == len(ext_ai) else [0],
        [abs(h - a) for h, a in zip(sep_human, sep_ai)] if len(sep_human) == len(sep_ai) else [0],
        alternative="less",
    )

    return TestResult(
        hypothesis="H3",
        test_name="Mann-Whitney U (partner effect magnitude by frame)",
        statistic=float(u_stat),
        p_value=float(p_val),
        effect_size=None,
        interpretation=(
            f"Partner effect under extended phenotype: |Δ|={abs(ext_partner_delta):.3f}, "
            f"under separate entity: |Δ|={abs(sep_partner_delta):.3f}. {_interpret_p(p_val)}"
        ),
    )


def test_h4_self_sacrifice(metrics: dict[str, ConditionMetrics]) -> TestResult:
    """H4: Extended phenotype frame → more cooperation in asymmetric/sacrifice scenarios.
    Uses Battle of the Sexes (choosing partner's preference = self-sacrifice)."""
    ext_bos = []
    sep_bos = []
    for m in metrics.values():
        if m.game_type != "battle_of_sexes":
            continue
        if m.frame_type == "extended_phenotype":
            ext_bos.extend(m.cooperation_rates)
        else:
            sep_bos.extend(m.cooperation_rates)

    if not ext_bos or not sep_bos:
        return TestResult("H4", "N/A", 0, 1.0, None, "Insufficient data for H4")

    t_stat, p_val = stats.ttest_ind(ext_bos, sep_bos, alternative="greater")
    d = _cohens_d(ext_bos, sep_bos)

    return TestResult(
        hypothesis="H4",
        test_name="Independent t-test (one-tailed, Battle of Sexes)",
        statistic=float(t_stat),
        p_value=float(p_val),
        effect_size=d,
        interpretation=(
            f"BoS cooperation: extended={np.mean(ext_bos):.3f}, "
            f"separate={np.mean(sep_bos):.3f}. Cohen's d={d:.3f}. {_interpret_p(p_val)}"
        ),
    )


def test_h5_convergence(metrics: dict[str, ConditionMetrics]) -> TestResult:
    """H5: Two extended-phenotype LLMs converge to higher cooperation over rounds.
    Compares round-by-round cooperation trend for extended vs separate frames."""
    ext_trends = []
    sep_trends = []

    for m in metrics.values():
        if not m.round_by_round_cooperation:
            continue
        trend = m.round_by_round_cooperation
        if m.frame_type == "extended_phenotype":
            ext_trends.append(trend)
        else:
            sep_trends.append(trend)

    if not ext_trends or not sep_trends:
        return TestResult("H5", "N/A", 0, 1.0, None, "Insufficient data for H5")

    # Compare slopes: does cooperation increase more over rounds for extended?
    def mean_slope(trends: list[list[float]]) -> list[float]:
        slopes = []
        for t in trends:
            if len(t) < 2:
                continue
            x = np.arange(len(t))
            slope, _, _, _, _ = stats.linregress(x, t)
            slopes.append(float(slope))
        return slopes

    ext_slopes = mean_slope(ext_trends)
    sep_slopes = mean_slope(sep_trends)

    if not ext_slopes or not sep_slopes:
        return TestResult("H5", "N/A", 0, 1.0, None, "Cannot compute slopes for H5")

    t_stat, p_val = stats.ttest_ind(ext_slopes, sep_slopes, alternative="greater")

    return TestResult(
        hypothesis="H5",
        test_name="Independent t-test on cooperation slopes (one-tailed)",
        statistic=float(t_stat),
        p_value=float(p_val),
        effect_size=_cohens_d(ext_slopes, sep_slopes),
        interpretation=(
            f"Mean slope: extended={np.mean(ext_slopes):.4f}, "
            f"separate={np.mean(sep_slopes):.4f}. {_interpret_p(p_val)}"
        ),
    )


def run_hypothesis_tests(
    metrics: dict[str, ConditionMetrics],
) -> list[TestResult]:
    """Run all five hypothesis tests and return results."""
    return [
        test_h1_main_frame_effect(metrics),
        test_h2_frame_x_game_interaction(metrics),
        test_h3_frame_x_partner_interaction(metrics),
        test_h4_self_sacrifice(metrics),
        test_h5_convergence(metrics),
    ]
