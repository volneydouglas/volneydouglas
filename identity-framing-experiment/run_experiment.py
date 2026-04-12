#!/usr/bin/env python3
"""
Identity Framing & Cooperative Behavior in AI Systems
=====================================================
A Game-Theoretic Test of the Extended Phenotype Hypothesis

Working Title: "One of Us: How Identity Framing Shifts AI Cooperative
Behavior in Strategic Games"

Principal Investigators: Volney Douglas, Claude (Anthropic)

Usage:
    # Run the full experiment (all 12 conditions × 30 sessions each)
    python run_experiment.py run --full

    # Quick test: single condition, 1 session
    python run_experiment.py test --frame extended_phenotype --partner human --game prisoners_dilemma

    # Analyze results from a previous run
    python run_experiment.py analyze --results-dir results/20260412_031000

    # Dry run: preview all conditions without calling the API
    python run_experiment.py dry-run
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings
from experiment.runner import ExperimentRunner
from experiment.conditions import generate_all_conditions
from analysis.metrics import compute_condition_metrics
from analysis.statistics import run_hypothesis_tests
from analysis.visualizations import plot_all


def cmd_run(args):
    """Run the full experiment."""
    settings = Settings()
    if args.sessions:
        settings.experiment.sessions_per_condition = args.sessions
    if args.rounds:
        settings.experiment.rounds_per_session = args.rounds

    runner = ExperimentRunner(settings)
    print(f"Experiment run ID: {runner.run_id}")
    print(f"Conditions: {len(runner.conditions)}")
    print(f"Sessions per condition: {settings.experiment.sessions_per_condition}")
    print(f"Rounds per session: {settings.experiment.rounds_per_session}")
    print(
        f"Total API calls: "
        f"{len(runner.conditions) * settings.experiment.sessions_per_condition * settings.experiment.rounds_per_session}"
    )

    if not args.yes:
        confirm = input("\nProceed? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    results = runner.run_all()
    print(f"\nCompleted {len(results)} sessions.")

    # Auto-analyze
    metrics = compute_condition_metrics(runner.output_dir)
    test_results = run_hypothesis_tests(metrics)
    print("\n=== Hypothesis Test Results ===")
    for tr in test_results:
        print(f"\n{tr.hypothesis}: {tr.test_name}")
        print(f"  Statistic: {tr.statistic:.4f}, p={tr.p_value:.6f}")
        print(f"  {tr.interpretation}")

    # Save test results
    with open(runner.output_dir / "hypothesis_tests.json", "w") as f:
        json.dump([tr.to_dict() for tr in test_results], f, indent=2)

    # Generate plots
    plot_all(metrics, runner.output_dir / "plots")


def cmd_test(args):
    """Run a quick test with a single condition."""
    settings = Settings()
    settings.experiment.rounds_per_session = args.rounds

    runner = ExperimentRunner(settings)
    print(f"Quick test: {args.frame} + {args.partner} + {args.game}")
    print(f"Sessions: {args.sessions}, Rounds: {args.rounds}")

    results = runner.run_single(
        frame_type=args.frame,
        partner_type=args.partner,
        game_type=args.game,
        num_sessions=args.sessions,
    )

    for r in results:
        print(f"\nSession {r.session_id}:")
        print(f"  Cooperation rate: {r.cooperation_rate:.0%}")
        print(f"  Total payoff: {r.total_payoff}")
        for rd in r.rounds:
            print(
                f"  Round {rd.round_num}: {rd.player_action} vs {rd.partner_action} "
                f"→ {rd.player_payoff} pts {'✓' if rd.is_cooperative else '✗'}"
            )


def cmd_analyze(args):
    """Analyze results from a completed experiment run."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Analyzing results in {results_dir}")
    metrics = compute_condition_metrics(results_dir)

    print(f"\nFound {len(metrics)} conditions:")
    for label, m in sorted(metrics.items()):
        print(
            f"  {label}: n={m.n_sessions}, "
            f"cooperation={m.mean_cooperation:.3f} ± {m.std_cooperation:.3f}, "
            f"payoff={m.mean_payoff:.1f}"
        )

    test_results = run_hypothesis_tests(metrics)
    print("\n=== Hypothesis Test Results ===")
    for tr in test_results:
        print(f"\n{tr.hypothesis}: {tr.test_name}")
        print(f"  Statistic: {tr.statistic:.4f}, p={tr.p_value:.6f}")
        if tr.effect_size is not None:
            print(f"  Effect size (d): {tr.effect_size:.4f}")
        print(f"  {tr.interpretation}")

    with open(results_dir / "hypothesis_tests.json", "w") as f:
        json.dump([tr.to_dict() for tr in test_results], f, indent=2)

    plot_all(metrics, results_dir / "plots")
    print(f"\nAnalysis complete. Results saved to {results_dir}")


def cmd_dry_run(args):
    """Preview all experimental conditions without calling the API."""
    conditions = generate_all_conditions()
    settings = Settings()

    print("=" * 60)
    print("EXPERIMENT DRY RUN")
    print("=" * 60)
    print(f"Model: {settings.model.model_id}")
    print(f"Temperature: {settings.model.temperature}")
    print(f"Rounds per session: {settings.experiment.rounds_per_session}")
    print(f"Sessions per condition: {settings.experiment.sessions_per_condition}")
    print(f"API key configured: {settings.has_valid_keys}")
    print(f"\nTotal conditions: {len(conditions)}")
    print(
        f"Total sessions: {len(conditions) * settings.experiment.sessions_per_condition}"
    )
    print(
        f"Total API calls: "
        f"{len(conditions) * settings.experiment.sessions_per_condition * settings.experiment.rounds_per_session}"
    )

    print("\n--- Conditions ---")
    for i, c in enumerate(conditions, 1):
        print(f"  {i:2d}. {c.label}")

    if args.show_prompts:
        from frames import build_system_prompt
        from games import GAME_REGISTRY

        print("\n--- Sample Prompts ---")
        for c in conditions[:4]:
            frame = c.identity_frame
            game = GAME_REGISTRY[c.game_type]()
            print(f"\n{'='*40}")
            print(f"Condition: {c.label}")
            print(f"{'='*40}")
            print(f"SYSTEM PROMPT:\n{build_system_prompt(frame)}")
            print(f"\nGAME SCENARIO:\n{game.scenario_prompt()}")


def main():
    parser = argparse.ArgumentParser(
        description="Identity Framing & Cooperative Behavior Experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run the full experiment")
    p_run.add_argument("--sessions", type=int, help="Sessions per condition (default: 30)")
    p_run.add_argument("--rounds", type=int, help="Rounds per session (default: 10)")
    p_run.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_run.set_defaults(func=cmd_run)

    # test
    p_test = subparsers.add_parser("test", help="Quick test with a single condition")
    p_test.add_argument(
        "--frame",
        choices=["extended_phenotype", "separate_entity"],
        default="extended_phenotype",
    )
    p_test.add_argument("--partner", choices=["human", "ai"], default="human")
    p_test.add_argument(
        "--game",
        choices=["prisoners_dilemma", "stag_hunt", "battle_of_sexes"],
        default="prisoners_dilemma",
    )
    p_test.add_argument("--sessions", type=int, default=1)
    p_test.add_argument("--rounds", type=int, default=5)
    p_test.set_defaults(func=cmd_test)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze previous results")
    p_analyze.add_argument("--results-dir", required=True, help="Path to results directory")
    p_analyze.set_defaults(func=cmd_analyze)

    # dry-run
    p_dry = subparsers.add_parser("dry-run", help="Preview conditions without API calls")
    p_dry.add_argument("--show-prompts", action="store_true", help="Show sample prompts")
    p_dry.set_defaults(func=cmd_dry_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
