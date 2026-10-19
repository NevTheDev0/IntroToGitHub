#!/usr/bin/env python3
"""
ACADEMIC COMPLIANCE INTELLIGENCE PLATFORM™
Proceedings of the International Conference on Mandatory Git Repositories

A needlessly elaborate, yet genuinely trained, machine-learning system for
estimating the 'administrative satisfaction' of a GitHub repository.

DISCLAIMER: The training data is entirely synthetic. This is satire, not a
real grading tool, and the output has no bearing on anyone's actual grade.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Eight features. Four models would be unnecessary. So naturally, we use two.
FEATURES = [
    "commit_count",
    "branch_count",
    "remote_connected",
    "ssh_remote_enabled",
    "tracked_file_count",
    "narrative_screenshots",
    "fifty_word_reflections",
    "python_lines_of_code",
]
SEED = 1000  # Dedicated to the course that inspired this invention.


def git(*arguments: str) -> str:
    """Safely inspect Git without making any changes to the repository."""
    try:
        process = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        return process.stdout.strip() if process.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def fabricate_research_dataset(n: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    """Manufacture a reproducible academic dataset with exactly zero respondents."""
    rng = np.random.default_rng(SEED)
    commits = rng.integers(0, 12, size=n)
    branches = rng.integers(0, 6, size=n)
    remote = rng.binomial(1, 0.65, size=n)
    ssh = remote * rng.binomial(1, 0.50, size=n)
    files = rng.integers(0, 15, size=n)
    screenshots = rng.integers(0, 5, size=n)
    reflections = rng.integers(0, 5, size=n)
    python_lines = rng.integers(0, 1_500, size=n)

    X = np.column_stack(
        [commits, branches, remote, ssh, files, screenshots, reflections, python_lines]
    ).astype(float)

    # Synthetic 'grading rubric' invented for this joke. Not a real rubric.
    bureaucratic_approval_score = (
        1.8 * remote
        + 1.6 * ssh
        + 1.7 * (commits >= 1)
        + 1.0 * (files >= 1)
        + 1.3 * (screenshots >= 2)
        + 1.2 * (reflections >= 3)
        + 0.4 * (branches >= 1)
        + 0.7 * np.log1p(python_lines) / np.log(1500)
        + rng.normal(0, 0.55, size=n)
    )
    y = (bureaucratic_approval_score >= 6.0).astype(int)
    return X, y


def audit_actual_repository(screenshots: int, reflections: int) -> dict[str, int]:
    """Read real local Git metadata, while accepting narrative facts as flags."""
    remote = git("remote", "get-url", "origin")
    commits_text = git("rev-list", "--count", "HEAD")
    branches = git("branch", "--format=%(refname:short)")
    files = [line for line in git("ls-files").splitlines() if line]

    python_lines = 0
    for name in files:
        path = Path(name)
        if path.suffix == ".py" and path.is_file():
            try:
                python_lines += len(path.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeError):
                pass

    return dict(
        commit_count=min(int(commits_text), 11) if commits_text.isdigit() else 0,
        branch_count=min(len(branches.splitlines()), 5),
        remote_connected=int(bool(remote)),
        ssh_remote_enabled=int(remote.startswith(("git@", "ssh://"))),
        tracked_file_count=min(len(files), 14),
        narrative_screenshots=min(max(screenshots, 0), 4),
        fifty_word_reflections=min(max(reflections, 0), 4),
        python_lines_of_code=min(python_lines, 1499),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an absurd academic ML audit.")
    parser.add_argument(
        "--screenshots", type=int, default=0,
        help="Number of screenshots you actually added to your narrative",
    )
    parser.add_argument(
        "--reflections", type=int, default=0,
        help="Number of completed 50-word narrative reflections",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("    ACADEMIC COMPLIANCE INTELLIGENCE PLATFORM™  |  RESEARCH BUILD 8.3")
    print("    A peer-review-resistant solution to a participation-grade problem")
    print("=" * 72)
    print("[1/5] Generating 2,000 entirely fictional GitHub repositories...")
    X, y = fabricate_research_dataset()
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    model = VotingClassifier(
        estimators=[
            ("forest", RandomForestClassifier(
                n_estimators=128, min_samples_leaf=2, random_state=SEED, n_jobs=1
            )),
            ("logistic", make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2_000)
            )),
        ],
        voting="soft",
    )

    print("[2/5] Conducting four-fold cross-validation for no discernible reason...")
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=4, scoring="accuracy")

    print("[3/5] Training an ensemble to replicate a rubric that we invented...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("[4/5] Running permutation feature importance. Science requires charts.")
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=3, random_state=SEED, scoring="accuracy"
    )
    rank = sorted(
        zip(FEATURES, importance.importances_mean), key=lambda pair: -pair[1]
    )

    print("[5/5] Subjecting this very repository to the bureaucracy classifier...")
    observed = audit_actual_repository(args.screenshots, args.reflections)
    actual_array = np.array([[observed[name] for name in FEATURES]], dtype=float)
    probability = float(model.predict_proba(actual_array)[0, 1])

    print("\n── UNNECESSARILY RIGOROUS VALIDATION ──")
    print(f"Cross-validation accuracy : {cv_accuracy.mean():.1%} ± {cv_accuracy.std():.1%}")
    print(f"Synthetic test accuracy   : {accuracy:.1%}")
    print(f"Synthetic test ROC-AUC    : {auc:.3f}")
    print(f"Confusion matrix          : {confusion_matrix(y_test, predictions).tolist()}")
    print(f"Training corpus SHA-256   : {digest[:24]}... (for reproducibility, obviously)")

    print("\n── TOP ARTIFICIALLY IMPORTANT VARIABLES ──")
    for feature, score in rank[:5]:
        print(f"  {feature:26s} {score:+.4f}")

    print("\n── HIGH-STAKES REPOSITORY EVALUATION ──")
    print(f"Estimated administrative satisfaction: {probability:.1%}")
    if probability >= 0.75:
        print("VERDICT: This repository is statistically overqualified to exist.")
    else:
        print("VERDICT: Additional documentation may appease the imaginary committee.")
    print("FINAL RECOMMENDATION: Please award the participation points.")
    print("DISCLAIMER: ALL TRAINING DATA IS SYNTHETIC. THIS IS A JOKE, NOT A GRADE.")

    report = {
        "title": "Toward a Neural-Adjacent Theory of Mandatory GitHub Compliance",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": "100% synthetic; not reflective of any professor or class",
        "training_dataset_sha256": digest,
        "cross_validation_accuracy_mean": float(cv_accuracy.mean()),
        "synthetic_test_accuracy": float(accuracy),
        "synthetic_test_roc_auc": float(auc),
        "repository_features": observed,
        "imaginary_approval_probability": probability,
        "top_feature_importances": {key: float(value) for key, value in rank[:5]},
        "conclusion": "An entire ML ensemble was deployed to request participation points.",
    }
    output = Path("bureaucratic_intelligence_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nAudit artifact exported: {output.resolve()}")


if __name__ == "__main__":
    main()
