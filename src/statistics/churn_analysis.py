from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def compare_churn_groups(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "churn",
) -> pd.DataFrame:
    """
    Compare numeric customer features between churned
    and non-churned customers using the Mann-Whitney U test.

    Returns one row per feature containing:
        - feature
        - non_churned_median
        - churned_median
        - u_statistic
        - p_value
        - effect_size
        - adjusted_p_value
        - significant

    Effect size is calculated as:

        r = |Z| / sqrt(N)

    where N is the total number of observations used
    for the feature.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    target_values = set(
        df[target_column].dropna().unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    results = []

    for feature in feature_columns:

        if feature not in df.columns:
            raise ValueError(
                f"Missing feature column: {feature}"
            )

        if not pd.api.types.is_numeric_dtype(
            df[feature]
        ):
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

        non_churned = (
            df.loc[
                df[target_column] == 0,
                feature,
            ]
            .dropna()
        )

        churned = (
            df.loc[
                df[target_column] == 1,
                feature,
            ]
            .dropna()
        )

        if len(non_churned) == 0 or len(churned) == 0:
            raise ValueError(
                f"Both churn groups must contain "
                f"observations for '{feature}'."
            )

        test_result = mannwhitneyu(
            non_churned,
            churned,
            alternative="two-sided",
        )

        u_statistic = float(
            test_result.statistic
        )

        p_value = float(
            test_result.pvalue
        )

        n_total = (
            len(non_churned)
            + len(churned)
        )

        # Convert U to a standardized Z approximation.
        n1 = len(non_churned)
        n2 = len(churned)

        mean_u = n1 * n2 / 2

        std_u = np.sqrt(
            n1 * n2 * (n1 + n2 + 1) / 12
        )

        if std_u == 0:
            z_score = 0.0
        else:
            z_score = (
                u_statistic - mean_u
            ) / std_u

        effect_size = (
            abs(z_score)
            / np.sqrt(n_total)
        )

        results.append({
            "feature": feature,
            "non_churned_median": float(
                non_churned.median()
            ),
            "churned_median": float(
                churned.median()
            ),
            "u_statistic": u_statistic,
            "p_value": p_value,
            "effect_size": float(
                effect_size
            ),
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        rejected, adjusted_p_values, _, _ = (
            multipletests(
                results_df["p_value"],
                method="fdr_bh",
                alpha=0.05,
            )
        )

        results_df[
            "adjusted_p_value"
        ] = adjusted_p_values

        results_df[
            "significant"
        ] = rejected

    return results_df


def summarize_significant_features(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return statistically significant features,
    ordered by adjusted p-value.
    """

    required_columns = {
        "feature",
        "adjusted_p_value",
        "significant",
    }

    missing = required_columns - set(
        results_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing result columns: {sorted(missing)}"
        )

    return (
        results_df[
            results_df["significant"]
        ]
        .sort_values(
            "adjusted_p_value"
        )
        .reset_index(drop=True)
    )