import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

def chi2_test_frequency(df, group_col, target_col='HasClaim'):
    """
    Chi-squared test for claim frequency (binary target) between two groups.
    Returns dict with chi2 statistic, p-value, degrees of freedom, and decision.
    """
    contingency = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency)
    decision = 'Reject H0' if p < 0.05 else 'Fail to reject'
    return {
        'statistic': chi2,
        'p_value': p,
        'degrees_freedom': dof,
        'decision': decision,
        'test': 'Chi-squared'
    }

def t_test_severity(df, group_col, claim_col='TotalClaims'):
    """
    Two-sample t-test for claim severity (claims > 0) between two groups.
    Assumes approximately normal distribution; use with caution.
    """
    df_claims = df[df[claim_col] > 0]
    groups = df_claims[group_col].unique()
    if len(groups) != 2:
        raise ValueError("t-test requires exactly two groups")
    group_a = df_claims[df_claims[group_col] == groups[0]][claim_col]
    group_b = df_claims[df_claims[group_col] == groups[1]][claim_col]
    t_stat, p = ttest_ind(group_a, group_b, equal_var=False)
    decision = 'Reject H0' if p < 0.05 else 'Fail to reject'
    return {
        'statistic': t_stat,
        'p_value': p,
        'decision': decision,
        'test': 't-test (unequal var)'
    }

def mannwhitney_margin(df, group_col, margin_col='Margin'):
    """
    Mann-Whitney U test for margin (non-parametric, robust to outliers).
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError("Mann-Whitney requires exactly two groups")
    a = df[df[group_col] == groups[0]][margin_col]
    b = df[df[group_col] == groups[1]][margin_col]
    u_stat, p = mannwhitneyu(a, b, alternative='two-sided')
    decision = 'Reject H0' if p < 0.05 else 'Fail to reject'
    return {
        'statistic': u_stat,
        'p_value': p,
        'decision': decision,
        'test': 'Mann-Whitney U'
    }
