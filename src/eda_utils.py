import pandas as pd
import numpy as np

def compute_loss_ratio(df):
    """Adds LossRatio column to dataframe."""
    df = df.copy()
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    return df

def summarize_by_group(df, group_col, metric='LossRatio'):
    """Returns mean, std, count of a metric grouped by a categorical column."""
    return df.groupby(group_col)[metric].agg(['mean', 'std', 'count'])

def detect_outliers_iqr(df, column):
    """Returns rows where column value is outside 1.5*IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] < lower) | (df[column] > upper)]

def plot_monthly_loss_ratio(df, date_col='TransactionMonth', premium_col='TotalPremium', claims_col='TotalClaims'):
    """Plots monthly loss ratio over time."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.groupby(df[date_col].dt.to_period('M')).agg(
        total_premium=(premium_col, 'sum'),
        total_claims=(claims_col, 'sum')
    )
    monthly['loss_ratio'] = monthly['total_claims'] / monthly['total_premium']
    monthly['loss_ratio'].plot(title='Monthly Loss Ratio', ylabel='Loss Ratio')
    return monthly

