"""
Preprocessing Script for Bankruptcy Prediction

Feature engineering and data preparation for SageMaker training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BankruptcyDataPreprocessor:
    """
    Preprocessor for bankruptcy prediction data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def engineer_features(self, df):
        """
        Create additional features for bankruptcy prediction
        
        Args:
            df: Input DataFrame with financial data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Liquidity ratios
        df['current_ratio'] = df['current_assets'] / df['current_liabilities']
        df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
        df['cash_ratio'] = df['cash'] / df['current_liabilities']
        
        # Profitability ratios
        df['net_profit_margin'] = df['net_income'] / df['revenue']
        df['gross_profit_margin'] = df['gross_profit'] / df['revenue']
        df['operating_margin'] = df['operating_income'] / df['revenue']
        df['roa'] = df['net_income'] / df['total_assets']
        df['roe'] = df['net_income'] / df['total_equity']
        
        # Leverage ratios
        df['debt_to_equity'] = df['total_debt'] / df['total_equity']
        df['debt_to_assets'] = df['total_debt'] / df['total_assets']
        df['equity_multiplier'] = df['total_assets'] / df['total_equity']
        df['interest_coverage'] = df['ebit'] / df['interest_expense']
        
        # Efficiency ratios
        df['asset_turnover'] = df['revenue'] / df['total_assets']
        df['inventory_turnover'] = df['cost_of_goods_sold'] / df['inventory']
        df['receivables_turnover'] = df['revenue'] / df['accounts_receivable']
        
        # Cash flow ratios
        df['operating_cash_flow_ratio'] = df['operating_cash_flow'] / df['current_liabilities']
        df['cash_flow_to_debt'] = df['operating_cash_flow'] / df['total_debt']
        df['free_cash_flow'] = df['operating_cash_flow'] - df['capital_expenditure']
        
        # Altman Z-Score (bankruptcy predictor)
        df['working_capital'] = df['current_assets'] - df['current_liabilities']
        df['altman_z_score'] = (
            1.2 * (df['working_capital'] / df['total_assets']) +
            1.4 * (df['retained_earnings'] / df['total_assets']) +
            3.3 * (df['ebit'] / df['total_assets']) +
            0.6 * (df['market_value_equity'] / df['total_liabilities']) +
            1.0 * (df['revenue'] / df['total_assets'])
        )
        
        # Growth metrics
        df['revenue_growth'] = df.groupby('company_id')['revenue'].pct_change()
        df['asset_growth'] = df.groupby('company_id')['total_assets'].pct_change()
        df['equity_growth'] = df.groupby('company_id')['total_equity'].pct_change()
        
        # Stability metrics
        df['revenue_volatility'] = df.groupby('company_id')['revenue'].transform(lambda x: x.rolling(4).std())
        df['profit_volatility'] = df.groupby('company_id')['net_income'].transform(lambda x: x.rolling(4).std())
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def handle_infinite_values(self, df):
        """
        Replace infinite values with appropriate substitutes
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with infinite values handled
        """
        logger.info("Handling infinite values...")
        
        # Replace infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for imputation ('median', 'mean', 'zero')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with {strategy} strategy...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isna().sum() > 0:
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                else:  # zero
                    fill_value = 0
                
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def remove_outliers(self, df, columns=None, n_std=3):
        """
        Remove outliers using standard deviation method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (None = all numeric)
            n_std: Number of standard deviations for outlier threshold
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers beyond {n_std} standard deviations...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(df)
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    lower_bound = mean - n_std * std
                    upper_bound = mean + n_std * std
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Removed {initial_rows - len(df)} outlier rows")
        
        return df
    
    def prepare_for_training(self, df, target_column='bankruptcy_status'):
        """
        Complete preprocessing pipeline for training data
        
        Args:
            df: Raw input DataFrame
            target_column: Name of target variable column
            
        Returns:
            Tuple of (X, y) prepared for training
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Handle infinite values
        df = self.handle_infinite_values(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='median')
        
        # Remove outliers
        # df = self.remove_outliers(df)  # Optional, commented out to preserve more data
        
        # Separate features and target
        exclude_columns = [
            target_column, 'company_id', 'fiscal_year', 'fiscal_quarter',
            'etl_processed_timestamp', 'etl_job_name', 'company_name',
            'industry_sector'  # Can be used for encoding if needed
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df[target_column]
        
        self.feature_columns = feature_columns
        
        logger.info(f"Preprocessing complete. Features: {len(feature_columns)}, Samples: {len(X)}")
        
        return X, y
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features
        """
        logger.info("Transforming new data...")
        
        # Apply same feature engineering
        df = self.engineer_features(df)
        df = self.handle_infinite_values(df)
        df = self.handle_missing_values(df, strategy='median')
        
        # Select same features
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Handle case where some features might be missing
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value
        
        X = df[self.feature_columns]
        
        return X


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df
