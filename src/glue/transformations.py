"""
Data Transformation Functions for Bankruptcy Prediction

Contains reusable transformation functions for the ETL pipeline.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import logging

logger = logging.getLogger()


def normalize_column_names(df: DataFrame) -> DataFrame:
    """
    Standardize column names to lowercase with underscores
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    logger.info("Normalizing column names...")
    
    for col in df.columns:
        new_col = col.lower().replace(' ', '_').replace('-', '_')
        df = df.withColumnRenamed(col, new_col)
    
    return df


def handle_missing_values(df: DataFrame, strategy: str = 'median') -> DataFrame:
    """
    Handle missing values in numeric columns
    
    Args:
        df: Input DataFrame
        strategy: 'median', 'mean', or 'zero'
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values with strategy: {strategy}")
    
    numeric_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, DoubleType)]
    
    for col in numeric_cols:
        if strategy == 'median':
            fill_value = df.approxQuantile(col, [0.5], 0.01)[0]
        elif strategy == 'mean':
            fill_value = df.select(F.mean(col)).first()[0]
        else:  # zero
            fill_value = 0.0
        
        df = df.fillna({col: fill_value})
    
    return df


def create_bankruptcy_features(df: DataFrame) -> DataFrame:
    """
    Create features specifically for bankruptcy prediction
    
    Args:
        df: DataFrame with financial data
        
    Returns:
        DataFrame with bankruptcy prediction features
    """
    logger.info("Creating bankruptcy-specific features...")
    
    # Working capital to total assets
    df = df.withColumn('working_capital_ratio',
                       (F.col('current_assets') - F.col('current_liabilities')) / 
                       F.col('total_assets'))
    
    # Retained earnings to total assets
    df = df.withColumn('retained_earnings_ratio',
                       F.col('retained_earnings') / F.col('total_assets'))
    
    # EBIT to total assets
    df = df.withColumn('ebit_to_assets',
                       F.col('ebit') / F.col('total_assets'))
    
    # Market value equity to book value liabilities
    df = df.withColumn('market_to_book_ratio',
                       F.col('market_value_equity') / F.col('total_liabilities'))
    
    # Sales to total assets
    df = df.withColumn('sales_to_assets',
                       F.col('revenue') / F.col('total_assets'))
    
    # Cash flow adequacy
    df = df.withColumn('cash_flow_adequacy',
                       F.col('operating_cash_flow') / F.col('total_debt'))
    
    # Interest coverage ratio
    df = df.withColumn('interest_coverage',
                       F.col('ebit') / F.col('interest_expense'))
    
    # Operating cash flow to current liabilities
    df = df.withColumn('cash_flow_coverage',
                       F.col('operating_cash_flow') / F.col('current_liabilities'))
    
    logger.info("Bankruptcy features created successfully")
    return df


def create_trend_features(df: DataFrame) -> DataFrame:
    """
    Create time-series trend features
    
    Args:
        df: DataFrame with temporal financial data
        
    Returns:
        DataFrame with trend features
    """
    logger.info("Creating trend features...")
    
    # Define window for time-based calculations
    window_1y = Window.partitionBy('company_id').orderBy('fiscal_year').rowsBetween(-3, 0)
    window_2y = Window.partitionBy('company_id').orderBy('fiscal_year').rowsBetween(-7, 0)
    
    # Moving averages
    df = df.withColumn('revenue_ma_1y', F.avg('revenue').over(window_1y))
    df = df.withColumn('net_income_ma_1y', F.avg('net_income').over(window_1y))
    
    # Volatility (standard deviation)
    df = df.withColumn('revenue_volatility', F.stddev('revenue').over(window_2y))
    df = df.withColumn('profit_volatility', F.stddev('net_income').over(window_2y))
    
    # Trend direction (positive or negative)
    df = df.withColumn('revenue_trend',
                       F.when(F.col('revenue') > F.col('revenue_ma_1y'), 1).otherwise(0))
    
    logger.info("Trend features created successfully")
    return df


def create_industry_benchmarks(df: DataFrame) -> DataFrame:
    """
    Create features based on industry benchmarks
    
    Args:
        df: DataFrame with company financial data
        
    Returns:
        DataFrame with industry benchmark features
    """
    logger.info("Creating industry benchmark features...")
    
    # Window for industry-level calculations
    industry_window = Window.partitionBy('industry_sector', 'fiscal_year')
    
    # Industry median metrics
    df = df.withColumn('industry_median_roa',
                       F.expr('percentile_approx(roa, 0.5)').over(industry_window))
    
    df = df.withColumn('industry_median_debt_ratio',
                       F.expr('percentile_approx(debt_to_equity, 0.5)').over(industry_window))
    
    # Company performance vs industry
    df = df.withColumn('roa_vs_industry',
                       F.col('roa') - F.col('industry_median_roa'))
    
    df = df.withColumn('debt_vs_industry',
                       F.col('debt_to_equity') - F.col('industry_median_debt_ratio'))
    
    logger.info("Industry benchmark features created successfully")
    return df


def apply_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Apply complete feature engineering pipeline
    
    Args:
        df: Raw financial DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Normalize column names
    df = normalize_column_names(df)
    
    # Handle missing values
    df = handle_missing_values(df, strategy='median')
    
    # Create bankruptcy features
    df = create_bankruptcy_features(df)
    
    # Create trend features
    df = create_trend_features(df)
    
    # Create industry benchmarks (if industry data available)
    if 'industry_sector' in df.columns:
        df = create_industry_benchmarks(df)
    
    # Replace infinite values with nulls
    numeric_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, DoubleType)]
    
    for col in numeric_cols:
        df = df.withColumn(col, 
                          F.when(F.col(col).isNotNull() & ~F.isnan(col) & 
                                (F.col(col) != float('inf')) & 
                                (F.col(col) != float('-inf')), 
                                F.col(col)).otherwise(None))
    
    logger.info("Feature engineering pipeline completed")
    return df


def partition_data_for_ml(df: DataFrame, 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.15) -> tuple:
    """
    Partition data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Partitioning data: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}")
    
    # Add random split column
    df = df.withColumn('split_random', F.rand(seed=42))
    
    train_df = df.filter(F.col('split_random') < train_ratio)
    val_df = df.filter((F.col('split_random') >= train_ratio) & 
                      (F.col('split_random') < train_ratio + val_ratio))
    test_df = df.filter(F.col('split_random') >= train_ratio + val_ratio)
    
    # Remove split column
    train_df = train_df.drop('split_random')
    val_df = val_df.drop('split_random')
    test_df = test_df.drop('split_random')
    
    logger.info(f"Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}")
    
    return train_df, val_df, test_df
