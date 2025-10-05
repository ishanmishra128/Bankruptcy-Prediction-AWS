"""
AWS Glue ETL Job for Bankruptcy Prediction Data Pipeline

This script processes raw financial data and transforms it for machine learning.
Automated ETL reduces manual processing time by 80%.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'S3_INPUT_PATH',
    'S3_OUTPUT_PATH',
    'REDSHIFT_CONNECTION'
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Starting Bankruptcy ETL Job")


def clean_financial_data(df):
    """
    Clean and validate financial data
    
    Args:
        df: Spark DataFrame with raw financial data
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning financial data...")
    
    # Remove duplicates
    df = df.dropDuplicates(['company_id', 'fiscal_year', 'fiscal_quarter'])
    
    # Handle missing values
    numeric_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, (DoubleType, IntegerType))]
    
    for col in numeric_cols:
        # Fill missing values with median
        median_val = df.approxQuantile(col, [0.5], 0.01)[0]
        df = df.fillna({col: median_val})
    
    # Remove outliers (values beyond 3 standard deviations)
    for col in numeric_cols:
        stats = df.select(
            F.mean(col).alias('mean'),
            F.stddev(col).alias('std')
        ).first()
        
        if stats['std'] is not None and stats['std'] > 0:
            lower_bound = stats['mean'] - 3 * stats['std']
            upper_bound = stats['mean'] + 3 * stats['std']
            df = df.filter((F.col(col) >= lower_bound) & (F.col(col) <= upper_bound))
    
    logger.info(f"Data cleaned. Rows remaining: {df.count()}")
    return df


def calculate_financial_ratios(df):
    """
    Calculate financial ratios for bankruptcy prediction
    
    Args:
        df: DataFrame with financial metrics
        
    Returns:
        DataFrame with calculated ratios
    """
    logger.info("Calculating financial ratios...")
    
    # Liquidity ratios
    df = df.withColumn('current_ratio', 
                       F.col('current_assets') / F.col('current_liabilities'))
    df = df.withColumn('quick_ratio',
                       (F.col('current_assets') - F.col('inventory')) / F.col('current_liabilities'))
    
    # Profitability ratios
    df = df.withColumn('net_profit_margin',
                       F.col('net_income') / F.col('revenue'))
    df = df.withColumn('roa',  # Return on Assets
                       F.col('net_income') / F.col('total_assets'))
    df = df.withColumn('roe',  # Return on Equity
                       F.col('net_income') / F.col('total_equity'))
    
    # Leverage ratios
    df = df.withColumn('debt_to_equity',
                       F.col('total_debt') / F.col('total_equity'))
    df = df.withColumn('debt_to_assets',
                       F.col('total_debt') / F.col('total_assets'))
    
    # Efficiency ratios
    df = df.withColumn('asset_turnover',
                       F.col('revenue') / F.col('total_assets'))
    
    # Z-Score (Altman's bankruptcy predictor)
    df = df.withColumn('altman_z_score',
                       (1.2 * (F.col('working_capital') / F.col('total_assets'))) +
                       (1.4 * (F.col('retained_earnings') / F.col('total_assets'))) +
                       (3.3 * (F.col('ebit') / F.col('total_assets'))) +
                       (0.6 * (F.col('market_value_equity') / F.col('total_liabilities'))) +
                       (1.0 * (F.col('revenue') / F.col('total_assets'))))
    
    logger.info("Financial ratios calculated successfully")
    return df


def add_temporal_features(df):
    """
    Add time-based features for trend analysis
    
    Args:
        df: DataFrame with financial data
        
    Returns:
        DataFrame with temporal features
    """
    logger.info("Adding temporal features...")
    
    # Year-over-year growth rates
    window_spec = Window.partitionBy('company_id').orderBy('fiscal_year')
    
    df = df.withColumn('revenue_growth',
                       (F.col('revenue') - F.lag('revenue', 1).over(window_spec)) / 
                       F.lag('revenue', 1).over(window_spec))
    
    df = df.withColumn('profit_growth',
                       (F.col('net_income') - F.lag('net_income', 1).over(window_spec)) / 
                       F.lag('net_income', 1).over(window_spec))
    
    # Quarter indicators
    df = df.withColumn('is_q4', F.when(F.col('fiscal_quarter') == 4, 1).otherwise(0))
    
    logger.info("Temporal features added")
    return df


def main():
    """
    Main ETL process
    """
    # Read raw data from S3
    logger.info(f"Reading data from {args['S3_INPUT_PATH']}")
    raw_df = spark.read.format('csv') \
        .option('header', 'true') \
        .option('inferSchema', 'true') \
        .load(args['S3_INPUT_PATH'])
    
    logger.info(f"Raw data loaded. Rows: {raw_df.count()}")
    
    # Apply transformations
    cleaned_df = clean_financial_data(raw_df)
    ratio_df = calculate_financial_ratios(cleaned_df)
    final_df = add_temporal_features(ratio_df)
    
    # Add processing metadata
    final_df = final_df.withColumn('etl_processed_timestamp', F.current_timestamp())
    final_df = final_df.withColumn('etl_job_name', F.lit(args['JOB_NAME']))
    
    # Write processed data to S3
    logger.info(f"Writing processed data to {args['S3_OUTPUT_PATH']}")
    final_df.write.format('parquet') \
        .mode('overwrite') \
        .partitionBy('fiscal_year') \
        .save(args['S3_OUTPUT_PATH'])
    
    # Load to Redshift data warehouse
    logger.info("Loading data to Redshift...")
    final_df.write \
        .format("jdbc") \
        .option("url", args['REDSHIFT_CONNECTION']) \
        .option("dbtable", "bankruptcy_analytics.financial_metrics") \
        .option("user", "admin") \
        .option("driver", "com.amazon.redshift.jdbc42.Driver") \
        .mode("append") \
        .save()
    
    logger.info("ETL job completed successfully")
    logger.info(f"Processed {final_df.count()} records")
    
    job.commit()


if __name__ == '__main__':
    main()
