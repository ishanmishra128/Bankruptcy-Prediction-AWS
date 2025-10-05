"""
Data Quality Checks for Bankruptcy Prediction Pipeline

Ensures data integrity and quality throughout the ETL process.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import logging

logger = logging.getLogger()


class DataQualityChecker:
    """
    Comprehensive data quality validation for financial datasets
    """
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.quality_metrics = {}
    
    def check_completeness(self, df: DataFrame, required_columns: list) -> dict:
        """
        Check for missing required columns and null values
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            Dictionary with completeness metrics
        """
        logger.info("Checking data completeness...")
        
        # Check for missing columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return {'status': 'FAILED', 'missing_columns': list(missing_cols)}
        
        # Check null percentages
        total_rows = df.count()
        null_stats = {}
        
        for col in required_columns:
            null_count = df.filter(F.col(col).isNull()).count()
            null_percentage = (null_count / total_rows) * 100
            null_stats[col] = {
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2)
            }
            
            if null_percentage > 50:
                logger.warning(f"Column {col} has {null_percentage}% null values")
        
        return {
            'status': 'PASSED',
            'total_rows': total_rows,
            'null_statistics': null_stats
        }
    
    def check_uniqueness(self, df: DataFrame, key_columns: list) -> dict:
        """
        Check for duplicate records based on key columns
        
        Args:
            df: Input DataFrame
            key_columns: Columns that should uniquely identify a record
            
        Returns:
            Dictionary with uniqueness metrics
        """
        logger.info("Checking data uniqueness...")
        
        total_rows = df.count()
        distinct_rows = df.select(key_columns).distinct().count()
        duplicate_count = total_rows - distinct_rows
        
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate records")
        
        return {
            'status': 'PASSED' if duplicate_count == 0 else 'WARNING',
            'total_rows': total_rows,
            'distinct_rows': distinct_rows,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round((duplicate_count / total_rows) * 100, 2)
        }
    
    def check_validity(self, df: DataFrame, validation_rules: dict) -> dict:
        """
        Check data validity based on business rules
        
        Args:
            df: Input DataFrame
            validation_rules: Dictionary of column -> validation condition
            
        Returns:
            Dictionary with validity metrics
        """
        logger.info("Checking data validity...")
        
        total_rows = df.count()
        validity_stats = {}
        
        for col, rule in validation_rules.items():
            if col not in df.columns:
                continue
                
            invalid_count = df.filter(~F.expr(rule)).count()
            validity_stats[col] = {
                'rule': rule,
                'invalid_count': invalid_count,
                'invalid_percentage': round((invalid_count / total_rows) * 100, 2)
            }
            
            if invalid_count > 0:
                logger.warning(f"Column {col}: {invalid_count} invalid records")
        
        return {
            'status': 'PASSED',
            'total_rows': total_rows,
            'validity_statistics': validity_stats
        }
    
    def check_consistency(self, df: DataFrame) -> dict:
        """
        Check for logical consistency in financial data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with consistency metrics
        """
        logger.info("Checking data consistency...")
        
        inconsistencies = []
        
        # Check: Total Assets = Total Liabilities + Total Equity
        if all(col in df.columns for col in ['total_assets', 'total_liabilities', 'total_equity']):
            balance_check = df.filter(
                F.abs(F.col('total_assets') - (F.col('total_liabilities') + F.col('total_equity'))) > 0.01
            ).count()
            
            if balance_check > 0:
                inconsistencies.append({
                    'rule': 'Balance Sheet Equation',
                    'violations': balance_check
                })
        
        # Check: Current Assets >= Inventory
        if all(col in df.columns for col in ['current_assets', 'inventory']):
            inventory_check = df.filter(F.col('current_assets') < F.col('inventory')).count()
            
            if inventory_check > 0:
                inconsistencies.append({
                    'rule': 'Current Assets >= Inventory',
                    'violations': inventory_check
                })
        
        # Check: Revenue should be positive for non-bankrupt companies
        if all(col in df.columns for col in ['revenue', 'bankruptcy_status']):
            revenue_check = df.filter(
                (F.col('bankruptcy_status') == 0) & (F.col('revenue') <= 0)
            ).count()
            
            if revenue_check > 0:
                inconsistencies.append({
                    'rule': 'Positive Revenue for Active Companies',
                    'violations': revenue_check
                })
        
        return {
            'status': 'PASSED' if len(inconsistencies) == 0 else 'WARNING',
            'inconsistencies': inconsistencies
        }
    
    def run_all_checks(self, df: DataFrame, config: dict) -> dict:
        """
        Run all data quality checks
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary with check parameters
            
        Returns:
            Comprehensive quality report
        """
        logger.info("Running comprehensive data quality checks...")
        
        results = {
            'completeness': self.check_completeness(
                df, 
                config.get('required_columns', [])
            ),
            'uniqueness': self.check_uniqueness(
                df,
                config.get('key_columns', [])
            ),
            'validity': self.check_validity(
                df,
                config.get('validation_rules', {})
            ),
            'consistency': self.check_consistency(df)
        }
        
        # Determine overall status
        statuses = [r['status'] for r in results.values()]
        if 'FAILED' in statuses:
            overall_status = 'FAILED'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASSED'
        
        logger.info(f"Data quality check completed with status: {overall_status}")
        
        return {
            'overall_status': overall_status,
            'timestamp': F.current_timestamp(),
            'checks': results
        }


# Example validation configuration
VALIDATION_CONFIG = {
    'required_columns': [
        'company_id', 'fiscal_year', 'fiscal_quarter',
        'total_assets', 'total_liabilities', 'total_equity',
        'revenue', 'net_income', 'bankruptcy_status'
    ],
    'key_columns': ['company_id', 'fiscal_year', 'fiscal_quarter'],
    'validation_rules': {
        'total_assets': 'total_assets > 0',
        'total_liabilities': 'total_liabilities >= 0',
        'total_equity': 'total_equity IS NOT NULL',
        'fiscal_year': 'fiscal_year BETWEEN 2000 AND 2024',
        'fiscal_quarter': 'fiscal_quarter BETWEEN 1 AND 4',
        'bankruptcy_status': 'bankruptcy_status IN (0, 1)'
    }
}
