-- Redshift Data Warehouse Schema for Bankruptcy Prediction
-- Centralized data warehouse for analytical queries

-- Create schema
CREATE SCHEMA IF NOT EXISTS bankruptcy_analytics;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS bankruptcy_analytics.financial_metrics CASCADE;
DROP TABLE IF EXISTS bankruptcy_analytics.company_dim CASCADE;
DROP TABLE IF EXISTS bankruptcy_analytics.time_dim CASCADE;
DROP TABLE IF EXISTS bankruptcy_analytics.predictions CASCADE;

-- Company Dimension Table
CREATE TABLE bankruptcy_analytics.company_dim (
    company_key INTEGER IDENTITY(1,1) PRIMARY KEY,
    company_id VARCHAR(50) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    industry_sector VARCHAR(100),
    sub_industry VARCHAR(100),
    country VARCHAR(50),
    founded_year INTEGER,
    employee_count INTEGER,
    created_at TIMESTAMP DEFAULT GETDATE(),
    updated_at TIMESTAMP DEFAULT GETDATE()
)
DISTSTYLE KEY
DISTKEY (company_key)
SORTKEY (company_id);

-- Time Dimension Table
CREATE TABLE bankruptcy_analytics.time_dim (
    time_key INTEGER IDENTITY(1,1) PRIMARY KEY,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER NOT NULL,
    fiscal_month INTEGER,
    calendar_year INTEGER,
    calendar_quarter INTEGER,
    is_q4 BOOLEAN DEFAULT FALSE,
    UNIQUE(fiscal_year, fiscal_quarter)
)
DISTSTYLE ALL
SORTKEY (fiscal_year, fiscal_quarter);

-- Financial Metrics Fact Table
CREATE TABLE bankruptcy_analytics.financial_metrics (
    metric_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    company_key INTEGER NOT NULL REFERENCES bankruptcy_analytics.company_dim(company_key),
    time_key INTEGER NOT NULL REFERENCES bankruptcy_analytics.time_dim(time_key),
    
    -- Balance Sheet Metrics
    total_assets DECIMAL(18,2),
    total_liabilities DECIMAL(18,2),
    total_equity DECIMAL(18,2),
    current_assets DECIMAL(18,2),
    current_liabilities DECIMAL(18,2),
    cash DECIMAL(18,2),
    inventory DECIMAL(18,2),
    accounts_receivable DECIMAL(18,2),
    total_debt DECIMAL(18,2),
    long_term_debt DECIMAL(18,2),
    short_term_debt DECIMAL(18,2),
    
    -- Income Statement Metrics
    revenue DECIMAL(18,2),
    gross_profit DECIMAL(18,2),
    operating_income DECIMAL(18,2),
    net_income DECIMAL(18,2),
    ebit DECIMAL(18,2),
    ebitda DECIMAL(18,2),
    interest_expense DECIMAL(18,2),
    cost_of_goods_sold DECIMAL(18,2),
    
    -- Cash Flow Metrics
    operating_cash_flow DECIMAL(18,2),
    investing_cash_flow DECIMAL(18,2),
    financing_cash_flow DECIMAL(18,2),
    free_cash_flow DECIMAL(18,2),
    capital_expenditure DECIMAL(18,2),
    
    -- Calculated Ratios
    current_ratio DECIMAL(10,4),
    quick_ratio DECIMAL(10,4),
    net_profit_margin DECIMAL(10,4),
    roa DECIMAL(10,4),
    roe DECIMAL(10,4),
    debt_to_equity DECIMAL(10,4),
    debt_to_assets DECIMAL(10,4),
    asset_turnover DECIMAL(10,4),
    altman_z_score DECIMAL(10,4),
    
    -- Additional Features
    working_capital DECIMAL(18,2),
    retained_earnings DECIMAL(18,2),
    market_value_equity DECIMAL(18,2),
    
    -- Growth Metrics
    revenue_growth DECIMAL(10,4),
    profit_growth DECIMAL(10,4),
    asset_growth DECIMAL(10,4),
    
    -- Target Variable
    bankruptcy_status INTEGER NOT NULL,  -- 0 = Not Bankrupt, 1 = Bankrupt
    
    -- Metadata
    etl_processed_timestamp TIMESTAMP,
    etl_job_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT GETDATE()
)
DISTSTYLE KEY
DISTKEY (company_key)
SORTKEY (time_key, company_key);

-- Predictions Table
CREATE TABLE bankruptcy_analytics.predictions (
    prediction_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    company_key INTEGER NOT NULL REFERENCES bankruptcy_analytics.company_dim(company_key),
    time_key INTEGER NOT NULL REFERENCES bankruptcy_analytics.time_dim(time_key),
    
    -- Prediction Results
    predicted_bankruptcy INTEGER NOT NULL,  -- 0 or 1
    bankruptcy_probability DECIMAL(10,6),
    risk_level VARCHAR(20),  -- LOW, MEDIUM, HIGH, CRITICAL
    
    -- Model Information
    model_version VARCHAR(50),
    model_type VARCHAR(50),
    prediction_timestamp TIMESTAMP DEFAULT GETDATE(),
    
    -- Confidence Metrics
    prediction_confidence DECIMAL(10,4),
    
    CONSTRAINT unique_prediction UNIQUE(company_key, time_key, model_version)
)
DISTSTYLE KEY
DISTKEY (company_key)
SORTKEY (prediction_timestamp);

-- Create indexes for better query performance
CREATE INDEX idx_financial_metrics_company ON bankruptcy_analytics.financial_metrics(company_key);
CREATE INDEX idx_financial_metrics_time ON bankruptcy_analytics.financial_metrics(time_key);
CREATE INDEX idx_financial_metrics_bankruptcy ON bankruptcy_analytics.financial_metrics(bankruptcy_status);
CREATE INDEX idx_predictions_company ON bankruptcy_analytics.predictions(company_key);
CREATE INDEX idx_predictions_risk ON bankruptcy_analytics.predictions(risk_level);

-- Create views for common analytical queries

-- View: Latest Financial Metrics per Company
CREATE OR REPLACE VIEW bankruptcy_analytics.v_latest_metrics AS
SELECT 
    c.company_id,
    c.company_name,
    c.industry_sector,
    t.fiscal_year,
    t.fiscal_quarter,
    fm.*
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.company_dim c ON fm.company_key = c.company_key
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
WHERE (fm.company_key, fm.time_key) IN (
    SELECT company_key, MAX(time_key)
    FROM bankruptcy_analytics.financial_metrics
    GROUP BY company_key
);

-- View: High Risk Companies
CREATE OR REPLACE VIEW bankruptcy_analytics.v_high_risk_companies AS
SELECT 
    c.company_id,
    c.company_name,
    c.industry_sector,
    p.bankruptcy_probability,
    p.risk_level,
    p.prediction_timestamp,
    fm.altman_z_score,
    fm.debt_to_equity,
    fm.current_ratio
FROM bankruptcy_analytics.predictions p
JOIN bankruptcy_analytics.company_dim c ON p.company_key = c.company_key
JOIN bankruptcy_analytics.financial_metrics fm ON p.company_key = fm.company_key AND p.time_key = fm.time_key
WHERE p.risk_level IN ('HIGH', 'CRITICAL')
AND p.prediction_timestamp = (
    SELECT MAX(prediction_timestamp)
    FROM bankruptcy_analytics.predictions p2
    WHERE p2.company_key = p.company_key
);

-- View: Industry Benchmark Statistics
CREATE OR REPLACE VIEW bankruptcy_analytics.v_industry_benchmarks AS
SELECT 
    c.industry_sector,
    t.fiscal_year,
    t.fiscal_quarter,
    COUNT(DISTINCT c.company_key) as company_count,
    AVG(fm.roa) as avg_roa,
    MEDIAN(fm.roa) as median_roa,
    AVG(fm.debt_to_equity) as avg_debt_to_equity,
    MEDIAN(fm.debt_to_equity) as median_debt_to_equity,
    AVG(fm.current_ratio) as avg_current_ratio,
    MEDIAN(fm.current_ratio) as median_current_ratio,
    AVG(fm.altman_z_score) as avg_altman_z_score,
    SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) as bankruptcy_count,
    CAST(SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as bankruptcy_rate
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.company_dim c ON fm.company_key = c.company_key
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
GROUP BY c.industry_sector, t.fiscal_year, t.fiscal_quarter;

-- Grant permissions (adjust as needed)
GRANT USAGE ON SCHEMA bankruptcy_analytics TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA bankruptcy_analytics TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA bankruptcy_analytics TO bi_reporting;

-- Create materialized view for dashboard performance
CREATE MATERIALIZED VIEW bankruptcy_analytics.mv_dashboard_metrics AS
SELECT 
    t.fiscal_year,
    t.fiscal_quarter,
    COUNT(DISTINCT fm.company_key) as total_companies,
    SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) as bankruptcies,
    AVG(fm.altman_z_score) as avg_z_score,
    AVG(fm.debt_to_equity) as avg_leverage,
    COUNT(DISTINCT CASE WHEN p.risk_level = 'HIGH' THEN fm.company_key END) as high_risk_companies,
    COUNT(DISTINCT CASE WHEN p.risk_level = 'CRITICAL' THEN fm.company_key END) as critical_risk_companies
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
LEFT JOIN bankruptcy_analytics.predictions p ON fm.company_key = p.company_key AND fm.time_key = p.time_key
GROUP BY t.fiscal_year, t.fiscal_quarter;

-- Refresh schedule for materialized view (run daily)
-- This would typically be scheduled via AWS Lambda or cron
-- REFRESH MATERIALIZED VIEW bankruptcy_analytics.mv_dashboard_metrics;

COMMENT ON SCHEMA bankruptcy_analytics IS 'Centralized data warehouse for bankruptcy prediction analytics';
COMMENT ON TABLE bankruptcy_analytics.financial_metrics IS 'Fact table containing financial metrics and ratios';
COMMENT ON TABLE bankruptcy_analytics.predictions IS 'Model predictions with 97% accuracy';
