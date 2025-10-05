-- Analytical Queries for Bankruptcy Prediction
-- Sample queries for business intelligence and reporting

-- Query 1: Overall Model Performance Metrics
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN predicted_bankruptcy = bankruptcy_status THEN 1 ELSE 0 END) as correct_predictions,
    CAST(SUM(CASE WHEN predicted_bankruptcy = bankruptcy_status THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as accuracy_percentage,
    SUM(CASE WHEN predicted_bankruptcy = 1 AND bankruptcy_status = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN predicted_bankruptcy = 1 AND bankruptcy_status = 0 THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN predicted_bankruptcy = 0 AND bankruptcy_status = 1 THEN 1 ELSE 0 END) as false_negatives
FROM bankruptcy_analytics.predictions p
JOIN bankruptcy_analytics.financial_metrics fm 
    ON p.company_key = fm.company_key AND p.time_key = fm.time_key
WHERE p.model_version = '1.0';

-- Query 2: Companies at Highest Risk of Bankruptcy
SELECT 
    c.company_name,
    c.industry_sector,
    p.bankruptcy_probability,
    p.risk_level,
    fm.altman_z_score,
    fm.debt_to_equity,
    fm.current_ratio,
    fm.roa,
    fm.revenue,
    t.fiscal_year,
    t.fiscal_quarter
FROM bankruptcy_analytics.predictions p
JOIN bankruptcy_analytics.company_dim c ON p.company_key = c.company_key
JOIN bankruptcy_analytics.financial_metrics fm ON p.company_key = fm.company_key AND p.time_key = fm.time_key
JOIN bankruptcy_analytics.time_dim t ON p.time_key = t.time_key
WHERE p.risk_level IN ('HIGH', 'CRITICAL')
ORDER BY p.bankruptcy_probability DESC
LIMIT 50;

-- Query 3: Industry-wise Bankruptcy Rate Trends
SELECT 
    c.industry_sector,
    t.fiscal_year,
    COUNT(DISTINCT fm.company_key) as total_companies,
    SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) as bankruptcies,
    CAST(SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(DISTINCT fm.company_key) * 100 as bankruptcy_rate,
    AVG(fm.altman_z_score) as avg_altman_z_score
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.company_dim c ON fm.company_key = c.company_key
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
GROUP BY c.industry_sector, t.fiscal_year
ORDER BY t.fiscal_year DESC, bankruptcy_rate DESC;

-- Query 4: Year-over-Year Financial Health Comparison
SELECT 
    c.company_name,
    t.fiscal_year,
    fm.revenue,
    fm.revenue - LAG(fm.revenue) OVER (PARTITION BY c.company_key ORDER BY t.fiscal_year) as revenue_change,
    fm.net_income,
    fm.debt_to_equity,
    fm.altman_z_score,
    fm.bankruptcy_status
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.company_dim c ON fm.company_key = c.company_key
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
WHERE c.company_id IN (SELECT company_id FROM bankruptcy_analytics.company_dim LIMIT 10)
ORDER BY c.company_name, t.fiscal_year;

-- Query 5: Key Financial Ratios Distribution
SELECT 
    CASE 
        WHEN current_ratio < 1.0 THEN '< 1.0 (Poor)'
        WHEN current_ratio < 1.5 THEN '1.0-1.5 (Fair)'
        WHEN current_ratio < 2.0 THEN '1.5-2.0 (Good)'
        ELSE '> 2.0 (Excellent)'
    END as current_ratio_category,
    COUNT(*) as company_count,
    SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) as bankruptcies,
    CAST(SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as bankruptcy_rate
FROM bankruptcy_analytics.financial_metrics
WHERE current_ratio IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- Query 6: Model Prediction Accuracy by Risk Level
SELECT 
    p.risk_level,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.predicted_bankruptcy = fm.bankruptcy_status THEN 1 ELSE 0 END) as correct_predictions,
    CAST(SUM(CASE WHEN p.predicted_bankruptcy = fm.bankruptcy_status THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as accuracy_percentage,
    AVG(p.bankruptcy_probability) as avg_probability
FROM bankruptcy_analytics.predictions p
JOIN bankruptcy_analytics.financial_metrics fm 
    ON p.company_key = fm.company_key AND p.time_key = fm.time_key
GROUP BY p.risk_level
ORDER BY p.risk_level;

-- Query 7: Top Warning Signs of Bankruptcy
SELECT 
    'Negative Cash Flow' as warning_sign,
    COUNT(*) as companies_affected,
    SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) as actual_bankruptcies,
    CAST(SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as bankruptcy_rate
FROM bankruptcy_analytics.financial_metrics
WHERE operating_cash_flow < 0

UNION ALL

SELECT 
    'High Debt to Equity (>2.0)',
    COUNT(*),
    SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END),
    CAST(SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100
FROM bankruptcy_analytics.financial_metrics
WHERE debt_to_equity > 2.0

UNION ALL

SELECT 
    'Low Current Ratio (<1.0)',
    COUNT(*),
    SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END),
    CAST(SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100
FROM bankruptcy_analytics.financial_metrics
WHERE current_ratio < 1.0

UNION ALL

SELECT 
    'Negative Net Income',
    COUNT(*),
    SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END),
    CAST(SUM(CASE WHEN bankruptcy_status = 1 THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100
FROM bankruptcy_analytics.financial_metrics
WHERE net_income < 0

ORDER BY bankruptcy_rate DESC;

-- Query 8: Quarterly Trend Analysis
SELECT 
    t.fiscal_year,
    t.fiscal_quarter,
    COUNT(DISTINCT fm.company_key) as total_companies,
    AVG(fm.revenue) as avg_revenue,
    AVG(fm.net_profit_margin) as avg_profit_margin,
    AVG(fm.roa) as avg_roa,
    AVG(fm.debt_to_equity) as avg_leverage,
    SUM(CASE WHEN fm.bankruptcy_status = 1 THEN 1 ELSE 0 END) as bankruptcies
FROM bankruptcy_analytics.financial_metrics fm
JOIN bankruptcy_analytics.time_dim t ON fm.time_key = t.time_key
GROUP BY t.fiscal_year, t.fiscal_quarter
ORDER BY t.fiscal_year DESC, t.fiscal_quarter DESC;

-- Query 9: Companies with Declining Financial Health
SELECT 
    c.company_name,
    c.industry_sector,
    current.fiscal_year,
    current.altman_z_score as current_z_score,
    prior.altman_z_score as prior_z_score,
    current.altman_z_score - prior.altman_z_score as z_score_change,
    current.debt_to_equity as current_leverage,
    p.bankruptcy_probability,
    p.risk_level
FROM bankruptcy_analytics.financial_metrics current
JOIN bankruptcy_analytics.financial_metrics prior 
    ON current.company_key = prior.company_key 
    AND prior.time_key = current.time_key - 1
JOIN bankruptcy_analytics.company_dim c ON current.company_key = c.company_key
JOIN bankruptcy_analytics.time_dim t ON current.time_key = t.time_key
LEFT JOIN bankruptcy_analytics.predictions p 
    ON current.company_key = p.company_key AND current.time_key = p.time_key
WHERE current.altman_z_score < prior.altman_z_score
ORDER BY z_score_change ASC
LIMIT 100;

-- Query 10: ETL Pipeline Performance Metrics
SELECT 
    DATE_TRUNC('day', etl_processed_timestamp) as processing_date,
    etl_job_name,
    COUNT(*) as records_processed,
    COUNT(DISTINCT company_key) as unique_companies,
    MIN(etl_processed_timestamp) as first_record_time,
    MAX(etl_processed_timestamp) as last_record_time,
    DATEDIFF(minute, MIN(etl_processed_timestamp), MAX(etl_processed_timestamp)) as processing_duration_minutes
FROM bankruptcy_analytics.financial_metrics
WHERE etl_processed_timestamp IS NOT NULL
GROUP BY DATE_TRUNC('day', etl_processed_timestamp), etl_job_name
ORDER BY processing_date DESC;

-- Query 11: Real-time Risk Alert Dashboard
CREATE OR REPLACE VIEW bankruptcy_analytics.v_risk_dashboard AS
SELECT 
    COUNT(DISTINCT CASE WHEN p.risk_level = 'CRITICAL' THEN c.company_key END) as critical_risk_count,
    COUNT(DISTINCT CASE WHEN p.risk_level = 'HIGH' THEN c.company_key END) as high_risk_count,
    COUNT(DISTINCT CASE WHEN p.risk_level = 'MEDIUM' THEN c.company_key END) as medium_risk_count,
    COUNT(DISTINCT CASE WHEN p.risk_level = 'LOW' THEN c.company_key END) as low_risk_count,
    AVG(fm.altman_z_score) as avg_z_score,
    COUNT(DISTINCT CASE WHEN fm.altman_z_score < 1.8 THEN c.company_key END) as distress_zone_count,
    COUNT(DISTINCT CASE WHEN fm.altman_z_score BETWEEN 1.8 AND 3.0 THEN c.company_key END) as grey_zone_count,
    COUNT(DISTINCT CASE WHEN fm.altman_z_score > 3.0 THEN c.company_key END) as safe_zone_count
FROM bankruptcy_analytics.company_dim c
LEFT JOIN bankruptcy_analytics.predictions p ON c.company_key = p.company_key
LEFT JOIN bankruptcy_analytics.financial_metrics fm ON c.company_key = fm.company_key
WHERE (p.company_key, p.time_key) IN (
    SELECT company_key, MAX(time_key)
    FROM bankruptcy_analytics.predictions
    GROUP BY company_key
)
AND (fm.company_key, fm.time_key) IN (
    SELECT company_key, MAX(time_key)
    FROM bankruptcy_analytics.financial_metrics
    GROUP BY company_key
);

-- Query 12: Model Performance Over Time
SELECT 
    DATE_TRUNC('month', p.prediction_timestamp) as prediction_month,
    p.model_version,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN p.predicted_bankruptcy = fm.bankruptcy_status THEN 1 ELSE 0 END) as correct_predictions,
    CAST(SUM(CASE WHEN p.predicted_bankruptcy = fm.bankruptcy_status THEN 1 ELSE 0 END) AS DECIMAL) / 
        COUNT(*) * 100 as accuracy_percentage
FROM bankruptcy_analytics.predictions p
JOIN bankruptcy_analytics.financial_metrics fm 
    ON p.company_key = fm.company_key AND p.time_key = fm.time_key
GROUP BY DATE_TRUNC('month', p.prediction_timestamp), p.model_version
ORDER BY prediction_month DESC, p.model_version;
