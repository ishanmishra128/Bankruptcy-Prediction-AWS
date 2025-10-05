import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from awsglue import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node S3 Ratios
S3Ratios_node1759345349135 = glueContext.create_dynamic_frame.from_options(format_options={"quoteChar": "\"", "withHeader": True, "separator": ",", "optimizePerformance": False}, connection_type="s3", format="csv", connection_options={"paths": ["s3://msba-financial-222/training-data/ratios-analyst-a/msba_fg_ratio_data.csv"], "recurse": True}, transformation_ctx="S3Ratios_node1759345349135")

# Script generated for node S3 Financials
S3Financials_node1759345413451 = glueContext.create_dynamic_frame.from_options(format_options={"quoteChar": "\"", "withHeader": True, "separator": ",", "optimizePerformance": False}, connection_type="s3", format="csv", connection_options={"paths": ["s3://msba-financial-222/training-data/financials/datacorp_financial_data.csv"], "recurse": True}, transformation_ctx="S3Financials_node1759345413451")

# Script generated for node Join
S3Financials_node1759345413451DF = S3Financials_node1759345413451.toDF()
S3Ratios_node1759345349135DF = S3Ratios_node1759345349135.toDF()
Join_node1759345505219 = DynamicFrame.fromDF(S3Financials_node1759345413451DF.join(S3Ratios_node1759345349135DF, (S3Financials_node1759345413451DF['company_id'] == S3Ratios_node1759345349135DF['ID']), "outer"), glueContext, "Join_node1759345505219")

# Script generated for node Change Schema
ChangeSchema_node1759345565368 = ApplyMapping.apply(frame=Join_node1759345505219, mappings=[("company_id", "string", "company_id", "string"), ("net_worth_to_assets", "string", "net_worth_to_assets", "string"), ("retained_earnings_to_total_assets", "string", "retained_earnings_to_total_assets", "string"), ("working_capital_to_total_assets", "string", "working_capital_to_total_assets", "string"), ("working_capital_to_equity", "string", "working_capital_to_equity", "string"), ("equity_to_longterm_liability", "string", "equity_to_longterm_liability", "string"), ("current_liabilities_to_equity", "string", "current_liabilities_to_equity", "string"), ("liability_to_equity", "string", "liability_to_equity", "string"), ("current_liability_to_current_assets", "string", "current_liability_to_current_assets", "string"), ("borrowing_dependency", "string", "borrowing_dependency", "string"), ("debt_ratio_percentage", "string", "debt_ratio_percentage", "string"), ("persistent_eps", "string", "persistent_eps", "string"), ("per_share_net_profit_pre_tax", "string", "per_share_net_profit_pre_tax", "string"), ("operating_profit_per_share", "string", "operating_profit_per_share", "string"), ("tax_rate", "string", "tax_rate", "string"), ("operating_gross_margin", "string", "operating_gross_margin", "string"), ("Net_Income_to_Total_Assets", "string", "Net_Income_to_Total_Assets", "decimal"), ("ROA_before_interest_percent_after_tax", "string", "ROA_before_interest_percent_after_tax", "decimal"), ("Net_profit_before_tax_to_Paid_in_capital", "string", "Net_profit_before_tax_to_Paid_in_capital", "decimal"), ("Net_Income_to_Stockholders_Equity", "string", "Net_Income_to_Stockholders_Equity", "decimal"), ("Operating_profit_Paid_in_capital", "string", "Operating_profit_Paid_in_capital", "decimal"), ("Total_Asset_Turnover", "string", "Total_Asset_Turnover", "decimal"), ("Total_expense_to_Assets", "string", "Total_expense_to_Assets", "decimal")], transformation_ctx="ChangeSchema_node1759345565368")

# Script generated for node Amazon Redshift
AmazonRedshift_node1759346396780 = glueContext.write_dynamic_frame.from_options(frame=ChangeSchema_node1759345565368, connection_type="redshift", connection_options={"redshiftTmpDir": "s3://aws-glue-assets-058264534855-us-east-1/temporary/", "useConnectionProperties": "true", "dbtable": "public.msba_financials_unified", "connectionName": "msba-jdbc-connection", "preactions": "CREATE TABLE IF NOT EXISTS public.msba_financials_unified (company_id VARCHAR, net_worth_to_assets VARCHAR, retained_earnings_to_total_assets VARCHAR, working_capital_to_total_assets VARCHAR, working_capital_to_equity VARCHAR, equity_to_longterm_liability VARCHAR, current_liabilities_to_equity VARCHAR, liability_to_equity VARCHAR, current_liability_to_current_assets VARCHAR, borrowing_dependency VARCHAR, debt_ratio_percentage VARCHAR, persistent_eps VARCHAR, per_share_net_profit_pre_tax VARCHAR, operating_profit_per_share VARCHAR, tax_rate VARCHAR, operating_gross_margin VARCHAR, Net_Income_to_Total_Assets DECIMAL, ROA_before_interest_percent_after_tax DECIMAL, Net_profit_before_tax_to_Paid_in_capital DECIMAL, Net_Income_to_Stockholders_Equity DECIMAL, Operating_profit_Paid_in_capital DECIMAL, Total_Asset_Turnover DECIMAL, Total_expense_to_Assets DECIMAL);"}, transformation_ctx="AmazonRedshift_node1759346396780")

job.commit()