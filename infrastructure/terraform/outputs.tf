# Terraform Outputs for Bankruptcy Prediction Infrastructure

output "data_lake_bucket_name" {
  description = "Name of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_bucket_arn" {
  description = "ARN of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.arn
}

output "model_artifacts_bucket_name" {
  description = "Name of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "model_artifacts_bucket_arn" {
  description = "ARN of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "glue_database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.bankruptcy_db.name
}

output "glue_job_name" {
  description = "Name of the Glue ETL job"
  value       = aws_glue_job.etl_job.name
}

output "glue_role_arn" {
  description = "ARN of the Glue IAM role"
  value       = aws_iam_role.glue_role.arn
}

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker IAM role"
  value       = aws_iam_role.sagemaker_role.arn
}

output "sagemaker_notebook_name" {
  description = "Name of the SageMaker notebook instance"
  value       = aws_sagemaker_notebook_instance.notebook.name
}

output "sagemaker_notebook_url" {
  description = "URL of the SageMaker notebook instance"
  value       = aws_sagemaker_notebook_instance.notebook.url
}

output "glue_logs_group" {
  description = "CloudWatch log group for Glue jobs"
  value       = aws_cloudwatch_log_group.glue_logs.name
}

output "sagemaker_logs_group" {
  description = "CloudWatch log group for SageMaker"
  value       = aws_cloudwatch_log_group.sagemaker_logs.name
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}
