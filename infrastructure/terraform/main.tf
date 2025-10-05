# Terraform configuration for Bankruptcy Prediction AWS Infrastructure
# Alternative to CloudFormation for infrastructure as code

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "bankruptcy-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# S3 Data Lake Bucket
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name = "Data Lake for Bankruptcy Prediction"
  }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Model Artifacts Bucket
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project_name}-models-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name = "Model Artifacts"
  }
}

resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Glue Database
resource "aws_glue_catalog_database" "bankruptcy_db" {
  name        = "${var.project_name}_${var.environment}"
  description = "Database for bankruptcy prediction pipeline"
}

# Glue IAM Role
resource "aws_iam_role" "glue_role" {
  name = "${var.project_name}-glue-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "glue.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3_access" {
  name = "s3-access"
  role = aws_iam_role.glue_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.data_lake.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn
        ]
      }
    ]
  })
}

# Glue ETL Job
resource "aws_glue_job" "etl_job" {
  name     = "${var.project_name}-etl-job-${var.environment}"
  role_arn = aws_iam_role.glue_role.arn
  
  command {
    name            = "glueetl"
    script_location = "s3://${aws_s3_bucket.data_lake.bucket}/scripts/glue/etl_job.py"
    python_version  = "3"
  }
  
  default_arguments = {
    "--job-language"                     = "python"
    "--job-bookmark-option"              = "job-bookmark-enable"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--S3_INPUT_PATH"                    = "s3://${aws_s3_bucket.data_lake.bucket}/raw/"
    "--S3_OUTPUT_PATH"                   = "s3://${aws_s3_bucket.data_lake.bucket}/processed/"
    "--TempDir"                          = "s3://${aws_s3_bucket.data_lake.bucket}/temp/"
  }
  
  max_retries       = 1
  timeout           = 120
  glue_version      = "4.0"
  number_of_workers = 10
  worker_type       = "G.1X"
  
  tags = {
    Name = "Bankruptcy ETL Job"
  }
}

# SageMaker IAM Role
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "notebook" {
  name          = "${var.project_name}-notebook-${var.environment}"
  instance_type = "ml.t3.medium"
  role_arn      = aws_iam_role.sagemaker_role.arn
  
  tags = {
    Name = "Bankruptcy Prediction Notebook"
  }
}

# CloudWatch Log Group for Glue
resource "aws_cloudwatch_log_group" "glue_logs" {
  name              = "/aws-glue/jobs/${var.project_name}"
  retention_in_days = 30
}

# CloudWatch Log Group for SageMaker
resource "aws_cloudwatch_log_group" "sagemaker_logs" {
  name              = "/aws/sagemaker/${var.project_name}"
  retention_in_days = 30
}

# Data sources
data "aws_caller_identity" "current" {}

# Outputs
output "data_lake_bucket_name" {
  description = "Name of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "model_artifacts_bucket_name" {
  description = "Name of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "glue_job_name" {
  description = "Name of the Glue ETL job"
  value       = aws_glue_job.etl_job.name
}

output "sagemaker_notebook_url" {
  description = "URL of the SageMaker notebook instance"
  value       = aws_sagemaker_notebook_instance.notebook.url
}
