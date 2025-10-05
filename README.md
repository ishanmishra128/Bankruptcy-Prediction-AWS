# Bankruptcy Prediction - AWS Data Pipeline

An end-to-end AWS data pipeline solution for predicting corporate bankruptcy using machine learning, featuring automated ETL, centralized data warehousing, and real-time analytics capabilities.

## 🏗️ Architecture Overview

This project implements a scalable, cloud-native architecture on AWS that consolidates financial and bankruptcy data to predict corporate financial distress with **97% accuracy**.

### Key Components

#### 📦 Amazon S3 - Data Lake
- **Raw Data Zone**: Stores unprocessed financial datasets and bankruptcy records
- **Processed Data Zone**: Contains cleaned, transformed, and feature-engineered datasets
- **Model Artifacts**: Stores trained models, evaluation metrics, and predictions

#### 🔄 AWS Glue - ETL Pipeline
- Automated ETL processes that reduced manual processing time by **80%**
- Data quality checks and validation
- Schema evolution and cataloging
- Incremental data loading strategies

#### 🗄️ Amazon Redshift - Data Warehouse
- Centralized analytical data warehouse
- Optimized star schema for bankruptcy analytics
- Support for complex analytical queries
- Historical trend analysis and reporting

#### 🤖 Amazon SageMaker - Machine Learning
- Trained bankruptcy prediction model with **97% accuracy** on validation data
- Automated model training and hyperparameter tuning
- Real-time inference endpoints
- Model monitoring and retraining pipelines

### Architecture Diagram

```
┌─────────────┐
│   Raw Data  │
│  (CSV/JSON) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Amazon S3     │
│   Data Lake     │
│  ┌───────────┐  │
│  │ Raw Zone  │  │
│  └─────┬─────┘  │
└────────┼────────┘
         │
         ▼
┌─────────────────┐
│   AWS Glue      │
│  ETL Pipeline   │
│  - Clean Data   │
│  - Transform    │
│  - Validate     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│  S3     │ │   Redshift   │
│Processed│ │ Data Warehouse│
│  Zone   │ │              │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            │
            ▼
   ┌─────────────────┐
   │  SageMaker      │
   │  ML Training    │
   │  97% Accuracy   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  Real-time      │
   │  Predictions    │
   └─────────────────┘
```

## 🚀 Features

- **Automated Data Pipeline**: End-to-end automation from raw data ingestion to predictions
- **Scalable Architecture**: Cloud-native design supporting millions of records
- **Real-time Analytics**: Low-latency predictions for bankruptcy risk assessment
- **80% Efficiency Gain**: Reduced manual data processing time through automation
- **High Accuracy**: 97% accuracy in bankruptcy prediction on validation dataset
- **Data Quality**: Built-in validation and quality checks at every stage
- **Cost-Optimized**: Efficient use of AWS services with automatic scaling

## 📊 Dataset

The pipeline processes financial datasets containing:
- Balance sheet metrics (assets, liabilities, equity)
- Income statement data (revenue, expenses, profit margins)
- Cash flow indicators
- Financial ratios (liquidity, solvency, profitability)
- Historical bankruptcy labels

## 🛠️ Technology Stack

- **Cloud Platform**: Amazon Web Services (AWS)
- **Data Lake**: Amazon S3
- **ETL Processing**: AWS Glue
- **Data Warehouse**: Amazon Redshift
- **Machine Learning**: Amazon SageMaker
- **Programming**: Python 3.8+
- **Infrastructure**: AWS CloudFormation / Terraform
- **Monitoring**: Amazon CloudWatch

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── glue/
│   │   ├── etl_job.py              # Main ETL job script
│   │   ├── data_quality.py         # Data validation functions
│   │   └── transformations.py      # Data transformation logic
│   ├── sagemaker/
│   │   ├── training.py             # Model training script
│   │   ├── inference.py            # Prediction endpoint
│   │   ├── preprocessing.py        # Feature engineering
│   │   └── evaluation.py           # Model evaluation metrics
│   └── lambda/
│       └── trigger_pipeline.py     # Pipeline orchestration
├── sql/
│   ├── redshift_schema.sql         # Data warehouse schema
│   └── analytical_queries.sql      # Sample analytical queries
├── infrastructure/
│   ├── cloudformation/
│   │   ├── s3_bucket.yaml         # S3 bucket configuration
│   │   ├── glue_jobs.yaml         # Glue job definitions
│   │   ├── redshift_cluster.yaml  # Redshift cluster setup
│   │   └── sagemaker_resources.yaml # SageMaker resources
│   └── terraform/
│       ├── main.tf                # Main Terraform config
│       ├── variables.tf           # Variable definitions
│       └── outputs.tf             # Output values
├── config/
│   ├── s3_structure.yaml          # S3 folder structure
│   └── pipeline_config.json       # Pipeline configuration
└── notebooks/
    ├── exploratory_analysis.ipynb # Data exploration
    └── model_experiments.ipynb    # ML experiments
```

## 🔧 Setup and Installation

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Python 3.8 or higher
- Terraform (optional, for IaC deployment)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ishanmishra128/Bankruptcy-Prediction-AWS.git
   cd Bankruptcy-Prediction-AWS
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Deploy infrastructure**
   ```bash
   # Using CloudFormation
   aws cloudformation create-stack --stack-name bankruptcy-pipeline \
     --template-body file://infrastructure/cloudformation/main.yaml

   # OR using Terraform
   cd infrastructure/terraform
   terraform init
   terraform apply
   ```

5. **Upload initial data to S3**
   ```bash
   aws s3 cp data/ s3://bankruptcy-data-lake/raw/ --recursive
   ```

## 🏃 Running the Pipeline

### ETL Pipeline

```bash
# Trigger Glue ETL job
aws glue start-job-run --job-name bankruptcy-etl-job
```

### Model Training

```bash
# Start SageMaker training job
python src/sagemaker/training.py --config config/pipeline_config.json
```

### Real-time Predictions

```bash
# Deploy SageMaker endpoint
python src/sagemaker/inference.py --deploy

# Make predictions
python src/sagemaker/inference.py --predict --input data.json
```

## 📈 Performance Metrics

- **Model Accuracy**: 97% on validation dataset
- **Precision**: 96%
- **Recall**: 95%
- **F1 Score**: 95.5%
- **ETL Processing Time**: Reduced by 80% through automation
- **Data Processing Capacity**: 1M+ records per hour
- **Prediction Latency**: < 100ms for real-time inference

## 🔍 Key Achievements

1. ✅ **Automated ETL Pipeline**: Reduced manual processing time by 80%
2. ✅ **High Accuracy Model**: Achieved 97% accuracy on validation data
3. ✅ **Scalable Architecture**: Cloud-native design supporting real-time analytics
4. ✅ **Centralized Data Warehouse**: Single source of truth for analytical queries
5. ✅ **Cost Optimization**: Efficient resource utilization with auto-scaling

## 📊 Data Flow

1. **Ingestion**: Raw financial data lands in S3 raw zone
2. **Cataloging**: AWS Glue Crawler catalogs the data
3. **ETL Processing**: Glue jobs clean, transform, and validate data
4. **Storage**: Processed data stored in S3 and loaded to Redshift
5. **Training**: SageMaker trains bankruptcy prediction models
6. **Deployment**: Models deployed as real-time endpoints
7. **Analytics**: Business users query Redshift for insights

## 🔐 Security

- IAM roles and policies for least-privilege access
- S3 bucket encryption at rest
- Redshift encryption and VPC isolation
- SageMaker endpoint security
- CloudTrail logging for audit

## 📝 License

This project is available for educational and portfolio purposes.

## 👤 Author

**Ishan Mishra**
- GitHub: [@ishanmishra128](https://github.com/ishanmishra128)

## 🙏 Acknowledgments

- AWS for providing robust cloud services
- Open-source community for tools and libraries
