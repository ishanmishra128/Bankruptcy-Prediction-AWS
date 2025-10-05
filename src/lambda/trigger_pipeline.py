"""
AWS Lambda Function for Pipeline Orchestration

Triggers and coordinates the bankruptcy prediction pipeline.
"""

import json
import boto3
import os
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
glue_client = boto3.client('glue')
sagemaker_client = boto3.client('sagemaker')
sns_client = boto3.client('sns')


def lambda_handler(event, context):
    """
    Main Lambda handler for pipeline orchestration
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        Response dictionary
    """
    logger.info("Starting bankruptcy prediction pipeline...")
    logger.info(f"Event: {json.dumps(event)}")
    
    try:
        # Get configuration from environment variables
        glue_job_name = os.environ.get('GLUE_JOB_NAME', 'bankruptcy-etl-job')
        s3_input_bucket = os.environ.get('S3_INPUT_BUCKET', 'bankruptcy-data-lake')
        s3_output_bucket = os.environ.get('S3_OUTPUT_BUCKET', 'bankruptcy-data-lake')
        sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
        
        # Determine trigger type
        trigger_type = event.get('trigger_type', 'manual')
        
        if trigger_type == 's3_upload':
            # Triggered by S3 upload
            response = handle_s3_trigger(event, glue_job_name)
        
        elif trigger_type == 'scheduled':
            # Triggered by CloudWatch Events (scheduled)
            response = handle_scheduled_trigger(glue_job_name)
        
        else:
            # Manual trigger
            response = handle_manual_trigger(event, glue_job_name)
        
        # Send success notification
        if sns_topic_arn:
            send_notification(
                sns_topic_arn,
                "Pipeline Started Successfully",
                f"Bankruptcy prediction pipeline initiated. Job Run ID: {response.get('JobRunId', 'N/A')}"
            )
        
        logger.info("Pipeline orchestration completed successfully")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Pipeline started successfully',
                'response': response
            })
        }
    
    except Exception as e:
        logger.error(f"Pipeline orchestration failed: {str(e)}", exc_info=True)
        
        # Send error notification
        if sns_topic_arn:
            send_notification(
                sns_topic_arn,
                "Pipeline Failed",
                f"Bankruptcy prediction pipeline failed: {str(e)}"
            )
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Pipeline failed',
                'error': str(e)
            })
        }


def handle_s3_trigger(event, glue_job_name):
    """
    Handle S3 upload trigger
    
    Args:
        event: S3 event
        glue_job_name: Name of Glue job to trigger
        
    Returns:
        Glue job response
    """
    logger.info("Processing S3 trigger...")
    
    # Extract S3 information from event
    s3_event = event['Records'][0]['s3']
    bucket = s3_event['bucket']['name']
    key = s3_event['object']['key']
    
    logger.info(f"New file uploaded: s3://{bucket}/{key}")
    
    # Start Glue ETL job
    response = start_glue_job(
        glue_job_name,
        input_path=f"s3://{bucket}/{key}"
    )
    
    return response


def handle_scheduled_trigger(glue_job_name):
    """
    Handle scheduled trigger from CloudWatch Events
    
    Args:
        glue_job_name: Name of Glue job to trigger
        
    Returns:
        Glue job response
    """
    logger.info("Processing scheduled trigger...")
    
    # Check for new data in S3
    # Start Glue ETL job with default paths
    response = start_glue_job(glue_job_name)
    
    return response


def handle_manual_trigger(event, glue_job_name):
    """
    Handle manual trigger
    
    Args:
        event: Event with manual trigger parameters
        glue_job_name: Name of Glue job to trigger
        
    Returns:
        Glue job response
    """
    logger.info("Processing manual trigger...")
    
    # Extract parameters from event
    input_path = event.get('input_path')
    output_path = event.get('output_path')
    
    response = start_glue_job(
        glue_job_name,
        input_path=input_path,
        output_path=output_path
    )
    
    return response


def start_glue_job(job_name, input_path=None, output_path=None):
    """
    Start AWS Glue ETL job
    
    Args:
        job_name: Name of Glue job
        input_path: Input S3 path (optional)
        output_path: Output S3 path (optional)
        
    Returns:
        Glue job run response
    """
    logger.info(f"Starting Glue job: {job_name}")
    
    # Prepare job arguments
    arguments = {}
    
    if input_path:
        arguments['--S3_INPUT_PATH'] = input_path
    
    if output_path:
        arguments['--S3_OUTPUT_PATH'] = output_path
    
    # Start job run
    response = glue_client.start_job_run(
        JobName=job_name,
        Arguments=arguments
    )
    
    job_run_id = response['JobRunId']
    logger.info(f"Glue job started. Run ID: {job_run_id}")
    
    return response


def trigger_sagemaker_training(training_job_name, training_data_path):
    """
    Trigger SageMaker training job
    
    Args:
        training_job_name: Name for training job
        training_data_path: S3 path to training data
        
    Returns:
        SageMaker training response
    """
    logger.info(f"Triggering SageMaker training: {training_job_name}")
    
    # This would typically include more configuration
    # Simplified for demonstration
    
    response = {
        'message': 'SageMaker training triggered',
        'training_job_name': training_job_name,
        'status': 'InProgress'
    }
    
    logger.info(f"SageMaker training job started: {training_job_name}")
    
    return response


def send_notification(topic_arn, subject, message):
    """
    Send SNS notification
    
    Args:
        topic_arn: SNS topic ARN
        subject: Email subject
        message: Message body
    """
    try:
        sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        logger.info(f"Notification sent: {subject}")
    
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")


def check_pipeline_status(job_run_id):
    """
    Check status of pipeline job
    
    Args:
        job_run_id: Glue job run ID
        
    Returns:
        Job status
    """
    response = glue_client.get_job_run(
        JobName=os.environ.get('GLUE_JOB_NAME'),
        RunId=job_run_id
    )
    
    status = response['JobRun']['JobRunState']
    logger.info(f"Job {job_run_id} status: {status}")
    
    return status
