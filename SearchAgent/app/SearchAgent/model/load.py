import os

from langchain_aws import ChatBedrock

# Uses global inference profile for Claude Sonnet 4.5
# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


def load_model() -> ChatBedrock:
    """Get Bedrock model client using IAM credentials."""
    # agentcore dev injects access keys but not the region from ~/.aws/config.
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    return ChatBedrock(model_id=MODEL_ID, region_name=region)
