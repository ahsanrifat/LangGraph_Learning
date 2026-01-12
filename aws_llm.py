from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
import os
load_dotenv()

llm = ChatBedrockConverse(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("aws_access_key_id"),
    aws_secret_access_key=os.getenv("aws_secret_access_key"),
    aws_session_token=os.getenv("aws_session_token")
    )