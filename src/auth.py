import os
from dotenv import load_dotenv
import httpx
from typing import Optional

try:
    from langchain_openai import AzureChatOpenAI
except Exception:
    AzureChatOpenAI = None 


AUTH_API_URL: Optional[str] = None
AUTH_API_SCOPE: Optional[str] = None
AUTH_CLIENT_ID: Optional[str] = None
AUTH_CLIENT_SECRET: Optional[str] = None
AZURE_OPENAI_ENDPOINT: Optional[str] = None
OPENAI_API_VERSION: Optional[str] = None
MODEL_DEPLOYMENT_NAME: Optional[str] = None
PROJECT_ID: Optional[str] = None

LLM_CLIENT = None


def init_auth(env_path: str = "./Data/claimaudit.env") -> None:
    """Load environment variables from `env_path` into module globals."""
    global AUTH_API_URL, AUTH_API_SCOPE, AUTH_CLIENT_ID, AUTH_CLIENT_SECRET
    global AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, MODEL_DEPLOYMENT_NAME, PROJECT_ID

    load_dotenv(env_path)

    AUTH_API_URL = os.environ.get("AUTH_API_URL")
    AUTH_API_SCOPE = os.environ.get("AUTH_API_SCOPE")
    AUTH_CLIENT_ID = os.environ.get("AUTH_CLIENT_ID")
    AUTH_CLIENT_SECRET = os.environ.get("AUTH_CLIENT_SECRET")

    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
    MODEL_DEPLOYMENT_NAME = os.environ.get("MODEL_DEPLOYMENT_NAME")
    PROJECT_ID = os.environ.get("PROJECT_ID")


def get_access_token() -> Optional[str]:
    """Request an access token from the configured auth endpoint using client credentials."""
    if not AUTH_API_URL or not AUTH_API_SCOPE or not AUTH_CLIENT_ID or not AUTH_CLIENT_SECRET:
        raise RuntimeError("Auth environment variables not configured. Call init_auth() first.")

    with httpx.Client() as client:
        body = {
            "grant_type": "client_credentials",
            "scope": AUTH_API_SCOPE,
            "client_id": AUTH_CLIENT_ID,
            "client_secret": AUTH_CLIENT_SECRET,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = client.post(AUTH_API_URL, headers=headers, data=body, timeout=60)
        resp.raise_for_status()
        return resp.json().get("access_token")


def make_llm_client() -> object:
    """Create and return an AzureChatOpenAI client configured from envs."""
    global LLM_CLIENT
    if AzureChatOpenAI is None:
        raise RuntimeError("AzureChatOpenAI library is not installed or failed to import.")

    token = get_access_token()
    if token is None:
        raise RuntimeError("Failed to obtain access token for Azure AD.")

    LLM_CLIENT = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=OPENAI_API_VERSION,
        azure_deployment=MODEL_DEPLOYMENT_NAME,
        temperature=0,
        azure_ad_token=token,
        default_headers={"projectId": PROJECT_ID},
    )
    return LLM_CLIENT
