import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load local .env if it exists
load_dotenv(PROJECT_ROOT / ".env")


def get_config(name: str, default=None):
    """
    Get config value from:
    1. Local environment / .env
    2. Streamlit secrets
    3. Default value
    """

    # Local .env / environment variable
    value = os.getenv(name)
    if value:
        return value

    # Streamlit Cloud secrets
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except StreamlitSecretNotFoundError:
        pass

    return default


def require_config(name: str):
    """
    Same as get_config(), but raises an error if missing.
    """
    value = get_config(name)

    if not value:
        raise ValueError(f"{name} was not found")

    return value