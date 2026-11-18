"""从环境变量中构建provider的config"""
from dataclasses import dataclass
from os import environ
from typing import Literal

DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    # 不支持 stream_options 的兼容服务可以显式关闭。
    include_usage: bool = True
    thinking: Literal["enabled", "disabled"] | None = None


def openai_compatible_config_from_env(
    *,
    api_key_var: str = "OPENAI_API_KEY",
    base_url_var: str = "OPENAI_BASE_URL"
) -> OpenAICompatibleConfig:
    api_key = environ.get(api_key_var)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_var}")

    base_url = environ.get(base_url_var, DEFAULT_OPENAI_COMPATIBLE_BASE_URL).rstrip("/")

    return OpenAICompatibleConfig(
        api_key=api_key, 
        base_url=base_url
    )
