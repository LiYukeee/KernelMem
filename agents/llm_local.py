"""
llm_vllm.py – vLLM client (OpenAI-compatible)
=============================================
* Connects to local vLLM server (multi-GPU backend)
* Replaces HF Transformers inference
* Compatible with query_server interface
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Any
from openai import OpenAI
import time

# -------------------------------
def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: Optional[int] = None,  # None means unlimited retries
    initial_delay: float = 1.0,
    max_delay: float = 300.0,  # 5 minutes
    backoff_factor: float = 2.0,
) -> Any:
    """
    带指数退避的重试函数
    
    Args:
        func: 要重试的函数（无参数）
        max_retries: 最大重试次数，None 表示无限重试
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒），默认 300 秒（5 分钟）
        backoff_factor: 退避因子
    
    Returns:
        函数执行结果
    
    Raises:
        最后一次尝试的异常（仅在手动中断时）
    """
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        retryable_exceptions = (APIConnectionError, APITimeoutError, RateLimitError)
    except ImportError:
        retryable_exceptions = (Exception,)
    
    delay = initial_delay
    attempt = 0
    
    while True:
        try:
            return func()
        except retryable_exceptions as e:
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                print(f"❌ Failed after {max_retries} attempts. Last error: {type(e).__name__}: {e}")
                raise
            
            error_name = type(e).__name__
            if max_retries is not None:
                print(f"⚠️  {error_name} occurred (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s...")
            else:
                print(f"⚠️  {error_name} occurred (attempt {attempt}, unlimited retries). Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)


# -------------------------------
@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40  # vLLM doesn't use top_k but kept for compatibility
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    stream: bool = False

# -------------------------------
class LLM:
    """vLLM OpenAI-compatible client."""

    def __init__(self, model: str, server_url: str = "http://localhost:8000/v1"):
        self.model = model
        self.client = OpenAI(
            base_url=server_url,
            api_key="EMPTY"  # required dummy key for vLLM
        )

    def chat(self, system: str, user: str, cfg: GenerationConfig | None = None, ) -> str:
        cfg = cfg or GenerationConfig()
        # For non-chat models like MPT, fallback to generate()
        if "mpt" in self.model.lower() or "deepseek-coder" in self.model.lower():
            prompt = f"{system.strip()}\n{user.strip()}"
            return self.generate(prompt, cfg)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        response = retry_with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                seed=cfg.seed,
            )
        )
        return response.choices[0].message.content


    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        response = retry_with_backoff(
            lambda: self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                seed=cfg.seed,
            )
        )
        return response.choices[0].text

# -------------------------------
from functools import lru_cache

@lru_cache(maxsize=2)
def get_llm(model_id: str, server_url: str = "http://localhost:8000/v1") -> LLM:
    return LLM(model=model_id, server_url=server_url)
