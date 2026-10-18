"""
Handling of Gemini API rate limits ("429 RESOURCE_EXHAUSTED").

Google returns a 429 when a quota is exceeded. Two kinds matter here:
  * per-minute limits (requests or tokens per minute) - waiting a little helps;
  * per-day limits - nothing helps until the quota resets the next day.

`call_with_retry()` waits and retries per-minute limits, using the delay Google
suggests. When retrying cannot help, it raises `RateLimitError` with a clear,
user-facing message that the web pages and the CLI show as-is.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

from app.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_WAIT_SECONDS = 15.0
MAX_WAIT_SECONDS = 60.0  # never sleep longer than this in one go


class RateLimitError(RuntimeError):
    """The Gemini API quota is exhausted and retrying did not help."""

    def __init__(self, retry_after: float | None, daily: bool):
        self.retry_after = retry_after
        self.daily = daily
        super().__init__(self.user_message())

    def user_message(self) -> str:
        if self.daily:
            return ("The daily Gemini API quota for this key is used up. It resets at midnight "
                    "Pacific Time. Try again tomorrow, or use a key with a higher quota.")
        wait = f"about {round(self.retry_after)} seconds" if self.retry_after else "a minute"
        return f"The Gemini API rate limit was reached. Please wait {wait} and try again."


# ---------- recognising a 429 ----------

def _error_chain(error: BaseException):
    """The error itself plus every error it was raised from."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def is_rate_limit_error(error: BaseException) -> bool:
    for e in _error_chain(error):
        if getattr(e, "code", None) == 429 or getattr(e, "status", None) == "RESOURCE_EXHAUSTED":
            return True
        if type(e).__name__ == "GoogleRateLimitError":
            return True
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or " 429" in text


def is_daily_limit(error: BaseException) -> bool:
    """Per-day quotas carry ids such as 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'."""
    return any("PerDay" in str(e) for e in _error_chain(error))


def suggested_wait(error: BaseException) -> float | None:
    """Seconds Google asks us to wait (RetryInfo.retryDelay or 'Please retry in 12.3s')."""
    for e in _error_chain(error):
        match = (re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e))
                 or re.search(r"retry in (\d+(?:\.\d+)?)\s*s", str(e), re.IGNORECASE))
        if match:
            return float(match.group(1))
    return None


# ---------- retrying ----------

def call_with_retry(func: Callable[[], T], what: str = "Gemini call") -> T:
    """
    Run `func()`. On a per-minute 429, wait (Google's suggested delay, or an
    exponential back-off) and try again, up to LLM_MAX_RETRIES times.
    Raises RateLimitError when the quota is daily or retries run out.
    """
    attempts = max(1, settings.llm_max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as error:
            if not is_rate_limit_error(error):
                raise
            wait = suggested_wait(error)
            if is_daily_limit(error):
                raise RateLimitError(wait, daily=True) from error
            if attempt == attempts:
                raise RateLimitError(wait, daily=False) from error
            sleep = min(wait or DEFAULT_WAIT_SECONDS * attempt, MAX_WAIT_SECONDS)
            log.warning("%s hit the rate limit (attempt %d/%d), waiting %.0fs",
                        what, attempt, attempts, sleep)
            time.sleep(sleep)
    raise AssertionError("unreachable")
