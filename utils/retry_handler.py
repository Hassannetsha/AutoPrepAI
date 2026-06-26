import re
import time


class GroqRetryHandler:
    """
    Retry loop with automatic API key rotation for Groq operations.
    Unifies the while-retry pattern duplicated across the codebase.
    """

    RATE_LIMIT_KEYWORDS = ("rate", "quota", "limit", "429")

    def __init__(self, key_manager, log_fn=None):
        self.key_manager = key_manager
        self.log_fn = log_fn or (lambda msg: None)

    def execute(self, task, after_rotate=None, task_name="operation"):
        """
        Execute *task* with retry and automatic key rotation on rate limits.

        Args:
            task: Zero-argument callable that performs the API operation.
            after_rotate: Optional callable(new_key) invoked after rotating
                          to a new key (e.g. to reconfigure DSPy LM).
            task_name: Label used in log / error messages.

        Returns:
            The return value of *task* on success.

        Raises:
            RuntimeError when all API keys are exhausted or the task keeps
            failing with non-rate-limit errors.
        """
        max_retries = self.key_manager.get_total_keys_count()
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                return task()
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                if any(x in error_msg for x in self.RATE_LIMIT_KEYWORDS):
                    self.key_manager.mark_key_failed()
                    available = self.key_manager.get_available_keys_count()
                    self.log_fn(
                        f"Rate limit hit for '{task_name}'. "
                        f"Keys available: {available}"
                    )

                    if available <= 1:
                        retry_seconds = "unknown"
                        match = re.search(
                            r"please try again in ([^s]+s)", error_msg
                        )
                        if match:
                            retry_seconds = match.group(1)
                        raise RuntimeError(
                            f"All {self.key_manager.get_total_keys_count()} "
                            f"API keys exhausted for '{task_name}'. "
                            f"Retry in {retry_seconds}."
                        )

                    try:
                        new_key = self.key_manager.rotate_key()
                        retry_count += 1
                        if after_rotate:
                            after_rotate(new_key)
                        time.sleep(1)
                        continue
                    except RuntimeError:
                        raise RuntimeError(
                            f"All {self.key_manager.get_total_keys_count()} "
                            f"API keys exhausted for '{task_name}'."
                        )

                raise

        raise RuntimeError(
            f"Operation '{task_name}' failed after "
            f"{max_retries} retries: {last_error}"
        )
