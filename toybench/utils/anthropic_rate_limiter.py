import time
import threading
import random
from typing import Optional, Dict


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        dt = max(0.0, now - self.last_refill)
        if dt > 0:
            self.tokens = min(self.capacity, self.tokens + dt * self.refill_per_sec)
            self.last_refill = now

    def available(self) -> float:
        self._refill()
        return self.tokens

    def reserve(self, amount: float):
        self._refill()
        self.tokens -= float(amount)
        if self.tokens < -self.capacity:
            # Safety: never let it drift too negative
            self.tokens = -self.capacity

    def time_to_avail(self, amount: float) -> float:
        self._refill()
        deficit = max(0.0, amount - self.tokens)
        if deficit <= 0:
            return 0.0
        return deficit / max(1e-9, self.refill_per_sec)


class AnthropicAdaptiveLimiter:
    """
    Adaptive per-model-family limiter for Anthropic, driven by API usage and tier caps.

    We maintain three token buckets:
      - requests/minute (RPM)
      - input tokens/minute (ITPM)
      - output tokens/minute (OTPM)

    Estimates for the next request use EWMA of observed usage, with OTPM pre-reserving
    based on max_tokens when provided (closer to Anthropic's pre-allocation behavior).
    """

    def __init__(self, rpm: int, itpm: int, otpm: int, target_utilization: float = 0.85):
        self.lock = threading.Lock()
        self.target_utilization = float(max(0.1, min(1.0, target_utilization)))

        # Convert per-minute limits to per-second refill rates and capacities
        self.req_bucket = TokenBucket(capacity=rpm * self.target_utilization,
                                      refill_per_sec=(rpm / 60.0) * self.target_utilization)
        self.in_bucket = TokenBucket(capacity=itpm * self.target_utilization,
                                     refill_per_sec=(itpm / 60.0) * self.target_utilization)
        self.out_bucket = TokenBucket(capacity=otpm * self.target_utilization,
                                      refill_per_sec=(otpm / 60.0) * self.target_utilization)

        # Rolling EWMAs for estimates (exclude cache reads for Sonnet 4.x standard context)
        self.ewma_in: Optional[float] = None
        self.ewma_out: Optional[float] = None
        self.alpha = 0.2

        # Cold start ramp (requests per second soft cap)
        self.cold_start = True
        self.start_ts = time.monotonic()

    def _update_ewma(self, attr: str, value: Optional[float]):
        if value is None:
            return
        cur = getattr(self, attr)
        if cur is None:
            setattr(self, attr, float(value))
        else:
            setattr(self, attr, (1.0 - self.alpha) * float(cur) + self.alpha * float(value))

    def estimate_next(self, max_tokens: Optional[int] = None) -> Dict[str, float]:
        # Conservative defaults for cold start
        est_in = self.ewma_in if self.ewma_in is not None else 1500.0
        # For output, Anthropic estimates via max_tokens; use it if provided
        if max_tokens is not None:
            est_out = float(max(1, max_tokens))
        else:
            est_out = self.ewma_out if self.ewma_out is not None else 800.0
        return {"requests": 1.0, "input": est_in, "output": est_out}

    def acquire(self, est_in: float, est_out: float, max_wait_sec: float = 120.0):
        with self.lock:
            # Cold start soft RPS ramp: ~1 rps for first few seconds, then ramp
            if self.cold_start:
                elapsed = time.monotonic() - self.start_ts
                # Allow up to 1 rps + 1 rps per second of stable run, capped at 10 rps soft
                soft_rps = min(10.0, 1.0 + max(0.0, elapsed))
            else:
                soft_rps = float('inf')

            # Compute waits for each bucket
            waits = [
                self.req_bucket.time_to_avail(1.0),
                self.in_bucket.time_to_avail(max(0.0, est_in)),
                self.out_bucket.time_to_avail(max(0.0, est_out)),
            ]
            wait_time = max(waits)

            # Also enforce soft RPS (if still in cold start)
            if soft_rps != float('inf') and soft_rps > 0:
                # Determine minimum interval between requests
                min_interval = 1.0 / soft_rps
                # Check how many requests worth of tokens are available in req_bucket
                req_deficit = max(0.0, 1.0 - self.req_bucket.available())
                # Convert deficit to time in a rough way: deficit/refill_rate
                if req_deficit > 0 and self.req_bucket.refill_per_sec > 0:
                    wait_time = max(wait_time, req_deficit / self.req_bucket.refill_per_sec, min_interval)
                else:
                    wait_time = max(wait_time, min_interval)

            if wait_time > 0:
                jitter = random.uniform(0.0, 0.25)
                sleep_time = min(max_wait_sec, wait_time + jitter)
                time.sleep(sleep_time)

            # Reserve from buckets (optimistic; adjusted post-response)
            self.req_bucket.reserve(1.0)
            self.in_bucket.reserve(max(0.0, est_in))
            self.out_bucket.reserve(max(0.0, est_out))

    def update_after(self,
                     input_tokens: Optional[int],
                     cache_creation_input_tokens: Optional[int],
                     cache_read_input_tokens: Optional[int],
                     output_tokens: Optional[int],
                     counts_cache_reads_toward_itpm: bool = False,
                     max_tokens_used_for_estimate: Optional[int] = None):
        # Determine what actually counted toward ITPM for standard Sonnet caps
        counted_in = 0
        for v in (input_tokens, cache_creation_input_tokens):
            if isinstance(v, int):
                counted_in += v
        if counts_cache_reads_toward_itpm and isinstance(cache_read_input_tokens, int):
            counted_in += cache_read_input_tokens

        counted_out = int(output_tokens) if isinstance(output_tokens, int) else None

        # Update EWMAs (exclude cache reads by default)
        self._update_ewma('ewma_in', counted_in if counted_in > 0 else None)
        self._update_ewma('ewma_out', counted_out if (counted_out is not None and counted_out > 0) else None)

        # Adjust buckets: we reserved est based on estimate; return surplus if over-reserved
        with self.lock:
            if counted_in >= 0:
                surplus_in = -min(0.0, -counted_in)  # no-op, kept for readability
            # For output, if we pre-reserved by max_tokens, return the unused portion
            if max_tokens_used_for_estimate is not None and isinstance(counted_out, int):
                surplus_out = max(0.0, float(max_tokens_used_for_estimate - counted_out))
                if surplus_out > 0:
                    # return unused output tokens to bucket
                    self.out_bucket.reserve(-surplus_out)

            # After a few successful updates, disable cold start
            if self.cold_start and self.ewma_in is not None and self.ewma_out is not None:
                self.cold_start = False

    def on_rate_limit(self, retry_after_seconds: Optional[float] = None):
        # Honor server hint; add jitter
        base = retry_after_seconds if (isinstance(retry_after_seconds, (int, float)) and retry_after_seconds >= 0) else 2.0
        time.sleep(base + random.uniform(0.1, 0.5))


# Singleton registry per model family
_family_limiters: Dict[str, AnthropicAdaptiveLimiter] = {}
_lock = threading.Lock()


def get_sonnet4x_family_limiter() -> AnthropicAdaptiveLimiter:
    global _family_limiters
    key = "claude-sonnet-4.x"
    with _lock:
        lim = _family_limiters.get(key)
        if lim is None:
            # Tier 4 standard caps for Sonnet 4.x (<=200k context)
            lim = AnthropicAdaptiveLimiter(rpm=4000, itpm=2_000_000, otpm=400_000, target_utilization=0.85)
            _family_limiters[key] = lim
        return lim
