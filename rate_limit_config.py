"""
Rate Limiting Configuration

Easy configuration for different environments and use cases.
"""

import os
from rate_limiter import RateLimitConfig

# Environment-based configuration
def get_rate_limit_config():
    """Get rate limiting configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "production").lower()
    
    if env == "development":
        return RateLimitConfig(
            # Development - more lenient
            requests_per_minute=30,
            tokens_per_minute=100000,
            requests_per_hour=200,
            tokens_per_hour=500000,
            requests_per_day=1000,
            tokens_per_day=2000000,
            cost_per_day=50.0,
            max_burst_requests=10,
            burst_window_seconds=10
        )
    
    elif env == "staging":
        return RateLimitConfig(
            # Staging - moderate limits
            requests_per_minute=20,
            tokens_per_minute=75000,
            requests_per_hour=150,
            tokens_per_hour=300000,
            requests_per_day=800,
            tokens_per_day=1500000,
            cost_per_day=25.0,
            max_burst_requests=8,
            burst_window_seconds=10
        )
    
    else:  # production
        return RateLimitConfig(
            # Production - strict limits
            requests_per_minute=10,
            tokens_per_minute=50000,
            requests_per_hour=100,
            tokens_per_hour=200000,
            requests_per_day=1000,
            tokens_per_day=1000000,
            cost_per_day=10.0,
            max_burst_requests=5,
            burst_window_seconds=10
        )


# Custom configurations for different use cases
def get_free_tier_config():
    """Configuration for free tier users."""
    return RateLimitConfig(
        requests_per_minute=5,
        tokens_per_minute=20000,
        requests_per_hour=50,
        tokens_per_hour=100000,
        requests_per_day=500,
        tokens_per_day=500000,
        cost_per_day=5.0,
        max_burst_requests=3,
        burst_window_seconds=10
    )


def get_premium_tier_config():
    """Configuration for premium tier users."""
    return RateLimitConfig(
        requests_per_minute=20,
        tokens_per_minute=100000,
        requests_per_hour=200,
        tokens_per_hour=500000,
        requests_per_day=2000,
        tokens_per_day=2000000,
        cost_per_day=50.0,
        max_burst_requests=10,
        burst_window_seconds=10
    )


def get_enterprise_config():
    """Configuration for enterprise users."""
    return RateLimitConfig(
        requests_per_minute=50,
        tokens_per_minute=200000,
        requests_per_hour=500,
        tokens_per_hour=1000000,
        requests_per_day=5000,
        tokens_per_day=10000000,
        cost_per_day=200.0,
        max_burst_requests=20,
        burst_window_seconds=10
    )


# Rate limiting strategies
RATE_LIMIT_STRATEGIES = {
    "strict": {
        "requests_per_minute": 5,
        "tokens_per_minute": 20000,
        "cost_per_day": 5.0
    },
    "moderate": {
        "requests_per_minute": 10,
        "tokens_per_minute": 50000,
        "cost_per_day": 10.0
    },
    "lenient": {
        "requests_per_minute": 20,
        "tokens_per_minute": 100000,
        "cost_per_day": 25.0
    }
}


def get_strategy_config(strategy: str):
    """Get configuration for a specific strategy."""
    if strategy not in RATE_LIMIT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    base_config = get_rate_limit_config()
    strategy_params = RATE_LIMIT_STRATEGIES[strategy]
    
    # Update base config with strategy parameters
    for key, value in strategy_params.items():
        setattr(base_config, key, value)
    
    return base_config
