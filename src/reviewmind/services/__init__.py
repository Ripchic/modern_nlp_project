# This file intentionally left minimal — services are imported directly.

from reviewmind.services.comparison_service import (
    COMPARISON_PROMPT_TEMPLATE,
    MAX_PRODUCTS_FOR_COMPARISON,
    MIN_PRODUCTS_FOR_COMPARISON,
    ComparisonResult,
    ProductRAGResult,
    compare_products,
    detect_comparison,
)
from reviewmind.services.limit_service import (
    FREE_DAILY_LIMIT,
    LIMIT_REACHED_MSG,
    PREMIUM_SUBSCRIPTION,
    LimitCheckResult,
    LimitService,
)
from reviewmind.services.payment_service import (
    SUBSCRIPTION_ACTIVATED_MSG,
    SUBSCRIPTION_DAYS,
    SUBSCRIPTION_ERROR_MSG,
    SUBSCRIPTION_PRICE_STARS,
    ActivationResult,
    PaymentService,
)

__all__ = [
    "COMPARISON_PROMPT_TEMPLATE",
    "ComparisonResult",
    "FREE_DAILY_LIMIT",
    "LIMIT_REACHED_MSG",
    "MAX_PRODUCTS_FOR_COMPARISON",
    "MIN_PRODUCTS_FOR_COMPARISON",
    "PREMIUM_SUBSCRIPTION",
    "ActivationResult",
    "LimitCheckResult",
    "LimitService",
    "PaymentService",
    "ProductRAGResult",
    "SUBSCRIPTION_ACTIVATED_MSG",
    "SUBSCRIPTION_DAYS",
    "SUBSCRIPTION_ERROR_MSG",
    "SUBSCRIPTION_PRICE_STARS",
    "compare_products",
    "detect_comparison",
]
