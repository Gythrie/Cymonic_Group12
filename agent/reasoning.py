"""
reasoning.py — Reasoning engine for Restaurant Reservation Concierge.
Conforms strictly to CONTRACT.md.

Primary path: Calls Gemini LLM via llm_client.call_llm(context).
Fallback path: Pure Python rule-based scoring via fallback_rules.fallback_score(context).
Validation: Enforces exact output contract via validator.validate(result).
"""

import logging
from typing import Dict, Any

from agent.llm_client import call_llm, LLMCallError
from agent.fallback_rules import fallback_score
from agent.validator import validate

logger = logging.getLogger(__name__)


def _personalize_offer(offer: str, context: Dict[str, Any]) -> str:
    """Replaces [Name] or generic greetings with actual guest name if available."""
    name = context.get("customer_name")
    if not name:
        return offer

    if "[Name]" in offer:
        return offer.replace("[Name]", name)
    if "Hi there!" in offer:
        return offer.replace("Hi there!", f"Hi {name}!", 1)
    return offer


def run_concierge(context: dict) -> dict:
    """
    Executes concierge reasoning for the given dining context.

    Parameters
    ----------
    context : dict
        Context dictionary adhering to CONTRACT.md:
        - occupancy_pct: int or float
        - cancellations_count: int
        - is_peak: bool
        - time_slot: str
        - customer_tier: str ("Gold", "Silver", "Regular")

    Returns
    -------
    dict
        {
            "decision": "notify" | "low_incentive" | "high_incentive",
            "reasoning": str,
            "offer": str
        }

    Error Handling:
        Never raises an unhandled exception to caller. If the LLM call fails,
        times out, or outputs malformed data, this function seamlessly
        reverts to rule-based fallback scoring.
    """
    # 1. Attempt LLM reasoning path
    try:
        raw_llm_result = call_llm(context)
        validated = validate(raw_llm_result)
        validated["offer"] = _personalize_offer(validated["offer"], context)
        return validated
    except (LLMCallError, Exception) as exc:
        logger.info(f"LLM pathway unavailable or errored ({exc}). Engaging fallback rule scoring.")

    # 2. Safe deterministic fallback path
    try:
        raw_fallback_result = fallback_score(context)
        validated = validate(raw_fallback_result)
        validated["offer"] = _personalize_offer(validated["offer"], context)
        return validated
    except Exception as exc:
        logger.error(f"Fallback engine error ({exc}). Returning baseline safe notification.")
        return {
            "decision": "notify",
            "reasoning": (
                "Automated safe fallback: Routine notification active. "
                "Table availability communicated to guest without promotional incentive."
            ),
            "offer": (
                f"Hi {context.get('customer_name', 'there')}, we have a table ready for you "
                f"at {context.get('time_slot', 'your requested time')}. "
                "We look forward to welcoming you to Le Bistro."
            )
        }
