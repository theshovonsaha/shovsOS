"""Compact LLM-facing capability cards.

These cards make platform abilities easy for the model to use without bloating
every prompt with full tool schemas. They describe when to use a capability,
which fields matter, and what structured output to expect.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestration.workflow_patterns import get_workflow_pattern
from orchestration.workflow_templates import get_workflow_template


@dataclass(frozen=True)
class CapabilityCard:
    id: str
    title: str
    triggers: tuple[str, ...]
    use: str
    required_inputs: tuple[str, ...]
    output_contract: tuple[str, ...]
    caution: tuple[str, ...]

    def render(self) -> str:
        return "\n".join([
            f"Capability: {self.title} ({self.id})",
            f"When to use: {'; '.join(self.triggers)}",
            f"Use: {self.use}",
            f"Inputs: {', '.join(self.required_inputs)}",
            "Output:",
            *[f"- {item}" for item in self.output_contract],
            "Cautions:",
            *[f"- {item}" for item in self.caution],
        ])


SHOPPING_ADVICE_CARD = CapabilityCard(
    id="shopping_advice",
    title="Local Store Shopping Advisor",
    triggers=(
        "user asks where/what to buy",
        "product comparison, price check, deal check, quality tradeoff",
        "stores like Costco, Canadian Tire, Shoppers, Metro, Dollarama, Walmart, Best Buy",
    ),
    use=(
        "Call shopping_advice once with the item, budget, location, region, priorities, and preferred stores. "
        "It performs divergence, verification, normalization, convergence, and returns answer_patch."
    ),
    required_inputs=(
        "query",
        "budget if stated",
        "location if stated or local stores matter",
        "stores if user names any",
        "priorities such as price, quality, size, warranty, return policy",
    ),
    output_contract=(
        "shopping_advice_result.candidates contains normalized store/item/price/rating/url records",
        "answer_patch.comparison_table is the source for final comparisons",
        "answer_patch.recommendation is the best verified lead",
        "verified_urls are the only URLs safe to cite",
    ),
    caution=(
        "Do not invent exact prices or availability outside extracted fields",
        "Do not claim local in-store stock unless fetched content explicitly says it",
        "If fields are missing, say not verified instead of guessing",
    ),
)


CAPABILITY_CARDS: dict[str, CapabilityCard] = {
    SHOPPING_ADVICE_CARD.id: SHOPPING_ADVICE_CARD,
}


def render_capability_cards(
    *,
    allowed_tools: list[str],
    workflow_template: str = "",
) -> str:
    ids: list[str] = []
    for tool in allowed_tools:
        if tool in CAPABILITY_CARDS and tool not in ids:
            ids.append(tool)

    template = get_workflow_template(workflow_template)
    pattern = get_workflow_pattern(template.workflow_pattern)
    rendered: list[str] = []
    for capability_id in ids:
        rendered.append(CAPABILITY_CARDS[capability_id].render())
    if pattern:
        rendered.append(pattern.render_for_prompt())
    return "\n\n".join(rendered)
