"""Nine-dimensional propagation-intent taxonomy used by IGTF."""

INTENT_NAMES = [
    "public-oriented",
    "emotion-driven",
    "individual-focused",
    "popularize",
    "clout-seeking",
    "conflict-creation",
    "smearing",
    "bias-injection",
    "connection-seeking",
]

INTENT_DESCRIPTIONS = {
    "public-oriented": "provides public information or serves public interest",
    "emotion-driven": "provokes emotions or strong affective responses",
    "individual-focused": "focuses on personal stories or specific individuals",
    "popularize": "uses accessible wording to pursue broad dissemination",
    "clout-seeking": "seeks attention, influence, clicks, or visibility",
    "conflict-creation": "creates opposition or intensifies conflict",
    "smearing": "attacks or damages a specific target",
    "bias-injection": "guides readers toward a particular stance",
    "connection-seeking": "connects events, actors, or issues to build association",
}


def build_nine_dim_prompt(text: str, max_chars: int = 500) -> str:
    """Return the offline annotation prompt used to create 9-d intent vectors."""
    clipped = text[:max_chars]
    lines = "\n".join(
        f"{idx + 1}. **{name}**: {INTENT_DESCRIPTIONS[name]}"
        for idx, name in enumerate(INTENT_NAMES)
    )
    return f"""You are an expert in analyzing news articles and social media posts.
Analyze the following text and identify the author's propagation intents.

Text:
{clipped}

---

Task: Evaluate the intent strength for these 9 dimensions using decimals between 0.0 and 1.0:

{lines}

Scoring guidelines:
- Strongly exhibited intent: 0.7-0.9
- Partially exhibited intent: 0.4-0.6
- Barely exhibited intent: 0.1-0.3
- Completely absent intent: 0.0

Return JSON only:
{{
  "intents": {{
    "public-oriented": <specific value>,
    "emotion-driven": <specific value>,
    "individual-focused": <specific value>,
    "popularize": <specific value>,
    "clout-seeking": <specific value>,
    "conflict-creation": <specific value>,
    "smearing": <specific value>,
    "bias-injection": <specific value>,
    "connection-seeking": <specific value>
  }},
  "reasoning": "brief evidence-grounded explanation",
  "key_features": ["feature1", "feature2", "feature3"]
}}
"""
