"""IGTF: game-inspired intent calibration for fake-news detection."""

from .taxonomy import INTENT_NAMES

__all__ = [
    "DifferentiableIntentGameRefiner",
    "IGTFClassifier",
    "IGTFConfig",
    "INTENT_NAMES",
    "IntentGameConfig",
]


def __getattr__(name):
    if name in {"DifferentiableIntentGameRefiner", "IntentGameConfig"}:
        from .intent_game import DifferentiableIntentGameRefiner, IntentGameConfig

        return {
            "DifferentiableIntentGameRefiner": DifferentiableIntentGameRefiner,
            "IntentGameConfig": IntentGameConfig,
        }[name]
    if name in {"IGTFClassifier", "IGTFConfig"}:
        from .model import IGTFClassifier, IGTFConfig

        return {
            "IGTFClassifier": IGTFClassifier,
            "IGTFConfig": IGTFConfig,
        }[name]
    raise AttributeError(name)
