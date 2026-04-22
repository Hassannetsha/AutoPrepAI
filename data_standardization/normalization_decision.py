from dataclasses import dataclass


@dataclass
class NormalizationDecision:
    original: str
    normalized: str
    confidence: float
    raw_confidence: float
    accepted: bool
    fallback_reason: str
    validation_passed: bool
    validation_reason: str
    reason: str = ""

    def to_dict(self):
        return self.__dict__
