from dataclasses import dataclass
from typing import Optional


@dataclass
class Finding:
    engine: str
    pattern_type: str
    confidence: float
    evidence: str
    element_ref: Optional[str] = None
