from typing import TypedDict, List, Optional

class Concept(TypedDict):
    concept_name: str
    category: str
    explanation: str
    important_keywords: List[str]
    related_concepts: List[str]
    importance_score: int
    source_note: str

class AgentState(TypedDict):
    notes: List[dict]
    new_notes: List[dict]
    raw_concepts: List[dict]
    concepts: List[Concept]
    new_concepts: List[Concept]
    raw_links: List[dict]
    links: List[dict]
    quality_score: float
    retry_count: int
    dir: str
    retry_reason: Optional[List[str]]
    cache_path: str
    file_mtimes: dict
