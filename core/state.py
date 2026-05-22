from typing import TypedDict, List, Optional

from pydantic import BaseModel, Field

class ConceptModel(BaseModel):
    concept_name: str = Field(description="The name of the concept.")
    category: str = Field(description="The broad category this concept belongs to.")
    explanation: str = Field(description="A concise explanation of the concept based on the text.")
    important_keywords: List[str] = Field(description="Important keywords associated with the concept.")
    related_concepts: List[str] = Field(description="Other related concepts mentioned in the text.")
    importance_score: int = Field(description="An importance score from 1 to 10.")
    source_note: Optional[str] = Field(default=None, description="The name of the source file.")

class ConceptExtractionOutput(BaseModel):
    concepts: List[ConceptModel]
    tags: List[str] = Field(default_factory=list, description="3-5 global Obsidian-style tags representing the entire note (e.g., '#machine-learning', '#python').")

class RelationshipModel(BaseModel):
    source: str = Field(description="The name of the source concept.")
    target: str = Field(description="The name of the target concept from a DIFFERENT note.")
    relationship: str = Field(description="The type of relationship (e.g., uses, depends_on, extends, similar_to, part_of, implemented_with, alternative_to, improves, causes).")
    evidence: str = Field(description="Brief explanation of why these concepts are related.")
    from_note: str = Field(description="The source_note of the source concept.")
    to_note: str = Field(description="The source_note of the target concept.")

class RelationshipExtractionOutput(BaseModel):
    links: List[RelationshipModel]

class AgentState(TypedDict):
    notes: List[dict]
    new_notes: List[dict]
    raw_concepts: List[dict]
    concepts: List[dict]
    new_concepts: List[dict]
    raw_links: List[dict]
    links: List[dict]
    quality_score: float
    retry_count: int
    dir: str
    retry_reason: Optional[List[str]]
    cache_path: str
    file_hashes: dict
    raw_tags_by_note: dict
    tags_by_note: dict
