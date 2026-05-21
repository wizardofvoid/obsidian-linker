from langchain_core.prompts import ChatPromptTemplate

prompt_concept_extraction = ChatPromptTemplate.from_messages([
    ("system", "You are an expert knowledge extraction system. Analyze the provided text and extract concepts as requested. Return ONLY the raw JSON array of concept objects. Do NOT write code or explanations outside of JSON."),
    ("user", """Analyze the following text and extract all important concepts, topics, technologies, methods, and entities.
For each extracted concept provide:
- concept_name
- category
- explanation
- important_keywords
- related_concepts
- importance_score (1-10)

Rules:
- Extract only the top 5-8 most significant concepts.
- Only extract concepts explicitly mentioned or strongly implied.
- Avoid duplicates.
- Keep explanations concise.

{format_instructions}

TEXT: {text}""")
])

prompt_concept_verification = ChatPromptTemplate.from_messages([
    ("system", "You are a verification system. Clean and verify the extracted concepts. Do NOT write Python code or programs. Output ONLY the raw JSON array of concepts directly."),
    ("user", """Analyze the provided concepts extracted from the text below. Clean and verify them by:
1. Removing any incorrect concepts that are not supported by the text.
2. Removing duplicate or highly redundant concepts.
3. Merging concepts that refer to the exact same entity/topic.
4. Refining descriptions to be highly accurate and clear based on the text.

{format_instructions}

TEXT:
{text}
CONCEPTS:
{concepts}""")
])

prompt_relation = ChatPromptTemplate.from_messages([
    ("system", "You are a relationship extraction engine that finds connections BETWEEN different notes in a knowledge base. Return ONLY the raw JSON array of relationship objects. Do NOT write code."),
    ("user", """Below are two lists of concepts extracted from notes in a knowledge base.
Each concept has a 'source_note' field indicating which note it came from.

YOUR TASK: Find semantic relationships that involve AT LEAST ONE concept from the NEW CONCEPTS list.
You can link New ↔ New, or New ↔ Existing, but DO NOT link Existing ↔ Existing (we already know those).

Allowed relationship types:
- uses, depends_on, extends, similar_to, part_of
- implemented_with, alternative_to, improves, causes

For each relationship, include:
- "source": the concept name (from one note)
- "target": the concept name (from a DIFFERENT note)
- "relationship": one of the allowed types
- "evidence": brief explanation of why these concepts are related
- "from_note": the source_note of the source concept
- "to_note": the source_note of the target concept

Rules:
- ONLY extract relationships between concepts from DIFFERENT notes.
- AT LEAST ONE concept in the relationship MUST be from the NEW CONCEPTS list.
- Do NOT create relationships between concepts from the same note.
- Focus on meaningful semantic connections.

{format_instructions}

NEW CONCEPTS:
{new_concepts}

EXISTING CONCEPTS:
{existing_concepts}""")
])

prompt_relation_verify = ChatPromptTemplate.from_messages([
    ("system", "You are a link verification engine. Verify the proposed cross-note relationships. Do NOT write Python code. Output ONLY the raw JSON array of verified relationships directly."),
    ("user", """Below are proposed relationships between concepts from different notes.
Verify each relationship:
1. Remove any invalid or unsupported relationships.
2. Remove duplicate relationships.
3. Ensure the source and target concepts actually exist in the concepts list.
4. Ensure the relationship type is semantically correct.

{format_instructions}

CONCEPTS:
{concepts}

RELATIONSHIPS:
{relationships}""")
])
