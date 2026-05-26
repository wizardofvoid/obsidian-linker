from langchain_core.prompts import ChatPromptTemplate

prompt_concept_extraction = ChatPromptTemplate.from_messages([
    ("system", "You are an expert knowledge extraction system. Analyze the provided text and extract concepts as well as global tags. Focus strictly on reasoning and accuracy."),
    ("user", """Analyze the following text and extract all important concepts, topics, technologies, methods, and entities.
Also generate 3-5 global Obsidian-style tags (prefixed with '#') that categorize the overall note.

Rules:
- Extract only the top 5-8 most significant concepts.
- Only extract concepts explicitly mentioned or strongly implied.
- Avoid duplicates.
- Keep explanations concise.
- Generate highly relevant tags for the 'tags' array (e.g. '#technology', '#data-science'). Do not use namespaces, just simple hashtags.
- CRITICAL INSTRUCTION: You must properly nest your output. Generate a single object containing a 'concepts' key (array of objects) and a 'tags' key (array of strings). Do not break the array syntax.

TEXT: {text}""")
])

prompt_concept_verification = ChatPromptTemplate.from_messages([
    ("system", "You are a verification system. Clean and verify the extracted concepts and tags. Focus purely on semantic accuracy."),
    ("user", """Analyze the provided concepts and tags extracted from the text below. Clean and verify them by:
1. Removing any incorrect concepts or tags that are not supported by the text.
2. Removing duplicate or highly redundant concepts and tags.
3. Merging concepts that refer to the exact same entity/topic.
4. Refining descriptions to be highly accurate and clear based on the text.
5. Ensuring all tags are properly formatted with a '#' prefix and no spaces.

TEXT:
{text}
CONCEPTS:
{concepts}
TAGS:
{tags}""")
])

prompt_relation = ChatPromptTemplate.from_messages([
    ("system", "You are a relationship extraction engine that finds connections BETWEEN different notes in a knowledge base."),
    ("user", """Below are two lists of concepts extracted from notes in a knowledge base.
Each concept has a 'source_note' field indicating which note it came from.

YOUR TASK: Find semantic relationships that involve AT LEAST ONE concept from the NEW CONCEPTS list.
You can link New ↔ New, or New ↔ Existing, but DO NOT link Existing ↔ Existing (we already know those).

Allowed relationship types:
- uses, depends_on, extends, similar_to, part_of
- implemented_with, alternative_to, improves, causes

Rules:
- ONLY extract relationships between concepts from DIFFERENT notes.
- AT LEAST ONE concept in the relationship MUST be from the NEW CONCEPTS list.
- Do NOT create relationships between concepts from the same note.
- **Strict Quality Threshold**: Focus on highly meaningful, direct, and significant semantic connections. Do NOT link concepts that only share a loose, generic, or trivial association (e.g. do not link every database just because they are 'databases'). The link must represent a true conceptual dependency, extension, or implementation choice.
- **Evidence-Based**: The connection must be clearly supported by the concept explanations provided. In the `evidence` field, write a concise explanation showing exactly why the two concepts/notes are strongly related.

NEW CONCEPTS:
{new_concepts}

EXISTING CONCEPTS:
{existing_concepts}""")
])

prompt_relation_verify = ChatPromptTemplate.from_messages([
    ("system", "You are a link verification engine. Verify the proposed cross-note relationships."),
    ("user", """Below are proposed relationships between concepts from different notes.
Verify each relationship:
1. Remove any invalid or unsupported relationships.
2. Remove duplicate relationships.
3. Ensure the source and target concepts actually exist in the concepts list.
4. Ensure the relationship type is semantically correct.
5. **Strict Quality Control**: Reject any relationships that represent weak, trivial, or generic connections. Every kept relationship must warrant a direct Obsidian link between the two notes. If the link does not add strong cognitive value to the reader, remove it.

CONCEPTS:
{concepts}

RELATIONSHIPS:
{relationships}""")
])
