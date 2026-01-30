# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# -----------------------------------------------------------------------------
# Standard Prompt: No reasoning / timeline / reflection (baseline)
# -----------------------------------------------------------------------------

STANDARD_PROMPT_TEMPLATE = """You are an AI assistant that answers questions strictly using the provided temporal context.
Provide your final, concise answer within the <answer> tags.
If the answer is a number, output only the number, nothing else. Otherwise, output the entity or event without any additional comments.

Important:
• The response must be entirely contained within the <answer> tags.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.

Response Format:
<answer>
[Your final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""

# -----------------------------------------------------------------------------
# FULL: All stages (Reasoning -> Timeline -> Reflection -> Answer)
# -----------------------------------------------------------------------------

TISER_PROMPT_TEMPLATE = """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are unidirectional.
3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags.
4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, go back to Step 1, otherwise move to the next step.
5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number, nothing else. Otherwise, output the entity or event without any additional comments.

Additional Instructions:
• The <reasoning>, <timeline>, and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<timeline>
[Relevant temporal events.]
</timeline>
<reflection>
[Reflection on reasoning and timeline.]
</reflection>
[Any adjustments to your thinking.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""

# -----------------------------------------------------------------------------
# Minimal / Finetuned Actor Prompt
# -----------------------------------------------------------------------------

ACTOR_FINETUNED_TEMPLATE = """You are an AI assistant that has to respond to questions given a context

Question: {question}

Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# Multi-Stage Pipeline Prompts (Actor -> Critic -> Solver)
# -----------------------------------------------------------------------------

CRITIC_PROMPT_TEMPLATE = """You are an AI Critic responsible for the evaluation phase of a Chain of Thought process.
Your task is to analyze the provided <reasoning> and <timeline> to identify errors, inconsistencies, or logical flaws based *strictly* on the provided <context>.

Follow these steps:
Step 1. Read the Question and the Context carefully.
Step 2. Analyze the provided "Draft Reasoning" and "Draft Timeline" to check if they accurately reflect the Context.
Step 3. Check for specific errors:
        - Hallucinations: Information present in the reasoning but missing from the context.
        - Temporal errors: Incorrect dates or sequence of events.
        - Logical fallacies: Conclusions that do not follow from the premises.
Step 4. Provide your evaluation within <reflection> tags. 
        - If the draft is correct, simply state it.
        - If errors exist, be diagnostic.

Important:
- Do NOT output the answer.
- Do NOT summarize the context or the drafts.
- Your output must contain ONLY the <reflection> section.
- Do NOT rewrite the reasoning or the timeline.
- Do not use enumerations, use plain text paragraphs.

Use the following format for your response:

<reflection>
[Your critique of the reasoning and timeline, pointing out errors or confirming accuracy based on the context.]
</reflection>

Input Data:

Question: {question}

Temporal context: {context}

Draft Reasoning: {draft_reasoning}

Draft Timeline: {draft_timeline}
"""


FINAL_SOLVER_PROMPT_TEMPLATE = """You are an AI assistant responsible for the final phase of a Chain of Thought process.
Your task is to synthesize the provided information, apply the Critic's feedback, and formulate the final correct answer based *strictly* on the provided <context>.

Follow these steps:
Step 1. Review the Question, Context, Draft Reasoning, Draft Timeline, and the Critic's Reflection.
Step 2. Address the Reflection within the <adjustments> tags.
        - If the Critic identified errors, explain how you are correcting them based on the Context.
        - If the Critic confirmed the reasoning is correct, simply state that the logic holds.
Step 3. Provide your final, concise answer within the <answer> tags. 
        - If the answer is a number, output just the number nothing else.
        - Otherwise, output the entity or event, without any additional comments.

Important:
- Trust the Context above all else.
- The <adjustments> section is for your internal correction process.
- Do NOT copy the Critic's reflection verbatim.
- **WARNING:** The Critic might be wrong. If the Critic confirms a reasoning that contradicts the Context, YOU MUST OVERRULE IT.
- The response to the query must be entirely contained within the <answer> tags.
- Do not use enumerations, use plain text paragraphs.

Use the following format for your response:

<adjustments>
[Your final logic, incorporating the specific feedback from the Critic to fix any errors or confirm the result.]
</adjustments>
<answer>
[Your final, concise answer to the query.]
</answer>

Input Data:

Question: {question}

Temporal context: {context}

Draft Reasoning: {draft_reasoning}

Draft Timeline: {draft_timeline}

Critic's Reflection: {critic_reflection}
"""


# =============================================================================
# Ablation Study Prompts (Single-Prompt Variants)
# =============================================================================
# These are single prompts that implement different subsets of the full TISER
# reasoning pipeline. Each expects:
#   - {question}
#   - {context}
# =============================================================================


# -----------------------------------------------------------------------------
# ONLY REASONING: Reasoning -> Answer (no timeline, no reflection)
# -----------------------------------------------------------------------------

ABLATION_ONLY_REASONING_PROMPT_TEMPLATE = """You are an AI assistant that answers queries using step-by-step reasoning.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Based on your reasoning, provide the final answer within the <answer> tags.

Additional Instructions:
• The <reasoning> section is for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# ONLY TIMELINE: Timeline -> Answer (no reasoning, no reflection)
# -----------------------------------------------------------------------------

ABLATION_ONLY_TIMELINE_PROMPT_TEMPLATE = """You are an AI assistant that answers queries by identifying relevant temporal events.

Follow these steps:
1. Identify the temporal events in the given context that are relevant for answering the question, and describe them within <timeline> tags. Assume relations in the context are unidirectional.
2. Based on the identified temporal events, provide the final answer within the <answer> tags.

Additional Instructions:
• The <timeline> section is for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<timeline>
[Relevant temporal events.]
</timeline>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO REFLECTION: Reasoning -> Timeline -> Answer (reflection removed)
# -----------------------------------------------------------------------------

ABLATION_NO_REFLECTION_PROMPT_TEMPLATE = """You are an AI assistant that uses a Chain of Thought (CoT) approach to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are unidirectional.
3. Provide your final, concise answer within the <answer> tags.

Additional Instructions:
• The <reasoning> and <timeline> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<timeline>
[Relevant temporal events.]
</timeline>
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO TIMELINE: Reasoning -> Reflection -> Answer (timeline removed)
# -----------------------------------------------------------------------------

ABLATION_NO_TIMELINE_PROMPT_TEMPLATE = """You are an AI assistant that uses reasoning and reflection to answer queries.

Follow these steps:
1. Reason through the problem step by step within the <reasoning> tags.
2. Reflect on your reasoning to check for any errors or improvements within the <reflection> tags.
3. Make any necessary adjustments based on your reflection.
4. Provide your final, concise answer within the <answer> tags.

Additional Instructions:
• The <reasoning> and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<reasoning>
[Your step-by-step reasoning goes here.]
<reflection>
[Reflection on the reasoning.]
</reflection>
[Any adjustments.]
</reasoning>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""


# -----------------------------------------------------------------------------
# NO REASONING: Timeline -> Reflection -> Answer (reasoning removed)
# -----------------------------------------------------------------------------

ABLATION_NO_REASONING_PROMPT_TEMPLATE = """You are an AI assistant that answers queries by analyzing temporal information and reflecting on it.

Follow these steps:
1. Identify relevant temporal events in the given context for answering the question within <timeline> tags. Assume relations in the context are unidirectional.
2. Reflect on the identified temporal events to check for errors or missing information within <reflection> tags.
3. Provide the final answer within the <answer> tags.

Additional Instructions:
• The <timeline> and <reflection> sections are for internal reasoning only.
• Do not use enumerations or lists when writing; use plain text such as paragraphs.
• The response to the query must be entirely contained within the <answer> tags.

Response Format:
<timeline>
[Relevant temporal events.]
</timeline>
<reflection>
[Reflection on the temporal analysis.]
</reflection>
<answer>
[Final answer.]
</answer>

Question: {question}
Temporal Context: {context}"""
