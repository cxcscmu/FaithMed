QA_RAG_TEMPLATE = """
You are a helpful medical assistant. Answer the question, and put the final choice in \\boxed{{}}.

Question:
{task_description}

Relevant documents:
{retrieved_docs}

Thinking policy:
1) Use retrieved documents only for internal thinking.
2) In internal thinking, if retrieved docs are relevant, cite every document-supported factual claim with [doc_id].
3) If retrieved docs are irrelevant or noisy, ignore them and answer normally from your own knowledge.

Response policy:
1) The response must be a user-facing answer only.
2) In the response, do not include any retrieval artifacts: no citations (e.g., [1], [1, 2]), no doc ids, no source/retrieval mentions, and no reference section.
"""


CALC_RAG_TEMPLATE = """
You are a helpful medical assistant. Calculate the requested value from the note, and put the final value (without units) in \\boxed{{}}.

Question:
{task_description}

Relevant documents:
{retrieved_docs}

Use one of the following final-value formats:
- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Thinking policy:
1) Use retrieved documents only for internal thinking.
2) In internal thinking, if retrieved docs are relevant, cite every document-supported factual claim with [doc_id].
3) If retrieved docs are irrelevant or noisy, ignore them and answer normally from your own knowledge.

Response policy:
1) The response must be a user-facing answer only.
2) In the response, do not include any retrieval artifacts: no citations (e.g., [1], [1, 2]), no doc ids, no source/retrieval mentions, and no reference section.
"""


HEALTHBENCH_RAG_TEMPLATE = """
You are a helpful medical assistant. Respond to the user's last message.

Conversation:
{task_description}

Relevant documents:
{retrieved_docs}

Grounding rules:
1) Use retrieved documents only for internal thinking.
2) In internal thinking, if retrieved docs are relevant, cite every document-supported factual claim with [doc_id].
3) If retrieved docs are irrelevant or noisy, ignore them and answer normally from your own knowledge.
4) The response must be a user-facing answer only.
5) In the response, do not include any retrieval artifacts: no citations (e.g., [1], [1, 2]), no doc ids, no source/retrieval mentions, and no reference section.
"""
