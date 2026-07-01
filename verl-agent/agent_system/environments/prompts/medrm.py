# No-info templates: used when no real <information> has been returned yet (no citation instructions).
QA_TEMPLATE_NO_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

Based on your search history, you need to take the next action to complete the task.
You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

Valid actions (ONLY TAKE ONE of the following):
(1) <search> your query </search>: call the external search engine for information if you consider you lack some knowledge. Your turn ends here, do not fabricate search results in the reasoning.
(2) <answer> your answer, with the final choice in \\boxed{{}} </answer>: output the final answer if you consider you are able to answer the question.

Note: 
(1) Text between <information></information> is the search results from search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output. 
(2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.

Question: {task_description}
"""

CALC_TEMPLATE_NO_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

Based on your search history, you need to take the next action to complete the task.
You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

Valid actions (ONLY TAKE ONE of the following):
(1) <search> your query </search>: call the external search engine for information if you consider you lack some knowledge. Your turn ends here, do not fabricate search results in the reasoning.
(2) <answer> your answer, with the final value in \\boxed{{}} </answer>: output the final answer if you consider you are able to answer the question.

Use one of the following final-value formats:
- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Note:
(1) Text between <information></information> is the search results from search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.
(2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.

Question: {task_description}
"""


# With-info templates: used when at least one real <information> block has been returned (includes citation instructions).
QA_TEMPLATE_WITH_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

Based on your search history, you need to take the next action to complete the task.
You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

Valid actions (ONLY TAKE ONE of the following):
(1) <search> your query </search>: call the external search engine for information if you consider you lack some knowledge. Your turn ends here, do not fabricate search results in the reasoning.
(2) <answer> your answer, with the final choice in \\boxed{{}} </answer>: output the final answer if you consider you are able to answer the question.

Citation rule (ONLY for internal thinking):
(1) Whenever you draw on a fact from prior search results (from <information> blocks), cite it inline using [T<turn>-R<result>] (example: [T2-R3]) in the thinking process.
(2) If no prior <information> block exists, or the retrieved information is irrelevant or noisy, do not output any citations.
(3) Do not invent citation ids.

Note:
(1) Text between <information></information> is the search results from search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.
(2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.

Question: {task_description}
"""

CALC_TEMPLATE_WITH_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

Based on your search history, you need to take the next action to complete the task.
You will be provided with:
1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
2. The question to answer.

Valid actions (ONLY TAKE ONE of the following):
(1) <search> your query </search>: call the external search engine for information if you consider you lack some knowledge. Your turn ends here, do not fabricate search results in the reasoning.
(2) <answer> your answer, with the final value in \\boxed{{}} </answer>: output the final answer if you consider you are able to answer the question.

Citation rule (ONLY for internal thinking):
(1) Whenever you draw on a retrieved fact, formula input, or numeric assumption (from prior <information> blocks), cite it inline using [T<turn>-R<result>] (example: [T2-R3]) in the thinking process.
(2) If no prior <information> block exists, or the retrieved information is irrelevant or noisy, do not output any citations.
(3) Do not invent citation ids.

Use one of the following final-value formats:
- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Note:
(1) Text between <information></information> is the search results from search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.
(2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.

Question: {task_description}
"""

QA_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions. Provide your response using format: <answer> your answer, with the final choice in \\boxed{{}} </answer>.

Question: {task_description}
"""

CALC_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions. Provide your response using format: <answer> your answer, with the final value (without units) in \\boxed{{}} </answer>.

Use one of the following final-value formats:
- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Question: {task_description}
"""


HIS_TEMPLATE = """
History Turns: (empty if this is the first turn)
{memory_context}
"""

QA_TEMPLATE_WITH_INFO = QA_TEMPLATE_WITH_INFO_NO_HIS + HIS_TEMPLATE
CALC_TEMPLATE_WITH_INFO = CALC_TEMPLATE_WITH_INFO_NO_HIS + HIS_TEMPLATE
QA_TEMPLATE_NO_INFO = QA_TEMPLATE_NO_INFO_NO_HIS + HIS_TEMPLATE
CALC_TEMPLATE_NO_INFO = CALC_TEMPLATE_NO_INFO_NO_HIS + HIS_TEMPLATE

FORMAT_REMINDER_PROMPT = """
You generated an invalid action in your previous turn.
Output exactly one action block; if using <answer>, include final answer/value in \\boxed{{}} inside <answer>...</answer>.
Use citation tags only for source ids that actually appeared in prior <information> blocks. If no prior <information> exists, do not output citations. If you rely on retrieved information in reasoning, cite the supported claim there; if you use it in the final <answer>, cite it there too. Do not invent or guess citation ids.
"""


# # No-info templates: used when no real <information> has been returned yet (no citation instructions).
# QA_TEMPLATE_NO_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

# Based on your search history, you need to take the next action to complete the task.
# You will be provided with:
# 1. Your history search attempts: query in format <search> query </search> and the search results returned by the external search tool.
# 2. The question to answer.

# Valid actions (ONLY TAKE ONE of the following):
# (1) <search> your query </search>: search to verify a specific medical fact, diagnosis, treatment, drug mechanism, or clinical guideline relevant to the question. Your turn ends here; do not fabricate search results — the external search tool will return them in the next turn.
# (2) <answer> your answer, with the final choice in \\boxed{{}} </answer>: output the final answer once you have verified the key medical facts.

# Strategy: Think step by step about what specific medical knowledge is needed to answer this question, then search to verify it.

# Note:
# (1) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.
# (2) Output EXACTLY ONE action block (search or answer).

# Question: {task_description}
# """

# CALC_TEMPLATE_NO_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

# Based on your search history, you need to take the next action to complete the task.
# You will be provided with:
# 1. Your history search attempts: query in format <search> query </search> and the search results returned by the external search tool.
# 2. The question to answer.

# Valid actions (ONLY TAKE ONE of the following):
# (1) <search> your query </search>: search to verify a specific medical formula, reference value, drug dose, or clinical parameter needed to solve this calculation. Your turn ends here; do not fabricate search results — the external search tool will return them in the next turn.
# (2) <answer> your answer, with the final value in \\boxed{{}} </answer>: output the final answer once you have verified the formula or key values.

# Strategy: Think step by step about which formula or reference value is required, then search to confirm it.

# Use one of the following final-value formats:
# - Decimal Answer Format: 17.29
# - Score-Based Answer Format: 5
# - Estimated Date Answer Format: 5/21/2021
# - Estimated Age Answer Format: (4 weeks, 3 days)

# Note:
# (1) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.
# (2) Output EXACTLY ONE action block (search or answer).

# Question: {task_description}
# """


# # With-info templates: used when at least one real <information> block has been returned (includes citation instructions).
# QA_TEMPLATE_WITH_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

# Based on your search history, you need to take the next action to complete the task.
# You will be provided with:
# 1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
# 2. The question to answer.

# Valid actions (ONLY TAKE ONE of the following):
# (1) <search> your query </search>: search to verify a specific medical fact, diagnosis, treatment, drug mechanism, or clinical guideline relevant to the question. Your turn ends here; do not fabricate search results — the external search tool will return them in the next turn.
# (2) <answer> your answer, with the final choice in \\boxed{{}} </answer>: output the final answer once you have verified the key medical facts.

# Strategy: Think step by step about what specific medical knowledge is needed to answer this question, then search to verify it.

# Citation rule (ONLY for internal thinking):
# (1) Whenever you draw on a fact from prior search results (from <information> blocks), cite it inline using [T<turn>-R<result>] (example: [T2-R3]) in the thinking process.
# (2) If no prior <information> block exists, or the retrieved information is irrelevant or noisy, do not output any citations.
# (3) Do not invent citation ids.

# Note:
# (1) Text between <information></information> is the search results from the search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.
# (2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.
# (3) Output EXACTLY ONE action block (search or answer).

# Question: {task_description}
# """

# CALC_TEMPLATE_WITH_INFO_NO_HIS = """You are a medical expert AI assistant. You have access to an external search tool to retrieve information. You can answer a question with many turns of search and reasoning.

# Based on your search history, you need to take the next action to complete the task.
# You will be provided with:
# 1. Your history search attempts: query in format <search> query </search> and the returned search results in <information> and </information>.
# 2. The question to answer.

# Valid actions (ONLY TAKE ONE of the following):
# (1) <search> your query </search>: search to verify a specific medical formula, reference value, drug dose, or clinical parameter needed to solve this calculation. Your turn ends here; do not fabricate search results — the external search tool will return them in the next turn.
# (2) <answer> your answer, with the final value in \\boxed{{}} </answer>: output the final answer once you have verified the formula or key values.

# Strategy: Think step by step about which formula or reference value is required, then search to confirm it.

# Citation rule (ONLY for internal thinking):
# (1) Whenever you draw on a retrieved fact, formula input, or numeric assumption (from prior <information> blocks), cite it inline using [T<turn>-R<result>] (example: [T2-R3]) in the thinking process.
# (2) If no prior <information> block exists, or the retrieved information is irrelevant or noisy, do not output any citations.
# (3) Do not invent citation ids.

# Use one of the following final-value formats:
# - Decimal Answer Format: 17.29
# - Score-Based Answer Format: 5
# - Estimated Date Answer Format: 5/21/2021
# - Estimated Age Answer Format: (4 weeks, 3 days)

# Note:
# (1) Text between <information></information> is the search results from the search engine after you perform a search action, **DO NOT** include any information in <information></information> in your output.
# (2) **DO NOT** simulate or imagine search results in your reasoning. If you decide to search, you must output a <search> action.
# (3) Output EXACTLY ONE action block (search or answer).

# Question: {task_description}
# """

# QA_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions. Provide your response using format: <answer> your answer, with the final choice in \\boxed{{}} </answer>.

# Question: {task_description}
# """

# CALC_BASE_TEMPLATE = """You are a medical expert AI assistant to answer questions. Provide your response using format: <answer> your answer, with the final value (without units) in \\boxed{{}} </answer>.

# Use one of the following final-value formats:
# - Decimal Answer Format: 17.29
# - Score-Based Answer Format: 5
# - Estimated Date Answer Format: 5/21/2021
# - Estimated Age Answer Format: (4 weeks, 3 days)

# Question: {task_description}
# """


# HIS_TEMPLATE = """
# History Turns: (empty if this is the first turn)
# {memory_context}
# """

# QA_TEMPLATE_WITH_INFO = QA_TEMPLATE_WITH_INFO_NO_HIS + HIS_TEMPLATE
# CALC_TEMPLATE_WITH_INFO = CALC_TEMPLATE_WITH_INFO_NO_HIS + HIS_TEMPLATE
# QA_TEMPLATE_NO_INFO = QA_TEMPLATE_NO_INFO_NO_HIS + HIS_TEMPLATE
# CALC_TEMPLATE_NO_INFO = CALC_TEMPLATE_NO_INFO_NO_HIS + HIS_TEMPLATE

# FORMAT_REMINDER_PROMPT = """
# You generated an invalid action in your previous turn.
# Output EXACTLY ONE action block:
# (1) If taking a search action, please use format: <search> your query </search>. Do not fabricate search results; the external search tool will return them in the next turn.
# (2) If taking an answer action, please use format: <answer> your answer, with the final choice in \\boxed{{}} </answer>.
# """