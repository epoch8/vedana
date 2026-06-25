from jims_core.thread.thread_context import StatusEvent

STATUS_PROCESSING_QUESTION = StatusEvent("Processing your question...", "processing_question")
STATUS_ANALYZING_QUERY = StatusEvent("Analyzing query structure...", "analyzing_query")
STATUS_SEARCHING_KNOWLEDGE_BASE = StatusEvent("Searching knowledge base...", "searching_knowledge_base")
STATUS_FORMULATING_ANSWER = StatusEvent("Formulating the answer...", "formulating_answer")
