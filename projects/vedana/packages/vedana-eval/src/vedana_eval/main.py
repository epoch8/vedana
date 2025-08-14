import asyncio
import datetime
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import grist_api
from dotenv import load_dotenv
from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent
from jims_core.thread.thread_context import ThreadContext
from pydantic import BaseModel, Field
from uuid_extensions import uuid7

from vedana_core.data_model import DataModel
from vedana_core.data_provider import GristSQLDataProvider
from vedana_core.graph import MemgraphGraph
from vedana_core.rag_pipeline import RagPipeline
from vedana_core.settings import settings as s

from settings import settings as eval_settings


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GdsQuestion:
    gds_question: str
    gds_answer: str
    question_context: str | None = None

    def question_with_context(self) -> str:
        if self.question_context:
            return f"{self.gds_question} {self.question_context}".strip()
        return self.gds_question


class JudgeResult(BaseModel):
    test_status: str = Field(description="pass or fail")
    comment: str = Field(description="short justification and hints")
    errors: list[str] | str | None = Field(description="critical errors found, if any; else empty")


judge_prompt_template = (
    "You are a strict evaluation judge. Compare the model's answer with the golden answer and the expected "
    "retrieval context. Consider whether the model's answer is factually aligned and sufficiently complete. "
    "Use the provided technical info (retrieval queries) only as hints for whether the context seems adequate. "
    "Return a JSON object with fields: test_status in {'pass','fail'}, comment, errors."
)


def get_test_set() -> pd.DataFrame:
    dp = GristSQLDataProvider(
        doc_id=eval_settings.grist_test_set_doc_id,
        grist_server=s.grist_server_url,
        api_key=s.grist_api_key,
    )
    gds = dp.get_table_df(eval_settings.gds_table_name)
    gds = gds.dropna(subset=["gds_question", "gds_answer"])
    gds = gds.loc[(gds["gds_question"] != "") & (gds["gds_answer"] != "")].copy()
    return gds


def _build_ctx(user_query: str) -> ThreadContext:
    history: list[CommunicationEvent] = [CommunicationEvent(role="user", content=user_query)]
    return ThreadContext(
        thread_id=uuid7(),
        history=history,
        events=[],
        llm=LLMProvider(),
    )


async def _run_pipeline_for_query(pipeline: RagPipeline, user_query: str) -> tuple[str, dict[str, Any]]:
    ctx = _build_ctx(user_query)
    await pipeline(ctx)

    llm_answer = ""
    technical_info: dict[str, Any] = {}
    for event in ctx.outgoing_events:
        if event.event_type == "comm.assistant_message":
            # last assistant message
            llm_answer = str(event.event_data.get("content", ""))
        elif event.event_type == "rag.query_processed":
            technical_info = dict(event.event_data.get("technical_info", {}))

    return llm_answer, technical_info


async def judge_answer(
    provider: LLMProvider,
    judge_prompt: str,
    golden_answer: str,
    golden_context: str | None,
    model_answer: str,
    technical_info: dict[str, Any],
) -> JudgeResult:
    sys_msg = {
        "role": "system",
        "content": judge_prompt,
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Golden answer:\n{golden_answer}\n\n"
            f"Expected context (if any):\n{golden_context or ''}\n\n"
            f"Model answer:\n{model_answer}\n\n"
            f"Technical info (for reference):\n{technical_info}"
        ),
    }
    res = await provider.chat_completion_structured([sys_msg, user_msg], JudgeResult)
    if res is None:
        return JudgeResult(test_status="fail", comment="Judge did not return a result", errors="judge_empty")
    return res


async def run_tests(gds: pd.DataFrame) -> None:
    graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd)

    data_model = await DataModel.load_from_graph(graph)
    if data_model is None:
        logger.info("No DataModel found in graph – loading from Grist …")
        data_model = DataModel.load_grist_online(
            s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )
        # todo
        # try:
        #     await data_model.update_data_model_node(graph)
        # except Exception as e:
        #     logger.warning(f"Unable to cache DataModel in graph: {e}")

    pipeline = RagPipeline(
        graph=graph,
        data_model=data_model,
        logger=logger,
    )

    judge_llm = LLMProvider()
    judge_llm.set_model(eval_settings.judge_model)

    judge_prompt = data_model.prompt_templates().get("eval_judge_prompt", judge_prompt_template)

    client = grist_api.GristDocAPI(
        eval_settings.grist_test_set_doc_id, server=s.grist_server_url, api_key=s.grist_api_key
    )

    rows_to_write: list[dict[str, Any]] = []
    today = datetime.date.today().isoformat()

    for row in gds.itertuples(index=False):
        logger.info(f"Processing row: {row}")
        # Workaround for grist changing template column names on imports/exports
        gq = getattr(row, "gds_question", getattr(row, "Gds_Question", ""))
        ga = getattr(row, "gds_answer", getattr(row, "Gds_Answer", ""))
        qc = getattr(row, "question_context", getattr(row, "Question_Context", None))

        q = GdsQuestion(gds_question=str(gq), gds_answer=str(ga), question_context=(None if qc is None else str(qc)))

        try:
            llm_answer, tech = await _run_pipeline_for_query(pipeline, q.question_with_context())
        except Exception as e:
            logger.exception("Pipeline failed for question")
            llm_answer = ""
            tech = {"error": str(e)}

        # Judge
        try:
            judge_res = await judge_answer(judge_llm, judge_prompt, q.gds_answer, q.question_context, llm_answer, tech)
        except Exception as e:
            logger.exception("Judge failed for question")
            judge_res = JudgeResult(test_status="fail", comment="Judge error", errors=[str(e)])

        record = {
            "gds_question": q.gds_question,
            "gds_answer": q.gds_answer,
            "LLM_answer": llm_answer,
            "test_status": judge_res.test_status,
            "comment": judge_res.comment,
            "errors": "\n".join(judge_res.errors) if isinstance(judge_res.errors, list) else judge_res.errors,
            "test_model": "\n".join(tech.get("model_stats")),
            "tool_calls": "\n".join(tech.get("vts_queries")) + "\n---\n" + "\n".join(tech.get("cypher_queries")),
            "test_environment": eval_settings.test_environment,
            "test_date": today,
        }
        rows_to_write.append(record)

        # Flush in small batches to reduce memory and see progress
        if len(rows_to_write) >= 10:
            client.add_records(eval_settings.tests_table_name, rows_to_write)
            rows_to_write.clear()

    if rows_to_write:
        client.add_records(eval_settings.tests_table_name, rows_to_write)


def main() -> None:
    gds = get_test_set()
    asyncio.run(run_tests(gds))


if __name__ == "__main__":
    main()
