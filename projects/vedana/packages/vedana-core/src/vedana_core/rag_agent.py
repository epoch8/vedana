import enum
import json
import logging
import re
from dataclasses import dataclass
from itertools import islice
from typing import Any, Mapping, Type

import neo4j
import neo4j.graph
from jims_core.thread.thread_context import ThreadContext
from pydantic import BaseModel, Field, create_model

from vedana_core.data_model import DataModel
from vedana_core.graph import Graph, Record
from vedana_core.llm import LLM, Tool

QueryResult = list[Record] | Exception


# TODO replace with VTSArgs and CypherArgs
class CypherQuery(str): ...


@dataclass
class VTSQuery:
    label: str
    param: str
    query: str

    def __str__(self) -> str:
        return f'vector_search("{self.label}","{self.param}","{self.query}")'


DBQuery = CypherQuery | VTSQuery | str


@dataclass
class RagResults:
    fts_res: list[Record] | None = None
    vts_res: list[Record] | None = None
    db_query_res: list[tuple[DBQuery, QueryResult]] | None = None


class VTSArgs(BaseModel):
    label: str = Field(description="node label")
    property: str = Field(description="node property to search in")
    text: str = Field(description="text to search similar")


class CypherArgs(BaseModel):
    query: str = Field(description="Cypher query")


class GetHistoryArgs(BaseModel):
    max_history: int = Field(20000, description="Maximum text length to retrieve from history. Cuts off on messages")


VTS_TOOL_NAME = "vector_text_search"
CYPHER_TOOL_NAME = "cypher"
HISTORY_TOOL_NAME = "get_conversation_history"


class RagAgent:
    _data_model: DataModel
    _graph_descr: str
    _vts_indices: dict[str, str]
    _vts_args: type[VTSArgs]

    def __init__(
        self,
        graph: Graph,
        data_model: DataModel,
        llm: LLM,
        ctx: ThreadContext,
        logger: logging.Logger | None = None,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.set_data_model(data_model)
        self.ctx = ctx

    def set_data_model(self, data_model: DataModel) -> None:
        self._data_model = data_model
        self._graph_descr = data_model.to_text_descr()
        self._vts_indices = data_model.vector_indices()
        self._vts_args = self._build_vts_arg_model()

    def _build_vts_arg_model(self) -> Type[VTSArgs]:
        """Create a Pydantic model with Enum-constrained fields for the VTS tool."""

        if not self._vts_indices:
            return VTSArgs

        # Label Enum – keys of `_vts_indices`
        LabelEnum = enum.Enum("LabelEnum", {name: name for name in self._vts_indices.keys()})  # type: ignore

        # Property Enum – unique values of `_vts_indices`
        unique_props = set(self._vts_indices.values())
        prop_member_mapping: dict[str, str] = {}

        used_names: set[str] = set()
        for idx, prop in enumerate(sorted(unique_props)):
            sanitized = re.sub(r"\W|^(?=\d)", "_", prop)
            if sanitized in used_names:
                sanitized = f"{sanitized}_{idx}"
            used_names.add(sanitized)
            prop_member_mapping[sanitized] = prop

        PropertyEnum = enum.Enum("PropertyEnum", prop_member_mapping)  # type: ignore

        VTSArgsEnum = create_model(
            "VTSArgsEnum",
            label=(LabelEnum, Field(description="node label")),
            property=(PropertyEnum, Field(description="node property to search in")),
            text=(str, Field(description="text for semantic search")),
            __base__=VTSArgs,
        )

        return VTSArgsEnum

    def search_vector_text(
        self,
        label: str,
        prop_name: str,
        search_value: str,
        threshold: float,
        top_n: int = 5,
    ) -> list[Record]:
        embed = self.llm.llm.create_embedding_sync(search_value)
        return self.graph.vector_search(label, prop_name, embed, threshold=threshold, top_n=top_n)

    # Unused for now
    def search_full_text(self, idx: str, query: str, limit: int = 10) -> list[Record]:
        return list(self.graph.text_search(idx, query, limit))

    def _llm_answer_to_queries(self, answer: str) -> list[DBQuery]:
        str_queries: list[str]
        if answer.startswith("["):
            str_queries = json.loads(answer)
        elif "---" in answer:
            str_queries = answer.split("---")
        else:
            str_queries = [answer]

        str_queries = [clear_cypher(q) for q in str_queries if q]

        vts_re = re.compile(r'vector_search\("(\S+)",\s*"(\S+)",\s*"(.+?)"\s*\)')

        queries: list[DBQuery] = []
        for str_q in str_queries:
            vts_args = next(iter(vts_re.findall(str_q)), None)
            if vts_args:
                queries.append(VTSQuery(*vts_args))
            else:
                queries.append(CypherQuery(str_q))

        return queries

    async def text_to_queries(self, text_query: str) -> list[DBQuery]:
        # filtered_graph_descr = await self.llm.filter_graph_structure(
        #     self._graph_descr, text_query
        # )
        # answer = await self.llm.generate_cypher_query_v5(filtered_graph_descr, text_query)
        answer = await self.llm.generate_cypher_query_v5(self._graph_descr, text_query)
        return self._llm_answer_to_queries(answer)

    @staticmethod
    def result_to_text(query: str, result: list[Record] | Exception) -> str:
        if isinstance(result, Exception):
            return f"Query: {query}\nResult: 'Error executing query'"
        rows_str = "\n".join(row_to_text(row) for row in result)
        return f"Query: {query}\nRows:\n{rows_str}"

    def execute_cypher_query(self, query, rows_limit: int = 30) -> QueryResult:
        try:
            return list(islice(self.graph.execute_ro_cypher_query(query), rows_limit))
        except Exception as e:
            self.logger.exception(e)
            return e

    def rag_results_to_text(self, results: RagResults) -> str:
        all_results = results.db_query_res or []
        if results.vts_res:
            all_results.append(("Vector text search", results.vts_res))
        if results.fts_res:
            all_results.append(("Full text search", results.fts_res))
        return "\n\n".join(self.result_to_text(str(q), r) for q, r in all_results)

    def _get_conversation_history_tool_func(self, args: GetHistoryArgs) -> str:
        if not self.ctx or len(self.ctx.history) <= 1:
            return "Conversation history is not available in the current context."

        relevant_events = [
            event for event in self.ctx.history if event.get("role") == "user" or event.get("role") == "assistant"
        ][:-1]  # last message is current query

        # Get the last N messages
        history_len = 0
        history_msgs = []

        for event in reversed(relevant_events):
            role = event.get("role")
            content = event.get("content", "")
            if content:  # Ensure there is content to add
                message_text = f"{role}: {content}"
                if len(message_text) + history_len > args.max_history and role != "user":  # break only on questions
                    break
                else:
                    history_len += len(message_text)
                    history_msgs.append(message_text)

        history_text = "\n---\n".join(reversed(history_msgs))
        if not history_text:
            return f"No relevant conversation history found (out of {len(relevant_events)} total messages)."

        self.logger.info(
            f"Retrieved conversation history (last {len(history_msgs)} messages with total length {len(history_text)})."
        )
        return history_text

    async def text_to_answer_with_vts_and_cypher(
        self, text_query: str, threshold: float, temperature: float = 0, top_n: int = 5
    ) -> tuple[str, list[VTSQuery], list[CypherQuery]]:
        vts_queries: list[VTSQuery] = []
        cypher_queries: list[CypherQuery] = []

        def vts_fn(args: VTSArgs) -> str:
            label = args.label.value if isinstance(args.label, enum.Enum) else args.label
            prop = args.property.value if isinstance(args.property, enum.Enum) else args.property

            th = self._data_model.embeddable_attributes().get(prop, {}).get("th") or threshold
            self.logger.info(f"vts_fn(label={label}, property={prop}, th={th}, n={top_n})")

            vts_queries.append(VTSQuery(label, prop, args.text))
            vts_res = self.search_vector_text(label, prop, args.text, threshold=th, top_n=top_n)
            return self.result_to_text(VTS_TOOL_NAME, vts_res)

        def cypher_fn(args: CypherArgs) -> str:
            self.logger.info(f"cypher_fn({args})")
            cypher_queries.append(CypherQuery(args.query))
            res = self.execute_cypher_query(args.query)
            return self.result_to_text(CYPHER_TOOL_NAME, res)

        vts_tool = Tool(
            VTS_TOOL_NAME,
            "Vector search for similar text in node properties, use for semantic search",
            self._vts_args,
            vts_fn,
        )

        cypher_tool = Tool(
            CYPHER_TOOL_NAME,
            "Execute a Cypher query against the graph database. Use for structured data retrieval and graph traversal.",
            CypherArgs,
            cypher_fn,
        )

        tools: list[Tool] = [vts_tool, cypher_tool]

        if self.ctx.history:
            tools.append(
                Tool(
                    name=HISTORY_TOOL_NAME,
                    description="Retrieves past messages from the current conversation. Use to get context for the current user query.",
                    args_cls=GetHistoryArgs,
                    fn=self._get_conversation_history_tool_func,
                )
            )

        msgs, answer = await self.llm.generate_cypher_query_with_tools(
            data_descr=self._graph_descr,
            text_query=text_query,
            tools=tools,
            temperature=temperature,
        )

        if not answer:
            self.logger.warning(f"No answer found for {text_query}. Generating empty answer...")
            answer = await self.llm.generate_no_answer(text_query, self.ctx.history)

        return answer, vts_queries, cypher_queries


def _remove_embeddings(val: Any):
    if isinstance(val, Mapping):
        return {k: _remove_embeddings(v) for k, v in val.items() if not k.endswith("_embedding")}
    if isinstance(val, list):
        return [_remove_embeddings(v) for v in val]
    return val


def _clear_record_val(val: Any):
    params = _remove_embeddings(val)
    if isinstance(val, neo4j.graph.Node) and isinstance(params, dict):
        params["labels"] = list(val.labels)
    return params


def row_to_text(row: Any) -> str:
    if isinstance(row, neo4j.Record):
        row = {k: _clear_record_val(v) for k, v in row.items()}
    try:
        return json.dumps(row, ensure_ascii=False, indent=2)
    except TypeError:
        return str(row)


def main():
    import logging

    from jims_core.llms.llm_provider import LLMProvider

    from vedana_core.graph import MemgraphGraph
    from vedana_core.settings import settings as s

    logging.basicConfig(level=logging.INFO)

    data_model = DataModel.load_grist_online(
        s.grist_data_model_doc_id, api_key=s.grist_api_key, grist_server=s.grist_server_url
    )
    llm = LLM(LLMProvider())
    q = "если я произвожу роботов, то какой окпд мне взять?"

    with MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd, "") as graph:
        agent = RagAgent(graph, data_model, llm)
        answer, vts_q, cypher_q = agent.text_to_answer_with_vts_and_cypher(q, threshold=0.8, temperature=0)
        print()
        print("vts_q")
        print(vts_q)
        print()
        print("cypher_q")
        print(cypher_q)
        print()
        print("answer")
        print(answer)


if __name__ == "__main__":
    main()
