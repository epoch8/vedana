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

from vedana_core.settings import settings
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


class RagAgent:
    _data_model: DataModel
    _graph_descr: str
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
        self._vts_args = self._build_vts_arg_model()

    def _build_vts_arg_model(self) -> Type[VTSArgs]:
        """Create a Pydantic model with Enum-constrained fields for the VTS tool."""

        _vts_indices = self._data_model.vector_indices()

        if not _vts_indices:
            return VTSArgs

        # Label Enum – keys of `_vts_indices`
        LabelEnum = enum.Enum("LabelEnum", {name: name for (name, _) in _vts_indices})  # type: ignore

        # Property Enum – unique values of `_vts_indices`
        unique_props = set(attr for (_, attr) in _vts_indices)
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

    async def search_vector_text(
        self,
        label: str,
        prop_name: str,
        search_value: str,
        threshold: float,
        top_n: int = 5,
    ) -> list[Record]:
        embed = await self.llm.llm.create_embedding(search_value)
        return await self.graph.vector_search(label, prop_name, embed, threshold=threshold, top_n=top_n)

    @staticmethod
    def result_to_text(query: str, result: list[Record] | Exception) -> str:
        if isinstance(result, Exception):
            return f"Query: {query}\nResult: 'Error executing query'"
        rows_str = "\n".join(row_to_text(row) for row in result)
        return f"Query: {query}\nRows:\n{rows_str}"

    async def execute_cypher_query(self, query, rows_limit: int = 30) -> QueryResult:
        try:
            return list(islice(await self.graph.execute_ro_cypher_query(query), rows_limit))
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

    async def text_to_answer_with_vts_and_cypher(
        self, text_query: str, threshold: float, temperature: float | None = None, top_n: int = 5
    ) -> tuple[str, list[VTSQuery], list[CypherQuery]]:
        vts_queries: list[VTSQuery] = []
        cypher_queries: list[CypherQuery] = []

        async def vts_fn(args: VTSArgs) -> str:
            label = args.label.value if isinstance(args.label, enum.Enum) else args.label
            prop = args.property.value if isinstance(args.property, enum.Enum) else args.property

            th = self._data_model.embeddable_attributes().get(prop, {}).get("th") or threshold
            self.logger.info(f"vts_fn(label={label}, property={prop}, th={th}, n={top_n})")

            vts_queries.append(VTSQuery(label, prop, args.text))
            vts_res = await self.search_vector_text(label, prop, args.text, threshold=th, top_n=top_n)
            return self.result_to_text(VTS_TOOL_NAME, vts_res)

        async def cypher_fn(args: CypherArgs) -> str:
            self.logger.info(f"cypher_fn({args})")
            cypher_queries.append(CypherQuery(args.query))
            res = await self.execute_cypher_query(args.query)
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

        msgs, answer = await self.llm.generate_cypher_query_with_tools(
            data_descr=self._graph_descr,
            messages=self.ctx.history[-settings.pipeline_history_length:],
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
