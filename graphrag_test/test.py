import os
import pandas as pd
import tiktoken
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

api_key = "EMPTY"
llm_model = "qwen3:latest"

llm = ChatOpenAI(
    api_base="http://127.0.0.1:11434/v1",
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

token_encoder = tiktoken.get_encoding("cl100k_base")

messages = [
    {
        "role": "user",
        "content": "Hi"
    }
]
response = llm.generate(messages=messages)
print(response)

#2. Loading Data and Building the Context:
# parquet files generated from indexing pipeline
INPUT_DIR = ".test/rag_graph/output/run-id" # replace the run-id with the created one
COMMUNITY_REPORT_TABLE = "artifacts/create_final_community_reports"
ENTITY_TABLE = "artifacts/create_final_nodes"
ENTITY_EMBEDDING_TABLE = "artifacts/create_final_entities"

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2

entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)


# Build global context based on community reports

context_builder = GlobalCommunityContext(
    community_reports=reports,
    entities=entities,  # default to None if you don't want to use community weights for ranking
    token_encoder=token_encoder,
)

#3. Configuring Global Search
context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 500,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

map_llm_params = {
    "max_tokens": 500,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

reduce_llm_params = {
    "max_tokens": 500,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}
search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    max_data_tokens=500,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    map_llm_params=map_llm_params,
    reduce_llm_params=reduce_llm_params,
    allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
    json_mode=True,  # set this to False if your LLM model does not support JSON mode.
    context_builder_params=context_builder_params,
    concurrent_coroutines=32,
    response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
)
result = search_engine.asearch(
    "Who has collaborated with Alice Johnson on any project?"
)
print(result.response)