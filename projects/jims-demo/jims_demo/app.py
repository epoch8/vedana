from jims_core.app import JimsApp
from jims_demo.db import get_sessionmaker
from jims_demo.simple_pipeline import say_hello, simple_pipeline

app = JimsApp(
    sessionmaker=get_sessionmaker(),
    conversation_start_pipeline=say_hello,
    pipeline=simple_pipeline,
)
