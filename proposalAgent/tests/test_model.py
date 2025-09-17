import uuid
from typing_extensions import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="qwen-plus",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                 api_key="sk-0e349a8dc24443988825b69a56d2b868"
                 )

def call_model(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,  
):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "1",
    }
}
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
    config,
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

config = {
    "configurable": {
        "thread_id": "2", # 跨越线程
        "user_id": "1",
    }
}

for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "what is my name?"}]},
    config,
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()