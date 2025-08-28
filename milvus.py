import os
from chromadb import URI
from openai import AsyncOpenAI
import json
from typing import List
import asyncio

client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


async def get_emb(s:str) -> List[float]:
    completion = await client.embeddings.create(
    model="text-embedding-v4",
    input=s,
    dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
    encoding_format="float",
    
    )
    embedding = json.loads(completion.model_dump_json())["data"][0]["embedding"]
    return embedding


# import numpy as np

# def cosine(a:str,b:str):
#     ae = f(a)
#     be = f(b)
#     a = np.asarray(ae)
#     b = np.asarray(be)
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# print(cosine("朝鲜族","中国少数民族"))

# # 交叉性评估：
# # 原因： 二级学科不一定能够直接通过embedding相似度找到对应一级学科，但是三级学科一般都可找到二级学科。同时因为现在的embedding模型不经过微调很难符合已有的规则，因此在不微调的前提下，我们需要加上一些规则来调整embedding的结果。
# # embedding 二级学科-> 存入向量数据库， embedding查询学科->查询所属二级学科 -> 找到一级学科。 
# # 然后需要根据相似度来调整学科树，（还没想到如何调整），最后综合查询学科找到的二级学科之间的树的直径综合他们之间的相似度最为最终的交叉性评估。

# # 未来影响力评估：
# #



import json
from pymilvus import MilvusClient, connections, Collection, FieldSchema, CollectionSchema, DataType, utility

async def main():
    json_file_path = "/Users/peelsannaw/Desktop/disciplines.json"

    with open(json_file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    seconds = []

    for first_level_dict in data:
        for second_level_list in first_level_dict.values():
            for second_level_dict in second_level_list:
                for second_level_key in second_level_dict.keys():
                    seconds.append(second_level_key)

    # connections.connect("./discipline.db")
    client = MilvusClient("./discipline.db")
    dim = 1024  # Vector dimension
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="discipline_name", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="discipline embeddings")
    index = client.prepare_index_params()
    collection_name = "discipline_embeddings"
    
    index_params = {
        "metric_type": "COSINE",
        "index_type": "FLAT",
        "params": {}
    }
    
    index.add_index(field_name="embedding", index_params=index_params)
    client.create_collection(collection_name, dimension=dim, schema=schema,index_params=index)
    semaphore = asyncio.Semaphore(5)
    
    async def get_emb_with_semaphore(discipline: str) -> List[float]:
        async with semaphore:
            return await get_emb(discipline)
    
    tasks = [get_emb_with_semaphore(discipline) for discipline in seconds]
    embeddings = await asyncio.gather(*tasks)
    
    names = seconds

    data = [
       {"discipline_name": name, "embedding": emb} for name, emb in zip(names, embeddings)
    ]
    
    client.insert(collection_name, data)
    client.flush(collection_name=collection_name)
    print(client.list_collections())
    print("flush done")

if __name__ == "__main__":
    asyncio.run(main())
