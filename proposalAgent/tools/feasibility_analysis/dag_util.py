import os
import chromadb
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

# # 原因： 二级学科不一定能够直接通过embedding相似度找到对应一级学科，但是三级学科一般都可以
# # embedding 二级学科-> 存入向量数据库， embedding查询学科->查询所属二级学科 -> 找到一级学科。 利用一级学科embedding计算距离(仅通过相似度？ )



import json

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

    # Setup ChromaDB client
    db_client = chromadb.PersistentClient(path="./discipline_cm_db")
    
    collection_name = "discipline_embeddings"
    # Get or create the collection
    collection = db_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # Using cosine distance
    )

    semaphore = asyncio.Semaphore(5)
    
    async def get_emb_with_semaphore(discipline: str) -> List[float]:
        async with semaphore:
            return await get_emb(discipline)
    
    tasks = [get_emb_with_semaphore(discipline) for discipline in seconds]
    embeddings = await asyncio.gather(*tasks)
    
    ids = [f"id_{i}" for i in range(len(seconds))]

    collection.add(
        embeddings=list(embeddings),
        documents=seconds,
        ids=ids
    )
    
    print(f"Collection '{collection_name}' created and data added.")
    print(f"Number of items in collection: {collection.count()}")


if __name__ == "__main__":
    asyncio.run(main())
