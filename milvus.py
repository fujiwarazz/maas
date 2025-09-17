import os
from openai import AsyncOpenAI
import json
from typing import List
import asyncio
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus import MilvusClient
# OpenAI客户端用于获取embedding
openai_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)


async def get_emb(s: str) -> List[float]:
    completion = await openai_client.embeddings.create(
        model="text-embedding-v4",
        input=s,
        dimensions=1024,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float",
    )
    embedding = json.loads(completion.model_dump_json())["data"][0]["embedding"]
    return embedding


async def main():
    # json_file_path = "/Users/peelsannaw/Desktop/codes/maas/mas4proposal/proposalAgent/data/course.json"

    # with open(json_file_path, "r", encoding='utf-8') as f:
    #     data = json.load(f)

    # # 提取二级学科信息
    # second_level_disciplines = []
    
    # for first_level in data:
    #     if "children" in first_level:
    #         for second_level in first_level["children"]:
    #             if "code" in second_level and "name" in second_level:
    #                 second_level_disciplines.append({
    #                     "code": second_level["code"],
    #                     "name": second_level["name"]
    #                 })
    
    # print(f"找到 {len(second_level_disciplines)} 个二级学科")
    # print("\n前10个二级学科示例:")
    # for i, discipline in enumerate(second_level_disciplines[:10]):
    #     print(f"{i+1:2d}. 代码: {discipline['code']}, 名称: {discipline['name']}")
    
    # # 按学科部统计
    # print("\n按学科部分类统计:")
    # department_count = {}
    # for first_level in data:
    #     if "children" in first_level:
    #         dept_name = first_level.get("name", "未知")
    #         dept_code = first_level.get("code", "未知")
    #         count = len([child for child in first_level["children"] if "code" in child and "name" in child])
    #         department_count[f"{dept_code} - {dept_name}"] = count
    #         print(f"{dept_code} - {dept_name}: {count} 个二级学科")
    
    # print(f"\n总计: {sum(department_count.values())} 个二级学科")

    # # 连接到Milvus
    # connections.connect(
    #     alias="default",
    #     uri="./discipline.db"
    # )
    
    # dim = 1024  # Vector dimension
    # fields = [
    #     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    #     FieldSchema(name="discipline_code", dtype=DataType.VARCHAR, max_length=10),
    #     FieldSchema(name="discipline_name", dtype=DataType.VARCHAR, max_length=100),
    #     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    # ]
    # schema = CollectionSchema(fields=fields, description="二级学科embeddings")
    # collection_name = "second_level_disciplines"
    
    # # 检查集合是否存在，如果存在则删除
    # if utility.has_collection(collection_name):
    #     utility.drop_collection(collection_name)
    
    # # 创建集合
    # collection = Collection(collection_name, schema)
    
    # # 创建索引
    # index_params = {
    #     "metric_type": "COSINE",
    #     "index_type": "FLAT",
    #     "params": {}
    # }
    
    # collection.create_index(field_name="embedding", index_params=index_params)
    
    # semaphore = asyncio.Semaphore(5)
    
    # async def get_emb_with_semaphore(discipline_name: str) -> List[float]:
    #     async with semaphore:
    #         return await get_emb(discipline_name)
    
    # # 为每个二级学科生成embedding
    # print("开始生成embeddings...")
    # tasks = [get_emb_with_semaphore(discipline["name"]) for discipline in second_level_disciplines]
    # embeddings = await asyncio.gather(*tasks)
    
    # # 准备插入数据
    # insert_data = []
    # for discipline, embedding in zip(second_level_disciplines, embeddings):
    #     insert_data.append({
    #         "discipline_code": discipline["code"],
    #         "discipline_name": discipline["name"],
    #         "embedding": embedding
    #     })
    
    # # 插入数据
    # collection.insert(insert_data)
    # collection.flush()
    # collection.load()
    
    # print("数据插入完成")
    
    content = """
**申请代码:** F0212. 数据科学与大数据计算 [P1]


"""
    # 搜索测试
    emb = await get_emb("大模型")
    client = MilvusClient(uri="./discipline.db")
    
    res = client.search(
        collection_name="second_level_disciplines",
        data=[emb],
        anns_field="embedding",
        search_params={"metric_type": "COSINE", "params": {}},
        limit=20,
        output_fields=["discipline_code", "discipline_name"]
    )

    print("搜索结果:")
    # 搜索结果是一个列表，每个元素对应一个查询的结果
    for hits in res:
        for hit in hits:
            print(f"ID: {hit.id}, 距离: {hit.distance:.4f}, 代码: {hit.entity.get('discipline_code', 'N/A')}, 名称: {hit.entity.get('discipline_name', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())