import os
import asyncio
import numpy as np
from openai import AsyncOpenAI, OpenAI
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Setup your API key and base URL
client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def get_embedding(text):
    response = await client.embeddings.create(
        model="text-embedding-v4",
        input=text
    )
    return response.data[0].embedding

async def main():
    # 1. Get embeddings for the words
    words = ["father", "male", "mother", "female"]
    embeddings = await asyncio.gather(*[get_embedding(word) for word in words])
    
    word_embeddings = {word: np.array(embedding) for word, embedding in zip(words, embeddings)}

    all_vectors = np.array(list(word_embeddings.values()))
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(all_vectors)
    
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(word_embeddings.keys()):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=12)
    
    # Draw arrows to show the relationships
    plt.arrow(reduced_vectors[1, 0], reduced_vectors[1, 1], reduced_vectors[0, 0] - reduced_vectors[1, 0], reduced_vectors[0, 1] - reduced_vectors[1, 1], 
              shape='full', lw=1, length_includes_head=True, head_width=.01, color='blue')
    plt.arrow(reduced_vectors[2, 0], reduced_vectors[2, 1], reduced_vectors[3, 0] - reduced_vectors[2, 0], reduced_vectors[3, 1] - reduced_vectors[2, 1], 
              shape='full', lw=1, length_includes_head=True, head_width=.01, color='red')

    plt.title("Mother - Female  = Father - Male")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.savefig("king_queen_analogy.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())