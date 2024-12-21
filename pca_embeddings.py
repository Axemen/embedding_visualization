import os

import numpy as np
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.decomposition import PCA

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"), server_api=ServerApi("1"))

db_name = "sample_mflix"
collection_name = "embedded_movies"

collection = client[db_name][collection_name]

pipeline = [
    {"$match": {"plot_embedding": {"$exists": 1}, "title": {"$exists": 1}}},
    {"$sample": {"size": 100}},
    {"$project": {"plot_embedding": 1, "title": 1}},
]


embeddings = []
titles = []
for doc in collection.aggregate(pipeline):
    embeddings.append(doc["plot_embedding"])
    titles.append(doc["title"])

embeddings = np.array(embeddings)
titles = np.array(titles)

pca = PCA(n_components=3).fit(embeddings)

output = pca.transform(embeddings)

np.save("pcad_embeddings.npy", output)
np.save("titles.npy", titles)
