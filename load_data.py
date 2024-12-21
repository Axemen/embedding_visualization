import os

import numpy as np
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"), server_api=ServerApi("1"))

db_name = "sample_mflix"
collection_name = "embedded_movies"

collection = client[db_name][collection_name]

pipeline = [
    {"$match": {"plot_embedding": {"$exists": 1}, "title": {"$exists": 1}, "genres": {"$exists": 1}}},
    {"$sample": {"size": 300}},
    {"$project": {"plot_embedding": 1, "title": 1, "genres": 1}},
]


embeddings = []
titles = []
genres = []
for doc in collection.aggregate(pipeline):
    embeddings.append(doc["plot_embedding"])
    titles.append(doc["title"])
    genres.append(doc["genres"])

embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)
pd.DataFrame({"titles": titles, "genres": genres}).to_csv("metadata.csv")

