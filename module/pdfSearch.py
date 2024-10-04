import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import asyncio

# Elasticsearch 설정
ELASTICSEARCH_HOST = os.getenv("elastic")
INDEX_NAME = 'pdf_array2'
es = Elasticsearch([ELASTICSEARCH_HOST])

async def search_resume_info(source_value):
    print(f"Searching for source value: {source_value}")

    keywords = ["name", "date_of_birth", "technical_skills", "work_experience", "number_of_projects", "project_description", "summary_keywords"]
    results = {"source": source_value}

    for keyword in keywords:
        query = {
            "bool": {
                "must": [
                    {"term": {"source": source_value}},
                    {"term": {"key": keyword}}
                ]
            }
        }

        response = es.search(
            index=INDEX_NAME,
            body={
                "query": query,
                "_source": ["key", "value"],
                "size": 1
            }
        )

        hits = response['hits']['hits']
        if hits:
            key = hits[0]['_source']['key']
            value = hits[0]['_source']['value']
            results[key] = value
        else:
            print(f"No hits found for keyword: {keyword}")

    return results

async def search(source):
    results = await search_resume_info(source)
    return results
