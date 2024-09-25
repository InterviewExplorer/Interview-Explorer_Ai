from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
ELASTICSEARCH_HOST="http://192.168.0.49:9200"
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="kookoo"

def delete_docs():
    es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}) 

def search_doc_nori(keyword):
    doc = {
    "query": {
        "match": {
        "resume": keyword
        }
    },
    }

    response = es.search(index=INDEX_NAME, body=doc)
    result=[]
    for hit in response['hits']['hits']:
        
        resume = hit['_source'].get('resume', 'No resume field')
        score = hit['_score']
        result.append({
            'resume': resume,
            'score': score,
        
    })
    
    return result

