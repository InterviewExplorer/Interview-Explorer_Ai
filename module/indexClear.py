from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
ELASTICSEARCH_HOST="http://192.168.0.49:9200"
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="pdf_array"

def delete_docs():
    es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}) 
    es.delete_by_query(index="fasttext_search", body={"query": {"match_all": {}}}) 


# ---------------------------------------------------------------------------