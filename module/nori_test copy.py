from langchain_community.document_loaders import PyPDFLoader
from elasticsearch import Elasticsearch
ELASTICSEARCH_HOST="http://192.168.0.49:9200"
es = Elasticsearch([ELASTICSEARCH_HOST])
INDEX_NAME="my_korean_index"


def read_pdf(resume):
    resume_content = ""
    loader = PyPDFLoader(resume)
    document = loader.load()
    resume_content = "\n".join([page.page_content for page in document])
    return resume_content

def add_doc_nori(doc_name, doc):

    doc = {
        "content": doc,
        "source" : doc_name
    }

    
    response = es.index(index=INDEX_NAME, body=doc)

def delete_docs():
    es.delete_by_query(index=INDEX_NAME, body={"query": {"match_all": {}}}) 


def search_doc_nori(keyword):
    doc = {
    "query": {
        "match": {
        "content": keyword
        }
    },
    
    
    }
    response = es.search(index=INDEX_NAME, body=doc)
    result=[]
    for hit in response['hits']['hits']:
        content = hit['_source'].get('content', 'No content field')
        # content = hit['highlight'].get('content', 'No content field')
        source = hit['_source'].get('source', 'No source field')
        score = hit['_score']
        result.append({
        'content': content,
        'score': score,
        'source': source
    })
    
    return result
