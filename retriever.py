import re
import dspy
from datetime import datetime
from typing import List, Union, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    entries = content.split('Date:')
    entries = [entry.strip() for entry in entries if entry.strip()]
    dates = []
    strings = []
    for entry in entries:
        date_match = re.match(r'(\d{2}-\d{2}-\d{4})', entry)
        if date_match:
            date = date_match.group(1)
            text = f"Date: {entry}"
            dates.append(date)
            strings.append(text)
    documents = [Document(page_content=text, metadata={"date": date}) for text, date in zip(strings, dates)]
    return documents


def setup_db():
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_hf = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    documents = load_documents('project_diary.txt')
    faiss_db = FAISS.from_documents(documents, embeddings_hf)
    return faiss_db


def add_entry_to_diary(entry, filename='project_diary.txt', db=setup_db()):
    date_str = datetime.now().strftime("%d-%m-%Y")
    new_entry_content = f"Date: {date_str}. {entry}\n"

    with open(filename, 'r') as file:
        content = file.read()

    if f"Date: {date_str}" in content:
        updated_content = re.sub(rf"(Date: {date_str}.*?)(?=\nDate:|$)", f"\\1\n{entry}", content,
                                 flags=re.DOTALL | re.MULTILINE)
    else:
        updated_content = content + new_entry_content

    with open(filename, 'w') as file:
        file.write(updated_content)

    new_document = Document(page_content=new_entry_content.strip(), metadata={"date": date_str})
    db.add_documents([new_document])


class DSPythonicRMClient(dspy.Retrieve):
    def __init__(self, k: int = 3, db=setup_db()):
        super().__init__(k=k)
        self.db = db

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        k = k if k else self.k

        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries

        results = []
        for query in queries:
            retrieved_docs = self.db.similarity_search(query, k=k)
            passages = [doc.page_content for doc in retrieved_docs]
            results.append(passages)

        return dspy.Prediction(
            passages=results if len(results) > 1 else results[0]
        )


