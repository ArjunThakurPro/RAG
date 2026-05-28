from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI()
template = Jinja2Templates(directory="templates")


class DialogueInput(BaseModel):
    prompt: str


def pdf_loader():
    pdf_path = "data"
    all_docs = []
    for file_name in os.listdir(pdf_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_path, file_name)
            loader = PyPDFLoader(file_path)
            doc = loader.load()
            all_docs.extend(doc)
    return all_docs


all_doc = pdf_loader()


def text_split(doc, chunk_size=500, chunk_overlap=50):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_spliter.split_documents(doc)
    return chunks


chunk = text_split(all_doc)


class Embedding:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embedding(self, chunks):
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings


import chromadb
import uuid


class Vector_database:
    def __init__(self, persist_dir="data/vector_store", collection_name="pdf_document"):
        self.persist_directory = persist_dir
        self.collection_name = collection_name
        self.collection = None
        self.client = None

    def initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Vector Store Collection for RAG"},
        )

    def add_documents(self, document, embedding):
        if len(document) != len(embedding):
            raise ValueError(
                "length of document does not match with length of embedding"
            )
        ids = []
        all_metadata = []
        document_content = []
        embedding_list = []
        for i, (doc, embedding) in enumerate(zip(document, embedding)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)
            meta_data = dict(doc.metadata)
            meta_data["index"] = i
            meta_data["content_length"] = len(doc.page_content)
            all_metadata.append(meta_data)
            document_content.append(doc.page_content)
            embedding_list.append(embedding.tolist())
        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=document_content,
            embeddings=embedding_list,
        )


vector_manager = Vector_database()
vector_manager.initialize_store()
embedding_manager = Embedding()
text = [doc.page_content for doc in chunk]
embedding = embedding_manager.generate_embedding(text)
vector_manager.add_documents(chunk, embedding)


class RAGRetriever:
    def __init__(self, embedding_manager, vector_manager):
        self.embedding_manager = embedding_manager
        self.vector_manager = vector_manager

    def retrieve(self, query, top_k=4, vector_threshold=0.0):
        query_embeddings = embedding_manager.generate_embedding([query])[0]
        result = self.vector_manager.collection.query(
            query_embeddings=[query_embeddings.tolist()], n_results=top_k
        )
        retrived_doc = []
        if result["documents"] and result["documents"][0]:
            ids = result["ids"][0]
            metadata = result["metadatas"][0]
            documents = result["documents"][0]
            distance = result["distances"][0]
            for i, (doc_ids, metadatas, documents, distance) in enumerate(
                zip(ids, metadata, documents, distance)
            ):
                similarity_score = 1 - distance
                if similarity_score >= vector_threshold:
                    retrived_doc.append(
                        {
                            "ids": doc_ids,
                            "metadatas": metadatas,
                            "documents": documents,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )
                else:
                    print("No document found")
        return retrived_doc


retriver_manager = RAGRetriever(embedding_manager, vector_manager)
# pip install lanchain-groq
Groq_API = "xxxxxxxxxxxxxxxxxxx"
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=Groq_API, model="qwen/qwen3-32b", temperature=0.1, max_tokens=1024
)


def generate_output(query, retrieve_manager, llm, top_k=3):
    results = retriver_manager.retrieve(query)
    context = "\n".join(doc["documents"] for doc in results) if results else " "
    prompt = f""" use given context for generating answer for query
           Context:{context}
           Query : {query}"""
    response = llm.invoke(
        [prompt.format(context=context, query=query)]
    )  # it expecting  a list as prompt
    return response.content


@app.post("/generation/")
async def generate(dialogue_input: DialogueInput):
    Answer = generate_output(dialogue_input.prompt, retriver_manager, llm)
    return {"Answer": Answer}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return template.TemplateResponse("frontend.html", {"request": request})
