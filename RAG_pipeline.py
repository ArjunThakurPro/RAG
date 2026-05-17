from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


# STEP 1
# Data -> Documents
def load_all_pdfs():
    folder_path = "data"
    num_docs = 0
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(pdf_path)
            doc = loader.load()
            all_docs.extend(doc)
            num_docs += 1
        # print(len(all_docs))
        # print(num_docs)
    return all_docs


doc = load_all_pdfs()


# STEP -2
# Documents -> chunks
def split_docs(doc, chunk_size=500, chunk_overlap=50):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # number of characters in every chunks
        chunk_overlap=chunk_overlap,
    )

    chunked_doc = text_spliter.split_documents(doc)
    return chunked_doc


chunk = split_docs(doc)
print(len(chunk))
# print(chunk)

# STEP 3
# chunks ->Embeding
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        print("Embedding_dimentions=", self.model.get_sentence_embedding_dimension())

    def generate_embedding(self, text):
        embeddings = self.model.encode(text, show_progress_bar=True)
        print("embedding_shape:", embeddings.shape)
        return embeddings


embedding_manager = EmbeddingManager()
# STEP -4
# Embedding -> Vector DB
import chromadb
import uuid  # uniniquely indentify


class VectorStoreDatabase:
    def __init__(
        self, persist_directory="data/vector_store", collection_name="pdf_document"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection = None
        self.client = None

    def initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        # create a client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        # creating collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"discription": "vector store collection of embedding for RAG"},
        )
        print("vector collection: ", self.collection)
        print("doc in collection: ", self.collection.count())

    def add_documents(self, document, embeddings):
        if len(document) != len(embeddings):
            raise ValueError("num of doc does not match num of embeddings")
        ids = []
        all_metadata = []
        documnent_content = []
        embedding_list = []
        for i, (doc, embedding) in enumerate(
            zip(document, embeddings)
        ):  # (0, ("Doc1", [0.1, 0.2]))
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)
            meta_data = dict(doc.metadata)
            meta_data["index"] = i
            meta_data["content_length"] = len(doc.page_content)
            all_metadata.append(meta_data)
            documnent_content.append(doc.page_content)
            embedding_list.append(embedding.tolist())
        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documnent_content,
            embeddings=embedding_list,
        )
        print("total Document added in Vectore Store: ", len(documnent_content))
        print("Docs in collection: ", self.collection.count())


VCM = VectorStoreDatabase()
VCM.initialize_store()
text = [doc.page_content for doc in chunk]
embedding = embedding_manager.generate_embedding(text)
VCM.add_documents(chunk, embedding)
from sklearn.metrics.pairwise import cosine_similarity


class RAGRetriever:
    def __init__(self, embedding_manager, VCM):
        self.vector_manager = embedding_manager
        self.Vector_store = VCM

    def retrieve(self, query, top_k=5, vector_threshold=0.0):
        query_embedding = self.vector_manager.generate_embedding([query])[0]
        # symantic Search
        result = self.Vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )
        retrived_docs = []
        if result["documents"] and result["documents"][0]:
            ids = result["ids"][0]
            metadatas = result["metadatas"][0]
            documents = result["documents"][0]
            distances = result["distances"][0]
            for i, (doc_id, metadata, document, distance) in enumerate(
                zip(ids, metadatas, documents, distances)
            ):
                similarity_score = 1 - distance
                if similarity_score >= vector_threshold:
                    retrived_docs.append(
                        {
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )
            print(f"retrived {len(retrived_docs)} documents")
        else:
            print("no documents found")
        return retrived_docs


rag_retriever = RAGRetriever(embedding_manager, VCM)
print(rag_retriever.retrieve("what is RAG"))
