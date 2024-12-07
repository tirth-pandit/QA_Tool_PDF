from uuid import uuid4
from typing import List, Dict, Tuple
import argparse 
import os 
import json 

from PyPDF2 import PdfReader 
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
import tiktoken
import matplotlib.pyplot as plt 
from langchain_openai import ChatOpenAI

class QATool: 
    def __init__(self, pdf_path: str):
        self.config = self.get_config() 
        self.pdf_path = pdf_path 
        self.conversation_history = []

        # LLM and Embeddings
        self.embeddings = OpenAIEmbeddings(model=self.config['OPENAI']['EMBEDDING_MODEL']) 
        self.llm = ChatOpenAI(model_name=self.config['OPENAI']['MODEL'], temperature=self.config['OPENAI']['TEMPERATURE'], max_tokens=self.config['OPENAI']['MAX_TOKENS']) 

        # Read PDF and Create Chunks
        self.raw_text = self.read_pdf(self.pdf_path)

        # Create Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=self.config['TOKENIZER']['CHUNK_SIZE'], 
                                chunk_overlap=self.config['TOKENIZER']['CHUNK_OVERLAP'],
                                length_function=self.tiktoken_len,
                                separators=["\n\n", "\n", " ", ""]
                            ) 
          
        # Create Chunks and Metadata
        self.chunks, self.metadata = self.create_chunks_metadata() 
        self.chunk_embs = self.embeddings.embed_documents(self.chunks)


        # Create Pinecone Index
        self.index = self.create_pinecone_index()

    def get_config(self)->Dict:
        with open("CONFIG.json", "r") as f:
            config = json.load(f)
        os.environ['OPENAI_API_KEY'] = config['OPENAI']['OPENAI_API_KEY']
        return config 

    def tiktoken_len(self, text: str)->int:
        tokenizer = tiktoken.get_encoding('cl100k_base')
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
        
    def read_pdf(self, pdf_path: str)->List[str]:
        print('Reading PDF...') 
        reader = PdfReader(pdf_path) 
        raw_text = [page.extract_text() for page in reader.pages]
        return raw_text 

    def create_chunks_metadata(self)->Tuple[List[str], List[Dict]]:
        print("Creating Chunks and Metadata...")
        chunks = []
        metadata = []

        for idx, page_text in enumerate(self.raw_text):
            page_chunks = self.text_splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(page_chunks):
                chunks.append(chunk)
                metadata.append({"page": idx + 1,"chunk_idx": chunk_idx, "chunk": chunk})
            
        return chunks, metadata

    def create_pinecone_index(self)->Pinecone.Index:
        print("Creating Vector store..")
        index_name = self.config['PINECONE']['DB_INDEX_NAME']
        pc = Pinecone(api_key=self.config['PINECONE']['PINECONE_API_KEY'])
        
        existing_index = [index['name'] for index in pc.list_indexes()] 
        spec = ServerlessSpec(cloud=self.config['PINECONE']['CLOUD'], region=self.config['PINECONE']['PINECONE_ENV'])

        if index_name not in existing_index:
            pc.create_index(index_name, spec=spec, dimension=len(self.chunk_embs[0]), metric='dotproduct')
            index = pc.Index(index_name)
            ids = [ str(uuid4()) for _ in range(len(self.chunk_embs))]
            index.upsert(vectors=zip(ids, self.chunk_embs, self.metadata), namespace=index_name)
            print(f"Index {index_name} created successfully, Populating vector store with data ...")
        else:
            index = pc.Index(index_name)
            print(f"Index {index_name} already exists, please check or delete the index before proceeding with the QA Tool")
        return index 

    def answer_query(self, query: str)->str:
        query_emb = self.embeddings.embed_query(query) 
        results = self.index.query(vector=query_emb, top_k=self.config['PINECONE']['TOP_K'], include_values=True, include_metadata=True, namespace=self.config['PINECONE']['DB_INDEX_NAME'])

        confidence_scores = [match['score'] for match in results['matches']]
        average_confidence = sum(confidence_scores) / len(confidence_scores)

        if average_confidence >= self.config['PINECONE']['HIGH_CONFIDENCE_THRESHOLD']:
            context_reliability = "Context is reliable, Use the context provided to answer the question confidently."
        elif average_confidence >= self.config['PINECONE']['LOW_CONFIDENCE_THRESHOLD']:
            context_reliability = ("The context may be incomplete or less reliable. Use it cautiously, "
                                    "and indicate areas where certainty is low.")
        else:
            context_reliability = "The context is unreliable. Just say that you don't know the answer."

        retrived_context = "\n".join([matches['metadata']['chunk'] for matches in results['matches']])
        conversation_history = "\n".join(self.conversation_history)

        prompt = f"""
            Answer the question based on the provided context and context reliability, also consider the conversation history.
            Context Reliability: {context_reliability}
            Context: {retrived_context}

            Conversation History: {conversation_history}

            Question: {query}

            Answer:
            """ 
        
        response = self.llm.invoke(prompt)
        self.conversation_history.append(f"\nUser: {query}\nAI: {response.content}")
        self.conversation_history = self.conversation_history[-self.config['CONVERSATION_HISTORY_LENGTH'] : ] 
        
        return response.content  

    def delete_pinecone_index(self):
        pc = Pinecone(api_key=self.config['PINECONE']['PINECONE_API_KEY'])
        pc.delete_index(self.config['PINECONE']['DB_INDEX_NAME'])
        print(f"Index {self.config['PINECONE']['DB_INDEX_NAME']} deleted successfully")