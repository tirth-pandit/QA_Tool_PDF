
# RAG-Based PDF Q&A Tool 
A robust Question-Answering (QA) tool built using Retrieval-Augmented Generation (RAG), OpenAI API, LangChain, and Pinecone. This tool enables users to upload a PDF document and ask questions, receiving accurate answers from the document.





## Features

- 📝 Upload and process PDF file.
- ❓ Ask natural language questions about the content of uploaded PDF.
- 📄 Retrieve accurate answers.
- ⚡ Powered by RAG combining retrieval and generation for context-aware answers.
- 💾 Persistent storage of vector embeddings using Pinecone for scalable performance.
- 🧠 Conversational Memory: Implementation of a conversational buffer to retain and reference past interactions, enabling context-aware and coherent dialogues.
## How it works 

- **PDF Processing**:
    - Extracts text from PDFs and splits it into manageable chunks.

- **Vector Embedding**:
    - Converts document chunks into vector embeddings using OpenAI's API.

- **Indexing**:
    - Stores vector embeddings in Pinecone for efficient retrieval.

- **Question-Answering**:
    - Queries Pinecone to retrieve relevant document chunks.
    - Uses OpenAI's GPT model to generate human-readable answers.
    - Generates answer depending on the retrieved context confidence scores. 

    






## Installation & Uasge

- ### **Prerequisites**
  - Python 3.8+
  - API keys for OpenAI and Pinecone
  - Conda Environment (recommended)

- ### **Steps** 
  - Clone the repository:
    ```
    git clone https://github.com/your-username/rag-pdf-qa-tool.git
    cd rag-pdf-qa-tool
    ```
  - Install dependencies:
    ```
    conda env create -f environment.yaml -n NewEnvName
    conda activate NewEnvName
    ```
  - Set up configurations:
    - Change the configurations in CONFIG.json
    - Add your api keys to openAI and Pinecone
    ```
    "OPENAI_API_KEY": 
    "PINECONE_API_KEY": 
    ...
    ```
  - Run the tool:
    ```
    python app.py
    ```

- ### **Usage**
  - Change the configurations in CONFIG.json
  - Run and intreact with the tool on CLI 
  - Give the path to PDF document.
  - Write the list of questions one by one 
  - Get the answers from tool displayed on CLI




## Configuration customization 
  - Many different customization can be made using CONFIG.json, below are details

  ```
  {
    "OPENAI":{
        "OPENAI_API_KEY": Add your openAI api key
        "MODEL": Add gpt model name to use as LLM,
        "TEMPERATURE": Temperature value to make LLM more creative,
        "MAX_TOKENS": Max number of token generation,
        "EMBEDDING_MODEL": Name of embeddings to use
    },
    
    "PINECONE":{
        "PINECONE_API_KEY": Pinecone api key,
        "PINECONE_ENV": Region name for Pinecone index,
        "DB_INDEX_NAME": Name of index,
        "CLOUD": Name of cloud service to be used,
        "TOP_K": Nuumber of retrivals to be made from vectorDB with given query,
        "HIGH_CONFIDENCE_THRESHOLD": High confidence threshold for retrieved chunk confidence value,
        "LOW_CONFIDENCE_THRESHOLD": Low confidence threshold for retrieved chunk confidence value.
    },

    "TOKENIZER":{
        "CHUNK_OVERLAP": Number of token to overlap between consicutive chunks,
        "CHUNK_SIZE": Number of token in a chunk.
    },

    "CONVERSATION_HISTORY_LENGTH": Conversational buffer size
}
  ```


## Future Improvements
 - #### **User Experiance**
   - Enhance the chatbot flow to make it more intuitive and seamless for users by employing prompt engineering, AI agents, and prebuilt chains.
   - Integrate output streaming to deliver real-time chat experiences.
   - Tailor personalized conversations by leveraging previous chats stored for specific users, combined with metadata retrieved from the corpus.
   - Deploy a user-friendly web application interface to improve accessibility and engagement.

- #### Modularization of code 
  - Separate Logic into Modules
  - Input Handling Module: For handling user inputs (e.g., PDF path, queries).
  - PDF Parsing Module: To manage PDF text extraction.
  - Embedding and Indexing Module: To handle vectorization and Pinecone indexing.
  - Query Processing Module: For query answering and response generation.
  - Configuration Module: Load and validate configurations.
  - Utilities Module: For helper functions like tiktoken_len, logging, and error handling

- #### Scalability 
   - **Cloud-Ready Design**: Containerize the application using Docker to deploy on cloud services (e.g., AWS, GCP, Azure).

   - **Asynchronous Operations**: Introduce async programming using asyncio or a task queue for embedding creation and database interactions to handle large documents more efficiently.

   - **Database Index Management**: Create a dynamic index manager for handling multiple PDFs simultaneously.Allow users to select, merge, or delete specific indices at runtime.

   - **Optimized Tokenization**: Use batch processing for embedding large chunks of text to improve efficiency. Cache embeddings for previously processed PDFs using libraries. 

    - **Confidence score-based retrieval usage**: More robust algorithms can be employed that filter or use the retrieved context from the vector database based on its confidence scores.

    - **Handling different data types**: More robust PDF scanning techniques can be used to handle various data types in PDFs, such as tables, headers, footers, etc.
   

