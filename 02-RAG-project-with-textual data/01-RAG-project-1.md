### **1️⃣ Install Required Libraries**
```python
!pip install -qU langchain-pinecone langchain-google-genai
```
🔹 This command installs two Python libraries:  
   - **`langchain-pinecone`** (for connecting LangChain with Pinecone).  
   - **`langchain-google-genai`** (for using Google Generative AI models).  

---

### **2️⃣ Connect to Pinecone**
```python
from pinecone import Pinecone, ServerlessSpec
from google.colab import userdata

pinecone_api_key = userdata.get('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)
```
🔹 **What’s happening here?**  
   - Imports Pinecone and Google Colab’s `userdata` module (for securely retrieving API keys).  
   - Retrieves the **Pinecone API Key** stored in Colab.  
   - Connects to **Pinecone** using this API key.  

---

### **3️⃣ Create a Pinecone Index**
```python
index_name = "rag-project-1"  # change if desired

pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(index_name)
```
🔹 **What’s happening here?**  
   - Creates a **Pinecone index** named `"rag-project-1"` (you can change this).  
   - **`dimension=768`**: This means the embeddings used have 768 dimensions (common for AI models).  
   - **`metric="cosine"`**: Uses cosine similarity for finding similar embeddings.  
   - Connects to this index so data can be stored and retrieved.  

---

### **4️⃣ Set Up Google Generative AI Embeddings**
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)
```
🔹 **What’s happening here?**  
   - Imports Google’s **Generative AI Embeddings** model.  
   - Retrieves the **Google API Key** securely from Colab.  
   - Loads the **embedding model** (`"models/embedding-001"`) to convert text into vector representations.  

---

### **5️⃣ Store Embeddings in Pinecone**
```python
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
```
🔹 **What’s happening here?**  
   - Connects **LangChain’s vector store** with **Pinecone**.  
   - This allows the system to **store** and **retrieve** embeddings for AI-powered search and retrieval.  

---

## **🔹 Summary:**
This notebook is **setting up a RAG system** using:  
✅ **Pinecone** to store and retrieve AI-generated embeddings.  
✅ **Google Generative AI** to generate text embeddings.  
✅ **LangChain** to connect these services seamlessly.  

---

## **6️⃣ Importing Required Libraries**
```python
from uuid import uuid4
from langchain_core.documents import Document
```
🔹 **What’s happening here?**  
   - **`uuid4()`**: Generates unique IDs for documents.  
   - **`Document`**: A class from **LangChain** used to structure data (text + metadata).  

---

## **7️⃣ Creating Sample Documents**
```python
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

document_11 = Document(
    page_content="Hello, My name is Mohsin Raza, I am a Student of piaic, My favorite subject is programming,and we are trying to make an agent.",
    metadata={"source": "tweet"},

)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    document_11,
]

```
🔹 **What’s happening here?**  
   - **Creates 11 documents**, each containing:  
     ✅ **Text Content** (e.g., news, tweets, website reviews)  
     ✅ **Metadata** (e.g., `"source": "tweet"`, `"source": "news"`)  
   - Your name (**Mohsin Raza**) is also included in one of the documents! 🎉  

---

## **8️⃣ Storing Documents in Pinecone**
```python
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```
🔹 **What’s happening here?**  
   - Generates a **unique ID** for each document.  
   - **Stores** the documents inside **Pinecone** as vector embeddings.  
   - This allows **fast retrieval** based on similarity search.  

---

## **9️⃣ Importing Google AI Chat Model**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
```
🔹 **What’s happening here?**  
   - Imports **Google Gemini**'s **chat model** for generating AI responses.  

---

## **🔟 Setting Up the AI Chat Model**
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_output_tokens=1024,
    top_k=40,
    top_p=0.95,
)
```
🔹 **What’s happening here?**  
   - **`model="gemini-2.0-flash-exp"`** → Uses **Google Gemini 2.0** for AI-generated text.  
   - **`temperature=0.7`** → Controls randomness (higher = more creative).  
   - **`max_output_tokens=1024`** → Limits response length.  
   - **`top_k=40, top_p=0.95`** → Tweaks response diversity.  

---

## **📌 Summary of the Notebook**
1️⃣ Installs **LangChain, Pinecone, and Google Generative AI**.  
2️⃣ Connects to **Pinecone** (a vector database).  
3️⃣ Creates an **AI-powered document storage system**.  
4️⃣ **Stores documents** (e.g., tweets, news, websites).  
5️⃣ Uses **Google Gemini AI** to process and generate responses.  

🔹 **This notebook is building a Retrieval-Augmented Generation (RAG) system!** 🚀  
🔹 **It can retrieve relevant information from documents & generate responses using AI.**  

---

## **🔹 11️⃣ Defining a Function to Answer User Queries**
```python
def answer_to_user(question: str) -> str:
    vector_results = vector_store.similarity_search(question, k=1)

    print(len(vector_results))

    final_answer = llm.invoke(f"""Answer the question: {question}, 
    Here are some references to answer: {vector_results} """)

    return final_answer
```
### **What’s happening here?**
✅ **`answer_to_user(question: str) -> str`**  
   - A function that takes a **user's question** and returns an **AI-generated answer**.  

✅ **`vector_store.similarity_search(question, k=1)`**  
   - Searches for **the most relevant document** in the Pinecone database.  
   - **`k=1`** → Retrieves **only the closest matching document**.  

✅ **`llm.invoke(f"... {vector_results} ...")`**  
   - Uses **Google Gemini AI** to **generate an answer** based on the retrieved document.  

---

## **🔹 12️⃣ Asking a Question**
```python
answer = answer_to_user("who is Mohsin Raza?")
```
🔹 **What’s happening here?**  
   - The function **searches the stored documents** for relevant information about `"Mohsin Raza"`.  
   - It then **uses AI** to generate a response.  

---

## **🔹 13️⃣ Printing the AI Response**
```python
print(answer.content)
```
🔹 **What’s happening here?**  
   - Displays the AI-generated **answer** in the notebook.  

---

## **📌 Final Summary**
1️⃣ **Stores text documents in Pinecone**.  
2️⃣ **Searches for relevant documents** using a query.  
3️⃣ **Uses Google Gemini AI** to generate an answer based on the retrieved documents.  
4️⃣ **Prints the AI-generated response**.  

This means your notebook is a **fully functional AI-powered chatbot** that can search stored knowledge and answer user queries! 🚀  

