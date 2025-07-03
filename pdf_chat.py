from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = None
retriever = None

def load_pdf(file_path):
    global qa_chain, retriever
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    return "PDF loaded and processed."

def chat_with_pdf(query, fallback_agent=None):
    """
    Attempts to answer using PDF content first. Falls back to agent if no good PDF answer.
    """
    if not qa_chain:
        return "No PDF is currently loaded."

    pdf_answer = qa_chain.run(query).strip()

    # Heuristic to decide if fallback is needed
    if not pdf_answer or len(pdf_answer.split()) < 5 or "I need" in pdf_answer or "please provide" in pdf_answer:
        if fallback_agent:
            agent_result = fallback_agent(query)
            return agent_result.get("summary", "[No valid answer found]")
        else:
            return "Couldn't get a good answer from PDF."

    return pdf_answer
