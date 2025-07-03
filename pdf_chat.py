from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from main import run_research_agent

qa_chain = None
pdf_qa_chain = None

def load_pdf(file_path):
    global qa_chain
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    global pdf_qa_chain
    pdf_qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)
    return "✅ PDF loaded and processed."

def chat_with_pdf(query):
    global pdf_qa_chain
    if not pdf_qa_chain:
        return "❌ No PDF loaded yet."

    # Step 1: Try answering from the PDF
    pdf_answer = pdf_qa_chain.run(query)

    # Step 2: Check if the answer is too vague or unhelpful
    if not pdf_answer or len(pdf_answer.strip()) < 30:
        agent_result = run_research_agent(query)
        return f"{agent_result['summary']} (From tools)"

    # PDF answer seems useful — try both
    agent_result = run_research_agent(query)
    return f"{pdf_answer}\n\n🔍 Additional info:\n{agent_result['summary']}"

    return pdf_answer
