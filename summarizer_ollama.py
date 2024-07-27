from langchain_community.llms import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

def file_preprocessing(file, start, end):
    loader =  PyPDFLoader(file)
    pages = loader.load()
    pages = pages[start - 1:end]
    return pages

def generate_summary_llama3_1(filepath, start_page, end_page, detailed):
    data = file_preprocessing(filepath, start_page, end_page)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="llama3.1",show_progress=True),
        collection_name="local-rag"
    )
    local_model = "llama3.1"
    llm = ChatOllama(model=local_model)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    template = """Answer the question in points based ONLY on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    strength = "detail" if detailed else "brief"
    strength_1 = "topic-wise" if detailed else "as a whole"
    que = f"""
    write the good summary in {strength} of the whole document without repititons {strength_1} .
    """
    summary = chain.invoke(que)
    vector_db.delete_collection()
    return summary
    
