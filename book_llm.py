from decouple import config
import os
os.environ['OPENAI_API_KEY'] = config("OPENAI_SECRET_KEY")

from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone

loader = UnstructuredPDFLoader("./books/advances-in-financial-machine-learning.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key= config("OPENAI_SECRET_KEY"))

pinecone.init(
    api_key= config("PINECONE_API_KEY"),
    environment= config("PINECONE_API_ENV")
)
index_name = "advances-in-financial-machine-learning"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embedding=embeddings, index_name=index_name)

query = "Explain meta-labelling from Chapter 3"
docs = docsearch.similarity_search(query)

llm = OpenAI(temperature= 0.7, openai_api_key= config("OPENAI_SECRET_KEY"))
chain = load_qa_chain(llm, chain_type="stuff")
result = chain.run(input_documents=docs, question=query)