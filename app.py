from flask import Flask, request, render_template
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Declare parsed_data globally
parsed_data = None

def resumeparser(pdfreader):
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    query = 'Name \n Email and \n mobile '
    docs = document_search.similarity_search(query)

    global parsed_data
    parsed_data = chain.run(input_documents=docs, question=query)

@app.route('/parse', methods=['POST'])
def parse_resume():
    resume_file = request.files['resume']

    # Ensure the file pointer is at the beginning
    resume_file.seek(0)

    pdfreader = PdfReader(resume_file)

    # Call resumeparser to update the global parsed_data
    resumeparser(pdfreader)

    # Print to check if parsed_data is populated
    

    return render_template('results.html', parsed_data=parsed_data)


if __name__ == '__main__':
    app.run(debug=True)
