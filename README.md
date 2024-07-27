



# Document Summarizer




https://github.com/user-attachments/assets/c530ba07-ba32-45a0-9144-96b55ef8d193







<details>
<summary>Table of Contents</summary>

- [Document Summarizer](#document-summarizer)
- [About The Project](#about-the-project)
- [Project Pipeline](#project-pipeline)
  - [Pre-Processing](#pre-processing)
  - [Summarization](#summarization)
  - [Question-Answering](#question-answering)
- [ADD-ONS](#add-ons)
- [Usage Instructions and Demo](#usage-instructions-and-demo)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

</details>

## :star2: About The Project

This project was created for IITISOC 24'. The goal of this project was to develop a robust pipeline that generates concise textual summaries of various types of documents. The documents can be research papers, novels, invoices, etc., and will contain text and images. Additionally, a Question and Answer (QnA) is integrated to  allow users to ask questions about the document and receive precise answers.

## :triangular_ruler: Project Pipeline


### :gear: Pre-Processing

#### 1. Text Extraction

- PDF Documents: Text and images are extracted from PDF files using the fitz library.
- DOCX Documents: Text and images are extracted from DOCX files using the python-docx library.


#### 2. Image Caption Generation
- The pipeline uses a pre-trained model [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) to generate detailed captions for images found within the documents.


#### 3. Text Chunking

- The extracted text is divided into smaller chunks for efficient processing by summarization models. The chunking function ensures each chunk stays within a maximum length and maintains context of previous paragraphs through overlapping sections.

### :memo: Summarization
The project allows for summary generation using either the BART model or the Llama 3.1 model.

#### 1. BART Model
- We used a [BART](https://huggingface.co/facebook/bart-large-cnn) (Bidirectional and Auto-Regressive Transformers) model which we fine-tuned on [BOOKSUM](https://huggingface.co/datasets/kmfoda/booksum) dataset.

#### 2. Llama Model
- The [Llama 3.1](https://llama.meta.com) model is used for high-quality summarization, tailored to our specific tasks for improved performance.

### :question: Question-Answering
The question answering (QnA) part of the pipeline enables users to ask specific questions about a document and receive detailed and accurate responses.

#### 1. Ollama Mistral Model

- The [Ollama Mistral](https://ollama.com/library/mistral) model offers advanced language understanding and generates precise responses to user queries based on the document.

#### 2. Text Embedding
- Text is converted into dense vectors using the [mxbai-embed](https://ollama.com/library/mxbai-embed-large) model, capturing semantic meaning for better model understanding.
- Vectors are stored in a Chroma database for fast and easy retrieval of relevant document chunks.
#### 3. Answer Generation
- The model generates multiple versions of questions to improve relevant retrieval from the vector database.
- It then formats the retrieved context and the user’s question to produce a clear, concise answer.




### :rocket: ADD-ONS

#### 1. Mind-Map
- Visualizes hierarchical information from text data as a mind map using Matplotlib.
- Offers customizable layouts based on summary size for clear and organized data representation.

#### 2. Invoice Summarization
- After extracting key details from receipts, it generates a concise summary, including merchant name, date, total amount, and item count.
#### 3. Audio 
- Converts text into spoken audio using the gTTS library, enabling text-to-speech functionality. 
#### 4. Language Tranlation
- Translates text between languages using the deep-translator library.






## :running: Set-Up Model Locally

### :desktop_computer: System Requirements

- Operating System: Windows , macOS, or Linux
- RAM: At least 16 GB CPU + 6 GB GPU or 32 GB CPU
- Disk Space: At least 18 GB free space

### :hammer_and_wrench: Conda Enviromenet Set-Up
- First create new anaconda enviornment. Open Anaconda terminal and write:
```bash
  conda create -n document_summarizer python=3.10 
```

- To active your new environment type:
```bash
  conda activate document_summarizer 
```
- To enable CUDA for GPU on this system, [click here](https://pytorch.org/get-started/locally/)

### :inbox_tray: Git Clone Repo
To Download the repo on your system, type
```bash
   git clone (repo here)
```
### :package: Downloading Ollama Models
- Download ollama from [here](https://ollama.com/)
- To Run Ollama:
```bash
   ollama serve
```
- To Download Models:
 ```bash
   ollama serve & run mxbai-embed-large
   ollama serve & run mistral
   ollama serve & run llama3.1
```
-  Check for Downloaded Models:
```bash
   ollama list
```
### :books: Imports 
First activate your environment and locate your dowloaded repo and then type:
```bash
   pip install -r requirements.txt
```

### :runner: Run model on Stream-Lit
```bash
   streamlit run streamlit.py
```





## 🔗 Authors

 #### Arihant Jain

 [![github](https://img.shields.io/badge/github-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Arihant779)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arihant-jain-a946962a6/)

#### Arunav Sameer

 [![github](https://img.shields.io/badge/github-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/arunavsameer)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arunav-sameer-012129230/)

#### Anmol Joshi

 [![github](https://img.shields.io/badge/github-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Anmol-Joshi004)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anmol-joshi-64a041286/)


#### Tanishq Godha 

 [![github](https://img.shields.io/badge/github-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Tanishq-Godha)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tanishq-godha-08a7a12b0/)








## :heart: Acknowledgements

We are grateful to the following individuals and organizations for their invaluable contributions and support throughout this project:

- **Organising Team of IITISOC 24**: For providing the opportunity, and platform to develop this project. 
- **Cynaptics Club, IIT Indore**: For designing high-quality problem statements that challenged and inspired us.
- **Our Mentor, Soham Pandit**: For his unwavering support, guidance, and mentorship, which were crucial to the project's success.



