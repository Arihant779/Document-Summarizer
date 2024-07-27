import streamlit as st
import fitz
from summarizer_BART import preprocessing_pipeline_pdf, preprocessing_pipeline_docx, pdf_doc, make_mind_map, make_points, make_points2
from summarizer_ollama import generate_summary_llama3_1
from QA import generate_answer
from Invoice import  invoice_summary, invoice_qna
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import subprocess
from gtts import gTTS
from deep_translator import GoogleTranslator                

# Initialize session state if not already done
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'view' not in st.session_state:
    st.session_state.view = 'summary'

def count_pdf_pages(pdf_path):
    try:
        document = fitz.open(pdf_path)
        num_pages = document.page_count
        document.close()
        return num_pages
    except Exception as e:
        print(f"Error: {e}")
        return None

def text_to_pdf(input_text, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    
    text = input_text.splitlines()  
    
    y_start = 750  
    line_height = 15  
    
    
    for line in text:
        c.drawString(100, y_start, line)  
        y_start -= line_height  
    c.save()
    
def text_to_pdf_invoice(text, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)  
    y_start = 750  
    line_height = 15    
    for line in text:
        c.drawString(100, y_start, line)  
        y_start -= line_height  
    c.save()

def convert_docx_to_pdf(docx_path, pdf_path):
    subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', docx_path, '--outdir', pdf_path])

def text_to_speech(text,lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("data/audio/output_audio.mp3")

def translate_text(input_text,lang):
    translator = GoogleTranslator(source='auto', target=lang)
    translation = translator.translate(input_text)
    return translation
  
  
def main():
    root_dir = "/run/media/arunav/Data/programming/Cynaptics/Document-Summarizer/streamlit/"
    st.title("Document Summarizer")

    # Upload file
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        filepath = "/run/media/arunav/Data/programming/Cynaptics/Document-Summarizer/testing/data/" + uploaded_file.name
        file_type = (uploaded_file.name[-3:])
        with open(filepath, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())                           
        if file_type == "ocx":
            docx_path = filepath
            pdf_output_path = "data/doc2pdf/"  # Specify the output directory
            convert_docx_to_pdf(docx_path, pdf_output_path)
            filepath = pdf_output_path + "Document1.pdf"
          
    
    #print(f"PDF generated successfully: {output_file}") 
    if st.session_state.uploaded_file is not None:
        st.write("File uploaded: ", st.session_state.uploaded_file.name)
        
        col1, col2, col3, col4 = st.columns(4)
        # Buttons for different views
        with col1:
            summary_button = st.button("Summary")
        with col2:
            mind_map_button = st.button("Mind Map")
        with col3:
            qa_button = st.button("Q&A")
        with col4:
            invoice_button = st.button("Invoice")
        
        if summary_button:
            st.session_state.view = 'summary'
        if mind_map_button:
            st.session_state.view = 'mind_map'
        if qa_button:
            st.session_state.view = 'qa'
        if invoice_button:
            st.session_state.view = 'invoice'
        
        
        if st.session_state.view == 'summary':
            total_pages = count_pdf_pages(filepath)
            values = st.slider("Select a range of page values", min_value= 1,max_value= total_pages, value=(1, total_pages), step=1) if total_pages > 1 else (1, 1)
            model_choice = st.radio("Choose Model", options=['llama3.1', 'BART'])
            language = st.selectbox("Language", ("English", "Hindi", "Spanish", "German", "French"))
            detailed = st.toggle("Detailed Summary")
            generate = st.button("Generate")
            if generate:
                display_summary(filepath, values, model_choice, language, detailed)
        elif st.session_state.view == 'mind_map':
            total_pages = count_pdf_pages(filepath)
            values = st.slider("Select a range of page values", min_value= 1,max_value= total_pages, value=(1, total_pages), step=1) if total_pages > 1 else (1, 1)
            option = st.radio("Choose Model",options=["llama3.1", "BART"])
            generate = st.button("Generate")
            if generate:
                display_mind_map(filepath, values, option)
        elif st.session_state.view == 'qa':
            display_qa(filepath)
        elif st.session_state.view == 'invoice':
            display_invoice(filepath)

def display_summary(filepath, values, model_choice, language, detailed):
    loading_placeholder = st.empty()
    loading_placeholder.write("Generating...")
    if language == "English":
        lang = "en"
    elif language == "Hindi":
        lang = "hi"
    elif language == "Spanish":
        lang = "es"
    elif language == "German":
        lang = "de"
    elif language == "French":
        lang = "fr"
    
    if model_choice == 'llama3.1':
        extracted_text = pdf_doc(filepath, values[0], values[1])
        input_text = " ".join(extracted_text)
        new_filepath ="data/gen_image_text.pdf"
        text_to_pdf(input_text, new_filepath) 
        summary = generate_summary_llama3_1(new_filepath, values[0], values[1], detailed)
        if lang != "en":
            summary = translate_text(summary, lang)
        loading_placeholder.empty()
        st.write(summary)
        audio_placeholder = st.empty()
        audio_placeholder.write("Generating Audio...") 
        text_to_speech(summary, lang)
        audio_placeholder.empty()
        st.audio("data/audio/output_audio.mp3", format="audio/mpeg")
    if model_choice == 'BART':
        summary = preprocessing_pipeline_pdf(filepath, values[0], values[1], detailed)
        if lang != "en":
            summary = translate_text(summary, lang)
        loading_placeholder.empty()
        st.write(summary)
        audio_placeholder = st.empty()
        audio_placeholder.write("Generating Audio...")
        text_to_speech(summary, lang)
        audio_placeholder.empty()
        st.audio("data/audio/output_audio.mp3", format="audio/mpeg")

def display_mind_map(filepath, values, option):
    # extracted_text = pdf_doc(filepath, values[0], values[1])
    # input_text = " ".join(extracted_text)
    loading_placeholder = st.empty()
    loading_placeholder.write("Generating...")
    if option == "llama3.1":
        summary = generate_summary_llama3_1(filepath, values[0], values[1], 1)
        points, titles = make_points2(summary)
    elif option == "BART":
        summary = preprocessing_pipeline_pdf(filepath, values[0], values[1], 1)
        points,titles=make_points(summary)
        
    make_mind_map(summary,points,titles)
    plt.draw()  # Ensure the plot is updated
    plt.savefig('data/images/my_plot1.png', bbox_inches='tight', pad_inches=0.5)
    loading_placeholder.empty()
    st.image('data/images/my_plot1.png', output_format='PNG')

def display_qa(filepath):
    Question = st.text_input("Ask Question")
    generate = st.button("Generate Answer")
    if generate:
        loading_placeholder = st.empty()
        loading_placeholder.write("Generating Answer...")
        st.write("Question: ", Question)
        extracted_text = pdf_doc(filepath, 0, count_pdf_pages(filepath))
        input_text = " ".join(extracted_text)
        new_filepath ="data/gen_image_text.pdf"
        text_to_pdf(input_text, new_filepath) 
        answer = generate_answer(new_filepath, Question)
        loading_placeholder.empty()
        st.write(answer)

def display_invoice(filepath):
    option = st.radio("Choose Option", options=["summary", "QnA"])
    if option == "summary":
        generate = st.button("Generate")
        if generate:
            summary = invoice_summary(filepath)
            st.write(summary)
    elif option == "QnA":    
        new_filepath =  "data/invoice/gen_invoice.pdf"
        input_text = invoice_qna(filepath)
        text_to_pdf(input_text, new_filepath)
        display_qa(new_filepath)    
    
if __name__ == "__main__":
    main()
