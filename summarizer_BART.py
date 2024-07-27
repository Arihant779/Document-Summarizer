# Imports
import io
import re
import fitz
import torch
from PIL import Image
import matplotlib.pyplot as plt
from rake_nltk import Rake
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

# Loading Models
bart_tokenizer = AutoTokenizer.from_pretrained("Arihant29/bart_summarize_tokenizer")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("Arihant29/bart_summarize").to(device)
florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(cpu_device)
florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# Pre-processing Functions

def generate_image_caption(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(cpu_device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"].to(cpu_device),
        pixel_values=inputs["pixel_values"].to(cpu_device),
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    caption = parsed_answer["<DETAILED_CAPTION>"]
    return caption

def pdf_doc(pdf_document, start, end):
    doc = fitz.open(pdf_document)

    page_text_list = []
    for page_number in range(start-1, end):
        page = doc.load_page(page_number)
        image_list = page.get_images(full=True)

        page_text = page.get_text()
        page_text_combined = page_text

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            prompt = "<DETAILED_CAPTION>"
            generated_caption = generate_image_caption(image, prompt)

            combined_text = f"\nImage here with Caption: {generated_caption}\n"
            page_text_combined += combined_text

        page_text_list.append(page_text_combined)
    return page_text_list

def docx_doc(docx_document):
    doc = Document(docx_document)
    doc_text_list = []
    for para in doc.paragraphs:
        doc_text_list.append(para.text)

        for run in para.runs:
            if run.element.xml.find("pic:blipFill") != -1:
                image = run.element.xpath(".//a:blip")[0]
                image_data = image.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                image_part = doc.part.related_parts[image_data]
                image_bytes = image_part.blob

                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                prompt = "<DETAILED_CAPTION>"
                generated_caption = generate_image_caption(image, prompt)
                combined_text = f"\nImage here with Caption: {generated_caption}\n"
                doc_text_list.append(combined_text)
    return doc_text_list

def break_in_chunks(text, max_chunk, overlap):
    sentences = re.findall(r'.+?\.', text)
    paras = []
    curr = ""
    i = 0
    while i < len(sentences):
        if len(curr) + len(sentences[i]) > max_chunk:
            paras.append(curr)
            curr = ""

            for j in range(i, 0, -1):
                if len(curr) + len(sentences[j]) > overlap:
                    i = j - 1
                    curr = ""
                    break
                else:
                    curr += sentences[j]
        else:
            curr += sentences[i]
        i += 1
    if len(text) < max_chunk:
        paras.append(text)
    return paras

def generate_summary(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=1024).to(device)
    summary_ids = model.generate(inputs.input_ids.to(device), max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def preprocessing_pipeline_pdf(pdf_filepath, start_page, end_page, detailed):
    chunk_size = 512 if detailed else 1024
    overlap = 50 if detailed else 300
    extracted_text = pdf_doc(pdf_filepath, start_page, end_page)
    text_in_chunks = break_in_chunks(" ".join(extracted_text), chunk_size, overlap)
    generated_summary = ""
    for para in text_in_chunks:
        generated_summary += generate_summary(para, bart_model, bart_tokenizer)
    return generated_summary

def preprocessing_pipeline_docx(docx_filepath):
    extracted_text = docx_doc(docx_filepath)
    text_in_chunks = break_in_chunks(" ".join(extracted_text), 1024, 300)
    generated_summary = ""
    for para in text_in_chunks:
        generated_summary += generate_summary(para, bart_model, bart_tokenizer)
    return generated_summary

def break_in_chunks(text, max_chunk, overlap):
    sentences = re.findall(r'.+?\.', text)
    paras = []
    curr = ""
    i = 0
    while i < len(sentences):
        if len(curr) + len(sentences[i]) > max_chunk:
            paras.append(curr)
            curr = ""
            for j in range(i, 0, -1):
                if len(curr) + len(sentences[j]) > overlap:
                    i = j - 1
                    curr = ""
                    break
                else:
                    curr += sentences[j]
        else:
            curr += sentences[i]
        i += 1
    if curr:
        paras.append(curr)
    return paras

def generate_summary_mind_map(text, model, tokenizer, maxi):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=maxi)
    summary_ids = model.generate(inputs.input_ids.to('cuda'), max_length=maxi, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize(text, maxi):
    paras = break_in_chunks(text, 1024, 200)
    summary = ""
    for para in paras:
        summary += generate_summary_mind_map(para, bart_model, bart_tokenizer, maxi)
    return summary

def similarity(paragraph1, paragraph2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([paragraph1, paragraph2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

def text_to_list(text):
    sentences = text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def form_contexts(text, alpha):
    new_text = text_to_list(text)
    contexts = []
    curr = [new_text[0]]
    for i in range(len(new_text) - 1):
        value = similarity(new_text[i], new_text[i + 1])
        if value > alpha:
            curr.append(new_text[i + 1])
        else:
            para = '. '.join(curr) + '.'
            if len(para) > 30:
                contexts.append(para)
            curr.clear()
            curr.append(new_text[i + 1])
    if curr:
        para = '. '.join(curr) + '.'
        contexts.append(para)
    return contexts

def count_tokens(text):
    tokens = bart_tokenizer.encode(text, add_special_tokens=True)
    num_tokens = len(tokens)
    return num_tokens

def generate_bullet_point(text):
    num_tokens = count_tokens(text)
    max_length = max(num_tokens // 2, 1)
    summary = summarize(text, 150)
    bullet_points = [sentence.strip() for sentence in summary.split('. ') if sentence]

    rake = Rake()
    rake.extract_keywords_from_text(summary)
    title = rake.get_ranked_phrases()[0] if rake.get_ranked_phrases() else "No title found"

    return bullet_points, title

def wrap_text(text, width):
    return '\n'.join(textwrap.wrap(text, width))

def get_bbox_center(ax, text_obj):
    bbox = text_obj.get_window_extent(renderer=ax.get_figure().canvas.get_renderer())
    bbox_data = bbox.transformed(ax.transData.inverted())
    x_center = (bbox_data.x0 + bbox_data.x1) / 2
    y_center = (bbox_data.y0 + bbox_data.y1) / 2
    return x_center, y_center

def connect(ax, text_obj1, text_obj2):
    x1, y1 = get_bbox_center(ax, text_obj1)
    x2, y2 = get_bbox_center(ax, text_obj2)
    ax.plot([x2, x1], [y2, y1], 'k-')

def para1(ax, x, y, bullet_points, title, body1, width, gap, fsize):
    if bullet_points:
        text_obj1 = ax.text(x, y, wrap_text(title.title(), width), fontsize=fsize + 1,
                            ha='center', va='center', rotation=0,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor=(64 / 255, 224 / 255, 208 / 255, 1.0), edgecolor='black'))
        connect(ax, text_obj1, body1)
        prev = text_obj1
        curr_y = y
        for bullet_point in bullet_points:
            text_obj = ax.text(x, curr_y - gap, wrap_text(bullet_point, width), fontsize=fsize,
                               ha='center', va='center', rotation=0,
                               bbox=dict(boxstyle="round,pad=0.5", facecolor=(189 / 255, 252 / 255, 201 / 255, 1.0), edgecolor='black'))
            connect(ax, text_obj, prev)
            prev = text_obj
            curr_y -= gap

def make_points(text):
    points = []
    titles = []
    paragraphs = form_contexts(text, 0.55)
    for paras in paragraphs:
        a, b = generate_bullet_point(paras)
        if len(a) > 1 and len(b) > 0:
            points.append(a)
            titles.append(b)
    return points, titles



def make_points2(text):
    pattern = r'(\d+)\.\s*'
    text = re.sub(pattern, r'\1- ', text)
    segments = text.split('**')
    
    bold_texts = []
    regular_texts = []
    bullet_points=[]

    for i in range(1, len(segments), 2):  
        bold_texts.append(segments[i].strip())
        if i + 1 < len(segments):
            regular_texts.append(segments[i + 1].strip())
    for points in regular_texts:
        sentences = points.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        bullet_points.append(sentences)
    return bullet_points, bold_texts

def get_document_title(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    doc_title = ""
    for phrase in rake.get_ranked_phrases():
        if len(phrase.title()) <= 25:
            doc_title = phrase.title()
            break
    if not doc_title:
        doc_title = rake.get_ranked_phrases()[0].title() if rake.get_ranked_phrases() else "No title found"
    return doc_title

def make_mind_map(text, points, titles):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0, right=(4.8 + 0.3 * (len(points))) - 0.1, top=2.9, bottom=0.1)
    ax.axis('off')

    x = 2.5
    y = 0
    width = 45

    document_title = get_document_title(text)

    text_obj1 = ax.text(x, y, wrap_text(document_title, width), fontsize=20,
                        ha='center', va='center', rotation=0,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=(255 / 255, 127 / 255, 80 / 255, 1.0), edgecolor='black'))

    lr = (0.2 * (len(points)) + 0.9) / 2

    plt.draw()
    for i in range(int(len(points) / 2)):
        para1(ax, x - lr, y - 5, points[2 * i + 1], titles[2 * i + 1], text_obj1, width, 6, 15)
        para1(ax, x + lr, y - 5, points[2 * i], titles[2 * i], text_obj1, width, 6, 15)
        lr += 0.2 * (len(points)) + 0.9
