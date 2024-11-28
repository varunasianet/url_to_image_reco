import os
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import requests
from urllib.parse import urlparse, urljoin
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from aiohttp import TCPConnector, ClientTimeout
import aiohttp
from bs4 import BeautifulSoup
import logging
import traceback
import uuid
import ssl
import certifi
import shutil
from google.cloud import storage
from google.oauth2 import service_account
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
import hashlib
from fastapi.responses import JSONResponse
import html
from google.cloud import aiplatform
import nltk
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess




async def download_image(session, base_url, img_tag, save_dir, folder_name, image_count):
    src = img_tag.get('src')
    if not src:
        return

    url = urljoin(base_url, src)
    alt_text = img_tag.get('alt', '')
    try:
        excluded, reason = is_excluded_image(url, alt_text)
        if excluded:
            logger.info(f"Excluded: {url} - {reason}")
            return

        passed, message = await check_image_aspect_ratio(session, url)
        if not passed:
            logger.info(f"Excluded: {url} - {message}")
            return

        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()

                parsed_url = urlparse(url)
                file_extension = os.path.splitext(parsed_url.path)[1].lower()
                if not file_extension:
                    file_extension = '.jpg'  # Default to .jpg if no extension is found

                filename = f"{folder_name}_image{image_count}{file_extension}"
                local_path = os.path.join(save_dir, filename)

                async with aiofiles.open(local_path, 'wb') as f:
                    await f.write(content)

                # Upload image to GCS
                gcs_image_path = f"{folder_name}/{filename}"
                image_url = upload_to_gcs(local_path, gcs_image_path)

                logger.info(f"Downloaded and uploaded: {url} - {message}")
                return image_url if image_url else local_path
            else:
                logger.warning(f"Failed to download: {url} - HTTP {response.status}")
                return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
    return None

async def check_image_aspect_ratio(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                width, height = image.size
                aspect_ratio = width / height
                area = width * height
                if aspect_ratio <= 0.3 or aspect_ratio >= 3:
                    return False, f"Aspect ratio out of range: {aspect_ratio:.2f}"
                if area <= 500:
                    return False, f"Image area too small: {area}"
                return True, f"Image passed: aspect ratio {aspect_ratio:.2f}, area {area}"
            else:
                return False, f"Failed to fetch image: HTTP {response.status}"
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

async def crawl_images(url):
    folder_name = get_folder_name(url)
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        base_url = url  # Set base_url here instead of modifying the session
        try:
            html_content = await fetch_url(session, url)
            soup = BeautifulSoup(html_content, 'html.parser')
            article_content = find_article_content(soup)

            if article_content:
                img_tags = article_content.find_all('img')
            else:
                logger.warning("Couldn't find the main article content. Falling back to all images.")
                img_tags = soup.find_all('img')

            # Pass base_url to download_image
            tasks = [download_image(session, base_url, img, save_dir, folder_name, i+1) for i, img in enumerate(img_tags)]
            image_urls = await asyncio.gather(*tasks)

            return [url for url in image_urls if url]
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return []

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

def find_article_content(soup):
    potential_containers = [
        soup.find('article'),
        soup.find('div', class_=['article-content', 'post-content', 'entry-content', 'content']),
        soup.find('div', id=['content', 'main-content', 'article-body'])
    ]
    return max(
        (container for container in potential_containers if container),
        key=lambda x: len(x.find_all('p')),
        default=None
    )
def preprocess_text(text):
    return [lemmatizer.lemmatize(token) for token in simple_preprocess(text) if token not in STOPWORDS and len(token) > 3]

def extract_topics(text, num_topics=5, num_words=3):
    logger.info("Extracting topics from text")
    processed_text = preprocess_text(text)
    id2word = corpora.Dictionary([processed_text])
    corpus = [id2word.doc2bow(processed_text)]
    
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                         update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    topics = lda_model.print_topics(num_words=num_words)
    return [' '.join([word.split('"')[1] for word in topic[1].split('+')]) for topic in topics]

def translate_article(llm, text, source_language, target_language="English"):
    if not text.strip():
        raise ValueError("No text provided for translation")
    logger.info(f"Translating text from {source_language} to {target_language}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    translation_prompt = PromptTemplate(
        input_variables=["source_language", "target_language", "text"],
        template="Translate the following text from {source_language} to {target_language}. Ensure the translation is contextual and accurate without adding any extra information:\n\n{text}"
    )
    
    translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
    
    translated_chunks = []
    for chunk in chunks:
        try:
            result = translation_chain.run({
                "source_language": source_language,
                "target_language": target_language,
                "text": chunk
            })
            translated_chunks.append(result)
        except Exception as e:
            logger.error(f"Error translating chunk: {str(e)}")
            # You might want to add a placeholder or skip this chunk
            translated_chunks.append("[Translation Error]")
    
    translated_text = " ".join(translated_chunks)
    
    if not translated_text.strip():
        raise ValueError("Translation resulted in empty text")
    
    return translated_text

def generate_keywords(llm, text):
    logger.info("Generating keywords from translated text")
    if not text.strip():
        raise ValueError("No text provided for keyword generation")
    
    lda_topics = extract_topics(text)
    
    keyword_prompt = PromptTemplate(
        input_variables=["text", "lda_topics"],
        template="""Generate 5 keywords in English from the following text, considering these requirements:
        - Identify the central theme, main topic, or subject matter.
        - If relevant, determine the key individuals involved (names, titles).
        - Pinpoint any significant events, occurrences, or actions.
        - Pay attention to relationships between entities and events for context.
        - Consider the following topics extracted from the text: {lda_topics}

        Prioritize:
        - Relevance: Keywords should be closely related to the core content.
        - Specificity: Avoid generic terms; be precise.
        - Conciseness: Use 1-3 words per keyword.
        - Contextual Significance: Highlight the importance of the main subject matter.
        - Names: Include if individuals are central.
        - Events: Include if specific events are crucial.
        - Main Subjects: Core topics or themes.
        - Key Elements: Objects, locations, or concepts vital to understanding.

        Provide exactly 5 keywords, one per line. Each keyword should be 1 to 3 words long and suitable for image search.
        Do not include any numbering, formatting, or additional text.

        Text: {text}

        Keywords:"""
    )
    
    keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)
    
    try:
        result = keyword_chain.run({"text": text, "lda_topics": ", ".join(lda_topics)})
        keywords = [line.strip() for line in result.split('\n') if line.strip()]
        
        if not keywords:
            raise ValueError("No keywords generated")
        
        return keywords[:5]
    except Exception as e:
        logger.error(f"Error generating keywords: {str(e)}")
        raise ValueError("Failed to generate keywords")


def upload_to_gcs(source_file_name, destination_blob_name):
    try:
        bucket = storage_client.bucket("urltowebstories")
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        return f"https://storage.googleapis.com/urltowebstories/{destination_blob_name}"
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        return None

def get_folder_name(url):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    folder_name = path[-6:] if len(path) >= 6 else path
    folder_name = folder_name.replace('/', '_')
    return folder_name

def save_slides_to_file(content_input, slides):
    folder_name = get_folder_name(content_input)
    
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{folder_name}_output.json")
    
    # Prepare the data to be saved
        # Prepare the data to be saved
    data_to_save = {
        "webstories": slides["webstories"],
        "slides": slides["slides"],
        "metadata": slides["metadata"]
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    # Upload JSON file to GCS
    gcs_json_path = f"{folder_name}/{folder_name}_output.json"
    json_url = upload_to_gcs(file_path, gcs_json_path)
    
    return file_path, json_url if json_url else file_path

def check_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' exists and is accessible.")
        return True
    except Exception as e:
        logger.error(f"Error accessing GCS bucket '{bucket_name}': {str(e)}")
        return False

def create_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating GCS bucket '{bucket_name}': {str(e)}")
        return False

def upload_folder_to_gcs(local_folder, bucket_name, bucket_folder):
    logger.info(f"Uploading folder {local_folder} to GCS bucket {bucket_name}/{bucket_folder}")
    credentials = service_account.Credentials.from_service_account_file(STORAGE_KEY_PATH)
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            blob_path = os.path.join(bucket_folder, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to {blob_path}")

            # Update metadata with GCS path
            if file.endswith('.json'):
                with open(local_path, 'r') as f:
                    metadata = json.load(f)
                
                for keyword, images in metadata.items():
                    # Replace spaces with underscores in the keyword
                    keyword_folder = keyword.replace(' ', '_')
                    for image in images:
                        image['gcs_path'] = f"gs://{bucket_name}/{bucket_folder}/{keyword_folder}/{image['image_name']}"
                
                with open(local_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

def get_folder_name(url_or_text):
    if url_or_text.startswith("http"):
        path = urlparse(url_or_text).path
        folder_name = os.path.splitext(os.path.basename(path))[0][-6:]
    else:
        folder_name = hashlib.md5(url_or_text.encode()).hexdigest()[:6]
    return folder_name

def validate_metadata(metadata):
    if len(metadata.get('webstorie_title', '')) > 70:
        return False, "Webstorie title exceeds 70 characters"
    
    english_title_words = metadata.get('english_title', '').split()
    if len(english_title_words) < 3 or len(english_title_words) > 10:
        return False, "English title should be between 3 and 10 words"
    
    if len(metadata.get('summary', '')) > 250:
        return False, "Summary exceeds 250 characters"
    
    if len(metadata.get('meta_title', '')) > 150:
        return False, "Meta title exceeds 150 characters"
    
    if len(metadata.get('meta_description', '')) > 250:
        return False, "Meta description exceeds 250 characters"
    
    keywords = metadata.get('meta_keywords', [])
    if len(keywords) != 10:
        return False, "There should be exactly 10 meta keywords"
    
    return True, "Valid"

def validate_slide_content(slide):
    webstorie_title_words = slide.get('webstorie_title', '').split()
    title_length = len(slide.get('title', ''))
    description_length = len(slide.get('description', ''))
    
    if len(webstorie_title_words) > 5:  # Changed from 3 to 5
        return False, f"Web story title exceeds 5 words (current: {len(webstorie_title_words)})"
    if title_length > 70:
        return False, f"Slide title exceeds 70 characters (current: {title_length})"
    if description_length > 180:
        return False, f"Slide description exceeds 180 characters (current: {description_length})"
    if 'metadata' in slide:
        return validate_metadata(slide['metadata'])
    return True, "Valid"

def validate_regenerated_content(part_to_regenerate, result, output_language, max_chars):
    if part_to_regenerate == "english_title":
        return 3 <= len(result.split()) <= 10
    elif part_to_regenerate == "meta_keywords":
        keywords = result.split(',')
        return len(keywords) == 10
    elif part_to_regenerate == "webstorie_title":
        if output_language.lower() == "english":
            return len(result.split()) <= 3
        else:
            return len(result) <= 30
    elif part_to_regenerate in ["title", "description", "summary", "meta_title", "meta_description"]:
        return len(result) <= max_chars[part_to_regenerate]
    elif part_to_regenerate == "all":
        return validate_slide_content(result)[0]
    else:
        return len(result) <= max_chars[part_to_regenerate]

def log_error(e, context):
    logger.error(f"Error in {context}: {str(e)}")
    logger.error(traceback.format_exc())

def preprocess_english_title(title):
    # Remove all special characters and replace them with spaces
    title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
    # Replace multiple spaces with a single space
    title = re.sub(r'\s+', ' ', title)
    # Trim leading and trailing spaces
    title = title.strip()
    # Capitalize the first letter of each word
    title = title.title()
    # Ensure the title is between 3 and 10 words
    words = title.split()
    if len(words) < 3:
        title = ' '.join(words + ['Story'] * (3 - len(words)))
    elif len(words) > 10:
        title = ' '.join(words[:10])
    return title

def extract_slide_info(raw_output):
    webstorie_title_match = re.search(r'"webstorie_title":\s*"([^"]*)"', raw_output)
    title_match = re.search(r'"title":\s*"([^"]*)"', raw_output)
    description_match = re.search(r'"description":\s*"([^"]*)"', raw_output)
    
    webstorie_title = webstorie_title_match.group(1) if webstorie_title_match else "N/A"
    title = title_match.group(1) if title_match else "N/A"
    description = description_match.group(1) if description_match else "N/A"
    
    return {
        "webstorie_title": webstorie_title,
        "title": title,
        "description": description
    }


def is_excluded_image(url, alt_text):
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    if path.endswith('.cms'):
        return True, "Excluded .cms file"
    if any(dir in path for dir in ['/assets/', '/static/', '/images/icons/', '/ui/', '/v1/images/']):
        return True, f"Excluded due to directory: {path}"
    filename = os.path.basename(path).lower()
    if any(keyword in filename for keyword in ['logo', 'icon', 'button', 'arrow', 'android', 'ios', 'windows', 'mac', 'facebook', 'twitter', 'instagram', 'youtube', 'pinterest', 'playstore', 'appstore', 'apple']):
        return True, f"Excluded due to filename: {filename}"
    if alt_text:
        alt_lower = alt_text.lower()
        if any(keyword in alt_lower for keyword in ['logo', 'icon', 'button', 'arrow']):
            return True, f"Excluded due to alt text: {alt_text}"
    if 'apple.png' in path or 'v1/images/' in path:
        return True, f"Excluded specific image pattern: {path}"
    return False, "Not excluded"


# Add these functions to utils.py
def validate_input(data):
    required_fields = ['input_language', 'output_language', 'content_input', 'target_audience', 'input_type']
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    if data['input_language'] not in supported_languages:
        return False, f"Unsupported input language: {data['input_language']}"
    
    if data['output_language'] not in supported_languages:
        return False, f"Unsupported output language: {data['output_language']}"
    
    if data['target_audience'] not in target_audiences:
        return False, f"Unsupported target audience: {data['target_audience']}"
    
    if data['input_type'] not in ['url', 'prompt']:
        return False, f"Invalid input type: {data['input_type']}"
    
    return True, ""

def log_error(e, context):
    logger.error(f"Error in {context}: {str(e)}")
    logger.error(traceback.format_exc())

def check_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' exists and is accessible.")
        return True
    except Exception as e:
        logger.error(f"Error accessing GCS bucket '{bucket_name}': {str(e)}")
        return False

def create_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating GCS bucket '{bucket_name}': {str(e)}")
        return False
