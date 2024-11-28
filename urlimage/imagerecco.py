import os
import json
import urllib.parse
import aiohttp
import asyncio
import aiofiles
import ssl
import certifi
from aiohttp import ClientTimeout, TCPConnector
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_vertexai import VertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from google.cloud import aiplatform, storage
from google.oauth2 import service_account
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from nltk.stem import WordNetLemmatizer
import shutil
import hashlib
from aiolimiter import AsyncLimiter
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = os.getenv("PROJECT_ID")
SERP_API_KEY = os.getenv("SERP_API_KEY")
STORAGE_KEY_PATH = os.getenv("STORAGE_KEY_PATH")
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_FOLDER = "images"

# Global variables for configuration
MAX_IMAGES_PER_KEYWORD = 10
MIN_IMAGE_WIDTH = 400
MIN_IMAGE_HEIGHT = 300

# Initialize Google AI Platform
aiplatform.init(project=PROJECT_ID, location="asia-southeast1")

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ArticleRequest(BaseModel):
    url: str = None
    text: str = None
    source_language: str

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

def create_llm():
    return VertexAI(
        model_name="gemini-1.5-flash-001",
        max_output_tokens=8192,
        temperature=0.7,
        top_p=0.95,
    )


def load_article(url):
    logger.info(f"Loading article from URL: {url}")
    loader = WebBaseLoader(url)
    return loader.load()

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
        result = translation_chain.run({
            "source_language": source_language,
            "target_language": target_language,
            "text": chunk
        })
        translated_chunks.append(result)
    
    translated_text = " ".join(translated_chunks)
    
    return translated_text

def generate_keywords(llm, text):
    logger.info("Generating keywords from translated text")
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
    
    result = keyword_chain.run({"text": text, "lda_topics": ", ".join(lda_topics)})
    
    keywords = [line.strip() for line in result.split('\n') if line.strip()]
    
    return keywords[:5]

async def download_and_filter_images(keyword, folder_name):
    logger.info(f"Downloading and filtering images for keyword: {keyword}")
    base_url = f"https://serpapi.com/search.json?q={keyword}&tbm=isch&api_key={SERP_API_KEY}"
    params = {
        "num": 100,
        "hl": "en",
        "gl": "in",
    }
    url = f"{base_url}&{urllib.parse.urlencode(params)}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error fetching images for keyword '{keyword}'. Status code: {response.status}")
                return None
            
            try:
                data = await response.json()
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON for keyword '{keyword}'")
                return None
            
            if 'images_results' not in data:
                logger.warning(f"No image results found for keyword: {keyword}")
                return None
            
            images_results = data['images_results']
            
            high_quality_images = [
                img for img in images_results 
                if img.get('original_width') and img.get('original_height') and 
                int(img['original_width']) >= MIN_IMAGE_WIDTH and 
                int(img['original_height']) >= MIN_IMAGE_HEIGHT
            ]
            
            if not high_quality_images:
                logger.warning(f"No high-quality images found for keyword: {keyword}")
                return None
            
            dir_path = os.path.join(folder_name, keyword.replace(' ', '_'))
            os.makedirs(dir_path, exist_ok=True)
            
            download_tasks = []
            metadata = []
            for i, img in enumerate(high_quality_images[:MAX_IMAGES_PER_KEYWORD]):
                filename = f"{keyword.replace(' ', '_')}_image_{i+1}.jpg"
                full_path = os.path.join(dir_path, filename)
                img_url = img['original']
                download_tasks.append(download_image_with_retry(img_url, full_path, i+1, len(high_quality_images), keyword))
                
                metadata.append({
                    "image_name": filename,
                    "source_url": img_url,
                    "thumbnail_url": img.get('thumbnail'),
                    "title": img.get('title', ''),
                    "source_name": img.get('source', ''),
                    "original_width": img.get('original_width', ''),
                    "original_height": img.get('original_height', ''),
                    "position": i + 1
                })
            
            downloaded_images = await asyncio.gather(*download_tasks)
            successful_downloads = [m for m, success in zip(metadata, downloaded_images) if success]
            
            return {keyword: successful_downloads} if successful_downloads else None


async def download_image(img_url, filename, index, total, keyword):
    logger.info(f"Downloading image {index}/{total} for '{keyword}'")
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        parsed_url = urlparse(img_url)
        domain = parsed_url.netloc
        conn = TCPConnector(ssl=ssl_context, limit=10)  # Limit concurrent connections
        timeout = ClientTimeout(total=30, connect=10, sock_read=10)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": f"https://{domain}",
        }

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            async with session.get(img_url, headers=headers, allow_redirects=True) as response:
                if response.status == 200:
                    content = await response.read()
                    if content:
                        async with aiofiles.open(filename, 'wb') as f:
                            await f.write(content)
                        logger.info(f"Successfully downloaded image {index} for '{keyword}'")
                        return True
                    else:
                        logger.warning(f"Empty content for image {index} for '{keyword}'")
                        return False
                else:
                    logger.warning(f"Failed to download image {index} for '{keyword}'. Status code: {response.status}")
                    return False
    except asyncio.TimeoutError:
        logger.error(f"Timeout error downloading image {index} for '{keyword}'")
    except aiohttp.ClientError as e:
        logger.error(f"Client error downloading image {index} for '{keyword}': {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading image {index} for '{keyword}': {str(e)}")
    return False

rate_limit = AsyncLimiter(10, 1)  # 10 requests per second

async def download_image_with_retry(img_url, filename, index, total, keyword, max_retries=3):
    async with rate_limit:
        for attempt in range(max_retries):
            try:
                result = await download_image(img_url, filename, index, total, keyword)
                if result:
                    return True
                logger.warning(f"Retry {attempt + 1}/{max_retries} for image {index} of '{keyword}'")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error during retry {attempt + 1} for image {index} of '{keyword}': {str(e)}")
        logger.error(f"Failed to download image {index} for '{keyword}' after {max_retries} attempts")
        return False

async def download_image_with_fallback(img_data, filename, index, total, keyword, max_retries=3):
    urls = [img_data['original'], img_data.get('thumbnail'), img_data.get('alternative_url')]
    urls = [url for url in urls if url]  # Remove None values
    
    for url in urls:
        result = await download_image_with_retry(url, filename, index, total, keyword, max_retries)
        if result:
            return True
    
    return False

def get_folder_name(url_or_text):
    if url_or_text.startswith("http"):
        path = urlparse(url_or_text).path
        folder_name = os.path.splitext(os.path.basename(path))[0][-6:]
    else:
        folder_name = hashlib.md5(url_or_text.encode()).hexdigest()[:6]
    return folder_name

async def process_keywords(keywords, folder_name):
    logger.info(f"Processing keywords: {keywords}")
    tasks = [download_and_filter_images(keyword, folder_name) for keyword in keywords]
    results = await asyncio.gather(*tasks)
    
    all_metadata = {}
    for result in results:
        if result:
            all_metadata.update(result)
    
    metadata_filename = os.path.join(folder_name, "all_metadata.json")
    async with aiofiles.open(metadata_filename, 'w') as f:
        await f.write(json.dumps(all_metadata, indent=2))
    logger.info(f"Combined metadata saved in: {metadata_filename}")

    logger.info("Uploading folder to Google Cloud Storage...")
    upload_folder_to_gcs(folder_name, BUCKET_NAME, os.path.join(BUCKET_FOLDER, folder_name))
    logger.info("Upload complete.")

    # Read the updated metadata file
    with open(metadata_filename, 'r') as f:
        updated_metadata = json.load(f)

    # Remove the local folder
    logger.info(f"Removing local folder: {folder_name}")
    shutil.rmtree(folder_name)
    logger.info("Local folder removed successfully.")

    return updated_metadata

async def article_translation_and_keyword_pipeline(url, text, source_language):
    logger.info(f"Starting pipeline for URL: {url}, Text: {text[:50]}..., Source Language: {source_language}")
    llm = create_llm()
    
    if url:
        documents = load_article(url)
        full_text = " ".join([doc.page_content for doc in documents])
    elif text:
        full_text = text
    else:
        raise ValueError("Either URL or text must be provided")
    
    translated_text = translate_article(llm, full_text, source_language)
    keywords = generate_keywords(llm, translated_text)
    
    folder_name = get_folder_name(url or text)
    os.makedirs(folder_name, exist_ok=True)

    metadata = await process_keywords(keywords, folder_name)
    
    return {
        "translated_text": translated_text,
        "keywords": keywords,
        "metadata": metadata
    }

@app.post("/process_article")
async def process_article(request: ArticleRequest):
    try:
        result = await article_translation_and_keyword_pipeline(request.url, request.text, request.source_language)
        return result
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)



