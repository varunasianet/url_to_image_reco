from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin  # Add cross_origin here
import os
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAI
import requests
from urllib.parse import urlparse, urljoin
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse, unquote
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from google.cloud import aiplatform
import html
from google.oauth2 import service_account
import os
import json
from urllib.parse import urlparse
from google.cloud import storage
import uuid
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # This allows all origins for all routes



credentials = service_account.Credentials.from_service_account_file(
    '/home/varun_saagar/urltoweb/vertexai-key.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-004",
    project="asianet-tech-staging",
    credentials=credentials
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/varun_saagar/service_key.json"
os.environ["PROJECT_ID"] = "asianet-tech-staging"
os.environ["GOOGLE_CSE_ID"] = "a1343a858e5ba4c1f"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCPvYQ-GRzdS2-Y_1hlboPzygNDC_1cB9c"

aiplatform.init(project="asianet-tech-staging", location="asia-southeast1")

llm = VertexAI(
    model_name="gemini-1.5-flash-001",
    max_output_tokens=8192,
    temperature=0.7,
    top_p=0.95,
)

# Set the path to your service account key file
SERVICE_ACCOUNT_FILE = '/home/varun_saagar/urltoweb/storage_service.json'

# Create credentials using the service account file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize GCS client
storage_client = storage.Client(credentials=credentials)

def upload_to_gcs(source_file_name, destination_blob_name):
    try:
        bucket = storage_client.bucket("urltowebstories")
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        return f"https://storage.googleapis.com/urltowebstories/{destination_blob_name}"
    except Exception as e:
        logger.error(f"Error uploading to GCS: {str(e)}")
        return None



class RealTimeWebScraper:
    def __init__(self, urls):
        self.urls = urls
        self.embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002",project="asianet-tech-staging",
            location="us-central1")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def scrape_and_process(self):
        all_text = []
        for url in self.urls:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                all_text.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        splits = self.text_splitter.split_text("\n\n".join(all_text))
        vectorstore = Chroma.from_texts(splits, self.embeddings)
        retriever = vectorstore.as_retriever()
        return retriever

def get_real_time_data(query):
    search = GoogleSearchAPIWrapper()
    search_results = search.results(query, num_results=5)
    processed_results = []
    for result in search_results:
        processed_results.append({
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "link": result.get("link", "")
        })
    summary = f"Search query: {query}\n\n"
    for i, result in enumerate(processed_results, 1):
        summary += f"Result {i}:\n"
        summary += f"Title: {result['title']}\n"
        summary += f"Snippet: {result['snippet']}\n"
        summary += f"Link: {result['link']}\n\n"
    return summary


def save_slides_to_file(content_input, slides):
    folder_name = get_folder_name(content_input)
    
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{folder_name}_output.json")
    
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

def get_folder_name(url):
    parsed_url = urlparse(url)
    folder_name = parsed_url.path[-5:] if len(parsed_url.path) >= 5 else parsed_url.path
    folder_name = folder_name.replace('/', '_')
    return folder_name

def search_and_scrape(query):
    search = GoogleSearchAPIWrapper()
    search_results = search.results(query, num_results=5)
    all_content = ""
    for result in search_results:
        url = result["link"]
        try:
            loader = WebBaseLoader(url)
            doc = loader.load()
            all_content += doc[0].page_content + "\n\n"
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(all_content)
    return texts

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

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

import urllib.parse

async def download_image(session, img_tag, save_dir, folder_name, image_count):
    src = img_tag.get('src')
    if not src:
        return
    url = urljoin(session.base_url, src)
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
                # Parse the URL and remove query parameters
                parsed_url = urllib.parse.urlparse(url)
                file_extension = os.path.splitext(parsed_url.path)[1].lower()
                if not file_extension:
                    file_extension = '.jpg'  # Default to .jpg if no extension is found
                filename = f"{folder_name}_image{image_count}{file_extension}"
                local_path = os.path.join(save_dir, filename)
                with open(local_path, 'wb') as f:
                    f.write(content)
                
                # Upload image to GCS
                gcs_image_path = f"{folder_name}/{filename}"
                image_url = upload_to_gcs(local_path, gcs_image_path)
                
                logger.info(f"Downloaded and uploaded: {url} - {message}")
                return image_url if image_url else local_path
            else:
                logger.warning(f"Failed to download: {url} - HTTP {response.status}")
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
    return None

def get_folder_name(url):
    parsed_url = urlparse(url)
    path = parsed_url.path.strip('/')
    folder_name = path[-6:] if len(path) >= 6 else path
    folder_name = folder_name.replace('/', '_')
    return folder_name


async def crawl_images(url):
    folder_name = get_folder_name(url)
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        session.base_url = url
        try:
            html_content = await fetch_url(session, url)
            soup = BeautifulSoup(html_content, 'html.parser')   
            article_content = find_article_content(soup)
            if article_content:
                img_tags = article_content.find_all('img')
            else:
                logger.warning("Couldn't find the main article content. Falling back to all images.")
                img_tags = soup.find_all('img')
            tasks = [download_image(session, img, save_dir, folder_name, i+1) for i, img in enumerate(img_tags)]
            image_urls = await asyncio.gather(*tasks)
            return [url for url in image_urls if url]
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            return []

main_model = VertexAI(
    model_name="gemini-1.5-flash-001",
    max_output_tokens=8192,
    temperature=1,
    top_p=0.95,
    project="asianet-tech-staging",
    location="us-central1",
)

regenerate_model = VertexAI(
    model_name="gemini-1.5-flash-001",
    max_output_tokens=8192,
    temperature=1.8,
)

imagen_client = aiplatform.ImageGenerationClient(
    client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
)

def generate_image_for_slide(text_content):
    prompt = f"Generate an image for a web story slide with the following content: {text_content}. Create a visually appealing scene that captures the essence of the given text. Avoid including any explicit text or logos. Focus on creating an engaging and relevant image for a web story."
    
    try:
        response = imagen_client.generate_images(
            prompt=prompt,
            number_of_images=1,
            image_parameters=aiplatform.types.ImageGenerationImageParameters(
                resolution="1024x1024"
            )
        )
        
        # The response contains the generated image data
        image_data = response.images[0]
        
        # Save the image to a file or upload it to storage
        image_filename = f"slide_image.png"
        with open(image_filename, "wb") as f:
            f.write(image_data)
        
        # Upload the image to your storage and get the URL
        image_url = upload_to_gcs(image_filename, f"generated_images/{image_filename}")
        
        return image_url
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return None


def scrape_images(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        images = []
        for img in img_tags:
            src = img.get('src')
            if src:
                absolute_url = urljoin(url, src)
                images.append(absolute_url)
        return images
    except Exception as e:
        print(f"Error scraping images: {e}")
        return []

prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in creating engaging web stories from either news articles or search results, and translating them from {input_language} to {output_language}. Your task is to transform the given content into a concise {num_slides}-slide web story format in {output_language}, tailored for the {target_audience} audience. Each slide should be presented in a specific JSON-like format with a title and description in {output_language}. The content should be informative, engaging, and easy to read in a visual story format.

    Please create a {num_slides}-slide web story in {output_language} based on the following content:
    {content}

    Target Audience: {target_audience}

    Follow these guidelines for the slides:
    1. Each slide title must be no more than 70 characters long.
    2. Each slide description must be no more than 180 characters long.
    3. Create {num_slides} slides, each in the specified JSON-like format, with all text in {output_language}.
    4. The first slide is crucial:
       - Its title and Description MUST be a compelling question that hooks the reader instantly.
       - The question should create curiosity and make the reader want to explore further.
       - Example: "Why do enemies fear Mossad, Israel's intelligence agency?"
       - The description should hint at intriguing answers without giving everything away.
    5. Use concise, vivid language suitable for a visual story format in {output_language}.
    6. Ensure that the story flows logically from one slide to the next, building on the intrigue created by the first slide.
    7. The final slide should summarize or conclude the story in a satisfying way, potentially circling back to answer the initial question.
    8. Translate all content from {input_language} to {output_language}.
    9. Tailor the language, tone, and content to suit the {target_audience} audience.

    Additionally, generate the following SEO-optimized metadata:

    1. Title: Craft a concise and engaging title of maximum 60 characters that accurately represents the entire content and includes primary keywords. The title should be grammatically accurate in {output_language} and should hint at the content's subject matter without giving away the ending or key findings. Evoke a sense of mystery and intrigue, making the reader want to learn more.

    2. Summary: Write a brief summary (2-3 sentences) that encapsulates the main points of the content in {output_language}. Stick strictly to the provided content without adding any new information.

    3. Meta Title: Create a meta title (60 characters or less) in {output_language} that is similar to the title but optimized for search engines with primary keywords.

    4. English Meta Title: Create a meta title (60 characters or less) in English that is similar to the title but optimized for search engines with primary keywords.

    5. Meta Description: Develop a meta description (155-160 characters) in {output_language} that provides a compelling summary of the content, includes primary and secondary keywords, and encourages users to click through.

    6. Meta Keywords: List strictly only top 6 meta keywords in {output_language} that are relevant to the content, focusing on primary and secondary keywords. Don't give one-word keywords; generate long-tail keywords which users would likely search in Google to find similar content. The keywords should not be standalone generic words.

    7. English Meta Keywords: List strictly only top 4 meta keywords in English that are relevant to the content, focusing on primary and secondary keywords. Ensure these are long-tail, unique, and engaging keywords that align with best SEO practices.

    8. Webstorie Title: Create a concise and engaging title in {output_language} that accurately represents the entire content. It MUST be up to 3 words for English, or up to 5 words (not exceeding 30 characters) for non-English languages.

    IMPORTANT: Your response must be a single, valid JSON object containing two keys: "slides" (an array of {num_slides} slide objects) and "metadata" (an object with metadata fields). Do not include any text before or after the JSON object. Ensure each slide object has exactly these keys: "title", "description".

    The JSON structure should look like this:
    {{
        "slides": [
            {{
                "title": "{{SLIDE_TITLE}}",
                "description": "{{SLIDE_DESCRIPTION}}"
            }},
            ...
        ],
        "metadata": {{
            "title": "{{TITLE}}",
            "summary": "{{SUMMARY}}",
            "meta_title": "{{META_TITLE}}",
            "english_meta_title": "{{ENGLISH_META_TITLE}}",
            "meta_description": "{{META_DESCRIPTION}}",
            "meta_keywords": ["{{KEYWORD1}}", "{{KEYWORD2}}", "{{KEYWORD3}}", "{{KEYWORD4}}", "{{KEYWORD5}}", "{{KEYWORD6}}"],
            "english_meta_keywords": ["{{ENG_KEYWORD1}}", "{{ENG_KEYWORD2}}", "{{ENG_KEYWORD3}}", "{{ENG_KEYWORD4}}"],
            "webstorie_title": "{{WEBSTORIE_TITLE}}"
        }}
    }}

    Remember, the first slide's title MUST be a captivating question that creates immediate interest!
    """
)




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

search_prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in creating engaging web stories from the most recent and up-to-date search results, translating them from {input_language} to {output_language}. Your task is to transform the given search summary into a concise {num_slides}-slide web story format in {output_language}, tailored for the {target_audience} audience. Each slide should be presented in a specific JSON-like format with a webstorie_title, title, and description in {output_language}. The content should be informative, engaging, easy to read in a visual story format, and focus on the most recent and relevant information.

    Search Summary:
    {search_summary}

    Please create a {num_slides}-slide web story in {output_language} based on the search summary above, focusing on the most recent and up-to-date information.

    Target Audience: {target_audience}

    Follow these guidelines:
    1. Analyze the search summary thoroughly and extract the most relevant, recent, and interesting information.
    2. Create {num_slides} slides, each in the specified JSON-like format, with all text in {output_language}.
    3. The first slide should introduce the topic and grab the reader's attention with the most recent developments.
    4. Use concise, vivid language suitable for a visual story format in {output_language}.
    5. Ensure that the story flows logically from one slide to the next, presenting different aspects or perspectives on the topic, with an emphasis on recent events or information.
    6. The final slide should summarize or conclude the story in a satisfying way, potentially highlighting future implications or developments.
    7. Each slide's description should be no longer than 2-3 sentences, but make every word count for maximum impact.
    8. Translate all content from {input_language} to {output_language}, including the webstorie_title and title.
    9. Tailor the language, tone, and content to suit the {target_audience} audience.
    10. Ensure that the content is factual, based on the provided search summary, and focuses on the most recent information available.

    Provide the {num_slides} slides, each in the following format:
    {{
        "webstorie_title": "{{WEBSTORY_TITLE}}",
        "title": "{{TITLE}}",
        "description": "{{DESCRIPTION}}"
    }}

    Ensure the output is a valid JSON array of {num_slides} slide objects.
    Remember, the web story should provide a comprehensive and up-to-date overview of the topic based on the most recent search results!
    """
)

regenerate_chain = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in regenerating specific parts of web story slides. Your task is to regenerate the {part_to_regenerate} for slide {slide_number} in a {num_slides}-slide web story, translating from {input_language} to {output_language}. The regenerated content should fit seamlessly into the existing story context and be tailored for the {target_audience} audience.

    Original content: {content_input}
    Slide number: {slide_number}
    Total number of slides: {num_slides}
    Part to regenerate: {part_to_regenerate}
    Target Audience: {target_audience}

    Context of surrounding slides:
    {context}

    Current content of the slide:
    {current_slide_content}

    Please regenerate the {part_to_regenerate} for slide {slide_number}. Ensure it:
    1. Fits logically within the story flow.
    2. Maintains the overall narrative of the web story.
    3. Uses engaging and informative language in {output_language}.
    4. Keeps the content concise and suitable for a web story format.
    5. Adheres to the facts presented in the original content.
    6. Is tailored to the {target_audience} audience in terms of language, tone, and content.

    Provide the regenerated {part_to_regenerate} as a single string.
    """
) | regenerate_model | StrOutputParser()

search_chain = search_prompt_template | main_model | StrOutputParser()

regenerate_prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in creating engaging web stories. Your task is to regenerate a specific part of a slide in a {num_slides}-slide web story in {output_language}, tailored for the {target_audience} audience. The regenerated content should fit seamlessly into the existing story context and adhere to strict character limits.

    {article_url_or_search_summary}

    Slide number: {slide_number}
    Total number of slides: {num_slides}
    Part to regenerate: {part_to_regenerate}
    Target Audience: {target_audience}

    Context of surrounding slides:
    {context}

    Current content of the slide:
    {current_slide_content}

    Please regenerate the {part_to_regenerate} for slide {slide_number}. Ensure it:
    1. Fits logically within the story flow.
    2. Maintains the overall narrative of the web story.
    3. Uses engaging and informative language in {output_language}.
    4. Keeps the content concise and suitable for a web story format.
    5. Adheres to the facts presented in the original content.
    6. Is tailored to the {target_audience} audience in terms of language, tone, and content.
    7. Strictly adheres to the following character limits:
       - If regenerating the web story title: EXACTLY 3 words
       - If regenerating a slide title: NO MORE THAN 70 characters
       - If regenerating a slide description: NO MORE THAN 180 characters

    Provide the regenerated {part_to_regenerate} as a single string, ensuring it meets the specified character limit.
    """
)

fix_content_prompt = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in fixing web story content to meet specific character limits, with expertise in handling multiple languages including Hindi, Malayalam, Kannada, Marathi, Bengali, Tamil, and Telugu. Your task is to modify the given content to fit within the required constraints while maintaining the essence of the original message.

    Original content: {original_content}
    Content type: {content_type}
    Current character count: {current_count}
    Maximum allowed characters: {max_chars}
    Output language: {output_language}
    Target audience: {target_audience}

    Please modify the content to fit within the character limit. Ensure that:
    1. The main message or key points are preserved.
    2. The language remains engaging and suitable for a web story format.
    3. The content is in {output_language}.
    4. The tone is appropriate for the {target_audience} audience.
    5. For non-English languages, consider the character count, not word count.

    Specific requirements based on content type:
    - If content_type is "webstorie_title":
        - It MUST be up to  3 words long for English.
        - For non-English languages, aim for a concise title that captures the essence in about 3-5 words, not exceeding 30 characters.
    - If content_type is "title":
        - It MUST be NO MORE THAN 70 characters long.
    - If content_type is "description":
        - It MUST be NO MORE THAN 180 characters long.

    Provide the modified content as a single string, ensuring it meets the specified character limit and requirements for the given content type.
    """
)

fix_content_chain = fix_content_prompt | llm | StrOutputParser()

main_chain = prompt_template | llm | StrOutputParser()
search_chain = search_prompt_template | llm | StrOutputParser()
regenerate_chain = regenerate_prompt_template | llm | StrOutputParser()

supported_languages = ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Kannada"]
target_audiences = ["Common", "Gen Z", "90s", "Middle-aged", "Older"]

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

def regenerate_slide_part(input_language, output_language, content_input, num_slides, slide_number, part_to_regenerate, context, current_slide_content, target_audience, input_type):
    global generated_web_story    
    max_attempts = 3

    logger.info(f"Regenerating {part_to_regenerate} for slide {slide_number}")

    # Input validation
    if not all([input_language, output_language, content_input, num_slides, part_to_regenerate, target_audience, input_type]):
        return {"error": "Missing required parameters"}

    if part_to_regenerate in ["webstorie_title", "english_title", "summary", "meta_title", "meta_description", "meta_keywords"]:
        max_chars = {
            "webstorie_title": 70,
            "english_title": None,  # We'll handle this separately
            "summary": 250,
            "meta_title": 150,
            "meta_description": 250,
            "meta_keywords": None  # We'll handle this separately
        }
        requirement = f"Regenerate the {part_to_regenerate} for the metadata."
    elif part_to_regenerate in ["title", "description"]:
        max_chars = {"title": 70, "description": 180}
        requirement = f"The {part_to_regenerate} MUST be {max_chars[part_to_regenerate]} characters or less."
    elif part_to_regenerate == "all":
        max_chars = None
        requirement = "Regenerate the entire slide content."
    else:
        return {"error": f"Invalid part_to_regenerate: {part_to_regenerate}"}
    
    for attempt in range(max_attempts):
        try:
            chain_input = {
                "input_language": input_language,
                "output_language": output_language,
                "num_slides": num_slides,
                "slide_number": slide_number,
                "part_to_regenerate": part_to_regenerate,
                "context": context,
                "current_slide_content": current_slide_content,
                "target_audience": target_audience,
                "additional_instructions": requirement
            }
            
            if input_type == "url":
                chain_input["article_url_or_search_summary"] = f"Original article URL: {content_input}"
            else:
                chain_input["article_url_or_search_summary"] = f"Search summary: {content_input}"
            
            result = regenerate_chain.invoke(chain_input)
            logger.info(f"Generated result: {result}")

            is_valid = validate_regenerated_content(part_to_regenerate, result, output_language, max_chars)
            
            if is_valid:
                break
            
            if max_chars and part_to_regenerate not in ["english_title", "meta_keywords"]:
                result = fix_content_chain.invoke({
                    "original_content": result,
                    "content_type": part_to_regenerate,
                    "current_count": len(result),
                    "max_chars": max_chars[part_to_regenerate] if isinstance(max_chars, dict) else max_chars,
                    "output_language": output_language,
                    "target_audience": target_audience,
                    "requirement": requirement
                })
            
            if attempt == max_attempts - 1:
                return {"error": f"Failed to generate valid content for {part_to_regenerate} after {max_attempts} attempts."}
        
        except Exception as e:
            logger.error(f"Error in regeneration attempt {attempt + 1}: {str(e)}")
            if attempt == max_attempts - 1:
                return {"error": f"An error occurred while regenerating content: {str(e)}"}
    
    # Update the generated_web_story
    if part_to_regenerate in ["webstorie_title", "english_title", "summary", "meta_title", "meta_description", "meta_keywords"]:
        generated_web_story["metadata"][part_to_regenerate] = result
    elif part_to_regenerate == "all":
        if 'generated_web_story' in globals() and generated_web_story and slide_number <= len(generated_web_story["slides"]):
            generated_web_story["slides"][slide_number - 1][f"slide{slide_number}"] = {
                "title": result.get('title', ''),
                "description": result.get('description', ''),
                "image_url": generated_web_story["slides"][slide_number - 1][f"slide{slide_number}"].get('image_url', '')
            }
    
    # Save and upload to GCS
    _, json_url = save_slides_to_file(content_input, generated_web_story)
    generated_web_story["json_url"] = json_url
    
    logger.info(f"Successfully regenerated {part_to_regenerate}")
    return result


def generate_web_story(input_language, output_language, content_input, num_slides, target_audience, input_type):
    global generated_web_story
    generated_web_story = {"webstories": {}, "slides": [], "metadata": {}}
    
    if input_language not in supported_languages or output_language not in supported_languages:
        return f"Invalid language selection. Please choose from: {', '.join(supported_languages)}"
    
    try:
        if input_type == "url":
            article_url = content_input
            scraped_images = asyncio.run(crawl_images(article_url))
            content = article_url
        else:  # prompt
            search_summary = get_real_time_data(content_input)
            content = search_summary
            scraped_images = []
        
        # Generate the web story
        result = main_chain.invoke({
            "input_language": input_language,
            "output_language": output_language,
            "content": content,
            "num_slides": num_slides,
            "target_audience": target_audience
        })
        logger.info("Raw output: %s", result)

        # Parse the JSON object within the result
        try:
            json_start = result.index('{')
            json_end = result.rindex('}') + 1
            json_str = result[json_start:json_end]
            parsed_output = json.loads(json_str)
        except ValueError:
            raise ValueError("Could not find a valid JSON object in the output")

        if not isinstance(parsed_output, dict) or "slides" not in parsed_output or "metadata" not in parsed_output:
            raise ValueError(f"Output is not in the expected format")
        
        slides = parsed_output["slides"]
        metadata = parsed_output["metadata"]
        
        if len(slides) != num_slides:
            raise ValueError(f"Output does not contain {num_slides} slides")
        
        # Set up the webstories section
        generated_web_story["webstories"]["webstorie_title"] = metadata.get('title', '')
        generated_web_story["webstories"]["english_title"] = metadata.get('english_title', '')
        generated_web_story["webstories"]["summary"] = metadata.get('summary', '')
        
        # Set up the metadata section
        generated_web_story["metadata"] = {
            "english_meta_keywords": metadata.get('english_meta_keywords', []),
            "meta_description": metadata.get('meta_description', ''),
            "meta_keywords": metadata.get('meta_keywords', []),
            "meta_title": metadata.get('meta_title', ''),
        }
        
        # Initialize the first_image_url
        first_image_url = ''

        # Set up the slides
        for i, slide in enumerate(slides, 1):
            # If a new image URL is available, use it
            if i <= len(scraped_images) and scraped_images[i-1]:
                current_image_url = scraped_images[i-1]
                if not first_image_url:
                    first_image_url = current_image_url
            # If no new image URL is available, use the first_image_url
            else:
                current_image_url = first_image_url

            slide_content = {
                "title": slide.get('title', ''),
                "description": slide.get('description', ''),
                "image_url": current_image_url
            }
            generated_web_story["slides"].append({f"slide{i}": slide_content})
        
        # Save slides to file and upload to GCS
        _, json_url = save_slides_to_file(content_input, generated_web_story)
        
        # Add the JSON URL to the web story data
        generated_web_story["json_url"] = json_url
        
        return generated_web_story
    except Exception as e:
        log_error(e, "generate_web_story - Unexpected error")
        return {"error": "An unexpected error occurred while generating the web story"}
    
import re

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

@app.route('/generate_web_story', methods=['POST'])
def generate_web_story_api():
    app.logger.debug(f"Received data: {request.json}")
    data = request.json
    
    is_valid, error_message = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400
    input_language = data.get('input_language')
    output_language = data.get('output_language')
    content_input = data.get('content_input')
    num_slides = data.get('num_slides', 8)
    target_audience = data.get('target_audience')
    input_type = data.get('input_type')

    if not content_input:
        return jsonify({"error": "Please provide content_input"}), 400

    try:
        generated_story = generate_web_story(input_language, output_language, content_input, num_slides, target_audience, input_type)
        
        if isinstance(generated_story, dict) and "error" in generated_story:
            return jsonify(generated_story), 400
        
        if isinstance(generated_story, str):
            return jsonify({"error": generated_story}), 400
        
        # Format the response
        formatted_story = {
            "json_url": generated_story.get("json_url", ""),
            "metadata": {
                "english_meta_keywords": generated_story["metadata"].get("english_meta_keywords", []),
                "meta_description": generated_story["metadata"].get("meta_description", ""),
                "meta_keywords": generated_story["metadata"].get("meta_keywords", []),
                "meta_title": generated_story["metadata"].get("meta_title", ""),
                "english_meta_title": preprocess_english_title(generated_story["metadata"].get("english_meta_title", ""))
            },
            "slides": {},
            "webstories": {
                "webstorie_title": generated_story["metadata"].get("webstorie_title", ""),
                "english_title": "",
                "summary": generated_story["metadata"].get("summary", "")
            }
        }
        
        # Ensure webstorie_title is in the output language and has the correct length
        if output_language.lower() == "english":
            if len(formatted_story["webstories"]["webstorie_title"].split()) > 3:
                formatted_story["webstories"]["webstorie_title"] = fix_content_chain.invoke({
                    "original_content": formatted_story["webstories"]["webstorie_title"],
                    "content_type": "webstorie_title",
                    "current_count": len(formatted_story["webstories"]["webstorie_title"].split()),
                    "max_chars": None,
                    "output_language": output_language,
                    "target_audience": target_audience
                })
        else:
            if len(formatted_story["webstories"]["webstorie_title"]) > 30:
                formatted_story["webstories"]["webstorie_title"] = fix_content_chain.invoke({
                    "original_content": formatted_story["webstories"]["webstorie_title"],
                    "content_type": "webstorie_title",
                    "current_count": len(formatted_story["webstories"]["webstorie_title"]),
                    "max_chars": 30,
                    "output_language": output_language,
                    "target_audience": target_audience
                })
        
        # Preprocess and ensure english_title has 3-10 words and no special characters
        english_title = formatted_story["metadata"]["english_meta_title"]
        english_title = preprocess_english_title(english_title)
        if len(english_title.split()) < 3 or len(english_title.split()) > 10:
            english_title = fix_content_chain.invoke({
                "original_content": english_title,
                "content_type": "english_title",
                "current_count": len(english_title.split()),
                "max_chars": None,
                "output_language": "English",
                "target_audience": target_audience
            })
            english_title = preprocess_english_title(english_title)
        formatted_story["webstories"]["english_title"] = english_title
        
        # Ensure summary is not empty and doesn't exceed 250 characters
        if not formatted_story["webstories"]["summary"] or len(formatted_story["webstories"]["summary"]) > 250:
            formatted_story["webstories"]["summary"] = fix_content_chain.invoke({
                "original_content": formatted_story["webstories"]["summary"] or formatted_story["metadata"]["meta_description"],
                "content_type": "summary",
                "current_count": len(formatted_story["webstories"]["summary"]),
                "max_chars": 250,
                "output_language": output_language,
                "target_audience": target_audience
            })
        
        for i, slide in enumerate(generated_story["slides"], 1):
            slide_content = slide[f"slide{i}"]
            formatted_story["slides"][f"slide{i}"] = {
                "description": slide_content.get('description', ''),
                "image_url": slide_content.get('image_url', ''),
                "title": slide_content.get('title', '')
            }
        
        return jsonify(formatted_story), 200
    except Exception as e:
        log_error(e, "generate_web_story_api")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/regenerate_slide_part', methods=['POST'])
def regenerate_slide_part_api():
    data = request.json
    input_language = data.get('input_language')
    output_language = data.get('output_language')
    content_input = data.get('content_input')
    num_slides = data.get('num_slides', 8)
    target_audience = data.get('target_audience')
    slide_index = data.get('slide_index')
    part_to_regenerate = data.get('part_to_regenerate')
    input_type = data.get('input_type')

    try:
        regenerated_part = regenerate_slide_part(
            input_language, output_language, content_input, num_slides,
            slide_index + 1 if slide_index is not None else None, 
            part_to_regenerate, "", "", target_audience, input_type
        )
        
        if isinstance(regenerated_part, dict) and "error" in regenerated_part:
            return jsonify(regenerated_part), 400
        
        response = {
            "regenerated_part": html.unescape(regenerated_part),
            "json_url": generated_web_story.get("json_url", "")
        }
        
        return jsonify(response), 200
    except Exception as e:
        log_error(e, "regenerate_slide_part_api")
        return jsonify({"error": str(e)}), 500

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Add this function to check if the GCS bucket exists
def check_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' exists and is accessible.")
        return True
    except Exception as e:
        logger.error(f"Error accessing GCS bucket '{bucket_name}': {str(e)}")
        return False

# Add this function to create the GCS bucket if it doesn't exist
def create_gcs_bucket(bucket_name):
    try:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info(f"GCS bucket '{bucket_name}' created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating GCS bucket '{bucket_name}': {str(e)}")
        return False

def log_error(e, context):
    logger.error(f"Error in {context}: {str(e)}")
    logger.error(traceback.format_exc())

# Modify the main function to check and create the GCS bucket if necessary
if __name__ == '__main__':
    bucket_name = "urltowebstories"
    if not check_gcs_bucket(bucket_name):
        logger.warning(f"GCS bucket '{bucket_name}' does not exist or is not accessible.")
        if create_gcs_bucket(bucket_name):
            logger.info(f"Created GCS bucket '{bucket_name}'.")
        else:
            logger.error(f"Failed to create GCS bucket '{bucket_name}'. Exiting.")
            exit(1)
    
    app.run(host='0.0.0.0', port=7862)