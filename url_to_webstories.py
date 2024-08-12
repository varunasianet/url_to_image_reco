import gradio as gr
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
from googlesearch import search
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from google.cloud import aiplatform
import html
import json
import re

# Set up Vertex AI credentials (unchanged)
os.environ["PROJECT_ID"] = "asianet-tech-staging"

os.environ["GOOGLE_CSE_ID"] = "a1343a858e5ba4c1f"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCPvYQ-GRzdS2-Y_1hlboPzygNDC_1cB9c"

aiplatform.init(project="asianet-tech-staging", location="asia-southeast1")

# Initialize VertexAI embeddings with a specific model name
embeddings = VertexAIEmbeddings(model_name="text-embedding-004",project="asianet-tech-staging",
    location="asia-southeast1")

# Initialize VertexAI model for text generation
llm = VertexAI(
    model_name="gemini-1.5-flash-001",
    max_output_tokens=8192,
    temperature=0.7,
    top_p=0.95,
)

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
                
                # Extract main content (adjust selectors based on website structure)
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                
                all_text.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        # Split text into chunks
        splits = self.text_splitter.split_text("\n\n".join(all_text))

        # Create vector store
        vectorstore = Chroma.from_texts(splits, self.embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever()

        return retriever

def get_real_time_data(query):
    # Use Google Search API to get top 5 results
    search = GoogleSearchAPIWrapper()
    search_results = search.results(query, num_results=5)
    
    # Extract relevant information from search results
    processed_results = []
    for result in search_results:
        processed_results.append({
            "title": result.get("title", ""),
            "snippet": result.get("snippet", ""),
            "link": result.get("link", "")
        })
    
    # Create a summary of the search results
    summary = f"Search query: {query}\n\n"
    for i, result in enumerate(processed_results, 1):
        summary += f"Result {i}:\n"
        summary += f"Title: {result['title']}\n"
        summary += f"Snippet: {result['snippet']}\n"
        summary += f"Link: {result['link']}\n\n"
    
    return summary


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
    
    # Exclude .cms files
    if path.endswith('.cms'):
        return True, "Excluded .cms file"
    
    # Exclude based on common asset directories
    if any(dir in path for dir in ['/assets/', '/static/', '/images/icons/', '/ui/', '/v1/images/']):
        return True, f"Excluded due to directory: {path}"
    
    # Exclude based on filename keywords
    filename = os.path.basename(path).lower()
    if any(keyword in filename for keyword in ['logo', 'icon', 'button', 'arrow', 'android', 'ios', 'windows', 'mac', 'facebook', 'twitter', 'instagram', 'youtube', 'pinterest', 'playstore', 'appstore', 'apple']):
        return True, f"Excluded due to filename: {filename}"
    
    # Exclude based on specific alt text keywords
    if alt_text:
        alt_lower = alt_text.lower()
        if any(keyword in alt_lower for keyword in ['logo', 'icon', 'button', 'arrow']):
            return True, f"Excluded due to alt text: {alt_text}"
    
    # Exclude specific image patterns
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
                if area <= 10000:
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

async def download_image(session, img_tag, save_dir):
    src = img_tag.get('src')
    if not src:
        return
    
    url = urljoin(session.base_url, src)
    alt_text = img_tag.get('alt', '')

    try:
        excluded, reason = is_excluded_image(url, alt_text)
        if excluded:
            print(f"Excluded: {url} - {reason}")
            return

        passed, message = await check_image_aspect_ratio(session, url)
        if not passed:
            print(f"Excluded: {url} - {message}")
            return

        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                filename = os.path.join(save_dir, os.path.basename(unquote(urlparse(url).path)))
                with open(filename, 'wb') as f:
                    f.write(content)
                print(f"Downloaded: {url} - {message}")
                return filename
            else:
                print(f"Failed to download: {url} - HTTP {response.status}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

def get_folder_name(url):
    folder_name = urlparse(url).path[-5:] if len(urlparse(url).path) >= 5 else urlparse(url).path
    folder_name = folder_name.replace('/', '_')  # Replace any slashes with underscores
    return folder_name

async def crawl_images(url):
    folder_name = get_folder_name(url)
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        session.base_url = url  # Set base URL for relative path resolution
        try:
            html_content = await fetch_url(session, url)
            soup = BeautifulSoup(html_content, 'html.parser')

            article_content = find_article_content(soup)
            
            if article_content:
                img_tags = article_content.find_all('img')
            else:
                print("Couldn't find the main article content. Falling back to all images.")
                img_tags = soup.find_all('img')

            tasks = [download_image(session, img, save_dir) for img in img_tags]
            await asyncio.gather(*tasks)

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

    # Return the list of downloaded image paths
    return [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]


# Initialize Vertex AI model (unchanged)
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
    temperature=1.8,  # Higher temperature for more varied regeneration
    top_p=0.95,
)


def scrape_images(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tags = soup.find_all('img')
        images = []
        for img in img_tags:
            src = img.get('src')
            if src:
                # Make sure to get the absolute URL
                absolute_url = urljoin(url, src)
                images.append(absolute_url)
        return images
    except Exception as e:
        print(f"Error scraping images: {e}")
        return []
    
# Define the main prompt template (unchanged)
prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in creating engaging web stories from either news articles or search results, and translating them from {input_language} to {output_language}. Your task is to transform the given content into a concise {num_slides}-slide web story format in {output_language}, tailored for the {target_audience} audience. Each slide should be presented in a specific JSON-like format with a webstorie_title, title, and description in {output_language}. The content should be informative, engaging, and easy to read in a visual story format.

    Please create a {num_slides}-slide web story in {output_language} based on the following content:
    {content}

    Target Audience: {target_audience}

    Follow these guidelines:
    1. Read and analyze the provided content thoroughly.
    2. Create {num_slides} slides, each in the specified JSON-like format, with all text in {output_language}.
    3. The first slide is crucial:
       - Make its title exceptionally engaging and intriguing.
       - Keep the description very short and crisp (1-2 sentences maximum).
       - End the first slide's description with a question that piques curiosity.
       - Do not provide any details about the story's content beyond the initial hook and question.    4. Use concise, vivid language suitable for a visual story format in {output_language}.
    5. Ensure that the story flows logically from one slide to the next, with each slide building anticipation for the next.
    6. The final slide should summarize or conclude the story in a satisfying way.
    7. Each slide's description should be no longer than 2-3 sentences, but make every word count for maximum impact.
    8. Translate all content from {input_language} to {output_language}, including the webstorie_title and title.
    9. Tailor the language, tone, and content to suit the {target_audience} audience.

    Provide the {num_slides} slides, each in the following format:
    {{
        "webstorie_title": "{{WEBSTORY_TITLE}}",
        "title": "{{TITLE}}",
        "description": "{{DESCRIPTION}}"
    }}

    Ensure the output is a valid JSON array of {num_slides} slide objects.
    Remember, the first slide should be so compelling that readers can't help but want to see what comes next!
    """
)

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

# New prompt template for regenerating a single slide
regenerate_prompt_template = PromptTemplate.from_template(
    """
    You are an AI assistant specialized in creating engaging web stories. Your task is to regenerate a specific part of a slide in a {num_slides}-slide web story in {output_language}, tailored for the {target_audience} audience. The regenerated content should fit seamlessly into the existing story context.

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

    Provide the regenerated {part_to_regenerate} as a single string.
    """
)

# Set up the chains
# Update the main chain and search chain
main_chain = prompt_template | llm | StrOutputParser()
search_chain = search_prompt_template | llm | StrOutputParser()
regenerate_chain = regenerate_prompt_template | llm | StrOutputParser()



# List of supported languages (unchanged)
supported_languages = ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Kannada"]

# List of target audiences
target_audiences = ["Common", "Gen Z", "90s", "Middle-aged", "Older"]





def extract_slide_info(raw_output):
    """
    Attempt to extract slide information from raw output even if JSON parsing fails.
    """
    title_match = re.search(r'"title":\s*"([^"]*)"', raw_output)
    description_match = re.search(r'"description":\s*"([^"]*)"', raw_output)
    
    title = title_match.group(1) if title_match else "N/A"
    description = description_match.group(1) if description_match else "N/A"
    
    return {"title": title, "description": description}

def regenerate_slide(input_language, output_language, article_url, num_slides, slide_number, context):
    global generated_web_story
    if slide_number < 1 or slide_number > num_slides:
        return f"Invalid slide number. Please choose a number between 1 and {num_slides}."
    
    result = regenerate_chain.invoke({
        "input_language": input_language,
        "output_language": output_language,
        "article_url": article_url,
        "num_slides": num_slides,
        "slide_number": slide_number,
        "context": context,
        "target_audience": target_audience
    })
    
    try:
        # Check if the result is already a dictionary
        if isinstance(result, dict):
            regenerated_slide = result
        else:
            regenerated_slide = json.loads(result)
        
        # Update the stored web story with the regenerated slide
        if generated_web_story and isinstance(generated_web_story, list):
            generated_web_story[slide_number - 1] = regenerated_slide
        return regenerated_slide
    except json.JSONDecodeError:
        # Attempt to extract information even if JSON parsing fails
        extracted_info = extract_slide_info(result)
        
        # Create a structured slide object from extracted information
        regenerated_slide = {
            "webstorie_title": generated_web_story[0].get("webstorie_title", "N/A") if generated_web_story else "N/A",
            "title": extracted_info['title'],
            "description": extracted_info['description']
        }
        
        # Update the stored web story with the extracted information
        if generated_web_story and isinstance(generated_web_story, list):
            generated_web_story[slide_number - 1] = regenerated_slide
        
        return regenerated_slide

def regenerate_individual_slide(input_language, output_language, article_url, num_slides, slide_index, *current_slides):
    global generated_web_story
    if not generated_web_story:
        return gr.Markdown("Please generate the web story first before regenerating individual slides.")
    
    context = "\n".join([f"Slide {i+1}: {json.dumps(slide)}" for i, slide in enumerate(generated_web_story)])
    regenerated_slide = regenerate_slide(input_language, output_language, article_url, num_slides, slide_index + 1, context)
    
    return gr.Markdown(f"**Slide {slide_index + 1}**\n\n**Title:** {regenerated_slide.get('title', 'N/A')}\n\n**Description:** {regenerated_slide.get('description', 'N/A')}")

generated_web_story = None

def regenerate_slide_part(input_language, output_language, content_input, num_slides, slide_number, part_to_regenerate, context, current_slide_content, target_audience, input_type):
    global generated_web_story
    if slide_number < 1 or slide_number > num_slides:
        return f"Invalid slide number. Please choose a number between 1 and {num_slides}."
    
    # Prepare the input for the regenerate_chain
    chain_input = {
        "input_language": input_language,
        "output_language": output_language,
        "num_slides": num_slides,
        "slide_number": slide_number,
        "part_to_regenerate": part_to_regenerate,
        "context": context,
        "current_slide_content": current_slide_content,
        "target_audience": target_audience
    }
    
    # Add the appropriate content input based on the input type
    if input_type == "url":
        chain_input["article_url"] = content_input
    else:  # prompt
        chain_input["search_summary"] = content_input
    
    result = regenerate_chain.invoke(chain_input)
    
    # Update the stored web story with the regenerated part
    if generated_web_story and isinstance(generated_web_story, list):
        generated_web_story[slide_number - 1][part_to_regenerate] = result
    
    return result



def create_slide_component(index):
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=2):
                slide_content = gr.Markdown(label=f"Slide {index + 1}")
            with gr.Column(scale=1):
                with gr.Row():
                    change_image_btn = gr.Button("Change", size="sm")
                    drop_image_btn = gr.Button("Drop", size="sm")
                image_display = gr.Image(label=f"Image for Slide {index + 1}", type="filepath")
        with gr.Row():
            regenerate_title_btn = gr.Button(f"Regenerate Title", size="sm")
            regenerate_description_btn = gr.Button(f"Regenerate Description", size="sm")
    return slide_content, image_display, change_image_btn, drop_image_btn, regenerate_title_btn, regenerate_description_btn


def save_slides_to_file(content_input, slides):
    if content_input.startswith(("http://", "https://")):
        folder_name = get_folder_name(content_input)
    else:
        folder_name = "prompt_" + content_input[:20].replace(" ", "_")
    
    save_dir = os.path.join('/tmp', 'scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "slides.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(slides, f, ensure_ascii=False, indent=2)
    return file_path



def generate_web_story(input_language, output_language, content_input, num_slides, target_audience, input_type):
    global generated_web_story
    if input_language not in supported_languages or output_language not in supported_languages:
        return f"Invalid language selection. Please choose from: {', '.join(supported_languages)}"
    
    if input_type == "url":
        article_url = content_input
        scraped_images = asyncio.run(crawl_images(article_url))
        content = article_url
    else:  # prompt
        search_summary = get_real_time_data(content_input)
        content = search_summary
        scraped_images = []  # We won't have specific images for search results
    
    # Generate the web story
    result = main_chain.invoke({
        "input_language": input_language,
        "output_language": output_language,
        "content": content,
        "num_slides": num_slides,
        "target_audience": target_audience
    })
    
    # Parse the output (same as before)
    try:
        parsed_output = json.loads(result)
        if not isinstance(parsed_output, list) or len(parsed_output) != num_slides:
            raise ValueError(f"Output is not a list of {num_slides} slides")
        
        # Assign scraped images to slides (if available)
        last_image = None
        for i, slide in enumerate(parsed_output):
            if i < len(scraped_images):
                slide['image_url'] = scraped_images[i]
                last_image = scraped_images[i]
            else:
                slide['image_url'] = last_image  # Replicate the last available image
        
        generated_web_story = parsed_output  # Store the generated web story
        save_slides_to_file(content_input, generated_web_story)  # Save slides to file
        return parsed_output
    except json.JSONDecodeError:
        # If parsing as JSON fails, try to extract JSON-like structures
        json_like_structures = re.findall(r'\{[^{}]*\}', result)
        if len(json_like_structures) == num_slides:
            generated_web_story = [json.loads(struct) for struct in json_like_structures]
            
            # Assign scraped images to slides (if available)
            last_image = None
            for i, slide in enumerate(generated_web_story):
                if i < len(scraped_images):
                    slide['image_url'] = scraped_images[i]
                    last_image = scraped_images[i]
                else:
                    slide['image_url'] = last_image  # Replicate the last available image
            
            save_slides_to_file(content_input, generated_web_story)  # Save slides to file
            return generated_web_story
        else:
            return f"Error parsing output. Raw output:\n{result}"

def regenerate_slide_part_interface(input_language, output_language, url_input, prompt_input, num_slides, target_audience, slide_index, part_to_regenerate):
    global generated_web_story
    num_slides = num_slides or 8
    slide_index = int(slide_index)
    
    content_input = url_input if url_input else prompt_input
    input_type = "url" if url_input else "prompt"
    
    if not generated_web_story:
        return gr.Markdown("Please generate the web story first before regenerating individual parts.")
    
    context = "\n".join([f"Slide {i+1}: {json.dumps(slide)}" for i, slide in enumerate(generated_web_story)])
    current_slide_content = json.dumps(generated_web_story[slide_index])
    
    regenerated_part = regenerate_slide_part(input_language, output_language, content_input, num_slides, slide_index + 1, part_to_regenerate, context, current_slide_content, target_audience, input_type)
    
    updated_slide = generated_web_story[slide_index]
    updated_slide[part_to_regenerate] = regenerated_part
    
    slide_content = f"**Slide {slide_index + 1}**\n\n**Title:** {updated_slide.get('title', 'N/A')}\n\n**Description:** {updated_slide.get('description', 'N/A')}"
    
    # Save the updated slides to file
    save_slides_to_file(content_input, generated_web_story)
    
    return gr.Markdown(slide_content)




def regenerate_slide_part(input_language, output_language, content_input, num_slides, slide_number, part_to_regenerate, context, current_slide_content, target_audience, input_type):
    global generated_web_story
    if slide_number < 1 or slide_number > num_slides:
        return f"Invalid slide number. Please choose a number between 1 and {num_slides}."
    
    # Prepare the input for the regenerate_chain
    chain_input = {
        "input_language": input_language,
        "output_language": output_language,
        "num_slides": num_slides,
        "slide_number": slide_number,
        "part_to_regenerate": part_to_regenerate,
        "context": context,
        "current_slide_content": current_slide_content,
        "target_audience": target_audience
    }
    
    # Set the article_url_or_search_summary based on the input type
    if input_type == "url":
        chain_input["article_url_or_search_summary"] = f"Original article URL: {content_input}"
    else:  # prompt
        chain_input["article_url_or_search_summary"] = f"Search summary: {content_input}"
    
    result = regenerate_chain.invoke(chain_input)
    
    # Update the stored web story with the regenerated part
    if generated_web_story and isinstance(generated_web_story, list):
        generated_web_story[slide_number - 1][part_to_regenerate] = result
    
    return result

    

def regenerate_slide_part_interface(input_language, output_language, url_input, prompt_input, num_slides, target_audience, slide_index, part_to_regenerate):
    global generated_web_story
    num_slides = num_slides or 8
    slide_index = int(slide_index)
    
    content_input = url_input if url_input else prompt_input
    input_type = "url" if url_input else "prompt"
    
    if not generated_web_story:
        return gr.Markdown("Please generate the web story first before regenerating individual parts.")
    
    context = "\n".join([f"Slide {i+1}: {json.dumps(slide)}" for i, slide in enumerate(generated_web_story)])
    current_slide_content = json.dumps(generated_web_story[slide_index])
    
    regenerated_part = regenerate_slide_part(input_language, output_language, content_input, num_slides, slide_index + 1, part_to_regenerate, context, current_slide_content, target_audience, input_type)
    
    updated_slide = generated_web_story[slide_index]
    updated_slide[part_to_regenerate] = regenerated_part
    
    slide_content = f"**Slide {slide_index + 1}**\n\n**Title:** {updated_slide.get('title', 'N/A')}\n\n**Description:** {updated_slide.get('description', 'N/A')}"
    
    # Save the updated slides to file
    save_slides_to_file(content_input, generated_web_story)
    
    return gr.Markdown(slide_content)


# Update the save_slides_to_file function to handle both URL and prompt inputs
def save_slides_to_file(content_input, slides):
    if content_input.startswith(("http://", "https://")):
        folder_name = get_folder_name(content_input)
    else:
        folder_name = "prompt_" + content_input[:20].replace(" ", "_")  # Use first 20 chars of prompt
    
    save_dir = os.path.join('scraped_content', folder_name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "slides.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(slides, f, ensure_ascii=False, indent=2)
    return file_path

# Update the change_image function to handle both URL and prompt inputs
def change_image(slide_index, new_image, input_type, content_input):
    global generated_web_story
    if generated_web_story and slide_index < len(generated_web_story):
        generated_web_story[slide_index]['image_url'] = new_image
        save_slides_to_file(content_input, generated_web_story)
        return update_slide_content(slide_index, generated_web_story[slide_index], new_image)
    return gr.Markdown("No changes made."), gr.Image(value=None)

def drop_image(slide_index, input_type, content_input):
    global generated_web_story
    if generated_web_story and slide_index < len(generated_web_story):
        generated_web_story[slide_index]['image_url'] = None
        save_slides_to_file(content_input, generated_web_story)
        return update_slide_content(slide_index, generated_web_story[slide_index], None)
    return gr.Markdown("No changes made."), gr.Image(value=None)

def prepare_export_data():
    global generated_web_story
    if not generated_web_story:
        return None
    
    export_data = []
    for slide in generated_web_story:
        export_slide = {
            "webstorie_title": slide.get("webstorie_title", ""),
            "title": slide.get("title", ""),
            "description": slide.get("description", ""),
            "image_url": slide.get("image_url", "")
        }
        export_data.append(export_slide)
    
    return export_data

def export_webstory():
    export_data = prepare_export_data()
    if not export_data:
        return gr.File(value=None), gr.Markdown("Please generate a web story before exporting.")
    
    export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
    file_name = "/tmp/webstory_export.json"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(export_json)
    
    return gr.File(value=file_name), gr.Markdown("Web story exported successfully!")


def update_visibility(num_slides):
    num_slides = num_slides or 8  # Use 8 as default if num_slides is None
    updates = [gr.update(visible=True)]  # For web story title
    for i in range(10):
        updates.extend([
            gr.update(visible=i < num_slides),  # For slide content
            gr.update(visible=i < num_slides),  # For image display
            gr.update(visible=i < num_slides),  # For change image button
            gr.update(visible=i < num_slides),  # For drop image button
            gr.update(visible=i < num_slides),  # For regenerate title button
            gr.update(visible=i < num_slides),  # For regenerate description button
        ])
    return updates

def update_slide_content(slide_index, new_content, image_url):
    slide_content = f"**Slide {slide_index + 1}**\n\n**Title:** {new_content.get('title', 'N/A')}\n\n**Description:** {new_content.get('description', 'N/A')}"
    try:
        return gr.Markdown(slide_content), gr.Image(value=image_url) if image_url else gr.Image(value=None)
    except UnidentifiedImageError:
        return gr.Markdown(slide_content), gr.Image(value=None)

def clear_inputs_and_outputs():
    return (
        gr.Textbox(value=""),  # Clear URL input
        gr.Textbox(value=""),  # Clear prompt input
        gr.Markdown(""),  # Clear web story title output
        *[gr.Markdown("") for _ in range(10)],  # Clear slide outputs
        *[gr.Image(value=None) for _ in range(10)],  # Clear image displays
        *[gr.update(visible=False) for _ in range(51)],  # Hide all buttons
        gr.File(value=None),  # Clear export output
        gr.Markdown("")  # Clear export status
    )


with gr.Blocks(css="button.sm { margin: 0.1rem; }") as demo:
    gr.Markdown("# Multilingual Web Story Generator ")
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                input_lang = gr.Dropdown(choices=supported_languages, label="Input Language", value="English")
                output_lang = gr.Dropdown(choices=supported_languages, label="Output Language", value="Hindi")
            
            with gr.Tabs() as tabs:
                with gr.TabItem("URL"):
                    url_input = gr.Textbox(label="Article URL")
                with gr.TabItem("Search Prompt"):
                    prompt_input = gr.Textbox(label="Search Prompt")
            
            num_slides = gr.Slider(minimum=3, maximum=10, step=1, value=8, label="Number of Slides")
            target_audience = gr.Dropdown(choices=target_audiences, label="Target Audience", value="Common")
            submit_btn = gr.Button("Generate Web Story", variant="primary")
        with gr.Column(scale=1):
            webstory_title_output = gr.Markdown(label="Web Story Title")
            regenerate_webstory_title_btn = gr.Button("Regenerate Web Story Title")
            export_btn = gr.Button("Export Web Story", variant="secondary")
            export_output = gr.File(label="Exported Web Story")
            export_status = gr.Markdown()

    slide_components = [create_slide_component(i) for i in range(10)]
    slide_outputs, image_displays, change_image_btns, drop_image_btns, regenerate_title_btns, regenerate_description_btns = zip(*slide_components)

    # Add event handlers for tab changes
    tabs.change(
        fn=clear_inputs_and_outputs,
        inputs=[],
        outputs=[url_input, prompt_input, webstory_title_output] + list(slide_outputs) + list(image_displays) + 
                [regenerate_webstory_title_btn] + 
                list(change_image_btns) + list(drop_image_btns) + 
                list(regenerate_title_btns) + list(regenerate_description_btns)
    )

    def gradio_interface(input_language, output_language, url_input, prompt_input, num_slides, target_audience):
        if not url_input and not prompt_input:
            return [gr.Markdown("Please enter a URL or a search prompt.")] + [gr.Markdown("")] * 10 + [gr.Image(value=None)] * 10 + [gr.update(visible=False)] * 51

        content_input = url_input if url_input else prompt_input
        input_type = "url" if url_input else "prompt"
    
        generated_story = generate_web_story(input_language, output_language, content_input, num_slides, target_audience, input_type)
         
        outputs = [gr.Markdown("")] * 11
        image_outputs = [gr.Image(value=None)] * 10
        button_visibility = [gr.update(visible=False)] * 61  # Increased to account for all buttons

        if isinstance(generated_story, str):
            outputs[0] = gr.Markdown(generated_story)
        else:
            # Decode the Unicode escape sequences for the web story title
            decoded_title = html.unescape(generated_story[0].get('webstorie_title', 'N/A'))
            outputs[0] = gr.Markdown(f"**Web Story Title:** {decoded_title}")
            button_visibility[0] = gr.update(visible=True)

            for i in range(10):
                if i < len(generated_story):
                    slide = generated_story[i]
                    # Decode the Unicode escape sequences for the slide title and description
                    decoded_title = html.unescape(slide.get('title', 'N/A'))
                    decoded_description = html.unescape(slide.get('description', 'N/A'))
                    slide_content = f"**Slide {i+1}**\n\n**Title:** {decoded_title}\n\n**Description:** {decoded_description}"
                    outputs[i+1] = gr.Markdown(slide_content)
                    image_outputs[i] = gr.Image(value=slide.get('image_url'))
                    # Update visibility for change, drop, regenerate title, and regenerate description buttons
                    button_visibility[i*6+1:i*6+7] = [gr.update(visible=True)] * 6
                else:
                    # Hide components for unused slides
                    button_visibility[i*6+1:i*6+7] = [gr.update(visible=False)] * 6

        export_output = gr.File(value=None)
        export_status = gr.Markdown("")

        return outputs + image_outputs + button_visibility + [export_output, export_status]



    submit_btn.click(
        fn=gradio_interface,
        inputs=[input_lang, output_lang, url_input, prompt_input, num_slides, target_audience],
        outputs=[webstory_title_output] + list(slide_outputs) + list(image_displays) + 
                [regenerate_webstory_title_btn] + 
                list(change_image_btns) + list(drop_image_btns) + 
                list(regenerate_title_btns) + list(regenerate_description_btns) +
                [export_output, export_status],
        show_progress=True
    )



    

    def regenerate_webstory_title_interface(input_language, output_language, url_input, prompt_input, num_slides, target_audience):
        global generated_web_story
        if not generated_web_story:
            return gr.Markdown("Please generate the web story first before regenerating the web story title.")
        
        content_input = url_input if url_input else prompt_input
        input_type = "url" if url_input else "prompt"
        
        context = "\n".join([f"Slide {i+1}: {json.dumps(slide)}" for i, slide in enumerate(generated_web_story)])
        regenerated_title = regenerate_slide_part(input_language, output_language, content_input, num_slides, 1, "webstorie_title", context, json.dumps(generated_web_story[0]), target_audience, input_type)
        
        # Decode the Unicode escape sequences
        decoded_title = html.unescape(regenerated_title)
        
        generated_web_story[0]['webstorie_title'] = decoded_title
        save_slides_to_file(content_input, generated_web_story)  # Save updated slides to file
        
        return gr.Markdown(f"**Web Story Title:** {decoded_title}")


    regenerate_webstory_title_btn.click(
        fn=regenerate_webstory_title_interface,
        inputs=[input_lang, output_lang, url_input, prompt_input, num_slides, target_audience],
        outputs=[webstory_title_output],
        show_progress=True
    )

    for i in range(10):
        change_image_btns[i].click(
            fn=change_image,
            inputs=[gr.Slider(value=i, visible=False), image_displays[i], gr.Radio(["url", "prompt"], visible=False), gr.Textbox(visible=False)],
            outputs=[slide_outputs[i], image_displays[i]],
            show_progress=True
        )

        drop_image_btns[i].click(
            fn=drop_image,
            inputs=[gr.Slider(value=i, visible=False), gr.Radio(["url", "prompt"], visible=False), gr.Textbox(visible=False)],
            outputs=[slide_outputs[i], image_displays[i]],
            show_progress=True
        )

        regenerate_title_btns[i].click(
            fn=regenerate_slide_part_interface,
            inputs=[input_lang, output_lang, url_input, prompt_input, num_slides, target_audience, gr.Slider(value=i, visible=False), gr.Textbox(value="title", visible=False)],
            outputs=[slide_outputs[i]],
            show_progress=True
        )

        regenerate_description_btns[i].click(
            fn=regenerate_slide_part_interface,
            inputs=[input_lang, output_lang, url_input, prompt_input, num_slides, target_audience, gr.Slider(value=i, visible=False), gr.Textbox(value="description", visible=False)],
            outputs=[slide_outputs[i]],
            show_progress=True
        )

        export_btn.click(
        fn=export_webstory,
        inputs=[],
        outputs=[export_output, export_status],
        show_progress=True
    )


# Launch the app
demo.launch()