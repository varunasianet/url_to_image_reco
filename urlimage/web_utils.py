import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

logger = logging.getLogger(__name__)

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
