import asyncio
import aiohttp
import aiofiles
import urllib.parse
import logging
import ssl
import certifi
import shutil
import traceback
from urllib.parse import urlparse, urljoin, unquote
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from PIL import UnidentifiedImageError
from google.cloud import storage
from google.cloud import aiplatform
from aiolimiter import AsyncLimiter

from utils import (
    SERP_API_KEY, MAX_IMAGES_PER_KEYWORD, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, BUCKET_NAME, BUCKET_FOLDER,
    upload_to_gcs, get_folder_name, is_excluded_image, check_image_aspect_ratio
)

logger = logging.getLogger(__name__)

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
                download_tasks.append(download_image_with_retry(img_url, full_path, i+1, len(high_quality_images), keyword, i+1))
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
        
async def download_image_serp(img_url, filename, index, total, keyword, image_count):
    
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

async def download_image_with_retry(img_url, filename, index, total, keyword, image_count, max_retries=3):
    async with rate_limit:
        for attempt in range(max_retries):
            try:
                result = await download_image_serp(img_url, filename, index, total, keyword, image_count)
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

                parsed_url = urllib.parse.urlparse(url)
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

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()
