import asyncio
import aiofiles
import aiohttp
import json
import logging
import os
import re
import traceback
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from PIL import Image
from PIL import UnidentifiedImageError
from io import BytesIO
from google.cloud import storage
from google.cloud import aiplatform
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
import hashlib
import html
from google.cloud import aiplatform
import nltk
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess


from utils import (
    llm, regenerate_model, prompt_template, search_prompt_template, regenerate_chain,
    search_chain, fix_content_chain, supported_languages, target_audiences,
    extract_slide_info, validate_metadata, validate_slide_content, preprocess_english_title,
    validate_regenerated_content, load_article, preprocess_text, extract_topics,
    translate_article, generate_keywords, download_and_filter_images, download_image_serp,
    rate_limit, download_image_with_retry, download_image_with_fallback, get_folder_name,
    upload_to_gcs, save_slides_to_file, crawl_images, find_article_content, is_excluded_image,
    check_image_aspect_ratio, get_real_time_data, search_and_scrape
)


logger = logging.getLogger(__name__)

async def generate_web_story(input_language, output_language, content_input, num_slides, target_audience, input_type):
    global generated_web_story
    generated_web_story = {"webstories": {}, "slides": [], "metadata": {}}
    
    if input_language not in supported_languages or output_language not in supported_languages:
        return f"Invalid language selection. Please choose from: {', '.join(supported_languages)}"
    
    try:
        if input_type == "url":
            article_url = content_input
            scraped_images = await crawl_images(article_url)  # Use await here
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

async def article_translation_and_keyword_pipeline(url, source_language):
    logger.info(f"Starting pipeline for URL: {url}, Source Language: {source_language}")
    llm = create_llm()
    
    translated_text = translate_article(llm, url, source_language)
    keywords = generate_keywords(llm, translated_text)
    
    folder_name = get_folder_name(url)
    os.makedirs(folder_name, exist_ok=True)

    metadata = await process_keywords(keywords, folder_name)
    
    return {
        "translated_text": translated_text,
        "keywords": keywords,
        "metadata": metadata
    }

