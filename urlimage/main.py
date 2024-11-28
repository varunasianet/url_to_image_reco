from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import uvicorn

from utils import (
    validate_input,
    log_error
)
from web_story_generation import generate_web_story, regenerate_slide_part
from nlp_utils import process_article


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class ArticleRequest(BaseModel):
    url: str = None
    text: str = None
    source_language: str

class WebStoryRequest(BaseModel):
    input_language: str
    output_language: str
    content_input: str
    num_slides: int = 8
    target_audience: str
    input_type: str

class RegenerateSlideRequest(BaseModel):
    input_language: str
    output_language: str
    content_input: str
    num_slides: int = 8
    target_audience: str
    slide_index: int = None
    part_to_regenerate: str
    input_type: str

# Define routes
@app.post("/process_article")
async def process_article(request: ArticleRequest):
    try:
        result = await process_article(request.url, request.source_language)
        return result
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_web_story")
async def generate_web_story_api(request: WebStoryRequest):
    logger.info(f"Received data: {request.dict()}")
    
    is_valid, error_message = validate_input(request.dict())
    if not is_valid:
        return JSONResponse(content={"error": error_message}, status_code=400)

    if not request.content_input:
        return JSONResponse(content={"error": "Please provide content_input"}, status_code=400)

    try:
        generated_story = await generate_web_story(
            request.input_language,
            request.output_language,
            request.content_input,
            request.num_slides,
            request.target_audience,
            request.input_type
        )
        
        if isinstance(generated_story, dict) and "error" in generated_story:
            return JSONResponse(content=generated_story, status_code=400)
        
        if isinstance(generated_story, str):
            return JSONResponse(content={"error": generated_story}, status_code=400)
        
        return JSONResponse(content=generated_story, status_code=200)
    except Exception as e:
        log_error(e, "generate_web_story_api")
        return JSONResponse(content={"error": "An unexpected error occurred"}, status_code=500)

@app.post("/regenerate_slide_part")
async def regenerate_slide_part_api(request: RegenerateSlideRequest):
    try:
        regenerated_part = regenerate_slide_part(
            request.input_language,
            request.output_language,
            request.content_input,
            request.num_slides,
            request.slide_index + 1 if request.slide_index is not None else None,
            request.part_to_regenerate,
            "",
            "",
            request.target_audience,
            request.input_type
        )
        
        if isinstance(regenerated_part, dict) and "error" in regenerated_part:
            return JSONResponse(content=regenerated_part, status_code=400)
        
        response = {
            "regenerated_part": html.unescape(regenerated_part),
            "json_url": generated_web_story.get("json_url", "")
        }
        
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        log_error(e, "regenerate_slide_part_api")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Main execution
if __name__ == "__main__":
    import uvicorn
    
    # Check and create GCS bucket if necessary
    bucket_name = "urltowebstories"
    if not check_gcs_bucket(bucket_name):
        logger.warning(f"GCS bucket '{bucket_name}' does not exist or is not accessible.")
        if create_gcs_bucket(bucket_name):
            logger.info(f"Created GCS bucket '{bucket_name}'.")
        else:
            logger.error(f"Failed to create GCS bucket '{bucket_name}'. Exiting.")
            exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=7863)
