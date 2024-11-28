import logging
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from langchain_core.prompts import PromptTemplate
from langchain_core.chains import LLMChain
from langchain_google_vertexai import VertexAI

from utils import llm, preprocess_english_title

logger = logging.getLogger(__name__)

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

def create_llm():
    return VertexAI(
        model_name="gemini-1.5-flash-001",
        max_output_tokens=8192,
        temperature=0.7,
        top_p=0.95,
    )

def load_article(url):
    logger.info(f"Loading article from URL: {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        if not documents:
            logger.warning(f"No content loaded from URL: {url}")
            return None
        return documents
    except Exception as e:
        logger.error(f"Error loading article from URL {url}: {str(e)}")
        return None
