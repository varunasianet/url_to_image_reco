import gradio as gr
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from google.cloud import aiplatform
from google.oauth2 import service_account

# Set up Vertex AI credentials

# Set up Vertex AI credentials
credentials = service_account.Credentials.from_service_account_file(
    '/home/varun_saagar/urltoweb/vertexai-key.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

os.environ["PROJECT_ID"] = "asianet-tech-staging"

aiplatform.init(project="asianet-tech-staging", location="asia-southeast1", credentials=credentials)

# Initialize the VertexAI model
llm = VertexAI(
    model_name="gemini-1.5-flash-001",
    max_output_tokens=8192,
    temperature=1,
    top_p=0.95,
    project="asianet-tech-staging",
    location="us-central1",
    credentials=credentials,
    streaming=True,
)

# Create PromptTemplates
translation_template = """Translate the following article from {input_language} to {output_language}. Follow these guidelines strictly:

1. Focus only on the main content of the article, including the title, description, and body.
2. Ignore any metadata, navigation elements, advertisements, or unrelated content.
3. Ensure accuracy and clarity in both meaning and style.
4. Maintain contextual translation, preserving idioms and cultural references when possible.
5. Retain the same theme, tone, and narrative structure as the original.
6. Do not add any extra facts or information not present in the original main content.
7. Aim for approximately {word_limit} words in the translation.
8. Preserve the original formatting, paragraph structure, and section breaks of the main content.
9. Maintain the same level of formality or informality as the source text.

Article content:
{text}

Translated article (focus on main content only):
"""

translation_prompt = PromptTemplate(template=translation_template, input_variables=["input_language", "output_language", "word_limit", "text"])

summary_template = """Summarize the following text in approximately {summary_length} words in {output_language}. Adhere to these guidelines:

1. Focus only on the main content of the article, including the title, description, and body.
2. Ignore any metadata, navigation elements, advertisements, or unrelated content.
3. Capture the main ideas and key points of the original text.
4. Maintain the original tone and perspective.
5. Do not introduce any new information or facts not present in the original main content.
6. Preserve the chronological or logical order of ideas from the original text.
7. Use clear and concise language while retaining the essence of the original style.
8. Include a brief mention of the main topic or theme in the opening sentence.
9. Avoid personal opinions or interpretations.
10. Ensure the summary is in {output_language}, even if the original text is in a different language.

Original text:
{text}

Summary of main content in {output_language}:
"""
summary_prompt = PromptTemplate(template=summary_template, input_variables=["summary_length", "output_language", "text"])


# Update the idea generation template
idea_generation_template = """Based on the main content of the following article, generate 5 unique and interesting ideas for new articles. Follow these guidelines:

1. Focus only on the main content of the article, including the title, description, and body.
2. Ignore any metadata, navigation elements, advertisements, or unrelated content.
3. Each idea should be directly related to the theme or subject matter of the original article's main content.
4. Present each idea as a brief title or concept, separated by newlines.
5. Ensure that the ideas explore different aspects or angles of the main topic.
6. Do not introduce completely unrelated topics or themes.
7. Aim for a mix of broader and more specific ideas within the same general subject.
8. Consider potential follow-up stories, related issues, or deeper dives into subtopics.
9. Avoid sensationalism or clickbait-style ideas.
10. Ensure that each idea is distinct and not a mere rephrasing of another.

Original article content:
{text}

5 New Article Ideas (based on main content only):
"""

idea_generation_prompt = PromptTemplate(template=idea_generation_template, input_variables=["text"])

article_generation_template = """Generate a new article based on the following idea, in the style and format of the original article. Adhere to these guidelines strictly:

1. Write the article in {language}, aiming for approximately {word_limit} words.
2. Maintain the same theme, tone, and style as the original article.
3. Do not add any extra facts or information that wouldn't be naturally derived from the original article's context.
4. Use a similar structure and formatting as the original article (e.g., subheadings, paragraph lengths).
5. Ensure the content is contextually relevant to the culture and region implied by the chosen language.
6. Include a compelling headline that reflects the new article's focus.
7. Maintain the same level of technicality or simplicity as the original article.
8. If the original included quotes or statistics, incorporate similar elements (without fabricating sources).
9. Conclude the article in a manner consistent with the original's style.

Original article content:
{text}

New article idea: {idea}

Generated article:
"""

article_generation_prompt = PromptTemplate(template=article_generation_template, input_variables=["language", "word_limit", "text", "idea"])

# Create LLMChains
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
idea_generation_chain = LLMChain(llm=llm, prompt=idea_generation_prompt)
article_generation_chain = LLMChain(llm=llm, prompt=article_generation_prompt)

def load_and_split_text(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    
    # Join the chunks into a single string
    full_text = " ".join([chunk.page_content for chunk in chunks])
    return full_text


# Update the translate_and_summarize function
def translate_and_summarize(url, input_language, output_language, word_limit, summary_length):
    full_text = load_and_split_text(url)
    translated_text = translation_chain.run(text=full_text, input_language=input_language, output_language=output_language, word_limit=word_limit)
    summary = summary_chain.run(text=translated_text, output_language=output_language, summary_length=summary_length)
    return translated_text, summary



def generate_ideas(url):
    full_text = load_and_split_text(url)
    ideas = idea_generation_chain.run(text=full_text)
    return ideas.split('\n')


def generate_article(url, idea, language, word_limit):
    full_text = load_and_split_text(url)
    new_article = article_generation_chain.run(text=full_text, idea=idea, language=language, word_limit=word_limit)
    return new_article

summary_prompt = PromptTemplate(template=summary_template, input_variables=["summary_length", "output_language", "text"])

# Update the summary_chain
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Create Gradio interface
languages = ["Malayalam", "English", "Kannada", "Telugu", "Tamil", "Bangla", "Hindi", "Marathi"]


with gr.Blocks() as app:
    gr.Markdown("# Article Translator, Summarizer, and Idea Generator")
    
    with gr.Tab("Translate and Summarize"):
        url_input = gr.Textbox(label="Article URL")
        input_lang = gr.Dropdown(choices=languages, label="Input Language")
        output_lang = gr.Dropdown(choices=languages, label="Output Language")
        word_limit = gr.Slider(minimum=300, maximum=1000, value=300, step=50, label="Translation Word Limit")
        summary_length = gr.Slider(minimum=50, maximum=200, value=50, step=10, label="Summary Word Length")
        translate_btn = gr.Button("Translate and Summarize")
        translated_output = gr.Textbox(label="Translated Article")
        summary_output = gr.Textbox(label="Summary")
        
        translate_btn.click(
            translate_and_summarize,
            inputs=[url_input, input_lang, output_lang, word_limit, summary_length],
            outputs=[translated_output, summary_output]
        )

    
    with gr.Tab("Generate Ideas and New Article"):
        idea_url_input = gr.Textbox(label="Article URL for Idea Generation")
        generate_ideas_btn = gr.Button("Generate Ideas")
        ideas_output = gr.Radio(label="Select an Idea", choices=[])
        idea_lang = gr.Dropdown(choices=languages, label="New Article Language")
        idea_word_limit = gr.Slider(minimum=300, maximum=1000, value=500, step=50, label="New Article Word Limit")
        generate_article_btn = gr.Button("Generate New Article")
        new_article_output = gr.Textbox(label="Generated Article")
        
        def update_ideas(url):
            ideas = generate_ideas(url)
            return gr.Radio(choices=ideas, label="Select an Idea")
        
        generate_ideas_btn.click(
            update_ideas,
            inputs=[idea_url_input],
            outputs=[ideas_output]
        )
        
        generate_article_btn.click(
            generate_article,
            inputs=[idea_url_input, ideas_output, idea_lang, idea_word_limit],
            outputs=[new_article_output]
        )

import time


if __name__ == "__main__":
    app.concurrency_limit = 10 # This replaces the concurrency_limit
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False  # Set to True if you want to generate a public link
    )
    
    # Keep the script running
    while True:
        time.sleep(1)

