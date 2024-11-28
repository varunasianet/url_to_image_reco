# import requests
# import json

# # API endpoint
# url = "http://35.207.211.198:7862/generate_web_story"

# # Request payload
# payload = {
#     "input_language": "Kannada",
#     "output_language": "Hindi",
#     "content_input": "https://kannada.asianetnews.com/tv-talk/zee-kannada-puttakkana-makkalu-serial-sahana-trying-to-lead-independent-life-her-husband-marali-going-for-another-marriage-six0z2",
#     "num_slides": 8,
#     "target_audience": "Common",
#     "input_type": "url"
# }

# # Make the API request
# response = requests.post(url, json=payload)

# # Check if the request was successful
# if response.status_code == 200:
#     # Parse the JSON response
#     web_story = response.json()
    
#     # Print the web story
#     print("Web Story:")
#     print(f"Web Story Title: {web_story['webstories']['webstorie_title']}")
    
#     print("\nSlides:")
#     for i, slide in web_story['slides'].items():
#         print(f"\n{i.capitalize()}:")
#         print(f"Title: {slide['title']}")
#         print(f"Description: {slide['description']}")
#         print(f"Image URL: {slide['image_url']}")
#     # Print metadata
#     # print("\nMetadata:")
#     # metadata = web_story['metadata']
#     # print(f"Web Story Title: {metadata['webstorie_title']}")
#     # print(f"English Title: {metadata['english_title']}")
#     # print(f"Summary: {metadata['summary']}")
#     # print(f"Meta Title: {metadata['meta_title']}")
#     # print(f"Meta Description: {metadata['meta_description']}")
#     # print("Meta Keywords:")
#     # for keyword in metadata['meta_keywords']:
#     #     print(f"- {keyword}")
    
#     # if 'json_url' in web_story:
#     #     print(f"\nJSON URL: {web_story['json_url']}")
# else:
#     print(f"Error: {response.status_code}")
#     print(response.text)

import requests
import json

# API endpoint
url= "https://ai-webstories.asianetnews.org/generate_web_story"


# Request payload
payload = {
    "input_language": "Kannada",
    "output_language": "English",
    "content_input": "https://kannada.asianetnews.com/karnataka-districts/union-minister-v-somanna-react-to-railway-projects-from-shivamogga-to-north-karnataka-grg-skg0ta",
    "num_slides": 8,
    "target_audience": "Common",
    "input_type": "url"
}

# Make the API request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    web_story = response.json()
    print(web_story)
    
    # # Print the web story
    # print("Web Story:")
    # print(f"Web Story Title: {web_story['webstories'].get('webstorie_title', 'N/A')}")
    
    # print("\nSlides:")
    # for i, slide in web_story['slides'].items():
    #     print(f"\n{i.capitalize()}:")
    #     print(f"Title: {slide['title']}")
    #     print(f"Description: {slide['description']}")
    #     print(f"Image URL: {slide.get('image_url', 'N/A')}")
    
    # # Print metadata
    # print("\nMetadata:")
    # metadata = web_story['metadata']
    # print(f"Title: {metadata.get('title', 'N/A')}")
    # print(f"Summary: {metadata.get('summary', 'N/A')}")
    # print(f"Meta Title: {metadata.get('meta_title', 'N/A')}")
    # print(f"English Meta Title: {metadata.get('english_meta_title', 'N/A')}")
    # print(f"Meta Description: {metadata.get('meta_description', 'N/A')}")
    # print("Meta Keywords:")
    # for keyword in metadata.get('meta_keywords', []):
    #     print(f"- {keyword}")
    # print("English Meta Keywords:")
    # for keyword in metadata.get('english_meta_keywords', []):
    #     print(f"- {keyword}")
    
    # if 'json_url' in web_story:
    #     print(f"\nJSON URL: {web_story['json_url']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
