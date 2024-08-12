import requests
import feedparser
from datetime import datetime
import json
from google.cloud.sql.connector import Connector
import sqlalchemy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_rss_feed(url):
    # Fetch the RSS feed
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch RSS feed. Status code: {response.status_code}")
        return None
    
    # Parse the RSS feed
    feed = feedparser.parse(response.content)
    
    # Extract entries
    webstories = []
    for entry in feed.entries:
        webstory_id = entry.get('id', '').split('/')[-1]  # Assuming the ID is the last part of the 'id' field
        webstory_data = {
            'webstory_id': webstory_id,
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'description': entry.get('summary', ''),
            'pubDate': entry.get('published', ''),
            'image': next((content['url'] for content in entry.get('media_content', []) if content.get('medium') == 'image'), ''),
            'category': [tag.term for tag in entry.get('tags', [])],
        }
        webstories.append(webstory_data)
    
    return webstories

def init_connection_engine():
    instance_connection_name = "asianet-tech-staging:asia-south1:webstories-asianet-db"
    db_user = "webstories-user"
    db_pass = "asianetweb"
    db_name = "webstoriesrss"

    connector = Connector()

    def getconn():
        conn = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    engine = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
    )
    return engine

def insert_webstories(engine, webstories):
    with engine.connect() as connection:
        metadata = sqlalchemy.MetaData()
        webstories_table = sqlalchemy.Table(
            'webstories',
            metadata,
            sqlalchemy.Column('webstory_id', sqlalchemy.String(255), primary_key=True),
            sqlalchemy.Column('title', sqlalchemy.String(255)),
            sqlalchemy.Column('link', sqlalchemy.String(255)),
            sqlalchemy.Column('description', sqlalchemy.Text),
            sqlalchemy.Column('pubDate', sqlalchemy.DateTime),
            sqlalchemy.Column('image', sqlalchemy.String(255)),
            sqlalchemy.Column('category', sqlalchemy.JSON),
        )
        
        metadata.create_all(engine)
        
        for webstory in webstories:
            insert_stmt = sqlalchemy.insert(webstories_table).values(
                webstory_id=webstory['webstory_id'],
                title=webstory['title'],
                link=webstory['link'],
                description=webstory['description'],
                pubDate=datetime.strptime(webstory['pubDate'], '%a, %d %b %Y %H:%M:%S %z'),
                image=webstory['image'],
                category=json.dumps(webstory['category'])
            )
            connection.execute(insert_stmt)
        
        connection.commit()

# URL of the RSS feed
rss_url = "https://newsable.asianetnews.com/rss/dh/webstories"

try:
    # Scrape the RSS feed
    scraped_data = scrape_rss_feed(rss_url)

    if scraped_data:
        logger.info(f"Scraped {len(scraped_data)} webstories.")
        
        # Initialize the database connection
        db_engine = init_connection_engine()
        
        # Insert the webstories into the database
        insert_webstories(db_engine, scraped_data)
        
        logger.info("Data successfully saved to Cloud SQL database.")
    else:
        logger.warning("No data scraped from RSS feed.")
except Exception as e:
    logger.error(f"An error occurred: {str(e)}")