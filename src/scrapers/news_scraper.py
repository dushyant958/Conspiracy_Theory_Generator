import requests
import json
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import time
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import newspaper
from newspaper import Article
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/news_scraper.log"),
        logging.StreamHandler()
    ]
)

class NewsConspiracyScraper:
    def __init__(self, output_dir: str = "data/raw/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Conspiracy Theory Academic Research Bot/1.0'
        })
        
        # API keys
        self.news_api_key = os.getenv("NEWS_API_KEY")
    
    def search_news_api(self, query: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Search for news articles using News API"""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            logging.info(f"Searching News API for: {query}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            logging.info(f"Found {len(articles)} articles for query: {query}")
            return articles
            
        except Exception as e:
            logging.error(f"Error searching News API for {query}: {str(e)}")
            return []
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a news article URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Natural Language Processing features
            try:
                article.nlp()
            except Exception as nlp_error:
                logging.warning(f"NLP processing failed for {url}: {str(nlp_error)}")
            
            result = {
                'title': article.title,
                'url': url,
                'text': article.text,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'authors': article.authors,
                'top_image': article.top_image,
                'summary': article.summary if hasattr(article, 'summary') else None,
                'keywords': article.keywords if hasattr(article, 'keywords') else [],
                'source': 'news',
                'scrape_date': time.strftime('%Y-%m-%d')
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error extracting content from {url}: {str(e)}")
            return {
                'title': '',
                'url': url,
                'text': '',
                'error': str(e),
                'source': 'news',
                'scrape_date': time.strftime('%Y-%m-%d')
            }
    
    def scrape_conspiracy_news(self) -> None:
        """Scrape news articles related to conspiracy theories"""
        search_queries = [
            "conspiracy theory", "deep state", "government coverup", 
            "secret society", "illuminati", "new world order", 
            "alien coverup", "ufo sighting", "chemtrails", 
            "vaccine conspiracy", "fake moon landing"
        ]
        
        for query in search_queries:
            try:
                # Sanitize query for filename
                query_filename = query.replace(' ', '_').replace('"', '').lower()
                
                # Get articles from News API
                articles = self.search_news_api(query)
                
                # Process and save each article
                processed_count = 0
                for i, article_meta in enumerate(articles):
                    try:
                        url = article_meta.get('url')
                        if not url:
                            continue
                        
                        # Extract full article content
                        article_data = self.extract_article_content(url)
                        
                        # Add metadata from API
                        article_data['source_name'] = article_meta.get('source', {}).get('name')
                        article_data['description'] = article_meta.get('description')
                        article_data['query'] = query
                        
                        # Generate unique filename
                        source_name = article_data['source_name'] or 'unknown'
                        source_name = ''.join(c for c in source_name if c.isalnum() or c in ' _-')
                        
                        filename = f"{query_filename}_{source_name}_{int(time.time())}_{i}.json"
                        output_file = self.output_dir / filename
                        
                        # Save to file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(article_data, f, ensure_ascii=False, indent=2)
                        
                        processed_count += 1
                        
                        # Respect rate limits
                        time.sleep(1)
                        
                    except Exception as article_error:
                        logging.error(f"Error processing article {i} for query '{query}': {str(article_error)}")
                
                logging.info(f"Processed {processed_count} articles for query: {query}")
                
                # Save query summary
                summary_file = self.output_dir / f"{query_filename}_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'query': query,
                        'article_count': processed_count,
                        'scrape_date': time.strftime('%Y-%m-%d')
                    }, f, ensure_ascii=False, indent=2)
                
                # Sleep between queries to respect API limits
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing query '{query}': {str(e)}")

if __name__ == "__main__":
    scraper = NewsConspiracyScraper()
    scraper.scrape_conspiracy_news()