import requests
from bs4 import BeautifulSoup
import json
import logging
from pathlib import Path
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/wikipedia_scraper.log"),
        logging.StreamHandler()
    ]
)

class WikipediaConspiracyScraper:
    def __init__(self, output_dir: str = "data/raw/wikipedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Conspiracy Theory Academic Research Bot/1.0'
        })
    
    def get_conspiracy_categories(self) -> List[str]:
        """Get list of conspiracy-related categories"""
        try:
            urls = [
                "https://en.wikipedia.org/wiki/Category:Conspiracy_theories",
                "https://en.wikipedia.org/wiki/Category:Conspiracy_theories_by_topic",
                "https://en.wikipedia.org/wiki/Category:UFO_conspiracy_theories",
                "https://en.wikipedia.org/wiki/Category:Political_conspiracy_theories"
            ]
            
            all_category_pages = []
            
            for url in urls:
                logging.info(f"Fetching category: {url}")
                response = self.session.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                category_links = soup.select('div.mw-category a')
                
                for link in category_links:
                    all_category_pages.append({
                        'title': link.text,
                        'url': f"https://en.wikipedia.org{link['href']}"
                    })
                
                # Respect Wikipedia's rate limits
                time.sleep(1)
            
            return all_category_pages
                
        except Exception as e:
            logging.error(f"Error getting conspiracy categories: {str(e)}")
            return []
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single Wikipedia page"""
        try:
            logging.info(f"Scraping page: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.select_one('h1#firstHeading').text
            
            # Get content
            content_div = soup.select_one('div.mw-parser-output')
            paragraphs = content_div.select('p')
            content = '\n\n'.join([p.text for p in paragraphs])
            
            # Get sections
            sections = []
            for heading in content_div.select('h2, h3'):
                section_title = heading.text.replace('[edit]', '').strip()
                section_content = []
                
                element = heading.next_sibling
                while element and not (element.name == 'h2' or element.name == 'h3'):
                    if element.name == 'p':
                        section_content.append(element.text)
                    element = element.next_sibling
                
                sections.append({
                    'title': section_title,
                    'content': '\n\n'.join(section_content)
                })
            
            # Get references
            references = []
            ref_list = soup.select('div.reflist span.reference-text')
            for ref in ref_list:
                references.append(ref.text)
            
            # Result object
            result = {
                'title': title,
                'url': url,
                'content': content,
                'sections': sections,
                'references': references,
                'source': 'wikipedia',
                'scrape_date': time.strftime('%Y-%m-%d')
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error scraping page {url}: {str(e)}")
            return {
                'title': url,
                'url': url,
                'content': '',
                'sections': [],
                'references': [],
                'source': 'wikipedia',
                'scrape_date': time.strftime('%Y-%m-%d'),
                'error': str(e)
            }
    
    def scrape_conspiracy_pages(self, limit: int = None) -> None:
        """Scrape all conspiracy-related pages"""
        categories = self.get_conspiracy_categories()
        
        logging.info(f"Found {len(categories)} conspiracy categories")
        
        for i, category in enumerate(categories[:limit] if limit else categories):
            try:
                page_data = self.scrape_page(category['url'])
                
                # Save to file
                output_file = self.output_dir / f"{page_data['title'].replace(' ', '_')}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                
                logging.info(f"Saved data for: {page_data['title']}")
                
                # Respect Wikipedia's rate limits
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing category {category['title']}: {str(e)}")
                
            # Log progress
            if i > 0 and i % 10 == 0:
                logging.info(f"Progress: {i}/{len(categories)} categories processed")

if __name__ == "__main__":
    scraper = WikipediaConspiracyScraper()
    scraper.scrape_conspiracy_pages(limit=100)  # Limit to 100 pages for testing