import requests
import json
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import time
import os
from typing import List, Dict, Any
import re
import PyPDF2
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/govt_docs_scraper.log"),
        logging.StreamHandler()
    ]
)

class GovtDocScraper:
    def __init__(self, output_dir: str = "data/raw/government"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Conspiracy Theory Academic Research Bot/1.0'
        })
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF binary content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def scrape_cia_reading_room(self, search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Scrape declassified documents from CIA FOIA Reading Room"""
        documents = []
        try:
            # Format search term for URL
            formatted_search = search_term.replace(' ', '+')
            
            # FOIA Reading Room search URL
            url = f"https://www.cia.gov/readingroom/search/site/{formatted_search}"
            
            logging.info(f"Searching CIA Reading Room for: {search_term}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result items
            results = soup.select('li.search-result')
            
            for i, result in enumerate(results[:limit]):
                try:
                    # Get title and link
                    title_element = result.select_one('h3.title a')
                    title = title_element.text.strip() if title_element else "Unknown Title"
                    link = title_element['href'] if title_element else None
                    
                    if not link:
                        continue
                        
                    # Make link absolute
                    if link.startswith('/'):
                        link = f"https://www.cia.gov{link}"
                    
                    # Get document page
                    doc_response = self.session.get(link)
                    doc_soup = BeautifulSoup(doc_response.text, 'html.parser')
                    
                    # Get metadata
                    meta_items = {}
                    meta_elements = doc_soup.select('div.field-item')
                    for elem in meta_elements:
                        label = elem.select_one('div.field-label')
                        value = elem.select_one('div.field-item')
                        
                        if label and value:
                            key = label.text.strip().replace(':', '')
                            meta_items[key] = value.text.strip()
                    
                    # Get PDF link if available
                    pdf_link = None
                    pdf_element = doc_soup.select_one('a[href$=".pdf"]')
                    if pdf_element:
                        pdf_link = pdf_element['href']
                        if pdf_link.startswith('/'):
                            pdf_link = f"https://www.cia.gov{pdf_link}"
                    
                    # Get text content
                    text_content = ""
                    if pdf_link:
                        pdf_response = self.session.get(pdf_link)
                        if pdf_response.status_code == 200:
                            text_content = self.extract_text_from_pdf(pdf_response.content)
                    
                    # Create document record
                    document = {
                        'title': title,
                        'url': link,
                        'pdf_url': pdf_link,
                        'metadata': meta_items,
                        'text_content': text_content,
                        'search_term': search_term,
                        'source': 'cia_reading_room',
                        'scrape_date': time.strftime('%Y-%m-%d')
                    }
                    
                    documents.append(document)
                    
                    # Save individual document
                    sanitized_title = re.sub(r'[^\w\-]', '_', title)[:50]  # Limit length
                    output_file = self.output_dir / f"cia_{sanitized_title}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(document, f, ensure_ascii=False, indent=2)
                    
                    logging.info(f"Saved document: {title}")
                    
                    # Respect site's rate limits
                    time.sleep(2)
                    
                except Exception as doc_error:
                    logging.error(f"Error processing document {i}: {str(doc_error)}")
            
            return documents
            
        except Exception as e:
            logging.error(f"Error scraping CIA Reading Room for {search_term}: {str(e)}")
            return []
    
    def scrape_fbi_vault(self, topic: str) -> List[Dict[str, Any]]:
        """Scrape declassified documents from FBI Vault"""
        documents = []
        try:
            # The FBI Vault has specific topic pages, not a search function
            # This is a simplified approach - in reality you'd map conspiracy topics to vault URLs
            
            url = f"https://vault.fbi.gov/search?SearchableText={topic}"
            
            logging.info(f"Checking FBI Vault for: {topic}")
            response = self.session.get(url)
            
            # Basic processing - would need to be customized for the actual structure
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result items (this selector would need adjustment)
            results = soup.select('div.result-item')
            
            for result in results:
                # Basic extraction - would need customization
                title = result.select_one('h3').text if result.select_one('h3') else "Unknown"
                link = result.select_one('a')['href'] if result.select_one('a') else None
                
                if not link:
                    continue
                
                # Create document record
                document = {
                    'title': title,
                    'url': link,
                    'topic': topic,
                    'source': 'fbi_vault',
                    'scrape_date': time.strftime('%Y-%m-%d')
                }
                
                documents.append(document)
                
                # Save individual document
                sanitized_title = re.sub(r'[^\w\-]', '_', title)[:50]  # Limit length
                output_file = self.output_dir / f"fbi_{sanitized_title}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(document, f, ensure_ascii=False, indent=2)
                
                # Respect site's rate limits
                time.sleep(2)
            
            return documents
            
        except Exception as e:
            logging.error(f"Error scraping FBI Vault for {topic}: {str(e)}")
            return []
    
    def scrape_government_documents(self) -> None:
        """Scrape conspiracy-related government documents"""
        # Search terms for CIA Reading Room
        cia_search_terms = [
            "UFO", "Unidentified Flying", "MKUltra", "Project Stargate", 
            "ESP", "Psychic", "CIA Conspiracy", "Kennedy Assassination", 
            "Area 51", "Roswell", "Remote Viewing"
        ]
        
        # Topics for FBI Vault
        fbi_topics = [
            "UFO", "Unexplained", "Majestic", "Roswell", 
            "Conspiracy", "Secret Society"
        ]
        
        # Scrape CIA Reading Room
        for term in cia_search_terms:
            try:
                docs = self.scrape_cia_reading_room(term)
                logging.info(f"Scraped {len(docs)} CIA documents for term: {term}")
                
                # Sleep between searches to respect rate limits
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Error processing CIA search term {term}: {str(e)}")
        
        # Scrape FBI Vault
        for topic in fbi_topics:
            try:
                docs = self.scrape_fbi_vault(topic)
                logging.info(f"Scraped {len(docs)} FBI documents for topic: {topic}")
                
                # Sleep between searches to respect rate limits
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Error processing FBI topic {topic}: {str(e)}")

if __name__ == "__main__":
    scraper = GovtDocScraper()
    scraper.scrape_government_documents()