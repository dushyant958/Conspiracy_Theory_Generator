import logging
import time
import argparse
from pathlib import Path
import json
from datetime import datetime

# Import scrapers
from wikipedia_scraper import WikipediaConspiracyScraper
from reddit_scraper import RedditConspiracyScraper
from news_scraper import NewsConspiracyScraper
from government_docs_scraper import GovtDocScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/data_collection.log"),
        logging.StreamHandler()
    ]
)

class DataCollectionOrchestrator:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self.metadata_file = self.output_dir / "collection_metadata.json"
        
        # Initialize scrapers
        self.wikipedia_scraper = WikipediaConspiracyScraper()
        self.reddit_scraper = RedditConspiracyScraper()
        self.news_scraper = NewsConspiracyScraper()
        self.govt_scraper = GovtDocScraper()
    
    def run_collection(self, sources: list = None) -> None:
        """Run data collection for specified or all sources"""
        start_time = time.time()
        
        if sources is None or not sources:
            sources = ["wikipedia", "reddit", "news", "government"]
        
        collection_metadata = {
            "start_time": datetime.now().isoformat(),
            "sources": sources,
            "status": "in_progress"
        }
        
        # Save initial metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(collection_metadata, f, ensure_ascii=False, indent=2)
        
        # Run scrapers based on selected sources
        try:
            if "wikipedia" in sources:
                logging.info("Starting Wikipedia scraping...")
                self.wikipedia_scraper.scrape_conspiracy_pages(limit=None)
                logging.info("Wikipedia scraping completed")
            
            if "reddit" in sources:
                logging.info("Starting Reddit scraping...")
                self.reddit_scraper.scrape_conspiracy_subreddits()
                logging.info("Reddit scraping completed")
            
            if "news" in sources:
                logging.info("Starting News scraping...")
                self.news_scraper.scrape_conspiracy_news()
                logging.info("News Scraping completed")

            if "news" in sources:
                logging.info("Starting News scraping...")
                self.news_scraper.scrape_conspiracy_articles()
                logging.info("News scraping completed")
            
            if "government" in sources:
                logging.info("Starting Government documents scraping...")
                self.govt_scraper.scrape_declassified_documents()
                logging.info("Government documents scraping completed")
            
            # Update metadata with completion information
            collection_metadata["status"] = "completed"
            collection_metadata["end_time"] = datetime.now().isoformat()
            collection_metadata["duration_seconds"] = time.time() - start_time
            collection_metadata["file_count"] = self._count_collected_files()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(collection_metadata, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Data collection completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            # Update metadata with error information
            collection_metadata["status"] = "failed"
            collection_metadata["error"] = str(e)
            collection_metadata["end_time"] = datetime.now().isoformat()
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(collection_metadata, f, ensure_ascii=False, indent=2)
            
            logging.error(f"Data collection failed: {str(e)}")
            raise
    
    def _count_collected_files(self) -> int:
        """Count the number of files collected in this run"""
        return len(list(self.output_dir.glob("*.json")))
    
    def save_data(self, source: str, data: list, batch_id: str = None) -> None:
        """Save collected data to JSON files"""
        if not data:
            logging.warning(f"No data to save for source: {source}")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        batch_suffix = f"_{batch_id}" if batch_id else ""
        filename = f"{source}_{timestamp}{batch_suffix}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved {len(data)} items from {source} to {filepath}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Conspiracy Theory Data Collection")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Directory to store collected data")
    parser.add_argument("--sources", nargs="+", 
                       choices=["wikipedia", "reddit", "news", "government"],
                       help="Specific sources to collect data from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create log directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    orchestrator = DataCollectionOrchestrator(output_dir=args.output_dir)
    orchestrator.run_collection(sources=args.sources)    