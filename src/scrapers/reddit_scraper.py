import praw
import json
import logging
from pathlib import Path
import time
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/reddit_scraper.log"),
        logging.StreamHandler()
    ]
)

class RedditConspiracyScraper:
    def __init__(self, output_dir: str = "data/raw/reddit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PRAW (you'll need to set up Reddit API credentials)
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="ConspiracyTheoryAcademicResearchBot/1.0"
        )
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 100, 
                         time_filter: str = "month") -> List[Dict[str, Any]]:
        """Scrape posts from a specific subreddit"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            logging.info(f"Scraping top {limit} posts from r/{subreddit_name}")
            
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'url': f"https://www.reddit.com{submission.permalink}",
                    'author': str(submission.author),
                    'score': submission.score,
                    'created_utc': submission.created_utc,
                    'text': submission.selftext,
                    'num_comments': submission.num_comments,
                    'comments': [],
                    'subreddit': subreddit_name,
                    'source': 'reddit',
                    'scrape_date': time.strftime('%Y-%m-%d')
                }
                
                # Get top comments
                submission.comments.replace_more(limit=0)  # Skip "more comments" objects
                for comment in submission.comments[:30]:  # Top 30 comments
                    post_data['comments'].append({
                        'id': comment.id,
                        'author': str(comment.author),
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc
                    })
                
                posts.append(post_data)
                
                # Save individual post
                output_file = self.output_dir / f"{subreddit_name}_{submission.id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(post_data, f, ensure_ascii=False, indent=2)
                
                # Respect Reddit's rate limits
                time.sleep(0.5)
            
            return posts
            
        except Exception as e:
            logging.error(f"Error scraping subreddit {subreddit_name}: {str(e)}")
            return []
    
    def scrape_conspiracy_subreddits(self) -> None:
        """Scrape top posts from conspiracy-related subreddits"""
        subreddits = [
            "conspiracy", "UFOs", "conspiracytheories", "conspiracyfact", 
            "conspiracyNOPOL", "HighStrangeness", "aliens"
        ]
        
        for subreddit in subreddits:
            try:
                posts = self.scrape_subreddit(subreddit, limit=200)
                logging.info(f"Scraped {len(posts)} posts from r/{subreddit}")
                
                # Create subreddit summary
                summary_file = self.output_dir / f"{subreddit}_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'subreddit': subreddit,
                        'post_count': len(posts),
                        'scrape_date': time.strftime('%Y-%m-%d')
                    }, f, ensure_ascii=False, indent=2)
                
                # Sleep between subreddits to respect rate limits
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error processing subreddit {subreddit}: {str(e)}")

if __name__ == "__main__":
    scraper = RedditConspiracyScraper()
    scraper.scrape_conspiracy_subreddits()