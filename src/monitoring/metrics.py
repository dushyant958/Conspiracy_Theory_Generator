import time
import logging
import os
import json
import psutil
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/metrics.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("metrics")

class MetricsCollector:
    """
    Collects, stores, and reports metrics about the API and RAG pipeline.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of requests to keep in history
        """
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.recent_requests = deque(maxlen=max_history)
        self.recent_errors = deque(maxlen=max_history)
        self.feedback = {}
        
        # Directory to store metrics logs
        self.metrics_dir = "logs/metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Create periodic metrics logging thread
        self.should_run = True
        self.metrics_thread = threading.Thread(target=self._periodic_metrics_logging)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        logger.info("Metrics collector initialized")
    
    def get_uptime(self) -> float:
        """Get the API uptime in seconds"""
        return time.time() - self.start_time
    
    def get_total_requests(self) -> int:
        """Get the total number of requests processed"""
        return self.request_count
    
    def increment_request_count(self) -> None:
        """Increment the request counter"""
        self.request_count += 1
    
    def increment_error_count(self) -> None:
        """Increment the error counter"""
        self.error_count += 1
    
    def track_response_time(self, response_time: float) -> None:
        """
        Track a response time.
        
        Args:
            response_time: Response time in seconds
        """
        self.response_times.append(response_time)
        
        # Trim response times list if it gets too large
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_average_response_time(self) -> float:
        """Get the average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "connections": len(psutil.net_connections())
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {
                "error": "Failed to get system metrics",
                "message": str(e)
            }
    
    def log_request_metrics(
        self,
        request_id: str,
        query_length: int,
        response_length: int,
        processing_time: float,
        creativity_level: float,
        validity_score: float
    ) -> None:
        """
        Log metrics for a processed request.
        
        Args:
            request_id: Request identifier
            query_length: Length of the user query
            response_length: Length of the generated response
            processing_time: Total processing time
            creativity_level: Creativity level used
            validity_score: Validity score of the response
        """
        timestamp = datetime.now().isoformat()
        
        # Create metrics record
        metrics = {
            "request_id": request_id,
            "timestamp": timestamp,
            "query_length": query_length,
            "response_length": response_length,
            "processing_time": processing_time,
            "creativity_level": creativity_level,
            "validity_score": validity_score
        }
        
        # Add to recent requests
        self.recent_requests.append(metrics)
        
        # Write to file (asynchronously would be better)
        try:
            with open(f"{self.metrics_dir}/requests.log", "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Error writing request metrics: {str(e)}")
    
    def log_error(self, request_id: str, error: str, query: str) -> None:
        """
        Log an error.
        
        Args:
            request_id: Request identifier
            error: Error message
            query: User query that caused the error
        """
        timestamp = datetime.now().isoformat()
        
        # Create error record
        error_record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "error": error,
            "query": query
        }
        
        # Add to recent errors
        self.recent_errors.append(error_record)
        
        # Write to file
        try:
            with open(f"{self.metrics_dir}/errors.log", "a") as f:
                f.write(json.dumps(error_record) + "\n")
        except Exception as e:
            logger.error(f"Error writing error record: {str(e)}")
    
    def log_feedback(self, request_id: str, rating: int, comments: str) -> None:
        """
        Log user feedback.
        
        Args:
            request_id: Request identifier
            rating: User rating (1-5)
            comments: User comments
        """
        timestamp = datetime.now().isoformat()
        
        # Create feedback record
        feedback = {
            "request_id": request_id,
            "timestamp": timestamp,
            "rating": rating,
            "comments": comments
        }
        
        # Store feedback
        self.feedback[request_id] = feedback
        
        # Write to file
        try:
            with open(f"{self.metrics_dir}/feedback.log", "a") as f:
                f.write(json.dumps(feedback) + "\n")
        except Exception as e:
            logger.error(f"Error writing feedback: {str(e)}")
    
    def get_recent_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent requests.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of recent request records
        """
        return list(self.recent_requests)[-limit:]
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error records
        """
        return list(self.recent_errors)[-limit:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metrics summary
        """
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate error rate
        error_rate = (self.error_count / self.request_count) * 100 if self.request_count > 0 else 0
        
        # Calculate average response time
        avg_response_time = self.get_average_response_time()
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        # Compile summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "average_response_time": avg_response_time,
            "system_metrics": system_metrics
        }
        
        return summary
    
    def _periodic_metrics_logging(self) -> None:
        """Periodically log metrics summary (runs in a separate thread)"""
        interval = 300  # Log every 5 minutes
        
        while self.should_run:
            try:
                # Sleep first to allow some metrics to be collected
                time.sleep(interval)
                
                # Get metrics summary
                summary = self.get_metrics_summary()
                
                # Log summary
                logger.info(f"Metrics summary: uptime={summary['uptime_seconds']:.2f}s, "
                           f"requests={summary['request_count']}, "
                           f"errors={summary['error_count']} ({summary['error_rate_percent']:.2f}%), "
                           f"avg_response_time={summary['average_response_time']:.3f}s")
                
                # Write to file
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                with open(f"{self.metrics_dir}/summary_{timestamp}.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error in metrics thread: {str(e)}")
    
    def __del__(self):
        """Clean up when the collector is destroyed"""
        self.should_run = False
        if hasattr(self, 'metrics_thread') and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=1.0)