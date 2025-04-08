import logging
import traceback
import inspect
import functools
import sys
import time
from typing import Callable, Any, Dict, Optional, Type
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/errors.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("error_handler")

# Directory for error logs
ERROR_LOG_DIR = "logs/errors"
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

class CustomException(Exception):
    """Base class for all custom exceptions in the application."""
    
    def __init__(self, message: str, code: str = None, details: Dict[str, Any] = None):
        """
        Initialize the custom exception.
        
        Args:
            message: Error message
            code: Error code (optional)
            details: Additional error details (optional)
        """
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

class DataError(CustomException):
    """Exception raised for errors related to data processing or validation."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, code="DATA_ERROR", details=details)

class ModelError(CustomException):
    """Exception raised for errors related to LLM or model operations."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, code="MODEL_ERROR", details=details)

class DatabaseError(CustomException):
    """Exception raised for errors related to database operations."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, code="DATABASE_ERROR", details=details)

class APIError(CustomException):
    """Exception raised for errors related to API operations."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, code="API_ERROR", details=details)

def error_boundary(
    fallback_return: Any = None,
    reraise: bool = False,
    log_level: int = logging.ERROR,
    exclude_exceptions: Optional[Type[Exception]] = None
) -> Callable:
    """
    Decorator to add error handling to a function.
    
    Args:
        fallback_return: Value to return if an exception occurs
        reraise: Whether to re-raise the exception after handling
        log_level: Logging level for the error
        exclude_exceptions: Exception types to not handle
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Skip handling for excluded exceptions
                if exclude_exceptions and isinstance(e, exclude_exceptions):
                    raise
                
                # Get details about the function and error
                module_name = func.__module__
                func_name = func.__name__
                line_no = inspect.trace()[-1][2]
                
                # Generate error details
                error_details = {
                    "timestamp": datetime.now().isoformat(),
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "module": module_name,
                    "function": func_name,
                    "line": line_no,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "traceback": traceback.format_exc()
                }
                
                # Log the error
                logger.log(
                    log_level,
                    f"Error in {module_name}.{func_name}: {str(e)}",
                    extra={"error_details": error_details}
                )
                
                # Log detailed error to file
                try:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    filename = f"{ERROR_LOG_DIR}/error_{timestamp}_{module_name}_{func_name}.json"
                    with open(filename, "w") as f:
                        json.dump(error_details, f, indent=2)
                except Exception as log_error:
                    logger.error(f"Failed to write error log: {str(log_error)}")
                
                # Re-raise if specified
                if reraise:
                    raise
                
                # Return fallback value
                return fallback_return
        
        return wrapper
    
    return decorator

def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    # Skip KeyboardInterrupt exceptions
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the exception
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Log detailed error to file
    try:
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "exception_type": exc_type.__name__,
            "exception_message": str(exc_value),
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{ERROR_LOG_DIR}/unhandled_error_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(error_details, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write unhandled error log: {str(e)}")

# Set the global exception handler
sys.excepthook = handle_exception

class Timer:
    """
    Context manager for timing code execution.
    
    Usage:
        with Timer("Operation name") as timer:
            # Code to time
            
        print(f"Operation took {timer.duration} seconds")
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Initialize the timer.
        
        Args:
            name: Name of the operation being timed
            log_level: Logging level for the timing message
        """
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the timer and log the duration"""
        self.duration = time.time() - self.start_time
        logger.log(
            self.log_level,
            f"Operation '{self.name}' completed in {self.duration:.3f} seconds"
        )

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Type[Exception] = Exception
) -> Callable:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        exceptions: Exception types to catch and retry
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.warning(
                            f"Function {func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise
                    
                    logger.info(
                        f"Attempt {attempt}/{max_attempts} for {func.__name__} failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
            
            # This should never be reached, but just in case
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

class RequestsMonitor:
    """
    Class to monitor and track API requests for rate limiting and debugging.
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize the requests monitor.
        
        Args:
            window_size: Size of the sliding window in seconds for rate limiting
        """
        self.window_size = window_size
        self.requests = []
        self.counters = {}
    
    def record_request(self, endpoint: str, status_code: int, duration: float):
        """
        Record a request to an endpoint.
        
        Args:
            endpoint: API endpoint that was called
            status_code: HTTP status code of the response
            duration: Duration of the request in seconds
        """
        timestamp = time.time()
        
        # Add request to history
        self.requests.append({
            "timestamp": timestamp,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration": duration
        })
        
        # Update counters
        if endpoint not in self.counters:
            self.counters[endpoint] = {
                "total": 0,
                "success": 0,
                "errors": 0,
                "avg_duration": 0
            }
        
        counter = self.counters[endpoint]
        counter["total"] += 1
        
        if 200 <= status_code < 300:
            counter["success"] += 1
        else:
            counter["errors"] += 1
        
        # Update average duration using rolling average
        counter["avg_duration"] = (
            (counter["avg_duration"] * (counter["total"] - 1) + duration) / counter["total"]
        )
        
        # Clean up old requests
        self._cleanup_old_requests(timestamp)
    
    def _cleanup_old_requests(self, current_time: float):
        """
        Remove requests older than the window size.
        
        Args:
            current_time: Current timestamp
        """
        cutoff = current_time - self.window_size
        self.requests = [req for req in self.requests if req["timestamp"] >= cutoff]
    
    def get_rate(self, endpoint: str = None) -> int:
        """
        Get the current request rate for an endpoint or all endpoints.
        
        Args:
            endpoint: Specific endpoint to get rate for (or None for all)
        
        Returns:
            Number of requests in the current window
        """
        if endpoint:
            return len([req for req in self.requests if req["endpoint"] == endpoint])
        return len(self.requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics for all endpoints.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "requests_in_window": len(self.requests),
            "endpoints": self.counters
        }

def rate_limit(
    max_calls: int,
    time_window: int = 60,
    key_func: Callable = None
) -> Callable:
    """
    Decorator to implement rate limiting for function calls.
    
    Args:
        max_calls: Maximum number of calls allowed in the time window
        time_window: Time window in seconds
        key_func: Function to generate a key for the rate limit (e.g., based on args)
    
    Returns:
        Decorated function
    """
    # Store call history
    call_history = {}
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__
            
            # Initialize call history for this key
            if key not in call_history:
                call_history[key] = []
            
            # Get current time
            current_time = time.time()
            
            # Remove calls outside the time window
            call_history[key] = [
                t for t in call_history[key] if current_time - t <= time_window
            ]
            
            # Check if rate limit exceeded
            if len(call_history[key]) >= max_calls:
                oldest_call = call_history[key][0]
                wait_time = time_window - (current_time - oldest_call)
                
                logger.warning(
                    f"Rate limit exceeded for {key}. "
                    f"Max {max_calls} calls per {time_window} seconds. "
                    f"Try again in {wait_time:.2f} seconds."
                )
                
                raise APIError(
                    f"Rate limit exceeded. Try again later.",
                    details={
                        "max_calls": max_calls,
                        "time_window": time_window,
                        "wait_time": wait_time
                    }
                )
            
            # Record this call
            call_history[key].append(current_time)
            
            # Execute the function
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_api_call(func: Callable) -> Callable:
    """
    Decorator to log API calls with request and response details.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get endpoint name from function or kwargs
        endpoint = kwargs.get("endpoint", func.__name__)
        
        # Log the request
        logger.info(
            f"API Request: {endpoint}",
            extra={
                "args": str(args),
                "kwargs": {k: v for k, v in kwargs.items() if k != "api_key"}
            }
        )
        
        # Time the API call
        start_time = time.time()
        try:
            # Execute the API call
            response = func(*args, **kwargs)
            status_code = getattr(response, "status_code", 200)
            duration = time.time() - start_time
            
            # Log the response
            log_level = logging.INFO if 200 <= status_code < 300 else logging.WARNING
            logger.log(
                log_level,
                f"API Response: {endpoint} ({status_code}) in {duration:.3f}s"
            )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            
            # Log the error
            logger.error(
                f"API Error: {endpoint} - {str(e)} in {duration:.3f}s",
                exc_info=True
            )
            
            raise
    
    return wrapper

# Optional: Add a metrics reporting function that could connect to a monitoring system
def report_metrics(metrics_url: Optional[str] = None):
    """
    Report metrics to a monitoring system.
    
    Args:
        metrics_url: URL of the metrics endpoint (optional)
    """
    if not metrics_url:
        return
    
    try:
        # Collect system metrics
        memory_usage = sys.getsizeof(sys.modules) / (1024 * 1024)  # in MB
        
        # Collect error metrics
        error_count = len(os.listdir(ERROR_LOG_DIR)) if os.path.exists(ERROR_LOG_DIR) else 0
        
        # Prepare metrics payload
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "memory_usage_mb": memory_usage,
            "error_count": error_count,
            "python_version": sys.version,
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown"
        }
        
        # Here you would send metrics to your monitoring system
        # For example, using requests:
        # import requests
        # response = requests.post(metrics_url, json=metrics)
        
        logger.debug(f"Metrics reported: {metrics}")
    except Exception as e:
        logger.error(f"Failed to report metrics: {str(e)}")