from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import logging
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from .router import router as conspiracy_router
from ..monitoring.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api")

# Initialize metrics collector
metrics = MetricsCollector()

# Create FastAPI app
app = FastAPI(
    title="Conspiracy Theory Generator API",
    description="API for generating conspiracy theories using RAG techniques with fact-checking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Track request
    metrics.increment_request_count()
    
    try:
        response = await call_next(request)
        
        # Add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Track response time in metrics
        metrics.track_response_time(process_time)
        
        return response
    except Exception as e:
        # Track error
        metrics.increment_error_count()
        logger.error(f"Request error: {str(e)}")
        
        # Return error response
        process_time = time.time() - start_time
        error_response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
        error_response.headers["X-Process-Time"] = str(process_time)
        return error_response

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for validation errors to provide cleaner error messages"""
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    # Log the error
    logger.warning(f"Validation error: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": errors},
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch unhandled exceptions"""
    # Log the error
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Track the error
    metrics.increment_error_count()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )

# Include routers
app.include_router(conspiracy_router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": "Conspiracy Theory Generator API",
        "version": "1.0.0",
        "description": "API for generating conspiracy theories using RAG with fact-checking",
        "endpoints": {
            "conspiracy_theory": "/api/v1/conspiracies",
            "health_check": "/api/v1/health",
            "history": "/api/v1/conspiracies/history",
            "feedback": "/api/v1/feedback"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("API_PORT", 8000))
    
    # Run the API server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )