from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, constr, StringConstraints
from typing import List, Dict, Any, Optional
import logging
import time
import os
import json

# Local imports
from ..rag.pipeline import RAGPipeline, create_pipeline_from_config
from ..monitoring.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_router.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api_router")

# Initialize the RAG pipeline
config_path = os.getenv("CONFIG_PATH", "config/rag_config.json")
try:
    pipeline = create_pipeline_from_config(config_path)
    logger.info(f"Successfully loaded RAG pipeline from config: {config_path}")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
    pipeline = None  # Will be checked later and raise appropriate error

# Initialize metrics collector
metrics = MetricsCollector()

# Define API models
class ConspiracyQuery(BaseModel):
    query: constr(min_length=5, max_length=1000) = Field(..., description="User query text")
    creativity_level: float = Field(0.7, ge=0.0, le=1.0, description="Level of creativity for response generation (0.0-1.0)")
    use_fact_check: bool = Field(True, description="Whether to apply fact-checking")
    filter_criteria: Optional[Dict[str, Any]] = Field(None, description="Optional filters for document retrieval")

class ConspiracyResponse(BaseModel):
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated conspiracy theory")
    validity_score: float = Field(..., description="Fact-check validity score (0.0-1.0)")
    sources: List[Dict[str, Any]] = Field(..., description="Top sources used")
    processing_time_seconds: float = Field(..., description="Total processing time")

# Create router
router = APIRouter(prefix="/api/v1", tags=["conspiracy"])

def check_pipeline():
    """Dependency to check if pipeline is initialized correctly"""
    if pipeline is None:
        logger.error("RAG pipeline not initialized")
        raise HTTPException(status_code=503, detail="Service unavailable: RAG pipeline not initialized")
    return pipeline

@router.get("/health", summary="Check API health")
async def health_check():
    """Endpoint to check API health status"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "RAG pipeline not initialized"}
        )
        
    # Check connection to vector DB
    try:
        # Simple check (adjust based on your implementation)
        db_status = "ok" if hasattr(pipeline, "vector_db") else "error"
    except:
        db_status = "error"
    
    # Check LLM availability
    try:
        # Simple ping to LLM (could be improved)
        llm_status = "ok" if hasattr(pipeline, "llm") else "error"
    except:
        llm_status = "error"
        
    # Get system metrics
    system_metrics = metrics.get_system_metrics()
    
    return {
        "status": "healthy" if db_status == "ok" and llm_status == "ok" else "degraded",
        "vector_db": db_status,
        "llm": llm_status,
        "uptime_seconds": metrics.get_uptime(),
        "total_requests": metrics.get_total_requests(),
        "system_metrics": system_metrics
    }

@router.post("/conspiracies", response_model=ConspiracyResponse, summary="Generate a conspiracy theory")
async def generate_conspiracy(
    background_tasks: BackgroundTasks,
    conspiracy_query: ConspiracyQuery,
    rag_pipeline: RAGPipeline = Depends(check_pipeline)
):
    """
    Generate a conspiracy theory based on the provided query.
    
    - **query**: User query text
    - **creativity_level**: Level of creativity/speculation (0.0-1.0)
    - **use_fact_check**: Whether to apply fact-checking
    - **filter_criteria**: Optional filters for document retrieval
    """
    request_id = f"req_{int(time.time() * 1000)}"
    start_time = time.time()
    
    # Log the request
    logger.info(f"Request {request_id}: {conspiracy_query.query[:100]}...")
    metrics.increment_request_count()
    
    try:
        # Process the query through the RAG pipeline
        result = rag_pipeline.process_query(
            query=conspiracy_query.query,
            creativity_level=conspiracy_query.creativity_level,
            filter_criteria=conspiracy_query.filter_criteria,
            use_fact_check=conspiracy_query.use_fact_check
        )
        
        # Background task to log metrics
        background_tasks.add_task(
            metrics.log_request_metrics,
            request_id=request_id,
            query_length=len(conspiracy_query.query),
            response_length=len(result["response"]),
            processing_time=time.time() - start_time,
            creativity_level=conspiracy_query.creativity_level,
            validity_score=result["validity_score"]
        )
        
        # Return the response
        return result
    
    except Exception as e:
        # Log the error
        error_msg = str(e)
        logger.error(f"Error processing request {request_id}: {error_msg}")
        metrics.increment_error_count()
        
        # Background task to log error metrics
        background_tasks.add_task(
            metrics.log_error,
            request_id=request_id,
            error=error_msg,
            query=conspiracy_query.query
        )
        
        # Return an error response
        raise HTTPException(status_code=500, detail=f"Error generating conspiracy theory: {error_msg}")

@router.get("/conspiracies/history", summary="Get recent conspiracy theory generations")
async def get_history(
    limit: int = Query(10, ge=1, le=100),
    rag_pipeline: RAGPipeline = Depends(check_pipeline)
):
    """Get a list of recent conspiracy theory generations (does not include full responses)"""
    try:
        # This would need a proper implementation to store and retrieve history
        # For now, return mock data from the metrics collector
        history = metrics.get_recent_requests(limit)
        
        if not history:
            return {"history": [], "count": 0}
            
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@router.post("/feedback", summary="Submit feedback for a generated conspiracy theory")
async def submit_feedback(
    background_tasks: BackgroundTasks,
    feedback: Dict[str, Any] = Body(...),
    rag_pipeline: RAGPipeline = Depends(check_pipeline)
):
    """
    Submit user feedback for a generated conspiracy theory.
    
    The feedback should include:
    - **request_id**: ID of the original request
    - **rating**: User rating (1-5)
    - **comments**: Optional user comments
    """
    try:
        request_id = feedback.get("request_id")
        if not request_id:
            raise HTTPException(status_code=400, detail="Missing request_id")
            
        rating = feedback.get("rating")
        if not rating or not isinstance(rating, int) or rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be an integer between 1 and 5")
            
        # Log the feedback
        logger.info(f"Received feedback for request {request_id}: rating={rating}")
        
        # Background task to store feedback
        background_tasks.add_task(
            metrics.log_feedback,
            request_id=request_id,
            rating=rating,
            comments=feedback.get("comments", "")
        )
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")