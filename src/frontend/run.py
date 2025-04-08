import os
import subprocess
import time
import sys
import signal
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/deployment.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("deployment")

# Global variables for process management
api_process = None
frontend_process = None

def start_api_server():
    """Start the FastAPI server"""
    global api_process
    
    logger.info("Starting API server...")
    cmd = ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    
    try:
        api_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"API server started with PID: {api_process.pid}")
        
        # Give the API server time to start
        time.sleep(3)
        
        # Check if process is still running
        if api_process.poll() is not None:
            stdout, stderr = api_process.communicate()
            logger.error(f"API server failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        return False

def start_streamlit():
    """Start the Streamlit frontend"""
    global frontend_process
    
    logger.info("Starting Streamlit frontend...")
    cmd = ["streamlit", "run", "src/frontend/app.py"]
    
    try:
        frontend_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Streamlit frontend started with PID: {frontend_process.pid}")
        
        # Give Streamlit time to start
        time.sleep(3)
        
        # Check if process is still running
        if frontend_process.poll() is not None:
            stdout, stderr = frontend_process.communicate()
            logger.error(f"Streamlit frontend failed to start:\nSTDOUT: {stdout}\nSTDERR: {stderr}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error starting Streamlit frontend: {str(e)}")
        return False

def shutdown_services(sig=None, frame=None):
    """Gracefully shut down all services"""
    logger.info("Shutting down services...")
    
    if frontend_process is not None:
        logger.info(f"Terminating Streamlit (PID: {frontend_process.pid})")
        try:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Streamlit didn't terminate gracefully, forcing...")
            frontend_process.kill()
        except Exception as e:
            logger.error(f"Error terminating Streamlit: {str(e)}")
    
    if api_process is not None:
        logger.info(f"Terminating API server (PID: {api_process.pid})")
        try:
            api_process.terminate()
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("API server didn't terminate gracefully, forcing...")
            api_process.kill()
        except Exception as e:
            logger.error(f"Error terminating API server: {str(e)}")
    
    logger.info("All services shut down")
    sys.exit(0)

def main():
    """Main function to start all services"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_services)
    signal.signal(signal.SIGTERM, shutdown_services)
    
    logger.info("Starting Conspiracy Theory Generator deployment...")
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Start API server
    if not start_api_server():
        logger.error("Failed to start API server, exiting.")
        shutdown_services()
        return
    
    # Start Streamlit frontend
    if not start_streamlit():
        logger.error("Failed to start Streamlit frontend, exiting.")
        shutdown_services()
        return
    
    logger.info("All services started successfully.")
    logger.info("API server running at http://localhost:8000")
    logger.info("Streamlit frontend running at http://localhost:8501")
    
    try:
        # Keep the script running until interrupted
        while True:
            # Check if processes are still running
            if api_process.poll() is not None:
                logger.error("API server has stopped unexpectedly. Shutting down...")
                shutdown_services()
                break
            
            if frontend_process.poll() is not None:
                logger.error("Streamlit frontend has stopped unexpectedly. Shutting down...")
                shutdown_services()
                break
            
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        shutdown_services()

if __name__ == "__main__":
    main()