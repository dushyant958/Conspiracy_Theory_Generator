import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

# API Configuration
API_URL = "http://localhost:8000/api/v1"  # Change this to your API URL when deployed

# Page configuration
st.set_page_config(
    page_title="Conspiracy Theory Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .validity-high {
        color: green;
        font-weight: bold;
    }
    .validity-medium {
        color: orange;
        font-weight: bold;
    }
    .validity-low {
        color: red;
        font-weight: bold;
    }
    .sidebar-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### Controls")
    
    # Creativity level slider
    creativity = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values produce more creative but potentially less grounded theories"
    )
    
    # Fact checking toggle
    fact_check = st.checkbox("Enable Fact Checking", value=True, 
                            help="Validates the generated theory against known facts")
    
    # Advanced options expander
    with st.expander("Advanced Options"):
        # Example filter criteria - customize based on your RAG pipeline
        filter_topics = st.multiselect(
            "Filter by Topics",
            options=["Government", "Technology", "Health", "Space", "History", "Military"],
            default=[]
        )
        
        filter_date_range = st.date_input(
            "Documents Date Range",
            value=(datetime(2000, 1, 1), datetime.now()),
            help="Filter documents by publication date"
        )
        
        filter_min_reliability = st.slider(
            "Minimum Source Reliability",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum reliability score for source documents"
        )
    
    # System status indicator
    st.markdown("### System Status")
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            status = health_data.get("status", "unknown")
            
            if status == "healthy":
                st.success("System: Online")
            elif status == "degraded":
                st.warning("System: Degraded")
            else:
                st.error("System: Issues Detected")
                
            st.text(f"Total Requests: {health_data.get('total_requests', 'N/A')}")
            st.text(f"Vector DB: {health_data.get('vector_db', 'N/A')}")
            st.text(f"LLM: {health_data.get('llm', 'N/A')}")
        else:
            st.error("System: Offline")
    except:
        st.error("Cannot connect to API")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Conspiracy Theory Generator</h1>', unsafe_allow_html=True)
st.markdown("Explore alternative explanations and conspiracy theories based on your queries.")

# Query input
query = st.text_area("Enter your query:", height=100, 
                    placeholder="Example: What really happened at Area 51?")

# Create filter criteria based on advanced options
filter_criteria = {}
if filter_topics:
    filter_criteria["topics"] = filter_topics
if filter_date_range and len(filter_date_range) == 2:
    filter_criteria["date_range"] = {
        "start": filter_date_range[0].isoformat(),
        "end": filter_date_range[1].isoformat()
    }
if filter_min_reliability > 0.0:
    filter_criteria["min_reliability"] = filter_min_reliability

# Submit button
if st.button("Generate Conspiracy Theory"):
    if len(query) < 5:
        st.error("Query must be at least 5 characters long")
    else:
        with st.spinner("Generating conspiracy theory..."):
            try:
                # Prepare request payload
                payload = {
                    "query": query,
                    "creativity_level": creativity,
                    "use_fact_check": fact_check,
                    "filter_criteria": filter_criteria if filter_criteria else None
                }
                
                # Make API request
                response = requests.post(
                    f"{API_URL}/conspiracies",
                    json=payload
                )
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display response time
                    st.text(f"Processing Time: {result['processing_time_seconds']:.2f} seconds")
                    
                    # Display validity score with color coding
                    validity = result.get('validity_score', 0)
                    validity_class = "validity-high" if validity >= 0.7 else \
                                    "validity-medium" if validity >= 0.4 else \
                                    "validity-low"
                    
                    st.markdown(f"<h3>Validity Score: <span class='{validity_class}'>{validity:.2f}</span></h3>", 
                                unsafe_allow_html=True)
                    
                    # Display the generated theory
                    st.markdown('<h2 class="sub-header">Generated Theory</h2>', unsafe_allow_html=True)
                    st.markdown(result['response'])
                    
                    # Display sources
                    st.markdown('<h2 class="sub-header">Sources</h2>', unsafe_allow_html=True)
                    for i, source in enumerate(result.get('sources', [])):
                        with st.expander(f"Source {i+1}: {source.get('title', 'Unknown')}"):
                            st.markdown(f"**Relevance Score:** {source.get('relevance', 'N/A')}")
                            st.markdown(f"**Source Type:** {source.get('type', 'Unknown')}")
                            st.markdown(f"**Publication Date:** {source.get('date', 'Unknown')}")
                            st.markdown("**Excerpt:**")
                            st.markdown(f"<div class='source-box'>{source.get('excerpt', 'No excerpt available')}</div>", 
                                        unsafe_allow_html=True)
                    
                    # Feedback section
                    st.markdown('<h2 class="sub-header">Feedback</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        feedback_text = st.text_area("Comments (optional):", key="feedback_text")
                    
                    with col2:
                        rating = st.select_slider(
                            "Rate this theory:",
                            options=[1, 2, 3, 4, 5],
                            value=3
                        )
                    
                    if st.button("Submit Feedback"):
                        # Prepare feedback payload
                        feedback_payload = {
                            "request_id": result.get("request_id", f"req_{int(time.time() * 1000)}"),
                            "rating": rating,
                            "comments": feedback_text
                        }
                        
                        # Send feedback
                        feedback_response = requests.post(
                            f"{API_URL}/feedback",
                            json=feedback_payload
                        )
                        
                        if feedback_response.status_code == 200:
                            st.success("Thank you for your feedback!")
                        else:
                            st.error("Failed to submit feedback. Please try again.")
                
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add Recent History tab
st.markdown('<h2 class="sub-header">Recent Queries</h2>', unsafe_allow_html=True)

try:
    history_response = requests.get(f"{API_URL}/conspiracies/history?limit=5")
    if history_response.status_code == 200:
        history_data = history_response.json()
        history_items = history_data.get("history", [])
        
        if history_items:
            # Convert to DataFrame for better display
            df = pd.DataFrame(history_items)
            
            # Display recent queries table
            st.dataframe(
                df[["request_id", "query", "validity_score", "processing_time", "timestamp"]],
                hide_index=True,
                column_config={
                    "request_id": "Request ID",
                    "query": "Query",
                    "validity_score": st.column_config.NumberColumn("Validity Score", format="%.2f"),
                    "processing_time": st.column_config.NumberColumn("Processing Time (s)", format="%.2f"),
                    "timestamp": "Time"
                }
            )
            
            # Optional: Add a simple visualization of validity scores
            if len(df) > 1:
                st.subheader("Validity Score Distribution")
                fig = px.histogram(df, x="validity_score", nbins=10,
                                 labels={"validity_score": "Validity Score"},
                                 title="Distribution of Recent Theory Validity Scores")
                st.plotly_chart(fig)
        else:
            st.info("No recent queries found.")
    else:
        st.warning("Could not load recent history.")
except:
    st.warning("Could not connect to history endpoint.")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: All theories generated are speculative and should be treated as fiction.*")