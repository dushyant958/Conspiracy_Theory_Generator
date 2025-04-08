import os
import argparse
import json
from pipeline import create_pipeline_from_config, RAGPipeline

def main():
    """Example usage of the RAG Pipeline"""
    parser = argparse.ArgumentParser(description="Conspiracy Theory Generator")
    parser.add_argument("--query", type=str, required=True,
                      help="User query for conspiracy theory generation")
    parser.add_argument("--config", type=str, default="config/rag_config.json",
                      help="Path to RAG configuration file")
    parser.add_argument("--creativity", type=float, default=0.7,
                      help="Creativity level (0.0-1.0)")
    parser.add_argument("--fact-check", action="store_true",
                      help="Enable fact-checking")
    parser.add_argument("--output", type=str,
                      help="Save result to output file")
    
    args = parser.parse_args()
    
    # Create pipeline from config
    pipeline = create_pipeline_from_config(args.config)
    
    # Process the query
    result = pipeline.process_query(
        query=args.query,
        creativity_level=args.creativity,
        use_fact_check=args.fact_check
    )
    
    # Print the result
    print("\n" + "="*50)
    print("CONSPIRACY THEORY GENERATOR")
    print("="*50)
    print(f"Query: {args.query}")
    print(f"Creativity Level: {args.creativity}")
    print("-"*50)
    print("\nResponse:")
    print(result["response"])
    print("-"*50)
    
    if args.fact_check:
        print(f"\nValidity Score: {result['validity_score']:.2f}")
    
    print("\nTop Sources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source['title']} ({source['source']}) - Relevance: {source['relevance']:.2f}")
    
    print(f"\nProcessing Time: {result['processing_time_seconds']:.2f} seconds")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to {args.output}")

if __name__ == "__main__":
    main()