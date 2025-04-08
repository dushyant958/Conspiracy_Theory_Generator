import logging
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import openai
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/factcheck_module.log"),
        logging.StreamHandler()
    ]
)

class FactChecker:
    """
    Validates conspiracy theories against facts and provides credibility scores.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Fact Checker.
        
        Args:
            model_name: Name of the LLM model to use for fact checking
            api_key: API key for the LLM provider
            config: Additional configuration parameters
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or {}
        
        # Configuration parameters
        self.credibility_threshold = self.config.get("credibility_threshold", 0.6)
        self.check_references = self.config.get("check_references", True)
        
        # Configure OpenAI client
        openai.api_key = self.api_key
        
        logging.info(f"Fact Checker initialized with model: {self.model_name}")
        
    def check_theory(self, theory: str, context: str) -> Tuple[str, float]:
        """
        Check a conspiracy theory against facts and calculate a credibility score.
        
        Args:
            theory: The conspiracy theory text to check
            context: The retrieved context used to generate the theory
            
        Returns:
            Tuple of (annotated_theory, credibility_score)
        """
        try:
            start_time = time.time()
            
            # Extract claims from the theory
            claims = self._extract_claims(theory)
            
            # Check each claim against the context
            checked_claims = []
            claim_scores = []
            
            for claim in claims:
                score, explanation = self._verify_claim(claim, context)
                checked_claims.append({
                    "claim": claim,
                    "score": score,
                    "explanation": explanation
                })
                claim_scores.append(score)
            
            # Calculate overall credibility score
            if claim_scores:
                credibility_score = sum(claim_scores) / len(claim_scores)
            else:
                credibility_score = 0.5  # Default score if no claims extracted
            
            # Annotate the theory with fact-check information
            annotated_theory = self._annotate_theory(theory, checked_claims, credibility_score)
            
            elapsed_time = time.time() - start_time
            logging.info(f"Fact-checked theory with score {credibility_score:.2f} in {elapsed_time:.2f}s")
            
            return annotated_theory, credibility_score
            
        except Exception as e:
            logging.error(f"Error in fact checking: {str(e)}")
            return theory, 0.5  # Return original theory with neutral score on error
    
    def _extract_claims(self, theory: str) -> List[str]:
        """
        Extract factual claims from the conspiracy theory.
        
        Args:
            theory: Conspiracy theory text
            
        Returns:
            List of extracted claims
        """
        try:
            # Use LLM to extract claims
            prompt = f"""
Please extract the main factual claims from the following conspiracy theory.
Focus only on statements presented as facts, not opinions or speculations.
For each claim, provide it as a separate line starting with "CLAIM: ".

Theory:
{theory}

Extract the factual claims:
"""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts factual claims from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse claims from the response
            claims = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith("CLAIM:"):
                    claim = line[6:].strip()
                    if claim:
                        claims.append(claim)
            
            logging.info(f"Extracted {len(claims)} claims from theory")
            return claims
            
        except Exception as e:
            logging.error(f"Error extracting claims: {str(e)}")
            # Fallback to simple sentence-based claim extraction
            sentences = re.split(r'(?<=[.!?])\s+', theory)
            # Filter out short sentences and sentences that are questions
            claims = [s for s in sentences if len(s) > 20 and not s.endswith('?')]
            return claims[:5]  # Limit to 5 claims
    
    def _verify_claim(self, claim: str, context: str) -> Tuple[float, str]:
        """
        Verify a claim against the retrieved context and calculate a credibility score.
        
        Args:
            claim: The claim to verify
            context: The retrieved context
            
        Returns:
            Tuple of (credibility_score, explanation)
        """
        try:
            prompt = f"""
Please fact-check the following claim against the provided context information.
Provide a credibility score between 0.0 (completely false) and 1.0 (completely true),
and a brief explanation of your reasoning.

Claim:
{claim}

Context Information:
{context}

Your response should be in this format:
SCORE: [number between 0.0 and 1.0]
EXPLANATION: [your explanation]
"""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant that evaluates claims for accuracy."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse score and explanation
            score_match = re.search(r'SCORE:\s*([\d.]+)', text)
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=$|\nSCORE:)', text, re.DOTALL)
            
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5  # Default neutral score
            
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            else:
                explanation = "No explanation provided."
            
            return score, explanation
            
        except Exception as e:
            logging.error(f"Error verifying claim: {str(e)}")
            return 0.5, "Error verifying claim."
    
    def _annotate_theory(self, theory: str, checked_claims: List[Dict[str, Any]], credibility_score: float) -> str:
        """
        Annotate the conspiracy theory with fact-check information.
        
        Args:
            theory: Original conspiracy theory text
            checked_claims: List of checked claims with scores and explanations
            credibility_score: Overall credibility score
            
        Returns:
            Annotated theory with fact-check information
        """
        try:
            # Use LLM to create an annotated version
            claims_info = "\n".join([
                f"Claim: {c['claim']}\nCredibility: {c['score']:.2f}\nExplanation: {c['explanation']}"
                for c in checked_claims
            ])
            
            prompt = f"""
Please create an annotated version of the following conspiracy theory.
Add fact-check information as footnotes or inline comments, and add a "FACT CHECK SUMMARY"
section at the end with an overall credibility assessment.

Original theory:
{theory}

Fact-check information for claims:
{claims_info}

Overall credibility score: {credibility_score:.2f}

Create an annotated version with fact-check information:
"""
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that annotates texts with fact-check information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            annotated_theory = response.choices[0].message.content.strip()
            
            # Ensure it has the FACT CHECK SUMMARY section
            if "FACT CHECK SUMMARY" not in annotated_theory:
                summary = self._generate_fact_check_summary(checked_claims, credibility_score)
                annotated_theory += f"\n\n{summary}"
            
            return annotated_theory
            
        except Exception as e:
            logging.error(f"Error annotating theory: {str(e)}")
            
            # Fallback to simple annotation
            summary = self._generate_fact_check_summary(checked_claims, credibility_score)
            return f"{theory}\n\n{summary}"
    
    def _generate_fact_check_summary(self, checked_claims: List[Dict[str, Any]], credibility_score: float) -> str:
        """
        Generate a fact-check summary.
        
        Args:
            checked_claims: List of checked claims with scores and explanations
            credibility_score: Overall credibility score
            
        Returns:
            Fact-check summary text
        """
        # Determine credibility level
        if credibility_score >= 0.8:
            level = "Highly credible"
        elif credibility_score >= 0.6:
            level = "Moderately credible"
        elif credibility_score >= 0.4:
            level = "Questionable credibility"
        elif credibility_score >= 0.2:
            level = "Low credibility"
        else:
            level = "Very low credibility"
        
        # Create summary text
        summary_lines = ["FACT CHECK SUMMARY"]
        summary_lines.append(f"Overall credibility: {credibility_score:.2f} - {level}")
        
        if checked_claims:
            summary_lines.append("\nKey claims assessment:")
            for claim in checked_claims:
                summary_lines.append(f"- {claim['claim'][:100]}... [{claim['score']:.2f}]")
        
        summary_lines.append("\nREMINDER: This is a fictional conspiracy theory for entertainment purposes only.")
        
        return "\n".join(summary_lines)

    def check_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the credibility of the sources used.
        
        Args:
            sources: List of source information
            
        Returns:
            Dictionary with source credibility assessments
        """
        if not self.check_references:
            return {"source_check": "disabled"}
            
        try:
            source_assessments = []
            
            for source in sources:
                assessment = {
                    "source": source.get("source", "Unknown"),
                    "title": source.get("title", "Unknown"),
                    "credibility": 0.5  # Default score
                }
                
                # Basic source type credibility scoring
                source_name = source.get("source", "").lower()
                
                # Very simple rule-based scoring - replace with a proper system
                if any(term in source_name for term in [
                    "university", "edu", "research", "science", "journal", "academic"
                ]):
                    assessment["credibility"] = 0.9
                    assessment["type"] = "Academic"
                elif any(term in source_name for term in [
                    "gov", "government", "official", "agency"
                ]):
                    assessment["credibility"] = 0.85
                    assessment["type"] = "Government"
                elif any(term in source_name for term in [
                    "news", "times", "post", "bbc", "cnn", "nyt", "reuters"
                ]):
                    assessment["credibility"] = 0.7
                    assessment["type"] = "Mainstream Media"
                elif any(term in source_name for term in [
                    "blog", "magazine", "opinion"
                ]):
                    assessment["credibility"] = 0.5
                    assessment["type"] = "Blog/Magazine"
                elif any(term in source_name for term in [
                    "social", "twitter", "facebook", "reddit", "forum"
                ]):
                    assessment["credibility"] = 0.3
                    assessment["type"] = "Social Media"
                else:
                    assessment["credibility"] = 0.4
                    assessment["type"] = "Unknown"
                
                source_assessments.append(assessment)
            
            # Overall assessment
            if source_assessments:
                avg_credibility = sum(s["credibility"] for s in source_assessments) / len(source_assessments)
            else:
                avg_credibility = 0.0
                
            return {
                "sources": source_assessments,
                "average_credibility": avg_credibility,
                "assessment": self._get_credibility_label(avg_credibility)
            }
            
        except Exception as e:
            logging.error(f"Error checking sources: {str(e)}")
            return {"source_check": "error", "message": str(e)}
    
    def _get_credibility_label(self, score: float) -> str:
        """Get a text label for a credibility score"""
        if score >= 0.8:
            return "Highly Credible Sources"
        elif score >= 0.6:
            return "Credible Sources"
        elif score >= 0.4:
            return "Mixed Credibility Sources"
        elif score >= 0.2:
            return "Low Credibility Sources"
        else:
            return "Very Low Credibility Sources"