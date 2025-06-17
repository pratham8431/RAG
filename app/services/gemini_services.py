import os
import json
import logging
from typing import Dict, Any

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the model globally to avoid reinitializing on every call
model = genai.GenerativeModel("gemini-1.5-pro")

async def generate_influencer_list(search_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate influencer list based on search parameters using Gemini
    """
    try:
        # System instruction
        system_instruction = """✅ System Prompt: Influencer Discovery AI
You are an expert influencer discovery assistant trained to match brand campaign needs with suitable influencers.
Based on the given structured query, generate a structured response that includes 15 influencers in a clean JSON format.

Expected format:
{
  "influencers": [
    {
      "name": "string",
      "platform": "Instagram | YouTube | etc.",
      "category": "Fashion | Fitness | etc.",
      "followers": "numeric",
      "engagement_rate": "percentage",
      "location": "string",
      "profile_link": "URL"
    },
    ...
  ]
}
"""

        # Compose the prompt
        prompt = f"""{system_instruction}

Search Parameters:
{json.dumps(search_params, indent=2)}
"""

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2
            }
        )


        # Parse response
        try:
            influencer_data = json.loads(response.text)
            influencers = influencer_data.get("influencers", [])
            if not isinstance(influencers, list):
                raise ValueError("Invalid influencer data format")

            return {
                "success": True,
                "influencers": influencers,
                "count": len(influencers),
                "logs": [{"message": "Influencer list generated successfully", "type": "success"}]
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            logger.error(f"Raw response: {response.text}")
            return {
                "success": False,
                "message": "Failed to parse influencer data",
                "error": str(e),
                "raw_response": response.text
            }

    except Exception as e:
        logger.error(f"Error in Gemini service: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": "Failed to generate influencer list",
            "error": str(e)
        }
