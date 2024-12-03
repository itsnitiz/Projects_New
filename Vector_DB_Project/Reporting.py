from openai import OpenAI
from typing import List, Dict
from config import load_config

# Load configuration
config = load_config()
client = OpenAI(api_key=config["OPENAI_API_KEY"])

def pointers(query: str, initial_analysis: str, reason_counts: Dict[str, int], total_transcripts: 70) -> str:
    """
    Generate a bullet-point response for the final analysis.
    """
    system_instruction = """You are an assistant for providing analysis of call transcripts in bullets. Your task is to:
    1. Summarize the findings based on the initial analysis and the metadata summary if provided.
    2. Ensure all conclusions are backed by specific numbers and percentages.
    3. Provide bullet answer to the user's query using the quantitative data."""

    prompt = f"""The user asked: {query}

Initial Analysis: {initial_analysis}

Quantitative Data:
Total Transcripts Analyzed: {total_transcripts}
Reason Counts:
{reason_counts}

Please provide a final analysis that answers the user's query, ensuring all conclusions are backed by the quantitative data provided. Format the response in bullet points as this is intended for senior leadership of a company."""

    # Construct prompt as messages for ChatCompletion
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    # Generate response with OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    # Return the response content
    return response.choices[0].message.content.strip()

def summary(query: str, initial_analysis: str, reason_counts: Dict[str, int], total_transcripts: 70) -> str:
    """
    Generate a summary-style response for the final analysis.
    """
    system_instruction = """You are an assistant for providing analysis of call transcripts in summary. Your task is to:
    1. Summarize the findings based on the initial analysis, the count data and metadata summary provided.
    2. Ensure all conclusions are backed by specific numbers and percentages.
    3. Provide a comprehensive answer to the user's query using this quantitative data."""

    prompt = f"""The user asked: {query}

Initial Analysis: {initial_analysis}

Quantitative Data:
Total Transcripts Analyzed: {total_transcripts}
Reason Counts:
{reason_counts}

Please provide a summary of the final analysis that answers the user's query, ensuring all conclusions are backed by the quantitative data provided. Format the response in a single paragraph to provide an overview for senior leadership of a company."""

    # Construct prompt as messages for ChatCompletion
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]

    # Generate response with OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    # Return the response content
    return response.choices[0].message.content.strip()



# from openai import OpenAI
# from typing import List, Tuple, Dict
# from config import load_config

# config = load_config()
# client = OpenAI(api_key=config["OPENAI_API_KEY"])

# def final_detailed_reporting(query: str, initial_analysis: str, reason_counts: Dict[str, int], total_transcripts: int) -> str:
#     system_instruction = """You are an assistant for providing quantitative analysis of call transcripts. Your task is to:
#     1. Summarize the findings based on the initial analysis and the count data provided.
#     2. Ensure all conclusions are backed by specific numbers and percentages.
#     3. Provide a comprehensive answer to the user's query using this quantitative data."""

#     prompt = f"""The user asked: {query}

# Initial Analysis: {initial_analysis}

# Quantitative Data:
# Total Transcripts Analyzed: {total_transcripts}
# Reason Counts:
# {reason_counts}

# Please provide a final analysis that answers the user's query, ensuring all conclusions are backed by the quantitative data provided. Include percentages where relevant. The response should be in bullets as this is intended for senior leadership of a company"""

#     # Construct prompt as messages for ChatCompletion
#     messages = [
#         {"role": "system", "content": system_instruction},
#         {"role": "user", "content": prompt}
#     ]

#     # Generate response with OpenAI
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         max_tokens=500
#     )

#     # Return the response content
#     return response.choices[0].message.content.strip()