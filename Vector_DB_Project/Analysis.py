from openai import OpenAI
from typing import List, Tuple, Dict
from config import load_config

config = load_config()
client = OpenAI(api_key=config["OPENAI_API_KEY"])

def general_analysis(query: str, relevant_transcripts: str) -> str:
    """
    Analyzes call transcripts and provides an summary based answer based on the query.

    :param query: The user's query.
    :param relevant_transcripts: A single string containing concatenated transcripts.
    :return: A string with the analysis and summary based on the question asked 
    """
    # Define system instructions
    system_instruction = """You are an assistant for analyzing call transcripts of Aavas. Calls from Aavas are telesales calls are usually done to sell their financial products like loans, etc. Aavas deals with financial products.
    Your task is to:
    1. Identify the intent related to the user's query. The intent should relate to whether a financial product is getting sold.
    2. Provide a brief answer in a summary."""

    # Construct prompt
    prompt = f"""The user asked: {query}

Analyze these transcripts and provide a brief summary answer:

{relevant_transcripts}

"""

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

def detailed_analysis(query: str, relevant_transcripts: str) -> str:
    """
    Analyzes call transcripts and provides an detailed analysis based on the query.

    :param query: The user's query.
    :param relevant_transcripts: A single string containing concatenated transcripts.
    :return: A string with the initial analysis and identified reasons or key points.
    """
    # Define system instructions
    system_instruction = """You are an assistant for analyzing call transcripts of Aavas. Calls from Aavas are telesales calls are usually done to sell their financial products like loans, etc. Aavas deals with financial products.
    Your task is to:
    1. Identify potential reasons or key points related to the user's query. The intent should relate to whether a financial product is getting sold.
    2. Provide a brief detailed analysis.
    3. List the identified reasons or key points in a numbered format for easy parsing."""

    # Construct prompt
    prompt = f"""The user asked: {query}

Analyze these transcripts and provide an initial response, listing potential reasons or key points:

{relevant_transcripts}

Format your response as:
Initial Analysis: [Your brief analysis here]

Reasons/Key Points:
1. [First reason/point]
2. [Second reason/point]
...
"""

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

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain.agents.agent_types import AgentType

def execute_query_on_metadata(query: str, documents: list) -> str:
    """
    Convert metadata JSON to a Pandas DataFrame and execute a query on it using LangChain's Pandas agent.

    :param query: The question to ask about the data.
    :param metadata_json: The metadata in JSON format retrieved from the database.
    :return: The result of the query as a string.
    """
    
    # Extract metadata from each document and organize it into a list of dictionaries
    metadata_list = [doc.metadata for doc in documents if hasattr(doc, "metadata")]
    
    # Convert the list of metadata dictionaries to a DataFrame
    df = pd.DataFrame(metadata_list)
    # Generate the statistical summary of the DataFrame
    summary_df = df.describe(include='all')
    
    # Convert the summary to a string format
    summary_str = summary_df.to_string()
    final_response = f"Metadata Statistical Summary :\n{summary_str}\n\n for Metadata Query:\n{query}"
    return final_response

# def execute_query_on_metadata(query: str, documents: list) -> str:
#     """
#     Convert metadata JSON to a Pandas DataFrame and execute a query on it using LangChain's Pandas agent.

#     :param query: The question to ask about the data.
#     :param metadata_json: The metadata in JSON format retrieved from the database.
#     :return: The result of the query as a string.
#     """
    
#     # Extract metadata from each document and organize it into a list of dictionaries
#     metadata_list = [doc.metadata for doc in documents if hasattr(doc, "metadata")]
    
#     # Convert the list of metadata dictionaries to a DataFrame
#     df = pd.DataFrame(metadata_list)
#     print(df)
    
#     # Initialize the language model and agent
#     llm = ChatOpenAI(api_key=config["OPENAI_API_KEY"],model="gpt-3.5-turbo", temperature=0)
#     agent = create_pandas_dataframe_agent(
#         llm,
#         df,
#         agent_type="tool-calling",
#         verbose=True,
#         allow_dangerous_code=True
#     )
    
#     # Invoke the query and return the response
#     response = agent.invoke(query)
#     return response

