from langchain_postgres.vectorstores import PGVector
from config import load_config
from typing import List, Tuple, Any, Dict


def retrieve_documents_by_serial_numbers(vectorstore, serial_numbers,query):
    """
    Retrieve documents from the vector store based on a list of serial numbers.

    Parameters:
    - vectorstore: The initialized PGVector store instance.
    - serial_numbers: List of serial numbers to filter by.

    Returns:
    - List of retrieved documents based on the specified serial numbers.
    """
    # Initialize the retriever with metadata filtering based on serial numbers
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {
                "Serial Number": {"$in": serial_numbers}
            }
        }
    )

    # Invoke the retriever without a query, as we're only filtering by metadata
    results = retriever.get_relevant_documents(query)  # Using an empty query

    return results

def counter_documents(llm_output: str, vector_store: Any, score_threshold: float = 0.8) -> Dict[str, int]:
    """
    Analyzes LLM output, extracts reasons, and searches documents in a vector store for each reason.
    
    Parameters:
    - llm_output (str): The output text from the LLM containing numbered reasons.
    - vector_store (Any): The vector store to perform the search.
    - score_threshold (float): Initial relevance threshold for similarity scores. Defaults to 0.8.
    
    Returns:
    - Dict[str, Union[int, str]]: A dictionary with each reason as a key and the count of relevant documents as the value.
      If no relevant documents are found down to a threshold of 0.4, "less evidence" is assigned.
    """
    # Extract reasons from the LLM output (supports any number of reasons)
    reasons = [line.split('. ', 1)[1] for line in llm_output.split('\n') if line.strip() and line.split()[0].rstrip('.').isdigit()]
    
    # Initialize dictionary to store counts for each reason
    reason_counts = {}

    for reason in reasons:
        # Start with the initial threshold
        current_threshold = score_threshold
        count = 0
        
        # Attempt to retrieve documents with progressively lower thresholds
        while current_threshold >= 0.4:
            # Perform similarity search for each reason with the specified threshold
            results = vector_store.similarity_search_with_relevance_scores(
                query=reason,
                k=300,
                score_threshold=current_threshold
            )
            
            # Count the number of documents with a score above the threshold
            count = sum(1 for _, score in results if score >= current_threshold)
            
            # If we find any documents, break out of the loop
            if count > 0:
                break
            
            # Decrease the threshold by 0.1 for the next iteration
            current_threshold -= 0.1

        # If no documents are found after lowering threshold to 0.4, set as "less evidence"
        reason_counts[reason] = count if count > 0 else "less evidence"

    return reason_counts




