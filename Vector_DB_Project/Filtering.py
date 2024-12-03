from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain.schema import Document
from config import load_config
from typing import List
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from Analysis import execute_query_on_metadata

config = load_config()

def retrieve_serial_numbers(queries: List[str], vectorstore_metadata) -> List[str]:
    """
    Retrieve unique serial numbers from documents matching the given queries.

    :param queries: List of queries to retrieve documents for.
    :param openai_api_key: API key for OpenAI.
    :param vectorstore_metadata: Metadata vector store used in SelfQueryRetriever.
    :return: List of unique serial numbers.
    """
    # Define metadata fields
    metadata_field_info = [
    AttributeInfo(
        name="Rolewise Transcript",
        description="The transcript of the call including the roles of the speakers.",
        type="string",
    ),
    AttributeInfo(
        name="Call Length",
        description="The duration of the call in seconds.",
        type="float",
    ),
    AttributeInfo(
        name="Lead Id",
        description="Unique identifier for the lead.",
        type="string",
    ),
    AttributeInfo(
        name="Call DateTime",
        description="The date and time when the call took place.",
        type="datetime",
    ),
    AttributeInfo(
        name="Language of the call",
        description="The language in which the call was conducted. Valid Values are ['Hindi', 'Marathi']",
        type="string",
    ),
    AttributeInfo(
        name="Purpose",
        description="The purpose of the call. Valid Values are ['Construction', 'Other Loans - Home Equity', 'Other Loans - MSME', 'Purchase', 'Purchase and Construction', 'Repair and Renovation Loan', 'Resale Property Purchase']",
        type="string",
    ),
    AttributeInfo(
        name="Product offered",
        description="The product that was offered during the call. Valid Values are ['Construction', 'Other Loans - Home Equity', 'Other Loans - MSME', 'Purchase', 'Purchase and Construction', 'Repair and Renovation Loan', 'Resale Property Purchase']",
        type="string",
    ),
    AttributeInfo(
        name="Lead Source",
        description="The source from which the lead/call was generated. Valid Values are ['Aavas Plus', 'BTLCanopy', 'BTLConstruction Visit', 'BTLMissed call', 'Chatbot', 'CustAppCrif', 'CustAppNew Lead', 'CustAppTop Up', 'Google Ads', 'MCVAN Activity', 'PhonePe', 'Reference', 'Self Sourced', 'Toll Free', 'Web_CRIF', 'Web_Sampark', 'Website', 'Whatsapp']",
        type="string",
    ),
    AttributeInfo(
        name="Location",
        description="The location of the customer. Valid Values are listed locations.['Gurgaon']",
        type="string",
    ),
    AttributeInfo(
        name="Branch",
        description="The branch associated with the lead. Valid Values are listed branches.",
        type="string",
    ),
    AttributeInfo(
        name="Opportunity Created",
        description="Indicates whether an opportunity was created from the call. Valid Values are [True]",
        type="boolean",
    ),
    AttributeInfo(
        name="Business Created",
        description="Indicates whether a business was created from the call. Valid Values are [True]",
        type="boolean",
    ),
    AttributeInfo(
        name="Agent Name",
        description="The name of the agent who handled the call. Valid Values are listed agent names.",
        type="string",
    ),
    AttributeInfo(
        name="Agent ID",
        description="Unique identifier for the agent.",
        type="integer",
    ),
    AttributeInfo(
        name="id",
        description="Unique identifier for the record.",
        type="integer",
    ),
]

    # Initialize language model
    llm = OpenAI(openai_api_key=config["OPENAI_API_KEY"], temperature=0)

    # Set up document content description for SelfQueryRetriever
    document_content_description = "Call Transcript of a telesales call."

    # Initialize SelfQueryRetriever
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore_metadata,
        document_contents=None,
        metadata_field_info=metadata_field_info,
        enable_limit=True,
        verbose=True,
        search_kwargs={"k": 1000}
    )

    # Set to collect unique documents
    all_retrieved_documents = set()
    summary_overall={}

    # Retrieve and combine unique documents for each query
    for query in queries:
        documents = retriever.get_relevant_documents(query)
        for doc in documents:
            all_retrieved_documents.add((doc.page_content, tuple(doc.metadata.items())))
        summary = execute_query_on_metadata(query, documents)
        summary_overall[query] = summary

    # Convert to list of Document objects
    unique_documents = [Document(page_content=content, metadata=dict(metadata)) for content, metadata in all_retrieved_documents]

    # Extract and return 'Serial Number' values from metadata
    serial_numbers = [doc.metadata.get("Serial Number") for doc in unique_documents if "Serial Number" in doc.metadata]
    return serial_numbers, summary_overall

from typing import List
from langchain_community.vectorstores.pgvector import PGVector

from typing import List, Dict

async def search_serial_numbers(
    vectorstore: PGVector,
    vectorstore2: PGVector,
    queries: List[str],
    serial_numbers: List[str]
) -> Dict[str, str]:
    """
    Perform an asynchronous maximal marginal relevance (MMR) search on each query,
    filtering results by serial numbers and returning unique serial numbers and
    transcripts for each query.

    :param vectorstore: The PGVector vector store instance.
    :param vectorstore2: The second PGVector vector store instance for content retrieval.
    :param queries: List of queries for the MMR search.
    :param serial_numbers: List of serial numbers to filter results.
    :return: Dictionary of query strings as keys and concatenated page content as values.
    """
    unique_serial_numbers = set()  # Set to collect unique serial numbers
    transcripts = {}  # Dictionary to store concatenated page content for each query

    # Iterate over each query in the list
    for query in queries:
        # Define search arguments, including MMR strategy and optional filtering by serial numbers
        search_kwargs = {
            "search_type": "mmr",
            "lambda_mult": 0.25,  # Adjust diversity if needed
        }

        # Apply filter by serial numbers if provided
        if serial_numbers:
            search_kwargs["filter"] = {"Serial Number": {"$in": serial_numbers}}

        # Perform MMR search for the current query on vectorstore
        results = await vectorstore.asearch(query, **search_kwargs)

        # Collect unique serial numbers from metadata
        for result in results:
            serial_number = result.metadata.get("Serial Number")
            if serial_number:
                unique_serial_numbers.add(serial_number)

        # Perform a second search on vectorstore2 based on unique serial numbers collected
        filter_kwargs = {
            "search_type": "mmr",
            "filter": {"Serial Number": {"$in": list(unique_serial_numbers)}},
            "k": 70  # Retrieve up to 70 results
        }
        vectorstore2_results = await vectorstore2.asearch(query, **filter_kwargs)

        # Concatenate page content for each unique serial number under the current query
        page_content = " ".join([res.page_content for res in vectorstore2_results])
        transcripts[query] = page_content

    return transcripts

