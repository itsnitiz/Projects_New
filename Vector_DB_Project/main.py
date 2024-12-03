import asyncio
from Query_Analysis import analyze_query
from Filtering import retrieve_serial_numbers, search_serial_numbers
from Retrieve import counter_documents
from Decider import module_chooser
from Analysis import general_analysis,detailed_analysis
from Reporting import pointers,summary
from config import load_config
from langchain.vectorstores import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from typing import List
import warnings

warnings.filterwarnings("ignore")

async def main():
    # Load configuration and initialize vector stores
    config = load_config()
    embedding_model = OpenAIEmbeddings(openai_api_key=config['OPENAI_API_KEY'], model="text-embedding-ada-002")
    
    vectorstore_metadata = PGVector(
        embedding_function=embedding_model,
        connection_string=config['PGVECTOR_CONNECTION_STRING'],
        collection_name="call_embeddings"
    )

    vectorstore_embeddings = PGVector(
        embedding_function=embedding_model,
        connection_string=config['PGVECTOR_CONNECTION_STRING'],
        collection_name="call_embeddings_detailed"
    )

    # Analyze the question and determine sub-queries and routing
    question = "No of calls where call duration is more than 600 seconds"
    sub_queries = analyze_query(question)
    print(sub_queries)
    
    module = module_chooser(sub_queries)
    print(module)
    final_response_format = module_chooser([question])
    print(final_response_format)
    # Initialize the dictionaries for each function type
    filtering_function = {}
    analysis_function = {}
    reporting_function = {}
    
    reporting_function_final = {}
    # Iterate over the items in the original module_data
    for query, functions in final_response_format.items():
        reporting_key = functions['reporting_function']
        
        # Update reporting_function dictionary
        if reporting_key not in reporting_function_final:
            reporting_function_final[reporting_key] = []
        reporting_function_final[reporting_key].append(query)
    
    # Iterate over the items in the original module_data
    for query, functions in module.items():
        filtering_key = functions['filtering_function']
        analysis_key = functions['analysis_function']
        reporting_key = functions['reporting_function']
        
        # Update filtering_function dictionary
        if filtering_key not in filtering_function:
            filtering_function[filtering_key] = []
        filtering_function[filtering_key].append(query)
    
        # Update analysis_function dictionary
        if analysis_key not in analysis_function:
            analysis_function[analysis_key] = []
        analysis_function[analysis_key].append(query)
        
        # Update reporting_function dictionary
        if reporting_key not in reporting_function:
            reporting_function[reporting_key] = []
        reporting_function[reporting_key].append(query)
    
    print(filtering_function)
    print(analysis_function)
    print(reporting_function)
    
    shortlisted_id_metadata = []
    metadata_summary = {}
    relevant_transcripts = {}
    analysis_collection = ""
    counts = {}
    counts_detailed ={}
    if 'metadata_filtering' in filtering_function and filtering_function['metadata_filtering']:
        shortlisted_id_metadata,metadata_summary = retrieve_serial_numbers(filtering_function['metadata_filtering'], vectorstore_metadata)
    if 'transcript_filtering' in filtering_function and filtering_function['transcript_filtering']:
        relevant_transcripts = await search_serial_numbers(vectorstore_embeddings,vectorstore_metadata,filtering_function['transcript_filtering'],shortlisted_id_metadata)
    if 'general_analysis' in analysis_function and analysis_function['general_analysis']: 
        for query in analysis_function['general_analysis']:
            if query in metadata_summary:
                metadata_summary.pop(query)
            relevant_docs = relevant_transcripts[query]
            analysis = general_analysis(query,relevant_docs)
            counts = counter_documents(analysis, vectorstore_metadata)
            analysis_collection = analysis_collection + analysis
    if 'detailed_analysis' in analysis_function and analysis_function['detailed_analysis']: 
        for query in analysis_function['detailed_analysis']:
            if query in metadata_summary:
                metadata_summary.pop(query)
            relevant_docs = relevant_transcripts[query]
            analysis = detailed_analysis(query,relevant_docs)
            counts_detailed = counter_documents(analysis, vectorstore_metadata)
            analysis_collection = analysis_collection + analysis
    
    print('length of shortlisted')
    print(shortlisted_id_metadata)
    counts.update(counts_detailed)
    metadata_analysis = ''.join(metadata_summary.values())
    analysis_collection = analysis_collection + metadata_analysis
    print('Analysis Done')
    print('Reporting process starts')
    if 'Summary' in reporting_function_final and reporting_function_final['Summary']:
        final_output = summary(question,analysis_collection,counts,70)
    if 'Pointers' in reporting_function_final and reporting_function_final['Pointers']:
        final_output = pointers(question,analysis_collection,counts,70)
    
    print(final_output)

# Run the main function in an asynchronous environment
if __name__ == "__main__":
    asyncio.run(main())
