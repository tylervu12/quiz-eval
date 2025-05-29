import os
import dotenv
from pinecone import Pinecone
from openai import OpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def main():
    """
    Main function to ingest documents into Pinecone.
    """
    print("Starting data ingestion process...")

    # Step 1: Directory Setup (fitness_docs)
    docs_folder = "fitness_docs" # Ensure this is 'data-ingestion/fitness_docs' if running from root, or adjust path
    # Correcting path to be relative to the script's location or an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_folder_path = os.path.join(script_dir, docs_folder)
    
    if not os.path.exists(docs_folder_path):
        # Fallback for common execution from project root where 'fitness_docs' might be directly there or inside 'data-ingestion'
        project_root_docs_path = "fitness_docs"
        data_ingestion_root_docs_path = os.path.join("data-ingestion", "fitness_docs")

        if os.path.exists(project_root_docs_path):
            docs_folder_path = project_root_docs_path
        elif os.path.exists(data_ingestion_root_docs_path):
            docs_folder_path = data_ingestion_root_docs_path
        else:
            print(f"Error: The folder '{docs_folder}' was not found in expected locations: {os.path.join(script_dir, docs_folder)}, {project_root_docs_path}, or {data_ingestion_root_docs_path}. Please create it and add your .txt files.")
            return
    print(f"Using document folder: {docs_folder_path}")

    # Step 2: Document Loading & Metadata Assignment
    documents = []
    for filename in os.listdir(docs_folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            metadata_source = filename[:-4]  # Remove .txt extension
            documents.append({"content": content, "source": metadata_source})
            print(f"Loaded document: {filename} with source: {metadata_source}")

    if not documents:
        print(f"No .txt files found in '{docs_folder_path}'. Exiting.")
        return

    # Step 3: Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks_with_metadata = []
    for doc_idx, doc in enumerate(documents):
        doc_chunks = text_splitter.split_text(doc["content"])
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunks_with_metadata.append({
                "id": f"{doc['source']}-{chunk_idx}", # Unique ID for each chunk
                "text": chunk_text,
                "source": doc['source']
            })
        print(f"Chunked document: {doc['source']} into {len(doc_chunks)} chunks.")

    if not chunks_with_metadata:
        print("No chunks were created. Exiting.")
        return

    # Determine Embedding Dimension before index creation
    print("Determining embedding dimension...")
    try:
        sample_text_for_embedding = chunks_with_metadata[0]["text"]
        response = openai_client.embeddings.create(
            input=sample_text_for_embedding,
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding_dimension = len(response.data[0].embedding)
        print(f"Determined embedding dimension: {embedding_dimension}")
    except Exception as e:
        print(f"Error determining embedding dimension: {e}")
        print("Cannot proceed without embedding dimension.")
        return

    # Step 4: Pinecone Index Setup (Moved before embedding loop)
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with dimension {embedding_dimension}...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embedding_dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            print("Please check your Pinecone account, API key, and plan limits.")
            return  # Exit if index creation fails
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Targeting Pinecone index: {PINECONE_INDEX_NAME}")

    # Step 5: Embedding Generation and Incremental Upsert to Pinecone
    print(f"Generating embeddings using model: {OPENAI_EMBEDDING_MODEL} and upserting incrementally...")
    successful_upserts = 0
    for i, chunk_data in enumerate(chunks_with_metadata):
        try:
            print(f"Processing chunk {i+1}/{len(chunks_with_metadata)}: ID {chunk_data['id']}...")
            response = openai_client.embeddings.create(
                input=chunk_data["text"],
                model=OPENAI_EMBEDDING_MODEL
            )
            embedding = response.data[0].embedding
            
            vector_to_upsert = {
                "id": chunk_data["id"],
                "values": embedding,
                "metadata": {"source": chunk_data["source"], "text": chunk_data["text"]}
            }
            
            upsert_response = index.upsert(vectors=[vector_to_upsert])
            print(f"  Upserted chunk {chunk_data['id']}. Response: {upsert_response}")
            successful_upserts += 1

        except Exception as e:
            print(f"Error processing or upserting chunk {chunk_data['id']}: {e}")
            print(f"  Skipping this chunk. {successful_upserts} chunks successfully upserted so far.")
            # Optionally, you could implement a retry mechanism here or log failed chunks for later processing.
            continue 
    
    print(f"Data ingestion process completed. {successful_upserts}/{len(chunks_with_metadata)} chunks successfully processed and upserted.")
    if successful_upserts > 0:
        print(f"Final index stats: {index.describe_index_stats()}")
    else:
        print("No chunks were successfully upserted.")

if __name__ == "__main__":
    main() 