import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone as LangchainPinecone # Old import
from langchain_pinecone import Pinecone as LangchainPinecone # New recommended import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") # This is the global one
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Kept for debug printing, not direct client use

# CRITICAL: Override EMBEDDING_MODEL for consistency
# Uncomment the next line to force a specific model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")  # Using latest model
print(f"EMBEDDING_MODEL explicitly set to: {EMBEDDING_MODEL}")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_EMBEDDING_DIMENSION = os.getenv("PINECONE_EMBEDDING_DIMENSION", 1536)

# Clean EMBEDDING_MODEL value (just in case)
if EMBEDDING_MODEL and "#" in EMBEDDING_MODEL:
    print(f"Warning: EMBEDDING_MODEL '{EMBEDDING_MODEL}' contains a # character. Stripping comment.")
    EMBEDDING_MODEL = EMBEDDING_MODEL.split("#")[0].strip()
if EMBEDDING_MODEL:
    EMBEDDING_MODEL = EMBEDDING_MODEL.strip().strip('"').strip("'")

# Critical check for EMBEDDING_MODEL after loading and cleaning
if not EMBEDDING_MODEL:
    print("Error: EMBEDDING_MODEL is not set or is invalid after cleaning. Please set a valid model in your .env file (e.g., text-embedding-3-small).")
    exit() # Exit if no valid model can be determined

# Debug: Print environment variables
print("Environment variables:")
print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not Set'}")
print(f"PINECONE_API_KEY: {'Set' if PINECONE_API_KEY else 'Not Set'}")
print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
print(f"PINECONE_CLOUD: {PINECONE_CLOUD}")
print(f"PINECONE_REGION: {PINECONE_REGION}")
print(f"PINECONE_EMBEDDING_DIMENSION: {PINECONE_EMBEDDING_DIMENSION}")

def prepare_cheese_data(cheese_item):
    '''Prepares a single cheese item for vector storage, creating a text representation and extracting metadata.'''
    related_str = "Related Items: "
    for item in cheese_item.get("related_items", []):
        related_str += f"{item.get('name', 'N/A').strip()}, "
    # Create a comprehensive text representation for embedding
    text_representation = f"Name: {cheese_item.get('name', 'N/A')}; " \
                          f"Category: {cheese_item.get('category', 'N/A')}; " \
                          f"Brand: {cheese_item.get('brand', 'N/A')}; " \
                          f"Description: {cheese_item.get('description', 'N/A')}; " \
                          f"Price: {cheese_item.get('price', 'N/A')}; " \
                          f"Price per unit: {cheese_item.get('per_price', 'N/A')}; " \
                          f"Warning text: {cheese_item.get('warning_text', 'N/A')}; " \
                          f"Related items: {related_str}." \
                          f"SKU: {cheese_item.get('SKU', 'N/A')}. " \
                          f"UPC: {cheese_item.get('UPC', 'N/A')}. "

    # if cheese_item.get('info'):
    #     info = cheese_item['info']
    #     text_representation += f"Case number: {info.get('case_number', 'N/A')}. " \
    #                            f"Case size: {info.get('case_size', 'N/A')}. " \
    #                            f"Case weight: {info.get('case_weight', 'N/A')}. " \
    #                            f"Each number: {info.get('each_number', 'N/A')}. " \
    #                            f"Each size: {info.get('each_size', 'N/A')}. " \
    #                            f"Each weight: {info.get('each_weight', 'N/A')}. "
    
    # Extract metadata
    metadata = {
        "name": str(cheese_item.get("name", "N/A")),
        "category": str(cheese_item.get("category", "Unknown Category")).lower(),
        "brand": str(cheese_item.get("brand", "Unknown Brand")).lower(),
        "price_str": str(cheese_item.get("price", "N/A")),
        "per_price_str": str(cheese_item.get("per_price", "N/A")),
        "status": str(cheese_item.get("status", "exist")).lower(),
        "image_url": str(cheese_item.get("image_url", "")),
        "sku": str(cheese_item.get("SKU", "N/A")),
        "upc": str(cheese_item.get("UPC", "N/A")),
        "warning_text": str(cheese_item.get("warning_text", "")),
        "small_images": ",".join(cheese_item.get("small_images", [])),
        "case_status": str(cheese_item.get("case_status", "N/A")),
        "description": str(cheese_item.get("description", "N/A")),
        "related_items": str(cheese_item.get("related_items", "N/A")),
        "more_url": str(cheese_item.get("more_url", "N/A"))
    }
    
    # Attempt to convert price to float for potential range queries
    try:
        price_value = cheese_item.get("price", "").replace("$", "")
        metadata["price_float"] = float(price_value) if price_value and price_value != "N/A" else -1.0
    except ValueError:
        metadata["price_float"] = -1.0

    # Clean metadata
    for key, value in metadata.items():
        if isinstance(value, str) and len(value) > 500:
            metadata[key] = value[:500] + "..."
        if value is None:
            metadata[key] = "N/A"

    # Generate a unique ID
    item_id = str(cheese_item.get("SKU"))
    if not item_id or item_id == "N/A":
        item_id = hashlib.md5(cheese_item.get("name", "").encode('utf-8')).hexdigest()
    
    return item_id, text_representation, metadata

def main():
    '''Main function to load data and store in Pinecone using LangChain.'''
    
    # Use a local variable for Pinecone API key to handle modifications
    processed_pinecone_api_key = PINECONE_API_KEY 

    if not all([OPENAI_API_KEY, processed_pinecone_api_key, PINECONE_INDEX_NAME]):
        print("Error: Missing one or more critical environment variables. Please check your .env file.")
        print(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not Set'}")
        print(f"PINECONE_API_KEY: {'Set' if processed_pinecone_api_key else 'Not Set'}")
        print(f"PINECONE_INDEX_NAME: {'Set' if PINECONE_INDEX_NAME else 'Not Set'}")
        return
    
    # Debug: Check API key format and clean the processed_pinecone_api_key
    if processed_pinecone_api_key:
        # Check for common issues with API keys
        if processed_pinecone_api_key.startswith('"') and processed_pinecone_api_key.endswith('"'):
            print("Warning: Your API key appears to have quotes around it. Removing them...")
            processed_pinecone_api_key = processed_pinecone_api_key.strip('"')
        
        if processed_pinecone_api_key.startswith("'") and processed_pinecone_api_key.endswith("'"):
            print("Warning: Your API key appears to have quotes around it. Removing them...")
            processed_pinecone_api_key = processed_pinecone_api_key.strip("'")
            
        if "#" in processed_pinecone_api_key:
            print("Warning: Your API key contains a # character, which might indicate a comment in the .env file")
            processed_pinecone_api_key = processed_pinecone_api_key.split("#")[0].strip()
        
        # After cleaning, check if it became empty
        if not processed_pinecone_api_key:
            print("Error: Pinecone API key became empty after cleaning. Please check its value in .env file.")
            return

        masked_key = processed_pinecone_api_key[:5] + "..." + processed_pinecone_api_key[-5:] if len(processed_pinecone_api_key) > 10 else "***"
        print(f"Processed PINECONE_API_KEY length: {len(processed_pinecone_api_key)}, format: {masked_key}")
    else:
        # This case should have been caught by the 'if not all' check if it was initially None/empty
        # but good to have a specific message if it's None here for any other reason.
        print("Error: PINECONE_API_KEY is not set.")
        return
    
    print("\nInitializing components...")
    
    try:
        # Initialize Pinecone
        print("Initializing Pinecone...")
        print(f"Using cloud: {PINECONE_CLOUD}, region: {PINECONE_REGION}")
        
        # Create Pinecone client without environment parameter, using the processed key
        pinecone_client = Pinecone(api_key=processed_pinecone_api_key)
        
        # Verify connection by listing indexes
        print("Checking Pinecone connection...")
        try:
            indexes_list_response = pinecone_client.list_indexes()
            # The response is an IndexList object, get names like: [index.name for index in indexes_list_response]
            # or directly from indexes_list_response.names if that attribute exists (check SDK version)
            # Assuming IndexList has a 'names' attribute or similar based on newer SDKs.
            # If it's a list of Index objects, use: index_names = [idx.name for idx in indexes_list_response]
            
            # Let's be safe and iterate if it's iterable, or access .names
            index_names = []
            if hasattr(indexes_list_response, 'names') and isinstance(indexes_list_response.names, list):
                 index_names = indexes_list_response.names
            elif isinstance(indexes_list_response, list): # list of Index objects
                 index_names = [idx.name for idx in indexes_list_response if hasattr(idx, 'name')]
            else: # Fallback for older SDKs or different response structures
                # This might need adjustment based on the exact structure of indexes_list_response
                # For now, let's assume it's a list of objects with a 'name' attribute
                try:
                    index_names = [idx['name'] for idx in indexes_list_response] # Common for older list_indexes()
                except (TypeError, KeyError):
                     print("Could not parse index list. Response:", indexes_list_response)

            # NEW: Check if our index already exists and delete it for a fresh start
            if PINECONE_INDEX_NAME in index_names:
                print(f"Found existing index '{PINECONE_INDEX_NAME}'. Deleting for a clean re-indexing...")
                confirm = input("Are you sure you want to delete the existing index and start fresh? (y/n): ")
                if confirm.lower() == 'y':
                    pinecone_client.delete_index(PINECONE_INDEX_NAME)
                    print(f"Index '{PINECONE_INDEX_NAME}' deleted.")
                    # Wait a moment for deletion to complete
                    time.sleep(5)
                    # Recreate index_names list after deletion
                    indexes_list_response = pinecone_client.list_indexes()
                    if hasattr(indexes_list_response, 'names') and isinstance(indexes_list_response.names, list):
                        index_names = indexes_list_response.names
                    elif isinstance(indexes_list_response, list):
                        index_names = [idx.name for idx in indexes_list_response if hasattr(idx, 'name')]
                    else:
                        try:
                            index_names = [idx['name'] for idx in indexes_list_response]
                        except (TypeError, KeyError):
                            print("Could not parse index list after deletion.")
                else:
                    print("Deletion cancelled. Continuing with existing index.")
                    
            # Create index if it doesn't exist (existing logic)
            if not indexes_list_response or PINECONE_INDEX_NAME not in index_names:
                print(f"Creating Pinecone index: '{PINECONE_INDEX_NAME}' with dimension {PINECONE_EMBEDDING_DIMENSION}...")
                print(f"Using model: {EMBEDDING_MODEL}")
                
                pinecone_client.create_index(
                    name=PINECONE_INDEX_NAME, 
                    dimension=int(PINECONE_EMBEDDING_DIMENSION), 
                    metric='cosine',
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
                )
                print(f"Index '{PINECONE_INDEX_NAME}' creation initiated. Waiting for initialization...")
                # Wait for index to be ready
                while True:
                    status = pinecone_client.describe_index(PINECONE_INDEX_NAME).status
                    if status['ready']:
                        print("Index is ready.")
                        break
                    print(f"Index not ready yet. Current state: {status.get('state', 'Unknown')}. Waiting...")
                    time.sleep(10) # Increased wait time
        except Exception as e:
            print(f"Error during Pinecone index check/creation: {str(e)}")
            print("Please check your API key, network connection, and Pinecone console (https://app.pinecone.io/).")
            import traceback
            traceback.print_exc()
            return
        
        # Get the index directly
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        
        # Initialize embeddings
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    
        # Initialize Pinecone vector store with the direct index
        print("Initializing Pinecone vector store...")
        vectorstore = LangchainPinecone(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace="cheese"
        )
    
        # Load cheese data from JSON file
        print("\nLoading cheese data...")
        try:
            with open("scraped_cheese_data.json", 'r', encoding='utf-8') as f:
                all_cheese_data = json.load(f)
                print(f"Successfully loaded JSON data with {len(all_cheese_data)} items")
        except FileNotFoundError:
            print("Error: scraped_cheese_data.json not found.")
            return
        except json.JSONDecodeError:
            print("Error: Could not decode JSON from scraped_cheese_data.json.")
            return
    
        print(f"Loaded {len(all_cheese_data)} cheese items from JSON.")
    
        # Process and store data
        print("\nProcessing documents...")
        documents = []
        for cheese_item in all_cheese_data:
            item_id, text_to_embed, metadata = prepare_cheese_data(cheese_item)
            
            if not text_to_embed:
                print(f"Skipping item {item_id} due to empty text representation.")
                continue
    
            # Create a Document object
            doc = Document(
                page_content=text_to_embed,
                metadata=metadata
            )
            documents.append(doc)
    
        print(f"Created {len(documents)} documents")
    
        # Split documents if they're too long
        print("\nSplitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
    
        # Add documents to vector store
        print(f"\nAdding {len(split_docs)} documents to vector store...")
        try:
            vectorstore.add_documents(split_docs)
            print("Successfully added documents to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
        print("\nData ingestion complete.")
    except Exception as e:
        print(f"An unexpected error occurred in main: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 