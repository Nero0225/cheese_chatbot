import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables from .env file in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Not directly used for client init now

# Explicitly set EMBEDDING_MODEL for consistency with ingest_data.py
# EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
print(f"EMBEDDING_MODEL in utils.py explicitly set to: {EMBEDDING_MODEL}")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# --- Clean environment variables ---
# Clean PINECONE_API_KEY
if PINECONE_API_KEY:
    if PINECONE_API_KEY.startswith(('"', "'")) and PINECONE_API_KEY.endswith(('"', "'")) :
        print(f"Warning (utils.py): Pinecone API key had quotes. Stripping them.")
        PINECONE_API_KEY = PINECONE_API_KEY.strip('"').strip("'")
    if "#" in PINECONE_API_KEY:
        print(f"Warning (utils.py): Pinecone API key '{PINECONE_API_KEY}' contained #. Stripping comment.")
        PINECONE_API_KEY = PINECONE_API_KEY.split("#")[0].strip()

# Clean EMBEDDING_MODEL
if EMBEDDING_MODEL:
    if EMBEDDING_MODEL.startswith(('"', "'")) and EMBEDDING_MODEL.endswith(('"', "'")):
        print(f"Warning (utils.py): Embedding model ID had quotes. Stripping them.")
        EMBEDDING_MODEL = EMBEDDING_MODEL.strip('"').strip("'")
    if "#" in EMBEDDING_MODEL:
        print(f"Warning (utils.py): Embedding model ID '{EMBEDDING_MODEL}' contained #. Stripping comment.")
        EMBEDDING_MODEL = EMBEDDING_MODEL.split("#")[0].strip()
# --- End cleaning ---

# Initialize clients
openai_client = None
pinecone_index = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        if not OPENAI_API_KEY: # Check the cleaned key
            raise ValueError("OpenAI API key not found or invalid after cleaning. Please set OPENAI_API_KEY in your .env file.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

def get_pinecone_index():
    global pinecone_index, PINECONE_API_KEY # Ensure we use the potentially modified global PINECONE_API_KEY
    if pinecone_index is None:
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME: # PINECONE_ENVIRONMENT removed from this check
            error_msg = "Pinecone configuration error in .env file (checked by utils.py):\n"
            if not PINECONE_API_KEY: error_msg += "- PINECONE_API_KEY is missing or invalid after cleaning.\n"
            if not PINECONE_INDEX_NAME: error_msg += "- PINECONE_INDEX_NAME is missing.\n"
            # if not PINECONE_ENVIRONMENT: error_msg += "- PINECONE_ENVIRONMENT is missing (though not always required for client init).\n"
            raise ValueError(error_msg)
        
        # Initialize Pinecone client without the environment parameter
        # The API key for serverless indexes typically includes region, or it's inferred.
        print(f"Debug (utils.py): Initializing Pinecone client with API key ending: ...{PINECONE_API_KEY[-5:] if PINECONE_API_KEY else 'N/A'}")
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY) 
        
        existing_indexes_response = pinecone_client.list_indexes()
        existing_indexes_names = [idx.name for idx in existing_indexes_response] # Assuming modern SDK response
        
        if PINECONE_INDEX_NAME not in existing_indexes_names:
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Available: {existing_indexes_names}. Please run the ingest_data.py script first.")
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    return pinecone_index

def get_embedding_openai(text, model=None): # Allow model override, default to cleaned global
    '''Generates an embedding for the given text using OpenAI.'''
    client = get_openai_client()
    current_model = model if model else EMBEDDING_MODEL # Use cleaned global EMBEDDING_MODEL
    if not current_model:
        raise ValueError("Embedding model ID not found or invalid after cleaning. Please set EMBEDDING_MODEL in .env or pass a valid model.")
    
    text = text.replace("\n", " ")
    try:
        # print(f"Debug (utils.py): Requesting embedding with model: {current_model}") # Optional: for debugging model ID issues
        response = client.embeddings.create(input=[text], model=current_model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding with model '{current_model}': {e}")
        # Consider re-raising or returning a more specific error to the UI
        raise  # Re-raise to make it visible in Streamlit

def query_pinecone(query_text, top_k=20, metadata_filter=None):
    '''Queries Pinecone for relevant documents based on the query text and optional metadata filter.'''
    print("query_text", query_text)
    index = get_pinecone_index()
    query_embedding = get_embedding_openai(query_text)
    if not query_embedding:
        return []
    
    # print("Debug (utils.py): metadata_filter", metadata_filter)
    # print("Debug (utils.py): query_embedding", len(query_embedding))
    
    # Define namespace consistently with ingest_data.py
    namespace = "cheese"
    # print(f"Debug (utils.py): Using namespace: {namespace}")
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,
            namespace=namespace  # Explicit namespace
        )
        # print("Debug (utils.py): results", results)
        return results.get('matches', [])
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

def get_llm_response(messages, stream=True):
    '''Gets a response from the LLM (e.g., GPT-4o) for the given list of messages.'''
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=stream
        )
        return response
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        # For non-streaming, you might return a default error message or raise
        # For streaming, the error might need to be handled where the stream is consumed
        if stream:
            # Simulate a stream with an error message if direct error streaming is complex
            def error_stream():
                yield f"Error from LLM: {str(e)}"
            return error_stream()
        return f"Error from LLM: {str(e)}"

# --- Placeholder for metadata filter extraction logic ---
# Updated function to use LLM for filter extraction
FILTER_EXTRACTION_PROMPT_TEMPLATE = """
You are a query understanding assistant. Your task is to analyze a user's query about cheese products and extract structured filters for a database query.
The database contains cheese items with metadata fields: 'name' (string), 'category' (string, e.g., "cheddar", "gouda"), 'brand' (string, e.g., "schreiber", "kimelos own"), 'price_float' (float), 'status' (string, e.g., "exist", "back in stock soon"), 'sku' (string, e.g., "123792").

Convert the user's query into a JSON object representing the filters.
- The JSON object should only contain keys for which a filter is found.
- ALL STRING VALUES IN THE FILTERS (like brand, category, or sku) MUST BE CONVERTED TO LOWERCASE where appropriate (e.g. brand, category), but SKUs should be preserved as extracted if they are case-sensitive or alphanumeric codes.

Supported filter operations for 'price_float':
- "$gte": greater than or equal to
- "$lte": less than or equal to
- If the user says "expensive" or "most expensive", use `{{"price_float": {{"$gte": 50.0}}}}`. (This is a general guideline, if a specific price is mentioned alongside "expensive", prioritize the specific price).
- If the user says "cheap" or "cheapest", use `{{"price_float": {{"$lte": 10.0}}}}`. (Again, prioritize specific prices if mentioned).
- If a price range like "between $X and $Y" is given, extract X and Y and use `{{"price_float": {{"$gte": X, "$lte": Y}}}}`.
- For "over $X", "more than $X", or "$X plus", use `{{"price_float": {{"$gte": X}}}}`.
- For "under $X", "less than $X", or "up to $X", use `{{"price_float": {{"$lte": X}}}}`.
- Ensure extracted price values are numbers (float).

Supported filter operations for 'brand' and 'category' (string fields):
- Exact match after converting the user's specified value to lowercase: e.g., if user says "BrandX", filter should be `{{"brand": "brandx"}}`.
- If the user lists multiple brands (e.g., "BrandX or BrandY"), you can use an "$in" filter if you are confident: `{{"brand": {{"$in": ["brandx", "brandy"]}}}}`. For simplicity, if multiple distinct filterable entities of the same type are mentioned disjunctively (OR), and you cannot form an $in query, it's better to return no filter for that field or pick the most prominent one if clear.

SKU filter ('sku' field, string type):
- If the user mentions an SKU, item number, or product ID, extract it as a string. 
- Example: "cheese with SKU 125731" or "item number 123792" should become `{{"sku": "125731"}}` or `{{"sku": "123792"}}` (preserve case or convert to lower based on typical SKU format, for now, let's say convert to lower if it seems to be a general product code, but preserve if it looks like a fixed ID like 'CHZ123'. Let's default to preserving the exact string as found in the query for SKU.).
- Final decision: SKUs should be extracted as exact strings. If the user says "SKU 125731", the filter is `{{"sku": "123792"}}`.

Special query types (if detected, prioritize returning this over other filters):
- If the query is a general request for information like "show me all brands", "list all categories", or "what is the price range of cheeses?", return a JSON like:
  `{{"query_type": "aggregate_request", "request_details": "all_brands"}}`
  `{{"query_type": "aggregate_request", "request_details": "all_categories"}}`
  `{{"query_type": "aggregate_request", "request_details": "price_range"}}`
  If an aggregate request is detected, do not include other filters like price or specific brand unless the aggregate request itself is constrained (e.g., "show me all brands that offer cheddar" - this is complex, for now, treat broad aggregate requests simply).

- If the query is asking about the count or total number of cheese categories or types (e.g., "How many different kinds of cheese products do you have?", "How many cheese categories in total?", "What's the total number of cheese types?"), return:
  `{{"query_type": "aggregate_request", "request_details": "count_categories"}}`

Status filter:
- If the query asks for items "out of stock" or "unavailable", use `{{"status": "back in stock soon"}}`.

If no filters or special query types are confidently extracted, return an empty JSON object `{{}}`.
Do not invent filters. Only use information explicitly present in the query.

User Query: {user_query}

Valid JSON Output:
"""

def extract_filters_from_query(user_query):
    """
    Uses an LLM to extract potential filters or identify aggregate query types
    from a user query.
    """
    client = get_openai_client()
    prompt = FILTER_EXTRACTION_PROMPT_TEMPLATE.format(user_query=user_query)

    # print(f"Debug (extract_filters_from_query): Sending prompt to LLM for filter extraction:\\n{prompt}")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, # Or a cheaper/faster model if preferred for this task
            messages=[{"role": "system", "content": "You are an expert at converting user queries into JSON filters."},
                      {"role": "user", "content": prompt}],
            temperature=0.0, # For deterministic output
            response_format={"type": "json_object"} # Request JSON output
        )
        
        llm_output_content = response.choices[0].message.content
        # print(f"Debug (extract_filters_from_query): LLM raw output: {llm_output_content}")
        
        if not llm_output_content:
            print("Warning (extract_filters_from_query): LLM returned empty content for filters.")
            return None # Or {}

        extracted_json = json.loads(llm_output_content)
        
        # Basic validation / cleaning of extracted JSON (optional, but good practice)
        # For example, ensure price_float values are actually floats
        if "price_float" in extracted_json and isinstance(extracted_json["price_float"], dict):
            for key, value in extracted_json["price_float"].items():
                if isinstance(value, str):
                    try:
                        extracted_json["price_float"][key] = float(value)
                    except ValueError:
                        print(f"Warning (extract_filters_from_query): Could not convert price value '{value}' to float. Removing filter.")
                        del extracted_json["price_float"] # Or handle error more gracefully
                        break
        
        print(f"Debug (extract_filters_from_query): Extracted filters: {extracted_json}")

        if not extracted_json: # If LLM returns an empty object explicitly
            return None
        return extracted_json

    except json.JSONDecodeError as e:
        print(f"Error (extract_filters_from_query): Could not decode JSON from LLM output: {llm_output_content}. Error: {e}")
        return None # Failed to get valid filters
    except Exception as e:
        print(f"Error (extract_filters_from_query): Exception during LLM call or processing: {e}")
        return None 

def get_enhanced_cheese_description_openai(cheese_name):
    """Gets an enhanced, customer-facing description for a cheese using OpenAI."""
    if not cheese_name or cheese_name == 'N/A':
        return None

    client = get_openai_client()
    prompt_messages = [
        {
            "role": "system", 
            "content": "You are a creative food writer. Your task is to generate an appealing and informative description for a cheese, suitable for a customer. Focus on general characteristics, potential taste profiles, texture, common culinary uses, or interesting facts. Do not invent specific brand details, prices, or availability. Make it sound delicious and inviting. Keep it concise, around 2-4 sentences."
        },
        {
            "role": "user", 
            "content": f"Please provide an enhanced description for the following cheese: {cheese_name}"
        }
    ]

    print(f"Debug (utils.py): Requesting enhanced description for: {cheese_name}")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, # Or a potentially cheaper model like gpt-3.5-turbo if cost is a concern for descriptions
            messages=prompt_messages,
            temperature=0.7, # Allow for some creativity
            max_tokens=150, # Limit length of description
            stream=False # Not streaming for this internal call
        )
        enhanced_description = response.choices[0].message.content.strip()
        print(f"Debug (utils.py): Enhanced description for {cheese_name}: {enhanced_description}")
        return enhanced_description
    except Exception as e:
        print(f"Error getting enhanced description for '{cheese_name}': {e}")
        return None # Return None if there's an error 

def generate_product_description_openai(cheese_name, category, brand):
    """Generates a product description using OpenAI, given name, category, and brand."""
    if not cheese_name or cheese_name == 'N/A':
        return "No specific description available for this product at the moment."

    client = get_openai_client() # Uses existing client from utils
    
    prompt_parts = [f"Product Name: {cheese_name}"]
    if category and category != 'N/A':
        prompt_parts.append(f"Category: {category}")
    if brand and brand != 'N/A':
        prompt_parts.append(f"Brand: {brand}")
    
    input_details = "\n".join(prompt_parts)

    prompt_messages = [
        {
            "role": "system",
            "content": "You are a product description writer for an online cheese shop. Based on the provided cheese name, and optionally its category and brand, write a concise and appealing product description of 2-3 sentences. Focus on its likely taste, texture, common culinary uses, or interesting characteristics. Do not mention price, availability, or the current year. If category or brand is not provided, focus on the cheese name itself to infer general properties."
        },
        {
            "role": "user",
            "content": f"Please generate a product description for the following cheese:\n{input_details}"
        }
    ]

    print(f"Debug (utils.py): Requesting LLM-generated product description for: {cheese_name}")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, # Uses existing LLM_MODEL from utils
            messages=prompt_messages,
            temperature=0.6, 
            max_tokens=120, # Adjusted for 2-3 sentences
            stream=False
        )
        generated_description = response.choices[0].message.content.strip()
        print(f"Debug (utils.py): Generated description for {cheese_name}: {generated_description}")
        return generated_description if generated_description else "A delightful cheese, perfect for various occasions." # Fallback if empty
    except Exception as e:
        print(f"Error generating product description for '{cheese_name}' via LLM: {e}")
        return "Explore this cheese to discover its unique qualities." # Fallback in case of error 