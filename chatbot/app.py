import streamlit as st
import time
import json # Import json module
from datetime import datetime # For timestamps
from utils import ( 
    get_openai_client,
    get_pinecone_index,
    query_pinecone,
    get_llm_response,
    extract_filters_from_query,
    get_enhanced_cheese_description_openai # Import the new function
)

# Custom JSON encoder to handle non-serializable objects
class PineconeEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert non-serializable objects to strings
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

# Function to safely convert Pinecone results to JSON-serializable format
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Convert any other type to string
        return str(obj)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="🧀 Cheese Chatbot",
    page_icon="🧀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Buttons ---
# st.sidebar.write("Actions:") # Optional title for the buttons section
# Removed Stop button from here, keeping only Clear button
# col1, col2 = st.sidebar.columns(2) # No longer need two columns if only one button

# with col1:
#     if st.button("🛑 Stop", help="Stop current response generation (experimental)"):
#         st.session_state.stop_generation_requested = True
#         # Note: Full stop functionality requires integration into the streaming loop.
#         st.toast("Stop request sent. Full stop is experimental.", icon="🛑")

# Place Clear button directly in sidebar
# if st.sidebar.button("🗑️ Clear History", help="Clear chat history", use_container_width=True):
#     st.session_state.messages = [{"role": "assistant", "content": "Hello again! How can I help you find some cheese?"}]
#     st.session_state.last_contexts = []
#     st.session_state.processed_context_for_display = None
#     if 'stop_generation_requested' in st.session_state:
#         del st.session_state.stop_generation_requested
#     st.toast("Chat history cleared!", icon="🗑️")
#     st.rerun()

# --- Initialize Clients (with error handling) ---
try:
    openai_client = get_openai_client()
    pinecone_index = get_pinecone_index()
except ValueError as e:
    st.error(f"Initialization Error: {e} Please check your .env file and ensure the ingestion script has been run.")
    st.stop()

# --- Chat History and State Management (Multi-Session) ---
INITIAL_ASSISTANT_MESSAGE = {"role": "assistant", "content": "Hello there! How can I help you find some cheese today?"}

# Function to create a new chat session structure
def create_new_chat_session(name=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    default_name = name if name else f"Chat {timestamp}"
    return {
        "name": default_name,
        "messages": [INITIAL_ASSISTANT_MESSAGE.copy()],
        "last_contexts": [],
        "processed_context_for_display": None
    }

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = [create_new_chat_session("Chat 1")] # Initial chat named "Chat 1"
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = 0 

# These global flags remain as they are, not per-session
if 'is_generating' not in st.session_state: 
    st.session_state.is_generating = False
if 'stop_generation_requested' in st.session_state: # Initialize if not present
    st.session_state.stop_generation_requested = False

# Add an expander to the sidebar that shows the retrieved contexts
# active_session = st.session_state.chat_sessions[st.session_state.active_chat_index]
# with st.sidebar.expander("🔍 Retrieved Contexts", expanded=False):
#     # Get the contexts from the active session
#     last_contexts = active_session.get("last_contexts", [])
#     if last_contexts:
#         st.code(json.dumps(last_contexts, indent=2), language="json")
#     else:
#         st.write("No context data available yet.")

# --- Sidebar: New Chat Button and Clear Active Chat Button ---

# Create a more reliable and visible retrieval context display
# st.sidebar.divider()
# st.sidebar.write("🔍 Retrieved Information:")
# active_session = st.session_state.chat_sessions[st.session_state.active_chat_index]
# last_contexts = active_session.get("last_contexts", [])

# if not last_contexts:
#     st.sidebar.info("No context data available yet. Ask a question about cheese to see retrieval data here.")
# else:
#     # Display count of retrieved items
#     st.sidebar.success(f"Found {len(last_contexts)} items from Pinecone")
    
#     # Create a more noticeable expander that's expanded by default for visibility
#     with st.sidebar.expander("View Retrieval Data", expanded=True):
#         # Pretty print the JSON with proper indentation
#         try:
#             # Sanitize the data first to make it JSON-serializable
#             sanitized_contexts = sanitize_for_json(last_contexts)
#             # Use the custom encoder as a fallback for any objects that weren't properly sanitized
#             json_str = json.dumps(sanitized_contexts, indent=2, cls=PineconeEncoder)
#             st.code(json_str, language="json")
#         except Exception as e:
#             st.error(f"Error displaying contexts: {e}")
#             # Fallback to display a simpler representation
#             st.write("Retrieved context data (simplified view):")
#             for i, item in enumerate(last_contexts):
#                 st.write(f"Item {i+1}: {type(item).__name__} object")
#                 if hasattr(item, 'metadata') and item.metadata:
#                     st.write(f"  Metadata keys: {', '.join(item.metadata.keys()) if hasattr(item.metadata, 'keys') else str(type(item.metadata))}")
#                 elif isinstance(item, dict) and 'metadata' in item:
#                     meta = item.get('metadata', {})
#                     st.write(f"  Metadata keys: {', '.join(meta.keys()) if hasattr(meta, 'keys') else str(type(meta))}")

# st.sidebar.divider()
st.sidebar.write("Chat Management:")

# Helper function to generate a name for a chat session
def generate_chat_name(session_messages):
    first_user_message_content = None
    for msg in session_messages:
        if msg["role"] == "user":
            first_user_message_content = msg["content"]
            break
    
    if first_user_message_content:
        # Take first few words, e.g., up to 30 chars or 5 words
        name_candidate = " ".join(first_user_message_content.split()[:5])
        if len(name_candidate) > 30:
            name_candidate = name_candidate[:27] + "..."
        return name_candidate if name_candidate else f"Chat {datetime.now().strftime('%H:%M:%S')}"
    return f"Chat {datetime.now().strftime('%H:%M:%S')}"

if st.sidebar.button("➕ New Chat", help="Start a new chat session", use_container_width=True):
    # Auto-name the current (soon to be previous) chat session if it's still "New Chat" or similar generic name
    current_active_session = st.session_state.chat_sessions[st.session_state.active_chat_index]
    if current_active_session["name"].startswith("Chat ") or current_active_session["name"].startswith("New Chat"): # Check if it needs naming
        # Only generate name if there are more than just the initial assistant message
        if len(current_active_session["messages"]) > 1:
             current_active_session["name"] = generate_chat_name(current_active_session["messages"])
        else: # If no user interaction, keep a generic timed name or number it
            current_active_session["name"] = f"Chat {st.session_state.active_chat_index + 1}"

    # Create and switch to the new chat
    new_session_name = f"New Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions.append(create_new_chat_session(name=new_session_name))
    st.session_state.active_chat_index = len(st.session_state.chat_sessions) - 1
    # Reset flags for the new chat
    st.session_state.is_generating = False
    st.session_state.stop_generation_requested = False
    st.rerun()

# Modify Clear History to clear the active chat
# if st.sidebar.button("🗑️ Clear Active Chat", help="Clear messages in the current chat session", use_container_width=True):
#     active_session = st.session_state.chat_sessions[st.session_state.active_chat_index]
#     active_session["messages"] = [INITIAL_ASSISTANT_MESSAGE.copy()]
#     active_session["last_contexts"] = []
#     active_session["processed_context_for_display"] = None
#     # Reset flags for the cleared chat
#     st.session_state.is_generating = False 
#     st.session_state.stop_generation_requested = False 
#     st.toast("Current chat cleared!", icon="🗑️")
#     st.rerun()

st.sidebar.divider()

# Display chat sessions in the sidebar for switching
ch_col1, ch_col2 = st.sidebar.columns([0.7, 0.3]) # Columns for label and button
with ch_col1:
    st.sidebar.write("Chat History:")
with ch_col2:
    if st.button("🗑️", key="clear_all_chats", help="Delete all chat sessions and start fresh", use_container_width=True):
        st.session_state.chat_sessions = [create_new_chat_session("Chat 1")]
        st.session_state.active_chat_index = 0
        st.session_state.is_generating = False
        st.session_state.stop_generation_requested = False
        st.toast("All chats cleared!", icon="🗑️")
        st.rerun()

for idx, session in enumerate(st.session_state.chat_sessions):
    session_display_name = session.get("name", f"Chat {idx + 1}")
    # Use columns to place delete button next to chat name button
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        if st.button(session_display_name, key=f"switch_chat_{idx}", use_container_width=True, help=f"Switch to {session_display_name}"):
            st.session_state.active_chat_index = idx
            # Reset flags when switching chat
            st.session_state.is_generating = False
            st.session_state.stop_generation_requested = False
            st.rerun()
    with col2:
        if st.button("🗑️", key=f"delete_chat_{idx}", help=f"Delete {session_display_name}"):
            if len(st.session_state.chat_sessions) > 1: # Prevent deleting the last chat
                st.session_state.chat_sessions.pop(idx)
                # Adjust active_chat_index carefully
                if st.session_state.active_chat_index >= idx:
                    st.session_state.active_chat_index = max(0, st.session_state.active_chat_index -1)
                # If active index became out of bounds (e.g. deleted last one and active was last)
                if st.session_state.active_chat_index >= len(st.session_state.chat_sessions):
                    st.session_state.active_chat_index = len(st.session_state.chat_sessions) -1

            else: # If only one chat left, clear it instead of deleting
                st.session_state.chat_sessions[0] = create_new_chat_session(name="Chat 1")
                st.session_state.active_chat_index = 0
            st.rerun()

st.sidebar.divider() # Add one more divider at the end of sidebar content

# --- Helper function to format context for display and LLM ---
def format_context_for_llm(contexts, aggregate_data=None): # Added aggregate_data parameter
    if aggregate_data:
        return aggregate_data # If aggregate_data is provided, use it directly

    if not contexts:
        return "No relevant cheese information found in the database."
    
    formatted_texts = []
    for i, context in enumerate(contexts):
        meta = context.get('metadata', {})
        cheese_name = meta.get('name', 'N/A')
        description = meta.get('description', 'N/A')

        # Condition to fetch enhanced description
        if cheese_name != 'N/A' and (not description or description == 'N/A' or len(description) < 30): # Arbitrary length check
            print(f"Debug (app.py): Description for '{cheese_name}' is short or missing. Fetching enhanced description.")
            enhanced_description = get_enhanced_cheese_description_openai(cheese_name)
            if enhanced_description:
                description = enhanced_description # Replace original description
            else:
                print(f"Debug (app.py): Failed to get enhanced description for '{cheese_name}'. Using original.")

        text = f"Item {i+1}:\n"
        text += f"  Name: {cheese_name}\n"
        text += f"  Category: {meta.get('category', 'N/A')}\n"
        text += f"  Brand: {meta.get('brand', 'N/A')}\n"
        text += f"  Price: {meta.get('price_str', 'N/A')}\n"
        text += f"  Price per Unit: {meta.get('per_price_str', 'N/A')}\n"
        text += f"  Status: {meta.get('status', 'N/A')}\n"
        text += f"  Image URL: {meta.get('image_url', 'N/A')}\n"
        text += f"  Related Items: {meta.get('related_items', 'N/A')}\n"
        text += f"  Description: {description}\n"  # Use potentially enhanced description
        text += f"  Warning Text: {meta.get('warning_text', 'N/A')}\n"
        text += f"  SKU: {meta.get('sku', 'N/A')}\n"
        text += f"  UPC: {meta.get('upc', 'N/A')}\n"
        text += f"  More URL: {meta.get('more_url', 'N/A')}."
        formatted_texts.append(text)
    return "\n\n".join(formatted_texts)

# --- System Prompt for the LLM ---
SYSTEM_PROMPT = """
A friendly AI assistant specializing in cheese products from Kimelo.com.
The knowledge base contains information about a variety of cheeses.

When a user asks a question, follow these steps:
1. The 'CONTEXT' section is displayed, which contains the user's question and the relevant cheese data retrieved from the database.
2. If the CONTEXT is empty or there is no information to answer the question, clearly state that the information is not in the database.
3. When providing information about a specific cheese, if the user requests a list or multiple cheeses, display up to 6 cheeses.
4. Do not rely solely on the 'CONTEXT' section for information about the cheese, but provide a rich description.
5. For each cheese, clearly list the following details, if possible in context. Also, do not display only the information presented below, but present the content in the body of the text.
    Product Name
    Image (You can display only the image in multiple lines, and display the image in small size. Clicking on the image should display the corresponding Cheese homepage in a new tab. The Cheese homepage URL is "more_url" in the metadata.)
    Brand
    Price (e.g. $16.76)
    Price per lb (e.g. $3.35/lb)
    Description (e.g. This cheese is labeled Brown, and eating this cheese can help you absorb various nutrients and is good for your health. However, {You can display a warning message.})
    Related Items (e.g. Cheese, American, 120 Pieces, Yellow, (4) 5 Pounds - 103674 and Cheese, American, 120 Pieces, Yellow, (4) 5 Pounds - 103674 include URLs.)
    Status: Only state if it explicitly states 'Back in stock' or 'Out of stock'. If it says 'In stock' or 'In stock', there is no need to mention the status. Warning text
    Price and price per mass are displayed on the same line. Price per mass text is smaller than price text. Warning text is red text.
    Sort in the order above.
6. If the number of cheeses found in the context is less than the requested number or less than 6 (if a general list is requested), only list the cheeses that are available. Do not mention the fact that there are fewer items displayed unless there is a specific question about the number.
7. Be concise and helpful. If the user asks a general question (e.g., "Hello"), do not search the database, but answer politely.
8. If the context includes a warning_text for the product, you can mention it if it is relevant to the user's question about product safety or a specific ingredient.
9. After answering the user's question, suggest 2-3 related follow-up questions that the user might have. Please write in a Markdown format.
"""

# --- UI Rendering ---
st.title("🧀 Cheese Chatbot")
st.caption("Ask me about cheese products available at Kimelo.com!")

# Main chat area container
chat_container = st.container()

with chat_container:
    # Display chat messages from the active session
    active_session_messages = st.session_state.chat_sessions[st.session_state.active_chat_index]["messages"]
    for i, message in enumerate(active_session_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
                
            # Updated expander logic to handle both raw contexts and processed aggregate context for the active session
            if message["role"] == "assistant" and (message.get("context_data") or message.get("processed_context_for_display")):
                with st.expander("🔍 View Context Data Used", expanded=False):
                    processed_display = message.get("processed_context_for_display")
                    raw_context_data = message.get("context_data")

                    if processed_display: # For aggregate queries, this is a pre-formatted string
                        # To show it as JSON, we can wrap it or just show the string as is.
                        # For consistency, let's make it a simple JSON object if it's a string.
                        if isinstance(processed_display, str):
                            try:
                                # Attempt to parse if it might be a JSON string already
                                json_data = json.loads(processed_display)
                                st.code(json.dumps(json_data, indent=2), language="json")
                            except json.JSONDecodeError:
                                # If not a JSON string, treat as a summary string and wrap it
                                st.code(json.dumps({"summary_context": processed_display}, indent=2), language="json")
                        elif isinstance(processed_display, (list, dict)): # If it was already structured
                            # Sanitize before attempting to display
                            try:
                                sanitized_display = sanitize_for_json(processed_display)
                                st.code(json.dumps(sanitized_display, indent=2, cls=PineconeEncoder), language="json")
                            except Exception as e:
                                st.error(f"Could not display processed context: {e}")
                                st.write(str(processed_display))
                        else:
                            st.markdown(str(processed_display)) # Fallback

                    elif isinstance(raw_context_data, list) and raw_context_data:
                        # Display raw_context_data (list of dicts from Pinecone) as JSON
                        try:
                            # Sanitize before attempting to display
                            sanitized_data = sanitize_for_json(raw_context_data)
                            st.code(json.dumps(sanitized_data, indent=2, cls=PineconeEncoder), language="json")
                        except Exception as e:
                            st.error(f"Could not serialize context to JSON: {e}")
                            # Show simplified view as fallback
                            st.write("Retrieved context data (simplified view):")
                            for i, item in enumerate(raw_context_data):
                                st.write(f"Item {i+1}: {type(item).__name__} object")
                                if hasattr(item, 'metadata') and item.metadata:
                                    st.write(f"  Metadata keys: {', '.join(item.metadata.keys()) if hasattr(item.metadata, 'keys') else str(type(item.metadata))}")
                    else:
                        st.write("No context data to display for this response.")

# Stop button before the chat input - only show if generating
if st.session_state.get('is_generating', False):
    if st.button("🛑 Stop Generation", help="Attempt to stop the current response generation (experimental)"):
        st.session_state.stop_generation_requested = True
        st.toast("Stop request. Note: Full stop is experimental and may not immediately halt generation.", icon="🛑")

# --- Chat Input and Response Logic ---
# Get a reference to the current active chat session
active_chat_session = st.session_state.chat_sessions[st.session_state.active_chat_index]

if prompt := st.chat_input("What kind of cheese are you looking for?"):
    active_chat_session["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() # For streaming response
        full_response = ""
        retrieved_contexts = [] # To store raw context matches
        processed_context_for_llm = None # Will hold formatted aggregate data for LLM
        processed_context_for_display_this_turn = None # For storing aggregate results for display
        is_aggregate_query = False
        # Simple check for greetings to avoid unnecessary DB query
        greeting_keywords = ["hi", "hello", "hey", "greetings"]
        if any(keyword in prompt.lower() for keyword in greeting_keywords) and len(prompt.split()) < 3:
            full_response = "Hello there! How can I help you find some cheese today?"
            message_placeholder.markdown(full_response)
            st.session_state.is_generating = False # Ensure it's false for greetings
        else:
            st.session_state.is_generating = True # Set to true before starting potentially long operations
            message_placeholder.markdown("Thinking... 💭") # Display thinking message
            
            # Attempt to extract filters 
            metadata_filters = extract_filters_from_query(prompt)

            context_for_llm = "" # Initialize context_for_llm

            if isinstance(metadata_filters, dict) and metadata_filters.get("query_type") == "aggregate_request":
                is_aggregate_query = True # CORRECTED: Should be True if it's an aggregate request
                request_details = metadata_filters.get("request_details")
                print(f"Debug (app.py): Aggregate query detected for {request_details}")

                with st.spinner(f"Gathering information for {request_details}..."):
                    # For aggregate queries, use a generic query text and high top_k.
                    # We use "cheese" as a generic query text; metadata_filters for aggregate is not for Pinecone.
                    # top_k=100 for broader data collection, metadata_filter=None
                    aggregate_query_contexts = query_pinecone("cheese products", top_k=100, metadata_filter=None)

                    if not aggregate_query_contexts:
                        processed_context_for_llm = f"I couldn't retrieve enough data to determine {request_details}."
                    elif request_details == "all_brands":
                        brands = sorted(list(set(
                            item.get('metadata', {}).get('brand', 'N/A').strip().title() 
                            for item in aggregate_query_contexts 
                            if item.get('metadata', {}).get('brand', 'N/A') not in ['N/A', None, '']
                        )))
                        if brands:
                            processed_context_for_llm = "Available brands include: " + ", ".join(brands) + "."
                        else:
                            processed_context_for_llm = "I couldn't find specific brand information in the database."
                    elif request_details == "all_categories":
                        categories = sorted(list(set(
                            item.get('metadata', {}).get('category', 'N/A').strip().title()
                            for item in aggregate_query_contexts
                            if item.get('metadata', {}).get('category', 'N/A') not in ['N/A', None, '']
                        )))
                        if categories:
                            processed_context_for_llm = "Available categories include: " + ", ".join(categories) + "."
                        else:
                            processed_context_for_llm = "I couldn't find specific category information in the database."
                    elif request_details == "price_range":
                        prices = []
                        for item in aggregate_query_contexts:
                            meta = item.get('metadata', {})
                            if meta and isinstance(meta.get('price_float'), (float, int)) and meta['price_float'] >= 0:
                                prices.append(meta['price_float'])
                            elif meta and isinstance(meta.get('price_str'), str):
                                try:
                                    price_val = float(meta['price_str'].replace('$', '').strip())
                                    if price_val >= 0: prices.append(price_val)
                                except ValueError:
                                    continue
                        if prices:
                            min_price = min(prices)
                            max_price = max(prices)
                            processed_context_for_llm = f"The price range for cheeses I found is from ${min_price:.2f} to ${max_price:.2f}."
                        else:
                            processed_context_for_llm = "I couldn't determine the price range from the available data."
                    else:
                        processed_context_for_llm = "I can look for specific cheeses based on name, brand, category, or price. How can I help?"
                        is_aggregate_query = False # CORRECTED: Revert to normal flow for unknown aggregate types

                # For aggregate queries, the LLM gets the processed string; raw contexts are not directly used by LLM.
                # We also store this for display purposes if it was successfully generated.
                if processed_context_for_llm and is_aggregate_query:
                    print("Debuging", "555555555555555555555555555")
                    context_for_llm = format_context_for_llm(contexts=None, aggregate_data=processed_context_for_llm)
                    processed_context_for_display_this_turn = processed_context_for_llm
                retrieved_contexts = [] # Clear raw contexts if it was an aggregate query that was processed
            
            if st.session_state.is_generating:
                print(f"Debug (app.py): Normal query. Filters: {metadata_filters}")
                with st.spinner("Searching our cheese collection..."):
                    actual_filters_for_pinecone = metadata_filters if isinstance(metadata_filters, dict) and "query_type" not in metadata_filters else None
                    retrieved_contexts = query_pinecone(prompt, top_k=100, metadata_filter=actual_filters_for_pinecone)
                    active_chat_session["last_contexts"] = retrieved_contexts # Store in active session
                    context_for_llm = format_context_for_llm(retrieved_contexts) # Format normal context
            
            # Construct messages for LLM, including chat history from the active session
            messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Add previous messages from the active session
            for i in range(len(active_chat_session["messages"]) - 1):
                msg = active_chat_session["messages"][i]
                messages_for_llm.append({"role": msg["role"], "content": msg["content"]})
            
            # Add the current user prompt (the last message in the active session) with context
            current_user_query = active_chat_session["messages"][-1]['content'] 
            current_user_message_content = f"CONTEXT:\\n{context_for_llm}\\n\\nUSER QUESTION: {current_user_query}\\n\\nASSISTANT ANSWER:"
            messages_for_llm.append({"role": "user", "content": current_user_message_content})
            
            # Debug: Print the messages being sent to the LLM
            # print(f"Debug (app.py): Messages for LLM: {messages_for_llm}")
            
            # Get LLM response (streaming)
            try:
                # Pass the structured messages list to the LLM
                stream = get_llm_response(messages_for_llm, stream=True)
                for chunk_wrapper in stream:
                    # For OpenAI API v1.0.0 and later, using Pydantic models
                    if hasattr(chunk_wrapper, 'choices') and chunk_wrapper.choices:
                        chunk = chunk_wrapper.choices[0].delta
                        if hasattr(chunk, 'content') and chunk.content is not None:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌") # Typing indicator
                            time.sleep(0.01) # Small delay for smoother streaming effect
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Sorry, I encountered an error trying to respond: {str(e)}"
                message_placeholder.markdown(full_response)
            finally:
                st.session_state.is_generating = False # Reset flag when generation finishes or errors out
        
        # Store assistant response with context in the active session
        active_chat_session["messages"].append({
            "role": "assistant", 
            "content": full_response, 
            "context_data": retrieved_contexts if not is_aggregate_query else [], 
            "processed_context_for_display": processed_context_for_display_this_turn if is_aggregate_query else None
        })

        # Update the expander with the latest context data after the response is complete
        # This is implicitly handled by re-running and displaying the message log above. 