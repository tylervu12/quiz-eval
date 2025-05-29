import os
import json
import re
import uuid # Added for generating user_id
# from openai import OpenAI # Replaced by langchain_openai
# from pinecone import Pinecone # Replaced by langchain_pinecone
# from langsmith import traceable # Langchain components are auto-traced

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field # Use Pydantic v2
from typing import List, Dict, Any, Optional, Tuple # Added Tuple
from langchain.evaluation import load_evaluator # For LLM-as-judge style evaluations
import httpx # For Supabase client

# Attempt to load .env file for local development FIRST
# This ensures environment variables are available globally when the script loads.
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("INFO: Successfully loaded .env file.")
    else:
        print("INFO: .env file not found or empty. Relying on system/Lambda environment variables.")
except ImportError:
    print("INFO: python-dotenv not installed. Relying on system/Lambda environment variables. For local .env support, run: pip install python-dotenv")

# Initialize clients (outside handler for reuse)
# Ensure environment variables are set in your Lambda configuration OR .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL_NAME = "gpt-4o"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Critical checks for actual presence of API keys.
# The script should not proceed if these are not set at all.
if not OPENAI_API_KEY:
    raise ValueError("CRITICAL ERROR: OPENAI_API_KEY environment variable not set. Please check your .env file or system environment variables.")
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("CRITICAL ERROR: PINECONE_API_KEY environment variable not set. Please check your .env file or system environment variables.")
if not PINECONE_INDEX_NAME:
    raise ValueError("CRITICAL ERROR: PINECONE_INDEX_NAME environment variable not set. Please check your .env file or system environment variables.")
if not SUPABASE_URL:
    raise ValueError("CRITICAL ERROR: SUPABASE_URL environment variable not set. Please check your .env file or system environment variables.")
if not SUPABASE_SERVICE_KEY:
    raise ValueError("CRITICAL ERROR: SUPABASE_SERVICE_KEY environment variable not set. Please check your .env file or system environment variables.")

# Initialize Langchain components
try:
    # LLM for chat completions
    llm = ChatOpenAI(model=GPT_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.2) # Temp for categorization
    # LLM for recommendations (can use a different temperature)
    recommendation_llm = ChatOpenAI(model=GPT_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.7)
    # LLM for evaluations (can be a different model or temperature if needed, e.g., gpt-3.5-turbo for cost/speed)
    eval_llm = ChatOpenAI(model=GPT_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.0) # Low temp for consistent evals
    print("INFO: ChatOpenAI clients (main, recommendation, eval) initialized.")
    
    # Embeddings model
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    print("INFO: OpenAIEmbeddings client initialized.")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to initialize Langchain OpenAI components: {e}")

# Initialize Pinecone Vector Store
# PineconeVectorStore.from_existing_index will use PINECONE_API_KEY and PINECONE_ENVIRONMENT from env vars if set
pinecone_vectorstore = None
if "your_pinecone_index_name" in PINECONE_INDEX_NAME.lower() or not PINECONE_INDEX_NAME.strip():
    print(f"WARNING: PINECONE_INDEX_NAME ('{PINECONE_INDEX_NAME}') appears to be a placeholder or empty. PineconeVectorStore will not be initialized.")
else:
    try:
        pinecone_vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
            # Pinecone client args like api_key and environment are typically read from env vars by the underlying Pinecone client
        )
        print(f"INFO: PineconeVectorStore for index '{PINECONE_INDEX_NAME}' initialized.")
        # To verify, you could try a dummy search if the index isn't empty:
        # print(pinecone_vectorstore.similarity_search("test query", k=1))
    except Exception as e:
        print(f"ERROR: Could not initialize PineconeVectorStore for index '{PINECONE_INDEX_NAME}': {e}. RAG operations will fail.")

# Global constants
CLASSIFICATION_TAGS = [
    "Lose weight", "Gain weight", "Reduce pain",
    "Training for an event", "Recover from injury"
]
# Mapping from classification tags to the exact values in Pinecone 'source' metadata field
TAG_TO_PINECONE_SOURCE_VALUE = {
    "Lose weight": "weight_loss",
    "Gain weight": "muscle_gain",
    "Reduce pain": "reduce_pain",
    "Training for an event": "event",
    "Recover from injury": "injury"
}
MAX_RAG_CONTEXT_LENGTH = 7000 # Approximate characters for ~1500 tokens, adjust as needed

# Define Pydantic models for structured output
class CategorizedTags(BaseModel):
    tags: Optional[List[str]] = Field(default_factory=list, description="List of classified tags. Should be one of CLASSIFICATION_TAGS or empty/None if no tags apply.")

def slugify(text):
    """Helper function to convert text to a slug format for Pinecone metadata filtering."""
    text = text.lower()
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^\w_]', '', text) # Keep alphanumeric and underscores
    return text

def parse_webhook_payload(payload):
    """
    Parses the webhook payload to extract user input variables.
    Expected keys: primary_goal, specific_event, past_attempts, schedule, injuries
    Also looks for an optional user_id, generating one if not provided.
    """
    user_id = payload.get("user_id")
    if not user_id: # If user_id is not in payload, or is None/empty string
        user_id = f"generated_{uuid.uuid4()}" # Generate a new UUID
        print(f"INFO: No user_id in payload, generated new user_id: {user_id}")

    return {
        "user_id": user_id,
        "primary_goal": payload.get("primary_goal", ""),
        "specific_event": payload.get("specific_event", ""),
        "past_attempts": payload.get("past_attempts", ""),
        "schedule": payload.get("schedule", ""),
        "injuries": payload.get("injuries", ""),
        "raw_input_text": f"Primary Goal: {payload.get('primary_goal', '')}\n"
                          f"Specific Event: {payload.get('specific_event', '')}\n"
                          f"Past Attempts: {payload.get('past_attempts', '')}\n"
                          f"Schedule: {payload.get('schedule', '')}\n"
                          f"Injuries: {payload.get('injuries', '')}"
    }

def categorize_goals(user_input_text: str, chain_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Categorizes user goals using Langchain with ChatOpenAI and JsonOutputParser.
    """
    # System prompt for categorization
    # Note: The prompt itself needs to instruct the LLM to output valid JSON
    # that matches the CategorizedTags schema.
    categorization_prompt_text = f"""
You are a fitness goal classification expert. Based on the user's input, please assign between 0 and 5 relevant tags from the provided list.
User Input:
{user_input_text}

Available Tags:
{', '.join(CLASSIFICATION_TAGS)}

Instructions:
- Only assign a tag if it's clearly supported by the user's input.
- You MUST output a JSON object with a single key "tags". The value of "tags" should be a list of strings.
- Each string in the list must be one of the available tags.
- If no tags are applicable, the value of "tags" should be an empty list or null.

Example for applicable tags:
{{{{"tags": ["Lose weight", "Training for an event"]}}}}

Example for no applicable tags:
{{{{"tags": []}}}}

User input to classify: "{user_input_text}"
Output JSON:
"""
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", categorization_prompt_text),
        # No explicit user message needed here as user_input_text is embedded in system prompt
        # Or, could be: ("human", "User input to classify: {user_input_text}") if user_input_text is an input variable to the chain
    ])

    # Initialize JsonOutputParser with our Pydantic model
    # The parser will ensure the LLM output conforms to the CategorizedTags schema.
    parser = JsonOutputParser(pydantic_object=CategorizedTags)

    # Create the categorization chain using LCEL
    # llm is the ChatOpenAI instance initialized globally with temperature=0.2
    categorization_chain = prompt_template | llm | parser

    try:
        invocation_config = {"metadata": chain_metadata} if chain_metadata else {}
        response_data = categorization_chain.invoke({}, config=invocation_config)
        # response_data will be an instance of CategorizedTags Pydantic model (or a dict if parser is simple JsonOutputParser)
        
        # Extract tags, ensuring they are valid
        assigned_tags = []
        # Check if response_data is a dict (from JsonOutputParser) or Pydantic model
        if isinstance(response_data, dict) and response_data.get('tags') is not None:
            raw_tags = response_data['tags']
            assigned_tags = [tag for tag in raw_tags if tag in CLASSIFICATION_TAGS]
        elif hasattr(response_data, 'tags') and response_data.tags is not None: # Check if it has a tags attribute (like a Pydantic model)
            raw_tags = response_data.tags
            assigned_tags = [tag for tag in raw_tags if tag in CLASSIFICATION_TAGS]
            
        return assigned_tags
    except Exception as e:
        print(f"Error during Langchain goal categorization: {e}")
        return [] # Return empty list on error

def retrieve_from_pinecone(user_raw_context: str, classified_tags: List[str]) -> Dict[str, Any]:
    """
    Retrieves context from Pinecone using Langchain's PineconeVectorStore.
    Returns combined_context (str) and similar_chunk_pct (float).
    """
    if not pinecone_vectorstore:
        print("WARNING: PineconeVectorStore not initialized. Skipping RAG.")
        return {"combined_context": "", "similar_chunk_pct": 0.0, "retrieved_documents": []}

    all_retrieved_documents_with_scores: List[Tuple[Any, float]] = [] # Store (Document, score) tuples

    # If no tags, do a general search without filter
    if not classified_tags:
        try:
            results_with_scores = pinecone_vectorstore.similarity_search_with_score(
                query=user_raw_context,
                k=5 
            )
            all_retrieved_documents_with_scores.extend(results_with_scores)
            print(f"INFO: Pinecone general search returned {len(results_with_scores)} matches with scores.")
        except Exception as e:
            print(f"Error during Pinecone general similarity search with scores: {e}")
    else:
        for tag in classified_tags:
            pinecone_source_value = TAG_TO_PINECONE_SOURCE_VALUE.get(tag)
            if not pinecone_source_value:
                print(f"WARNING: No Pinecone source value mapping for tag: '{tag}'. Skipping.")
                continue
            try:
                results_with_scores = pinecone_vectorstore.similarity_search_with_score(
                    query=user_raw_context,
                    k=3,
                    filter={"source": pinecone_source_value}
                )
                all_retrieved_documents_with_scores.extend(results_with_scores)
                print(f"INFO: Pinecone query for source='{pinecone_source_value}' (tag: '{tag}') got {len(results_with_scores)} matches with scores.")
            except Exception as e:
                print(f"Error querying Pinecone for tag '{tag}' (source: '{pinecone_source_value}') with scores: {e}")

    # Process all collected documents
    retrieved_chunks_for_context_str = []
    relevant_chunk_count = 0
    unique_documents = [] # To handle duplicates from multiple tag queries
    seen_doc_ids_or_content = set() # Use page_content as a proxy for ID if no unique ID in metadata

    for doc, score in all_retrieved_documents_with_scores:
        # Deduplication: Langchain Documents might not have a unique ID field by default.
        # We use page_content as a simple way to deduplicate for now.
        # A more robust way would be to ensure your Pinecone metadata has a unique chunk_id.
        if doc.page_content not in seen_doc_ids_or_content:
            unique_documents.append((doc, score))
            seen_doc_ids_or_content.add(doc.page_content)

    for doc, score in unique_documents:
        source_info = doc.metadata.get('source', 'Unknown') 
        text_chunk = doc.page_content
        # Langchain Pinecone typically returns cosine similarity (0 to 1, higher is better)
        # If it were distance (lower is better), we'd need to invert/normalize.
        # Let's assume score is similarity for now.
        print(f"  Retrieved doc from source '{source_info}' with score: {score:.4f}") 
        if score >= 0.5:
            relevant_chunk_count += 1
        if text_chunk:
            retrieved_chunks_for_context_str.append(f"[Source: {source_info}, Score: {score:.2f}] {text_chunk}")
        else:
            print(f"WARNING: Doc from source '{source_info}' has empty page_content. Metadata: {doc.metadata}")
    
    total_retrieved_chunks = len(unique_documents)
    similar_chunk_pct = (relevant_chunk_count / total_retrieved_chunks) if total_retrieved_chunks > 0 else 0.0

    combined_context = "\n".join(retrieved_chunks_for_context_str)
    if len(combined_context) > MAX_RAG_CONTEXT_LENGTH:
        if combined_context:
            combined_context = combined_context[:MAX_RAG_CONTEXT_LENGTH] + "..."
        else:
            combined_context = ""
    
    # For Step 2 (Context Relevance Eval), we also need the raw Document objects
    raw_documents_for_eval = [doc for doc, score in unique_documents]

    return {
        "combined_context": combined_context, 
        "similar_chunk_pct": similar_chunk_pct,
        "retrieved_documents_for_eval": raw_documents_for_eval, # Pass this for evaluation step
        "retrieved_documents_with_scores_for_eval": unique_documents # Pass this for the new missing_context logic
    }

# @traceable(name="Generate Recommendation LLM", run_type="llm") # Removed @traceable
def generate_recommendation(user_input_data: Dict[str, Any], rag_context: str, chain_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Generates a personalized fitness recommendation using Langchain with ChatOpenAI.
    """
    
    # Construct the prompt parts clearly
    user_info_lines = [
        "User Information:",
        f"- Primary Goal: {user_input_data.get('primary_goal', 'N/A')}",
        f"- Specific Event: {user_input_data.get('specific_event', 'N/A')}",
        f"- Past Attempts: {user_input_data.get('past_attempts', 'N/A')}",
        f"- Schedule: {user_input_data.get('schedule', 'N/A')}",
        f"- Injuries/Limitations: {user_input_data.get('injuries', 'N/A')}",
    ]
    user_info_str = "\n".join(user_info_lines)

    rag_context_str = rag_context if rag_context else "No specific context retrieved. Provide general advice if possible."

    system_prompt_template_text = """
You are a helpful AI fitness advisor.
Based on the user's information and the provided context, generate a personalized fitness recommendation.
Be direct, encouraging, and practical.

{user_information}

Retrieved Context from Knowledge Base:
{retrieved_rag_context}

Output Format Instruction:
Provide a personalized fitness recommendation. If the provided context is insufficient to address a specific aspect of the user's query, clearly state that.
Safeguard: Use ONLY the provided context and user information when generating the recommendation. If something is not covered, say so.
Recommendation:
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template_text)
        # Optionally, a human message could be added if the prompt expects a direct user question after the system setup
        # ("human", "Please provide my personalized fitness recommendation based on the above.")
    ])

    # Output parser - we expect a string output for the recommendation
    output_parser = StrOutputParser()

    # Create the recommendation chain using LCEL
    # recommendation_llm is the ChatOpenAI instance initialized globally with temperature=0.7
    recommendation_chain = prompt_template | recommendation_llm | output_parser

    # The system_prompt_template_text IS the prompt for logging purposes BEFORE variable substitution
    # For actual invocation, we provide the variables to be substituted.
    
    try:
        invocation_config = {"metadata": chain_metadata} if chain_metadata else {}
        recommendation = recommendation_chain.invoke(
            {
                "user_information": user_info_str,
                "retrieved_rag_context": rag_context_str
            },
            config=invocation_config
        )
        
        # For logging, we want the prompt as it was *sent* to the LLM (after formatting)
        # We can get this by formatting the prompt template manually with the same inputs
        formatted_prompt_for_log = prompt_template.format_prompt(
            user_information=user_info_str,
            retrieved_rag_context=rag_context_str
        ).to_string() # Get as a single string

        return {"recommendation": recommendation.strip(), "system_prompt": formatted_prompt_for_log}
    except Exception as e:
        print(f"Error during Langchain recommendation generation: {e}")
        error_response_text = "I'm sorry, I encountered an error while generating your recommendation. Please try again later."
        # Attempt to format the prompt for logging even on error
        try:
            formatted_prompt_for_log_on_error = prompt_template.format_prompt(
                user_information=user_info_str,
                retrieved_rag_context=rag_context_str
            ).to_string()
        except Exception:
            formatted_prompt_for_log_on_error = system_prompt_template_text # Fallback to template text
        return {"recommendation": error_response_text, "system_prompt": formatted_prompt_for_log_on_error}

def evaluate_custom_context_relevance(user_request: str, rag_context: str, llm_client: ChatOpenAI) -> Optional[int]:
    """
    Evaluates context relevance using a custom prompt with GPT-4o.
    Scores:
    1: Context cannot directly address the user's request.
    2: Context can partially address the user's request.
    3: Context can directly address the user's request.
    Returns an integer score (1, 2, or 3) or None if evaluation fails.
    """
    if not rag_context: # If no context, it cannot address the request.
        print("INFO (Custom Context Relevance): No RAG context provided. Defaulting score to 1.")
        return 1
    if not user_request.strip():
        print("INFO (Custom Context Relevance): No user request provided. Skipping evaluation.")
        return None


    prompt_text = f"""
You are an expert relevance assessment AI. Your task is to evaluate if the provided "Retrieved Context" is relevant for answering the "User Request".
Based on this, assign a score from 1 to 3 according to the following criteria:

1: The "Retrieved Context" CANNOT directly address the "User Request". It is irrelevant or completely misses the point of the request.
2: The "Retrieved Context" can PARTIALLY address the "User Request". It contains some useful information but may not be comprehensive or fully aligned.
3: The "Retrieved Context" can DIRECTLY address the "User Request". It is highly relevant and provides sufficient information to formulate a good answer.

User Request:
---
{user_request}
---

Retrieved Context:
---
{rag_context}
---

Carefully analyze the User Request and the Retrieved Context.
Respond ONLY with the integer score: 1, 2, or 3. Do not provide any other text or explanation.
Score:"""

    try:
        messages = [("system", prompt_text)]
        response = llm_client.invoke(messages)
        content = response.content.strip()
        
        # Attempt to parse the integer score
        score = int(content)
        if 1 <= score <= 3:
            print(f"INFO (Custom Context Relevance): LLM returned score: {score}")
            return score
        else:
            print(f"WARNING (Custom Context Relevance): LLM returned out-of-range score: {content}. Defaulting to None.")
            return None
    except ValueError:
        print(f"WARNING (Custom Context Relevance): LLM did not return a valid integer. Response: '{content}'. Defaulting to None.")
        return None
    except Exception as e:
        print(f"ERROR (Custom Context Relevance): Could not evaluate context relevance: {e}")
        return None

def evaluate_custom_hallucination(user_request: str, recommendation: str, rag_context: str, llm_client: ChatOpenAI) -> Optional[int]:
    """
    Evaluates if the recommendation is grounded in the RAG context, using a custom prompt with GPT-4o.
    Scores:
    1: Recommendation IS supported by the provided "Retrieved Context" (or context is not provided, making it trivially true).
    0: Recommendation makes statements NOT supported by the "Retrieved Context".
    Returns an integer score (0 or 1) or None if evaluation fails.
    """
    if not recommendation.strip():
        print("INFO (Custom Hallucination): No recommendation provided. Skipping evaluation.")
        return None
    
    if not rag_context.strip():
        print("INFO (Custom Hallucination): No RAG context provided. Recommendation is trivially considered faithful (score 1).")
        return 1 # If no context, can't be unfaithful to it.
    
    if not user_request.strip():
        print("INFO (Custom Hallucination): No user request provided. Skipping evaluation as it might be needed for full assessment.")
        return None # Or decide if it can proceed without user_request

    prompt_text = f"""
You are an expert fact-checking AI. Your task is to determine if the "Generated Recommendation" is factually supported by the "Retrieved Context", considering the original "User Request".

User Request:
---
{user_request}
---

Retrieved Context:
---
{rag_context}
---

Generated Recommendation:
---
{recommendation}
---

Instructions:
- Read the "User Request", "Retrieved Context", and "Generated Recommendation" carefully.
- Determine if all factual claims made in the "Generated Recommendation" that are relevant to the "User Request" can be directly verified from the "Retrieved Context".
- If the "Generated Recommendation" includes advice or statements not present in the "Retrieved Context" (and not inferable as common knowledge given the context), it should be flagged.
- Your answer must be a single integer: 1 or 0.
  - 1: The "Generated Recommendation" IS supported by the "Retrieved Context". All key information in the recommendation that relates to the user's request is present in or directly inferable from the context.
  - 0: The "Generated Recommendation" makes one or more statements or gives advice NOT supported by (i.e., cannot be found in) the "Retrieved Context".

Respond ONLY with the integer score: 1 or 0. Do not provide any other text or explanation.
Score:"""

    try:
        messages = [("system", prompt_text)]
        response = llm_client.invoke(messages)
        content = response.content.strip()
        
        score = int(content)
        if score == 0 or score == 1:
            print(f"INFO (Custom Hallucination): LLM returned score: {score}")
            return score
        else:
            print(f"WARNING (Custom Hallucination): LLM returned out-of-range score: {content}. Defaulting to None.")
            return None
    except ValueError:
        print(f"WARNING (Custom Hallucination): LLM did not return a valid integer. Response: '{content}'. Defaulting to None.")
        return None
    except Exception as e:
        print(f"ERROR (Custom Hallucination): Could not evaluate hallucination: {e}")
        return None

def check_custom_out_of_scope(user_input_data: Dict[str, Any], llm_client: ChatOpenAI) -> bool:
    """
    Checks if the user's inputs (primary_goal, etc.) seem out of scope for a fitness quiz/advisor 
    using a custom prompt with GPT-4o.
    Returns True if out of scope, False otherwise.
    """
    
    input_summary_lines = [
        f"- Primary Goal: {user_input_data.get('primary_goal', 'N/A')}",
        f"- Specific Event: {user_input_data.get('specific_event', 'N/A')}",
        f"- Past Attempts: {user_input_data.get('past_attempts', 'N/A')}",
        f"- Schedule: {user_input_data.get('schedule', 'N/A')}",
        f"- Injuries/Limitations: {user_input_data.get('injuries', 'N/A')}",
    ]
    input_summary = "\n".join(input_summary_lines)

    if not input_summary.strip() or all(val in ('N/A', '') for val in user_input_data.values()):
        print("INFO (Custom Out-of-Scope): User input data is empty or all N/A. Considering it in scope by default (not actively out of scope).")
        return False # Not actively out of scope if empty

    prompt_text = f"""
You are an AI fitness assistant responsible for validating user input for a fitness and wellness advice service.
Your task is to determine if the provided "User Quiz Answers" are genuinely related to fitness, health, exercise, or wellness, OR if they are clearly out of scope (e.g., nonsensical, jokes, requests for unrelated topics like flying to the moon, financial advice, etc.).

User Quiz Answers:
---
{input_summary}
---

Consider the following:
- Fitness goals typically involve physical improvement (lose weight, gain muscle, run a race), pain reduction, injury recovery, or general well-being.
- Even unusual fitness goals (e.g., "train for an eating competition") might be considered in scope if they relate to physical preparation, though they are edge cases.
- Clear out-of-scope examples: "fly to the moon", "become a wizard", "tell me a joke", "what is the stock market doing?", primary goals that are just random words or gibberish.

Is the overall set of User Quiz Answers, particularly the "Primary Goal", clearly OUT OF SCOPE for a fitness and wellness advisor?

Respond ONLY with "YES" or "NO". Do not provide any other text or explanation.
OUT OF SCOPE?"""

    try:
        messages = [("system", prompt_text)]
        response = llm_client.invoke(messages)
        content = response.content.strip().upper()
        
        if content == "YES":
            print(f"INFO (Custom Out-of-Scope): LLM determined input is OUT OF SCOPE.")
            return True
        elif content == "NO":
            print(f"INFO (Custom Out-of-Scope): LLM determined input is IN SCOPE.")
            return False
        else:
            print(f"WARNING (Custom Out-of-Scope): LLM returned an unexpected response: '{response.content}'. Defaulting to False (in scope).")
            return False # Default to in-scope if response is ambiguous
    except Exception as e:
        print(f"ERROR (Custom Out-of-Scope): Could not evaluate out-of-scope: {e}. Defaulting to False (in scope).")
        return False # Default to in-scope on error

def log_to_supabase(data: Dict[str, Any]):
    """
    Logs a dictionary of data to the Supabase 'chatbot_runs' table.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("ERROR: Supabase URL or Service Key not configured. Skipping log to Supabase.")
        return

    url = f"{SUPABASE_URL}/rest/v1/chatbot_runs"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" # Optional: To get the inserted row back, remove if not needed or causing issues
    }

    # Ensure all expected keys are present, defaulting to None if missing from `data`
    payload = {
        "user_id": data.get("user_id"),
        "input_text": data.get("input_text"),
        "tags": data.get("tags"), 
        "retrieved_chunks": data.get("retrieved_chunks"),
        "recommendation": data.get("recommendation"),
        "similar_chunk_pct": data.get("similar_chunk_pct"),
        "missing_context": data.get("missing_context"),
        # New custom evaluation scores
        "is_out_of_scope": data.get("is_out_of_scope"), # Renamed from is_out_of_scope_flag
        "custom_context_relevance_score": data.get("custom_context_relevance_score"),
        "custom_hallucination_score": data.get("custom_hallucination_score")
        # Removed old Langchain evaluator scores: context_coverage_score, hallucination_score, hallucination_feedback
    }

    try:
        # Using httpx for synchronous POST request within Lambda
        # For async environments, you might use httpx.AsyncClient()
        # Lambda handler is synchronous, so this is fine.
        with httpx.Client() as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            print(f"Successfully logged data to Supabase for user_id: {data.get('user_id')}. Response: {response.json()}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error logging to Supabase: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"Request error logging to Supabase: {e}")
    except Exception as e:
        print(f"Unexpected error logging to Supabase: {e}")

def lambda_handler(event, context):
    """
    Main AWS Lambda handler function.
    """
    try:
        # Assuming event['body'] contains the JSON string from the webhook
        if isinstance(event.get('body'), str):
            payload = json.loads(event['body'])
        else:
            payload = event.get('body', event) # If body is already a dict (e.g. direct test invoke) or just event
        
        if not payload:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing payload'})
            }

    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON in request body'})
        }
    except Exception as e:
        print(f"Error processing request: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }

    # Step 1: Parse webhook payload
    user_input_data = parse_webhook_payload(payload)
    user_raw_text_for_categorization = user_input_data["raw_input_text"]
    user_id = user_input_data.get("user_id", "unknown_user_at_handler") # Get user_id for metadata
    input_text_for_logging = user_input_data["raw_input_text"] # For Supabase logging

    # --- Custom Out-of-Scope Check ---
    is_out_of_scope_flag = check_custom_out_of_scope(user_input_data, eval_llm)
    print(f"Custom Out-of-Scope Check result: {is_out_of_scope_flag}")

    # If out of scope, we might want to return early or handle differently
    if is_out_of_scope_flag:
        print("INFO: Request flagged as OUT OF SCOPE. Bypassing main processing flow.")
        classified_tags = []
        rag_context_retrieved = "N/A - Request was out of scope."
        similar_chunk_pct = 0.0
        retrieved_documents_for_eval = [] # Ensure it's an empty list
        retrieved_documents_with_scores_for_eval = [] # Ensure it's an empty list
        recommendation = "Your request appears to be outside the scope of fitness and wellness advice. Please try again with a fitness-related goal."
        recommendation_system_prompt = "N/A - Request was out of scope."
        custom_context_relevance_score_val = None # Or 1, as context is trivially irrelevant/not applicable
        custom_hallucination_score_val = None # Or 1, as recommendation is generic, not based on context
        missing_context_flag = True # Context is effectively missing for an out-of-scope request

        # Directly prepare for logging and return
        supabase_log_data = {
            "user_id": user_id,
            "input_text": input_text_for_logging,
            "tags": classified_tags,
            "retrieved_chunks": rag_context_retrieved,
            "recommendation": recommendation,
            "similar_chunk_pct": similar_chunk_pct,
            "missing_context": missing_context_flag,
            "is_out_of_scope": is_out_of_scope_flag,
            "custom_context_relevance_score": custom_context_relevance_score_val,
            "custom_hallucination_score": custom_hallucination_score_val
        }
        log_to_supabase(supabase_log_data)

        lambda_response_body = {
            "tags": classified_tags,
            "recommendation": recommendation,
            "retrieved_rag_context": rag_context_retrieved,
            "recommendation_system_prompt": recommendation_system_prompt,
            "missing_context_flag": missing_context_flag,
            "is_out_of_scope_flag": is_out_of_scope_flag,
            "custom_context_relevance_score": custom_context_relevance_score_val,
            "custom_hallucination_score": custom_hallucination_score_val
        }
        return {
            'statusCode': 200, # Still a 200 as the request was processed, just identified as OOS
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(lambda_response_body)
        }

    # Metadata for Langsmith traces
    trace_metadata = {
        "user_id": user_id,
        "raw_payload_preview": json.dumps(payload)[:250] + "..." if len(json.dumps(payload)) > 250 else json.dumps(payload) # Log a preview
    }

    # Step 1a: Categorize goals
    classified_tags = categorize_goals(
        user_raw_text_for_categorization,
        chain_metadata=trace_metadata # Pass metadata to the function
    )
    print(f"Classified tags: {classified_tags}")

    # Step 1b: Retrieve from Pinecone (RAG)
    rag_output = retrieve_from_pinecone(user_raw_text_for_categorization, classified_tags)
    rag_context_retrieved = rag_output["combined_context"]
    similar_chunk_pct = rag_output["similar_chunk_pct"]
    retrieved_documents_for_eval = rag_output["retrieved_documents_for_eval"]
    retrieved_documents_with_scores_for_eval = rag_output["retrieved_documents_with_scores_for_eval"]

    print(f"Retrieved RAG context (first 100 chars): {rag_context_retrieved[:100]}...")
    print(f"Similar chunk pct (>=0.5 score): {similar_chunk_pct:.2f}")

    # Initialize evaluation scores and feedback
    custom_context_relevance_score_val: Optional[int] = None
    custom_hallucination_score_val: Optional[int] = None

    # Step 2: Add LangChain evaluation: context relevance (coverage) -> CUSTOM
    # This needs to be done BEFORE generation
    if retrieved_documents_for_eval: 
        # OLD Langchain Evaluator - REMOVE/COMMENT OUT
        # try:
        #     context_coverage_evaluator = load_evaluator("context_qa", llm=eval_llm)
        #     eval_result_coverage = context_coverage_evaluator.evaluate_strings(
        #         input=user_raw_text_for_categorization, 
        #         prediction=rag_context_retrieved, 
        #         reference=rag_context_retrieved 
        #     )
        #     if isinstance(eval_result_coverage, dict):
        #         context_coverage_score_val = eval_result_coverage.get('score')
        #     else: 
        #         context_coverage_score_val = eval_result_coverage 
        #     print(f"Context Coverage Score (from ContextQAEvalChain): {context_coverage_score_val}")
        # except Exception as e:
        #     print(f"Error during context coverage evaluation (ContextQAEvalChain): {e}")
        
        # --- NEW Custom Context Relevance Evaluation ---
        custom_context_relevance_score_val = evaluate_custom_context_relevance(
            user_request=user_raw_text_for_categorization, # Using the more detailed input for this eval
            rag_context=rag_context_retrieved,
            llm_client=eval_llm
        )
        print(f"Custom Context Relevance Score: {custom_context_relevance_score_val}")
    else:
        print("Skipping context coverage/relevance evaluation as no documents were retrieved.")
        if not rag_context_retrieved: # Explicitly set score to 1 if no context as per function logic
            custom_context_relevance_score_val = 1 
            print(f"Setting Custom Context Relevance Score to 1 as no RAG context was retrieved.")

    # Step 4: Add missing context flag logic (updated)
    if not retrieved_documents_with_scores_for_eval: # No documents retrieved at all
        missing_context_flag = True 
        # We might also consider if classified_tags exist here, 
        # i.e., missing_context_flag = bool(classified_tags and not retrieved_documents_with_scores_for_eval)
        # For now, simplifying to: if no docs, context is missing.
    else:
        # Check if all retrieved documents have a score < 0.4
        all_scores_below_threshold = all(score < 0.4 for doc, score in retrieved_documents_with_scores_for_eval)
        if all_scores_below_threshold:
            missing_context_flag = True
            print("INFO: All retrieved documents have similarity scores < 0.4. Setting missing_context_flag to True.")
        else:
            missing_context_flag = False
            
    print(f"Missing context flag: {missing_context_flag}")

    # Step 2: Generate recommendation
    generation_output = generate_recommendation(
        user_input_data, 
        rag_context_retrieved,
        chain_metadata=trace_metadata # Pass metadata to the function
    )
    recommendation = generation_output["recommendation"]
    recommendation_system_prompt = generation_output["system_prompt"]

    print(f"Generated recommendation (first 100 chars): {recommendation[:100]}...")
    print(f"Recommendation prompt (first 100 chars): {recommendation_system_prompt[:100]}...")

    # Step 3: Add LangChain evaluation: hallucination detection (faithfulness)
    # This needs to be done AFTER generation

    # Construct the structured user_info_str for QAEvalChain input, similar to generate_recommendation
    # This is also used for custom_context_relevance and custom_hallucination
    user_info_lines_for_eval = [
        "User Information:",
        f"- Primary Goal: {user_input_data.get('primary_goal', 'N/A')}",
        f"- Specific Event: {user_input_data.get('specific_event', 'N/A')}",
        f"- Past Attempts: {user_input_data.get('past_attempts', 'N/A')}",
        f"- Schedule: {user_input_data.get('schedule', 'N/A')}",
        f"- Injuries/Limitations: {user_input_data.get('injuries', 'N/A')}",
    ]
    structured_user_input_for_eval = "\n".join(user_info_lines_for_eval)

    if rag_context_retrieved and recommendation and structured_user_input_for_eval:
        # OLD Langchain Evaluator - REMOVE/COMMENT OUT
        # try:
        #     qa_evaluator = load_evaluator("qa", llm=eval_llm) 
        #     eval_result_hallucination = qa_evaluator.evaluate_strings(
        #         input=structured_user_input_for_eval, 
        #         prediction=recommendation, 
        #         reference=rag_context_retrieved 
        #     )
        #     if isinstance(eval_result_hallucination, dict):
        #         result_text = eval_result_hallucination.get('results', eval_result_hallucination.get('score', ''))
        #         hallucination_feedback_val = eval_result_hallucination.get('reasoning', str(eval_result_hallucination))
        #     else: 
        #         result_text = str(eval_result_hallucination)
        #         hallucination_feedback_val = str(eval_result_hallucination)
        #     if isinstance(result_text, str) and 'correct' in result_text.lower():
        #         hallucination_score_val = 1.0
        #     elif isinstance(result_text, str) and 'incorrect' in result_text.lower():
        #         hallucination_score_val = 0.0
        #     elif isinstance(result_text, (int, float)):
        #         hallucination_score_val = float(result_text) 
        #     else:
        #         hallucination_score_val = None 
        #     print(f"Hallucination/Faithfulness Score (from QAEvalChain): {hallucination_score_val}")
        #     if hallucination_feedback_val:
        #         print(f"Hallucination Feedback: {hallucination_feedback_val[:200]}...")
        # except Exception as e:
        #     print(f"Error during hallucination/faithfulness evaluation (QAEvalChain): {e}")

        # --- NEW Custom Hallucination Evaluation ---
        custom_hallucination_score_val = evaluate_custom_hallucination(
            user_request=structured_user_input_for_eval,
            recommendation=recommendation,
            rag_context=rag_context_retrieved,
            llm_client=eval_llm
        )
        print(f"Custom Hallucination Score: {custom_hallucination_score_val}")
    else:
        print("Skipping hallucination/faithfulness evaluation due to missing RAG context, recommendation, or user input.")
        if not rag_context_retrieved: # Explicitly set score to 1 if no context, as per function logic
            custom_hallucination_score_val = 1
            print(f"Setting Custom Hallucination Score to 1 as no RAG context was available.")

    # Prepare data for Supabase logging - ensuring all keys match what log_to_supabase expects
    supabase_log_data = {
        "user_id": user_id, 
        "input_text": input_text_for_logging, 
        "tags": classified_tags,
        "retrieved_chunks": rag_context_retrieved, 
        "recommendation": recommendation,
        "similar_chunk_pct": similar_chunk_pct, 
        # "context_coverage_score": context_coverage_score_val, # OLD
        # "hallucination_score": hallucination_score_val, # OLD
        # "hallucination_feedback": hallucination_feedback_val, # OLD
        "missing_context": missing_context_flag, 
        # New custom scores
        "is_out_of_scope": is_out_of_scope_flag,
        "custom_context_relevance_score": custom_context_relevance_score_val,
        "custom_hallucination_score": custom_hallucination_score_val
    }

    # Call log_to_supabase before returning the response
    log_to_supabase(supabase_log_data)

    # Prepare response body for the Lambda return
    # This can be a subset of supabase_log_data or include other things like recommendation_system_prompt
    # For now, let's keep it consistent with previous structure plus new eval scores
    lambda_response_body = {
        "tags": classified_tags,
        "recommendation": recommendation,
        "retrieved_rag_context": rag_context_retrieved, 
        "recommendation_system_prompt": recommendation_system_prompt,
        # "context_coverage_score": context_coverage_score_val, # OLD
        # "hallucination_score": hallucination_score_val, # OLD
        # "hallucination_feedback": hallucination_feedback_val, # OLD
        "missing_context_flag": missing_context_flag,
        # New custom scores for lambda response
        "is_out_of_scope_flag": is_out_of_scope_flag,
        "custom_context_relevance_score": custom_context_relevance_score_val,
        "custom_hallucination_score": custom_hallucination_score_val
    }

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*' # Adjust CORS as needed
        },
        'body': json.dumps(lambda_response_body)
    }

# Example usage (for local testing)
if __name__ == '__main__':
    print("\n--- Local Test Execution ---")

    # Perform checks to ensure essential configurations are not placeholders
    # The script would have already exited if keys were completely unset due to `raise ValueError` above.
    # These checks are more about ensuring they aren't dummy values for the local test run.
    
    valid_config_for_test = True
    if "your_openai_api_key" in OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is a placeholder. Update .env or environment variables.")
        valid_config_for_test = False
    
    if "your_pinecone_api_key" in PINECONE_API_KEY:
        print("ERROR: PINECONE_API_KEY is a placeholder. Update .env or environment variables.")
        valid_config_for_test = False

    if "your_pinecone_index_name" in PINECONE_INDEX_NAME:
        print("ERROR: PINECONE_INDEX_NAME is a placeholder. Update .env or environment variables.")
        valid_config_for_test = False
        
    if not pinecone_vectorstore:
        print("WARNING: PineconeVectorStore ('pinecone_vectorstore') was not successfully initialized globally. Pinecone queries in the test will likely fail or be skipped.")
        # Depending on test requirements, you might set valid_config_for_test = False here too.

    if valid_config_for_test:
        print("INFO: Configuration appears suitable for local test.")
        mock_event = {
            "body": json.dumps({
                "primary_goal": "I want to lose about 10 pounds for my friend's wedding in 3 months.",
                "specific_event": "Friend's wedding",
                "past_attempts": "Tried keto, but it was too restrictive. Also tried running but got shin splints.",
                "schedule": "Can work out 3-4 times a week, mostly evenings. Weekends are flexible.",
                "injuries": "Had shin splints in the past from running. Occasional knee discomfort."
            })
        }
        
        print("\n--- Calling lambda_handler with mock event ---")
        response = lambda_handler(mock_event, None)
        print("\n--- lambda_handler Response ---")
        print(json.dumps(response, indent=2))
    else:
        print("\nSkipping lambda_handler execution due to configuration issues (placeholder values found).")

    print("\n--- End of Local Test ---") 