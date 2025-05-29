# Fitness Quiz Intelligence API

## Overview

The Fitness Quiz Intelligence API is a Python-based backend system designed to integrate with a Landbot chatbot. It processes user responses from a fitness quiz, leverages AI for analysis and personalized recommendations, and logs detailed information for monitoring and improvement. The core functionality is deployed as an AWS Lambda function, with data and metrics tracked in Supabase and Langsmith, and visualized through a Streamlit dashboard.

## Core Features

*   **Webhook Processing**: Parses JSON payloads from Landbot containing user quiz answers (primary goal, specific event, past attempts, schedule, injuries).
*   **User ID Management**: Generates a unique `user_id` if one is not provided in the payload.
*   **AI-Powered Goal Categorization**: Uses Langchain with OpenAI (GPT-4o) to classify user inputs into predefined fitness tags (e.g., "Lose weight," "Gain weight," "Training for an event").
*   **Retrieval Augmented Generation (RAG)**:
    *   Embeds user's raw quiz context using OpenAI's `text-embedding-3-small`.
    *   Queries a Pinecone vector index to retrieve relevant documents, filtering by categorized tags.
    *   Deduplicates retrieved documents and calculates similarity scores.
*   **Personalized Recommendation Generation**: Employs Langchain with OpenAI (GPT-4o) to generate fitness recommendations based on user information and RAG context.
*   **Custom LLM-Based Evaluations**:
    *   **Out-of-Scope Check**: Determines if user quiz answers are outside the scope of fitness/wellness advice.
    *   **Context Relevance Evaluation**: Scores how well the RAG context addresses the user's request (1-3 scale).
    *   **Hallucination/Faithfulness Evaluation**: Scores if the recommendation is factually supported by the RAG context (0 or 1 scale).
*   **Comprehensive Logging**: Logs all processed data, AI outputs, and evaluation scores to a Supabase `chatbot_runs` table using `httpx`.
*   **Langsmith Tracing**: Integrates with Langsmith for end-to-end tracing and observability of Langchain components.
*   **Streamlit Dashboard**: Provides a visual interface to monitor key metrics and trends from the Supabase logs.

## Project Structure

```
.env
lambda_function.py
test_lambda_locally.py
dashboard.py
requirements.txt
README.md
```

*   `lambda_function.py`: Contains the core AWS Lambda handler and all business logic for processing quiz data, AI interactions, evaluations, and Supabase logging.
*   `test_lambda_locally.py`: A script for generating test cases and running the `lambda_function.py` locally to simulate AWS Lambda invocations.
*   `dashboard.py`: The Streamlit application for visualizing data from the Supabase `chatbot_runs` table.
*   `requirements.txt`: Lists all Python dependencies for the project.
*   `.env`: (To be created by user) Stores sensitive API keys and configuration variables.
*   `README.md`: This file.

## Key Technologies Used

*   **Programming Language**: Python 3.x
*   **Cloud Platform**: AWS Lambda (for deployment)
*   **AI/LLM**: OpenAI (GPT-4o for categorization, recommendations, and evaluations)
*   **LLM Orchestration**: Langchain (Chains, Prompts, Parsers, OpenAI & Pinecone integrations)
*   **Vector Database**: Pinecone (for RAG)
*   **Data Logging/Storage**: Supabase (PostgreSQL)
*   **Observability/Tracing**: Langsmith
*   **Dashboarding**: Streamlit
*   **Data Validation**: Pydantic
*   **HTTP Client**: HTTPLibX (for Supabase communication)
*   **Environment Management**: `python-dotenv`

## Setup and Installation

**1. Prerequisites:**

*   Python 3.8 or higher.
*   `pip` (Python package installer).

**2. Clone the Repository (Example):**

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**3. Create and Populate `.env` File:**

   Create a file named `.env` in the root directory of the project. Populate it with your actual API keys and service URLs:

   ```env
   OPENAI_API_KEY="sk-your-openai-api-key"
   PINECONE_API_KEY="your-pinecone-api-key"
   PINECONE_INDEX_NAME="your-pinecone-index-name"
   # PINECONE_ENVIRONMENT="your-pinecone-environment" # Often not needed if index name is unique globally or client handles it
   LANGCHAIN_API_KEY="ls__your-langchain-api-key" # For Langsmith
   LANGCHAIN_TRACING_V2="true"
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_PROJECT="Your-Langsmith-Project-Name" # Optional: define a project in Langsmith

   SUPABASE_URL="https://your-project-ref.supabase.co"
   SUPABASE_SERVICE_KEY="your-supabase-service-role-key"

   # Optional: Specify a different embedding model if needed
   # OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
   ```

   **Important**: 
   *   Ensure `PINECONE_INDEX_NAME` matches your existing and populated Pinecone index.
   *   For Langsmith, ensure `LANGCHAIN_API_KEY` is set and `LANGCHAIN_TRACING_V2` is "true".

**4. Install Dependencies:**

   Navigate to the project's root directory in your terminal and run:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

**1. Locally Testing the Lambda Function:**

   The `test_lambda_locally.py` script allows you to test the `lambda_handler` function from `lambda_function.py` as if it were invoked by an AWS Lambda event. It includes a sample event payload.

   To run the local test:

   ```bash
   python test_lambda_locally.py
   ```

   The script will print the response from the `lambda_handler`, including categorized tags, the generated recommendation, and evaluation scores. It also logs this information to your configured Supabase table.

**2. Running the Streamlit Dashboard:**

   The `dashboard.py` script launches a Streamlit web application to visualize metrics logged to Supabase.

   To run the dashboard:

   ```bash
   streamlit run dashboard.py
   ```

   This will typically open the dashboard in your default web browser (e.g., at `http://localhost:8501`).

## Deployment

   The primary component for deployment is `lambda_function.py`.

   **AWS Lambda:**

1.  Package your Lambda function along with its dependencies. This typically involves creating a ZIP file containing `lambda_function.py` and all installed packages from `requirements.txt`.
2.  Create a new Lambda function in the AWS Management Console.
    *   Choose the Python runtime (e.g., Python 3.9 or as appropriate).
    *   Upload your ZIP file.
    *   Configure the handler to `lambda_function.lambda_handler`.
    *   **Crucially, set all the environment variables** (from your `.env` file) in the Lambda function's configuration settings (Configuration > Environment variables). This includes `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `LANGCHAIN_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, etc.
    *   Set an appropriate timeout and memory allocation for the Lambda function.
3.  Configure a trigger for your Lambda function. This would typically be an API Gateway endpoint that Landbot can send its webhook POST requests to.

## Supabase Logging

The API logs detailed information about each quiz processed to the `chatbot_runs` table in your Supabase database. This allows for ongoing monitoring, evaluation, and data analysis.

**Key Logged Metrics:**

*   `user_id`: Identifier for the user.
*   `input_text`: The raw input from the user's quiz answers.
*   `tags`: JSON array of classified fitness goals.
*   `retrieved_chunks`: The context retrieved from Pinecone via RAG.
*   `recommendation`: The AI-generated fitness advice.
*   `similar_chunk_pct`: Percentage of retrieved RAG chunks deemed relevant (score >= 0.5).
*   `missing_context`: Flag indicating if RAG context was missing or all retrieved chunks had low similarity scores.
*   `is_out_of_scope`: Boolean flag from the custom out-of-scope LLM evaluation.
*   `custom_context_relevance_score`: Score (1-3) indicating how relevant the RAG context was to the user's query.
*   `custom_hallucination_score`: Score (0-1) indicating if the recommendation was faithful to the RAG context.

## Evaluation and Monitoring

*   **Custom LLM Evaluations**: The API uses GPT-4o to perform nuanced checks for out-of-scope requests, context relevance, and factual grounding (hallucination) of recommendations. These scores are logged to Supabase.
*   **Langsmith**: All Langchain-based operations (categorization, RAG, recommendation generation) are automatically traced to Langsmith, providing deep insights into LLM calls, prompt structures, and performance.
*   **Streamlit Dashboard**: The `dashboard.py` application provides a user-friendly interface to visualize trends in the logged metrics, such as the distribution of tags, relevance scores, hallucination rates, and out-of-scope percentages. 
