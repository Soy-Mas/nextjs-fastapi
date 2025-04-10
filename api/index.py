from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, UUID4
from typing import List, Optional, Dict, Any, Union
import numpy as np
import os
from dotenv import load_dotenv
import httpx
import json
import uuid
from supabase import create_client, Client
import asyncio
from openai import OpenAI

# --- Configuration Loading ---
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- NEW: Embedding Dimension Configuration ---
# Fetch from environment or use default. Ensure this matches your pgvector index setup (e.g., vector(2000)).
try:
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 2000))
except ValueError:
    print("Warning: Invalid EMBEDDING_DIMENSION in .env, using default 2000.")
    EMBEDDING_DIMENSION = 2000

# Validate essential variables
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL y SUPABASE_KEY deben estar definidos en el archivo .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY debe estar definido en el archivo .env")
if EMBEDDING_DIMENSION <= 0:
     raise ValueError("EMBEDDING_DIMENSION debe ser un entero positivo.")

# --- FastAPI App Initialization ---
# Define the app instance ONCE with all configurations
app = FastAPI(
    title="API de Recomendaciones Personalizadas con Supabase",
    description="API para obtener recomendaciones de contenido personalizadas basadas en embeddings con pgvector",
    version="1.1.0", # Consider bumping version
    # These URLs are relative to the server root, Vercel handles the /api/py prefix automatically for them
    docs_url="/api/py/docs",
    openapi_url="/api/py/openapi.json"
)

# --- Middleware Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Client Initializations ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Pydantic Models (Keep as previously refined) ---
class RecommendationRequest(BaseModel):
    content_id: str # Expected to be external_id
    user_id: Optional[str] = None
    num_recommendations: int = 10
    alpha: float = 0.7
    exploration_ratio: float = 0.2

class RecommendationItem(BaseModel):
    content_id: str # external_id
    title: str
    slug: str
    similarity_score: float
    is_exploration: bool

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    user_id: Optional[str] = None
    source_content_id: Optional[str] = None # external_id

# Model for manual embedding creation/update requests via API (if needed, often done via batch)
class ContentEmbeddingRequest(BaseModel):
    content_id: str  # external_id
    title: str
    slug: str
    text: str

class UserEmbeddingRequest(BaseModel): # Potentially useful for direct update trigger
    user_id: str

class BatchUserEmbeddingRequest(BaseModel):
    limit: int = 100

class OperationResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# --- Supabase Dependency ---
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Utility & Logic Functions ---

# (Keep the improved `safe_rpc_execute` helper)
async def safe_rpc_execute(supabase: Client, query: str):
    try:
        # print(f"Executing RPC query: {query[:200]}...") # Debug logging
        result = supabase.rpc('execute_sql', {'query': query}).execute()
        if hasattr(result, 'error') and result.error:
             error_detail = f"Database query failed: {getattr(result.error, 'message', str(result.error))}"
             print(f"Supabase RPC Error: {error_detail}")
             # Consider specific error codes if needed
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_detail)
        return result
    except httpx.RequestError as e:
         print(f"Network error communicating with Supabase: {e}")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Could not connect to database service.")
    except Exception as e:
         print(f"Unexpected error during Supabase RPC call: {e}")
         # Log traceback here for debugging
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during database operation: {str(e)}")

# (Keep improved get_content_embedding - fetch by external_id)
async def get_content_embedding(supabase: Client, external_content_id: str):
    """
    Obtiene el embedding de un contenido específico desde Supabase por external_id.
    Returns the internal id_content (UUID) and the embedding list.
    """
    response = supabase.table('content_embeddings') \
        .select('id_content, external_id, title, slug, embedding') \
        .eq('external_id', external_content_id) \
        .limit(1) \
        .execute()

    if not response.data or len(response.data) == 0:
        return None

    data = response.data[0]
    # Ensure embedding is list of floats
    if isinstance(data.get('embedding'), str):
         try:
            data['embedding'] = json.loads(data['embedding'])
         except json.JSONDecodeError:
             print(f"Warning: Could not decode embedding string for content external_id {external_content_id}")
             data['embedding'] = None
    # Validate dimension (optional but good practice)
    if data.get('embedding') and len(data['embedding']) != EMBEDDING_DIMENSION:
        print(f"Warning: Mismatched embedding dimension for external_id {external_content_id}. DB: {len(data['embedding'])}, Expected: {EMBEDDING_DIMENSION}")
        # Decide handling: return None, try to use it, etc.
        # data['embedding'] = None # Or log and continue

    return data

# (Keep improved get_user_embedding)
async def get_user_embedding(supabase: Client, user_id: str):
    """
    Obtiene el embedding promedio de un usuario desde Supabase.
    """
    response = supabase.table('user_embeddings') \
        .select('id_user, avg_embedding, content_count') \
        .eq('id_user', user_id) \
        .limit(1) \
        .execute()

    if not response.data or len(response.data) == 0:
        return None

    data = response.data[0]
    # Ensure avg_embedding is list of floats
    if isinstance(data.get('avg_embedding'), str):
         try:
            data['avg_embedding'] = json.loads(data['avg_embedding'])
         except json.JSONDecodeError:
             print(f"Warning: Could not decode avg_embedding string for user {user_id}")
             data['avg_embedding'] = None
    # Validate dimension
    if data.get('avg_embedding') and len(data['avg_embedding']) != EMBEDDING_DIMENSION:
         print(f"Warning: Mismatched user avg_embedding dimension for user {user_id}. DB: {len(data['avg_embedding'])}, Expected: {EMBEDDING_DIMENSION}")
         # data['avg_embedding'] = None

    return data

# --- UPDATED: generate_embedding with Truncation ---
def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Genera un embedding utilizando el modelo de OpenAI y lo trunca/recorta
    a la dimensión configurada (EMBEDDING_DIMENSION).
    """
    if not text:
        print("Warning: generate_embedding called with empty text.")
        return None
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large", # Or use env var
            input=text,
            encoding_format="float",
            # Specify dimensions if the model supports it directly (check OpenAI docs for text-embedding-3-large)
            # dimensions=EMBEDDING_DIMENSION # Add this if supported and preferred over truncation
        )
        if response.data and len(response.data) > 0:
            full_embedding = response.data[0].embedding

            # --- Truncation/Clipping Logic ---
            if len(full_embedding) >= EMBEDDING_DIMENSION:
                embedding = full_embedding[:EMBEDDING_DIMENSION]
            else:
                # Handle cases where the source embedding is shorter than target dimension
                print(f"Warning: OpenAI returned embedding shorter ({len(full_embedding)}) than target dimension ({EMBEDDING_DIMENSION}). Padding with zeros.")
                embedding = full_embedding + [0.0] * (EMBEDDING_DIMENSION - len(full_embedding))

            # Final type/format check
            if isinstance(embedding, list) and len(embedding) == EMBEDDING_DIMENSION and all(isinstance(x, float) for x in embedding):
                return embedding
            else:
                print(f"Error: Embedding format issue after processing. Got length {len(embedding)}, expected {EMBEDDING_DIMENSION}. Type: {type(embedding)}")
                return None
        else:
            print("Warning: No embedding data received from OpenAI.")
            return None
    except Exception as e:
        print(f"Error generating embedding from OpenAI: {e}")
        # Optionally re-raise as HTTPException or return None
        # raise HTTPException(...)
        return None # Return None to allow batch processing to potentially continue

# (Keep improved recommendation functions - they will use truncated embeddings transparently)
async def get_recommendations_with_pgvector(
    supabase: Client,
    external_content_id: str, # Expecting external_id from request
    user_id: Optional[str] = None,
    num_recommendations: int = 10,
    alpha: float = 0.7,
    exploration_ratio: float = 0.2
):
    # 1. Get source content embedding (using external_id)
    content = await get_content_embedding(supabase, external_content_id)
    if not content or not content.get('embedding'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contenido fuente con external_id {external_content_id} no encontrado o sin embedding válido."
        )
    source_embedding_list = content['embedding']
    source_embedding_str = f"[{','.join(map(str, source_embedding_list))}]"
    source_internal_id = content['id_content'] # Use internal UUID for DB queries

    # 2. Calculate counts
    core_count = max(1, int(num_recommendations * (1 - exploration_ratio)))
    exploration_count = max(0, num_recommendations - core_count)

    recommendations = []
    seen_in_core_external_ids = set() # Store external_ids

    # 3. Get CORE recommendations (content similarity)
    # Query uses internal ID for exclusion, returns external ID for API response
    content_recommendations_query = f"""
    SELECT
        ce.external_id as content_id,
        ce.title,
        ce.slug,
        1 - (ce.embedding <=> '{source_embedding_str}'::vector({EMBEDDING_DIMENSION})) as similarity_score
    FROM content_embeddings ce
    WHERE ce.id_content != '{source_internal_id}'::uuid
      AND ce.embedding IS NOT NULL
      -- Optional: Add dimension check if needed: AND vector_dims(ce.embedding) = {EMBEDDING_DIMENSION}
    ORDER BY ce.embedding <=> '{source_embedding_str}'::vector({EMBEDDING_DIMENSION})
    LIMIT {core_count};
    """
    core_results = await safe_rpc_execute(supabase, content_recommendations_query)

    if core_results.data:
        for item in core_results.data:
            recommendations.append({
                "content_id": item['content_id'], # external_id
                "title": item['title'],
                "slug": item['slug'],
                "similarity_score": item['similarity_score'],
                "is_exploration": False
            })
            seen_in_core_external_ids.add(item['content_id'])

    # 4. Get user embedding if user_id is provided
    user_embedding_list = None
    user_embedding_str = None
    if user_id:
        user = await get_user_embedding(supabase, user_id)
        if user and user.get('avg_embedding'):
            user_embedding_list = user['avg_embedding']
            user_embedding_str = f"[{','.join(map(str, user_embedding_list))}]"

    # 5. WEIGHT core recommendations if user embedding exists (using asyncio.gather)
    if user_embedding_str:
        weighted_core_recommendations = []
        pref_tasks = []
        for item in recommendations: # Use initial core list based on content similarity
            # Query uses external_id to find the item, calculates similarity to user embedding
            user_pref_query = f"""
            SELECT 1 - (embedding <=> '{user_embedding_str}'::vector({EMBEDDING_DIMENSION})) as user_preference
            FROM content_embeddings
            WHERE external_id = '{item['content_id'].replace("'", "''")}' AND embedding IS NOT NULL LIMIT 1;
            """
            pref_tasks.append(safe_rpc_execute(supabase, user_pref_query))

        preference_results = await asyncio.gather(*pref_tasks, return_exceptions=True)

        for i, item in enumerate(recommendations):
            user_preference = 0.0
            result = preference_results[i]
            if isinstance(result, Exception):
                 print(f"Error calculating user preference for {item['content_id']}: {result}")
            elif result.data:
                user_preference = result.data[0].get('user_preference', 0.0)

            combined_score = alpha * item['similarity_score'] + (1 - alpha) * user_preference
            item['similarity_score'] = combined_score # Update score
            weighted_core_recommendations.append(item)
        recommendations = weighted_core_recommendations # Replace with weighted list

    # 6. Get EXPLORATION recommendations (based on user preference, excluding core items)
    if exploration_count > 0 and user_embedding_str:
        exclude_ids_formatted = [f"'{cid.replace("'", "''")}'" for cid in seen_in_core_external_ids]
        exclusion_clause = ""
        if exclude_ids_formatted:
            # Exclude based on external_id
            exclusion_clause = f"AND ce.external_id NOT IN ({','.join(exclude_ids_formatted)})"

        exploration_query = f"""
        SELECT
            ce.external_id as content_id,
            ce.title,
            ce.slug,
            1 - (ce.embedding <=> '{user_embedding_str}'::vector({EMBEDDING_DIMENSION})) as similarity_score
        FROM content_embeddings ce
        WHERE ce.id_content != '{source_internal_id}'::uuid -- Exclude source by internal id
          {exclusion_clause} -- Exclude core results by external id
          AND ce.embedding IS NOT NULL
        ORDER BY ce.embedding <=> '{user_embedding_str}'::vector({EMBEDDING_DIMENSION})
        LIMIT {exploration_count};
        """
        exploration_results = await safe_rpc_execute(supabase, exploration_query)

        if exploration_results.data:
            for item in exploration_results.data:
                 if item['content_id'] not in seen_in_core_external_ids: # Double check
                     recommendations.append({
                         "content_id": item['content_id'],
                         "title": item['title'],
                         "slug": item['slug'],
                         "similarity_score": item['similarity_score'],
                         "is_exploration": True
                     })

    # 7. Sort final list and limit
    recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
    final_recommendations = recommendations[:num_recommendations]

    return {
        "recommendations": final_recommendations,
        "user_id": user_id,
        "source_content_id": external_content_id # Return the external_id used in request
    }


async def get_user_recommendations_from_embedding(
    supabase: Client,
    user_id: str,
    num_recommendations: int = 10
):
    # 1. Get user embedding
    user = await get_user_embedding(supabase, user_id)
    if not user or not user.get('avg_embedding'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario con ID {user_id} no encontrado o sin embedding promedio válido."
        )
    user_embedding_list = user['avg_embedding']
    user_embedding_str = f"[{','.join(map(str, user_embedding_list))}]"

    # 2. Get seen content history (assuming history stores internal id_content)
    # Ensure user_id is treated as UUID if necessary
    safe_user_id = user_id.replace("'", "''")
    history_query = f"""
    SELECT h.id_content FROM history h WHERE h.id_user = '{safe_user_id}'::uuid;
    """
    history_result = await safe_rpc_execute(supabase, history_query)
    seen_content_internal_ids = {item['id_content'] for item in history_result.data} if history_result.data else set()

    # 3. Build exclusion clause (using internal id_content)
    exclusion_clause = ""
    if seen_content_internal_ids:
        formatted_ids = [f"'{id}'::uuid" for id in seen_content_internal_ids]
        exclusion_clause = f"AND ce.id_content NOT IN ({','.join(formatted_ids)})"

    # 4. Get recommendations query (returning external_id)
    recommendations_query = f"""
    SELECT
        ce.external_id as content_id,
        ce.title,
        ce.slug,
        1 - (ce.embedding <=> '{user_embedding_str}'::vector({EMBEDDING_DIMENSION})) as similarity_score
    FROM content_embeddings ce
    WHERE ce.embedding IS NOT NULL
      {exclusion_clause}
    ORDER BY ce.embedding <=> '{user_embedding_str}'::vector({EMBEDDING_DIMENSION})
    LIMIT {num_recommendations};
    """
    recommendations_result = await safe_rpc_execute(supabase, recommendations_query)

    recommendations = []
    if recommendations_result.data:
        for item in recommendations_result.data:
            recommendations.append({
                "content_id": item['content_id'], # external_id
                "title": item['title'],
                "slug": item['slug'],
                "similarity_score": item['similarity_score'],
                "is_exploration": False
            })

    return {
        "recommendations": recommendations,
        "user_id": user_id,
        "source_content_id": None # No single source content here
    }

# (Keep improved batch processing functions - they use the updated generate_embedding)
async def process_content_from_posts(supabase: Client, external_content_id: Optional[str] = None, limit: int = 50):
    """
    Processes content from 'posts' table (identified by posts.id_content as external_id),
    generates truncated embeddings, and upserts into 'content_embeddings'.
    Uses asyncio.gather for concurrency. Returns count and list of failed external_ids.
    """
    query_filter = ""
    current_limit = limit
    if external_content_id:
        safe_external_id = external_content_id.replace("'", "''")
        # Assume posts.id_content is the external identifier
        query_filter = f"WHERE p.id_content = '{safe_external_id}'"
        current_limit = 1
    else:
        query_filter = "WHERE p.concatenated_text IS NOT NULL AND p.concatenated_text != ''"

    # Fetch posts assuming posts.id_content maps to content_embeddings.external_id
    posts_query = f"""
    SELECT p.id_content as external_id, p.title, p.slug, p.concatenated_text
    FROM posts p
    {query_filter}
    ORDER BY p.created_at DESC
    LIMIT {current_limit};
    """
    posts_result = await safe_rpc_execute(supabase, posts_query)
    processed_count = 0
    failed_ids = []

    if not posts_result.data:
        print("No posts found matching criteria for embedding processing.")
        return processed_count, failed_ids

    tasks = []

    async def process_single_post(post_info):
        ext_id = post_info['external_id']
        txt = post_info['text_to_embed']
        ttl = post_info['title']
        slg = post_info['slug']
        try:
            print(f"Processing post external_id: {ext_id}")
            # --- Uses the updated generate_embedding ---
            embedding_list = generate_embedding(txt)
            if embedding_list is None:
                print(f"Embedding generation failed for {ext_id}.")
                return ext_id, False

            # Ensure embedding is correct dimension before creating string
            if len(embedding_list) != EMBEDDING_DIMENSION:
                 print(f"Error: Final embedding dimension mismatch for {ext_id}. Got {len(embedding_list)}, expected {EMBEDDING_DIMENSION}")
                 return ext_id, False

            embedding_str = f"[{','.join(map(str, embedding_list))}]"
            safe_text = txt.replace("'", "''") # Basic sanitize

            # Upsert based on external_id (requires UNIQUE constraint on external_id)
            upsert_query = f"""
            INSERT INTO content_embeddings (id_content, external_id, title, slug, concatenated_text, embedding, created_at, updated_at)
            VALUES (gen_random_uuid(), '{ext_id}', '{ttl}', '{slg}', '{safe_text}', '{embedding_str}'::vector({EMBEDDING_DIMENSION}), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (external_id)
            DO UPDATE SET title = EXCLUDED.title, slug = EXCLUDED.slug, concatenated_text = EXCLUDED.concatenated_text,
                         embedding = EXCLUDED.embedding, updated_at = CURRENT_TIMESTAMP;
            """
            await safe_rpc_execute(supabase, upsert_query)
            print(f"Success: Upserted embedding for external_id: {ext_id}")
            return ext_id, True
        except HTTPException as e:
             # Catch DB/Network errors from safe_rpc_execute
             print(f"HTTP error during DB upsert for post {ext_id}: {e.detail}")
             return ext_id, False
        except Exception as e:
            # Catch unexpected errors during this post's processing
            print(f"Unexpected error processing post {ext_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return ext_id, False

    for post in posts_result.data:
        external_id = post.get('external_id')
        text_to_embed = post.get('concatenated_text')
        title = post.get('title', '').replace("'", "''")
        slug = post.get('slug', '').replace("'", "''")

        if not external_id or not text_to_embed:
            print(f"Skipping post due to missing external_id or text: {post}")
            failed_ids.append(external_id or "UNKNOWN_ID")
            continue

        task_info = {'external_id': external_id, 'text_to_embed': text_to_embed, 'title': title, 'slug': slug}
        tasks.append(process_single_post(task_info))

    results = await asyncio.gather(*tasks)
    for external_id, success in results:
        if success:
            processed_count += 1
        else:
            failed_ids.append(external_id)

    print(f"Content Batch Finished. Processed: {processed_count}, Failed: {len(failed_ids)}. Failed IDs: {failed_ids}")
    return processed_count, failed_ids


async def update_user_embedding_from_history(supabase: Client, user_id: str):
    """
    Calls the SQL function 'update_user_embedding' to recalculate and store
    the user's average embedding (which should also be truncated).
    Returns True on success, False on failure.
    """
    safe_user_id = user_id.replace("'", "''")
    # Ensure the SQL function `update_user_embedding` correctly handles
    # fetching content embeddings, averaging them, TRUNCATING the average,
    # and storing it in user_embeddings. The dimension needs to be consistent.
    query = f"SELECT update_user_embedding('{safe_user_id}'::uuid);"
    try:
        result = await safe_rpc_execute(supabase, query)
        # Check return value of your specific SQL function
        if result.data and len(result.data) > 0 and result.data[0].get('update_user_embedding') is True:
            print(f"Success: SQL function updated embedding for user {user_id}")
            return True
        else:
            print(f"Warning: SQL function 'update_user_embedding' did not indicate success for user {user_id}. Result: {result.data}")
            return False
    except Exception as e:
        # Handles exceptions from safe_rpc_execute or the function call itself
        print(f"Error calling/executing update_user_embedding function for user {user_id}: {e}")
        return False


async def process_batch_user_embeddings(supabase: Client, limit: int = 100):
    """
    Finds users needing embedding updates and calls update_user_embedding_from_history
    concurrently using asyncio.gather. Returns count and list of failed user_ids.
    """
    # Query to find users needing update (example)
    query = f"""
    SELECT DISTINCT h.id_user
    FROM history h
    LEFT JOIN user_embeddings ue ON h.id_user = ue.id_user
    WHERE ue.id_user IS NULL
       OR ue.updated_at < (SELECT MAX(hist.created_at) FROM history hist WHERE hist.id_user = h.id_user)
       -- Optional: Add check for dimension mismatch if storing dimension in user_embeddings
       -- OR vector_dims(ue.avg_embedding) != {EMBEDDING_DIMENSION}
    LIMIT {limit};
    """
    print("Finding users for batch embedding update...")
    users_result = await safe_rpc_execute(supabase, query)
    processed_count = 0
    failed_ids = []

    if not users_result.data:
        print("No users found requiring embedding updates.")
        return processed_count, failed_ids

    user_ids_to_process = [user['id_user'] for user in users_result.data]
    print(f"Found {len(user_ids_to_process)} users for update. Processing...")

    tasks = [update_user_embedding_from_history(supabase, user_id) for user_id in user_ids_to_process]
    results = await asyncio.gather(*tasks) # Run updates concurrently

    for i, success in enumerate(results):
        user_id = user_ids_to_process[i]
        if success:
            processed_count += 1
        else:
            failed_ids.append(user_id)

    print(f"User Batch Finished. Updated: {processed_count}, Failed: {len(failed_ids)}. Failed IDs: {failed_ids}")
    return processed_count, failed_ids


# --- API Endpoints (with /api/py/ prefix) ---

@app.get("/api/py/helloFastApi")
def hello_fast_api():
    # Simple test endpoint
    return {"message": f"Hello from FastAPI! Using Embedding Dim: {EMBEDDING_DIMENSION}"}

@app.get("/api/py/")
async def root():
    return {"message": "API de Recomendaciones con pgvector - Root"}

# Use RecommendationRequest which expects external_id for content_id
@app.post("/api/py/recommendations/", response_model=RecommendationResponse)
async def get_recommendations_endpoint( # Renamed for clarity
    request: RecommendationRequest,
    supabase: Client = Depends(get_supabase)
):
    """
    Obtiene recomendaciones basadas en un contenido fuente (por external_id),
    opcionalmente personalizadas para un usuario.
    """
    try:
        recommendations_data = await get_recommendations_with_pgvector(
            supabase,
            request.content_id, # Pass external_id
            request.user_id,
            request.num_recommendations,
            request.alpha,
            request.exploration_ratio
        )
        # Ensure the response model matches the returned data structure
        # The function already returns the correct structure
        return recommendations_data
    except HTTPException as e:
        raise e # Re-raise validation errors, 404s etc.
    except Exception as e:
        print(f"Error in /api/py/recommendations/ endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error obteniendo recomendaciones: {str(e)}")

@app.get("/api/py/user-recommendations/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations_endpoint( # Renamed for clarity
    user_id: str,
    num_recommendations: int = 10,
    supabase: Client = Depends(get_supabase)
):
    """
    Obtiene recomendaciones generales para un usuario basado en su embedding promedio.
    """
    if num_recommendations <= 0:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="num_recommendations must be positive")
    try:
        recommendations_data = await get_user_recommendations_from_embedding(
            supabase,
            user_id,
            num_recommendations
        )
        return recommendations_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in /api/py/user-recommendations/{user_id} endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error obteniendo recomendaciones para usuario: {str(e)}")

# Endpoint to process/update embedding for a SINGLE content item by external_id
@app.post("/api/py/content-embeddings/process-one/{external_content_id}", response_model=OperationResponse)
async def process_single_content_endpoint( # Renamed for clarity
    external_content_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Busca un post por su ID (externo), (re)genera su embedding truncado,
    y lo guarda/actualiza en content_embeddings.
    """
    try:
        # Calls the improved function that handles upsert and error reporting
        processed_count, failed_ids = await process_content_from_posts(supabase, external_content_id=external_content_id, limit=1)
        if processed_count > 0:
            return OperationResponse(success=True, message=f"Embedding para '{external_content_id}' procesado con éxito.", data={"external_id": external_content_id})
        else:
             # Provide more specific feedback if possible
             reason = "Contenido no encontrado en 'posts', texto vacío, o error durante el procesamiento (ver logs)."
             return OperationResponse(success=False, message=f"Fallo al procesar embedding para '{external_content_id}'. {reason}", data={"external_id": external_content_id, "failed_ids": failed_ids})
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error en endpoint /api/py/content-embeddings/process-one/{external_content_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error procesando embedding: {str(e)}")

# Endpoint to trigger update for a SINGLE user embedding
@app.post("/api/py/user-embeddings/update-one/{user_id}", response_model=OperationResponse)
async def update_single_user_endpoint( # Renamed for clarity
    user_id: str,
    supabase: Client = Depends(get_supabase)
):
    """
    Dispara la actualización del embedding promedio de un usuario específico
    llamando a la función SQL 'update_user_embedding'.
    """
    try:
        success = await update_user_embedding_from_history(supabase, user_id)
        if success:
            return OperationResponse(success=True, message=f"Actualización de embedding para usuario '{user_id}' iniciada con éxito (vía SQL function).", data={"user_id": user_id})
        else:
            # Failure could be due to SQL function error, no history, etc.
            return OperationResponse(success=False, message=f"Fallo al actualizar embedding para usuario '{user_id}'. Verifique logs y la función SQL 'update_user_embedding'.", data={"user_id": user_id})
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error en endpoint /api/py/user-embeddings/update-one/{user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error actualizando embedding de usuario: {str(e)}")

# Batch processing endpoints remain largely the same, relying on the updated core functions
@app.post("/api/py/batch/content-embeddings", response_model=OperationResponse)
async def batch_process_content_endpoint( # Renamed for clarity
    limit: int = 50,
    supabase: Client = Depends(get_supabase)
):
    """
    Procesa un lote de contenidos desde 'posts', generando embeddings truncados y guardándolos.
    """
    if limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be positive")
    try:
        processed_count, failed_ids = await process_content_from_posts(supabase, external_content_id=None, limit=limit)
        success = not failed_ids # Consider success if no items failed
        return OperationResponse(
            success=success,
            message=f"Lote de contenido: Procesados={processed_count}, Fallidos={len(failed_ids)}.",
            data={"processed_count": processed_count, "failed_count": len(failed_ids), "requested_limit": limit, "failed_ids": failed_ids}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error en el endpoint /api/py/batch/content-embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error en procesamiento por lotes de contenidos: {str(e)}")

@app.post("/api/py/batch/user-embeddings", response_model=OperationResponse)
async def batch_process_users_endpoint( # Renamed for clarity
    request: BatchUserEmbeddingRequest,
    supabase: Client = Depends(get_supabase)
):
    """
    Procesa un lote de usuarios, actualizando sus embeddings promedio (vía SQL function).
    """
    if request.limit <= 0:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be positive")
    try:
        processed_count, failed_ids = await process_batch_user_embeddings(supabase, request.limit)
        success = not failed_ids
        return OperationResponse(
            success=success,
            message=f"Lote de usuarios: Actualizados={processed_count}, Fallidos={len(failed_ids)}.",
            data={"processed_count": processed_count, "failed_count": len(failed_ids), "requested_limit": request.limit, "failed_ids": failed_ids}
        )
    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Error en el endpoint /api/py/batch/user-embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error en procesamiento por lotes de usuarios: {str(e)}")

# (Keep refined health check)
@app.get("/api/py/health")
async def health_check():
    """
    Punto de verificación de salud del API
    """
    supabase_healthy = False
    try:
        client = get_supabase()
        # Simple check: list tables or similar non-intensive operation
        # response = client.rpc('execute_sql', {'query': 'SELECT count(*) FROM pg_catalog.pg_tables;'}).execute()
        # if response.data:
        #    supabase_healthy = True
        if client: # Basic check
             supabase_healthy = True
    except Exception as e:
        print(f"Health check Supabase connection failed: {e}")
        supabase_healthy = False

    openai_healthy = bool(OPENAI_API_KEY)
    overall_status = "healthy" if supabase_healthy and openai_healthy else "unhealthy"

    return {
        "status": overall_status,
        "version": app.version,
        "embedding_dimension": EMBEDDING_DIMENSION, # Add dimension info
        "dependencies": {
             "supabase_connection": "ok" if supabase_healthy else "error",
             "openai_configured": "ok" if openai_healthy else "error"
        }
    }

# --- Main Execution Guard (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Uvicorn server for local development on port 8000...")
    print(f"Using Embedding Dimension: {EMBEDDING_DIMENSION}")
    # Use "index:app" to correctly reference the app instance for reload
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
