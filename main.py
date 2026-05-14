from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD CSV
df = pd.read_csv("Export7254358788563184402.csv")
df = df.fillna("")


# USE ONLY ISSUE TITLE COLUMN
# ==========================================================
# Excel Column A = "Issue title"
# Duplicate check will be performed only against this column
# ==========================================================
df["full_issue"] = df["Issue title"].astype(str)

# Remove empty titles
df = df[df["full_issue"].str.strip() != ""]

# Load AI model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings once during startup
embeddings = model.encode(
    df["full_issue"].tolist()
)

# Home API
@app.get("/")
def home():
    return {
        "message": "AI Duplicate API Running"
    }

# Duplicate Check API
@app.post("/check")
async def check_duplicate(
    text: str = Form(...)
):
    # Normalize input
    input_text = text.strip()
    normalized_input = input_text.lower()

    # ======================================================
    # STEP 1: EXACT MATCH CHECK
    # ======================================================
    for issue in df["full_issue"]:
        if issue.strip().lower() == normalized_input:
            return {
                "Duplicate": True,
                "MatchedIssue": issue,
                "Score": 1.0,
                "TopMatches": [
                    {
                        "Issue": issue,
                        "Score": 1.0
                    }
                ]
            }

    # ======================================================
    # STEP 2: AI SEMANTIC SIMILARITY CHECK
    # ======================================================
    new_embedding = model.encode([input_text])

    similarities = cosine_similarity(
        new_embedding,
        embeddings
    )[0]

    # Copy dataframe and attach similarity scores
    result_df = df.copy()
    result_df["Score"] = similarities

    # Sort by highest similarity
    result_df = result_df.sort_values(
        by="Score",
        ascending=False
    )

    # Top 10 similar issues
    top_results = result_df
#top_results = result_df[result_df["Score"] >= 0.30].head(100)

    # Threshold for duplicate detection
    threshold = 0.55

    # Best match
    best_score = float(top_results.iloc[0]["Score"])
    matched_issue = top_results.iloc[0]["full_issue"]

    # Duplicate decision
    is_duplicate = best_score >= threshold

    # Build TopMatches list
    matches = []

    for _, row in top_results.iterrows():
        matches.append({
            "Issue": row["full_issue"],
            "Score": round(float(row["Score"]), 4)
        })

    # Final response
    return {
        "Duplicate": is_duplicate,
        "MatchedIssue": matched_issue,
        "Score": best_score,
        "TopMatches": matches
    }