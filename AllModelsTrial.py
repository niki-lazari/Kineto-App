import os
import shutil
import glob
import subprocess
import sys
import importlib
import os, json, re, ast, pickle, warnings
import numpy as np
import pandas as pd
#from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

warnings.filterwarnings("ignore")

#---------------------------------------------------------------
# Navigate to the project folder
os.chdir('C:/Users/Niki/OneDrive/Desktop/Northwestern/Fall 2025 - Q7/MSDS 498 - Capstone/Application')
ARTIFACTS_DIR = ('C:/Users/Niki/OneDrive/Desktop/Northwestern/Fall 2025 - Q7/MSDS 498 - Capstone/Application/artifacts')

# ---------------------------------------------------------------------
# 3️⃣  Optionally copy your parser to the working directory
# ---------------------------------------------------------------------
parser_src = os.path.join(ARTIFACTS_DIR, "movie_query_parser.py")
parser_dst = os.path.join(os.getcwd(), "movie_query_parser.py")
if os.path.exists(parser_src) and not os.path.exists(parser_dst):
    shutil.copy(parser_src, parser_dst)
    print("✅ Copied movie_query_parser.py into current directory.")
else:
    print("ℹ️ movie_query_parser.py already present or missing in artifacts.")

# ---------------------------------------------------------------------
# 4️⃣  Verify copied files
# ---------------------------------------------------------------------
print("\n📂 Files in artifacts folder:")
for f in os.listdir(ARTIFACTS_DIR):
    print("  -", f)

# ---------------------------------------------------------------------
# 6️⃣  Optional: show versions
# ---------------------------------------------------------------------
print("\n📦 Installed package versions:")
subprocess.run(
    [sys.executable, "-m", "pip", "list"],
    text=True
)
# ============ MASTER: Imports & Paths ============

# ML & DL
# from surprise import SVD  # COMMENTED OUT - loading pickled model directly


ART = Path(r"C:/Users/Niki/OneDrive/Desktop/Northwestern/Fall 2025 - Q7/MSDS 498 - Capstone/Application/artifacts")
DATA = Path(".")  # this just means current folder
ART.mkdir(exist_ok=True)


print("✅ Imports OK")

# ============ Load CF Models ============
try:
    with open(ART/"svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)
        print("✅ SVD model loaded successfully")
except ModuleNotFoundError:
    print("⚠️ scikit-surprise not installed — skipping collaborative filtering model.")
    svd_model = None

# ============ Load NCF Model ============
try:
    from tensorflow import keras
    ncf_model = keras.models.load_model(ART / "ncf_model.keras")
    print("✅ NCF model loaded successfully")
except Exception as e:
    ncf_model = None
    print(f"⚠️ NCF model skipped (Libraries compatibility issue): {e}")

# ============ Load ID maps ============
with open(ART/"user_to_idx.json") as f:
    user_to_idx = json.load(f)
with open(ART/"movie_to_idx.json") as f:
    movie_to_idx = json.load(f)

print("✅ CF loaded (SVD only)",
      f"users={len(user_to_idx):,} movies={len(movie_to_idx):,}")

# ============ Load enriched embeddings & FAISS (robust keying) ============
import re

emb = np.load(ART/"movie_embeddings.npy")            # (N, 384)
emb_ids = np.load(ART/"movie_ids.npy", allow_pickle=True).tolist()
faiss_index = faiss.read_index(str(ART/"faiss_index.bin"))

def is_intable(x):
    try:
        int(str(x))
        return True
    except Exception:
        return False

num_total   = len(emb_ids)
num_numeric = sum(is_intable(x) for x in emb_ids)

# If most IDs are non-numeric, the file is keyed by TITLES (e.g., '!Women Art Revolution')
TITLE_MODE = num_numeric < 0.5 * num_total

def norm_title(s: str) -> str:
    s = (str(s) or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

if TITLE_MODE:
    # Key embeddings by normalized title
    emb_keys = [norm_title(x) for x in emb_ids]
    emb_id2idx = {k: i for i, k in enumerate(emb_keys)}
    EMB_KEY_TYPE = "title"
else:
    # Key embeddings by numeric ID
    emb_keys = [int(str(x)) for x in emb_ids]
    emb_id2idx = {k: i for i, k in enumerate(emb_keys)}
    EMB_KEY_TYPE = "numeric"

emb_key_set = set(emb_keys)

print("✅ Embeddings/FAISS loaded")
print(f"   shape={emb.shape}  ntotal={faiss_index.ntotal:,}")
print(f"   keying mode: {EMB_KEY_TYPE}  (numeric={num_numeric}/{num_total})")

# ============ Load TMDB, make title_norm, robust genres, weighted rating ============

import re, ast, numpy as np
tmdb = pd.read_parquet("tmdb_fully_enriched.parquet")
print(f"✅ TMDB loaded: {len(tmdb):,} rows")

# 1) title_norm (needed because embeddings are keyed by title)
def norm_title(s):
    s = (str(s) or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

tmdb["title_norm"] = tmdb["tmdb_title"].apply(norm_title)
print("🔑 title_norm ready")

# 2) robust genre_names from genre_ids (handles list/ndarray/string)
GENRE_MAPPING = {
    28:"Action",12:"Adventure",16:"Animation",35:"Comedy",80:"Crime",99:"Documentary",
    18:"Drama",10751:"Family",14:"Fantasy",36:"History",27:"Horror",10402:"Music",
    9648:"Mystery",10749:"Romance",878:"Science Fiction",10770:"TV Movie",53:"Thriller",
    10752:"War",37:"Western"
}

def to_genre_names(g):
    if isinstance(g, (list, np.ndarray)):
        ids = list(g)
    elif isinstance(g, str):
        try:
            parsed = ast.literal_eval(g)
            ids = list(parsed) if isinstance(parsed,(list,tuple,np.ndarray)) else []
        except Exception:
            ids = []
    else:
        ids = []
    out = []
    for x in ids:
        try:
            name = GENRE_MAPPING.get(int(x))
            if name: out.append(name)
        except Exception:
            pass
    return out

tmdb["genre_names"] = tmdb["genre_ids"].apply(to_genre_names)
print("🎭 genre_names non-empty:", (tmdb["genre_names"].str.len() > 0).sum())

# 3) IMDB-style Bayesian weighted rating (must exist before filtering)
if "weighted_rating" not in tmdb.columns:
    C = tmdb["vote_average"].mean()
    m = 1000  # min votes to trust
    def wr(row, C=C, m=m):
        v = float(row["vote_count"]) if pd.notna(row["vote_count"]) else 0.0
        R = float(row["vote_average"]) if pd.notna(row["vote_average"]) else C
        return (v/(v+m))*R + (m/(v+m))*C
    tmdb["weighted_rating"] = tmdb.apply(wr, axis=1)
print("⚖️ weighted_rating ready; sample:", tmdb["weighted_rating"].iloc[0])

# ============ Merge LDA themes + BERT sentiments onto TMDB (title_norm) ============
# 1) Themes
with open(ART/"movie_themes.pkl", "rb") as f:
    theme_art = pickle.load(f)

theme_df = pd.DataFrame({
    "rt_title": theme_art["movie_titles"],
    "lda_themes": theme_art["themes"]
})
theme_df["title_norm"] = theme_df["rt_title"].apply(norm_title)

before = tmdb["lda_themes"].notna().sum() if "lda_themes" in tmdb.columns else 0
tmdb = tmdb.merge(theme_df[["title_norm","lda_themes"]], on="title_norm", how="left")
after = tmdb["lda_themes"].notna().sum()
print(f"🎭 Themes merged by title_norm: {after:,} rows (was {before:,})")

# 2) Sentiments
movie_sentiments = pd.read_pickle(ART/"movie_sentiments.pkl")
emo_cols = [c for c in movie_sentiments.columns
            if c not in ["movie_title","rotten_tomatoes_link"]]
movie_sentiments["title_norm"] = movie_sentiments["movie_title"].apply(norm_title)

before_emo = tmdb["sentiment_happy"].notna().sum() if "sentiment_happy" in tmdb.columns else 0
tmdb = tmdb.merge(movie_sentiments[["title_norm"]+emo_cols], on="title_norm", how="left")
after_emo = tmdb["sentiment_happy"].notna().sum()
print(f"🧠 Sentiments merged by title_norm: {after_emo:,} rows (was {before_emo:,})")

# 3) Safety: ensure lists where expected
tmdb["lda_themes"] = tmdb["lda_themes"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

# 4) Preview one enriched row
sample_row = tmdb[tmdb["lda_themes"].map(len)>0].head(1)
if not sample_row.empty:
    r = sample_row.iloc[0]
    print(f"🔎 Sample: {r['tmdb_title']} ({r['year']}) | themes={r['lda_themes'][:3]} | WR={r['weighted_rating']:.2f}")
else:
    print("⚠️ No rows with lda_themes found after merge (title mismatch?)")

# Define paths
current_dir = os.getcwd()
target_dir = current_dir
parser_file = "movie_query_parser.py"

# Copy parser file
try:
    shutil.copy(parser_file, target_dir)
    print(f"✅ Copied {parser_file} to {target_dir}")
except FileNotFoundError:
    print(f"⚠️ {parser_file} not found in {current_dir}")
except Exception as e:
    print(f"⚠️ Error copying file: {e}")

# List the result for verification
if os.path.exists(os.path.join(target_dir, parser_file)):
    print(f"\n📂 Verified: {os.path.join(target_dir, parser_file)} exists")
else:
    print(f"❌ {parser_file} not found at {target_dir}")

# Make sure /content is in Python path
import sys
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

print("✅ File copied and path updated")

# ============ Parser + Encoder + Helpers (title-key aware) ============


# 1) Load the production parser
import sys, importlib
if "." not in sys.path: sys.path.insert(0, ".")

# Direct import without reload
from movie_query_parser import parse_query, filter_by_metadata

print("✅ Parser loaded")

# 2) Sentence encoder for query vectors and title seeding
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ SentenceTransformer ready")

# 3) Because your embeddings are KEYED BY TITLE, keep a normalized title key
#    emb_keys was created in Cell 3; it's the list of normalized titles
#    Build quick lookup structures (if not already present)
try:
    emb_keys
except NameError:
    # If you restarted, rebuild from emb_ids using the same normalization
    import re
    def norm_title(s):
        s = (str(s) or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return " ".join(s.split())
    emb_keys = [norm_title(x) for x in emb_ids]

key2idx = {k: i for i, k in enumerate(emb_keys)}
key_set  = set(key2idx.keys())
print(f"🔑 Embedding key mode: title  | keys={len(key_set):,}")

# 4) Title seeding (for 'like/similar to <title>')
import re
def seed_from_title(query, k=2000):
    m = re.search(r"(?:like|similar to)\s+([A-Za-z0-9: '&!-]+)", query, flags=re.I)
    if not m:
        return None
    t = m.group(1).strip()
    vec = encoder.encode([t]).astype(np.float32)
    D, I = faiss_index.search(vec, min(k, faiss_index.ntotal))
    return set(emb_keys[i] for i in I[0] if i != -1)

# 5) Intent → theme tag steering
THEME_SYNONYMS = {
    "inspiring": {"prestige_drama","biographical_drama"},
    "family": {"holiday_family","general_comedy_drama","family_friendly"},
    "family-friendly": {"holiday_family","general_comedy_drama","family_friendly"},
    "dark": {"dark_psychological","crime_mystery","horror"},
    "gritty": {"dark_psychological","crime_mystery"},
    "romantic": {"romantic_drama"},
    "horror": {"horror"},
    "documentary": {"documentary_biopic"},
    "arthouse": {"foreign_arthouse"},
}

def query_theme_set(query: str):
    q = query.lower(); out=set()
    for kw, labs in THEME_SYNONYMS.items():
        if kw in q: out |= labs
    return out

def tag_overlap_score(tags, q_themes:set):
    tags = set(map(str, (tags or [])))
    if not q_themes or not tags: return 0.7
    inter = len(tags & q_themes); uni = len(tags | q_themes)
    return 0.6 + 0.4 * (inter / max(1, uni))  # 0.6–1.0 band

print("🧩 Helpers ready")

# ============ CompleteRecommender (title-key mode) ============

class CompleteRecommender:
    def __init__(self, tmdb_df, svd, ncf, user_map, movie_map, faiss_idx, emb, key2idx, key_set):
        self.df = tmdb_df
        self.svd = svd
        self.ncf = ncf
        self.user_to_idx = user_map
        self.movie_to_idx = movie_map
        self.faiss = faiss_idx
        self.emb = emb
        self.key2idx = key2idx      # normalized-title -> row index in emb
        self.key_set = key_set

    @staticmethod
    def _content(qv, key, key2idx, emb):
        idx = key2idx.get(key)
        if idx is None:
            return 3.0
        sim = float(np.dot(qv[0], emb[idx]))  # cosine on L2-normalized vectors
        return float(np.clip(1 + 4*((sim+1)/2), 1, 5))  # [1..5]

def recommend(self, query, user_id=None, n=10):
    """
    Returns Top-N recommendations using content + FAISS + optional CF.
    CF gracefully skipped if Surprise or NCF models are missing.
    """

    # --- 1. Parse query ---
    try:
        filters = parse_query_safe(query)
    except Exception:
        filters = parse_query(query)

    cand = filter_by_metadata(self.df, filters)
    if len(cand) == 0:
        return []

    # --- 2. Keep only embeddings present ---
    cand["__key__"] = cand["title_norm"].astype(str)
    cand = cand[cand["__key__"].isin(self.key_set)].copy()
    if len(cand) == 0:
        return []

    # --- 3. Optional title seeding ---
    seed_pool = seed_from_title(query, k=2000)
    if seed_pool:
        cand = cand[cand["__key__"].isin(seed_pool)]
        if len(cand) == 0:
            return []

    # --- 4. FAISS shortlist ---
    qv = encoder.encode([query]).astype(np.float32)
    D, I = self.faiss.search(qv, min(3000, self.faiss.ntotal))
    faiss_keys = set(emb_keys[i] for i in I[0] if i != -1)
    cand = cand[cand["__key__"].isin(faiss_keys)].head(500).copy()
    if len(cand) == 0:
        return []

    # --- 5. Tag steering ---
    q_themes = query_theme_set(query)
    cand["tag_score"] = cand.apply(
        lambda r: tag_overlap_score(r.get("lda_themes"), q_themes), axis=1
    )

    # --- 6. Compute scores ---
    out = []
    for _, row in cand.iterrows():
        key = row["__key__"]
        content = self._content(qv, key, self.key2idx, self.emb) / 5.0

        # --- Safe CF (only if model loaded) ---
        cf = 0.5
        if self.svd is not None and user_id is not None:
            try:
                cf_pred = self.svd.predict(str(user_id), str(key)).est
                cf = np.clip(cf_pred / 5.0, 0.2, 1.0)
            except Exception:
                cf = 0.5

        tags = float(row.get("tag_score", 0.7))
        quality = float(row.get("weighted_rating", 5.0)) / 10.0
        score = 0.40 * content + 0.40 * cf + 0.10 * tags + 0.10 * quality

        genres = row.get("genre_names") or []
        if isinstance(genres, str):
            try:
                genres = ast.literal_eval(genres)
            except Exception:
                genres = [genres]

        out.append({
            "title": row.get("tmdb_title") or row.get("title"),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": genres if isinstance(genres, list) else [],
            "themes": row.get("lda_themes") or [],
            "rating": row.get("vote_average", 0.0),
            "weighted_rating": row.get("weighted_rating", 0.0),
            "content": content,
            "cf": cf,
            "theme": tags,
            "quality": quality,
            "score": float(score)
        })

    out.sort(key=lambda x: (x["score"], x.get("weighted_rating", 0.0)), reverse=True)
    return out[:n]

# instantiate
recommender = CompleteRecommender(
    tmdb_df=tmdb,
    svd=svd_model, ncf=ncf_model,
    user_map=user_to_idx, movie_map=movie_to_idx,
    faiss_idx=faiss_index, emb=emb,
    key2idx=key2idx, key_set=key_set
)

print("✅ Recommender ready (title-key mode).")

# ============ Enable CF scoring (safe, optional per-row) ============

# 1) Add a CF method to the existing class (handles unknown users/items safely)
def _cf(self, user_raw, movie_raw_key):
    """
    Returns a rating in [1..5] using SVD + NeuMF, or 2.5 if we can't score.
    - SVD uses raw string IDs (Surprise).
    - NeuMF uses mapped indices (user_to_idx, movie_to_idx).
    """
    # If either is missing, return neutral
    if user_raw is None or movie_raw_key is None:
        return 2.5

    # SVD (handles strings)
    try:
        svd_pred = self.svd.predict(str(user_raw), str(movie_raw_key)).est
        svd_scaled = float(np.clip(svd_pred, 1, 5))
    except Exception:
        svd_scaled = 3.0  # neutral-ish

    # NeuMF (mapped indices -> sigmoid -> [1..5])
    uid = self.user_to_idx.get(str(user_raw))
    mid = self.movie_to_idx.get(str(movie_raw_key))
    if uid is not None and mid is not None:
        try:
            p01 = float(self.ncf.predict([np.array([uid]), np.array([mid])], verbose=0).reshape(-1)[0])
            ncf_scaled = float(np.clip(1 + 4*p01, 1, 5))
        except Exception:
            ncf_scaled = 3.0
    else:
        ncf_scaled = 3.0

    return 0.5 * svd_scaled + 0.5 * ncf_scaled

CompleteRecommender._cf = _cf

# 2) Replace recommend() to use CF when possible
def recommend_with_cf(self, query, user_id=None, n=10):
    filters = parse_query(query)
    cand = filter_by_metadata(self.df, filters)
    if len(cand) == 0:
        return []

    # keep only titles we can score by content
    cand["__key__"] = cand["title_norm"].astype(str)
    cand = cand[cand["__key__"].isin(self.key_set)].copy()
    if len(cand) == 0:
        return []

    # optional title seed
    seed_pool = seed_from_title(query, k=2000)
    if seed_pool:
        cand = cand[cand["__key__"].isin(seed_pool)]
        if len(cand) == 0:
            return []

    # FAISS shortlist
    qv = encoder.encode([query]).astype(np.float32)
    D, I = self.faiss.search(qv, min(3000, self.faiss.ntotal))
    faiss_keys = set(emb_keys[i] for i in I[0] if i != -1)
    cand = cand[cand["__key__"].isin(faiss_keys)].head(500).copy()
    if len(cand) == 0:
        return []

    # tag steering
    q_themes = query_theme_set(query)
    cand["tag_score"] = cand.apply(lambda r: tag_overlap_score(r.get("lda_themes"), q_themes), axis=1)

    # score blend (content + CF + theme + quality)
    out = []
    for _, row in cand.iterrows():
        key = row["__key__"]
        content = CompleteRecommender._content(None, qv, key, self.key2idx, self.emb) / 5.0

        # --- CF hook ---
        # Use CF only if we have a movie id present in the CF mappings
        cf_key = row.get("original_movie_id")
        can_cf = (cf_key is not None) and (str(cf_key) in self.movie_to_idx or str(cf_key) in self.movie_to_idx.keys())
        if user_id is not None and can_cf:
            cf = self._cf(user_id, cf_key) / 5.0
        else:
            cf = 0.5  # neutral when CF not available

        tags    = float(row.get("tag_score", 0.7))
        quality = float(row.get("weighted_rating", 5.0)) / 10.0

        score = 0.40*content + 0.40*cf + 0.10*tags + 0.10*quality

        genres = row.get("genre_names") or []
        if isinstance(genres, str):
            try: genres = ast.literal_eval(genres)
            except: genres = [genres]

        out.append({
            "title": row.get("tmdb_title") or row.get("title"),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": genres if isinstance(genres, list) else [],
            "themes": row.get("lda_themes") or [],
            "rating": row.get("vote_average", 0.0),
            "weighted_rating": row.get("weighted_rating", 0.0),
            "content": content, "cf": cf, "theme": tags, "quality": quality,
            "score": float(score)
        })

    out.sort(key=lambda x: (x["score"], x.get("weighted_rating", 0.0)), reverse=True)
    return out[:n]

CompleteRecommender.recommend = recommend_with_cf

print("✅ Collaborative filtering enabled (uses CF when movie_id is known).")

# (Optional) helper: pick a known training user for testing CF
# WARNING: Only use this if you don't have a real user_id handy.
# example_user_id = next(iter(user_to_idx.keys()))
# print("Example known user_id:", example_user_id)

# ============ Fix parser misfire: ignore emotion/theme words in actor extraction ============

# extend the movie_query_parser's internal stoplist dynamically
# (no need to modify the source file)
BAD_ACTOR_WORDS = {
    "dark","psychological","family","inspiring","romantic","sad","happy",
    "thriller","drama","comedy","action","horror","documentary"
}

# wrap parse_query to post-clean actor field
def parse_query_safe(q):
    f = parse_query(q)
    if "actor" in f:
        a = str(f["actor"]).lower()
        if any(w in a for w in BAD_ACTOR_WORDS):
            f.pop("actor", None)
    return f

# patch the recommender to use the safe version
CompleteRecommender.parse_query = staticmethod(parse_query_safe)

print("✅ Parser patched to ignore emotion/theme words as actors")

# ============ Ensure recommender uses the patched parser ============
def parse_query_safe(q):
    f = parse_query(q)
    if "actor" in f:
        a = str(f["actor"]).lower()
        bad_words = {
            "dark","psychological","family","inspiring","romantic","sad","happy",
            "thriller","drama","comedy","action","horror","documentary"
        }
        if any(w in a for w in bad_words):
            f.pop("actor", None)
    return f

# override the call inside recommend()
def recommend_with_safe_parse(self, query, user_id=None, n=10):
    filters = parse_query_safe(query)
    cand = filter_by_metadata(self.df, filters)
    if len(cand) == 0:
        return []

    cand["__key__"] = cand["title_norm"].astype(str)
    cand = cand[cand["__key__"].isin(self.key_set)].copy()
    if len(cand) == 0:
        return []

    seed_pool = seed_from_title(query, k=2000)
    if seed_pool:
        cand = cand[cand["__key__"].isin(seed_pool)]
        if len(cand) == 0:
            return []

    qv = encoder.encode([query]).astype(np.float32)
    D, I = self.faiss.search(qv, min(3000, self.faiss.ntotal))
    faiss_keys = set(emb_keys[i] for i in I[0] if i != -1)
    cand = cand[cand["__key__"].isin(faiss_keys)].head(500).copy()
    if len(cand) == 0:
        return []

    q_themes = query_theme_set(query)
    cand["tag_score"] = cand.apply(lambda r: tag_overlap_score(r.get("lda_themes"), q_themes), axis=1)

    out = []
    for _, row in cand.iterrows():
        key = row["__key__"]
        content = CompleteRecommender._content(None, qv, key, self.key2idx, self.emb) / 5.0

        cf_key = row.get("original_movie_id")
        can_cf = (cf_key is not None) and (str(cf_key) in self.movie_to_idx)
        if user_id is not None and can_cf:
            cf = self._cf(user_id, cf_key) / 5.0
        else:
            cf = 0.5

        tags    = float(row.get("tag_score", 0.7))
        quality = float(row.get("weighted_rating", 5.0)) / 10.0
        score   = 0.40*content + 0.40*cf + 0.10*tags + 0.10*quality

        genres = row.get("genre_names") or []
        if isinstance(genres, str):
            try: genres = ast.literal_eval(genres)
            except: genres = [genres]

        out.append({
            "title": row.get("tmdb_title") or row.get("title"),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": genres if isinstance(genres, list) else [],
            "themes": row.get("lda_themes") or [],
            "rating": row.get("vote_average", 0.0),
            "weighted_rating": row.get("weighted_rating", 0.0),
            "content": content, "cf": cf, "theme": tags, "quality": quality,
            "score": float(score)
        })

    out.sort(key=lambda x: (x["score"], x.get("weighted_rating", 0.0)), reverse=True)
    return out[:n]

CompleteRecommender.recommend = recommend_with_safe_parse

print("✅ recommender updated to use safe parse (ignores emotion/theme words as actors)")

# ============ FIX: call _content with 4 args (not 5) ============

def recommend_with_safe_parse(self, query, user_id=None, n=10):
    filters = parse_query_safe(query)
    cand = filter_by_metadata(self.df, filters)
    if len(cand) == 0:
        return []

    cand["__key__"] = cand["title_norm"].astype(str)
    cand = cand[cand["__key__"].isin(self.key_set)].copy()
    if len(cand) == 0:
        return []

    seed_pool = seed_from_title(query, k=2000)
    if seed_pool:
        cand = cand[cand["__key__"].isin(seed_pool)]
        if len(cand) == 0:
            return []

    qv = encoder.encode([query]).astype(np.float32)
    D, I = self.faiss.search(qv, min(3000, self.faiss.ntotal))
    faiss_keys = set(emb_keys[i] for i in I[0] if i != -1)
    cand = cand[cand["__key__"].isin(faiss_keys)].head(500).copy()
    if len(cand) == 0:
        return []

    q_themes = query_theme_set(query)
    cand["tag_score"] = cand.apply(lambda r: tag_overlap_score(r.get("lda_themes"), q_themes), axis=1)

    out = []
    for _, row in cand.iterrows():
        key = row["__key__"]
        # FIX HERE: use self._content(qv, key, self.key2idx, self.emb)
        content = self._content(qv, key, self.key2idx, self.emb) / 5.0

        cf_key = row.get("original_movie_id")
        can_cf = (cf_key is not None) and (str(cf_key) in self.movie_to_idx)
        if user_id is not None and can_cf:
            cf = self._cf(user_id, cf_key) / 5.0
        else:
            cf = 0.5

        tags    = float(row.get("tag_score", 0.7))
        quality = float(row.get("weighted_rating", 5.0)) / 10.0
        score   = 0.40*content + 0.40*cf + 0.10*tags + 0.10*quality

        genres = row.get("genre_names") or []
        if isinstance(genres, str):
            try: genres = ast.literal_eval(genres)
            except: genres = [genres]

        out.append({
            "title": row.get("tmdb_title") or row.get("title"),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": genres if isinstance(genres, list) else [],
            "themes": row.get("lda_themes") or [],
            "rating": row.get("vote_average", 0.0),
            "weighted_rating": row.get("weighted_rating", 0.0),
            "content": content, "cf": cf, "theme": tags, "quality": quality,
            "score": float(score)
        })

    out.sort(key=lambda x: (x["score"], x.get("weighted_rating", 0.0)), reverse=True)
    return out[:n]

CompleteRecommender.recommend = recommend_with_safe_parse
print("✅ recommend() patched")

# ============ Exported interface ============

def get_recommender():
    """
    Returns the fully initialized CompleteRecommender instance.
    """
    return recommender
