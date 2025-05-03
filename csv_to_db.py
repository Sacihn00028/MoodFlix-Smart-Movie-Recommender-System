import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# ——————— Initialize Firebase ———————
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ——————— Load CSV ———————
# Replace 'movies.csv' with your file; ensure it’s in the working directory
df = pd.read_csv("movies.csv")

# ——————— Choose a collection name ———————
# You can call this 'movies', or anything that makes sense
coll = db.collection("movies")

# ——————— Write rows to Firestore ———————
batch = db.batch()
for idx, row in df.iterrows():
    # Pick a document ID—could be your movie_id column, or Firestore auto-ID
    doc_id = str(row["movie_id"])
    doc_ref = coll.document(doc_id)
    # Convert the row to a plain dict (columns become keys)
    data = row.to_dict()
    # (Optional) drop NaNs or convert types here
    batch.set(doc_ref, data)
    # Commit every 500 writes (Firestore batch limit)
    if (idx + 1) % 500 == 0:
        batch.commit()
        batch = db.batch()

# Commit any leftovers
batch.commit()

print(f"Loaded {len(df)} documents into Firestore collection '{coll.id}'.")
