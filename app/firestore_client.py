
"""Step 1 of RAG: LOAD the data.

Used ONLY by scripts/ingest.py on your laptop. It authenticates with the Firebase
service-account key whose path is in GOOGLE_APPLICATION_CREDENTIALS. Admin access bypasses
Firestore security rules, so the key must never be committed, deployed or sent to a browser.
"""
from google.cloud import firestore
from dotenv import load_dotenv

load_dotenv()

COLLECTIONS = ("projects", "skills", "experience", "education", "certifications")
SETTINGS_DOCS = ("site", "next", "contact", "imports")


def get_db(project_id: str, database: str = "(default)") -> firestore.Client:
    return firestore.Client(project=project_id, database=database)


def _sort_key(item: dict):
    # same ordering the website uses: `order`, then `createdAt`
    order = item.get("order")
    return (order if isinstance(order, (int, float)) else 9999, item.get("createdAt") or 0)


def fetch_portfolio(db: firestore.Client) -> dict:
    """Read every collection the portfolio uses. Returns plain dicts."""
    data: dict = {}
    for name in COLLECTIONS:
        items = []
        for snap in db.collection(name).stream():
            item = snap.to_dict() or {}
            item.setdefault("id", snap.id)
            if name == "projects":
                item.setdefault("slug", snap.id)  # project doc id == slug
            items.append(item)
        items.sort(key=_sort_key)
        data[name] = items

    data["settings"] = {}
    for sid in SETTINGS_DOCS:
        snap = db.collection("settings").document(sid).get()
        data["settings"][sid] = (snap.to_dict() or {}) if snap.exists else {}
    return data
