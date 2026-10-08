"""Create the Atlas Search keyword index on chunks.text (idempotent).

Run with::

    uv run python -m building_with_rag.ingestion.keyword_index
"""

import sys
import time

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from building_with_rag.config import get_settings
from building_with_rag.ingestion import mongodb_schema as schema

WAIT_SECONDS = 120


def _find(coll):
    return next(
        (i for i in coll.list_search_indexes() if i.get("name") == schema.KEYWORD_INDEX_NAME), None
    )


def _ready(idx) -> bool:
    return bool(idx and (idx.get("queryable") or idx.get("status") == "READY"))


def _diff(want, have, path="") -> list[str]:
    if isinstance(want, dict) and isinstance(have, dict):
        out = []
        for k in sorted(set(want) | set(have)):
            out += _diff(want.get(k), have.get(k), f"{path}.{k}" if path else k)
        return out
    return [] if want == have else [f"{path}: expected {want!r}, found {have!r}"]


def main() -> int:
    settings = get_settings()
    if not settings.mongodb_uri:
        print("MONGODB_URI is not set in .env.")
        return 1
    testing = settings.app_env == "testing"
    name = settings.mongodb_test_db_name if testing else settings.mongodb_db_name
    try:
        coll = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)[name][
            schema.CHUNKS_COLLECTION
        ]
        idx = _find(coll)
        if idx is None:
            coll.create_search_index(
                SearchIndexModel(
                    name=schema.KEYWORD_INDEX_NAME,
                    type="search",
                    definition=schema.KEYWORD_INDEX_DEFINITION,
                )
            )
            print(f"{schema.KEYWORD_INDEX_NAME}: created")
        else:
            differences = _diff(schema.KEYWORD_INDEX_DEFINITION, idx.get("latestDefinition"))
            if differences:
                print(f"{schema.KEYWORD_INDEX_NAME}: exists with a different definition.")
                for line in differences:
                    print(f"  {line}")
                print("Drop it manually in Atlas, then rerun.")
                return 1
            print(f"{schema.KEYWORD_INDEX_NAME}: reused")
        deadline = time.monotonic() + WAIT_SECONDS
        while not _ready(idx) and time.monotonic() < deadline:
            time.sleep(5)
            idx = _find(coll)
    except PyMongoError as exc:
        print(f"MongoDB error ({type(exc).__name__}).")
        return 1
    status = (idx or {}).get("status", "MISSING")
    print(f"{schema.KEYWORD_INDEX_NAME}: {status}, queryable={_ready(idx)}")
    return 0 if _ready(idx) else 1


if __name__ == "__main__":
    sys.exit(main())
