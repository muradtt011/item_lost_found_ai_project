
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root on sys.path so imports work from any cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Settings, configure_logging
from src.core.matching_service import MatchingService
from src.models import ItemStatus
from src.services.ai_service import AIService
from src.storage.repository import InMemoryRepository

# Offline providers reused from demo_ai.py pattern
import json as _json
import numpy as np
from ai.providers.base import VLMProvider, EmbeddingProvider


class _OfflineVLM(VLMProvider):
    def describe(self, image_path: str, prompt: str, *, json_schema=None) -> str:
        name = Path(image_path).stem.lower()
        for kw, obj, colors, brand, marks in [
            ("umbrella", "umbrella", ["black"], None, []),
            ("backpack", "backpack", ["navy"], "JanSport", ["worn front zipper"]),
            ("phone", "phone", ["black"], "Apple", ["cracked screen corner"]),
            ("wallet", "wallet", ["brown"], None, ["leather, tri-fold"]),
            ("keys", "key ring", ["silver"], None, ["3 keys, blue keychain"]),
        ]:
            if kw in name:
                return _json.dumps({
                    "object_class": obj, "colors": colors, "brand": brand,
                    "distinguishing_marks": marks, "location_hints": [],
                    "confidence": 0.85,
                })
        return _json.dumps({
            "object_class": "unknown object", "colors": [], "brand": None,
            "distinguishing_marks": [], "location_hints": [], "confidence": 0.3,
        })


class _OfflineEmbedder(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 64

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2 ** 31))
        v = rng.standard_normal(64).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


async def run_demo(offline: bool) -> None:
    configure_logging()
    logger = logging.getLogger("demo")

    lost_dir = ROOT / "data" / "lost"
    found_dir = ROOT / "data" / "found"

    if not lost_dir.exists() or not found_dir.exists():
        print(f"ERROR: Sample data missing under {ROOT / 'data'}", file=sys.stderr)
        sys.exit(2)

    settings = Settings(
        llm_provider="offline",
        embedding_provider="offline",
        ai_max_retries=0,
        ai_timeout_seconds=30.0,
        ai_concurrency_limit=4,
        image_storage_dir=str(ROOT / "storage" / "demo_images"),
        database_url="sqlite+aiosqlite:///./demo.db",
        log_level="INFO",
    )

    vlm = _OfflineVLM() if offline else None
    embedder = _OfflineEmbedder() if offline else None

    repo = InMemoryRepository()
    ai_svc = AIService(settings=settings, vlm=vlm, embedder=embedder)
    matching = MatchingService(repository=repo, ai_service=ai_svc, settings=settings)

    mode = "offline" if offline else "online"
    print(f"\n{'='*60}")
    print(f"  Smart Lost & Found — Demo ({mode} mode)")
    print(f"{'='*60}\n")

    # --- Register lost items ---
    print("📦 Registering LOST items…")
    lost_records = []
    for img in sorted(lost_dir.glob("*.png")) + sorted(lost_dir.glob("*.jpg")):
        try:
            rec = await matching.register_item(ItemStatus.lost, img, "")
            desc = rec.vlm_description or {}
            print(f"  ✓ {img.name:<30s}  [{desc.get('object_class', '?')}]  id={rec.id[:8]}…")
            lost_records.append(rec)
        except Exception as exc:
            print(f"  ✗ {img.name}: {exc}", file=sys.stderr)

    # --- Register found items ---
    print("\n🔍 Registering FOUND items…")
    found_records = []
    for img in sorted(found_dir.glob("*.png")) + sorted(found_dir.glob("*.jpg")):
        try:
            rec = await matching.register_item(ItemStatus.found, img, "")
            desc = rec.vlm_description or {}
            print(f"  ✓ {img.name:<30s}  [{desc.get('object_class', '?')}]  id={rec.id[:8]}…")
            found_records.append(rec)
        except Exception as exc:
            print(f"  ✗ {img.name}: {exc}", file=sys.stderr)

    if not lost_records or not found_records:
        print("\nERROR: Not enough items to match.", file=sys.stderr)
        sys.exit(2)

    # --- Query first lost item ---
    query = lost_records[0]
    q_desc = query.vlm_description or {}
    print(f"\n{'─'*60}")
    print(f"🔎 Top-3 matches for LOST item: {Path(query.image_path).name}")
    print(f"   Object : {q_desc.get('object_class', '?')}")
    print(f"   Colors : {', '.join(q_desc.get('colors', []))}")
    print(f"   Item ID: {query.id}")
    print(f"{'─'*60}")

    response = await matching.find_matches(query.id, k=3)

    artefact = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "query_item": {
            "id": query.id,
            "image": Path(query.image_path).name,
            "vlm_description": query.vlm_description,
        },
        "matches": [],
    }

    for rank, m in enumerate(response.matches, 1):
        m_desc = m.vlm_description or {}
        print(f"\n  [{rank}] {Path(m.image_path).name}")
        print(f"       Score  : {m.score:+.4f}")
        print(f"       Object : {m_desc.get('object_class', '?')}")
        print(f"       Colors : {', '.join(m_desc.get('colors', []))}")
        print(f"       Reason : {m.reason}")
        artefact["matches"].append({
            "rank": rank,
            "item_id": m.item_id,
            "image": Path(m.image_path).name,
            "score": m.score,
            "reason": m.reason,
            "vlm_description": m.vlm_description,
        })

    # --- Write artefact ---
    artefacts_dir = ROOT / "artefacts"
    artefacts_dir.mkdir(exist_ok=True)
    out_path = artefacts_dir / "demo_results.json"
    out_path.write_text(json.dumps(artefact, indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"✅ Artefact written to: {out_path.relative_to(ROOT)}")
    print(f"{'='*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true", help="Use fake providers (no API keys)")
    args = p.parse_args()
    asyncio.run(run_demo(offline=args.offline))


if __name__ == "__main__":
    main()
