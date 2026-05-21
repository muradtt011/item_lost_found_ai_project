"""Command-line interface for Smart Lost & Found.

Commands
--------
register-lost    Register a lost item
register-found   Register a found item
search-matches   Find top-k matches for an item
list             List all registered items

Usage
-----
python -m src.cli --help
python -m src.cli register-lost --image path/to/img.png --text "lost my black umbrella"
python -m src.cli search-matches --id <item-id> --k 5
python -m src.cli list --status lost
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is importable when run as `python -m src.cli`
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import configure_logging, get_settings
from src.core.matching_service import MatchingService
from src.models import ItemStatus
from src.services.ai_service import AIService
from src.storage.repository import make_repository


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lost-found",
        description="Smart Lost & Found — command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # register-lost
    p_lost = sub.add_parser("register-lost", help="Register a lost item")
    p_lost.add_argument("--image", required=True, help="Path to image (JPEG/PNG)")
    p_lost.add_argument("--text", default="", help="Free-text description")

    # register-found
    p_found = sub.add_parser("register-found", help="Register a found item")
    p_found.add_argument("--image", required=True, help="Path to image (JPEG/PNG)")
    p_found.add_argument("--text", default="", help="Free-text description")

    # search-matches
    p_match = sub.add_parser("search-matches", help="Find top-k matches for an item")
    p_match.add_argument("--id", required=True, dest="item_id", help="Item ID")
    p_match.add_argument("--k", type=int, default=3, help="Number of results (default 3)")

    # list
    p_list = sub.add_parser("list", help="List registered items")
    p_list.add_argument(
        "--status",
        choices=["lost", "found", "all"],
        default="all",
        help="Filter by status (default: all)",
    )

    return parser


async def _async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings)

    repo = make_repository()
    ai_service = AIService(settings=settings)
    matching = MatchingService(
        repository=repo, ai_service=ai_service, settings=settings
    )

    if args.command in ("register-lost", "register-found"):
        status = ItemStatus.lost if args.command == "register-lost" else ItemStatus.found
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
            return 1

        print(f"Registering {status.value} item: {image_path.name} …")
        try:
            record = await matching.register_item(
                status=status,
                image_path=image_path,
                user_text=args.text,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        print(json.dumps(
            {
                "id": record.id,
                "status": record.status.value,
                "image_path": record.image_path,
                "user_text": record.user_text,
                "vlm_description": record.vlm_description,
                "created_at": record.created_at.isoformat(),
            },
            indent=2,
        ))
        return 0

    elif args.command == "search-matches":
        print(f"Searching top-{args.k} matches for item {args.item_id} …")
        try:
            response = await matching.find_matches(args.item_id, k=args.k)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        if not response.matches:
            print("No matches found.")
            return 0

        for i, m in enumerate(response.matches, 1):
            print(f"\n  [{i}] item_id={m.item_id}  score={m.score:+.3f}")
            print(f"       status={m.status.value}  text={m.user_text!r}")
            print(f"       reason={m.reason}")
        return 0

    elif args.command == "list":
        filter_status = None if args.status == "all" else ItemStatus(args.status)
        records = await repo.list_items(status=filter_status)
        if not records:
            print("No items found.")
            return 0

        for r in records:
            desc = r.vlm_description or {}
            obj_class = desc.get("object_class", "(no description yet)")
            print(
                f"  {r.id[:8]}…  [{r.status.value:5s}]  {obj_class:<20s}  {r.user_text[:40]!r}"
            )
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
