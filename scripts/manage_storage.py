#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from orchestration.agent_profiles import ProfileManager
from orchestration.session_manager import SessionManager
from services.storage_admin import StorageAdminService, StoreSelection


def _selection_from_args(args: argparse.Namespace) -> StoreSelection:
    if not args.include:
        return StoreSelection()

    include = set(args.include)
    return StoreSelection(
        sessions="sessions" in include,
        agents="agents" in include,
        semantic_memory="semantic_memory" in include,
        tool_results="tool_results" in include,
        vector_memory="vector_memory" in include,
        session_rag="session_rag" in include,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup and reset ShovsAI storage safely.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show current storage paths and counts.")
    backups_parser = subparsers.add_parser("backups", help="List recent backups.")
    backups_parser.add_argument("--limit", type=int, default=20)

    for name in ("backup", "reset"):
        sp = subparsers.add_parser(name, help=f"{name.title()} selected stores.")
        sp.add_argument(
            "--include",
            nargs="*",
            choices=["sessions", "agents", "semantic_memory", "tool_results", "vector_memory", "session_rag"],
            help="Stores to include. Defaults to sessions, semantic_memory, tool_results, vector_memory, session_rag.",
        )
        sp.add_argument("--label", default="", help="Optional label appended to the backup folder name.")
        if name == "reset":
            sp.add_argument("--no-backup", action="store_true", help="Skip backup before reset.")
            sp.add_argument(
                "--drop-default-agent",
                action="store_true",
                help="Also clear the default agent profile when resetting agent storage.",
            )

    args = parser.parse_args()

    service = StorageAdminService(SessionManager(max_sessions=200), ProfileManager())

    if args.command == "status":
        payload = service.status()
    elif args.command == "backups":
        payload = service.list_backups(limit=args.limit)
    elif args.command == "backup":
        payload = service.backup(_selection_from_args(args), label=args.label)
    else:
        payload = service.reset(
            _selection_from_args(args),
            backup_first=not args.no_backup,
            backup_label=args.label,
            preserve_default_agent=not args.drop_default_agent,
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
