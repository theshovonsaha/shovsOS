"""
Skill Loader
-------------
Reads SKILL.md files from the workspace skill directory.
Strips YAML frontmatter and returns the instruction body.
Returns empty string gracefully if skill is not found.

Frontmatter fields supported:
  name        — display name (defaults to dir name)
  description — one-line summary used by the planner
  triggers    — comma-separated keywords that activate the skill automatically
  requirements — comma-separated tool/capability names this skill needs
  eligibility — "always" | "explicit_only" | "auto" (default: "auto")
                always      → always inject into context
                explicit_only → only when user or planner explicitly names it
                auto        → activated by trigger match (default)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.logger import log

YAML_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

SKILL_FILENAME = "SKILL.md"
SKILLS_SUBDIR = ".agent/skills"


@dataclass(frozen=True)
class SkillManifest:
    """Parsed skill metadata from YAML frontmatter."""

    name: str
    description: str
    body: str
    triggers: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    eligibility: str = "auto"  # "always" | "explicit_only" | "auto"

    def matches_message(self, message: str) -> bool:
        """Return True if any trigger keyword appears in the message (case-insensitive)."""
        if not self.triggers:
            return False
        lower = message.lower()
        return any(t.lower() in lower for t in self.triggers)

    def is_eligible_for_auto(self, message: str, available_tools: list[str]) -> bool:
        """Return True if this skill should be auto-activated for the given message/tools."""
        if self.eligibility == "always":
            return True
        if self.eligibility == "explicit_only":
            return False
        # "auto" — activate if triggers match and required tools are present
        if self.requirements and available_tools:
            if not all(r in available_tools for r in self.requirements):
                return False
        return self.matches_message(message)


def _strip_yaml_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Strip YAML frontmatter and return (metadata_dict, body).

    Only parses simple key: value pairs — no nested YAML.
    Falls back to empty metadata if no frontmatter found.
    """
    match = YAML_FRONTMATTER_RE.match(content)
    if not match:
        return {}, content.strip()

    raw_front = match.group(0)
    body = content[match.end():].strip()

    metadata: dict[str, str] = {}
    for line in raw_front.splitlines():
        line = line.strip()
        if line == "---" or not line:
            continue
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip().lower()
            value = line[colon_idx + 1:].strip()
            metadata[key] = value

    return metadata, body


def load_skill_manifest(
    skill_name: str,
    workspace_path: Optional[str] = None,
) -> Optional[SkillManifest]:
    """Load and parse a skill's SKILL.md file.

    Args:
        skill_name: Name of the skill directory (e.g., "pdf").
        workspace_path: Root workspace path. Skill is resolved at
            ``{workspace_path}/.agent/skills/{skill_name}/SKILL.md``.

    Returns:
        SkillManifest if found and readable, None otherwise.
    """
    if not skill_name or not workspace_path:
        return None

    skill_dir = Path(workspace_path) / SKILLS_SUBDIR / skill_name
    skill_file = skill_dir / SKILL_FILENAME

    if not skill_file.is_file():
        log("skill", "loader", f"SKILL.md not found for '{skill_name}' at {skill_file}", level="warn")
        return None

    try:
        raw = skill_file.read_text(encoding="utf-8")
    except Exception as exc:
        log("skill", "loader", f"Failed to read SKILL.md for '{skill_name}': {exc}", level="error")
        return None

    metadata, body = _strip_yaml_frontmatter(raw)

    if not body.strip():
        log("skill", "loader", f"SKILL.md for '{skill_name}' has no instruction body.", level="warn")
        return None

    name = metadata.get("name", skill_name)
    description = metadata.get("description", "")

    return SkillManifest(name=name, description=description, body=body)


def load_skill_context(
    skill_name: str,
    workspace_path: Optional[str] = None,
    max_chars: int = 3000,
) -> str:
    """Load a skill's instruction body for context injection.

    Returns empty string if skill is not found — graceful degradation.
    """
    manifest = load_skill_manifest(skill_name, workspace_path)
    if manifest is None:
        return ""

    body = manifest.body
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n[...skill instructions truncated...]"

    log("skill", "loader", f"Loaded skill '{manifest.name}' ({len(body)} chars)")
    return body


def list_available_skills(workspace_path: Optional[str] = None) -> list[SkillManifest]:
    """Discover all skills in the workspace skill directory.

    Returns a list of SkillManifest objects with name, description, and body
    for each skill that has a valid SKILL.md file.
    """
    if not workspace_path:
        return []

    skills_dir = Path(workspace_path) / SKILLS_SUBDIR
    if not skills_dir.is_dir():
        return []

    manifests: list[SkillManifest] = []
    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        manifest = load_skill_manifest(child.name, workspace_path)
        if manifest is not None:
            manifests.append(manifest)

    return manifests
