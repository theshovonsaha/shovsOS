"""Tests for run_engine/skill_loader.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

from run_engine.skill_loader import (
    SkillManifest,
    _strip_yaml_frontmatter,
    list_available_skills,
    load_skill_context,
    load_skill_manifest,
)


def _make_skill_dir(tmp: Path, skill_name: str, content: str) -> Path:
    skill_dir = tmp / ".agent" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class TestStripYamlFrontmatter:
    def test_with_frontmatter(self):
        content = "---\nname: pdf\ndescription: PDF tools\n---\n\n# Instructions\nDo stuff."
        metadata, body = _strip_yaml_frontmatter(content)
        assert metadata["name"] == "pdf"
        assert metadata["description"] == "PDF tools"
        assert body.startswith("# Instructions")

    def test_without_frontmatter(self):
        content = "# No Frontmatter\nJust instructions."
        metadata, body = _strip_yaml_frontmatter(content)
        assert metadata == {}
        assert body == content.strip()

    def test_empty_content(self):
        metadata, body = _strip_yaml_frontmatter("")
        assert metadata == {}
        assert body == ""


class TestLoadSkillManifest:
    def test_loads_skill_with_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(
                Path(tmp), "pdf",
                "---\nname: pdf\ndescription: Handle PDFs\n---\n\n# PDF Guide\nUse pypdf."
            )
            manifest = load_skill_manifest("pdf", tmp)
            assert manifest is not None
            assert manifest.name == "pdf"
            assert manifest.description == "Handle PDFs"
            assert "pypdf" in manifest.body

    def test_loads_skill_without_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(
                Path(tmp), "backend",
                "# Backend Skill\nDo backend things."
            )
            manifest = load_skill_manifest("backend", tmp)
            assert manifest is not None
            assert manifest.name == "backend"
            assert manifest.description == ""
            assert "backend things" in manifest.body

    def test_returns_none_for_missing_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = load_skill_manifest("nonexistent", tmp)
            assert manifest is None

    def test_returns_none_for_empty_workspace(self):
        manifest = load_skill_manifest("pdf", None)
        assert manifest is None

    def test_returns_none_for_empty_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(Path(tmp), "empty", "---\nname: empty\n---\n")
            manifest = load_skill_manifest("empty", tmp)
            assert manifest is None


class TestLoadSkillContext:
    def test_returns_body_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(
                Path(tmp), "pdf",
                "---\nname: pdf\n---\n\n# PDF Processing\nUse pypdf for merging."
            )
            ctx = load_skill_context("pdf", tmp)
            assert "PDF Processing" in ctx
            assert "pypdf" in ctx

    def test_truncates_long_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_body = "---\nname: big\n---\n\n" + ("x" * 5000)
            _make_skill_dir(Path(tmp), "big", long_body)
            ctx = load_skill_context("big", tmp, max_chars=100)
            assert len(ctx) <= 140  # 100 + truncation suffix
            assert "truncated" in ctx

    def test_returns_empty_for_missing(self):
        ctx = load_skill_context("nope", "/nonexistent")
        assert ctx == ""


class TestListAvailableSkills:
    def test_discovers_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(Path(tmp), "pdf", "---\nname: pdf\ndescription: PDF\n---\n\nInstructions.")
            _make_skill_dir(Path(tmp), "backend", "# Backend\nInstructions.")
            skills = list_available_skills(tmp)
            names = [s.name for s in skills]
            assert "pdf" in names
            assert "backend" in names

    def test_returns_empty_for_no_workspace(self):
        assert list_available_skills(None) == []

    def test_skips_invalid_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_skill_dir(Path(tmp), "good", "# Good\nBody.")
            _make_skill_dir(Path(tmp), "empty", "---\nname: empty\n---\n")
            skills = list_available_skills(tmp)
            names = [s.name for s in skills]
            assert "good" in names
            assert "empty" not in names
