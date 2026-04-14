"""Tests for run_engine/code_intent.py"""

from __future__ import annotations

from run_engine.code_intent import CodeIntent, classify_code_intent


class TestClassifyCodeIntent:
    def test_explicit_code_request(self):
        result = classify_code_intent("Write me a Python script to parse CSV files")
        assert result.code_warranted is True
        assert result.execution_risk_tier in ("read_only", "write")
        assert "explicit" in result.reason

    def test_implicit_code_from_domain(self):
        result = classify_code_intent("Analyze this CSV and extract the top 10 rows")
        assert result.code_warranted is True
        assert "domain" in result.reason or "file type" in result.reason

    def test_file_type_reference(self):
        result = classify_code_intent("Create a .py file that sorts a list")
        assert result.code_warranted is True
        assert "explicit" in result.reason or "file type" in result.reason

    def test_no_code_intent_for_chat(self):
        result = classify_code_intent("Hi, how are you?")
        assert result.code_warranted is False
        assert result.execution_risk_tier == "none"

    def test_no_code_intent_for_research(self):
        result = classify_code_intent("What is the capital of France?")
        assert result.code_warranted is False

    def test_destructive_risk_tier(self):
        result = classify_code_intent("Write a script to delete all log files")
        assert result.code_warranted is True
        assert result.execution_risk_tier == "destructive"

    def test_write_risk_tier(self):
        result = classify_code_intent("Create a script to save data to a JSON file")
        assert result.code_warranted is True
        assert result.execution_risk_tier == "write"

    def test_read_only_risk_tier(self):
        result = classify_code_intent("Analyze the database entries for errors")
        assert result.code_warranted is True
        assert result.execution_risk_tier == "read_only"

    def test_ambiguous_scope_triggers_missing_context(self):
        result = classify_code_intent("Write me a script.")
        assert result.code_warranted is True
        assert result.missing_context is not None
        assert "specific" in result.missing_context.lower() or "goal" in result.missing_context.lower()

    def test_specific_scope_no_missing_context(self):
        result = classify_code_intent("Write a script to convert CSV to JSON using pandas")
        assert result.code_warranted is True
        assert result.missing_context is None

    def test_empty_message(self):
        result = classify_code_intent("")
        assert result.code_warranted is False

    def test_neutral_result_is_singleton(self):
        r1 = classify_code_intent("")
        r2 = classify_code_intent("hello")
        assert r1.code_warranted is False
        assert r2.code_warranted is False


class TestCodeIntentNotes:
    def test_phase_note_for_code_task(self):
        result = classify_code_intent("Build a Python app to track expenses")
        note = result.to_phase_note()
        assert "Code intent detected" in note

    def test_phase_note_empty_for_non_code(self):
        result = classify_code_intent("What time is it?")
        assert result.to_phase_note() == ""

    def test_risk_note_for_destructive(self):
        result = classify_code_intent("Write a script to delete all temporary files")
        note = result.to_risk_note()
        assert "destructive" in note.lower() or "WARNING" in note

    def test_risk_note_for_write(self):
        result = classify_code_intent("Create a Python script to save results")
        note = result.to_risk_note()
        assert "write" in note.lower() or "verify" in note.lower()

    def test_risk_note_empty_for_read_only(self):
        result = classify_code_intent("Parse this JSON data")
        note = result.to_risk_note()
        assert note == ""

    def test_risk_note_empty_for_non_code(self):
        result = classify_code_intent("Tell me a joke")
        assert result.to_risk_note() == ""


class TestFallbackStrategy:
    def test_destructive_fallback(self):
        result = classify_code_intent("Write a script to remove all old data files")
        assert "preview" in result.fallback_if_failed.lower() or "dry-run" in result.fallback_if_failed.lower()

    def test_write_fallback(self):
        result = classify_code_intent("Create a script to save the report")
        assert "review" in result.fallback_if_failed.lower() or "chat" in result.fallback_if_failed.lower()
