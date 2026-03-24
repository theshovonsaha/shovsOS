import json
import sys
import types

import pytest

from plugins.tools import _pdf_processor, _safe_path


@pytest.mark.asyncio
async def test_pdf_processor_create_returns_preview_payload(monkeypatch):
    class FakeCanvas:
        def __init__(self, out_path, pagesize=None):
            self.out_path = out_path

        def drawString(self, x, y, text):
            return None

        def save(self):
            with open(self.out_path, "wb") as f:
                f.write(b"%PDF-1.4\n")

    fake_reportlab = types.ModuleType("reportlab")
    fake_lib = types.ModuleType("reportlab.lib")
    fake_pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    fake_pagesizes.letter = (612, 792)
    fake_pdfgen = types.ModuleType("reportlab.pdfgen")
    fake_canvas_mod = types.ModuleType("reportlab.pdfgen.canvas")
    fake_canvas_mod.Canvas = FakeCanvas

    monkeypatch.setitem(sys.modules, "reportlab", fake_reportlab)
    monkeypatch.setitem(sys.modules, "reportlab.lib", fake_lib)
    monkeypatch.setitem(sys.modules, "reportlab.lib.pagesizes", fake_pagesizes)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen", fake_pdfgen)
    monkeypatch.setitem(sys.modules, "reportlab.pdfgen.canvas", fake_canvas_mod)

    result = await _pdf_processor(
        action="create",
        output_path="preview/sample output.pdf",
        content="hello",
    )
    payload = json.loads(result)
    assert payload["status"] == "success"
    assert payload["type"] == "pdf_preview"
    assert payload["action"] == "create"
    assert payload["file"] == "preview/sample output.pdf"
    assert payload["url"] == "/sandbox/preview/sample output.pdf"
    created = _safe_path("preview/sample output.pdf")
    assert created.exists()
    if created.exists():
        created.unlink()
    assert not created.exists()


@pytest.mark.asyncio
async def test_pdf_processor_read_returns_preview_payload(monkeypatch):
    target = _safe_path("preview/readme.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"%PDF-1.4\n")

    class FakePage:
        def extract_text(self):
            return "hello pdf"

    class FakePdf:
        def __init__(self):
            self.pages = [FakePage(), FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_pdfplumber = types.ModuleType("pdfplumber")
    fake_pdfplumber.open = lambda _: FakePdf()
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    result = await _pdf_processor(action="read", path="preview/readme.pdf")
    payload = json.loads(result)
    assert payload["status"] == "success"
    assert payload["type"] == "pdf_preview"
    assert payload["action"] == "read"
    assert payload["pages"] == 2
    assert payload["content"] == "hello pdfhello pdf"
    assert payload["path"] == "/sandbox/preview/readme.pdf"
    assert target.exists()
    if target.exists():
        target.unlink()
    assert not target.exists()
