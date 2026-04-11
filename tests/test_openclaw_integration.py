import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from plugins.shovs_meta_gateway import (
    _get_platform_manifest,
    _spatial_query,
    _list_spatial_loci
)

@pytest.fixture(autouse=True)
def setup_mocks():
    from plugins.shovs_meta_gateway import inject_gateway_dependencies
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = [{"name": "shovs_memory_query"}]
    
    mock_graph = MagicMock()
    mock_graph.traverse = AsyncMock(return_value=[])
    mock_graph.list_loci.return_value = []
    
    inject_gateway_dependencies(mock_registry, mock_graph)
    return mock_registry, mock_graph

@pytest.mark.asyncio
async def test_meta_gateway_manifest(setup_mocks):
    manifest_str = await _get_platform_manifest()
    manifest = json.loads(manifest_str)
    assert "platform" in manifest
    capabilities = manifest.get("capabilities", [])
    assert any("Spatial Memory" in c for c in capabilities)

@pytest.mark.asyncio
async def test_meta_gateway_memory_bridge(setup_mocks):
    topic = "bridge_test_topic"
    
    # We must patch where it's IMPORTED in shovs_meta_gateway, not where it's defined
    import plugins.shovs_meta_gateway
    original_search = plugins.shovs_meta_gateway.unified_memory_search
    mock_search = AsyncMock(return_value={
        "results": [{"kind": "fact", "object": "bridge_pass", "sources": ["test"]}],
        "stats": {"source_counts": {"test": 1}}
    })
    plugins.shovs_meta_gateway.unified_memory_search = mock_search
    
    try:
        results_str = await _spatial_query(query=topic, _owner_id="test_owner")
        results = json.loads(results_str)
        assert any(r.get("object") == "bridge_pass" for r in results.get("results", []))
    finally:
        plugins.shovs_meta_gateway.unified_memory_search = original_search

@pytest.mark.asyncio
async def test_meta_gateway_loci_list(setup_mocks):
    _, mock_graph = setup_mocks
    mock_graph.list_loci.return_value = [{"id": "kitchen", "name": "Kitchen"}]

    results_str = await _list_spatial_loci(_owner_id="test_owner")
    results = json.loads(results_str)
    assert any(l.get("id") == "kitchen" for l in results)
