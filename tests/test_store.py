from mcp_vector.store import VectorStore
from mcp_vector.cli import main


def test_ann_and_metadata_filter() -> None:
    store = VectorStore()
    store.upsert("a", "离心泵滚动轴承每 2000 小时补充锂基脂", {"source": "pump"})
    store.upsert("b", "电机绕组绝缘电阻应大于规定值", {"source": "motor"})
    store.upsert("c", "齿轮箱油位在视镜中线", {"source": "gear"})
    hits = store.search("轴承 润滑 脂", k=3)
    assert hits
    assert hits[0][0].doc_id == "a"
    filtered = store.search("轴承", k=3, where={"source": "motor"})
    assert all(rec.metadata["source"] == "motor" for rec, _ in filtered)


def test_cli_roundtrip(tmp_path) -> None:
    index = tmp_path / "index.json"
    assert main(["upsert", "--id", "x", "--text", "停机后冷却再开箱", "--index", str(index)]) == 0
    assert main(["search", "冷却", "--index", str(index)]) == 0
