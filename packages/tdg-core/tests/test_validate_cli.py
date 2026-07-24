import json

from tdg_core.validate import main


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "schema_version": "1.0", "document_id": "d",
        "facts": [], "dependencies": []}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"facts": "not-a-list"}))
    assert main(["validate", str(good)]) == 0
    assert main(["validate", str(bad)]) == 1
    assert main(["validate", str(tmp_path)]) == 1  # dir mode hits the bad file
