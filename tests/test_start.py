from importlib import machinery, util
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load():
    loader = machinery.SourceFileLoader("start", str(ROOT / "start.pyw"))
    mod = util.module_from_spec(util.spec_from_loader("start", loader))
    loader.exec_module(mod)
    return mod


def test_install_uninstall_startup_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    start = load()
    vbs = start.startup_entry()
    vbs.parent.mkdir(parents=True)  # exists on a real Windows profile
    start.main(["--install"])
    assert "start.pyw" in vbs.read_text() and vbs.parent.parts[-1] == "Startup"
    start.main(["--uninstall"])
    assert not vbs.exists()
    start.main(["--uninstall"])  # already gone: no error
