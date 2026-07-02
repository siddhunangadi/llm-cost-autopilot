from backend.api.paths import STATIC_DIR


def test_htmx_asset_is_vendored():
    path = STATIC_DIR / "js" / "htmx.min.js"
    assert path.exists()
    assert path.stat().st_size > 10_000
    assert "htmx" in path.read_text()[:2000].lower()


def test_chartjs_asset_is_vendored():
    path = STATIC_DIR / "js" / "chart.min.js"
    assert path.exists()
    assert path.stat().st_size > 50_000


def test_dashboard_css_exists():
    path = STATIC_DIR / "css" / "dashboard.css"
    assert path.exists()
    assert path.stat().st_size > 0
