"""Verify the replacement frontend is served as local modules; browser behavior is exercised separately."""
from fastapi.testclient import TestClient
from app.main import create_app


def test_frontend_and_local_modules_are_served():
    client = TestClient(create_app())
    page = client.get('/')
    assert page.status_code == 200
    assert 'lang="zh-CN"' in page.text
    assert '/assets/bootstrap.js' in page.text
    assert 'id="demo-gate"' not in page.text
    for filename in ('bootstrap.js', 'app.js', 'views.js', 'components.js', 'api.js', 'icons.js', 'styles.css', 'mark.svg', 'model-settings.js'):
        response = client.get(f'/assets/{filename}')
        assert response.status_code == 200, filename
        assert len(response.content) > 10


def test_discarded_frontend_has_no_legacy_route():
    client = TestClient(create_app())
    assert client.get('/legacy.html').status_code == 404
    assert client.get('/assets/mark.svg').headers['content-type'].startswith('image/svg+xml')
