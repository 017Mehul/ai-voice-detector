"""Smoke tests for the static web UI served by FastAPI."""

from fastapi.testclient import TestClient

from voice_detector.api.app import create_app


def test_web_ui_routes_return_index_html():
    client = TestClient(create_app())

    for route in ["/app", "/app/workspace", "/app/report", "/app/status"]:
        response = client.get(route)

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert '<div id="app-shell"></div>' in response.text
        assert "/app-assets/app.css" in response.text
        assert "/app-assets/app.js" in response.text


def test_web_ui_assets_are_served():
    client = TestClient(create_app())

    css_response = client.get("/app-assets/app.css")
    js_response = client.get("/app-assets/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert ".topbar" in css_response.text

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert "renderRoute" in js_response.text


def test_api_root_remains_json_service_descriptor():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "AI Voice Detection System"
    assert body["endpoints"]["detect_voice"] == "/api/v1/detect-voice"
