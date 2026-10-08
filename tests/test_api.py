"""
Integration Tests for Flask REST API Endpoints.
"""

import pytest
from app.api.routes import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "database_connected" in data
    assert "model_available" in data


def test_api_upload_missing_file(client):
    response = client.post("/api/v1/videos/upload")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


def test_api_process_missing_video_id(client):
    response = client.post("/api/v1/videos/process", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
