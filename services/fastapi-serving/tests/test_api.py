import pytest
from pydantic import ValidationError

from app.main import ControlStepRequest, control_step, health, ready


def test_health_and_ready() -> None:
    assert health().status == "ok"
    assert ready().status == "ready"


def test_control_step_returns_stubbed_response() -> None:
    request = ControlStepRequest(
        run_id="api_smoke_001",
        state={
            "position_km": [7000.0, 0.0, 0.0],
            "velocity_kmps": [0.0, 7.5, 0.0],
            "mission_time_s": 10.0,
        },
        policy_backend="mock",
    )
    response = control_step(request)

    assert response.run_id == "api_smoke_001"
    assert response.model_status == "stubbed_mock_backend"
    assert len(response.control_vector) == 3


def test_libtorch_request_stays_stubbed_when_endpoint_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORBITAL_CPP_CONTROL_ENDPOINT", "http://orbital-backend:8080")
    request = ControlStepRequest(
        run_id="api_smoke_002",
        state={
            "position_km": [7000.0, 0.0, 0.0],
            "velocity_kmps": [0.0, 7.5, 0.0],
            "mission_time_s": 10.0,
        },
    )

    response = control_step(request)

    assert response.policy_backend == "libtorch"
    assert response.model_status == "stubbed_cpp_endpoint_configured"


def test_control_step_validates_run_id_and_vector_length() -> None:
    with pytest.raises(ValidationError):
        ControlStepRequest(
            run_id="../bad",
            state={
                "position_km": [7000.0, 0.0],
                "velocity_kmps": [0.0, 7.5, 0.0],
                "mission_time_s": 10.0,
            },
        )


def test_control_step_rejects_unknown_policy_backend() -> None:
    with pytest.raises(ValidationError):
        ControlStepRequest(
            run_id="api_smoke_003",
            state={
                "position_km": [7000.0, 0.0, 0.0],
                "velocity_kmps": [0.0, 7.5, 0.0],
                "mission_time_s": 10.0,
            },
            policy_backend="unknown",
        )
