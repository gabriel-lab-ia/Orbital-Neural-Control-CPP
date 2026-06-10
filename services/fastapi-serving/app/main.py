from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


FiniteFloat = Annotated[float, Field(ge=-1.0e9, le=1.0e9)]


class OrbitalState(BaseModel):
    position_km: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    velocity_kmps: Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
    mission_time_s: Annotated[float, Field(ge=0.0, le=1.0e9)]


class ControlStepRequest(BaseModel):
    run_id: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
    state: OrbitalState
    policy_backend: Annotated[str, Field(min_length=1, max_length=32)] = "libtorch"

    @field_validator("policy_backend")
    @classmethod
    def validate_policy_backend(cls, value: str) -> str:
        allowed = {"libtorch", "mock"}
        if value not in allowed:
            raise ValueError(f"policy_backend must be one of {sorted(allowed)}")
        return value


class ControlStepResponse(BaseModel):
    run_id: str
    control_vector: Annotated[list[float], Field(min_length=3, max_length=3)]
    control_magnitude: float
    policy_backend: str
    model_status: str


class HealthResponse(BaseModel):
    status: str
    service: str
    detail: str | None = None


app = FastAPI(
    title="Orbital FastAPI Serving",
    version="0.1.0",
    description="Optional API layer for deployment experiments around the C++20 orbital control core.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="fastapi-serving")


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    return HealthResponse(
        status="ready_stub_only",
        service="fastapi-serving",
        detail="No real C++/LibTorch control bridge is implemented.",
    )


@app.post("/control-step", response_model=ControlStepResponse)
def control_step(request: ControlStepRequest) -> ControlStepResponse:
    if request.policy_backend == "libtorch":
        # TODO: Replace this stub with pybind11 or an internal RPC call to the C++/LibTorch pipeline.
        model_status = "stubbed_cpp_endpoint_configured" if os.getenv("ORBITAL_CPP_CONTROL_ENDPOINT") else "stubbed_no_cpp_binding"
    elif request.policy_backend == "mock":
        model_status = "stubbed_mock_backend"

    position = request.state.position_km
    velocity = request.state.velocity_kmps
    control_vector = [
        round(-0.0001 * position[0] - 0.01 * velocity[0], 8),
        round(-0.0001 * position[1] - 0.01 * velocity[1], 8),
        round(-0.0001 * position[2] - 0.01 * velocity[2], 8),
    ]
    control_magnitude = sum(component * component for component in control_vector) ** 0.5

    return ControlStepResponse(
        run_id=request.run_id,
        control_vector=control_vector,
        control_magnitude=control_magnitude,
        policy_backend=request.policy_backend,
        model_status=model_status,
    )


@app.post("/predict", response_model=ControlStepResponse)
def predict(request: ControlStepRequest) -> ControlStepResponse:
    return control_step(request)
