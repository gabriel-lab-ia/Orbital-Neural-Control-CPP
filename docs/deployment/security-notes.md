# Deployment Security Notes

The optional deployment stack is a starter, not a hardened production platform.

- Keep `ORBITAL_JOB_EXECUTOR=0` unless the backend is isolated and authenticated.
- Put REST, WebSocket, MLflow, FastAPI, and PostgreSQL behind private networking and TLS.
- Add authentication, authorization, rate limits, request body limits, and connection limits before public exposure.
- Replace example secrets and use a managed secret store with rotation.
- Restrict artifact mounts and validate checkpoint/model provenance.
- Add Kubernetes NetworkPolicies, RBAC, Pod Security settings, backups, and image signing.
- Treat FastAPI `/control-step` as a deterministic stub until a real C++/LibTorch bridge exists.
- TensorRT engines are hardware/runtime-specific artifacts and require provenance and compatibility validation.

