# Security Policy

## Supported Practices

- Keep real secrets out of source control; use environment variables, Docker secrets, or Kubernetes Secrets.
- Use `.env.example` for documented local configuration and keep `.env` ignored.
- Prefer SQLite for local/dev and PostgreSQL credentials from the runtime environment for production-like deployments.
- Use parameterized SQL in backend persistence code.
- Validate API inputs at the FastAPI boundary before invoking model or control logic.
- Run containers as non-root where the current Dockerfiles support it.
- Treat Kubernetes files in `k8s/` as starter manifests, and replace `secret.example.yaml` before applying anything to a real cluster.

## Reporting Vulnerabilities

If you find a vulnerability, please report it privately when the hosting platform supports private security advisories. Include the affected component, reproduction steps, impact, and any suggested mitigation.

## Known Non-Production Limitations

- The FastAPI serving layer currently uses a documented stub until a C++ binding or internal RPC call is implemented.
- Kubernetes manifests are minimal starter manifests, not hardened production infrastructure.
- MLflow and database backups, network policies, TLS termination, secret rotation, and cluster-specific RBAC are not fully implemented here.
- TensorRT/GPU deployment hardening is not claimed as complete.
