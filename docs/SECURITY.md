# Local Security Notes

This workspace is configured for private local development.

## What is already in place

- Build artifacts are ignored by `.gitignore`
- Common secret files such as `.env`, `.pem`, `.key`, and `secrets/` are ignored
- Linux linker hardening flags are enabled for normal builds
- Debug sanitizers are available through the `debug-sanitized` preset
- VS Code uses `compile_commands.json` to avoid manual include-path drift

## Good habits for this workspace

- Keep remote publishing disabled unless you intentionally create a repository
- Review files under `build/`, `logs/`, `runs/`, and `checkpoints/` before sharing anything
- Avoid storing production credentials in source files
- Prefer CPU-only experiments unless you explicitly add and validate GPU dependencies

## If you initialize Git later

- Run `git init`
- Confirm `.gitignore` is present before the first commit
- Consider removing vendored dependency metadata like nested `.git` folders only if you no longer need their history

