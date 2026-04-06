# Contributing

Thank you for considering a contribution to `infer`.
This project is focused on building a clean, lightweight Rust inference crate
with a small and maintainable codebase.

## How to contribute

1. Open an issue for any bug or feature request before starting work.
2. Create a branch from `dev` with a descriptive name.
3. Keep changes focused and small.
4. Run the existing tests and formatting checks locally.
5. Submit a pull request targeting `dev` with a short summary and any relevant context.

## Development workflow

- Run tests:
  ```sh
  cargo test
  ```
- Run the default build:
  ```sh
  cargo build
  ```
- Run clippy with warnings denied:
  ```sh
  cargo clippy -- -D warnings
  ```
- Verify formatting:
  ```sh
  cargo fmt --check
  ```

## Code style

- Prefer explicit, readable Rust over clever shortcuts.
- Avoid `unwrap()` in library code.
- Keep public APIs stable and consistent.
- Use feature flags for optional dependencies.

## Branches and pull requests

- Base changes on `dev`.
- Do not open PRs directly against `main`.
- Use meaningful commit messages.
- Add tests for bug fixes and new functionality.
- Mention related issues in the PR description.
