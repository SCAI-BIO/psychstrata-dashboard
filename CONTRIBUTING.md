Contributing
============

Thanks for your interest in the Psych-STRATA Dashboard. Contributions are
welcome — please open an issue or a pull request on GitHub.

## Development setup

See the [README](README.md) for backend (uv), frontend (pnpm), and system-test
setup instructions.

## Before opening a pull request

- Run the backend tests: `cd backend && uv run pytest`
- Run the frontend tests and typecheck: `cd frontend && pnpm test && pnpm build`

## Conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`type(optional-scope): subject`).
- Branch names follow `type/description` in kebab-case, e.g. `fix/42-broken-url`.

## License of contributions

By submitting a contribution, you agree that it is licensed under the
[Apache License 2.0](LICENSE), the same license as this project
(inbound = outbound). You confirm that you have the right to submit the
contribution under that license.

Please note that the project may change its license in future versions;
contributions may be relicensed accordingly as part of the project.
