# Environments & Promotion

The portal runs as two fully separate deployments. This is the "separate the
entire flow, not just the database" decision from the team discussion: the
Production portal works only with Production resources, the Test portal only
with Test resources.

| | **Test** | **Production** |
| --- | --- | --- |
| Purpose | validate every change before it ships | live use |
| Backend `ENV` | `test` | `production` |
| Env file | `.env.test` (`ENV_FILE=.env.test`) | `.env.production` (`ENV_FILE=.env.production`) |
| Database | Test Postgres | Production Postgres (never shared) |
| `SECRET_KEY` | its own | its own (distinct) |
| Frontend origin | `bc-portal-test.…` | `bc-portal.…` |
| `CORS_ALLOW_ORIGINS` | Test portal origin only | Production portal origin only |
| Business Central | disabled, or a BC **sandbox** company | Production BC (once step 6 lands) |
| Who deploys | anyone, freely | only after approval (below) |

`local` is a third `ENV` for developer machines: SQLite, permissive CORS, no
guard rails. It is not a deployed environment.

## Why the split is enforced in code, not just convention

`backend/app/config/config.py`:

- `ENV` must be `local`, `test`, or `production`.
- When `ENV=production` the app **refuses to start** if `DATABASE_URL` is still
  the SQLite default, `SECRET_KEY` is shorter than 32 chars, or
  `CORS_ALLOW_ORIGINS` is empty. A misconfigured Production process fails fast
  instead of coming up pointed at the wrong data.
- CORS: `local` allows any origin; `test`/`production` allow **only** the exact
  origins listed in `CORS_ALLOW_ORIGINS`. The Production API cannot be called
  from the Test portal, or vice versa.
- Logs are written to a per-environment file (`app.test.log`,
  `app.production.log`) so a shared host never interleaves them.
- `BC_ENABLED` lets Test hard-disable the Business Central push while
  Production has it on.

## Deploying to Test

1. Merge your branch into the branch Test deploys from (`feature/authentication`
   for now; `master` once this work lands).
2. On the Test host: pull, `pip install -r backend/requirements.txt`, run any
   pending DB migration, restart the backend with `ENV_FILE=.env.test`.
3. Rebuild the frontend with `VITE_API_URL` pointing at the Test backend.
4. Validate the change end to end in the Test portal.

## Promoting Test → Production

Only after the change has been validated in Test:

1. Open a **Pull Request / Change Request** describing what changed and the
   Test validation done.
2. Get it reviewed and **approved**.
3. Merge to `master`.
4. On the Production host: pull the approved commit, `pip install`, run
   migrations, restart with `ENV_FILE=.env.production`.
5. Rebuild the frontend against the Production backend.
6. Smoke-test the Production portal (login, extract, save a vendor and a
   customer).

No change reaches Production that has not been through Test and an approved
PR — including config and `.env` changes.

## First-time setup per host

```bash
# backend
cd backend
cp .env.test.example .env.test          # or .env.production.example
#  → set SECRET_KEY (python -c "import secrets; print(secrets.token_hex(32))")
#  → set DATABASE_URL to that environment's Postgres
#  → set CORS_ALLOW_ORIGINS to that environment's portal origin
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ENV_FILE=.env.test alembic upgrade head              # create / update schema
ENV_FILE=.env.test python -m app.cli.seed_users      # create login accounts
ENV_FILE=.env.test uvicorn main:app --host 0.0.0.0 --port 8000
```

## Database migrations

The schema is managed by **Alembic** (`backend/alembic/`). `ENV=local`
auto-creates missing tables on startup for convenience; **Test and Production
do not** -- every deploy to those runs:

```bash
ENV_FILE=.env.test alembic upgrade head
```

as a step *before* restarting the backend. Migration files live in
`backend/alembic/versions/` and are committed alongside the model change that
produced them. See `backend/alembic/README` for the workflow.

## Business Central push (vendor)

Currently a **manual** flow — the portal cannot reach `ntz-srv-bcdb:2248`
directly. Per environment:

- `BC_ENABLED=false` by default. Set `true` only where the *Business Central*
  panel should appear.
- In **Test**, only enable it against a BC **sandbox company**, never
  Production BC.
- The operator step (download payload → run `scripts/push_to_bc.ps1` on a VPN
  machine → mark pushed) is documented in `scripts/README.md`.
- `BC_ODATA_BASE` / `BC_COMPANY` and the optional `BC_*_POSTING_GROUP` values
  are per-environment `.env` settings.

An automatic in-portal push (backend calls BC itself) is possible later, but
needs the backend hosted where it can reach the BC server over the VPN, plus a
BC service account with OData write permission.

## Not yet decided / out of scope here

- **Hosting** for each environment (which machines, process manager, TLS).
  Note that an *automatic* Business Central push would need the backend to run
  somewhere that can reach `ntz-srv-bcdb` over the VPN.
- **Postgres provisioning** for Test and Production.
- **Backup schedule** for each Postgres instance.
- **CI / branch protection** to enforce "Test + approved PR before Production".
