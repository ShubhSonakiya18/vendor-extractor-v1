# scripts/

## push_to_bc.ps1 — push a vendor to Business Central (manual)

The portal cannot reach the BC OData endpoint (`ntz-srv-bcdb:2248`) directly —
it is only reachable from inside the VPN. So the push is a two-person-minutes
manual step:

1. **In the portal**: open the saved vendor → *Business Central* panel →
   **Get BC payload** → **Download JSON**. You get `vendor_<id>_bc.json`
   containing `target_url` and `payload`.

2. **On a VPN machine** (Remote Desktop into the BC host, or any domain machine
   that can reach `ntz-srv-bcdb`), in PowerShell:

   ```powershell
   # preview what would be sent
   .\push_to_bc.ps1 -File .\vendor_5_bc.json -WhatIf

   # actually send it
   .\push_to_bc.ps1 -File .\vendor_5_bc.json
   ```

   It POSTs with `-UseDefaultCredentials` (your logged-in Windows account —
   NTLM), and on success prints the `No.` BC assigned, e.g. `EMPV/0123`.

3. **Back in the portal**: in the same *Business Central* panel, enter that
   `No.` in **Mark as pushed**. The vendor's `bc_status` becomes `pushed` and
   it will not be offered for push again.

### Requirements
- `BC_ENABLED=true` in the backend's `.env` (otherwise the payload endpoint
  returns 503).
- The Windows account used on the VPN machine must have permission to create
  vendors via BC OData — not just read them. Confirm with the BC admin before
  the first real push; test against a BC sandbox company if one exists.

### If the POST fails
The script prints BC's HTTP status and response body. Common causes:
- **401 / 403** — the Windows account lacks OData write permission.
- **400** with a message about a posting group — set
  `BC_GEN_BUS_POSTING_GROUP` / `BC_VAT_BUS_POSTING_GROUP` /
  `BC_VENDOR_POSTING_GROUP` in the backend `.env`, regenerate the payload.
- **404** — `BC_ODATA_BASE` / `BC_COMPANY` is wrong for this server.
