# Guide: Pushing a Vendor to Business Central (manual)

This is the step-by-step for getting a vendor that was created in the portal
into Business Central. It is a **manual** process because the portal's backend
cannot reach the BC server (`ntz-srv-bcdb`) — only machines on the Netsmartz
VPN can.

There are two phases:

- **Phase A — one-time setup** (do once, then never again)
- **Phase B — per vendor** (repeat for every vendor you push)


---

## Phase A — One-time setup

### A1. Turn the feature on in the backend

In the backend's env file (`backend/.env` for local, `.env.test` /
`.env.production` on a server), set:

```
BC_ENABLED=true
BC_ODATA_BASE=http://ntz-srv-bcdb:2248/BC220/ODataV4
BC_COMPANY=Netsmartz Infotech (India) Pri
```

Leave the posting-group lines blank for now:

```
BC_GEN_BUS_POSTING_GROUP=
BC_VAT_BUS_POSTING_GROUP=
BC_VENDOR_POSTING_GROUP=
```

Restart the backend. If `BC_ENABLED` is still `false`, the portal's
"Get BC payload" button returns an error (503) and nothing else works.

### A2. Get the push script onto a VPN machine

The script is `scripts/push_to_bc.ps1` in this repo.

1. Remote Desktop into the VPN machine
   (`192.168.10.106`, user `NETSMARTZ\BCD`, password as shared).
2. Copy `push_to_bc.ps1` onto that machine — e.g. into `C:\bc-push\`.
   (Paste through the RDP clipboard, or a shared folder, or email it to
   yourself and download it there.)
3. Open **PowerShell** on that machine and `cd` to that folder:
   ```powershell
   cd C:\bc-push
   ```
4. If PowerShell refuses to run the script ("running scripts is disabled"),
   allow it for this session only:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```

### A3. Prove the account can talk to BC

Still on the VPN machine, in PowerShell, run a **read** test:

```powershell
Invoke-RestMethod -UseDefaultCredentials `
  -Uri "http://ntz-srv-bcdb:2248/BC220/ODataV4/Company('Netsmartz Infotech (India) Pri')/VendorCard?`$top=1"
```

- If it prints a vendor's fields → read access works. Good.
- If it errors with 401/403 → the account has no BC access; stop and ask the
  BC admin.

> A read working does **not** guarantee a write (POST) will work. The first
> real push in Phase B is the true test. If possible, ask the BC admin whether
> there is a **test/sandbox company** to push to first, so a mistake doesn't
> create a junk vendor in the live company.


---

## Phase B — Push one vendor

### B1. Create and save the vendor in the portal

Dashboard → **Vendor Creation** → upload the documents → review the extracted
fields on the compare screen → **Validate & Submit**. You land on a success
screen showing "Vendor saved (ref #N)".

### B2. Open the vendor and get its BC payload

1. Go to **Dashboard → Saved Records → Vendors** (or click "View saved record"
   on the success screen).
2. Click the vendor to open its detail page.
3. Scroll to the **Business Central** section.
4. Click **Get BC payload**.
   - The portal shows a block of JSON (the vendor translated into BC's field
     names) and the target URL.
   - If you see "Business Central integration is turned off" → Phase A1 wasn't
     done / backend not restarted.
5. Click **Download JSON**. You get a file named `vendor_<N>_bc.json`.

### B3. Move the file to the VPN machine

Get `vendor_<N>_bc.json` onto the machine from A2 — RDP clipboard paste, a
shared folder, or download it there.

### B4. Send it to Business Central

On the VPN machine, in PowerShell, in the folder with the script and the file:

```powershell
# 1. Preview — shows exactly what will be sent, sends nothing
.\push_to_bc.ps1 -File .\vendor_5_bc.json -WhatIf

# 2. Send it for real
.\push_to_bc.ps1 -File .\vendor_5_bc.json
```

**On success** it prints:

```
CREATED.
BC No.        : EMPV/0123
Name          : M.B. Control & Systems Pvt. Ltd.

Next: in the portal, open this vendor and 'Mark as pushed' with BC No. = EMPV/0123
```

Write down that **BC No.** (`EMPV/0123` in the example).

**On failure** it prints an HTTP status and BC's message. See
"Troubleshooting" below.

### B5. Record the BC No. back in the portal

1. Return to the vendor's detail page in the portal.
2. In the **Business Central** section, the "BC No. returned" box is there.
3. Type the BC No. from B4 (`EMPV/0123`) and click **Mark as pushed**.
4. The section now shows: **"Pushed to Business Central as EMPV/0123 on <date>."**

That vendor is done. It will not be offered for push again (the button is
replaced by the "pushed" message).

### B6. Verify in Business Central (optional but recommended)

On the VPN machine, open BC in a browser, go to the Vendors list, and confirm
the new vendor is there with the details you expect.


---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Portal: "Business Central integration is turned off (503)" | `BC_ENABLED` not `true`, or backend not restarted | Phase A1 |
| Script: "running scripts is disabled on this system" | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Script: `HTTP 401` or `403` | The Windows account can't **write** to BC OData (read may still work) | Ask the BC admin to grant the account permission to create vendors via web services |
| Script: `HTTP 400` mentioning a **posting group** | BC requires a posting group on insert | Ask the BC admin which values to use, set `BC_GEN_BUS_POSTING_GROUP` / `BC_VAT_BUS_POSTING_GROUP` / `BC_VENDOR_POSTING_GROUP` in the backend `.env`, restart, redo B2–B4 |
| Script: `HTTP 400` about another required field | That field isn't in the portal's data / mapper | Note the field name and raise it — the mapper (`backend/app/services/bc_mapper.py`) needs updating |
| Script: `HTTP 404` | `BC_ODATA_BASE` or `BC_COMPANY` wrong for this server | Re-check the URL that worked in A3, update `.env` |
| Script: cannot connect / timeout | The machine you're on is not on the VPN / can't reach `ntz-srv-bcdb` | Use the correct VPN machine |
| Portal: "This vendor is already marked as pushed" (409) | You already recorded a BC No. for it | Nothing to do — it's already done |

## Notes

- **One vendor at a time.** There is no bulk push.
- The portal only *records* that a vendor was pushed — it does not re-check BC.
  If a vendor is deleted in BC, the portal still shows it as "pushed"; you'd
  fix that manually in the database if it ever happens.
- Customer push is **not built yet** — this guide is vendors only.
- When the backend is eventually hosted on a machine that can reach
  `ntz-srv-bcdb`, this whole manual flow can be replaced by a single button
  that does B2–B5 automatically. The field mapping stays the same.
