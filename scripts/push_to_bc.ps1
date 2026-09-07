<#
.SYNOPSIS
  POST a Business Central VendorCard or CustomerCard payload from a machine
  on the VPN.

.DESCRIPTION
  The portal (GET /business-central/vendors/{id}/payload or
  GET /business-central/customers/{id}/payload) produces a JSON file with two
  keys: "target_url" and "payload". This script is agnostic to which one it
  is -- it just POSTs "payload" to "target_url" using the logged-in Windows
  account (NTLM / -UseDefaultCredentials), which is how the BC OData endpoint
  is reachable from inside the VPN.

  On success it prints the "No." Business Central assigned. Paste that back into
  the portal ("Mark as pushed") so the record is not sent twice.

.PARAMETER File
  Path to the JSON file downloaded from the portal (e.g. vendor_5_bc.json or
  customer_12_bc.json -- the "Download JSON" button on the record's detail
  page names it after the record kind).

.PARAMETER WhatIf
  Show the request that would be sent, without sending it.

.EXAMPLE
  .\push_to_bc.ps1 -File .\vendor_5_bc.json
  .\push_to_bc.ps1 -File .\customer_12_bc.json -WhatIf
#>
param(
  [Parameter(Mandatory = $true)]
  [string]$File,

  [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $File)) {
  Write-Error "File not found: $File"
  exit 1
}

$doc = Get-Content -LiteralPath $File -Raw | ConvertFrom-Json

if (-not $doc.target_url -or -not $doc.payload) {
  Write-Error "JSON must contain 'target_url' and 'payload'. Is this the file from GET /business-central/vendors/{id}/payload ?"
  exit 1
}

$url  = [string]$doc.target_url
$body = $doc.payload | ConvertTo-Json -Depth 10

Write-Host "Target : $url"
Write-Host "Body   :"
Write-Host $body
Write-Host ""

if ($WhatIf) {
  Write-Host "(-WhatIf) Not sending."
  exit 0
}

try {
  $resp = Invoke-RestMethod -Method Post -Uri $url `
    -UseDefaultCredentials `
    -ContentType 'application/json' `
    -Body $body
}
catch {
  Write-Host "FAILED." -ForegroundColor Red
  if ($_.Exception.Response) {
    $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
    Write-Host ("HTTP {0}" -f [int]$_.Exception.Response.StatusCode)
    Write-Host $reader.ReadToEnd()
  } else {
    Write-Host $_.Exception.Message
  }
  exit 1
}

Write-Host "CREATED." -ForegroundColor Green
Write-Host ("BC No.        : {0}" -f $resp.No)
Write-Host ("Name          : {0}" -f $resp.Name)
Write-Host ""
Write-Host "Next: in the portal, open this record and 'Mark as pushed' with BC No. = $($resp.No)"
