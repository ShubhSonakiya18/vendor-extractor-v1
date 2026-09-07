/**
 * api.js — Central API service layer
 *
 * All backend calls go through here so components stay clean.
 * Proxied by Vite to http://127.0.0.1:8000 in development.
 *
 * Key endpoint: POST /extract
 *   Body (multipart/form-data):
 *     documents      — one or more File objects (PDF, DOCX, image)
 *     vendor_template — optional Excel template File
 *   Response:
 *     {
 *       run_id: string,
 *       values: { [fieldName]: value },
 *       fields: { [fieldName]: { value, confidence, source, flagged } },
 *       needs_review: string[],
 *       summary: { filled, total_fields, … },
 *       verification: { … } | null,
 *       timings: { load, extract, total },
 *     }
 */

// Empty in dev: vite.config.js proxies these paths to 127.0.0.1:8000.
// Set VITE_API_URL at build time (e.g. in Netlify) to point a static build at
// a backend on another origin, such as an ngrok URL -- see ar-portal/.env.example.
const BASE = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || ''

const SKIP_NGROK_WARNING = {
  'ngrok-skip-browser-warning': 'true',
  'Bypass-Tunnel-Reminder': 'true',
  'bypass-tunnel-reminder': 'true',
}

/* ─── Auth ──────────────────────────────────────────────────────────────
 * Normal email + password auth against the backend /auth flow. The access
 * token is kept in localStorage and attached to every API call by
 * authFetch(). Accounts are admin-provisioned (see backend seed_users.py) --
 * there is no sign-up here.
 */
const TOKEN_KEY   = 'bc_access_token'
const REFRESH_KEY = 'bc_refresh_token'

export function getToken()   { return localStorage.getItem(TOKEN_KEY) }
export function isLoggedIn()  { return !!getToken() }

function storeTokens({ access_token, refresh_token }) {
  if (access_token)  localStorage.setItem(TOKEN_KEY, access_token)
  if (refresh_token) localStorage.setItem(REFRESH_KEY, refresh_token)
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
  localStorage.removeItem('userEmail')
}

/** Log in. Stores tokens on success; throws with a readable message otherwise. */
export async function login(email, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...SKIP_NGROK_WARNING },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    let body = {}
    try { body = await res.json() } catch (_) {}
    throw new Error(body.detail || (res.status === 401
      ? 'Incorrect email or password.'
      : `Sign-in failed (${res.status}).`))
  }
  const tokens = await res.json()
  storeTokens(tokens)
  localStorage.setItem('userEmail', email)
  return tokens
}

/** Revoke the refresh token server-side (best effort) and clear local state. */
export async function logout() {
  const refresh = localStorage.getItem(REFRESH_KEY)
  if (refresh) {
    try {
      await fetch(`${BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...SKIP_NGROK_WARNING },
        body: JSON.stringify({ refresh_token: refresh }),
      })
    } catch (_) { /* offline / already gone — clear locally anyway */ }
  }
  clearAuth()
}

/**
 * Exchange the stored refresh token for a fresh access (and refresh) token.
 * Returns the new access token, or null if there is nothing to refresh with /
 * the refresh was rejected. Concurrent callers share one in-flight request so
 * a burst of 401s triggers a single refresh, not one per request.
 */
let _refreshInFlight = null
async function refreshAccessToken() {
  const refresh = localStorage.getItem(REFRESH_KEY)
  if (!refresh) return null
  if (!_refreshInFlight) {
    _refreshInFlight = (async () => {
      try {
        const res = await fetch(`${BASE}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...SKIP_NGROK_WARNING },
          body: JSON.stringify({ refresh_token: refresh }),
        })
        if (!res.ok) return null
        const tokens = await res.json()
        storeTokens(tokens)
        return tokens.access_token || null
      } catch (_) {
        return null
      } finally {
        _refreshInFlight = null
      }
    })()
  }
  return _refreshInFlight
}

/**
 * fetch() wrapper that attaches the bearer token. On a 401 it tries once to
 * refresh the access token and replay the request; if that fails it clears
 * auth and throws an AUTH_EXPIRED sentinel the caller catches to redirect to
 * /login.
 */
export async function authFetch(path, options = {}) {
  const send = token => fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...SKIP_NGROK_WARNING,
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  })

  let res = await send(getToken())
  if (res.status === 401) {
    const fresh = await refreshAccessToken()
    if (fresh) {
      res = await send(fresh)
    }
    if (res.status === 401) {
      clearAuth()
      const err = new Error('Your session has expired. Please sign in again.')
      err.code = 'AUTH_EXPIRED'
      throw err
    }
  }
  return res
}

// Turn a FastAPI error body into a readable string. `detail` can be a string,
// an object with a `message`, or (on 422) an array of {loc, msg} validation
// errors -- which must not be stringified directly ([object Object]).
function errorMessage(body, status) {
  const d = body && body.detail
  if (typeof d === 'string') return d
  if (d && typeof d === 'object' && !Array.isArray(d) && d.message) return d.message
  if (Array.isArray(d)) {
    return d
      .map(e => {
        const field = Array.isArray(e.loc) ? e.loc.filter(p => p !== 'body').join('.') : ''
        return field ? `${field}: ${e.msg}` : e.msg
      })
      .join('; ')
  }
  if (body && body.error) return body.error
  return `Request failed (${status}).`
}

async function authJson(path, method, payload) {
  const res = await authFetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  })
  let body = null
  try { body = await res.json() } catch (_) {}
  if (!res.ok) {
    const err = new Error(errorMessage(body, res.status))
    err.status = res.status
    err.body = body
    throw err
  }
  return body
}

/** Persist a reviewed vendor record. Throws err with .status === 409 on a
 *  duplicate GSTIN (err.body.detail carries existing_vendor_id). */
export function createVendor(payload)   { return authJson('/vendors', 'POST', payload) }

/** Persist a reviewed customer record. 409 on duplicate GSTIN, as above. */
export function createCustomer(payload) { return authJson('/customers', 'POST', payload) }

/* ─── Saved records ─────────────────────────────────────────────────── */
export function listVendors()             { return authJson('/vendors', 'GET') }
export function getVendorById(id)          { return authJson(`/vendors/${id}`, 'GET') }
export function updateVendor(id, changes)  { return authJson(`/vendors/${id}`, 'PATCH', changes) }
export function deleteVendor(id)           { return authFetch(`/vendors/${id}`, { method: 'DELETE' }).then(r => {
  if (!r.ok && r.status !== 204) throw new Error(`Delete failed (${r.status}).`)
}) }
export function listCustomers()            { return authJson('/customers', 'GET') }
export function getCustomerById(id)        { return authJson(`/customers/${id}`, 'GET') }
export function updateCustomer(id, changes){ return authJson(`/customers/${id}`, 'PATCH', changes) }
export function deleteCustomer(id)         { return authFetch(`/customers/${id}`, { method: 'DELETE' }).then(r => {
  if (!r.ok && r.status !== 204) throw new Error(`Delete failed (${r.status}).`)
}) }

/* ─── Business Central (manual push) ────────────────────────────────
 * getVendorBcPayload -> { target_url, payload, already_pushed, bc_no }.
 * The operator POSTs `payload` to `target_url` from a VPN machine
 * (scripts/push_to_bc.ps1), then records the assigned No. here.
 * Both throw err.status === 503 when BC_ENABLED is false.
 */
export function getVendorBcPayload(id)      { return authJson(`/business-central/vendors/${id}/payload`, 'GET') }
export function markVendorPushed(id, bcNo)  { return authJson(`/business-central/vendors/${id}/mark-pushed`, 'PATCH', { bc_no: bcNo }) }

/** Same manual-push flow as the vendor pair above, for a saved customer --
 *  see backend/app/routers/business_central.py. */
export function getCustomerBcPayload(id)     { return authJson(`/business-central/customers/${id}/payload`, 'GET') }
export function markCustomerPushed(id, bcNo) { return authJson(`/business-central/customers/${id}/mark-pushed`, 'PATCH', { bc_no: bcNo }) }

/** The pipeline's needs_review is a list of {field, reason} objects; the
 *  /vendors and /customers APIs want a plain list of field-name strings. */
export function reviewFieldNames(needsReview) {
  if (!Array.isArray(needsReview)) return null
  return needsReview
    .map(x => (typeof x === 'string' ? x : x?.field))
    .filter(Boolean)
}

/**
 * Map a /extract response (pipeline `fields` / `values`, keyed by internal
 * names like `vendor_name`, `pin_code`, `gst_number`) onto the /vendors
 * request body (Vendor Creation Request Form field names).
 */
export function extractionToVendorPayload(result) {
  const vals = result?.values || {}
  const fields = result?.fields || {}
  const get = k => vals[k] || fields[k]?.value || ''
  return {
    vendor_name:        get('vendor_name'),
    address_1:          get('address_1'),
    address_2:          get('address_2'),
    address_3:          get('address_3'),
    address_4:          get('address_4'),
    city:              get('city'),
    state:             get('state'),
    country:           get('country'),
    pin_code:          get('pin_code'),
    telephone_1:       get('telephone'),
    telephone_2:       '',
    email:             get('email'),
    website:           get('website'),
    company_type:      get('company_type'),
    nature_of_business: get('nature_of_business'),
    tan_no:            get('tan'),
    pan:               get('pan'),
    tds_applicable:    get('tds_applicable'),
    gst_no:            get('gst_number'),
    esic_no:           get('esic_number'),
    udyam_no:          get('udyam_number'),
    bank_name:         get('bank_name'),
    branch_address:    get('branch_address'),
    ifsc_swift_code:   get('ifsc'),
    account_type:      get('account_type'),
    account_number:    get('account_number'),
    raw_extraction:    result ?? null,
    source_documents:  Array.isArray(result?.documents) ? result.documents : null,
    fields_needing_review: reviewFieldNames(result?.needs_review),
  }
}

/**
 * Run OCR extraction on uploaded documents.
 *
 * @param {File[]} documentFiles  – PDFs / images / DOCX files to extract from
 * @param {File|null} templateFile – optional Excel template to fill
 * @param {string} mapping        – excel mapping name (default: vendor_creation_v1)
 * @returns {Promise<object>}     – backend JSON response (see header above)
 */
export async function extractDocuments(documentFiles, templateFile = null, mapping = 'vendor_creation_v1') {
  const formData = new FormData()

  documentFiles.forEach(f => formData.append('documents', f))
  if (templateFile) formData.append('vendor_template', templateFile)
  formData.append('mapping', mapping)
  formData.append('models', 'small')

  const res = await fetch(`${BASE}/extract`, {
    method: 'POST',
    headers: SKIP_NGROK_WARNING,
    body: formData,
  })

  if (!res.ok) {
    let errBody = {}
    try { errBody = await res.json() } catch (_) { }
    const msg = errBody.error || errBody.detail || `Server error ${res.status}`
    throw new Error(msg)
  }

  return res.json()
}

/**
 * Run customer-onboarding extraction on uploaded documents.
 *
 * Hits POST /onboarding/extract, not /extract -- there is no vendor-style
 * Excel template and no document-type field: each file is OCR'd and
 * classified from its own content (cancelled cheque, GST certificate, Udyam
 * certificate, PAN card, ...) automatically, then reshaped straight into the
 * customer-onboarding form's fixed schema (company_name, billing_address,
 * gst_registration_number, bank_details, ...). See
 * backend/app/routers/onboarding.py and services/onboarding_mapper.py.
 *
 * @param {File[]} documentFiles – PDFs / images to extract from
 * @returns {Promise<object>}    – the onboarding schema (see onboarding_mapper.py)
 */
export async function extractOnboardingDocuments(documentFiles) {
  const formData = new FormData()
  documentFiles.forEach(f => formData.append('documents', f))

  const res = await fetch(`${BASE}/onboarding/extract`, {
    method: 'POST',
    headers: SKIP_NGROK_WARNING,
    body: formData,
  })

  if (!res.ok) {
    let errBody = {}
    try { errBody = await res.json() } catch (_) { }
    const msg = errBody.error || errBody.detail || `Server error ${res.status}`
    throw new Error(msg)
  }

  return res.json()
}

/**
 * Reload a previously processed run by its run_id.
 * Useful if the page is refreshed after extraction.
 *
 * @param {string} runId
 * @returns {Promise<object>}
 */
export async function getRunResult(runId) {
  const res = await fetch(`${BASE}/results/${runId}/json`, { headers: SKIP_NGROK_WARNING })
  if (!res.ok) throw new Error(`Could not load run ${runId}`)
  return res.json()
}

/**
 * Returns a URL that, when navigated to, downloads a run artifact.
 *
 * @param {string} runId
 * @param {'xlsx'|'json'|'report'|'extraction'|'spans'} kind
 * @returns {string} URL
 */
export function downloadUrl(runId, kind) {
  return `${BASE}/download/${runId}/${kind}`
}

/**
 * Check that the backend is reachable and correctly configured.
 *
 * @returns {Promise<{status: string, fields: number, …}>}
 */
export async function checkHealth() {
  const res = await fetch(`${BASE}/health`, { headers: SKIP_NGROK_WARNING })
  if (!res.ok) throw new Error('Backend health check failed')
  return res.json()
}
