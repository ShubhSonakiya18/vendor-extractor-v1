import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import NavBar from '../components/NavBar'
import {
  getVendorById, getCustomerById, getVendorBcPayload, markVendorPushed,
  getCustomerBcPayload, markCustomerPushed,
  updateVendor, deleteVendor, updateCustomer, deleteCustomer,
} from '../api'
import './RecordsPage.css'

// Field groups per record kind, in display order. Keys match the *Out schema.
const VENDOR_GROUPS = [
  ['Identity', [
    ['vendor_name', 'Vendor Name'], ['company_type', 'Company / Non-Company'],
    ['nature_of_business', 'Nature of Business'],
  ]],
  ['Address', [
    ['address_1', 'Address 1'], ['address_2', 'Address 2'], ['address_3', 'Address 3'],
    ['address_4', 'Address 4'], ['city', 'City'], ['state', 'State'],
    ['country', 'Country'], ['pin_code', 'Pin Code'],
  ]],
  ['Contact', [
    ['telephone_1', 'Telephone 1'], ['telephone_2', 'Telephone 2'],
    ['email', 'E-mail'], ['website', 'Website'],
  ]],
  ['Statutory', [
    ['pan', 'PAN'], ['gst_no', 'GST No.'], ['tan_no', 'TAN No.'],
    ['esic_no', 'ESIC No.'], ['udyam_no', 'Udyam No.'], ['tds_applicable', 'TDS Applicable'],
  ]],
  ['Bank', [
    ['bank_name', 'Bank Name'], ['branch_address', 'Branch Address'],
    ['ifsc_swift_code', 'IFSC / SWIFT'], ['account_type', 'Account Type'],
    ['account_number', 'Account Number'],
  ]],
]

const CUSTOMER_GROUPS = [
  ['Identity', [
    ['company_name', 'Company Name'], ['contact_name', 'Contact Name'], ['type', 'Type'],
  ]],
  ['Address', [
    ['billing_address', 'Billing Address'], ['city', 'City'], ['state', 'State'],
    ['zip_code', 'Zip / Pin Code'], ['country', 'Country'],
  ]],
  ['Contact', [
    ['email_id_to', 'Email ID TO'], ['email_id_cc', 'Email ID CC'], ['phone_number', 'Phone Number'],
  ]],
  ['Statutory', [
    ['gst_registration_number', 'GST Registration No.'], ['pan_number', 'PAN'],
  ]],
  ['Commercial', [
    ['payment_terms', 'Payment Terms'], ['salesperson', 'Salesperson'],
    ['region', 'Region'], ['customer_agreement', 'Customer Agreement'],
  ]],
]

const CFG = {
  vendor: {
    fetch: getVendorById, update: updateVendor, remove: deleteVendor,
    groups: VENDOR_GROUPS, nameKey: 'vendor_name', title: 'Vendor',
    bcFetch: getVendorBcPayload, bcMark: markVendorPushed,
  },
  customer: {
    fetch: getCustomerById, update: updateCustomer, remove: deleteCustomer,
    groups: CUSTOMER_GROUPS, nameKey: 'company_name', title: 'Customer',
    bcFetch: getCustomerBcPayload, bcMark: markCustomerPushed,
  },
}

// The one required field per kind — cannot be blanked while editing.
const REQUIRED = { vendor: 'vendor_name', customer: 'company_name' }
// customer.type is a fixed choice
const TYPE_OPTIONS = ['Services', 'License']

function fmtDate(s) {
  if (!s) return '—'
  const d = new Date(s)
  return isNaN(d) ? s : d.toLocaleString()
}

export default function RecordDetailPage() {
  const navigate = useNavigate()
  const { kind = 'vendor', id } = useParams()
  const cfg = CFG[kind] || CFG.vendor
  const requiredKey = REQUIRED[kind] || REQUIRED.vendor

  const [rec, setRec] = useState(null)
  const [error, setError] = useState('')

  // edit state
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState({})
  const [saving, setSaving] = useState(false)
  const [editErr, setEditErr] = useState('')

  // delete state
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)

  // BC manual-push panel state (vendor and customer)
  const [bc, setBc] = useState(null)
  const [bcErr, setBcErr] = useState('')
  const [bcNoInput, setBcNoInput] = useState('')
  const [marking, setMarking] = useState(false)

  const editableKeys = useMemo(
    () => cfg.groups.flatMap(([, fields]) => fields.map(([k]) => k)),
    [cfg],
  )

  function loadRecord() {
    setRec(null); setError('')
    cfg.fetch(id)
      .then(setRec)
      .catch(err => {
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setError(err.status === 404 ? 'Record not found.' : (err.message || 'Could not load record.'))
      })
  }

  useEffect(() => {
    loadRecord()
    setEditing(false); setEditErr(''); setConfirmDelete(false)
    setBc(null); setBcErr(''); setBcNoInput('')
  }, [kind, id]) // eslint-disable-line react-hooks/exhaustive-deps

  function startEdit() {
    const seed = {}
    editableKeys.forEach(k => { seed[k] = rec[k] ?? '' })
    setForm(seed)
    setEditErr('')
    setEditing(true)
  }

  function saveEdit() {
    if (!String(form[requiredKey] || '').trim()) {
      setEditErr(`${requiredKey.replace(/_/g, ' ')} cannot be empty.`)
      return
    }
    // send only changed fields
    const changes = {}
    editableKeys.forEach(k => {
      const now = form[k] ?? ''
      const was = rec[k] ?? ''
      if (now !== was) changes[k] = now
    })
    if (Object.keys(changes).length === 0) { setEditing(false); return }

    setSaving(true); setEditErr('')
    cfg.update(id, changes)
      .then(updated => { setSaving(false); setEditing(false); setRec(updated) })
      .catch(err => {
        setSaving(false)
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setEditErr(err.status === 409
          ? (err.body?.detail?.message || 'Another record already has this GSTIN.')
          : (err.message || 'Could not save changes.'))
      })
  }

  function doDelete() {
    setDeleting(true)
    cfg.remove(id)
      .then(() => navigate(`/records/${kind}`, { replace: true }))
      .catch(err => {
        setDeleting(false)
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setError(err.message || 'Could not delete this record.')
        setConfirmDelete(false)
      })
  }

  function fetchBcPayload() {
    setBcErr('')
    cfg.bcFetch(id)
      .then(setBc)
      .catch(err => {
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setBcErr(err.status === 503
          ? 'Business Central integration is turned off (BC_ENABLED=false).'
          : (err.message || 'Could not build the BC payload.'))
      })
  }

  function copyPayload() {
    if (bc) navigator.clipboard?.writeText(JSON.stringify(bc, null, 2))
  }

  function downloadPayload() {
    if (!bc) return
    const blob = new Blob(
      [JSON.stringify({ target_url: bc.target_url, method: 'POST', payload: bc.payload }, null, 2)],
      { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${kind}_${id}_bc.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  function submitBcNo() {
    if (!bcNoInput.trim()) return
    setMarking(true); setBcErr('')
    cfg.bcMark(id, bcNoInput.trim())
      .then(() => { setMarking(false); loadRecord(); fetchBcPayload() })
      .catch(err => {
        setMarking(false)
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setBcErr(err.status === 409 ? `This ${cfg.title.toLowerCase()} is already marked as pushed.` : (err.message || 'Could not save.'))
      })
  }

  const isPushed = rec?.bc_status === 'pushed'

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">

          <a className="back-link" onClick={() => navigate(`/records/${kind}`)} style={{ cursor: 'pointer' }}>
            ‹ Saved {cfg.title}s
          </a>

          {error && <p className="records-error">{error}</p>}
          {!rec && !error && <p className="records-empty">Loading…</p>}

          {rec && (
            <>
              <div className="record-header-row">
                <h1 className="page-title" style={{ marginBottom: 0 }}>
                  {rec[cfg.nameKey] || `${cfg.title} #${rec.id}`}
                </h1>
                {!editing && (
                  <div className="record-header-actions">
                    <button className="btn btn-secondary" onClick={startEdit}>Edit</button>
                    <button className="btn btn-danger-outline" onClick={() => setConfirmDelete(true)}>Delete</button>
                  </div>
                )}
              </div>

              <div className="record-meta">
                <span>Ref #{rec.id}</span>
                <span className={`badge badge--${isPushed ? 'success' : 'neutral'}`}>
                  BC: {rec.bc_status || 'not_pushed'}{rec.bc_no ? ` (${rec.bc_no})` : ''}
                </span>
                <span>Created {fmtDate(rec.created_at)}</span>
                {rec.updated_at && rec.updated_at !== rec.created_at && (
                  <span>Updated {fmtDate(rec.updated_at)}</span>
                )}
              </div>

              {editing && isPushed && (
                <p className="records-review-note">
                  This record is already marked as pushed to Business Central. Editing it here
                  does <strong>not</strong> update Business Central — the two will be out of sync.
                </p>
              )}

              {!editing && Array.isArray(rec.fields_needing_review) && rec.fields_needing_review.length > 0 && (
                <p className="records-review-note">
                  Flagged at extraction for review: {rec.fields_needing_review.join(', ')}
                </p>
              )}

              {editErr && <p className="records-error">{editErr}</p>}

              <div className="record-view">
                {cfg.groups.map(([groupName, fields]) => (
                  <section key={groupName} className="record-group">
                    <h2 className="record-group-title">{groupName}</h2>
                    <dl className="record-grid">
                      {fields.map(([key, label]) => (
                        <div key={key} className="record-row">
                          <dt>{label}{editing && key === requiredKey ? ' *' : ''}</dt>
                          <dd>
                            {editing ? (
                              key === 'type' ? (
                                <select
                                  className="form-input"
                                  value={form[key] ?? 'Services'}
                                  onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                                >
                                  {TYPE_OPTIONS.map(o => <option key={o} value={o}>{o}</option>)}
                                </select>
                              ) : (
                                <input
                                  className="form-input"
                                  value={form[key] ?? ''}
                                  onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                                />
                              )
                            ) : (
                              rec[key] ? String(rec[key]) : <span className="val-absent">—</span>
                            )}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </section>
                ))}
              </div>

              {editing && (
                <div className="record-edit-bar">
                  <button className="btn btn-secondary" disabled={saving} onClick={() => setEditing(false)}>
                    Cancel
                  </button>
                  <button className="btn btn-primary" disabled={saving} onClick={saveEdit}>
                    {saving ? 'Saving…' : 'Save changes'}
                  </button>
                </div>
              )}

              {!editing && (
                <section className="record-group bc-panel">
                  <h2 className="record-group-title">Business Central</h2>

                  {isPushed ? (
                    <p className="bc-pushed-note">
                      Pushed to Business Central{rec.bc_no ? ` as ${rec.bc_no}` : ''}
                      {rec.bc_synced_at ? ` on ${fmtDate(rec.bc_synced_at)}` : ''}.
                    </p>
                  ) : (
                    <>
                      <p className="bc-help">
                        The portal cannot reach Business Central directly. Get the payload,
                        run <code>scripts/push_to_bc.ps1</code> on a VPN machine, then record
                        the No. it returns.
                      </p>

                      {!bc && (
                        <button className="btn btn-secondary" onClick={fetchBcPayload}>
                          Get BC payload
                        </button>
                      )}

                      {bcErr && <p className="records-error" style={{ marginTop: 12 }}>{bcErr}</p>}

                      {bc && (
                        <>
                          <div className="bc-payload-actions">
                            <button className="btn btn-secondary" onClick={downloadPayload}>Download JSON</button>
                            <button className="btn btn-secondary" onClick={copyPayload}>Copy</button>
                          </div>
                          <pre className="bc-payload-pre">{JSON.stringify(bc.payload, null, 2)}</pre>
                          <p className="bc-help" style={{ marginTop: 4 }}>
                            POST target: <code>{bc.target_url}</code>
                          </p>

                          <div className="bc-mark-row">
                            <input
                              className="form-input"
                              placeholder="BC No. returned, e.g. EMPV/0123"
                              value={bcNoInput}
                              onChange={e => setBcNoInput(e.target.value)}
                            />
                            <button className="btn btn-primary" disabled={marking || !bcNoInput.trim()} onClick={submitBcNo}>
                              {marking ? 'Saving…' : 'Mark as pushed'}
                            </button>
                          </div>
                        </>
                      )}
                    </>
                  )}
                </section>
              )}

              {confirmDelete && (
                <div className="modal-backdrop" onClick={() => !deleting && setConfirmDelete(false)}>
                  <div className="modal-card" onClick={e => e.stopPropagation()}>
                    <h3>Delete this {cfg.title.toLowerCase()}?</h3>
                    <p>
                      Ref #{rec.id} — {rec[cfg.nameKey] || '(no name)'}. This permanently removes
                      the record from the portal.
                      {isPushed && ' It has already been pushed to Business Central; that record in BC is not affected.'}
                    </p>
                    <div className="modal-actions">
                      <button className="btn btn-secondary" disabled={deleting} onClick={() => setConfirmDelete(false)}>
                        Cancel
                      </button>
                      <button className="btn btn-danger" disabled={deleting} onClick={doDelete}>
                        {deleting ? 'Deleting…' : 'Delete'}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

        </main>
      </div>
    </>
  )
}
