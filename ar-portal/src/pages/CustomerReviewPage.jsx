import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import NavBar from '../components/NavBar'
import Stepper from '../components/Stepper'
import './CustomerReviewPage.css'

const CUSTOMER_STEPS = [{ label: 'Upload' }, { label: 'Review' }, { label: 'Submit' }]

const DocsIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/>
    <line x1="16" y1="17" x2="8" y2="17"/>
    <polyline points="10 9 9 9 8 9"/>
  </svg>
)

const AlertIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
)

/**
 * Normalises response from either /onboarding/extract or /extract into a
 * consistent customer dictionary.
 */
function extractCustomerValues(result) {
  if (!result) return {}

  // Direct schema from /onboarding/extract
  if (result.company_name !== undefined || result.billing_address !== undefined) {
    return {
      company_name:            result.company_name ?? '',
      contact_name:            result.contact_name ?? '',
      billing_address:         result.billing_address ?? '',
      city:                    result.city ?? '',
      state:                   result.state ?? '',
      zip_code:                result.zip_code ?? '',
      country:                 result.country ?? '',
      gst_registration_number: result.gst_registration_number ?? '',
      pan_number:              result.pan_number ?? '',
      email_id_to:             result.email_id_to ?? '',
      email_id_cc:             result.email_id_cc ?? '',
      phone_number:            result.phone_number ?? '',
      payment_terms:           result.payment_terms ?? '',
      salesperson:             result.salesperson ?? '',
      region:                  result.region ?? '',
      customer_agreement:      result.customer_agreement ?? '',
      type:                    result.type || 'Services',
    }
  }

  // Schema from standard /extract
  const vals = result.values || {}
  const fields = result.fields || {}
  const getVal = key => vals[key] || fields[key]?.value || ''

  return {
    company_name:            getVal('vendor_name') || getVal('company_name'),
    contact_name:            getVal('contact_name'),
    billing_address:         [getVal('address_1'), getVal('address_2'), getVal('address_3')].filter(Boolean).join(', ') || getVal('billing_address'),
    city:                    getVal('city'),
    state:                   getVal('state'),
    zip_code:                getVal('pin_code') || getVal('zip_code'),
    country:                 getVal('country'),
    gst_registration_number: getVal('gst_number') || getVal('gst_registration_number'),
    pan_number:              getVal('pan') || getVal('pan_number'),
    email_id_to:             getVal('email') || getVal('email_id_to'),
    email_id_cc:             getVal('email_id_cc'),
    phone_number:            getVal('telephone') || getVal('phone_number'),
    payment_terms:           getVal('payment_terms'),
    salesperson:             getVal('salesperson'),
    region:                  getVal('region'),
    customer_agreement:      getVal('customer_agreement'),
    type:                    getVal('type') || 'Services',
  }
}

/**
 * The backend (onboarding_mapper.to_onboarding_schema) returns
 * `fields_needing_review` -- onboarding field names whose value was inferred
 * rather than read straight off a labelled field: a PAN derived from the
 * GSTIN, or address_1/city/state/zip_code split out of a single run-on
 * address line by the address resolver. Those inputs are highlighted and
 * badged so the reviewer checks them before submitting.
 */
function reviewSet(result) {
  const list = Array.isArray(result?.fields_needing_review)
    ? result.fields_needing_review
    : []
  // The engine reports "pin_code"; this form's field is "zip_code".
  return new Set(list.map(f => (f === 'pin_code' ? 'zip_code' : f)))
}

// Human-readable names for the review banner.
const FIELD_TITLES = {
  company_name: 'Company Name',
  billing_address: 'Billing Address',
  city: 'City',
  state: 'State',
  zip_code: 'Zip / Pin code',
  country: 'Country',
  gst_registration_number: 'GST Registration Number',
  pan_number: 'PAN Number',
  email_id_to: 'Email ID TO',
  phone_number: 'Phone Number',
}

// Every editable field, in display order. `full` spans both grid columns.
const FIELDS = [
  { key: 'company_name',            label: 'Company Name' },
  { key: 'contact_name',            label: 'Contact Name' },
  { key: 'billing_address',         label: 'Billing Address', full: true },
  { key: 'city',                    label: 'City' },
  { key: 'state',                   label: 'State' },
  { key: 'zip_code',                label: 'Zip code / Pin code' },
  { key: 'country',                 label: 'Country' },
  { key: 'gst_registration_number', label: 'GST(ABN,TRN) Registration Certificate' },
  { key: 'pan_number',              label: 'PAN Card (Company/Individual)' },
  { key: 'email_id_to',             label: 'Email ID TO', type: 'email' },
  { key: 'email_id_cc',             label: 'Email ID CC', type: 'email' },
  { key: 'phone_number',            label: 'Phone Number' },
  { key: 'payment_terms',           label: 'Payment Terms' },
  { key: 'salesperson',             label: 'SALESPERSON' },
  { key: 'region',                  label: 'REGION' },
  { key: 'customer_agreement',      label: 'Customer Agreement / Contract / Purchase Order / Sale Order', full: true },
  { key: 'type',                    label: 'Type', type: 'select', options: ['Services', 'License'] },
]

export default function CustomerReviewPage() {
  const navigate  = useNavigate()
  const location  = useLocation()
  const [loading, setLoading] = useState(false)

  const result = location.state?.result
  const [formData, setFormData] = useState(() => extractCustomerValues(result))
  const [review, setReview] = useState(() => reviewSet(result))

  if (!result) {
    return (
      <>
        <NavBar />
        <div className="page-wrapper">
          <main className="page-content">
            <p style={{ color: 'var(--color-text-muted)', marginTop: 40 }}>
              No extraction data found.{' '}
              <a className="back-link" onClick={() => navigate('/customer/upload')} style={{ cursor: 'pointer', display: 'inline' }}>
                Go back to Upload
              </a>
            </p>
          </main>
        </div>
      </>
    )
  }

  const sourceDocs = result.source_documents?.map(d => d.file_name) ||
                     (Array.isArray(result.documents) ? result.documents.map(d => typeof d === 'string' ? d : d.document) : [])

  const reviewList = [...review].filter(k => k in FIELD_TITLES || FIELDS.some(f => f.key === k))

  // Fields the extractor actually produced a value for (excludes the manual
  // business fields like salesperson/region that never come from a document).
  const filledCount = FIELDS.filter(f => f.key !== 'type' && String(formData[f.key] ?? '').trim()).length

  function handleChange(field, value) {
    setFormData(prev => ({ ...prev, [field]: value }))
    // Editing a flagged field clears its flag -- the reviewer has now seen it.
    if (review.has(field)) {
      setReview(prev => {
        const next = new Set(prev)
        next.delete(field)
        return next
      })
    }
  }

  function handleSubmit() {
    setLoading(true)
    setTimeout(() => navigate('/customer/confirm', { state: { result: { ...result, customerData: formData } } }), 800)
  }

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">

          <a className="back-link" onClick={() => navigate('/customer/upload')} style={{ cursor: 'pointer' }}>
            ‹ Upload
          </a>

          <h1 className="page-title">Customer Creation</h1>

          {result.timings && (
            <p style={{ fontSize: '0.8rem', color: 'var(--color-text-subtle)', marginBottom: 4 }}>
              Extracted {filledCount} field{filledCount === 1 ? '' : 's'} in {result.timings.total}s
            </p>
          )}

          <Stepper steps={CUSTOMER_STEPS} currentStep={1} />

          {reviewList.length > 0 && (
            <div className="review-banner" role="alert">
              <div className="review-banner-icon" aria-hidden="true"><AlertIcon /></div>
              <div>
                <div className="review-banner-title">
                  {reviewList.length} field{reviewList.length > 1 ? 's' : ''} need{reviewList.length > 1 ? '' : 's'} a check
                </div>
                <div className="review-banner-sub">
                  These were inferred (e.g. split out of a single address line, or a PAN
                  derived from the GSTIN) rather than read from a labelled field. Confirm
                  or correct each, then submit — editing one clears its flag.
                </div>
                <div className="review-banner-chips">
                  {reviewList.map(k => (
                    <span key={k} className="review-chip">{FIELD_TITLES[k] || k}</span>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="review-card" role="region" aria-label="Extracted Customer Fields">

            <div className="fields-grid">
              {FIELDS.map(f => {
                const flagged = review.has(f.key)
                const inputClass = 'form-input' + (flagged ? ' input-low-confidence' : '')
                return (
                  <div
                    key={f.key}
                    className={'form-group' + (f.full ? ' field-span-full' : '')}
                  >
                    <div className="field-header">
                      <label className="field-label" htmlFor={f.key}>{f.label}</label>
                      {flagged && (
                        <span className="conf-badge conf-low" title="Inferred value — please verify">
                          Review
                        </span>
                      )}
                    </div>

                    {f.type === 'select' ? (
                      <select
                        id={f.key}
                        className={inputClass}
                        style={{ cursor: 'pointer', background: 'var(--color-surface)' }}
                        value={formData[f.key]}
                        onChange={e => handleChange(f.key, e.target.value)}
                      >
                        {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                    ) : (
                      <input
                        id={f.key}
                        type={f.type || 'text'}
                        className={inputClass}
                        aria-invalid={flagged || undefined}
                        value={formData[f.key]}
                        onChange={e => handleChange(f.key, e.target.value)}
                      />
                    )}
                  </div>
                )
              })}
            </div>

            {sourceDocs.length > 0 && (
              <div className="attached-docs-card">
                <div className="attached-docs-info">
                  <div className="docs-icon" aria-hidden="true"><DocsIcon /></div>
                  <div>
                    <div className="docs-title">Documents Required &amp; Attached</div>
                    <div className="docs-sub">{sourceDocs.join(' · ')}</div>
                  </div>
                </div>
                <span className="docs-pill">{sourceDocs.length} Document{sourceDocs.length > 1 ? 's' : ''} Verified</span>
              </div>
            )}

          </div>

          <div className="action-bar" style={{ marginTop: 32 }}>
            <button
              type="button"
              className="btn btn-primary"
              id="validate-btn"
              disabled={loading}
              onClick={handleSubmit}
              aria-label="Validate and submit customer to Business Central"
            >
              {loading && <span className="btn-spinner" aria-hidden="true" />}
              <span>Validate &amp; Submit →</span>
            </button>
          </div>

        </main>
      </div>
    </>
  )
}
