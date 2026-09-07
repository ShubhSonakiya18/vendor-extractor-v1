import { useEffect, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import NavBar from '../components/NavBar'
import Stepper from '../components/Stepper'
import { createCustomer } from '../api'
import './ConfirmPage.css'

const CUSTOMER_STEPS = [{ label: 'Upload' }, { label: 'Review' }, { label: 'Submit' }]

const CheckIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
       strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20,6 9,17 4,12"/>
  </svg>
)

const AlertIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
       strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="8" x2="12" y2="12"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
)

// Fields the /customers endpoint accepts, in the shape it expects. Anything
// not extracted / not entered goes as "".
function toCustomerPayload(result) {
  const d = result?.customerData || {}
  const pick = k => d[k] ?? ''
  return {
    company_name:            pick('company_name'),
    contact_name:            pick('contact_name'),
    billing_address:         pick('billing_address'),
    city:                    pick('city'),
    state:                   pick('state'),
    zip_code:                pick('zip_code'),
    country:                 pick('country'),
    gst_registration_number: pick('gst_registration_number'),
    pan_number:              pick('pan_number'),
    email_id_to:             pick('email_id_to'),
    email_id_cc:             pick('email_id_cc'),
    phone_number:            pick('phone_number'),
    payment_terms:           pick('payment_terms'),
    salesperson:             pick('salesperson'),
    region:                  pick('region'),
    customer_agreement:      pick('customer_agreement'),
    type:                    d.type || 'Services',
    raw_extraction:          result?.rawExtraction ?? result ?? null,
    source_documents:        result?.source_documents ?? null,
    fields_needing_review:   result?.fields_needing_review ?? null,
  }
}

export default function CustomerConfirmPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const result   = location.state?.result

  // idle → saving → saved | duplicate | error
  const [status, setStatus] = useState('saving')
  const [message, setMessage] = useState('')
  const [saved, setSaved] = useState(null)
  const sent = useRef(false)

  useEffect(() => {
    if (!result) { setStatus('error'); setMessage('No customer data to submit.'); return }
    if (sent.current) return          // guard React 18 StrictMode double-invoke
    sent.current = true

    createCustomer(toCustomerPayload(result))
      .then(row => { setSaved(row); setStatus('saved') })
      .catch(err => {
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        if (err.status === 409) {
          const id = err.body?.detail?.existing_customer_id
          setStatus('duplicate')
          setMessage(id
            ? `This GSTIN is already registered as customer #${id}.`
            : 'A customer with this GSTIN already exists.')
          return
        }
        setStatus('error')
        setMessage(err.message || 'Could not save this customer.')
      })
  }, [result, navigate])

  const customerName = result?.customerData?.company_name

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">
          <Stepper steps={CUSTOMER_STEPS} currentStep={3} />

          {status === 'saving' && (
            <div className="success-wrap fade-up">
              <div className="success-icon-outer">
                <div className="success-icon"><span className="btn-spinner" aria-hidden="true" /></div>
              </div>
              <h1 className="success-heading">Saving customer…</h1>
              <p className="success-sub">Recording the reviewed details.</p>
            </div>
          )}

          {status === 'saved' && (
            <div className="success-wrap fade-up">
              <div className="success-icon-outer">
                <div className="success-icon"><CheckIcon /></div>
              </div>
              <h1 className="success-heading">Customer saved</h1>
              <p className="success-sub">
                {(customerName || saved?.company_name)
                  ? <><strong>{customerName || saved.company_name}</strong> has been recorded{saved?.id ? ` (ref #${saved.id})` : ''}.</>
                  : 'The customer has been recorded.'}
                {' '}Pushing to Business Central is a separate step.
              </p>
              <div className="btn-group" role="group" aria-label="Next steps">
                <button className="btn btn-secondary" onClick={() => navigate('/customer/upload')}>
                  Create another customer
                </button>
                {saved?.id && (
                  <button className="btn btn-secondary" onClick={() => navigate(`/records/customer/${saved.id}`)}>
                    View saved record
                  </button>
                )}
                <button className="btn btn-primary" onClick={() => navigate('/dashboard')}>
                  Back to Dashboard
                </button>
              </div>
            </div>
          )}

          {(status === 'duplicate' || status === 'error') && (
            <div className="success-wrap fade-up">
              <div className="success-icon-outer">
                <div className="success-icon" style={{ background: 'var(--color-danger, #dc2626)' }}>
                  <AlertIcon />
                </div>
              </div>
              <h1 className="success-heading">
                {status === 'duplicate' ? 'Already registered' : 'Could not save'}
              </h1>
              <p className="success-sub">{message}</p>
              <div className="btn-group" role="group" aria-label="Next steps">
                <button className="btn btn-secondary" onClick={() => navigate('/customer/review', { state: { result } })}>
                  Back to review
                </button>
                <button className="btn btn-primary" onClick={() => navigate('/dashboard')}>
                  Back to Dashboard
                </button>
              </div>
            </div>
          )}

        </main>
      </div>
    </>
  )
}
