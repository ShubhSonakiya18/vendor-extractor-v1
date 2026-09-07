import { useEffect, useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import NavBar from '../components/NavBar'
import Stepper from '../components/Stepper'
import { createVendor, extractionToVendorPayload } from '../api'
import './ConfirmPage.css'

const VENDOR_STEPS = [{ label: 'Upload' }, { label: 'Compare' }, { label: 'Submit' }]

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

export default function VendorConfirmPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const result   = location.state?.result

  const [status, setStatus] = useState('saving')   // saving | saved | duplicate | error
  const [message, setMessage] = useState('')
  const [saved, setSaved] = useState(null)
  const sent = useRef(false)

  useEffect(() => {
    if (!result) { setStatus('error'); setMessage('No vendor data to submit.'); return }
    if (sent.current) return
    sent.current = true

    createVendor(extractionToVendorPayload(result))
      .then(row => { setSaved(row); setStatus('saved') })
      .catch(err => {
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        if (err.status === 409) {
          const d = err.body?.detail || {}
          const id = d.existing_vendor_id
          // Backend message names which identifier clashed (GSTIN or Udyam).
          const what = d.udyam_no ? 'Udyam number' : 'GSTIN'
          setStatus('duplicate')
          setMessage(id
            ? `This ${what} is already registered as vendor #${id}.`
            : (d.message || `A vendor with this ${what} already exists.`))
          return
        }
        setStatus('error')
        setMessage(err.message || 'Could not save this vendor.')
      })
  }, [result, navigate])

  const vendorName = result?.fields?.vendor_name?.value || result?.values?.vendor_name

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">
          <Stepper steps={VENDOR_STEPS} currentStep={3} />

          {status === 'saving' && (
            <div className="success-wrap fade-up">
              <div className="success-icon-outer">
                <div className="success-icon"><span className="btn-spinner" aria-hidden="true" /></div>
              </div>
              <h1 className="success-heading">Saving vendor…</h1>
              <p className="success-sub">Recording the reviewed details.</p>
            </div>
          )}

          {status === 'saved' && (
            <div className="success-wrap fade-up">
              <div className="success-icon-outer">
                <div className="success-icon"><CheckIcon /></div>
              </div>
              <h1 className="success-heading">Vendor saved</h1>
              <p className="success-sub">
                {(vendorName || saved?.vendor_name)
                  ? <><strong>{vendorName || saved.vendor_name}</strong> has been recorded{saved?.id ? ` (ref #${saved.id})` : ''}.</>
                  : 'The vendor has been recorded.'}
                {' '}Pushing to Business Central is a separate step.
              </p>
              <div className="btn-group" role="group" aria-label="Next steps">
                <button className="btn btn-secondary" onClick={() => navigate('/vendor/upload')}>
                  Create another vendor
                </button>
                {saved?.id && (
                  <button className="btn btn-secondary" onClick={() => navigate(`/records/vendor/${saved.id}`)}>
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
                <button className="btn btn-secondary" onClick={() => navigate('/vendor/compare', { state: { result } })}>
                  Back to compare
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
