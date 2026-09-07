import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import NavBar from '../components/NavBar'
import Stepper from '../components/Stepper'
import FileDropzone from '../components/FileDropzone'
import { extractOnboardingDocuments } from '../api'

const CUSTOMER_STEPS = [
  { label: 'Upload' },
  { label: 'Review' },
  { label: 'Submit' },
]

export default function CustomerUploadPage() {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const ready = files.length > 0

  async function handleExtract() {
    if (!ready) return
    setLoading(true)
    setError('')

    try {
      // Every file goes as a "document" -- no Excel template and no
      // document-type field for the customer flow. Each upload is OCR'd and
      // classified from its own content (cancelled cheque, GST certificate,
      // Udyam certificate, PAN card, ...) by the backend; the caller never
      // says which file is which. See onboarding_mapper.py / routers/onboarding.py.
      const docFiles = files.map(f => f.fileObject)
      const result = await extractOnboardingDocuments(docFiles)

      navigate('/customer/review', { state: { result } })
    } catch (err) {
      setError(err.message || 'Extraction failed. Is the backend running?')
      setLoading(false)
    }
  }

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">

          <a className="back-link" onClick={() => navigate('/dashboard')} style={{ cursor: 'pointer' }}>
            ‹ Dashboard
          </a>

          <h1 className="page-title">Customer Creation</h1>
          <Stepper steps={CUSTOMER_STEPS} currentStep={0} />

          {/* API error banner */}
          {error && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: '#FEF2F2', border: '1px solid #FECACA',
              borderRadius: 10, padding: '11px 14px',
              fontSize: '0.85rem', color: '#991B1B', fontWeight: 500,
              marginBottom: 20,
            }} role="alert">
              ⚠ {error}
            </div>
          )}

          <FileDropzone
            files={files}
            setFiles={setFiles}
            accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp"
          />

          <p className="helper-text" style={{ marginTop: 16 }}>
            Upload the customer's documents as PDFs or images — <strong>GST certificate</strong>,{' '}
            <strong>cancelled cheque</strong>, <strong>Udyam certificate</strong>,{' '}
            <strong>PAN card</strong>, whatever you have. There's no document-type field to fill
            in; each file is read and matched to its type automatically.
          </p>

          <div className="action-bar">
            <div className="action-bar-inner">
              <span
                className={`action-hint${ready ? ' ready' : ''}`}
                aria-live="polite"
              >
                {ready
                  ? `${files.length} file${files.length > 1 ? 's' : ''} ready — click to extract fields.`
                  : 'Upload at least one file to continue.'}
              </span>
              <button
                type="button"
                className="btn btn-primary"
                id="extract-btn"
                disabled={!ready || loading}
                onClick={handleExtract}
                aria-label="Extract fields from uploaded documents"
              >
                {loading && <span className="btn-spinner" aria-hidden="true" />}
                <span>{loading ? 'Extracting…' : 'Extract Fields →'}</span>
              </button>
            </div>
          </div>

        </main>
      </div>
    </>
  )
}
