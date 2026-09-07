import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import NavBar from '../components/NavBar'
import { listVendors, listCustomers } from '../api'
import './RecordsPage.css'

const KIND = {
  vendor:   { label: 'Vendors',   fetch: listVendors,   nameKey: 'vendor_name',  gstKey: 'gst_no' },
  customer: { label: 'Customers', fetch: listCustomers, nameKey: 'company_name', gstKey: 'gst_registration_number' },
}

function fmtDate(s) {
  if (!s) return '—'
  const d = new Date(s)
  return isNaN(d) ? s : d.toLocaleString()
}

export default function RecordsPage() {
  const navigate = useNavigate()
  const { kind = 'vendor' } = useParams()
  const cfg = KIND[kind] || KIND.vendor

  const [rows, setRows] = useState(null)   // null = loading
  const [error, setError] = useState('')

  useEffect(() => {
    setRows(null); setError('')
    cfg.fetch()
      .then(setRows)
      .catch(err => {
        if (err.code === 'AUTH_EXPIRED') { navigate('/', { replace: true }); return }
        setError(err.message || 'Could not load records.')
        setRows([])
      })
  }, [kind]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">

          <a className="back-link" onClick={() => navigate('/dashboard')} style={{ cursor: 'pointer' }}>
            ‹ Dashboard
          </a>

          <h1 className="page-title">Saved Records</h1>

          <div className="records-tabs" role="tablist">
            {Object.entries(KIND).map(([k, v]) => (
              <button
                key={k}
                role="tab"
                aria-selected={k === kind}
                className={`records-tab${k === kind ? ' is-active' : ''}`}
                onClick={() => navigate(`/records/${k}`)}
              >
                {v.label}
              </button>
            ))}
          </div>

          {error && <p className="records-error">{error}</p>}

          {rows === null ? (
            <p className="records-empty">Loading…</p>
          ) : rows.length === 0 && !error ? (
            <p className="records-empty">No {cfg.label.toLowerCase()} saved yet.</p>
          ) : (
            <div className="table-wrapper">
              <table className="compare-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Name</th>
                    <th>GSTIN</th>
                    <th>BC Status</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(r => (
                    <tr
                      key={r.id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => navigate(`/records/${kind}/${r.id}`)}
                    >
                      <td>{r.id}</td>
                      <td>{r[cfg.nameKey] || '—'}</td>
                      <td>{r[cfg.gstKey] || '—'}</td>
                      <td>
                        <span className={`badge badge--${r.bc_status === 'pushed' ? 'success' : 'neutral'}`}>
                          {r.bc_status || 'not_pushed'}
                        </span>
                      </td>
                      <td>{fmtDate(r.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

        </main>
      </div>
    </>
  )
}
