import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { login } from '../api'
import './LoginPage.css'

/* ─── SVG icons ─────────────────────────────────────────── */
const MailIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"
       strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2"/>
    <polyline points="2,4 12,13 22,4"/>
  </svg>
)

const EyeIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
)

const EyeOffIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
)

const CheckCircleIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
       strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
    <polyline points="22 4 12 14.01 9 11.01"/>
  </svg>
)

const LockIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round" width="20" height="20">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
  </svg>
)

const features = [
  'Extract vendor & customer fields instantly',
  'Compare PDF vs Excel data side-by-side',
  'Validate and push directly to Business Central',
  'Audit trail for every submission',
]

export default function LoginPage() {
  const navigate = useNavigate()
  const [email,       setEmail]       = useState('')
  const [password,    setPassword]    = useState('')
  const [showPass,    setShowPass]    = useState(false)
  const [loading,     setLoading]     = useState(false)
  const [authError,   setAuthError]   = useState('')
  const [fieldErrors, setFieldErrors] = useState({ email: '', password: '' })

  function validate() {
    const errs = { email: '', password: '' }
    if (!email.trim())         errs.email    = 'Email is required.'
    else if (!/\S+@\S+\.\S+/.test(email)) errs.email = 'Enter a valid email address.'
    if (!password)             errs.password = 'Password is required.'
    setFieldErrors(errs)
    return !errs.email && !errs.password
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setAuthError('')
    if (!validate()) return

    setLoading(true)
    try {
      // Real email + password auth against the backend /auth/login endpoint.
      // login() stores the JWT (and refresh token) in localStorage; every
      // subsequent API call attaches it via authFetch().
      await login(email.trim(), password)
      navigate('/dashboard')
    } catch (err) {
      setAuthError(err.message || 'Sign-in failed. Please try again.')
      setLoading(false)
    }
  }

  function handleForgotPassword(e) {
    e.preventDefault()
    alert('A password reset link has been sent to your email.')
  }

  return (
    <div className="login-root">
      {/* ── Left panel ── */}
      <div className="login-left" aria-hidden="true">
        <div className="login-left-bg" />
        <div className="login-left-content">
          <div className="login-left-brand">
            <div className="login-left-icon"><MailIcon /></div>
            <span className="login-left-wordmark">Business Central Portal</span>
          </div>
          <div className="login-left-badge">Netsmartz Infotech (India) Pvt · BC220</div>

          <h2 className="login-left-headline">
            Streamline your<br/>Business Central<br/>document intake.
          </h2>
          <p className="login-left-subline">
            One portal to extract, compare, and submit vendor &amp; customer
            data — directly into Business Central.
          </p>

          <ul className="login-left-features">
            {features.map(f => (
              <li key={f}>
                <span className="login-left-check"><CheckCircleIcon /></span>
                {f}
              </li>
            ))}
          </ul>

          {/* Decorative circles */}
          <div className="deco-circle deco-circle-1" />
          <div className="deco-circle deco-circle-2" />
          <div className="deco-circle deco-circle-3" />
        </div>
      </div>

      {/* ── Right panel (form) ── */}
      <div className="login-right">
        <div className="login-form-wrap">

          {/* Mobile-only brand strip */}
          <div className="login-mobile-brand">
            <div className="login-mobile-icon"><MailIcon /></div>
            <span className="login-mobile-wordmark">Business Central Portal</span>
          </div>

          <div className="login-form-header">
            <div className="login-lock-icon"><LockIcon /></div>
            <h1 className="login-form-title">Welcome back</h1>
            <p className="login-form-sub">Sign in to your Business Central Portal account</p>
          </div>

          {/* Global auth error */}
          {authError && (
            <div className="login-auth-error" role="alert">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                   strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              {authError}
            </div>
          )}

          <form id="login-form" noValidate onSubmit={handleSubmit}>

            {/* Email */}
            <div className="form-group">
              <label className="form-label" htmlFor="email">Email</label>
              <div className="input-wrapper">
                <input
                  id="email"
                  type="email"
                  className={`form-input${fieldErrors.email ? ' input-error' : ''}`}
                  placeholder="you@netsmartz.com"
                  autoComplete="username"
                  value={email}
                  onChange={e => { setEmail(e.target.value); setFieldErrors(p => ({...p, email: ''})); setAuthError('') }}
                />
              </div>
              {fieldErrors.email && <p className="form-error">{fieldErrors.email}</p>}
            </div>

            {/* Password */}
            <div className="form-group">
              <label className="form-label" htmlFor="password">Password</label>
              <div className="input-wrapper">
                <input
                  id="password"
                  type={showPass ? 'text' : 'password'}
                  className={`form-input has-toggle${fieldErrors.password ? ' input-error' : ''}`}
                  placeholder="••••••••"
                  autoComplete="current-password"
                  value={password}
                  onChange={e => { setPassword(e.target.value); setFieldErrors(p => ({...p, password: ''})); setAuthError('') }}
                />
                <button type="button" className="toggle-btn"
                        aria-label={showPass ? 'Hide password' : 'Show password'}
                        onClick={() => setShowPass(v => !v)}>
                  {showPass ? <EyeOffIcon /> : <EyeIcon />}
                </button>
              </div>
              {fieldErrors.password && <p className="form-error">{fieldErrors.password}</p>}
            </div>

            {/* Submit */}
            <button type="submit" className="btn btn-primary btn-full login-submit-btn"
                    id="login-btn" disabled={loading}
                    aria-label="Log in to Business Central Portal">
              {loading && <span className="btn-spinner" aria-hidden="true" />}
              <span>{loading ? 'Signing in…' : 'Log in'}</span>
            </button>

          </form>

          <p className="login-forgot">
            Forgot your password?&nbsp;
            <a href="#" id="reset-link" onClick={handleForgotPassword}>Reset it</a>
          </p>

        </div>
      </div>
    </div>
  )
}
