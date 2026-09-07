import { useNavigate } from 'react-router-dom'
import { logout } from '../api'
import './NavBar.css'

const MailIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"
       strokeLinecap="round" strokeLinejoin="round">
    <rect x="2" y="4" width="20" height="16" rx="2"/>
    <polyline points="2,4 12,13 22,4"/>
  </svg>
)

// LoginPage stores the email of whichever hardcoded account was used to sign
// in (there is no real backend session -- see app/routers/auth.py). Turn
// "agamjot@netsmartz.com" into "Agamjot" for display; a name typed via the
// userName prop always wins, and a direct visit with no login (no route
// guard exists) falls back to "Guest".
function deriveDisplayName(email) {
  const local = email.split('@')[0]
  return local.charAt(0).toUpperCase() + local.slice(1)
}

export default function NavBar({ userName }) {
  const navigate = useNavigate()
  const storedEmail = typeof window !== 'undefined' ? localStorage.getItem('userEmail') : null
  const displayName = userName || (storedEmail ? deriveDisplayName(storedEmail) : 'Guest')
  const initials = displayName.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase()

  async function handleLogout() {
    await logout()
    navigate('/', { replace: true })
  }

  return (
    <header className="nav-bar">
      <div className="nav-inner">
        <a className="nav-brand" onClick={() => navigate('/dashboard')} style={{ cursor: 'pointer' }}>
          <div className="nav-brand-icon" aria-hidden="true">
            <MailIcon />
          </div>
          <span className="nav-wordmark">Business Central Portal</span>
        </a>
        <div className="nav-user">
          <div className="nav-avatar" aria-hidden="true">{initials}</div>
          <span className="nav-user-name">{displayName}</span>
          <button type="button" className="nav-logout" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </div>
    </header>
  )
}
