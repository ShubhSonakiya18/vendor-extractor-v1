import { useNavigate } from 'react-router-dom'
import NavBar from '../components/NavBar'
import './DashboardPage.css'

const VendorIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
    <polyline points="9,22 9,12 15,12 15,22"/>
  </svg>
)

const CustomerIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round">
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
    <circle cx="9" cy="7" r="4"/>
    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
    <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
  </svg>
)

const ChevronRight = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
       strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9,18 15,12 9,6"/>
  </svg>
)

const ListIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
       strokeLinecap="round" strokeLinejoin="round">
    <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>
    <line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>
    <line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
  </svg>
)

const ACTION_CARDS = [
  {
    id:     'card-vendor',
    title:  'Vendor Creation',
    desc:   'Upload a Vendor Registration Form (PDF + Excel) to onboard a new vendor.',
    icon:   <VendorIcon />,
    route:  '/vendor/upload',
  },
  {
    id:     'card-customer',
    title:  'Customer Creation',
    desc:   'Upload a customer document to extract and review before onboarding.',
    icon:   <CustomerIcon />,
    route:  '/customer/upload',
  },
  {
    id:     'card-records',
    title:  'Saved Records',
    desc:   'View vendors and customers already saved in the portal.',
    icon:   <ListIcon />,
    route:  '/records/vendor',
  },
]

export default function DashboardPage() {
  const navigate = useNavigate()

  return (
    <>
      <NavBar />
      <div className="page-wrapper">
        <main className="page-content">

          <div className="greeting-section fade-up">
            <h1 className="greeting-heading">Welcome to Vendor &amp; Customer Onboarding</h1>
            <p className="greeting-sub">Choose what you'd like to create today.</p>
          </div>

          <div className="action-grid fade-up" style={{ animationDelay: '0.08s' }}>
            {ACTION_CARDS.map(card => (
              <div
                key={card.id}
                id={card.id}
                className="action-card"
                role="button"
                tabIndex={0}
                aria-label={`Start ${card.title} flow`}
                onClick={() => navigate(card.route)}
                onKeyDown={e => e.key === 'Enter' && navigate(card.route)}
              >
                <div className="action-card-icon">{card.icon}</div>
                <p className="action-card-title">{card.title}</p>
                <p className="action-card-desc">{card.desc}</p>
                <span className="action-card-cta" aria-hidden="true">
                  Get started <ChevronRight />
                </span>
              </div>
            ))}
          </div>

        </main>
      </div>
    </>
  )
}
