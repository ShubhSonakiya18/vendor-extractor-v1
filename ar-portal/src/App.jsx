import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import VendorUploadPage from './pages/VendorUploadPage'
import VendorComparePage from './pages/VendorComparePage'
import VendorConfirmPage from './pages/VendorConfirmPage'
import CustomerUploadPage from './pages/CustomerUploadPage'
import CustomerReviewPage from './pages/CustomerReviewPage'
import CustomerConfirmPage from './pages/CustomerConfirmPage'
import RecordsPage from './pages/RecordsPage'
import RecordDetailPage from './pages/RecordDetailPage'
import { isLoggedIn } from './api'

// Normal auth: an unauthenticated visit to any app route redirects to the
// login page. authFetch() also clears the token on a 401, so an expired
// session lands back here on the next guarded navigation.
function RequireAuth({ children }) {
  return isLoggedIn() ? children : <Navigate to="/" replace />
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                    element={<LoginPage />} />
        <Route path="/dashboard"           element={<RequireAuth><DashboardPage /></RequireAuth>} />
        <Route path="/vendor/upload"       element={<RequireAuth><VendorUploadPage /></RequireAuth>} />
        <Route path="/vendor/compare"      element={<RequireAuth><VendorComparePage /></RequireAuth>} />
        <Route path="/vendor/confirm"      element={<RequireAuth><VendorConfirmPage /></RequireAuth>} />
        <Route path="/customer/upload"     element={<RequireAuth><CustomerUploadPage /></RequireAuth>} />
        <Route path="/customer/review"     element={<RequireAuth><CustomerReviewPage /></RequireAuth>} />
        <Route path="/customer/confirm"    element={<RequireAuth><CustomerConfirmPage /></RequireAuth>} />
        <Route path="/records"             element={<RequireAuth><RecordsPage /></RequireAuth>} />
        <Route path="/records/:kind"       element={<RequireAuth><RecordsPage /></RequireAuth>} />
        <Route path="/records/:kind/:id"   element={<RequireAuth><RecordDetailPage /></RequireAuth>} />
        <Route path="*"                    element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
