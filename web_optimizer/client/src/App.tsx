import React, { useState, Suspense, lazy, useEffect } from 'react';
import { SimpleLoader } from './components/SkeletonLoader';
import { useAuth } from './contexts/AuthContext';
import { Toaster } from 'react-hot-toast';

// Lazy load components for better performance
const Homepage = lazy(() => import('./components/Homepage'));
const Dashboard = lazy(() => import('./components/Dashboard'));
const LoginPage = lazy(() => import('./components/LoginPage'));
const RegisterPage = lazy(() => import('./components/RegisterPage'));

function App() {
  const { user, logout } = useAuth();
  const [currentView, setCurrentView] = useState<'home' | 'login' | 'register' | 'dashboard'>('home');

  // Auto-navigate based on auth state
  useEffect(() => {
    if (user && (currentView === 'home' || currentView === 'login' || currentView === 'register')) {
      setCurrentView('dashboard');
    } else if (!user && currentView === 'dashboard') {
      setCurrentView('home');
    }
  }, [user, currentView]);

  const handleLogin = () => {
    setCurrentView('login');
  };

  const handleSignUp = () => {
    setCurrentView('register');
  };

  const handleLoginSuccess = () => {
    setCurrentView('dashboard');
  };

  const handleRegisterSuccess = () => {
    setCurrentView('dashboard');
  };

  const handleLogout = async () => {
    await logout();
    setCurrentView('home');
  };

  const handleSwitchToRegister = () => {
    setCurrentView('register');
  };

  const handleSwitchToLogin = () => {
    setCurrentView('login');
  };

  const handleBackToHome = () => {
    setCurrentView('home');
  };

  return (
    <div className="min-h-screen">
      <Toaster position="top-right" />
      <Suspense fallback={<SimpleLoader />}>
        {currentView === 'home' && (
          <Homepage onLogin={handleLogin} onSignUp={handleSignUp} />
        )}
        {currentView === 'login' && (
          <LoginPage 
            onLogin={handleLoginSuccess} 
            onSwitchToRegister={handleSwitchToRegister}
          />
        )}
        {currentView === 'register' && (
          <RegisterPage 
            onRegister={handleRegisterSuccess} 
            onSwitchToLogin={handleSwitchToLogin}
          />
        )}
        {currentView === 'dashboard' && (
          <Dashboard onLogout={handleLogout} />
        )}
      </Suspense>
    </div>
  );
}

export default App;
