import React from 'react';
import LandingIntro from './components/LandingIntro';
import Dashboard from './pages/Dashboard';
import './App.css';

export default function App() {
  return (
    <div className="app-root">
      <LandingIntro />
      <Dashboard />
    </div>
  );
}
