import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import TestCheckbox from './TestCheckbox';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const showTest = window.location.pathname === '/test' || window.location.hash.includes('test') || window.location.search.includes('test');

root.render(
  <React.StrictMode>
    {showTest ? <TestCheckbox /> : <App />}
  </React.StrictMode>
);
