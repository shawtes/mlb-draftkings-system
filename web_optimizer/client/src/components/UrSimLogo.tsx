import React from 'react';

interface UrSimLogoProps {
  size?: number;
  className?: string;
}

/**
 * UrSim brand emblem — stylized "U" shield with simulation wave motif.
 * Represents quantitative simulation (wave) + protection (shield) + upside (arrow).
 */
const UrSimLogo: React.FC<UrSimLogoProps> = ({ size = 28, className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 40 40"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
  >
    {/* Shield body */}
    <path
      d="M20 2L4 8v12c0 10.5 6.8 17.2 16 20 9.2-2.8 16-9.5 16-20V8L20 2Z"
      fill="url(#shield-gradient)"
      stroke="url(#border-gradient)"
      strokeWidth="1.5"
    />
    {/* Inner shield highlight */}
    <path
      d="M20 5L7 10v10c0 8.8 5.6 14.4 13 16.8 7.4-2.4 13-8 13-16.8V10L20 5Z"
      fill="url(#inner-gradient)"
      opacity="0.6"
    />
    {/* Simulation wave — represents Monte Carlo / probability distribution */}
    <path
      d="M10 22c2-4 4-6 6-2s4 6 6 0 4-8 6-4"
      stroke="url(#wave-gradient)"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
    />
    {/* Upward arrow — represents optimization / upside */}
    <path
      d="M20 10v8M16.5 13.5L20 10l3.5 3.5"
      stroke="#00FFD4"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <defs>
      <linearGradient id="shield-gradient" x1="20" y1="2" x2="20" y2="40" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#0F2027" />
        <stop offset="0.5" stopColor="#0A1628" />
        <stop offset="1" stopColor="#060D1A" />
      </linearGradient>
      <linearGradient id="border-gradient" x1="4" y1="2" x2="36" y2="40" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#00D9FF" />
        <stop offset="0.5" stopColor="#00FFD4" />
        <stop offset="1" stopColor="#7C3AED" />
      </linearGradient>
      <linearGradient id="inner-gradient" x1="20" y1="5" x2="20" y2="32" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#00D9FF" stopOpacity="0.08" />
        <stop offset="1" stopColor="#7C3AED" stopOpacity="0.04" />
      </linearGradient>
      <linearGradient id="wave-gradient" x1="10" y1="20" x2="28" y2="20" gradientUnits="userSpaceOnUse">
        <stop offset="0" stopColor="#00D9FF" />
        <stop offset="0.6" stopColor="#00FFD4" />
        <stop offset="1" stopColor="#7C3AED" />
      </linearGradient>
    </defs>
  </svg>
);

export default UrSimLogo;
