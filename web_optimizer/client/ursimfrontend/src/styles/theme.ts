// Professional Betting Platform Design System
// Custom branding for UrSim DFS Optimizer

export const theme = {
  // Brand Colors - Professional betting aesthetic
  colors: {
    // Primary Brand Colors
    brand: {
      primary: '#00D9FF', // Vibrant cyan - main brand color
      secondary: '#7C3AED', // Purple - accent
      tertiary: '#F59E0B', // Amber - warnings/alerts
      success: '#10B981', // Green - wins/positive
      danger: '#EF4444', // Red - losses/negative
      warning: '#F59E0B', // Amber
      info: '#3B82F6', // Blue
    },
    
    // Neutral Grays - Professional dark theme
    neutral: {
      50: '#F8FAFC',
      100: '#F1F5F9',
      200: '#E2E8F0',
      300: '#CBD5E1',
      400: '#94A3B8',
      500: '#64748B',
      600: '#475569',
      700: '#334155',
      800: '#1E293B',
      900: '#0F172A',
      950: '#020617',
    },
    
    // Betting Specific Colors
    betting: {
      favoriteRed: '#DC2626',
      underdogGreen: '#16A34A',
      pushYellow: '#FBBF24',
      liveRed: '#EF4444',
      oddsMoveUp: '#10B981',
      oddsMoveDown: '#EF4444',
    },
    
    // Background Gradients
    gradients: {
      primary: 'linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%)',
      secondary: 'linear-gradient(135deg, #7C3AED 0%, #EC4899 100%)',
      success: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
      danger: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
      dark: 'linear-gradient(180deg, #0F172A 0%, #1E293B 100%)',
      card: 'linear-gradient(135deg, #1E293B 0%, #334155 100%)',
    },
  },
  
  // Typography - Professional betting font stack
  typography: {
    fonts: {
      primary: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "'Montserrat', 'Inter', sans-serif",
      mono: "'JetBrains Mono', 'Fira Code', monospace",
    },
    
    sizes: {
      xs: '0.75rem',    // 12px
      sm: '0.875rem',   // 14px
      base: '1rem',     // 16px
      lg: '1.125rem',   // 18px
      xl: '1.25rem',    // 20px
      '2xl': '1.5rem',  // 24px
      '3xl': '1.875rem', // 30px
      '4xl': '2.25rem',  // 36px
      '5xl': '3rem',     // 48px
      '6xl': '3.75rem',  // 60px
    },
    
    weights: {
      light: 300,
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
      extrabold: 800,
    },
  },
  
  // Spacing System
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    '2xl': '3rem',   // 48px
    '3xl': '4rem',   // 64px
    '4xl': '6rem',   // 96px
  },
  
  // Border Radius
  radius: {
    none: '0',
    sm: '0.25rem',   // 4px
    md: '0.375rem',  // 6px
    lg: '0.5rem',    // 8px
    xl: '0.75rem',   // 12px
    '2xl': '1rem',   // 16px
    full: '9999px',
  },
  
  // Shadows - Professional depth
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
    '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
    glow: '0 0 20px rgba(0, 217, 255, 0.3)',
    glowPurple: '0 0 20px rgba(124, 58, 237, 0.3)',
  },
  
  // Transitions
  transitions: {
    fast: '150ms cubic-bezier(0.4, 0, 0.2, 1)',
    base: '250ms cubic-bezier(0.4, 0, 0.2, 1)',
    slow: '350ms cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: '500ms cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
  
  // Breakpoints
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
} as const;

// Odds Display Formats
export const oddsFormats = {
  american: 'american',
  decimal: 'decimal',
  fractional: 'fractional',
} as const;

// Bet Types
export const betTypes = {
  straight: 'Straight Bet',
  parlay: 'Parlay',
  teaser: 'Teaser',
  prop: 'Player Prop',
  future: 'Future',
  live: 'Live Bet',
} as const;

// Security Levels
export const securityLevels = {
  verified: 'Verified Account',
  twoFactor: '2FA Enabled',
  sslSecure: 'SSL Encrypted',
  bankVerified: 'Bank Verified',
} as const;

export type OddsFormat = keyof typeof oddsFormats;
export type BetType = keyof typeof betTypes;
export type SecurityLevel = keyof typeof securityLevels;


