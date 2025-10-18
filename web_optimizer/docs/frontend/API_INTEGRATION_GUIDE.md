# Backend API Integration Guide

## Overview

This frontend is **designed to work with your backend API**. All calculations (odds, payouts, Kelly Criterion, ROI, etc.) are handled server-side. The frontend focuses on beautiful UI/UX and data display.

---

## 🎯 Architecture

### Frontend Responsibilities
- ✅ Display formatted data
- ✅ Handle user interactions
- ✅ Send requests to backend API
- ✅ Format numbers/currency/dates for display
- ✅ Show loading/error states
- ✅ Manage UI state

### Backend Responsibilities
- ✅ All betting calculations (odds, payouts, parlays)
- ✅ Kelly Criterion recommendations
- ✅ ROI and profit/loss calculations
- ✅ Edge/value detection
- ✅ Player projections and analytics
- ✅ Risk management calculations
- ✅ User data persistence
- ✅ Payment processing
- ✅ Authentication token verification

---

## 📡 API Endpoints Needed

### Authentication
```
POST /api/auth/login
POST /api/auth/register
POST /api/auth/logout
GET  /api/auth/me
POST /api/auth/refresh
```

### Betting Slip
```
POST /api/bets/calculate-payout
  Body: { selections: BetSelection[], betType: 'straight' | 'parlay', stake: number }
  Returns: { totalOdds, potentialPayout, profit, probability, kellyRecommendation }

POST /api/bets/place
  Body: { selections, betType, stake, oddsFormat }
  Returns: { betId, status, confirmation }
```

### Odds & Props
```
GET /api/odds/props?sport={sport}&date={date}
  Returns: Array<{ player, team, prop, line, overOdds, underOdds, edge, confidence }>

GET /api/odds/games?sport={sport}&slate={slate}
  Returns: Array<{ id, home, away, time, spread, total, status }>

GET /api/odds/convert?odds={odds}&toFormat={format}
  Returns: { american, decimal, fractional, impliedProbability }
```

### DFS Tools
```
GET /api/dfs/players?sport={sport}&slate={slate}&filters={filters}
  Returns: Array<{ id, name, position, team, salary, projection, status }>

POST /api/dfs/generate-lineups
  Body: { sport, slate, count, rules, lockedPlayers, excludedPlayers }
  Returns: Array<{ lineup, totalSalary, projectedPoints }>
```

### Bankroll Management
```
GET /api/bankroll/summary
  Returns: { currentBankroll, initialBankroll, totalProfit, roi, winRate }

GET /api/bankroll/transactions?limit={limit}&offset={offset}
  Returns: Array<{ id, date, type, amount, description, balance }>

GET /api/bankroll/charts
  Returns: { balanceHistory, sportAllocation, weeklyPerformance }

PUT /api/bankroll/limits
  Body: { dailyLimit, weeklyLimit }
  Returns: { success, limits }
```

### Account Management
```
GET  /api/account/profile
PUT  /api/account/profile
POST /api/account/change-password
GET  /api/account/billing
GET  /api/account/invoices
PUT  /api/account/preferences
```

### Subscription & Pricing
```
POST /api/subscription/create-checkout
  Body: { planId, billingCycle }
  Returns: { checkoutUrl }

POST /api/subscription/cancel
GET  /api/subscription/current
POST /api/subscription/update
```

---

## 🔧 How to Integrate

### Step 1: Create API Client

```typescript
// src/services/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('authToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

### Step 2: Create Service Functions

```typescript
// src/services/bettingService.ts
import api from './api';

export const bettingService = {
  calculatePayout: async (selections, betType, stake) => {
    const response = await api.post('/bets/calculate-payout', {
      selections,
      betType,
      stake,
    });
    return response.data;
  },

  placeBet: async (betData) => {
    const response = await api.post('/bets/place', betData);
    return response.data;
  },

  getProps: async (sport, filters = {}) => {
    const response = await api.get('/odds/props', {
      params: { sport, ...filters },
    });
    return response.data;
  },
};
```

### Step 3: Update Components

```typescript
// Example: BettingSlip.tsx
import { useEffect, useState } from 'react';
import { bettingService } from '../services/bettingService';

export default function BettingSlip() {
  const [payoutData, setPayoutData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Replace mock calculations with API call
  useEffect(() => {
    if (selections.length > 0 && stake > 0) {
      setLoading(true);
      bettingService
        .calculatePayout(selections, betType, stake)
        .then(setPayoutData)
        .finally(() => setLoading(false));
    }
  }, [selections, betType, stake]);

  // Use payoutData from backend instead of mock values
  const potentialPayout = payoutData?.potentialPayout || 0;
  const totalOdds = payoutData?.totalOdds || '';
  const kellyRecommendation = payoutData?.kellyRecommendation;
}
```

---

## 📦 Data Models

### BetSelection
```typescript
interface BetSelection {
  id: string;
  player: string;
  team: string;
  prop: string;
  line: number;
  odds: number;
  type: 'over' | 'under';
}
```

### PayoutResponse
```typescript
interface PayoutResponse {
  totalOdds: string;        // e.g., "+265"
  potentialPayout: number;  // e.g., 265.00
  profit: number;          // e.g., 165.00
  probability: number;     // e.g., 37.74
  kellyRecommendation?: {
    recommendedStake: number;
    fractionalStake: number;
    kellyPercentage: number;
  };
}
```

### PropBet
```typescript
interface PropBet {
  id: string;
  player: string;
  team: string;
  position: string;
  prop: string;           // "Passing Yards"
  line: number;          // 287.5
  overOdds: number;      // -110
  underOdds: number;     // -110
  projection: number;    // 312.4
  edge: number;         // 8.2
  confidence: number;   // 95
  hitRate: number;      // 68
}
```

---

## 🔐 Authentication Flow

```typescript
// Login example
const handleLogin = async (email, password) => {
  try {
    const response = await api.post('/auth/login', { email, password });
    const { token, user } = response.data;
    
    // Store token
    localStorage.setItem('authToken', token);
    
    // Update auth context
    setUser(user);
    
    // Navigate to dashboard
    navigate('/dashboard');
  } catch (error) {
    setError(error.response?.data?.message || 'Login failed');
  }
};
```

---

## 🎨 Display Formatters

The frontend includes simple display formatters (NO calculations):

```typescript
// src/utils/formatters.ts

formatCurrency(amount: number) → "$100.00"
formatOdds(odds: number, format: 'american' | 'decimal' | 'fractional') → "+265"
formatPercentage(value: number) → "8.2%"
formatDate(date: Date) → "Feb 14, 2025"
```

---

## ⚡ Quick Start Integration

### 1. Set up environment variables
```bash
# .env
VITE_API_URL=https://your-backend-api.com/api
VITE_FIREBASE_API_KEY=your_key
# ... other Firebase config
```

### 2. Create API services
```bash
mkdir src/services
touch src/services/api.ts
touch src/services/bettingService.ts
touch src/services/dfsService.ts
touch src/services/bankrollService.ts
```

### 3. Replace mock data in components
- BettingSlip: Replace mock calculations with API calls
- PropBetFinder: Fetch props from API
- LineupBuilder: Fetch players and generate lineups via API
- BankrollManager: Fetch all data from API
- DashboardOverview: Fetch games and props from API

---

## 🧪 Testing with Mock Backend

You can use tools like:
- **JSON Server** for quick mock API
- **MSW (Mock Service Worker)** for browser mocking
- **Postman Mock Server** for API testing

Example with JSON Server:
```bash
npm install -g json-server
json-server --watch db.json --port 8000
```

---

## 📊 State Management Recommendations

For complex apps, consider:
- **React Query** for data fetching/caching
- **Zustand** for global state
- **Context API** for auth state (already implemented)

Example with React Query:
```typescript
const { data, isLoading } = useQuery({
  queryKey: ['props', sport],
  queryFn: () => bettingService.getProps(sport),
});
```

---

## 🔒 Security Considerations

### Frontend
- ✅ Store auth tokens securely
- ✅ Validate all user inputs
- ✅ Use HTTPS only
- ✅ Implement CSRF protection
- ✅ Don't expose sensitive data

### Backend (Your Responsibility)
- ✅ Rate limiting
- ✅ Input sanitization
- ✅ Authentication & authorization
- ✅ SQL injection prevention
- ✅ Secure password storage
- ✅ API key management

---

## 📝 Next Steps

1. ✅ Set up your backend API
2. ✅ Create API client (`src/services/api.ts`)
3. ✅ Create service functions for each feature
4. ✅ Replace mock data in components with API calls
5. ✅ Add loading states
6. ✅ Add error handling
7. ✅ Test integration
8. ✅ Deploy!

---

## 💬 Support

The frontend is ready to integrate with your backend. All components are designed to receive data via props, making it easy to connect to any API.

**Key Points:**
- No complex calculations in frontend
- All components accept data via props
- Easy to swap mock data with real API
- Professional UI ready for production
- Type-safe TypeScript interfaces

---

**Last Updated:** October 14, 2025


