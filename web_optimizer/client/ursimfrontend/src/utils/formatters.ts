// Simple display formatters - no calculations, just formatting for display
// All calculations will come from backend API

/**
 * Format currency for display
 */
export function formatCurrency(amount: number, showCents: boolean = true): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: showCents ? 2 : 0,
    maximumFractionDigits: showCents ? 2 : 0,
  }).format(amount);
}

/**
 * Format odds display based on format type
 * This just formats the display - odds come from backend
 */
export function formatOdds(odds: number, format: 'american' | 'decimal' | 'fractional' = 'american'): string {
  // Backend will provide the odds, we just format for display
  switch (format) {
    case 'american':
      return odds > 0 ? `+${odds}` : `${odds}`;
    case 'decimal':
      return odds.toFixed(2);
    case 'fractional':
      // Backend should provide this pre-formatted
      return String(odds);
    default:
      return odds > 0 ? `+${odds}` : `${odds}`;
  }
}

/**
 * Format percentage for display
 */
export function formatPercentage(value: number, decimals: number = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format date for display
 */
export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(d);
}


