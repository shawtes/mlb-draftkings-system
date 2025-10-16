import { Component, ReactNode } from 'react';
import { Button } from './ui/button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-slate-800 rounded-lg shadow-xl p-8 text-center">
            <div className="mb-6">
              <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-8 h-8 text-red-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">Oops! Something went wrong</h2>
              <p className="text-slate-400 mb-4">
                We encountered an unexpected error. Don't worry, your data is safe.
              </p>
              {this.state.error && (
                <details className="text-left bg-slate-900 rounded p-3 mb-4">
                  <summary className="text-sm text-slate-300 cursor-pointer hover:text-white">
                    Error Details
                  </summary>
                  <p className="text-xs text-red-400 mt-2 font-mono break-words">
                    {this.state.error.toString()}
                  </p>
                </details>
              )}
            </div>
            <div className="space-y-3">
              <Button
                onClick={this.handleReset}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white"
              >
                Try Again
              </Button>
              <Button
                onClick={() => window.location.href = '/'}
                variant="outline"
                className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                Go to Homepage
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

