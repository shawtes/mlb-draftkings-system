import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card } from './ui/card';
import { TrendingUp, Eye, EyeOff, ArrowRight, Sparkles, CheckCircle } from 'lucide-react';
import { motion } from 'motion/react';
import { useAuth } from '../contexts/AuthContext';
import type { RegisterPageProps, FirebaseError } from '../types';

export default function RegisterPage({ onRegister, onSwitchToLogin }: RegisterPageProps) {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const { signup, loginWithGoogle } = useAuth();

  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required';
    }

    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    
    try {
      const displayName = `${formData.firstName} ${formData.lastName}`.trim();
      await signup(formData.email, formData.password, displayName);
      onRegister();
    } catch (err) {
      const error = err as FirebaseError;
      console.error('Registration error:', error);
      const newErrors: Record<string, string> = {};
      
      if (error.code === 'auth/email-already-in-use') {
        newErrors.email = 'This email is already registered';
      } else if (error.code === 'auth/weak-password') {
        newErrors.password = 'Password is too weak';
      } else if (error.code === 'auth/invalid-email') {
        newErrors.email = 'Invalid email address';
      } else {
        newErrors.general = error.message || 'Failed to create account. Please try again.';
      }
      
      setErrors(newErrors);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignup = async () => {
    setIsLoading(true);
    setErrors({});
    
    try {
      await loginWithGoogle();
      onRegister();
    } catch (err) {
      const error = err as FirebaseError;
      console.error('Google signup error:', error);
      setErrors({ general: error.message || 'Failed to sign up with Google. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Animated Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      
      {/* Gradient Orbs */}
      <motion.div 
        className="absolute top-0 -left-40 w-96 h-96 bg-cyan-500/30 rounded-full blur-[120px]"
        animate={{
          scale: [1, 1.1, 1],
          opacity: [0.3, 0.5, 0.3]
        }}
        transition={{ duration: 4, repeat: Infinity }}
      />
      <motion.div 
        className="absolute top-40 right-0 w-96 h-96 bg-blue-600/20 rounded-full blur-[120px]"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.2, 0.4, 0.2]
        }}
        transition={{ duration: 5, repeat: Infinity, delay: 1 }}
      />

      <div className="relative z-10 min-h-screen flex">
        {/* Left Side - Register Form */}
        <div className="flex-1 flex items-center justify-center p-8">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="w-full max-w-md"
          >
            {/* Header */}
            <div className="text-center mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                className="flex items-center justify-center gap-2 mb-6"
              >
                <div className="relative">
                  <TrendingUp className="w-8 h-8 text-cyan-400" />
                  <motion.div
                    className="absolute inset-0 bg-cyan-400/30 blur-xl rounded-full"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                </div>
                <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent text-2xl font-bold">
                  DFS Optimizer
                </span>
              </motion.div>
              
              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className="text-white text-3xl font-bold mb-2"
              >
                Create Account
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                className="text-slate-400"
              >
                Join the winning community and start optimizing today
              </motion.p>
            </div>

            {/* Register Form */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.5 }}
            >
              <Card className="bg-gradient-to-br from-slate-900/50 to-black/50 border-cyan-500/20 p-8 backdrop-blur-sm">
                <form onSubmit={handleSubmit} className="space-y-6">
                  {errors.general && (
                    <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                      <p className="text-red-400 text-sm">{errors.general}</p>
                    </div>
                  )}
                  
                  {/* Name Fields */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label htmlFor="firstName" className="text-slate-300 text-sm font-medium">
                        First Name <span className="text-red-400">*</span>
                      </label>
                        <Input
                          id="firstName"
                          name="firstName"
                          type="text"
                          value={formData.firstName}
                          onChange={handleInputChange}
                        placeholder="Enter first name"
                        className={`bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-cyan-500/20 transition-all ${
                            errors.firstName ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                          }`}
                          required
                        />
                      {errors.firstName && (
                        <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                          <span>⚠</span> {errors.firstName}
                        </p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <label htmlFor="lastName" className="text-slate-300 text-sm font-medium">
                        Last Name <span className="text-red-400">*</span>
                      </label>
                        <Input
                          id="lastName"
                          name="lastName"
                          type="text"
                          value={formData.lastName}
                          onChange={handleInputChange}
                        placeholder="Enter last name"
                        className={`bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-cyan-500/20 transition-all ${
                            errors.lastName ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                          }`}
                          required
                        />
                      {errors.lastName && (
                        <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                          <span>⚠</span> {errors.lastName}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Email Field */}
                  <div className="space-y-2">
                    <label htmlFor="email" className="text-slate-300 text-sm font-medium">
                      Email Address <span className="text-red-400">*</span>
                    </label>
                      <Input
                        id="email"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleInputChange}
                      placeholder="you@example.com"
                      className={`bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-cyan-500/20 transition-all ${
                          errors.email ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                        }`}
                        required
                      />
                    {errors.email && (
                      <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                        <span>⚠</span> {errors.email}
                      </p>
                    )}
                  </div>

                  {/* Password Field */}
                  <div className="space-y-2">
                    <label htmlFor="password" className="text-slate-300 text-sm font-medium">
                      Password <span className="text-red-400">*</span>
                    </label>
                    <div className="relative">
                      <Input
                        id="password"
                        name="password"
                        type={showPassword ? "text" : "password"}
                        value={formData.password}
                        onChange={handleInputChange}
                        placeholder="At least 8 characters"
                        className={`bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-cyan-500/20 pr-10 transition-all ${
                          errors.password ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                        }`}
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-cyan-400 transition-colors"
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    {errors.password && (
                      <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                        <span>⚠</span> {errors.password}
                      </p>
                    )}
                  </div>

                  {/* Confirm Password Field */}
                  <div className="space-y-2">
                    <label htmlFor="confirmPassword" className="text-slate-300 text-sm font-medium">
                      Confirm Password <span className="text-red-400">*</span>
                    </label>
                    <div className="relative">
                      <Input
                        id="confirmPassword"
                        name="confirmPassword"
                        type={showConfirmPassword ? "text" : "password"}
                        value={formData.confirmPassword}
                        onChange={handleInputChange}
                        placeholder="Re-enter your password"
                        className={`bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 focus:border-cyan-500 focus:ring-cyan-500/20 pr-10 transition-all ${
                          errors.confirmPassword ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''
                        }`}
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-cyan-400 transition-colors"
                      >
                        {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    {errors.confirmPassword && (
                      <p className="text-red-400 text-xs mt-1 flex items-center gap-1">
                        <span>⚠</span> {errors.confirmPassword}
                      </p>
                    )}
                  </div>

                  {/* Terms and Conditions */}
                  <div className="flex items-start gap-3">
                    <input 
                      type="checkbox" 
                      id="terms"
                      className="mt-1 rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/20" 
                      required
                    />
                    <label htmlFor="terms" className="text-sm text-slate-400">
                      I agree to the{' '}
                      <button type="button" className="text-cyan-400 hover:text-cyan-300 transition-colors">
                        Terms of Service
                      </button>
                      {' '}and{' '}
                      <button type="button" className="text-cyan-400 hover:text-cyan-300 transition-colors">
                        Privacy Policy
                      </button>
                    </label>
                  </div>

                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] transition-all duration-300"
                  >
                    {isLoading ? (
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                      />
                    ) : (
                      <>
                        Create Account
                        <ArrowRight className="ml-2 w-4 h-4" />
                      </>
                    )}
                  </Button>

                  <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="w-full border-t border-slate-700"></div>
                    </div>
                    <div className="relative flex justify-center text-sm">
                      <span className="px-2 bg-slate-900/50 text-slate-400">Or sign up with</span>
                    </div>
                  </div>

                  <Button
                    type="button"
                    onClick={handleGoogleSignup}
                    disabled={isLoading}
                    variant="outline"
                    className="w-full border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 text-white"
                  >
                    <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                      <path
                        fill="currentColor"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                      />
                      <path
                        fill="currentColor"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                      />
                      <path
                        fill="currentColor"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                      />
                      <path
                        fill="currentColor"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                      />
                    </svg>
                    Sign up with Google
                  </Button>
                </form>
              </Card>
            </motion.div>

            {/* Switch to Login */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.7 }}
              className="text-center mt-6"
            >
              <p className="text-slate-400">
                Already have an account?{' '}
                <button
                  onClick={onSwitchToLogin}
                  className="text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
                >
                  Sign in here
                </button>
              </p>
            </motion.div>
          </motion.div>
        </div>

        {/* Right Side - Benefits/Features */}
        <div className="hidden lg:flex flex-1 items-center justify-center p-8 bg-gradient-to-br from-cyan-500/5 to-blue-500/5">
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="max-w-md"
          >
            <div className="text-center mb-8">
              <motion.div
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{ duration: 4, repeat: Infinity }}
                className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-[0_0_30px_rgba(6,182,212,0.3)]"
              >
                <Sparkles className="w-8 h-8 text-white" />
              </motion.div>
              <h2 className="text-white text-2xl font-bold mb-4">
                Join the Winning Community
              </h2>
              <p className="text-slate-400">
                Start your journey to consistent DFS success with our proven optimization tools.
              </p>
            </div>

            <div className="space-y-4">
              {[
                { icon: CheckCircle, text: "Free 7-day trial - no credit card required" },
                { icon: TrendingUp, text: "AI-powered lineup optimization" },
                { icon: Sparkles, text: "Real-time data and projections" },
                { icon: ArrowRight, text: "Export to all major DFS platforms" }
              ].map((benefit, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.5 + i * 0.1 }}
                  className="flex items-center gap-3 text-slate-300"
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center">
                    <benefit.icon className="w-4 h-4 text-white" />
                  </div>
                  <span>{benefit.text}</span>
                </motion.div>
              ))}
            </div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.9 }}
              className="mt-8 grid grid-cols-2 gap-4"
            >
              <div className="text-center">
                <div className="text-2xl font-bold text-cyan-400">10,000+</div>
                <div className="text-sm text-slate-500">Active Users</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">98%</div>
                <div className="text-sm text-slate-500">Win Rate Increase</div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
