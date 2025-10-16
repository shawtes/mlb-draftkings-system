import { Button } from './ui/button';
import { Card } from './ui/card';
import { TrendingUp, Target, BarChart3, Shield, CheckCircle, ArrowRight, Sparkles, Zap, Code } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useState, useCallback } from 'react';
import type { HomepageProps } from '../types';

export default function Homepage({ onLogin, onSignUp }: HomepageProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // Throttle mouse movement for better performance using requestAnimationFrame
  useEffect(() => {
    let ticking = false;
    
    const handleMouseMove = (e: MouseEvent) => {
      if (!ticking) {
        requestAnimationFrame(() => {
          setMousePosition({ x: e.clientX, y: e.clientY });
          ticking = false;
        });
        ticking = true;
      }
    };
    
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    element?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Animated Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0a0a0a_1px,transparent_1px),linear-gradient(to_bottom,#0a0a0a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />
      
      {/* Gradient Orbs */}
      <motion.div 
        className="absolute top-0 -left-40 w-96 h-96 bg-cyan-500/30 rounded-full blur-[120px]"
        animate={{
          x: mousePosition.x * 0.02,
          y: mousePosition.y * 0.02,
        }}
        transition={{ type: "spring", damping: 30 }}
      />
      <motion.div 
        className="absolute top-40 right-0 w-96 h-96 bg-blue-600/20 rounded-full blur-[120px]"
        animate={{
          x: -mousePosition.x * 0.015,
          y: mousePosition.y * 0.015,
        }}
        transition={{ type: "spring", damping: 30 }}
      />

      {/* Navbar */}
      <nav className="fixed top-8 w-full bg-black/50 backdrop-blur-xl border-b border-cyan-500/20 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <motion.div 
              className="flex items-center gap-2"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="relative">
                <TrendingUp className="w-8 h-8 text-cyan-400" />
                <motion.div
                  className="absolute inset-0 bg-cyan-400/30 blur-xl rounded-full"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                DFS Web Optimizer
              </span>
            </motion.div>
            <div className="hidden md:flex items-center gap-8">
              <button onClick={() => scrollToSection('features')} className="text-slate-400 hover:text-cyan-400 transition-colors relative group">
                Features
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-cyan-400 to-blue-500 group-hover:w-full transition-all duration-300" />
              </button>
              <button onClick={() => scrollToSection('how-it-works')} className="text-slate-400 hover:text-cyan-400 transition-colors relative group">
                How it Works
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-cyan-400 to-blue-500 group-hover:w-full transition-all duration-300" />
              </button>
              <button onClick={() => scrollToSection('pricing')} className="text-slate-400 hover:text-cyan-400 transition-colors relative group">
                Pricing
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-cyan-400 to-blue-500 group-hover:w-full transition-all duration-300" />
              </button>
              <Button variant="ghost" onClick={onLogin} className="text-slate-300 hover:text-cyan-400 hover:bg-cyan-500/10">
                Login
              </Button>
              <Button 
                onClick={onSignUp} 
                className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] transition-all duration-300"
              >
                Sign Up Free
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Promo Ticker Bar */}
      <div className="ticker-static bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-600 text-white py-3 overflow-hidden shadow-lg border-b border-white/20">
        <div className="flex animate-scroll-homepage whitespace-nowrap items-center">
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🎯 Use code: PlayNow - Get 50% off your first month! 🎯
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🚀 Limited Time Offer - Start Your DFS Journey Today! 🚀
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            💎 Premium Analytics - Unlock Your Winning Potential! 💎
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🎯 Use code: PlayNow - Get 50% off your first month! 🎯
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🚀 Limited Time Offer - Start Your DFS Journey Today! 🚀
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            💎 Premium Analytics - Unlock Your Winning Potential! 💎
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🎯 Use code: PlayNow - Get 50% off your first month! 🎯
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            🚀 Limited Time Offer - Start Your DFS Journey Today! 🚀
          </span>
          <span className="text-base font-extrabold tracking-wide mr-20 text-white drop-shadow-lg">
            💎 Premium Analytics - Unlock Your Winning Potential! 💎
          </span>
        </div>
      </div>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Floating Elements */}
          <motion.div
            className="absolute top-20 left-20 opacity-20"
            animate={{ y: [0, -20, 0], rotate: [0, 5, 0] }}
            transition={{ duration: 5, repeat: Infinity }}
          >
            <Code className="w-16 h-16 text-cyan-400" />
          </motion.div>
          <motion.div
            className="absolute top-40 right-32 opacity-20"
            animate={{ y: [0, 20, 0], rotate: [0, -5, 0] }}
            transition={{ duration: 4, repeat: Infinity, delay: 1 }}
          >
            <Zap className="w-12 h-12 text-blue-400" />
          </motion.div>
          <motion.div
            className="absolute bottom-20 right-40 opacity-20"
            animate={{ y: [0, -15, 0], rotate: [0, 5, 0] }}
            transition={{ duration: 6, repeat: Infinity, delay: 2 }}
          >
            <Sparkles className="w-14 h-14 text-cyan-400" />
          </motion.div>

          <div className="text-center relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-6"
            >
              <motion.div
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/30 backdrop-blur-sm mb-8"
                whileHover={{ scale: 1.05 }}
              >
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-cyan-400 text-sm">AI-Powered Optimization Engine</span>
              </motion.div>
              
              <h1 className="text-white mb-6 relative">
                <span className="block text-7xl bg-gradient-to-r from-white via-cyan-200 to-blue-400 bg-clip-text text-transparent">
                  Dominate Daily
                </span>
                <span className="block text-7xl mt-2">
                  <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent animate-pulse">
                    Fantasy Sports
                  </span>
                </span>
              </h1>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-slate-400 mb-8 max-w-2xl mx-auto text-xl"
            >
              Unlock winning lineups and prop bets with our{' '}
              <span className="text-cyan-400">cutting-edge</span> optimization engine.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="flex items-center justify-center gap-4 mb-12"
            >
              <Button 
                onClick={onSignUp} 
                size="lg" 
                className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-[0_0_30px_rgba(6,182,212,0.4)] hover:shadow-[0_0_50px_rgba(6,182,212,0.6)] transition-all duration-300 text-lg px-8 py-6 group"
              >
                Start Winning Today 
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-400 text-lg px-8 py-6"
                onClick={() => scrollToSection('features')}
              >
                Explore Features
              </Button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="flex items-center justify-center gap-8 text-slate-500"
            >
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span>10,000+ Active Users</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                <span>Powered by AI</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
                <span>Real-time Analytics</span>
              </div>
            </motion.div>

            {/* Stats Display */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
              className="mt-20 grid grid-cols-3 gap-8 max-w-4xl mx-auto"
            >
              {[
                { value: '98%', label: 'Win Rate Increase', icon: TrendingUp },
                { value: '2.5M+', label: 'Lineups Generated', icon: Target },
                { value: '<30s', label: 'Optimization Time', icon: Zap },
              ].map((stat, i) => (
                <motion.div
                  key={i}
                  className="relative group"
                  whileHover={{ y: -5 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="bg-gradient-to-b from-slate-900/50 to-black/50 border border-cyan-500/20 rounded-xl p-6 backdrop-blur-sm group-hover:border-cyan-400/50 transition-all duration-300">
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-blue-500/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity" />
                    <stat.icon className="w-8 h-8 text-cyan-400 mb-3 mx-auto" />
                    <div className="text-3xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-1">
                      {stat.value}
                    </div>
                    <div className="text-slate-500 text-sm">{stat.label}</div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* Problem/Solution Section */}
      <section className="relative py-20 px-4 border-t border-cyan-500/10">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-white mb-4">
                Tired of manual research and{' '}
                <span className="text-red-400">losing lineups</span>?
              </h2>
              <p className="text-slate-400 mb-6">
                Daily Fantasy Sports and prop betting require hours of research, complex analysis, and overwhelming data. 
                Most players lose money because they can't keep up with the competition.
              </p>
              <div className="space-y-3">
                {['Hours of manual research', 'Complex data analysis', 'Overwhelming competition'].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 text-slate-500">
                    <div className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 blur-3xl rounded-3xl" />
              <div className="relative bg-gradient-to-br from-slate-900/90 to-black/90 border border-cyan-500/30 rounded-2xl p-8 backdrop-blur-sm">
                <h2 className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-4">
                  Our Optimizer is Your Secret Weapon.
                </h2>
                <p className="text-slate-300 mb-6">
                  Leverage advanced algorithms, real-time data, and customizable strategies to build winning lineups 
                  in minutes, not hours.
                </p>
                <div className="space-y-3">
                  {['AI-powered optimization', 'Real-time data processing', 'Automated lineup generation'].map((item, i) => (
                    <div key={i} className="flex items-center gap-3 text-cyan-400">
                      <CheckCircle className="w-5 h-5" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section id="features" className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-white mb-4">
              How We Help You{' '}
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Win More
              </span>
            </h2>
            <p className="text-slate-400 text-xl">
              Powerful tools designed for DFS pros and beginners alike.
            </p>
          </motion.div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Target,
                title: 'Advanced Lineup Optimization',
                description: 'Generate optimal lineups across all major DFS sites with custom rules and projections.',
                gradient: 'from-cyan-500 to-blue-500'
              },
              {
                icon: TrendingUp,
                title: 'A-Play (Prop Bet) Analysis',
                description: 'Identify high-value player prop bets with predictive analytics and real-time odds.',
                gradient: 'from-blue-500 to-purple-500'
              },
              {
                icon: BarChart3,
                title: 'Custom Player Projections',
                description: 'Import your own projections or use ours, fine-tune them, and see the immediate impact.',
                gradient: 'from-purple-500 to-pink-500'
              },
              {
                icon: Shield,
                title: 'Contest Simulators',
                description: 'Test lineups against various contest types and manage your bankroll effectively.',
                gradient: 'from-pink-500 to-cyan-500'
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -8 }}
                className="group relative"
              >
                <div className={`absolute -inset-0.5 bg-gradient-to-r ${feature.gradient} rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-300`} />
                <Card className="relative bg-black border-slate-800 p-6 h-full group-hover:border-cyan-500/50 transition-all duration-300">
                  <div className={`w-12 h-12 bg-gradient-to-r ${feature.gradient} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-white mb-2 group-hover:text-cyan-400 transition-colors">{feature.title}</h3>
                  <p className="text-slate-500 group-hover:text-slate-400 transition-colors">
                    {feature.description}
                  </p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="relative py-20 px-4 border-t border-cyan-500/10">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent" />
        <div className="max-w-7xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-white mb-4">
              Winning in{' '}
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                3 Easy Steps
              </span>
            </h2>
          </motion.div>
          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connection Lines */}
            <div className="hidden md:block absolute top-8 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />
            
            {[
              {
                step: '01',
                title: 'Select Your Sport & Slate',
                description: 'Choose from NFL, NBA, MLB, NHL, and more. Pick your preferred contest slate.',
              },
              {
                step: '02',
                title: 'Input Your Preferences & Rules',
                description: 'Set stacking rules, exposure limits, salary constraints, and customize projections.',
              },
              {
                step: '03',
                title: 'Generate & Export Lineups',
                description: 'Get optimized lineups instantly and export them to your favorite DFS platform.',
              }
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.2 }}
                className="text-center relative"
              >
                <motion.div 
                  className="relative w-16 h-16 mx-auto mb-6"
                  whileHover={{ scale: 1.1, rotate: 5 }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur-xl opacity-50" />
                  <div className="relative w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.3)]">
                    <span className="text-white text-2xl">{item.step}</span>
                  </div>
                </motion.div>
                <h3 className="text-white mb-3 text-xl">{item.title}</h3>
                <p className="text-slate-500">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-white mb-4">What Our Users Are Saying</h2>
          </motion.div>
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                quote: "This optimizer has completely changed my DFS game. I've won more in the last month than I did all last year!",
                author: 'Mike R.',
                role: 'Pro DFS Player',
                rating: 5
              },
              {
                quote: "The A-Play analyzer is incredible. I'm finding value bets I would have never spotted on my own.",
                author: 'Sarah L.',
                role: 'Sports Bettor',
                rating: 5
              },
              {
                quote: "Simple enough for beginners but powerful enough for pros. Worth every penny and then some.",
                author: 'James K.',
                role: 'DFS Enthusiast',
                rating: 5
              }
            ].map((testimonial, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <Card className="bg-gradient-to-br from-slate-900/50 to-black/50 border-cyan-500/20 p-6 h-full backdrop-blur-sm hover:border-cyan-400/40 transition-all duration-300">
                  <div className="flex gap-1 mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Sparkles key={i} className="w-4 h-4 text-cyan-400 fill-cyan-400" />
                    ))}
                  </div>
                  <p className="text-slate-300 mb-4 italic">
                    "{testimonial.quote}"
                  </p>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                      <span className="text-white">{testimonial.author[0]}</span>
                    </div>
                    <div>
                      <p className="text-cyan-400">{testimonial.author}</p>
                      <p className="text-slate-500 text-sm">{testimonial.role}</p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="relative py-20 px-4 border-t border-cyan-500/10">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-white mb-4">
              Flexible Plans for{' '}
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                Every Player
              </span>
            </h2>
          </motion.div>
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                name: 'Free Trial',
                price: '$0',
                period: '/7 days',
                features: ['3 Lineups per day', 'Basic projections', '1 sport access'],
                popular: false,
                cta: 'Start Free Trial'
              },
              {
                name: 'Pro',
                price: '$49',
                period: '/month',
                features: ['Unlimited lineups', 'Advanced projections', 'All sports', 'A-Play analyzer'],
                popular: true,
                cta: 'Choose Plan'
              },
              {
                name: 'Elite',
                price: '$99',
                period: '/month',
                features: ['Everything in Pro', 'Contest simulator', 'Custom projections', 'Priority support'],
                popular: false,
                cta: 'Choose Plan'
              }
            ].map((plan, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ y: -8 }}
                className="relative group"
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 z-10">
                    <div className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-4 py-1 rounded-full text-sm shadow-[0_0_20px_rgba(6,182,212,0.4)]">
                      Most Popular
                    </div>
                  </div>
                )}
                <div className={`absolute -inset-0.5 bg-gradient-to-r ${plan.popular ? 'from-cyan-500 to-blue-600' : 'from-slate-700 to-slate-800'} rounded-2xl blur ${plan.popular ? 'opacity-50' : 'opacity-20'} group-hover:opacity-70 transition duration-300`} />
                <Card className={`relative ${plan.popular ? 'bg-gradient-to-b from-slate-900 to-black border-cyan-500/50' : 'bg-black border-slate-800'} p-6 h-full`}>
                  <h3 className="text-white mb-2 text-xl">{plan.name}</h3>
                  <div className="mb-6">
                    <span className="text-white text-5xl bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">{plan.price}</span>
                    <span className="text-slate-500">{plan.period}</span>
                  </div>
                  <ul className="space-y-3 mb-8">
                    {plan.features.map((feature, j) => (
                      <li key={j} className="flex items-center gap-2 text-slate-300">
                        <CheckCircle className={`w-4 h-4 ${plan.popular ? 'text-cyan-400' : 'text-slate-600'}`} />
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <Button 
                    onClick={onSignUp} 
                    className={`w-full ${plan.popular 
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 shadow-[0_0_20px_rgba(6,182,212,0.3)]' 
                      : 'bg-slate-800 hover:bg-slate-700 border border-slate-700'
                    } text-white`}
                  >
                    {plan.cta}
                  </Button>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="relative py-20 px-4">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent" />
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto text-center relative"
        >
          <div className="absolute -inset-8 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 blur-3xl rounded-full" />
          <div className="relative bg-gradient-to-br from-slate-900/80 to-black/80 border border-cyan-500/30 rounded-3xl p-12 backdrop-blur-sm">
            <h2 className="text-white mb-6">
              Ready to Transform Your{' '}
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                DFS Game?
              </span>
            </h2>
            <Button 
              onClick={onSignUp} 
              size="lg" 
              className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-[0_0_40px_rgba(6,182,212,0.4)] hover:shadow-[0_0_60px_rgba(6,182,212,0.6)] transition-all duration-300 text-lg px-10 py-7 mb-4"
            >
              <Sparkles className="mr-2 w-5 h-5" />
              Sign Up for Free & Start Optimizing Now!
            </Button>
            <p className="text-slate-500 flex items-center justify-center gap-2">
              <CheckCircle className="w-4 h-4 text-cyan-400" />
              No Credit Card Required
            </p>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-cyan-500/10 py-12 px-4 bg-black">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="relative">
                  <TrendingUp className="w-6 h-6 text-cyan-400" />
                  <div className="absolute inset-0 bg-cyan-400/30 blur-lg rounded-full" />
                </div>
                <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  DFS Web Optimizer
                </span>
              </div>
              <p className="text-slate-500">
                Your competitive edge in Daily Fantasy Sports.
              </p>
            </div>
            <div>
              <h4 className="text-white mb-4">Product</h4>
              <ul className="space-y-2 text-slate-500">
                <li>
                  <button onClick={() => scrollToSection('features')} className="hover:text-cyan-400 transition-colors">
                    Features
                  </button>
                </li>
                <li>
                  <button onClick={() => scrollToSection('pricing')} className="hover:text-cyan-400 transition-colors">
                    Pricing
                  </button>
                </li>
                <li>
                  <button onClick={() => scrollToSection('how-it-works')} className="hover:text-cyan-400 transition-colors">
                    How it Works
                  </button>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="text-white mb-4">Company</h4>
              <ul className="space-y-2 text-slate-500">
                <li><a href="#" className="hover:text-cyan-400 transition-colors">About Us</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Contact</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Careers</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white mb-4">Legal</h4>
              <ul className="space-y-2 text-slate-500">
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Terms of Service</a></li>
                <li><a href="#" className="hover:text-cyan-400 transition-colors">Cookie Policy</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-cyan-500/10 pt-8 text-center">
            <p className="text-slate-600">
              &copy; 2025 DFS Web Optimizer. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
