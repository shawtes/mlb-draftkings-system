# UrSim DFS Optimizer

A modern, high-performance daily fantasy sports optimizer built with React, TypeScript, and Firebase.

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Firebase account
- Your backend API (if applicable)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ctsc/UrSimFrontend.git
   cd UrSimFrontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your Firebase credentials
   # Get these from Firebase Console: https://console.firebase.google.com/project/ursim-8ce06
   ```

4. **Start development server:**
   ```bash
   npm run dev
   ```

5. **Open your browser:**
   ```
   http://localhost:3001
   ```

---

## 📁 Project Structure

```
UrSimFrontend/
├── src/
│   ├── components/          # React components
│   │   ├── ui/             # Shadcn/Radix UI components
│   │   ├── Dashboard.tsx   # Main dashboard
│   │   ├── Homepage.tsx    # Landing page
│   │   ├── LoginPage.tsx   # Login page
│   │   ├── RegisterPage.tsx # Registration page
│   │   ├── ErrorBoundary.tsx # Error handling
│   │   └── SkeletonLoader.tsx # Loading states
│   ├── contexts/           # React contexts
│   │   └── AuthContext.tsx # Firebase auth state
│   ├── firebase/           # Firebase configuration
│   │   └── config.ts       # Firebase init
│   ├── styles/            # Global styles
│   ├── App.tsx            # Main app component
│   ├── main.tsx           # App entry point
│   └── index.css          # Tailwind + custom CSS
├── .env                   # Environment variables (DO NOT COMMIT)
├── .env.example          # Environment template (COMMIT THIS)
├── package.json          # Dependencies
└── vite.config.ts        # Vite configuration
```

---

## 🛠️ Development Guide for Team Members

### Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** in the appropriate files

3. **Test locally:**
   ```bash
   npm run dev
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

5. **Push and create Pull Request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Common Tasks

**Adding a new component:**
```bash
# Create in src/components/
# Import in parent component
# Use lazy loading if it's a large component
```

**Updating styles:**
```bash
# Use Tailwind classes directly in components
# Add custom CSS to src/index.css if needed
```

**Adding a new page/route:**
```bash
# 1. Create component in src/components/
# 2. Add lazy import in App.tsx
# 3. Add route logic in App.tsx
```

**Connecting to backend API:**
```bash
# 1. Add API URL to .env
# 2. Create API service in src/services/ (recommended)
# 3. Use in components with proper error handling
```

---

## 🔥 Firebase Setup

### First-Time Setup

1. **Go to Firebase Console:** https://console.firebase.google.com/project/ursim-8ce06

2. **Enable Authentication:**
   - Navigate to **Authentication** → **Sign-in method**
   - Enable **Email/Password**
   - Enable **Google** (optional)

3. **Get Your Config:**
   - Go to **Project Settings** → **Your apps**
   - Copy the Firebase config
   - Add to `.env` file

4. **Add Authorized Domains:**
   - In Authentication → Settings → Authorized domains
   - Add your production domain when deploying

---

## 📦 Building for Production

### Build the app:
```bash
npm run build
```

This creates optimized production files in the `build/` folder.

### Test production build locally:
```bash
npm run preview
```

---

## 🚀 Deployment

### Option 1: Vercel (Recommended - Fastest)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel --prod
   ```

3. **Set environment variables** in Vercel dashboard

### Option 2: Netlify

1. **Build the app:**
   ```bash
   npm run build
   ```

2. **Drag `build/` folder to Netlify** or use CLI:
   ```bash
   npm install -g netlify-cli
   netlify deploy --prod --dir=build
   ```

3. **Set environment variables** in Netlify dashboard

### Option 3: Traditional Hosting

1. Build: `npm run build`
2. Upload `build/` folder to your server
3. Configure server for SPA (redirect all routes to index.html)

---

## 🔧 Environment Variables

**Required for production:**

```env
VITE_FIREBASE_API_KEY=your_key_here
VITE_FIREBASE_AUTH_DOMAIN=your_domain_here
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_bucket_here
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
VITE_FIREBASE_MEASUREMENT_ID=your_measurement_id
```

**Never commit the `.env` file to Git!** ⚠️

---

## 🧪 Testing

### Manual Testing Checklist:
- [ ] Test signup with email/password
- [ ] Test login with email/password
- [ ] Test Google Sign-In
- [ ] Test logout
- [ ] Test on mobile device
- [ ] Test on different browsers
- [ ] Test with slow network
- [ ] Test error scenarios

### Browser Support:
- Chrome 90+ ✅
- Firefox 88+ ✅
- Safari 14+ ✅
- Edge 90+ ✅

---

## 🐛 Troubleshooting

### "Firebase: Error (auth/configuration-not-found)"
- Enable Email/Password auth in Firebase Console

### "Firebase: Error (auth/popup-blocked)"
- Allow popups for your domain
- Code will auto-fallback to redirect method

### "Cannot connect to backend API"
- Check your API URL in `.env`
- Verify CORS is configured on backend
- Check network tab for error details

### App won't start
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

---

## 📚 Additional Documentation

- **[TIMELINE.md](./TIMELINE.md)** - Development roadmap and task timeline
- **[COMMITS.md](./COMMITS.md)** - Detailed commit history and changes
- **[Firebase Console](https://console.firebase.google.com/project/ursim-8ce06)** - Manage authentication

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is proprietary and confidential.

---

## 🆘 Need Help?

- Check [TIMELINE.md](./TIMELINE.md) for development roadmap
- Check [COMMITS.md](./COMMITS.md) for what's been implemented
- Open an issue on GitHub
- Contact the team lead

---

## ⚡ Performance

This app is optimized for production with:
- Code splitting and lazy loading
- Optimized bundle sizes
- Firebase authentication
- Error boundaries
- Loading states
- Mobile responsive design

**Target Metrics:**
- Lighthouse Score: 90+
- Load Time: <2 seconds
- Time to Interactive: <3 seconds

---

**Built with ❤️ for DFS players**
