# Deployment Guide - MLB DFS Optimizer

This guide covers various deployment options for the MLB DFS Optimizer web application.

## Table of Contents
1. [Local Development](#local-development)
2. [Production Build](#production-build)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Development

### Quick Start
```bash
# Use the provided startup script
start.bat  # Windows
./setup.sh  # Linux/Mac

# Or manually:
cd server && npm install && npm start &
cd client && npm install && npm start
```

### Development URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- WebSocket: ws://localhost:5000

## Production Build

### 1. Build Frontend for Production
```bash
cd client
npm run build
```
This creates an optimized build in `client/build/`

### 2. Configure Production Server
```bash
cd server
# Set environment
export NODE_ENV=production
export PORT=8080

# Install production dependencies only
npm install --only=production

# Start server
npm start
```

### 3. Serve Static Files
The Express server automatically serves the built React app from `client/build/`

## Docker Deployment

### 1. Create Dockerfile
```dockerfile
# Dockerfile
FROM node:16-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY server/package*.json ./server/
COPY client/package*.json ./client/

# Install dependencies
RUN npm install --only=production
RUN cd server && npm install --only=production
RUN cd client && npm install

# Copy source code
COPY . .

# Build frontend
RUN cd client && npm run build

# Expose port
EXPOSE 5000

# Start server
CMD ["npm", "start"]
```

### 2. Build and Run Container
```bash
# Build image
docker build -t mlb-dfs-optimizer .

# Run container
docker run -p 5000:5000 mlb-dfs-optimizer
```

### 3. Docker Compose (with Database)
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=mongodb://mongo:27017/mlb-dfs
    depends_on:
      - mongo
    
  mongo:
    image: mongo:4.4
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

## Cloud Deployment

### 1. Heroku Deployment

**Prerequisites:**
- Heroku CLI installed
- Git repository

**Steps:**
```bash
# Login to Heroku
heroku login

# Create app
heroku create mlb-dfs-optimizer

# Set environment variables
heroku config:set NODE_ENV=production
heroku config:set NPM_CONFIG_PRODUCTION=false

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

**Procfile:**
```
web: cd server && npm start
```

### 2. AWS EC2 Deployment

**Launch EC2 Instance:**
- Choose Ubuntu Server 20.04 LTS
- t2.micro for testing (t2.small+ for production)
- Configure security group (ports 22, 80, 443)

**Setup on EC2:**
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install PM2 for process management
sudo npm install -g pm2

# Clone your repository
git clone your-repo-url
cd mlb-dfs-optimizer

# Install dependencies and build
cd server && npm install --only=production
cd ../client && npm install && npm run build

# Start with PM2
cd ../server
pm2 start index.js --name "mlb-dfs-optimizer"
pm2 startup
pm2 save
```

### 3. Digital Ocean App Platform

**app.yaml:**
```yaml
name: mlb-dfs-optimizer
services:
- name: web
  source_dir: /
  github:
    repo: your-username/mlb-dfs-optimizer
    branch: main
  run_command: cd server && npm start
  environment_slug: node-js
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: NODE_ENV
    value: production
```

### 4. Vercel Deployment

**vercel.json:**
```json
{
  "version": 2,
  "builds": [
    {
      "src": "server/index.js",
      "use": "@vercel/node"
    },
    {
      "src": "client/package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/server/index.js"
    },
    {
      "src": "/(.*)",
      "dest": "/client/build/$1"
    }
  ]
}
```

## Environment Configuration

### Production Environment Variables

**Server (.env file):**
```bash
NODE_ENV=production
PORT=5000
CORS_ORIGIN=https://your-domain.com
MAX_FILE_SIZE=10485760
JWT_SECRET=your-jwt-secret
DATABASE_URL=your-database-url
REDIS_URL=your-redis-url
LOG_LEVEL=info
```

**Security Considerations:**
- Use HTTPS in production
- Set secure CORS origins
- Implement rate limiting
- Add request validation
- Use environment variables for secrets

### SSL/HTTPS Setup

**With Let's Encrypt (Nginx):**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Database Integration (Optional)

### MongoDB Setup
```javascript
// server/database.js
const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.DATABASE_URL, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('MongoDB connected');
  } catch (error) {
    console.error('Database connection error:', error);
    process.exit(1);
  }
};
```

### Player Schema
```javascript
// server/models/Player.js
const mongoose = require('mongoose');

const playerSchema = new mongoose.Schema({
  name: { type: String, required: true },
  position: { type: String, required: true },
  salary: { type: Number, required: true },
  team: { type: String, required: true },
  projection: { type: Number, required: true },
  selected: { type: Boolean, default: false },
  minExposure: { type: Number, default: 0 },
  maxExposure: { type: Number, default: 100 },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Player', playerSchema);
```

## Monitoring and Maintenance

### Health Checks
```javascript
// Add to server/index.js
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: process.env.npm_package_version
  });
});
```

### Logging Setup
```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }));
}
```

### Process Management with PM2
```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'mlb-dfs-optimizer',
    script: 'server/index.js',
    instances: 'max',
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 5000
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
};
```

### Backup Strategy
- Database backups (if using database)
- Configuration file backups
- User data export functionality
- Automated backup scheduling

### Updates and Maintenance
```bash
# Update dependencies
npm update
npm audit fix

# Deploy updates
git pull origin main
npm install
npm run build  # if frontend changes
pm2 reload all
```

## Performance Optimization

### Frontend Optimization
- Code splitting with React.lazy()
- Image optimization
- Bundle analysis
- Service worker for caching

### Backend Optimization
- Connection pooling
- Response caching
- Request rate limiting
- Database indexing

### CDN Integration
- Static asset delivery
- Global distribution
- Cache optimization

## Troubleshooting

### Common Production Issues
1. **Memory leaks** - Monitor memory usage, restart if needed
2. **High CPU usage** - Check optimization algorithms
3. **WebSocket disconnections** - Implement retry logic
4. **File upload failures** - Check disk space and permissions

### Debugging Tools
- PM2 monitoring: `pm2 monit`
- Log analysis: `tail -f logs/combined.log`
- Memory profiling: Node.js profiler
- Performance monitoring: New Relic, DataDog

## Security Checklist

- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] Input validation implemented
- [ ] File upload restrictions in place
- [ ] Rate limiting configured
- [ ] Environment variables secured
- [ ] Regular security updates
- [ ] Error messages don't expose sensitive data

---

This deployment guide should cover most scenarios for getting your MLB DFS Optimizer running in production. Choose the deployment method that best fits your needs and infrastructure.
