# MLB DFS Optimizer - Web Application

A modern, web-based MLB Daily Fantasy Sports (DFS) optimizer built with Node.js, Express, React, and TypeScript. This application helps you optimize your MLB DFS lineups with advanced features like team stacking, exposure controls, and real-time optimization progress.

## Features

### 🏆 **Advanced Optimization**
- Multiple optimization strategies (Greedy, Balanced, Value, Projection-based)
- Team stacking with customizable parameters
- Player exposure controls and limits
- Salary cap management ($45,000 - $50,000)
- Unique player requirements per lineup

### 📊 **Player Management**
- CSV file upload for player data
- Bulk player selection/deselection
- Individual player exposure settings
- Real-time player filtering and search
- Position-based organization

### 🎯 **Team Stacking**
- Multi-team stacking options
- Configurable minimum players per stack
- Stack exposure controls
- Automatic stack detection and analysis

### 📈 **Real-time Updates**
- WebSocket connection for live optimization progress
- Real-time status updates and notifications
- Progress tracking during optimization runs

### 💻 **Modern UI/UX**
- Material-UI design system
- Dark theme optimized for DFS users
- Responsive design for all devices
- Smooth animations and transitions
- Toast notifications for user feedback

## Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer
   ```

2. **Run the startup script (Windows)**
   ```bash
   start.bat
   ```
   
   Or manually start both servers:

3. **Install dependencies**
   ```bash
   # Install server dependencies
   cd server
   npm install

   # Install client dependencies
   cd ../client
   npm install
   ```

4. **Start the application**
   ```bash
   # Start backend server (Terminal 1)
   cd server
   npm start

   # Start frontend dev server (Terminal 2)
   cd ../client
   npm start
   ```

5. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Usage Guide

### 1. Upload Player Data
- Click "Upload CSV" button in the Players tab
- Select your DraftKings MLB CSV export file
- Players will be automatically parsed and loaded

### 2. Configure Settings
- **Control Panel**: Set number of lineups, salary range, uniqueness requirements
- **Players Tab**: Select/deselect players, set individual exposure limits
- **Team Stacks**: Configure team stacking preferences
- **Stack Exposure**: Set exposure limits for team combinations

### 3. Run Optimization
- Click "Run Optimization" button
- Monitor real-time progress in the status bar
- View results as they're generated

### 4. Export Results
- Download optimized lineups as CSV
- Compatible with DraftKings upload format

## CSV File Format

Your player CSV should include these columns:
- `Name` - Player name
- `Position` - Player position (P, C, 1B, 2B, 3B, SS, OF)
- `Salary` - Player salary
- `TeamAbbrev` - Team abbreviation
- `AvgPointsPerGame` or `Projection` - Projected points

## API Endpoints

### Player Management
- `GET /api/health` - Health check
- `POST /api/upload` - Upload player CSV
- `GET /api/players` - Get all players
- `POST /api/players/bulk-update` - Bulk update player settings

### Optimization
- `POST /api/optimize` - Run optimization
- `GET /api/results` - Get optimization results
- `GET /api/export/:format` - Export results (csv/draftkings)

### WebSocket Events
- `OPTIMIZATION_STARTED` - Optimization began
- `OPTIMIZATION_PROGRESS` - Progress updates
- `OPTIMIZATION_COMPLETED` - Optimization finished
- `OPTIMIZATION_ERROR` - Error occurred

## Architecture

### Backend (Node.js/Express)
- RESTful API with comprehensive endpoints
- WebSocket support for real-time updates
- Advanced optimization algorithms
- File upload and CSV processing
- Security middleware (Helmet, CORS)

### Frontend (React/TypeScript)
- Material-UI components and theming
- TypeScript for type safety
- Real-time WebSocket integration
- Responsive design patterns
- Modern React hooks and patterns

### Optimization Engine
- Multi-strategy lineup generation
- Position requirement validation
- Salary cap optimization
- Team stacking algorithms
- Exposure limit enforcement
- Duplicate lineup prevention

## Configuration

### Environment Variables
Create a `.env` file in the server directory:
```
PORT=5000
NODE_ENV=development
MAX_LINEUPS=1000
ALLOWED_ORIGINS=http://localhost:3000
```

### Optimization Settings
Default settings can be modified in `server/optimizer.js`:
- Position requirements
- Salary limits
- Stacking parameters
- Optimization strategies

## Development

### Project Structure
```
web_optimizer/
├── server/                 # Backend Express server
│   ├── index.js           # Main server file
│   ├── optimizer.js       # Optimization engine
│   ├── package.json       # Server dependencies
│   └── uploads/           # Uploaded CSV files
├── client/                # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API and WebSocket services
│   │   ├── App.tsx        # Main application
│   │   └── index.tsx      # Application entry point
│   ├── public/            # Static files
│   ├── package.json       # Client dependencies
│   └── tsconfig.json      # TypeScript configuration
├── package.json           # Root package.json
└── start.bat             # Windows startup script
```

### Adding New Features
1. **Backend**: Add endpoints in `server/index.js`
2. **Frontend**: Create components in `client/src/components/`
3. **Optimization**: Modify algorithms in `server/optimizer.js`

## Deployment

### Production Build
```bash
# Build frontend for production
cd client
npm run build

# Start production server
cd ../server
NODE_ENV=production npm start
```

### Docker Deployment
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY . .
RUN npm install --production
EXPOSE 5000
CMD ["npm", "start"]
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change PORT in server/.env file
   - Kill existing processes: `taskkill /f /im node.exe`

2. **CSV upload fails**
   - Ensure CSV has required columns
   - Check file size (max 10MB)
   - Verify CSV format (comma-separated)

3. **Optimization errors**
   - Ensure enough players selected (minimum 10)
   - Check salary constraints
   - Verify position requirements can be met

4. **WebSocket connection issues**
   - Check firewall settings
   - Verify both frontend and backend are running
   - Try refreshing the browser

### Performance Tips
- Limit concurrent optimizations
- Use reasonable number of lineups (1-100)
- Filter players before optimization
- Monitor memory usage for large player pools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper TypeScript types
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review console logs for errors
3. Ensure all dependencies are installed
4. Verify CSV file format

---

**Built with ❤️ for the DFS community**
