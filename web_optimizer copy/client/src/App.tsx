import React, { useState, useEffect } from 'react';
import { 
  ThemeProvider, 
  createTheme, 
  CssBaseline,
  Box,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Tabs,
  Tab,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Badge,
  LinearProgress,
  Tooltip,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Fab,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Switch,
  Alert,
  AlertTitle,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Breadcrumbs,
  Link,
  Stack,
  Avatar
} from '@mui/material';
import {
  Sports,
  Dashboard,
  People,
  Timeline,
  Settings,
  Star,
  Refresh,
  CloudUpload,
  Download,
  PlayArrow,
  Group,
  ShowChart,
  AccountTree,
  Favorite,
  Menu,
  Close,
  Save,
  Analytics,
  TrendingUp,
  BarChart,
  PieChart,
  Assessment,
  AttachMoney,
  Security,
  Speed,
  MonetizationOn,
  TrendingFlat,
  Psychology,
  Build,
  FileUpload,
  GetApp,
  MoneyOff,
  ExpandMore,
  NavigateNext,
  Home,
  Stop,
  CheckCircle,
  Error,
  Warning,
  Info
} from '@mui/icons-material';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import PlayersTab from './components/PlayersTab';
import TeamStacksTab from './components/TeamStacksTab';
import StackExposureTab from './components/StackExposureTab';
import TeamCombosTab from './components/TeamCombosTab';
import ControlPanelTab from './components/ControlPanelTabEnhanced';
import FavoritesTab from './components/FavoritesTab';
import ResultsTab from './components/ResultsTab';
import StatusBar from './components/StatusBar';
import ErrorBoundary from './components/ErrorBoundary';
import WebSocketConnection from './services/WebSocketConnection';

// Custom theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#667eea',
      light: '#8fa5ff',
      dark: '#3f51b5',
    },
    secondary: {
      main: '#764ba2',
      light: '#a478d4',
      dark: '#4a2c73',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1f3a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0bec5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.1)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.1)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  try {
    return <MainApp />;
  } catch (error) {
    console.error('App crashed:', error);
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <h2>Something went wrong</h2>
        <p>Please refresh the page to try again.</p>
        <button onClick={() => window.location.reload()}>Refresh</button>
      </div>
    );
  }
}

function MainApp() {
  const [currentTab, setCurrentTab] = useState(0);
  const [playersData, setPlayersData] = useState<any[]>([]);
  const [optimizationResults, setOptimizationResults] = useState<any[]>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState({
    connected: false,
    playersLoaded: 0,
    optimizationProgress: 0
  });

  // New state for enhanced UI
  const [controlPanelExpanded, setControlPanelExpanded] = useState(true);
  const [quickStatsVisible, setQuickStatsVisible] = useState(true);
  const [riskEngineEnabled, setRiskEngineEnabled] = useState(true);
  const [mobileControlPanelCollapsed, setMobileControlPanelCollapsed] = useState(false);

  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocketConnection('ws://localhost:8080');
    
    ws.on('PLAYERS_LOADED', (data: any) => {
      console.log('PLAYERS_LOADED event received:', data);
      setSystemStatus(prev => ({
        ...prev,
        playersLoaded: data?.count || 0
      }));
    });
    
    ws.on('OPTIMIZATION_STARTED', (data: any) => {
      console.log('OPTIMIZATION_STARTED event received:', data);
      setIsOptimizing(true);
      setSystemStatus(prev => ({
        ...prev,
        optimizationProgress: 0
      }));
    });
    
    ws.on('OPTIMIZATION_PROGRESS', (data: any) => {
      console.log('OPTIMIZATION_PROGRESS event received:', data);
      setSystemStatus(prev => ({
        ...prev,
        optimizationProgress: data?.progress || 0
      }));
    });
    
    ws.on('OPTIMIZATION_COMPLETED', (data: any) => {
      console.log('OPTIMIZATION_COMPLETED event received:', data);
      setIsOptimizing(false);
      setSystemStatus(prev => ({
        ...prev,
        optimizationProgress: 100
      }));
      // Reload results
      if (data?.results) {
        setOptimizationResults(data.results);
      }
      loadOptimizationResults();
    });
    
    ws.on('OPTIMIZATION_ERROR', (data: any) => {
      console.log('OPTIMIZATION_ERROR event received:', data);
      setIsOptimizing(false);
    });

    ws.on('connected', () => {
      console.log('WebSocket connected');
      setSystemStatus(prev => ({ ...prev, connected: true }));
    });

    ws.on('disconnected', () => {
      console.log('WebSocket disconnected');
      setSystemStatus(prev => ({ ...prev, connected: false }));
    });

    return () => {
      ws.disconnect();
    };
  }, []);

  const loadOptimizationResults = async () => {
    try {
      const response = await fetch('/api/results');
      const data = await response.json();
      setOptimizationResults(data.lineups || []);
    } catch (error) {
      console.error('Error loading results:', error);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const tabs = [
    { label: 'Players', icon: <People />, component: PlayersTab, color: '#4CAF50' },
    { label: 'Team Stacks', icon: <Sports />, component: TeamStacksTab, color: '#2196F3' },
    { label: 'Stack Exposure', icon: <Timeline />, component: StackExposureTab, color: '#FF9800' },
    { label: 'Team Combos', icon: <Dashboard />, component: TeamCombosTab, color: '#9C27B0' },
    { label: 'My Entries', icon: <Star />, component: FavoritesTab, color: '#FF5722' },
    { label: 'Results', icon: <Analytics />, component: ResultsTab, color: '#4CAF50' },
  ];

  const QuickStatsCard = () => (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        sx={{ 
          mb: 2,
          background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
          border: '1px solid rgba(102, 126, 234, 0.3)'
        }}
      >
        <CardContent sx={{ pb: '16px !important' }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#667eea' }}>
              📊 Quick Stats
            </Typography>
            <IconButton 
              size="small" 
              onClick={() => setQuickStatsVisible(!quickStatsVisible)}
              sx={{ color: '#667eea' }}
            >
              <ExpandMore sx={{ transform: quickStatsVisible ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s' }} />
            </IconButton>
          </Box>
          
          {quickStatsVisible && (
            <Grid container spacing={{ xs: 1, sm: 2 }}>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      color: '#4CAF50', 
                      fontWeight: 700,
                      fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
                    }}
                  >
                    {playersData.length}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    color="textSecondary"
                    sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
                  >
                    Players Loaded
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      color: '#2196F3', 
                      fontWeight: 700,
                      fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
                    }}
                  >
                    {optimizationResults.length}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    color="textSecondary"
                    sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
                  >
                    Lineups Generated
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      color: '#FF9800', 
                      fontWeight: 700,
                      fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
                    }}
                  >
                    {isOptimizing ? `${systemStatus.optimizationProgress}%` : '✓'}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    color="textSecondary"
                    sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
                  >
                    Progress
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Box textAlign="center">
                  <Typography 
                    variant="h4" 
                    sx={{ 
                      color: systemStatus.connected ? '#4CAF50' : '#F44336', 
                      fontWeight: 700,
                      fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
                    }}
                  >
                    {systemStatus.connected ? '🟢' : '🔴'}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    color="textSecondary"
                    sx={{ fontSize: { xs: '0.7rem', sm: '0.75rem' } }}
                  >
                    Connection
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box 
        sx={{ 
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2d1b69 100%)'
        }}
      >
        {/* Enhanced Header Bar */}
        <AppBar 
          position="static" 
          elevation={0}
          sx={{ 
            background: 'rgba(10, 14, 39, 0.95)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid rgba(255,255,255,0.1)'
          }}
        >
          <Toolbar sx={{ minHeight: '72px !important' }}>
            <IconButton 
              edge="start" 
              color="inherit" 
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <Menu />
            </IconButton>
            
            <Sports sx={{ mr: 2, color: '#667eea', fontSize: 32 }} />
            <Typography variant="h5" component="div" sx={{ flexGrow: 1, fontWeight: 700 }}>
              Advanced MLB DFS Optimizer
            </Typography>
            
            {/* Status Indicators */}
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip 
                avatar={<Avatar sx={{ bgcolor: systemStatus.connected ? '#4CAF50' : '#F44336', width: 20, height: 20 }}>
                  {systemStatus.connected ? <CheckCircle sx={{ fontSize: 12 }} /> : <Error sx={{ fontSize: 12 }} />}
                </Avatar>}
                label={systemStatus.connected ? 'Connected' : 'Disconnected'}
                color={systemStatus.connected ? 'success' : 'error'}
                size="small"
                variant="outlined"
                sx={{ fontWeight: 600 }}
              />
              <Chip 
                avatar={<Avatar sx={{ bgcolor: '#2196F3', width: 20, height: 20 }}>
                  <People sx={{ fontSize: 12 }} />
                </Avatar>}
                label={`${systemStatus.playersLoaded} Players`}
                color="primary"
                size="small"
                variant="outlined"
                sx={{ fontWeight: 600 }}
              />
              {isOptimizing && (
                <Chip 
                  avatar={<Avatar sx={{ bgcolor: '#FF9800', width: 20, height: 20 }}>
                    <Speed sx={{ fontSize: 12 }} />
                  </Avatar>}
                  label={`${systemStatus.optimizationProgress}%`}
                  color="warning"
                  size="small"
                  variant="outlined"
                  sx={{ fontWeight: 600 }}
                />
              )}
            </Stack>
          </Toolbar>
        </AppBar>

        {/* Navigation Drawer */}
        <Drawer
          anchor="left"
          open={drawerOpen}
          onClose={toggleDrawer}
          sx={{
            '& .MuiDrawer-paper': {
              width: 280,
              background: 'rgba(26, 31, 58, 0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255,255,255,0.1)',
            },
          }}
        >
          <Box sx={{ p: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, color: '#667eea', mb: 2 }}>
              📊 DFS Navigation
            </Typography>
            <Divider sx={{ mb: 2, borderColor: 'rgba(255,255,255,0.1)' }} />
            
            <List>
              {tabs.map((tab, index) => (
                <ListItemButton
                  key={index}
                  selected={currentTab === index}
                  onClick={() => {
                    setCurrentTab(index);
                    setDrawerOpen(false);
                  }}
                  sx={{
                    borderRadius: 2,
                    mb: 1,
                    '&.Mui-selected': {
                      background: `linear-gradient(45deg, ${tab.color}20, ${tab.color}10)`,
                      borderLeft: `3px solid ${tab.color}`,
                    },
                    '&:hover': {
                      background: `${tab.color}10`,
                    }
                  }}
                >
                  <ListItemIcon sx={{ color: tab.color, minWidth: 40 }}>
                    {tab.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={tab.label} 
                    sx={{ 
                      '& .MuiListItemText-primary': { 
                        fontWeight: currentTab === index ? 600 : 400 
                      } 
                    }} 
                  />
                </ListItemButton>
              ))}
            </List>
            
            <Divider sx={{ my: 2, borderColor: 'rgba(255,255,255,0.1)' }} />
            
            {/* Quick Actions */}
            <Typography variant="subtitle2" sx={{ color: '#667eea', mb: 1, fontWeight: 600 }}>
              🚀 Quick Actions
            </Typography>
            <Stack spacing={1}>
              <Button
                startIcon={<CloudUpload />}
                variant="outlined"
                size="small"
                fullWidth
                sx={{ justifyContent: 'flex-start', borderColor: '#4CAF50', color: '#4CAF50' }}
              >
                Load Players
              </Button>
              <Button
                startIcon={<PlayArrow />}
                variant="outlined"
                size="small"
                fullWidth
                disabled={isOptimizing}
                sx={{ justifyContent: 'flex-start', borderColor: '#2196F3', color: '#2196F3' }}
              >
                {isOptimizing ? 'Optimizing...' : 'Run Optimization'}
              </Button>
              <Button
                startIcon={<Download />}
                variant="outlined"
                size="small"
                fullWidth
                sx={{ justifyContent: 'flex-start', borderColor: '#FF9800', color: '#FF9800' }}
              >
                Export Results
              </Button>
            </Stack>
          </Box>
        </Drawer>

        <Container 
          maxWidth="xl" 
          sx={{ 
            py: { xs: 2, sm: 2, md: 3 },
            px: { xs: 1, sm: 2, md: 3 }
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Breadcrumb Navigation */}
            <Breadcrumbs 
              separator={<NavigateNext fontSize="small" />}
              sx={{ mb: 2, color: 'rgba(255,255,255,0.7)' }}
            >
              <Link 
                color="inherit" 
                href="#"
                onClick={() => setCurrentTab(0)}
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  textDecoration: 'none',
                  '&:hover': { color: '#667eea' }
                }}
              >
                <Home sx={{ mr: 0.5, fontSize: 16 }} />
                DFS Optimizer
              </Link>
              <Typography color="primary" sx={{ display: 'flex', alignItems: 'center' }}>
                {tabs[currentTab]?.icon && React.cloneElement(tabs[currentTab].icon, { sx: { mr: 0.5, fontSize: 16 } })}
                {tabs[currentTab]?.label}
              </Typography>
            </Breadcrumbs>

            {/* Quick Stats Card */}
            <QuickStatsCard />

            {/* Responsive Multi-Column Layout */}
            <Grid container spacing={{ xs: 1, sm: 2, md: 2, lg: 3 }}>
              {/* Main Content - Responsive Width */}
              <Grid item xs={12} sm={12} md={6} lg={7} xl={8}>
                {/* Main Tab Navigation */}
                <Paper 
                  elevation={0} 
                  sx={{ 
                    mb: { xs: 2, sm: 2, md: 3 },
                    background: 'rgba(255,255,255,0.05)',
                    backdropFilter: 'blur(20px)',
                    borderRadius: { xs: 2, sm: 2, md: 3 }
                  }}
                >
                  <Tabs
                    value={currentTab}
                    onChange={handleTabChange}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{
                      '& .MuiTabs-indicator': {
                        background: 'linear-gradient(45deg, #667eea, #764ba2)',
                        height: 4,
                        borderRadius: 2,
                      },
                      '& .MuiTab-root': {
                        minHeight: 72,
                        fontWeight: 500,
                        textTransform: 'none',
                        fontSize: '0.95rem',
                      }
                    }}
                  >
                    {tabs.map((tab, index) => (
                      <Tab
                        key={index}
                        label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {tab.icon}
                            <Box>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {tab.label}
                              </Typography>
                              <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                                {index === 0 && `${playersData.length} players`}
                                {index === 1 && 'Stacking strategy'}
                                {index === 2 && 'Exposure control'}
                                {index === 3 && 'Team combinations'}
                                {index === 4 && 'My saved lineups'}
                                {index === 5 && `${optimizationResults.length} results`}
                              </Typography>
                            </Box>
                          </Box>
                        }
                        sx={{ 
                          px: 3,
                          '&.Mui-selected': {
                            color: tabs[index].color,
                            background: `linear-gradient(135deg, ${tabs[index].color}10, transparent)`
                          }
                        }}
                      />
                    ))}
                  </Tabs>
                </Paper>

                {/* Tab Content with Enhanced Animation */}
                <AnimatePresence mode="wait">
                  {tabs.map((tab, index) => (
                    <TabPanel key={index} value={currentTab} index={index}>
                      <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                      >
                        <Paper
                          elevation={0}
                          sx={{
                            p: { xs: 2, sm: 2, md: 3 },
                            background: 'rgba(255,255,255,0.03)',
                            backdropFilter: 'blur(20px)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            borderRadius: { xs: 2, sm: 2, md: 3 },
                            minHeight: { xs: '400px', sm: '450px', md: '500px' }
                          }}
                        >
                          {(() => {
                            switch (index) {
                              case 0: // Players tab
                                return (
                                  <PlayersTab 
                                    players={playersData}
                                    onPlayersUpdate={setPlayersData}
                                  />
                                );
                              case 1: // Team Stacks tab
                                return (
                                  <TeamStacksTab 
                                    teams={[]}
                                    stackSizes={[2, 3, 4, 5]}
                                    teamSelections={{}}
                                    onTeamSelectionsUpdate={(selections) => {
                                      console.log('Team selections updated:', selections);
                                    }}
                                    onTeamStacksUpdate={() => {}}
                                  />
                                );
                              case 2: // Stack Exposure tab
                                return (
                                  <StackExposureTab 
                                    teamStacks={[]}
                                    stackExposures={[]}
                                    onStackExposuresUpdate={() => {}}
                                    totalLineups={20}
                                  />
                                );
                              case 3: // Team Combos tab
                                return (
                                  <TeamCombosTab 
                                    availableTeams={[...new Set(playersData.map((p: any) => p.team))]}
                                    onCombinationsGenerated={(combinations) => {
                                      console.log('Generated combinations:', combinations);
                                    }}
                                    onGenerateLineups={(combinations) => {
                                      console.log('Generating lineups for combinations:', combinations);
                                      // TODO: Integrate with optimization API
                                    }}
                                    isGenerating={isOptimizing}
                                  />
                                );
                              case 4: // My Entries tab
                                return (
                                  <FavoritesTab 
                                    players={playersData}
                                    favoriteGroups={[]}
                                    onPlayersUpdate={setPlayersData}
                                    onFavoriteGroupsUpdate={() => {}}
                                  />
                                );
                              case 5: // Results tab
                                return (
                                  <ResultsTab 
                                    optimizationResults={optimizationResults}
                                    isOptimizing={isOptimizing}
                                    onExportResults={() => {
                                      console.log('Exporting results...');
                                      // TODO: Implement CSV export functionality
                                    }}
                                    onSaveToFavorites={(lineups) => {
                                      console.log('Saving to favorites:', lineups);
                                      // TODO: Implement save to favorites functionality
                                    }}
                                  />
                                );
                              default:
                                return <div>Tab not found</div>;
                            }
                          })()}
                        </Paper>
                      </motion.div>
                    </TabPanel>
                  ))}
                </AnimatePresence>
              </Grid>

              {/* Control Panel - Compact Sidebar */}
              <Grid item xs={12} sm={6} md={3} lg={2.5} xl={2}>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <Paper
                    elevation={0}
                    sx={{
                      p: { xs: 1, sm: 1.5, md: 1.5 },
                      background: 'rgba(244, 67, 54, 0.05)',
                      backdropFilter: 'blur(20px)',
                      border: '1px solid rgba(244, 67, 54, 0.2)',
                      borderRadius: { xs: 2, sm: 2, md: 2 },
                      position: { xs: 'relative', md: 'sticky' },
                      top: { xs: 0, md: 20 },
                      maxHeight: { xs: 'none', md: 'calc(50vh - 60px)' },
                      overflow: { xs: 'visible', md: 'auto' },
                      '& .MuiTypography-h6': {
                        fontSize: '0.9rem'
                      },
                      '& .MuiTypography-body2': {
                        fontSize: '0.75rem'
                      },
                      '& .MuiButton-root': {
                        fontSize: '0.75rem',
                        minHeight: '28px',
                        padding: '3px 8px'
                      },
                      '& .MuiTextField-root': {
                        '& .MuiInputBase-input': {
                          padding: '6px 8px',
                          fontSize: '0.8rem'
                        }
                      },
                      '& .MuiAccordion-root': {
                        '&:before': {
                          display: 'none'
                        },
                        boxShadow: 'none',
                        background: 'transparent'
                      },
                      '& .MuiAccordionSummary-root': {
                        minHeight: '32px',
                        '&.Mui-expanded': {
                          minHeight: '32px'
                        },
                        '& .MuiAccordionSummary-content': {
                          margin: '4px 0',
                          '&.Mui-expanded': {
                            margin: '4px 0'
                          }
                        }
                      },
                      '& .MuiAccordionDetails-root': {
                        padding: { xs: '4px 8px 8px', sm: '6px 12px 12px' }
                      }
                    }}
                  >
                    {/* Control Panel Header */}
                    <Box sx={{ 
                      mb: { xs: 0.5, sm: 0.5 }, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between',
                      gap: 1 
                    }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Settings sx={{ color: '#F44336', fontSize: { xs: 16, sm: 18 } }} />
                        <Typography variant="h6" sx={{ 
                          fontWeight: 600, 
                          color: '#F44336', 
                          fontSize: { xs: '0.8rem', sm: '0.85rem' } 
                        }}>
                          Controls
                        </Typography>
                      </Box>
                      {/* Mobile Collapse Button */}
                      <IconButton
                        size="small"
                        sx={{ 
                          display: { xs: 'flex', md: 'none' },
                          color: '#F44336'
                        }}
                        onClick={() => setMobileControlPanelCollapsed(!mobileControlPanelCollapsed)}
                      >
                          <ExpandMore sx={{ 
                            transform: mobileControlPanelCollapsed ? 'rotate(180deg)' : 'none',
                            transition: 'transform 0.3s'
                          }} />
                        </IconButton>
                      </Box>
                      <Divider sx={{ mb: { xs: 1, sm: 2 }, borderColor: 'rgba(244, 67, 54, 0.2)' }} />
                    
                    {/* Control Panel Content - Collapsible on Mobile */}
                    <Box 
                      data-control-panel
                      sx={{
                        display: { xs: mobileControlPanelCollapsed ? 'none' : 'block', md: 'block' },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <ControlPanelTab 
                        playersData={playersData}
                        setPlayersData={setPlayersData}
                        optimizationResults={optimizationResults}
                        isOptimizing={isOptimizing}
                        systemStatus={systemStatus}
                      />
                    </Box>
                  </Paper>
                </motion.div>
              </Grid>

              {/* Results Section - Right Sidebar */}
              <Grid item xs={12} sm={6} md={3} lg={2.5} xl={2}>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <Paper
                    elevation={0}
                    sx={{
                      p: { xs: 1, sm: 1.5, md: 1.5 },
                      background: 'rgba(76, 175, 80, 0.05)',
                      backdropFilter: 'blur(20px)',
                      border: '1px solid rgba(76, 175, 80, 0.2)',
                      borderRadius: { xs: 2, sm: 2, md: 2 },
                      position: { xs: 'relative', md: 'sticky' },
                      top: { xs: 0, md: 20 },
                      maxHeight: { xs: 'none', md: 'calc(100vh - 140px)' },
                      overflow: { xs: 'visible', md: 'auto' },
                    }}
                  >
                    {/* Results Header */}
                    <Box sx={{ 
                      mb: { xs: 0.5, sm: 1 }, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between',
                      gap: 1 
                    }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Analytics sx={{ color: '#4CAF50', fontSize: { xs: 16, sm: 18 } }} />
                        <Typography variant="h6" sx={{ 
                          fontWeight: 600, 
                          color: '#4CAF50', 
                          fontSize: { xs: '0.8rem', sm: '0.85rem' } 
                        }}>
                          Results
                        </Typography>
                      </Box>
                      <Chip 
                        size="small" 
                        label={`${optimizationResults.length}`}
                        sx={{ 
                          background: 'rgba(76, 175, 80, 0.2)',
                          color: '#4CAF50',
                          fontSize: '0.7rem'
                        }}
                      />
                    </Box>
                    <Divider sx={{ mb: { xs: 1, sm: 1.5 }, borderColor: 'rgba(76, 175, 80, 0.2)' }} />

                    {/* Quick Stats */}
                    <Stack spacing={1} sx={{ mb: 2 }}>
                      <Card 
                        elevation={0}
                        sx={{ 
                          p: 1, 
                          background: 'rgba(76, 175, 80, 0.1)',
                          border: '1px solid rgba(76, 175, 80, 0.2)'
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#4CAF50', fontWeight: 600 }}>
                          📊 Top Lineup Score
                        </Typography>
                        <Typography variant="h6" sx={{ color: '#4CAF50', fontSize: '1.1rem' }}>
                          {optimizationResults.length > 0 ? 
                            `${Math.max(...optimizationResults.map(r => r.projectedPoints || 0)).toFixed(1)} pts` : 
                            '0.0 pts'
                          }
                        </Typography>
                      </Card>

                      <Card 
                        elevation={0}
                        sx={{ 
                          p: 1, 
                          background: 'rgba(33, 150, 243, 0.1)',
                          border: '1px solid rgba(33, 150, 243, 0.2)'
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#2196F3', fontWeight: 600 }}>
                          💰 Salary Usage
                        </Typography>
                        <Typography variant="h6" sx={{ color: '#2196F3', fontSize: '1.1rem' }}>
                          {optimizationResults.length > 0 ? 
                            `${((optimizationResults[0]?.totalSalary || 0) / 50000 * 100).toFixed(1)}%` : 
                            '0%'
                          }
                        </Typography>
                      </Card>

                      <Card 
                        elevation={0}
                        sx={{ 
                          p: 1, 
                          background: 'rgba(255, 152, 0, 0.1)',
                          border: '1px solid rgba(255, 152, 0, 0.2)'
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#FF9800', fontWeight: 600 }}>
                          🎯 Optimization Status
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#FF9800', fontSize: '0.8rem' }}>
                          {isOptimizing ? 'Running...' : 'Ready'}
                        </Typography>
                      </Card>
                    </Stack>

                    {/* Recent Results List */}
                    <Typography variant="subtitle2" sx={{ 
                      color: '#4CAF50', 
                      mb: 1, 
                      fontWeight: 600,
                      fontSize: '0.8rem'
                    }}>
                      🏆 Recent Lineups
                    </Typography>
                    
                    <Stack spacing={0.5} sx={{ maxHeight: '200px', overflow: 'auto' }}>
                      {optimizationResults.length === 0 ? (
                        <Box sx={{ 
                          textAlign: 'center', 
                          py: 2,
                          color: 'rgba(255,255,255,0.5)'
                        }}>
                          <Typography variant="caption">
                            No results yet. Run optimization to see lineups here.
                          </Typography>
                        </Box>
                      ) : (
                        optimizationResults.slice(0, 5).map((result, index) => (
                          <Card 
                            key={index}
                            elevation={0}
                            sx={{ 
                              p: 0.5, 
                              background: 'rgba(255,255,255,0.03)',
                              border: '1px solid rgba(255,255,255,0.1)',
                              cursor: 'pointer',
                              transition: 'all 0.2s',
                              '&:hover': {
                                background: 'rgba(76, 175, 80, 0.1)',
                                borderColor: 'rgba(76, 175, 80, 0.3)'
                              }
                            }}
                            onClick={() => setCurrentTab(5)} // Go to results tab
                          >
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                Lineup #{index + 1}
                              </Typography>
                              <Typography variant="caption" sx={{ color: '#4CAF50' }}>
                                {result.projectedPoints?.toFixed(1) || '0.0'} pts
                              </Typography>
                            </Box>
                            <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                              ${result.totalSalary?.toLocaleString() || '0'}
                            </Typography>
                          </Card>
                        ))
                      )}
                    </Stack>

                    {/* Quick Actions */}
                    <Stack spacing={0.5} sx={{ mt: 2 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        fullWidth
                        startIcon={<Analytics />}
                        onClick={() => setCurrentTab(5)}
                        sx={{ 
                          borderColor: '#4CAF50', 
                          color: '#4CAF50',
                          fontSize: '0.7rem',
                          py: 0.5,
                          '&:hover': {
                            background: 'rgba(76, 175, 80, 0.1)'
                          }
                        }}
                      >
                        View All Results
                      </Button>
                      
                      <Button
                        size="small"
                        variant="outlined"
                        fullWidth
                        startIcon={<Download />}
                        disabled={optimizationResults.length === 0}
                        sx={{ 
                          borderColor: '#2196F3', 
                          color: '#2196F3',
                          fontSize: '0.7rem',
                          py: 0.5,
                          '&:hover': {
                            background: 'rgba(33, 150, 243, 0.1)'
                          }
                        }}
                      >
                        Export CSV
                      </Button>
                    </Stack>
                  </Paper>
                </motion.div>
              </Grid>
            </Grid>
          </motion.div>
        </Container>

        {/* Enhanced Status Bar */}
        <StatusBar 
          connectionStatus={systemStatus.connected ? 'connected' : 'disconnected'}
          optimizationStatus={isOptimizing ? 'running' : 'idle'}
          progress={systemStatus.optimizationProgress}
          totalPlayers={playersData.length}
          totalLineups={optimizationResults.length}
        />

        {/* Mobile Control Panel Toggle - Only visible on mobile */}
        <Fab
          color="primary"
          aria-label="toggle control panel"
          sx={{
            position: 'fixed',
            bottom: { xs: 16, md: 'auto' },
            right: { xs: 16, md: 'auto' },
            display: { xs: 'flex', md: 'none' },
            background: 'linear-gradient(45deg, #F44336, #d32f2f)',
            '&:hover': {
              background: 'linear-gradient(45deg, #d32f2f, #b71c1c)',
            },
            zIndex: 1000
          }}
          onClick={() => setMobileControlPanelCollapsed(!mobileControlPanelCollapsed)}
        >
          <Settings />
        </Fab>

        {/* Desktop Floating Action Button for Quick Access */}
        <SpeedDial
          ariaLabel="Quick actions"
          sx={{ 
            position: 'fixed', 
            bottom: 80, 
            right: 16,
            display: { xs: 'none', md: 'flex' },
            '& .MuiFab-primary': {
              background: 'linear-gradient(45deg, #667eea, #764ba2)',
            }
          }}
          icon={<SpeedDialIcon />}
        >
          <SpeedDialAction
            icon={<CloudUpload />}
            tooltipTitle="Load Players"
            onClick={() => setCurrentTab(0)} // Go to players tab
          />
          <SpeedDialAction
            icon={<PlayArrow />}
            tooltipTitle="Run Optimization"
            onClick={() => {
              // Control panel is always visible now
              const controlPanel = document.querySelector('[data-control-panel]');
              if (controlPanel) {
                controlPanel.scrollIntoView({ behavior: 'smooth' });
              }
            }}
          />
          <SpeedDialAction
            icon={<Download />}
            tooltipTitle="Export Results"
            onClick={() => {
              // Control panel is always visible now
              const controlPanel = document.querySelector('[data-control-panel]');
              if (controlPanel) {
                controlPanel.scrollIntoView({ behavior: 'smooth' });
              }
            }}
          />
          <SpeedDialAction
            icon={<Star />}
            tooltipTitle="My Entries"
            onClick={() => setCurrentTab(4)} // Go to favorites
          />
          <SpeedDialAction
            icon={<Analytics />}
            tooltipTitle="View Results"
            onClick={() => setCurrentTab(5)} // Go to results
          />
        </SpeedDial>

        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'rgba(26, 31, 58, 0.9)',
              color: '#fff',
              border: '1px solid rgba(255,255,255,0.1)',
              backdropFilter: 'blur(10px)',
            },
          }}
        />
      </Box>
    </ThemeProvider>
  );
}

export default App;
