// Legacy file - This component has been replaced by ControlPanelTabEnhanced.tsx
// This file is kept for compatibility but should not be used
import React, { useState, useCallback } from 'react';
import { 
  Box, 
  Typography, 
  Alert, 
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Chip,
  Divider,
  Paper,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Avatar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  ButtonGroup,
  Grid,
  TextField,
  Card,
  CardContent,
  FormControl,
  Button
} from '@mui/material';
import {
  CloudUpload,
  PlayArrow,
  Stop,
  Download,
  Settings,
  TrendingUp,
  AttachMoney,
  Group,
  ExpandMore,
  Security,
  Speed,
  MonetizationOn,
  Psychology,
  FileUpload,
  GetApp,
  MoneyOff,
  Assessment,
  CheckCircle,
  Info,
  Error,
  Save,
  TableChart,
  BarChart,
  Favorite
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';

interface ControlPanelTabProps {
  playersData: any[];
  setPlayersData: (data: any[]) => void;
  optimizationResults: any[];
  isOptimizing: boolean;
  systemStatus: any;
}

const ControlPanelTab: React.FC<ControlPanelTabProps> = ({
  playersData,
  setPlayersData,
  optimizationResults,
  isOptimizing,
  systemStatus
}) => {
  const [optimizationSettings, setOptimizationSettings] = useState({
    numLineups: 100,
    minSalary: 45000,
    maxSalary: 50000,
    uniquePlayers: 3,
    maxExposure: 40,
    stackSize: 4,
    minPoints: 1,
    bankroll: 1000,
    riskTolerance: 'medium' as 'conservative' | 'medium' | 'aggressive'
  });

  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [riskEngineEnabled, setRiskEngineEnabled] = useState(true);
  const [kellyDisabled, setKellyDisabled] = useState(false);
  const [accordionExpanded, setAccordionExpanded] = useState({
    fileManagement: true,
    optimization: true,
    riskManagement: false,
    results: false
  });

  const handleAccordionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setAccordionExpanded(prev => ({ ...prev, [panel]: isExpanded }));
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadedFile(file);
    setUploading(true);

    const formData = new FormData();
    formData.append('playersFile', file);

    try {
      const response = await fetch('/api/upload-players', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        toast.success(`✅ Loaded ${result.playersCount} players successfully!`);
              const playersResponse = await fetch('/api/players');
      const responseData = await playersResponse.json();
      console.log('API Response:', responseData); // Debug log
      setPlayersData(responseData.players || []);
      } else {
        toast.error(`❌ ${result.error || 'Upload failed'}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('❌ Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  }, [setPlayersData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  const handleOptimization = async () => {
    if (playersData.length === 0) {
      toast.error('Please upload player data first');
      return;
    }

    try {
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(optimizationSettings),
      });

      const result = await response.json();

      if (result.success) {
        toast.success(`🚀 Generated ${result.lineups?.length || 0} lineups!`);
      } else {
        toast.error(`❌ ${result.error || 'Optimization failed'}`);
      }
    } catch (error) {
      console.error('Optimization error:', error);
      toast.error('❌ Optimization failed. Please try again.');
    }
  };

  // Calculate statistics
  const selectedCount = playersData.filter((player: any) => player.selected).length;
  const avgSalary = playersData.length > 0 
    ? playersData.reduce((sum: number, player: any) => sum + (player.salary || 0), 0) / playersData.length 
    : 0;
  const avgProjection = playersData.length > 0 
    ? playersData.reduce((sum: number, player: any) => sum + (player.projection || 0), 0) / playersData.length 
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Typography variant="h4" gutterBottom sx={{ 
        fontWeight: 700, 
        background: 'linear-gradient(45deg, #667eea, #764ba2)',
        backgroundClip: 'text',
        WebkitBackgroundClip: 'text',
        color: 'transparent',
        mb: 3
      }}>
        🎯 Control Panel
      </Typography>

      <Grid container spacing={3}>
        {/* Left Column - Controls */}
        <Grid item xs={12} md={8}>
          {/* File Management Section */}
          <Accordion 
            expanded={accordionExpanded.fileManagement} 
            onChange={handleAccordionChange('fileManagement')}
            sx={{ mb: 2, background: 'rgba(76, 175, 80, 0.1)', borderLeft: '4px solid #4CAF50' }}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box display="flex" alignItems="center" gap={2}>
                <Avatar sx={{ bgcolor: '#4CAF50', width: 40, height: 40 }}>
                  <FileUpload />
                </Avatar>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    📁 File Management
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Load player data and manage files
                  </Typography>
                </Box>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Paper
                    {...getRootProps()}
                    sx={{
                      p: 3,
                      textAlign: 'center',
                      cursor: 'pointer',
                      border: '2px dashed',
                      borderColor: isDragActive ? '#4CAF50' : 'rgba(255,255,255,0.3)',
                      backgroundColor: isDragActive ? 'rgba(76, 175, 80, 0.1)' : 'transparent',
                      borderRadius: 2,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.05)',
                      }
                    }}
                  >
                    <input {...getInputProps()} />
                    {uploading ? (
                      <Box>
                        <CircularProgress size={40} sx={{ color: '#4CAF50', mb: 2 }} />
                        <Typography>Uploading players...</Typography>
                      </Box>
                    ) : (
                      <Box>
                        <CloudUpload sx={{ fontSize: 48, color: '#4CAF50', mb: 2 }} />
                        <Typography variant="h6" gutterBottom>
                          {isDragActive ? 'Drop CSV file here' : 'Load CSV / Drag & Drop Players File'}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Supports CSV files with player data
                        </Typography>
                        {uploadedFile && (
                          <Chip 
                            label={uploadedFile.name} 
                            color="success" 
                            sx={{ mt: 2 }}
                            icon={<CheckCircle />}
                          />
                        )}
                      </Box>
                    )}
                  </Paper>
                </Grid>
                
                <Grid item xs={12}>
                  <ButtonGroup variant="outlined" fullWidth>
                    <Button 
                      startIcon={<TableChart />}
                      sx={{ borderColor: '#2196F3', color: '#2196F3' }}
                    >
                      Load DraftKings Predictions
                    </Button>
                    <Button 
                      startIcon={<BarChart />}
                      sx={{ borderColor: '#9C27B0', color: '#9C27B0' }}
                    >
                      Load Entries CSV
                    </Button>
                    <Button 
                      startIcon={<Assessment />}
                      sx={{ borderColor: '#FF9800', color: '#FF9800' }}
                    >
                      Load DK Entries File
                    </Button>
                  </ButtonGroup>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Optimization Settings */}
          <Accordion 
            expanded={accordionExpanded.optimization} 
            onChange={handleAccordionChange('optimization')}
            sx={{ mb: 2, background: 'rgba(33, 150, 243, 0.1)', borderLeft: '4px solid #2196F3' }}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box display="flex" alignItems="center" gap={2}>
                <Avatar sx={{ bgcolor: '#2196F3', width: 40, height: 40 }}>
                  <Settings />
                </Avatar>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    ⚙️ Optimization Settings
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Configure lineup generation parameters
                  </Typography>
                </Box>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Number of Lineups"
                    type="number"
                    value={optimizationSettings.numLineups}
                    onChange={(e) => setOptimizationSettings(prev => ({ 
                      ...prev, 
                      numLineups: Number(e.target.value) 
                    }))}
                    helperText="1-500 lineups (more take longer)"
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Min Unique Players"
                    type="number"
                    value={optimizationSettings.uniquePlayers}
                    onChange={(e) => setOptimizationSettings(prev => ({ 
                      ...prev, 
                      uniquePlayers: Number(e.target.value) 
                    }))}
                    helperText="0-10 unique players between lineups"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Minimum Salary"
                    type="number"
                    value={optimizationSettings.minSalary}
                    onChange={(e) => setOptimizationSettings(prev => ({ 
                      ...prev, 
                      minSalary: Number(e.target.value) 
                    }))}
                    helperText="Force higher budget usage"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Maximum Salary"
                    type="number"
                    value={optimizationSettings.maxSalary}
                    onChange={(e) => setOptimizationSettings(prev => ({ 
                      ...prev, 
                      maxSalary: Number(e.target.value) 
                    }))}
                    helperText="Salary cap limit"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Sorting Method</InputLabel>
                    <Select value="Points" onChange={() => {}}>
                      <MenuItem value="Points">Points</MenuItem>
                      <MenuItem value="Value">Value</MenuItem>
                      <MenuItem value="Salary">Salary</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={kellyDisabled}
                        onChange={(e) => setKellyDisabled(e.target.checked)}
                        color="warning"
                      />
                    }
                    label="Disable Kelly Sizing (Generate All Requested)"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Right Column - Status and Actions */}
        <Grid item xs={12} md={4}>
          {/* Status Card */}
          <Card sx={{ mb: 3, background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, color: '#667eea' }}>
                📈 System Status
              </Typography>
              
              <Stack spacing={2}>
                <Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2">Players Loaded</Typography>
                    <Typography variant="h6" sx={{ color: '#4CAF50', fontWeight: 600 }}>
                      {playersData.length}
                    </Typography>
                  </Box>
                  
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2">Lineups Generated</Typography>
                    <Typography variant="h6" sx={{ color: '#2196F3', fontWeight: 600 }}>
                      {optimizationResults.length}
                    </Typography>
                  </Box>
                  
                  {isOptimizing && (
                    <Box>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                        <Typography variant="body2">Progress</Typography>
                        <Typography variant="body2">{systemStatus.optimizationProgress}%</Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={systemStatus.optimizationProgress} 
                        sx={{ height: 8, borderRadius: 4 }}
                      />
                    </Box>
                  )}
                </Box>
              </Stack>
            </CardContent>
          </Card>

          {/* Main Action Buttons */}
          <Stack spacing={2}>
            <Button
              variant="contained"
              size="large"
              fullWidth
              startIcon={isOptimizing ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
              disabled={isOptimizing || playersData.length === 0}
              onClick={handleOptimization}
              sx={{
                py: 2,
                background: 'linear-gradient(45deg, #4CAF50, #45a049)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #45a049, #4CAF50)',
                },
                fontSize: '1.1rem',
                fontWeight: 600
              }}
            >
              {isOptimizing ? 'Running Contest Sim...' : 'Run Contest Sim'}
            </Button>

            <Button
              variant="outlined"
              size="large"
              fullWidth
              startIcon={<Stop />}
              disabled={!isOptimizing}
              sx={{
                py: 2,
                borderColor: '#F44336',
                color: '#F44336',
                '&:hover': {
                  borderColor: '#d32f2f',
                  backgroundColor: 'rgba(244, 67, 54, 0.04)'
                }
              }}
            >
              Stop Optimization
            </Button>

            <Divider sx={{ my: 2 }} />

            <Typography variant="h6" sx={{ fontWeight: 600, color: '#FF9800', mb: 1 }}>
              🌟 Favorites Management
            </Typography>

            <Button
              variant="outlined"
              fullWidth
              startIcon={<Favorite />}
              sx={{ borderColor: '#FF9800', color: '#FF9800' }}
            >
              Add Current to Favorites
            </Button>

            <Button
              variant="outlined"
              fullWidth
              startIcon={<Save />}
              sx={{ borderColor: '#9C27B0', color: '#9C27B0' }}
            >
              Export Favorites as New Lineups
            </Button>
          </Stack>
        </Grid>
      </Grid>
    </motion.div>
  );
};

export default ControlPanelTab;
