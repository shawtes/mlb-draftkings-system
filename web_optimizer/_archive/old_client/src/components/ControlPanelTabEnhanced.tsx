import React, { useState, useCallback } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Button, 
  Alert,
  LinearProgress,
  Grid,
  Paper,
  Chip,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Switch,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  IconButton,
  Tooltip,
  Stack
} from '@mui/material';
import { 
  Settings, 
  CloudUpload, 
  PlayArrow, 
  Stop,
  CheckCircle,
  Error as ErrorIcon,
  ExpandMore,
  AttachMoney,
  Security,
  Psychology,
  Assessment,
  Save,
  GetApp,
  Refresh,
  Star,
  Info,
  Warning
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface ControlPanelTabProps {
  onAction?: (action: string, data?: any) => void;
  playersData?: any[];
  setPlayersData?: (data: any[]) => void;
  optimizationResults?: any[];
  isOptimizing?: boolean;
  systemStatus?: any;
}

const ControlPanelTab: React.FC<ControlPanelTabProps> = ({ 
  onAction,
  playersData = [],
  setPlayersData,
  optimizationResults = [],
  isOptimizing = false,
  systemStatus
}) => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  
  // Optimization Settings (matching optimizer01.py)
  const [minUnique, setMinUnique] = useState(3);
  const [numLineups, setNumLineups] = useState(100);
  const [minSalary, setMinSalary] = useState(45000);
  const [maxSalary, setMaxSalary] = useState(50000);
  const [sortingMethod, setSortingMethod] = useState('Points');
  const [disableKelly, setDisableKelly] = useState(false);
  
  // Risk Management Settings
  const [bankroll, setBankroll] = useState(1000);
  const [riskTolerance, setRiskTolerance] = useState('medium');
  const [enableRiskManagement, setEnableRiskManagement] = useState(true);
  
  // Advanced Settings
  const [monteCarloIterations, setMonteCarloIterations] = useState(1000);
  const [stackExposure, setStackExposure] = useState(true);
  const [teamExposure, setTeamExposure] = useState(true);
  
  // File Upload Handler
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setIsUploading(true);
      setUploadStatus('Uploading...');
      setUploadProgress(0);
      
      const formData = new FormData();
      formData.append('file', file);
      
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 100);
      
      // Upload to backend
      fetch('/api/upload', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        setUploadProgress(100);
        setUploadStatus('Upload complete!');
        setIsUploading(false);
        
        if (data.players) {
          setPlayersData?.(data.players);
          onAction?.('players_loaded', data.players);
        }
      })
      .catch(error => {
        setUploadStatus('Upload failed: ' + error.message);
        setIsUploading(false);
      });
    }
  }, [onAction, setPlayersData]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls', '.xlsx']
    },
    multiple: false
  });
  
  const handleOptimization = () => {
    const settings = {
      minUnique,
      numLineups,
      minSalary,
      maxSalary,
      sortingMethod,
      disableKelly,
      bankroll,
      riskTolerance,
      enableRiskManagement,
      monteCarloIterations,
      stackExposure,
      teamExposure
    };
    
    onAction?.('run_optimization', settings);
  };
  
  const handleSalaryPreset = (value: number) => {
    setMinSalary(value);
  };
  
  const handleSaveResults = () => {
    onAction?.('save_results', { format: 'draftkings' });
  };
  
  const handleLoadDKPredictions = () => {
    onAction?.('load_dk_predictions');
  };
  
  const handleLoadDKEntries = () => {
    onAction?.('load_dk_entries');
  };
  
  const handleFillEntries = () => {
    onAction?.('fill_dk_entries');
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Settings sx={{ mr: 2, color: '#667eea', fontSize: 32 }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#667eea' }}>
            Control Panel
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Upload files, configure settings, and manage optimization
          </Typography>
        </Box>
      </Box>

      {/* File Upload Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom color="primary">
            📁 File Management
          </Typography>
          
          {/* CSV Upload */}
          <Box {...getRootProps()} sx={{
            border: '2px dashed #667eea',
            borderRadius: 2,
            p: 3,
            textAlign: 'center',
            cursor: 'pointer',
            bgcolor: isDragActive ? '#f5f5f5' : 'transparent',
            mb: 2
          }}>
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: '#667eea', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop CSV file here' : 'Drag & drop CSV file or click to select'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Supported formats: .csv, .xls, .xlsx
            </Typography>
          </Box>
          
          {/* Upload Progress */}
          {isUploading && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress variant="determinate" value={uploadProgress} />
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                {uploadStatus}
              </Typography>
            </Box>
          )}
          
          {/* Additional Load Buttons */}
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="outlined"
                color="secondary"
                onClick={handleLoadDKPredictions}
                startIcon={<Assessment />}
              >
                Load DK Predictions
              </Button>
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="outlined"
                color="info"
                onClick={handleLoadDKEntries}
                startIcon={<GetApp />}
              >
                Load DK Entries
              </Button>
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                fullWidth
                variant="outlined"
                color="success"
                onClick={handleFillEntries}
                startIcon={<Save />}
              >
                Fill DK Entries
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Optimization Settings */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom color="primary">
            ⚙️ Optimization Settings
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Number of Lineups"
                type="number"
                value={numLineups}
                onChange={(e) => setNumLineups(parseInt(e.target.value) || 100)}
                helperText="Number of lineups to generate (1-500)"
                inputProps={{ min: 1, max: 500 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Min Unique Players"
                type="number"
                value={minUnique}
                onChange={(e) => setMinUnique(parseInt(e.target.value) || 3)}
                helperText="Minimum unique players between lineups (0-10)"
                inputProps={{ min: 0, max: 10 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Sorting Method</InputLabel>
                <Select
                  value={sortingMethod}
                  label="Sorting Method"
                  onChange={(e) => setSortingMethod(e.target.value)}
                >
                  <MenuItem value="Points">Points</MenuItem>
                  <MenuItem value="Value">Value</MenuItem>
                  <MenuItem value="Salary">Salary</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={disableKelly}
                    onChange={(e) => setDisableKelly(e.target.checked)}
                  />
                }
                label="Disable Kelly Sizing"
              />
              <Typography variant="body2" color="textSecondary">
                Generate all requested lineups regardless of risk
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Salary Constraints */}
      <Card sx={{ mb: 3, bgcolor: '#E8F5E8' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ color: '#2E7D32' }}>
            💰 Salary Constraints
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                label="Minimum Salary"
                type="number"
                value={minSalary}
                onChange={(e) => setMinSalary(parseInt(e.target.value) || 45000)}
                helperText="Minimum total salary to spend"
                inputProps={{ min: 0, max: 50000 }}
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Stack direction="row" spacing={1}>
                <Button
                  size="small"
                  variant={minSalary === 40000 ? "contained" : "outlined"}
                  color="success"
                  onClick={() => handleSalaryPreset(40000)}
                >
                  40K
                </Button>
                <Button
                  size="small"
                  variant={minSalary === 45000 ? "contained" : "outlined"}
                  color="success"
                  onClick={() => handleSalaryPreset(45000)}
                >
                  45K
                </Button>
                <Button
                  size="small"
                  variant={minSalary === 48000 ? "contained" : "outlined"}
                  color="success"
                  onClick={() => handleSalaryPreset(48000)}
                >
                  48K
                </Button>
              </Stack>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Maximum Salary"
                type="number"
                value={maxSalary}
                onChange={(e) => setMaxSalary(parseInt(e.target.value) || 50000)}
                helperText="Maximum total salary to spend"
                inputProps={{ min: 0, max: 50000 }}
              />
            </Grid>
          </Grid>
          
          <Alert severity="info" sx={{ mt: 2 }}>
            💡 Higher minimum salaries force more expensive, potentially better players
          </Alert>
        </CardContent>
      </Card>

      {/* Risk Management */}
      <Card sx={{ mb: 3, bgcolor: '#FFF3E0' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ color: '#FF5722' }}>
            🔥 Risk Management
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Bankroll ($)"
                type="number"
                value={bankroll}
                onChange={(e) => setBankroll(parseInt(e.target.value) || 1000)}
                helperText="Your total bankroll for position sizing"
                inputProps={{ min: 100, max: 100000 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Risk Profile</InputLabel>
                <Select
                  value={riskTolerance}
                  label="Risk Profile"
                  onChange={(e) => setRiskTolerance(e.target.value)}
                >
                  <MenuItem value="conservative">Conservative</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="aggressive">Aggressive</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={enableRiskManagement}
                    onChange={(e) => setEnableRiskManagement(e.target.checked)}
                  />
                }
                label="Enable Advanced Risk Management"
              />
              <Typography variant="body2" color="textSecondary">
                Use Kelly criterion, volatility analysis, and portfolio theory
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      <Accordion sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">🔧 Advanced Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Monte Carlo Iterations"
                type="number"
                value={monteCarloIterations}
                onChange={(e) => setMonteCarloIterations(parseInt(e.target.value) || 1000)}
                helperText="Number of simulation iterations"
                inputProps={{ min: 100, max: 10000 }}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={stackExposure}
                      onChange={(e) => setStackExposure(e.target.checked)}
                    />
                  }
                  label="Stack Exposure Control"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={teamExposure}
                      onChange={(e) => setTeamExposure(e.target.checked)}
                    />
                  }
                  label="Team Exposure Control"
                />
              </Box>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Action Buttons */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Button
            fullWidth
            variant="contained"
            color="primary"
            size="large"
            onClick={handleOptimization}
            disabled={isOptimizing || playersData.length === 0}
            startIcon={isOptimizing ? <Stop /> : <PlayArrow />}
            sx={{ py: 2 }}
          >
            {isOptimizing ? 'Stop Optimization' : 'Run Contest Sim'}
          </Button>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Button
            fullWidth
            variant="contained"
            color="success"
            size="large"
            onClick={handleSaveResults}
            disabled={optimizationResults.length === 0}
            startIcon={<Save />}
            sx={{ py: 2 }}
          >
            Save CSV for DraftKings
          </Button>
        </Grid>
      </Grid>

      {/* System Status */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📊 System Status
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <Chip
                label={`${playersData.length} Players Loaded`}
                color={playersData.length > 0 ? 'success' : 'default'}
                icon={<CheckCircle />}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <Chip
                label={`${optimizationResults.length} Lineups Generated`}
                color={optimizationResults.length > 0 ? 'success' : 'default'}
                icon={<Assessment />}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <Chip
                label={isOptimizing ? 'Optimizing...' : 'Ready'}
                color={isOptimizing ? 'warning' : 'success'}
                icon={isOptimizing ? <Warning /> : <CheckCircle />}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <Chip
                label={enableRiskManagement ? 'Risk Mgmt On' : 'Risk Mgmt Off'}
                color={enableRiskManagement ? 'success' : 'default'}
                icon={<Security />}
              />
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ControlPanelTab;
