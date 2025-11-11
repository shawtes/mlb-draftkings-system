// Legacy file - This component has been replaced by ControlPanelTabEnhanced.tsx
// This file is kept for compatibility but should not be used
import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Chip,
  Switch,
  FormControlLabel,
  Stack,
  CircularProgress,
  TextField,
  FormControl,
  Button
} from '@mui/material';
import {
  CloudUpload,
  PlayArrow,
  Stop,
  Download,
  FileUpload,
  Assessment,
  CheckCircle,
  Save,
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
    riskTolerance: 'medium' as 'conservative' | 'medium' | 'aggressive',
    sortingMethod: 'Points' as 'Points' | 'Value' | 'Salary'
  });

  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [kellyDisabled, setKellyDisabled] = useState(false);

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

      <Stack spacing={3}>
        <Box
          {...getRootProps()}
          sx={{
            p: 3,
            textAlign: 'center',
            cursor: 'pointer',
            border: '2px dashed',
            borderColor: isDragActive ? '#4CAF50' : 'rgba(255,255,255,0.3)',
            borderRadius: 2,
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: '#4CAF50',
              backgroundColor: 'rgba(76, 175, 80, 0.05)'
            },
            backgroundColor: isDragActive ? 'rgba(76, 175, 80, 0.08)' : 'transparent'
          }}
        >
          <input {...getInputProps()} />
          {uploading ? (
            <Stack alignItems="center" spacing={2}>
              <CircularProgress size={40} sx={{ color: '#4CAF50' }} />
              <Typography>Uploading players...</Typography>
            </Stack>
          ) : (
            <Stack alignItems="center" spacing={1.5}>
              <CloudUpload sx={{ fontSize: 48, color: '#4CAF50' }} />
              <Typography variant="h6">
                {isDragActive ? 'Drop CSV file here' : 'Load CSV / Drag & Drop Players File'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Supports CSV files with player data
              </Typography>
              {uploadedFile && (
                <Chip
                  label={uploadedFile.name}
                  color="success"
                  icon={<CheckCircle />}
                  sx={{ mt: 1 }}
                />
              )}
            </Stack>
          )}
        </Box>

        <Stack spacing={2}>
          <TextField
            label="Min Unique Players"
            type="number"
            value={optimizationSettings.uniquePlayers}
            onChange={(e) =>
              setOptimizationSettings((prev) => ({
                ...prev,
                uniquePlayers: Number(e.target.value),
              }))
            }
            helperText="0-10 unique players between lineups"
            fullWidth
          />

          <TextField
            label="Number of Lineups"
            type="number"
            value={optimizationSettings.numLineups}
            onChange={(e) =>
              setOptimizationSettings((prev) => ({
                ...prev,
                numLineups: Number(e.target.value),
              }))
            }
            helperText="1-500 lineups (more take longer)"
            fullWidth
          />

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

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <TextField
              label="Minimum Salary"
              type="number"
              value={optimizationSettings.minSalary}
              onChange={(e) =>
                setOptimizationSettings((prev) => ({
                  ...prev,
                  minSalary: Number(e.target.value),
                }))
              }
              helperText="Force higher budget usage"
              fullWidth
            />
            <TextField
              label="Maximum Salary"
              type="number"
              value={optimizationSettings.maxSalary}
              onChange={(e) =>
                setOptimizationSettings((prev) => ({
                  ...prev,
                  maxSalary: Number(e.target.value),
                }))
              }
              helperText="Salary cap limit"
              fullWidth
            />
          </Stack>

          <FormControl fullWidth>
            <InputLabel id="sorting-method-label">Sorting Method</InputLabel>
            <Select
              labelId="sorting-method-label"
              label="Sorting Method"
              value={optimizationSettings.sortingMethod}
              onChange={(e) =>
                setOptimizationSettings((prev) => ({
                  ...prev,
                  sortingMethod: e.target.value as 'Points' | 'Value' | 'Salary',
                }))
              }
            >
              <MenuItem value="Points">Points</MenuItem>
              <MenuItem value="Value">Value</MenuItem>
              <MenuItem value="Salary">Salary</MenuItem>
            </Select>
          </FormControl>
        </Stack>

        <Stack spacing={2}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <Button
              variant="contained"
              startIcon={isOptimizing ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
              disabled={isOptimizing || playersData.length === 0}
              onClick={handleOptimization}
              sx={{
                flex: 1,
                py: 1.8,
                fontSize: '1rem',
                fontWeight: 600,
                background: 'linear-gradient(45deg, #4CAF50, #45a049)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #45a049, #4CAF50)',
                },
              }}
            >
              {isOptimizing ? 'Running Contest Sim...' : 'Run Contest Sim (Optimize)'}
            </Button>
            <Button
              variant="outlined"
              startIcon={<Stop />}
              disabled={!isOptimizing}
              sx={{
                flex: 1,
                py: 1.8,
                borderColor: '#F44336',
                color: '#F44336',
                '&:hover': {
                  borderColor: '#d32f2f',
                  backgroundColor: 'rgba(244, 67, 54, 0.04)',
                },
              }}
            >
              Stop Optimization
            </Button>
          </Stack>

          <Button
            variant="outlined"
            startIcon={<Download />}
            sx={{ justifyContent: 'flex-start', py: 1.6 }}
            fullWidth
          >
            Save CSV for DraftKings
          </Button>

          <Button
            variant="outlined"
            startIcon={<FileUpload />}
            sx={{ justifyContent: 'flex-start', py: 1.6 }}
            fullWidth
          >
            Load DraftKings Entries File
          </Button>

          <Button
            variant="outlined"
            startIcon={<Assessment />}
            sx={{ justifyContent: 'flex-start', py: 1.6 }}
            fullWidth
          >
            Fill Entries with Optimized Lineups
          </Button>
        </Stack>

        <Stack spacing={1.5}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Favorites Management
          </Typography>

          <Button
            variant="outlined"
            startIcon={<Favorite />}
            sx={{ justifyContent: 'flex-start', py: 1.6, borderColor: '#FF9800', color: '#FF9800' }}
            fullWidth
          >
            Add Current to Favorites
          </Button>

          <Button
            variant="outlined"
            startIcon={<Save />}
            sx={{ justifyContent: 'flex-start', py: 1.6, borderColor: '#9C27B0', color: '#9C27B0' }}
            fullWidth
          >
            Export Favorites as New Lineups
          </Button>
        </Stack>

        <Stack spacing={1.5}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            📈 System Status
          </Typography>

          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="body2" color="text.secondary">
              Players Loaded
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {playersData.length}
            </Typography>
          </Box>

          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="body2" color="text.secondary">
              Lineups Generated
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {optimizationResults.length}
            </Typography>
          </Box>

          {isOptimizing && (
            <Stack spacing={1}>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  Progress
                </Typography>
                <Typography variant="body2">{systemStatus.optimizationProgress}%</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={systemStatus.optimizationProgress}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Stack>
          )}
        </Stack>
      </Stack>
    </motion.div>
  );
};

export default ControlPanelTab;
