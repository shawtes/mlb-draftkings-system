import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Chip,
  Stack,
  Alert,
  Divider,
  LinearProgress
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Settings,
  TrendingUp,
  Speed
} from '@mui/icons-material';

interface ControlPanelTabProps {
  onStartOptimization?: () => void;
}

const ControlPanelTab: React.FC<ControlPanelTabProps> = ({ onStartOptimization }) => {
  const [numLineups, setNumLineups] = useState(20);
  const [maxSalary, setMaxSalary] = useState(50000);
  const [uniqueness, setUniqueness] = useState(50);
  const [optimizing, setOptimizing] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleStartOptimization = () => {
    setOptimizing(true);
    setProgress(0);
    
    // Simulate optimization progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setOptimizing(false);
          return 100;
        }
        return prev + 10;
      });
    }, 500);

    if (onStartOptimization) {
      onStartOptimization();
    }
  };

  const handleStopOptimization = () => {
    setOptimizing(false);
    setProgress(0);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, color: '#2196F3', mb: 1 }}>
          Control Panel
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Configure optimization settings and generate lineups
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ background: 'rgba(33, 150, 243, 0.1)' }}>
            <CardContent>
              <Typography variant="h6" color="primary">{numLineups}</Typography>
              <Typography variant="caption">Lineups to Generate</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ background: 'rgba(76, 175, 80, 0.1)' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#4CAF50' }}>${maxSalary.toLocaleString()}</Typography>
              <Typography variant="caption">Max Salary</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ background: 'rgba(156, 39, 176, 0.1)' }}>
            <CardContent>
              <Typography variant="h6" sx={{ color: '#9C27B0' }}>{uniqueness}%</Typography>
              <Typography variant="caption">Uniqueness</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Settings */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={3}>
            <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Settings /> Optimization Settings
            </Typography>
            <Divider />

            <TextField
              label="Number of Lineups"
              type="number"
              value={numLineups}
              onChange={(e) => setNumLineups(Number(e.target.value))}
              fullWidth
              inputProps={{ min: 1, max: 150 }}
            />

            <TextField
              label="Max Salary"
              type="number"
              value={maxSalary}
              onChange={(e) => setMaxSalary(Number(e.target.value))}
              fullWidth
              inputProps={{ min: 10000, max: 60000 }}
            />

            <Box>
              <Typography gutterBottom>Uniqueness: {uniqueness}%</Typography>
              <Slider
                value={uniqueness}
                onChange={(_, value) => setUniqueness(value as number)}
                min={0}
                max={100}
                valueLabelDisplay="auto"
              />
            </Box>

            <FormControl fullWidth>
              <InputLabel>Contest Type</InputLabel>
              <Select defaultValue="cash" label="Contest Type">
                <MenuItem value="cash">Cash Game</MenuItem>
                <MenuItem value="gpp">GPP/Tournament</MenuItem>
                <MenuItem value="h2h">Head-to-Head</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </CardContent>
      </Card>

      {/* Progress */}
      {optimizing && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2" sx={{ mb: 1 }}>
            Optimization in progress... {progress}%
          </Typography>
          <LinearProgress variant="determinate" value={progress} />
        </Alert>
      )}

      {/* Action Buttons */}
      <Stack direction="row" spacing={2}>
        {!optimizing ? (
          <Button
            variant="contained"
            size="large"
            startIcon={<PlayArrow />}
            onClick={handleStartOptimization}
            sx={{
              background: 'linear-gradient(45deg, #667eea, #764ba2)',
              '&:hover': {
                background: 'linear-gradient(45deg, #5a6fd8, #6a4190)',
              },
            }}
          >
            Start Optimization
          </Button>
        ) : (
          <Button
            variant="outlined"
            size="large"
            startIcon={<Stop />}
            onClick={handleStopOptimization}
            color="error"
          >
            Stop Optimization
          </Button>
        )}
        <Button variant="outlined" startIcon={<Speed />}>
          Quick Optimize
        </Button>
      </Stack>

      {/* Status Chips */}
      <Box sx={{ mt: 3 }}>
        <Stack direction="row" spacing={1}>
          <Chip icon={<TrendingUp />} label="Ready to Optimize" color="success" />
          <Chip label="Players: Loaded" color="primary" />
          <Chip label="Stacks: Configured" />
        </Stack>
      </Box>
    </Box>
  );
};

export default ControlPanelTab;


