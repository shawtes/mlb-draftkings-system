import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CloudDone,
  CloudOff,
  Sync,
  CheckCircle,
  Error,
  Warning,
  Info,
  Circle
} from '@mui/icons-material';

interface StatusBarProps {
  connectionStatus: 'disconnected' | 'connecting' | 'connected';
  optimizationStatus: 'idle' | 'running' | 'completed' | 'error';
  progress?: number;
  message?: string;
  totalPlayers?: number;
  selectedPlayers?: number;
  totalLineups?: number;
  completedLineups?: number;
  isConnected?: boolean;
  playersLoaded?: number;
  optimizationProgress?: number;
  isOptimizing?: boolean;
}

const StatusBar: React.FC<StatusBarProps> = ({
  connectionStatus,
  optimizationStatus,
  progress = 0,
  message = '',
  totalPlayers = 0,
  selectedPlayers = 0,
  totalLineups = 0,
  completedLineups = 0,
  isConnected = false,
  playersLoaded = 0,
  optimizationProgress = 0,
  isOptimizing = false
}) => {
  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <CloudDone color="success" />;
      case 'connecting':
        return <Sync color="warning" />;
      case 'disconnected':
      default:
        return <CloudOff color="error" />;
    }
  };

  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'success';
      case 'connecting':
        return 'warning';
      case 'disconnected':
      default:
        return 'error';
    }
  };

  const getOptimizationIcon = () => {
    switch (optimizationStatus) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'error':
        return <Error color="error" />;
      case 'running':
        return <Sync color="primary" />;
      case 'idle':
      default:
        return <Info color="disabled" />;
    }
  };

  const getOptimizationColor = () => {
    switch (optimizationStatus) {
      case 'completed':
        return 'success';
      case 'error':
        return 'error';
      case 'running':
        return 'primary';
      case 'idle':
      default:
        return 'default';
    }
  };

  const getOptimizationText = () => {
    switch (optimizationStatus) {
      case 'completed':
        return 'Optimization Complete';
      case 'error':
        return 'Optimization Error';
      case 'running':
        return 'Optimizing...';
      case 'idle':
      default:
        return 'Ready';
    }
  };

  return (
    <AppBar 
      position="fixed" 
      sx={{ 
        top: 'auto', 
        bottom: 0,
        bgcolor: 'background.paper',
        color: 'text.primary',
        borderTop: 1,
        borderColor: 'divider'
      }}
      elevation={1}
    >
      <Toolbar variant="dense" sx={{ minHeight: 48 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
          
          {/* Connection Status */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title={`Server ${connectionStatus}`}>
              <IconButton size="small">
                <Circle 
                  sx={{ 
                    fontSize: 8, 
                    color: isConnected ? '#4CAF50' : '#F44336' 
                  }} 
                />
              </IconButton>
            </Tooltip>
            <Chip
              size="small"
              label={isConnected ? 'Connected' : 'Disconnected'}
              color={getConnectionColor() as any}
              variant="outlined"
            />
          </Box>

          {/* Optimization Status */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Optimization status">
              <IconButton size="small">
                {getOptimizationIcon()}
              </IconButton>
            </Tooltip>
            <Chip
              size="small"
              label={getOptimizationText()}
              color={getOptimizationColor() as any}
              variant="outlined"
            />
          </Box>

          {/* Progress Bar */}
          {isOptimizing && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
              <Typography variant="caption">Optimizing...</Typography>
              <LinearProgress 
                variant="determinate" 
                value={optimizationProgress} 
                sx={{ flex: 1, height: 4, borderRadius: 2 }}
              />
              <Typography variant="caption">{optimizationProgress}%</Typography>
            </Box>
          )}

          {/* Player Stats */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="textSecondary">
              Players: {selectedPlayers}/{totalPlayers}
            </Typography>
            
            {totalLineups > 0 && (
              <Typography variant="body2" color="textSecondary">
                Lineups: {completedLineups}/{totalLineups}
              </Typography>
            )}
          </Box>

          {/* Status Message */}
          {message && (
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography 
                variant="body2" 
                color="textSecondary"
                sx={{ 
                  overflow: 'hidden', 
                  textOverflow: 'ellipsis', 
                  whiteSpace: 'nowrap' 
                }}
              >
                {message}
              </Typography>
            </Box>
          )}

          {/* Timestamp */}
          <Typography variant="caption" color="textSecondary">
            {new Date().toLocaleTimeString()}
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default StatusBar;
