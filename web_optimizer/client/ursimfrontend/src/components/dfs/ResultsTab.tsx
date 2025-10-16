import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  IconButton,
  Collapse,
  Grid,
  Card,
  CardContent,
  Divider,
  Stack,
  Alert,
  AlertTitle,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  KeyboardArrowDown,
  KeyboardArrowUp,
  Download,
  Star,
  Analytics,
  Assessment,
  AttachMoney,
  People,
  TrendingUp,
  Sports,
  Visibility,
  VisibilityOff
} from '@mui/icons-material';

interface Player {
  id: string;
  name: string;
  position: string;
  team: string;
  salary: number;
  projection: number;
}

interface Lineup {
  id: number;
  players: Player[];
  totalSalary: number;
  totalProjection: number;
  ownership?: number;
  uniqueness?: number;
}

interface ResultsTabProps {
  optimizationResults: Lineup[];
  isOptimizing: boolean;
  onExportResults: () => void;
  onSaveToFavorites: (lineups: Lineup[]) => void;
}

const ResultsTab: React.FC<ResultsTabProps> = ({
  optimizationResults,
  isOptimizing,
  onExportResults,
  onSaveToFavorites
}) => {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const [showTopOnly, setShowTopOnly] = useState(false);

  const toggleRowExpansion = (lineupId: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(lineupId)) {
      newExpanded.delete(lineupId);
    } else {
      newExpanded.add(lineupId);
    }
    setExpandedRows(newExpanded);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatProjection = (projection: number) => {
    return projection.toFixed(2);
  };

  const displayResults = showTopOnly ? optimizationResults.slice(0, 10) : optimizationResults;

  const ResultsHeader = () => (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Analytics sx={{ color: '#4CAF50', fontSize: 28 }} />
          <Typography variant="h5" sx={{ fontWeight: 600, color: '#4CAF50' }}>
            Optimization Results
          </Typography>
        </Box>
        
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={showTopOnly ? <Visibility /> : <VisibilityOff />}
            onClick={() => setShowTopOnly(!showTopOnly)}
            size="small"
            sx={{ borderColor: '#2196F3', color: '#2196F3' }}
          >
            {showTopOnly ? 'Show All' : 'Top 10 Only'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Star />}
            onClick={() => onSaveToFavorites(displayResults)}
            size="small"
            disabled={optimizationResults.length === 0}
            sx={{ borderColor: '#FF9800', color: '#FF9800' }}
          >
            Save to Favorites
          </Button>
          <Button
            variant="contained"
            startIcon={<Download />}
            onClick={onExportResults}
            size="small"
            disabled={optimizationResults.length === 0}
            sx={{ 
              background: 'linear-gradient(45deg, #4CAF50, #45a049)',
              '&:hover': { background: 'linear-gradient(45deg, #45a049, #3d8b40)' }
            }}
          >
            Export CSV
          </Button>
        </Stack>
      </Box>

      {/* Quick Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Assessment sx={{ color: '#4CAF50' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#4CAF50', fontWeight: 700 }}>
                    {optimizationResults.length}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Total Lineups
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'rgba(33, 150, 243, 0.1)', border: '1px solid rgba(33, 150, 243, 0.3)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp sx={{ color: '#2196F3' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#2196F3', fontWeight: 700 }}>
                    {optimizationResults.length > 0 ? formatProjection(Math.max(...optimizationResults.map(r => r.totalProjection))) : '0.00'}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Best Projection
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'rgba(255, 152, 0, 0.1)', border: '1px solid rgba(255, 152, 0, 0.3)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AttachMoney sx={{ color: '#FF9800' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#FF9800', fontWeight: 700 }}>
                    {optimizationResults.length > 0 ? formatCurrency(optimizationResults[0]?.totalSalary || 0) : '$0'}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Avg Salary Used
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'rgba(156, 39, 176, 0.1)', border: '1px solid rgba(156, 39, 176, 0.3)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <People sx={{ color: '#9C27B0' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#9C27B0', fontWeight: 700 }}>
                    {optimizationResults.length > 0 ? optimizationResults[0]?.players?.length || 0 : 0}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Players/Lineup
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const LineupRow = ({ lineup, index }: { lineup: Lineup; index: number }) => {
    const isExpanded = expandedRows.has(lineup.id);
    
    return (
      <>
        <TableRow 
          hover 
          sx={{ 
            '& > *': { borderBottom: 'unset' },
            '&:hover': { backgroundColor: 'rgba(255,255,255,0.05)' },
            cursor: 'pointer'
          }}
          onClick={() => toggleRowExpansion(lineup.id)}
        >
          <TableCell>
            <IconButton size="small" sx={{ color: '#667eea' }}>
              {isExpanded ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
            </IconButton>
          </TableCell>
          <TableCell>
            <Chip 
              label={`#${index + 1}`} 
              size="small" 
              color={index < 3 ? 'success' : 'default'}
              sx={{ fontWeight: 600, minWidth: 50 }}
            />
          </TableCell>
          <TableCell>
            <Typography variant="body2" sx={{ fontWeight: 600, color: '#4CAF50' }}>
              {formatProjection(lineup.totalProjection)}
            </Typography>
          </TableCell>
          <TableCell>
            <Typography variant="body2" sx={{ fontWeight: 600, color: '#FF9800' }}>
              {formatCurrency(lineup.totalSalary)}
            </Typography>
          </TableCell>
          <TableCell>
            <LinearProgress 
              variant="determinate" 
              value={(lineup.totalSalary / 50000) * 100} 
              sx={{ 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(255,255,255,0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: lineup.totalSalary > 49000 ? '#4CAF50' : '#2196F3'
                }
              }}
            />
          </TableCell>
          <TableCell>
            {lineup.uniqueness && (
              <Chip 
                label={`${lineup.uniqueness.toFixed(1)}%`} 
                size="small" 
                color="secondary"
                variant="outlined"
              />
            )}
          </TableCell>
        </TableRow>
        
        <TableRow>
          <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
            <Collapse in={isExpanded} timeout="auto" unmountOnExit>
              <Box sx={{ margin: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ color: '#667eea', fontWeight: 600, mb: 2 }}>
                  Lineup Details
                </Typography>
                <Grid container spacing={1}>
                  {lineup.players?.map((player, playerIndex) => (
                    <Grid item xs={12} sm={6} md={4} key={playerIndex}>
                      <Paper sx={{ 
                        p: 1.5, 
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.1)',
                        borderRadius: 2
                      }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <Box sx={{ flexGrow: 1 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#fff' }}>
                              {player.name}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                              <Chip 
                                label={player.position} 
                                size="small" 
                                color="primary" 
                                sx={{ fontSize: '0.7rem', height: 20 }}
                              />
                              <Chip 
                                label={player.team} 
                                size="small" 
                                color="secondary" 
                                sx={{ fontSize: '0.7rem', height: 20 }}
                              />
                            </Box>
                          </Box>
                          <Box sx={{ textAlign: 'right', minWidth: 80 }}>
                            <Typography variant="caption" sx={{ color: '#4CAF50', fontWeight: 600 }}>
                              {formatProjection(player.projection)}
                            </Typography>
                            <Typography variant="caption" sx={{ color: '#FF9800', display: 'block' }}>
                              {formatCurrency(player.salary)}
                            </Typography>
                          </Box>
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </Collapse>
          </TableCell>
        </TableRow>
      </>
    );
  };

  if (isOptimizing) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <LinearProgress sx={{ mb: 2, maxWidth: 400, mx: 'auto' }} />
        <Typography variant="h6" sx={{ color: '#667eea', mb: 1 }}>
          Optimization in Progress...
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Please wait while we generate your optimal lineups
        </Typography>
      </Box>
    );
  }

  if (optimizationResults.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Assessment sx={{ fontSize: 64, color: 'rgba(255,255,255,0.3)', mb: 2 }} />
        <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.7)', mb: 1 }}>
          No Results Yet
        </Typography>
        <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
          Run an optimization to see your generated lineups here
        </Typography>
        <Alert severity="info" sx={{ maxWidth: 400, mx: 'auto' }}>
          <AlertTitle>Getting Started</AlertTitle>
          Load player data and run optimization from the Control Panel to generate lineups.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <ResultsHeader />
      
      <TableContainer 
        component={Paper} 
        sx={{ 
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.1)',
          maxHeight: 600,
          '& .MuiTableCell-head': {
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderBottom: '1px solid rgba(102, 126, 234, 0.3)',
            fontWeight: 600
          }
        }}
      >
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell width="50px"></TableCell>
              <TableCell>Rank</TableCell>
              <TableCell>Projection</TableCell>
              <TableCell>Salary</TableCell>
              <TableCell width="150px">Salary Usage</TableCell>
              <TableCell>Uniqueness</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {displayResults.map((lineup, index) => (
              <LineupRow key={lineup.id} lineup={lineup} index={index} />
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      {showTopOnly && optimizationResults.length > 10 && (
        <Box sx={{ textAlign: 'center', mt: 2 }}>
          <Typography variant="body2" color="textSecondary">
            Showing top 10 of {optimizationResults.length} lineups
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ResultsTab; 