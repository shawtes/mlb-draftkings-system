import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Paper,
  Tabs,
  Tab,
  Button,
  Stack,
  Alert,
  Chip,
  Badge,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  TextField
} from '@mui/material';
import {
  Sports,
  Timeline,
  Group,
  CheckCircle,
  Assessment,
  TrendingUp,
  Refresh,
  Star
} from '@mui/icons-material';

interface Player {
  id: string;
  name: string;
  position: string;
  team: string;
  salary: number;
  projection: number;
  selected: boolean;
}

interface TeamStackSettings {
  team: string;
  selected: boolean;
  status: 'Active' | 'Inactive' | 'Limited';
  projectedRuns: number;
  minExposure: number;
  maxExposure: number;
  actualExposure: number;
  playerCount: number;
}

interface TeamStacksTabProps {
  players: Player[];
  teamStacks: any[];
  onTeamStacksUpdate: (stacks: {[key: string]: string[]}) => void;
}

const TeamStacksTab: React.FC<TeamStacksTabProps> = ({
  players = [],
  teamStacks = [],
  onTeamStacksUpdate
}) => {
  const [currentStackTab, setCurrentStackTab] = useState(0);
  const [teamStackSettings, setTeamStackSettings] = useState<{[stackSize: string]: {[team: string]: TeamStackSettings}}>({});
  const [loading, setLoading] = useState(false);

  const stackSizes = [
    { value: 'All Stacks', label: 'All Stacks', description: 'View and configure all team stack settings' },
    { value: '2 Stack', label: '2 Stack', description: 'Stack 2 players from the same team' },
    { value: '3 Stack', label: '3 Stack', description: 'Stack 3 players from the same team' },
    { value: '4 Stack', label: '4 Stack', description: 'Stack 4 players from the same team' },
    { value: '5 Stack', label: '5 Stack', description: 'Stack 5 players from the same team' }
  ];

  // Initialize team stack settings
  useEffect(() => {
    const uniqueTeams = [...new Set(players.map(p => p.team))].filter(Boolean);
    const newSettings: {[stackSize: string]: {[team: string]: TeamStackSettings}} = {};
    
    stackSizes.forEach(stackSize => {
      newSettings[stackSize.value] = {};
      uniqueTeams.forEach(team => {
        newSettings[stackSize.value][team] = {
          team,
          selected: false,
          status: 'Active',
          projectedRuns: Math.random() * 6 + 3,
          minExposure: 0,
          maxExposure: 100,
          actualExposure: 0,
          playerCount: players.filter(p => p.team === team).length
        };
      });
    });
    
    setTeamStackSettings(newSettings);
  }, [players]);

  const handleTeamSelection = (stackSize: string, team: string, selected: boolean) => {
    setTeamStackSettings(prev => ({
      ...prev,
      [stackSize]: {
        ...prev[stackSize],
        [team]: {
          ...prev[stackSize][team],
          selected
        }
      }
    }));
  };

  const handleExposureChange = (stackSize: string, team: string, field: 'minExposure' | 'maxExposure', value: number) => {
    setTeamStackSettings(prev => ({
      ...prev,
      [stackSize]: {
        ...prev[stackSize],
        [team]: {
          ...prev[stackSize][team],
          [field]: Math.max(0, Math.min(100, value))
        }
      }
    }));
  };

  const getSelectedTeamsCount = (stackSize: string) => {
    return Object.values(teamStackSettings[stackSize] || {}).filter(team => team.selected).length;
  };

  const getStackSizeColor = (stackSize: string) => {
    const colors = {
      'All Stacks': '#9C27B0',
      '2 Stack': '#2196F3',
      '3 Stack': '#4CAF50',
      '4 Stack': '#FF9800',
      '5 Stack': '#F44336'
    };
    return colors[stackSize as keyof typeof colors] || '#9C27B0';
  };

  const renderTeamStackTable = (stackSize: string) => {
    const teams = Object.values(teamStackSettings[stackSize] || {});
    
    if (teams.length === 0) {
      return (
        <Alert severity="info">
          No teams available. Please upload player data first.
        </Alert>
      );
    }

    return (
      <TableContainer component={Paper} sx={{ background: 'rgba(255,255,255,0.02)' }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Select</TableCell>
              <TableCell>Team</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Players</TableCell>
              <TableCell>Proj Runs</TableCell>
              <TableCell>Min Exp %</TableCell>
              <TableCell>Max Exp %</TableCell>
              <TableCell>Actual %</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {teams.map((team) => (
              <TableRow key={team.team}>
                <TableCell>
                  <Checkbox
                    checked={team.selected}
                    onChange={(e) => handleTeamSelection(stackSize, team.team, e.target.checked)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {team.team}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={team.status}
                    size="small"
                    color={team.status === 'Active' ? 'success' : 'default'}
                  />
                </TableCell>
                <TableCell>{team.playerCount}</TableCell>
                <TableCell>{team.projectedRuns.toFixed(1)}</TableCell>
                <TableCell>
                  <TextField
                    type="number"
                    value={team.minExposure}
                    onChange={(e) => handleExposureChange(stackSize, team.team, 'minExposure', Number(e.target.value))}
                    size="small"
                    inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ width: 70 }}
                  />
                </TableCell>
                <TableCell>
                  <TextField
                    type="number"
                    value={team.maxExposure}
                    onChange={(e) => handleExposureChange(stackSize, team.team, 'maxExposure', Number(e.target.value))}
                    size="small"
                    inputProps={{ min: 0, max: 100, step: 1 }}
                    sx={{ width: 70 }}
                  />
                </TableCell>
                <TableCell>{team.actualExposure.toFixed(1)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Sports sx={{ mr: 2, color: '#2196F3', fontSize: 32 }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#2196F3' }}>
            Team Stacks Configuration
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Configure team stacking strategies and exposure settings
          </Typography>
        </Box>
      </Box>

      {/* Quick Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Group sx={{ color: '#4CAF50' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#4CAF50', fontWeight: 600 }}>
                    {Object.keys(teamStackSettings['All Stacks'] || {}).length}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Available Teams
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
                <CheckCircle sx={{ color: '#2196F3' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#2196F3', fontWeight: 600 }}>
                    {getSelectedTeamsCount(stackSizes[currentStackTab]?.value || 'All Stacks')}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Selected Teams
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
                <Assessment sx={{ color: '#FF9800' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#FF9800', fontWeight: 600 }}>
                    {Object.values(teamStackSettings['All Stacks'] || {}).filter(t => t.status === 'Active').length}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Active Teams
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
                <TrendingUp sx={{ color: '#9C27B0' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#9C27B0', fontWeight: 600 }}>
                    {Object.values(teamStackSettings['All Stacks'] || {}).reduce((sum, t) => sum + t.projectedRuns, 0).toFixed(0)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Total Projected
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Stack Size Tabs */}
      <Paper sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Tabs
          value={currentStackTab}
          onChange={(_, newValue) => setCurrentStackTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            borderBottom: '1px solid rgba(255,255,255,0.1)',
            '& .MuiTab-root': {
              textTransform: 'none',
              fontWeight: 500,
              minHeight: 72
            }
          }}
        >
          {stackSizes.map((stackSize, index) => (
            <Tab
              key={stackSize.value}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Badge 
                    badgeContent={getSelectedTeamsCount(stackSize.value)} 
                    color="primary"
                    sx={{ '& .MuiBadge-badge': { fontSize: '0.7rem' } }}
                  >
                    <Chip
                      icon={
                        stackSize.value === 'All Stacks' ? <Timeline /> :
                        <Typography variant="caption" sx={{ fontWeight: 600 }}>
                          {stackSize.value.split(' ')[0]}
                        </Typography>
                      }
                      label={stackSize.label}
                      size="small"
                      sx={{ 
                        backgroundColor: getStackSizeColor(stackSize.value),
                        color: 'white',
                        '& .MuiChip-icon': { color: 'white' }
                      }}
                    />
                  </Badge>
                </Box>
              }
              sx={{ 
                '&.Mui-selected': {
                  background: `linear-gradient(135deg, ${getStackSizeColor(stackSize.value)}20, transparent)`
                }
              }}
            />
          ))}
        </Tabs>

        {/* Tab Content */}
        <Box sx={{ flex: 1, p: 3, overflow: 'auto' }}>
          {stackSizes.map((stackSize, index) => (
            <Box key={stackSize.value} hidden={currentStackTab !== index}>
              {currentStackTab === index && (
                <Box>
                  {/* Stack Size Description */}
                  <Alert severity="info" sx={{ mb: 3 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {stackSize.label}
                    </Typography>
                    <Typography variant="body2">
                      {stackSize.description}
                    </Typography>
                  </Alert>

                  {/* Team Stack Table */}
                  {renderTeamStackTable(stackSize.value)}
                </Box>
              )}
            </Box>
          ))}
        </Box>
      </Paper>

      {/* Footer Actions */}
      <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="body2" color="textSecondary">
          Configure team selections and exposure limits for each stack size
        </Typography>
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => {}}
            disabled={loading}
            sx={{ borderColor: '#2196F3', color: '#2196F3' }}
          >
            Refresh Team Stacks
          </Button>
          <Button
            variant="contained"
            startIcon={<Star />}
            onClick={() => {
              // Convert settings to match the expected format
              const formattedSettings = Object.entries(teamStackSettings).reduce((acc, [stackSize, teams]) => {
                acc[stackSize] = Object.values(teams).filter(team => team.selected).map(team => team.team);
                return acc;
              }, {} as {[key: string]: string[]});
              
              onTeamStacksUpdate(formattedSettings);
              console.log('Saving team stack settings:', formattedSettings);
            }}
            sx={{ 
              background: 'linear-gradient(45deg, #667eea, #764ba2)',
              '&:hover': {
                background: 'linear-gradient(45deg, #5a6fd8, #6a4190)'
              }
            }}
          >
            Save Configuration
          </Button>
        </Stack>
      </Box>
    </Box>
  );
};

export default TeamStacksTab; 