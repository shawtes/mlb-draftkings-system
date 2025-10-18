import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  TextField,
  Paper,
  Tabs,
  Tab,
  Grid,
  Chip,
  FormControlLabel,
  Switch,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Player } from '../types';

interface TeamStackSettings {
  team: string;
  selected: boolean;
  status: 'Active' | 'Inactive' | 'Limited';
  projectedRuns: number;
  minExposure: number;
  maxExposure: number;
  actualExposure: number;
  playerCount: number;
  stackSize: number;
}

interface TeamStacksTabProps {
  players: Player[];
  teamStacks: any[];
  onTeamStacksUpdate: (stacks: { [key: string]: string[] }) => void;
}

const TeamStacksTab: React.FC<TeamStacksTabProps> = ({
  players,
  teamStacks,
  onTeamStacksUpdate
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [stackSizes] = useState([2, 3, 4, 5, 6]);
  const [teamStackData, setTeamStackData] = useState<{ [key: number]: TeamStackSettings[] }>({});
  const [enableAutoBalancing, setEnableAutoBalancing] = useState(true);
  const [globalMinExposure, setGlobalMinExposure] = useState(0);
  const [globalMaxExposure, setGlobalMaxExposure] = useState(100);

  // Initialize team data based on available players
  useEffect(() => {
    if (players && players.length > 0) {
      const uniqueTeams = [...new Set(players.map(p => p.team).filter(Boolean))];
      
      const initialData: { [key: number]: TeamStackSettings[] } = {};
      
      stackSizes.forEach(stackSize => {
        initialData[stackSize] = uniqueTeams.map(team => ({
          team,
          selected: false,
          status: 'Active' as const,
          projectedRuns: Math.floor(Math.random() * 10) + 5,
          minExposure: 0,
          maxExposure: 100,
          actualExposure: 0,
          playerCount: players.filter(p => p.team === team).length,
          stackSize
        }));
      });
      
      setTeamStackData(initialData);
    }
  }, [players, stackSizes]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleTeamSelection = (stackSize: number, team: string, selected: boolean) => {
    setTeamStackData(prev => ({
      ...prev,
      [stackSize]: prev[stackSize]?.map(teamData =>
        teamData.team === team ? { ...teamData, selected } : teamData
      ) || []
    }));
    
    // Update the parent component with the new selections
    updateParentStacks();
  };

  const handleExposureChange = (stackSize: number, team: string, field: 'minExposure' | 'maxExposure', value: number) => {
    setTeamStackData(prev => ({
      ...prev,
      [stackSize]: prev[stackSize]?.map(teamData =>
        teamData.team === team ? { ...teamData, [field]: value } : teamData
      ) || []
    }));
  };

  const handleGlobalExposureApply = () => {
    setTeamStackData(prev => {
      const updated = { ...prev };
      stackSizes.forEach(stackSize => {
        if (updated[stackSize]) {
          updated[stackSize] = updated[stackSize].map(team => ({
            ...team,
            minExposure: globalMinExposure,
            maxExposure: globalMaxExposure
          }));
        }
      });
      return updated;
    });
  };

  const updateParentStacks = () => {
    const stacks: { [key: string]: string[] } = {};
    
    stackSizes.forEach(stackSize => {
      const selectedTeams = teamStackData[stackSize]?.filter(team => team.selected).map(team => team.team) || [];
      if (selectedTeams.length > 0) {
        stacks[`${stackSize}_team_stacks`] = selectedTeams;
      }
    });
    
    onTeamStacksUpdate(stacks);
  };

  const getStackSummary = (stackSize: number) => {
    const teams = teamStackData[stackSize] || [];
    const selectedCount = teams.filter(t => t.selected).length;
    const totalTeams = teams.length;
    const avgExposure = teams.length > 0 ? teams.reduce((sum, t) => sum + t.actualExposure, 0) / teams.length : 0;
    
    return {
      selectedCount,
      totalTeams,
      avgExposure: Math.round(avgExposure * 100) / 100
    };
  };

  const currentStackSize = stackSizes[activeTab];
  const currentTeams = teamStackData[currentStackSize] || [];
  const summary = getStackSummary(currentStackSize);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Team Stacks Configuration
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure team stacking strategies for different stack sizes. Select teams to stack together and set exposure limits.
      </Alert>

      {/* Global Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Global Settings
          </Typography>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} sm={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={enableAutoBalancing}
                    onChange={(e) => setEnableAutoBalancing(e.target.checked)}
                  />
                }
                label="Auto Balancing"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                label="Global Min Exposure %"
                type="number"
                value={globalMinExposure}
                onChange={(e) => setGlobalMinExposure(Number(e.target.value))}
                inputProps={{ min: 0, max: 100 }}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <TextField
                label="Global Max Exposure %"
                type="number"
                value={globalMaxExposure}
                onChange={(e) => setGlobalMaxExposure(Number(e.target.value))}
                inputProps={{ min: 0, max: 100 }}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={12} sm={3}>
              <Button
                variant="contained"
                onClick={handleGlobalExposureApply}
                fullWidth
              >
                Apply to All
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Stack Size Tabs */}
      <Card>
        <CardContent>
          <Tabs value={activeTab} onChange={handleTabChange} variant="fullWidth">
            {stackSizes.map((size, index) => (
              <Tab
                key={size}
                label={
                  <Box>
                    <Typography variant="body2">{size}-Team Stacks</Typography>
                    <Chip
                      label={`${summary.selectedCount}/${summary.totalTeams} selected`}
                      size="small"
                      color={summary.selectedCount > 0 ? "primary" : "default"}
                    />
                  </Box>
                }
              />
            ))}
          </Tabs>

          <Divider sx={{ my: 2 }} />

          {/* Stack Summary */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={4}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="primary">
                    {summary.selectedCount}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Teams Selected
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="secondary">
                    {summary.totalTeams}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Available
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="success.main">
                    {summary.avgExposure}%
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Avg Exposure
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Team Selection Table */}
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      indeterminate={summary.selectedCount > 0 && summary.selectedCount < summary.totalTeams}
                      checked={summary.selectedCount === summary.totalTeams && summary.totalTeams > 0}
                      onChange={(e) => {
                        const allSelected = e.target.checked;
                        currentTeams.forEach(team => {
                          handleTeamSelection(currentStackSize, team.team, allSelected);
                        });
                      }}
                    />
                  </TableCell>
                  <TableCell><strong>Team</strong></TableCell>
                  <TableCell align="center"><strong>Status</strong></TableCell>
                  <TableCell align="center"><strong>Players</strong></TableCell>
                  <TableCell align="center"><strong>Proj. Runs</strong></TableCell>
                  <TableCell align="center"><strong>Min %</strong></TableCell>
                  <TableCell align="center"><strong>Max %</strong></TableCell>
                  <TableCell align="center"><strong>Actual %</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {currentTeams.map((teamData) => (
                  <TableRow
                    key={teamData.team}
                    hover
                    selected={teamData.selected}
                  >
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={teamData.selected}
                        onChange={(e) => handleTeamSelection(currentStackSize, teamData.team, e.target.checked)}
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="body2" fontWeight="medium">
                          {teamData.team}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={teamData.status}
                        size="small"
                        color={teamData.status === 'Active' ? 'success' : teamData.status === 'Limited' ? 'warning' : 'error'}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2">
                        {teamData.playerCount}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2">
                        {teamData.projectedRuns}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <TextField
                        type="number"
                        value={teamData.minExposure}
                        onChange={(e) => handleExposureChange(currentStackSize, teamData.team, 'minExposure', Number(e.target.value))}
                        inputProps={{ min: 0, max: 100, step: 1 }}
                        size="small"
                        sx={{ width: 80 }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <TextField
                        type="number"
                        value={teamData.maxExposure}
                        onChange={(e) => handleExposureChange(currentStackSize, teamData.team, 'maxExposure', Number(e.target.value))}
                        inputProps={{ min: 0, max: 100, step: 1 }}
                        size="small"
                        sx={{ width: 80 }}
                      />
                    </TableCell>
                    <TableCell align="center">
                      <Typography variant="body2" color={teamData.actualExposure > teamData.maxExposure ? 'error' : 'inherit'}>
                        {teamData.actualExposure}%
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {currentTeams.length === 0 && (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body1" color="textSecondary">
                No teams available. Please load player data first.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default TeamStacksTab;
