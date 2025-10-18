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
  Paper,
  Checkbox,
  TextField,
  Button,
  Tabs,
  Tab,
  Chip,
  Grid,
  FormControlLabel,
  Switch,
  Alert,
  LinearProgress,
  Divider
} from '@mui/material';
import { 
  People, 
  CheckCircle, 
  Cancel,
  SelectAll,
  ClearAll
} from '@mui/icons-material';

interface Player {
  id: string;
  name: string;
  team: string;
  position: string;
  salary: number;
  predicted_dk_points: number;
  value?: number;
  selected?: boolean;
  min_exposure?: number;
  max_exposure?: number;
  actual_exposure?: number;
}

interface PlayersTabProps {
  players?: Player[];
  onPlayersUpdate?: (players: Player[]) => void;
}

const PlayersTab: React.FC<PlayersTabProps> = ({ players = [], onPlayersUpdate }) => {
  const [selectedTab, setSelectedTab] = useState(0);
  const [localPlayers, setLocalPlayers] = useState<Player[]>(players);
  const [searchTerm, setSearchTerm] = useState('');
  const [showOnlySelected, setShowOnlySelected] = useState(false);

  // Position tabs matching optimizer01.py
  const positionTabs = [
    'All Batters',
    'C',
    '1B', 
    '2B',
    '3B',
    'SS',
    'OF',
    'P'
  ];

  useEffect(() => {
    setLocalPlayers(players);
  }, [players]);

  const handlePlayerUpdate = (updatedPlayers: Player[]) => {
    setLocalPlayers(updatedPlayers);
    onPlayersUpdate?.(updatedPlayers);
  };

  const handlePlayerSelection = (playerId: string, selected: boolean) => {
    const updated = localPlayers.map(p => 
      p.id === playerId ? { ...p, selected } : p
    );
    handlePlayerUpdate(updated);
  };

  const handleExposureChange = (playerId: string, field: 'min_exposure' | 'max_exposure', value: number) => {
    const updated = localPlayers.map(p => 
      p.id === playerId ? { ...p, [field]: value } : p
    );
    handlePlayerUpdate(updated);
  };

  const selectAllForPosition = (position: string) => {
    const updated = localPlayers.map(p => {
      if (position === 'All Batters') {
        return p.position !== 'P' ? { ...p, selected: true } : p;
      }
      return p.position.includes(position) ? { ...p, selected: true } : p;
    });
    handlePlayerUpdate(updated);
  };

  const deselectAllForPosition = (position: string) => {
    const updated = localPlayers.map(p => {
      if (position === 'All Batters') {
        return p.position !== 'P' ? { ...p, selected: false } : p;
      }
      return p.position.includes(position) ? { ...p, selected: false } : p;
    });
    handlePlayerUpdate(updated);
  };

  const getFilteredPlayers = (position: string) => {
    let filtered = localPlayers;
    
    // Filter by position
    if (position === 'All Batters') {
      filtered = filtered.filter(p => p.position !== 'P');
    } else if (position !== 'All') {
      filtered = filtered.filter(p => p.position.includes(position));
    }
    
    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(p => 
        p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        p.team.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Filter by selection if enabled
    if (showOnlySelected) {
      filtered = filtered.filter(p => p.selected);
    }
    
    return filtered;
  };

  const getSelectedCount = () => {
    return localPlayers.filter(p => p.selected).length;
  };

  const getTotalSalary = () => {
    return localPlayers.filter(p => p.selected).reduce((sum, p) => sum + p.salary, 0);
  };

  const getTotalPoints = () => {
    return localPlayers.filter(p => p.selected).reduce((sum, p) => sum + p.predicted_dk_points, 0);
  };

  const renderPlayerTable = (position: string) => {
    const filteredPlayers = getFilteredPlayers(position);
    
    return (
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Select</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Team</TableCell>
              <TableCell>Position</TableCell>
              <TableCell>Salary</TableCell>
              <TableCell>Predicted Points</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Min Exp (%)</TableCell>
              <TableCell>Max Exp (%)</TableCell>
              <TableCell>Actual Exp (%)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredPlayers.map((player) => (
              <TableRow key={player.id}>
                <TableCell>
                  <Checkbox
                    checked={player.selected || false}
                    onChange={(e) => handlePlayerSelection(player.id, e.target.checked)}
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: player.selected ? 'bold' : 'normal' }}>
                    {player.name}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip label={player.team} size="small" color="primary" />
                </TableCell>
                <TableCell>{player.position}</TableCell>
                <TableCell>${player.salary.toLocaleString()}</TableCell>
                <TableCell>{player.predicted_dk_points.toFixed(1)}</TableCell>
                <TableCell>{player.value?.toFixed(1) || 'N/A'}</TableCell>
                <TableCell>
                  <TextField
                    type="number"
                    size="small"
                    value={player.min_exposure || 0}
                    onChange={(e) => handleExposureChange(player.id, 'min_exposure', parseInt(e.target.value) || 0)}
                    inputProps={{ min: 0, max: 100, style: { width: '60px' } }}
                  />
                </TableCell>
                <TableCell>
                  <TextField
                    type="number"
                    size="small"
                    value={player.max_exposure || 100}
                    onChange={(e) => handleExposureChange(player.id, 'max_exposure', parseInt(e.target.value) || 100)}
                    inputProps={{ min: 0, max: 100, style: { width: '60px' } }}
                  />
                </TableCell>
                <TableCell>{player.actual_exposure?.toFixed(1) || '0.0'}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <People sx={{ mr: 2, color: '#4CAF50', fontSize: 32 }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#4CAF50' }}>
            Players Management
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Select players, set exposure limits, and manage your roster
          </Typography>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#E8F5E8' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#4CAF50">
                {getSelectedCount()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Selected Players
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#E3F2FD' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#2196F3">
                ${getTotalSalary().toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Salary
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#FFF3E0' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#FF9800">
                {getTotalPoints().toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Points
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#FCE4EC' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#E91E63">
                {localPlayers.length}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Available Players
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Filters */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, alignItems: 'center' }}>
        <TextField
          label="Search players..."
          variant="outlined"
          size="small"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ minWidth: 200 }}
        />
        <FormControlLabel
          control={
            <Switch
              checked={showOnlySelected}
              onChange={(e) => setShowOnlySelected(e.target.checked)}
            />
          }
          label="Show only selected"
        />
      </Box>

      {/* Position Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={selectedTab} onChange={(_, newValue) => setSelectedTab(newValue)}>
          {positionTabs.map((position, index) => (
            <Tab key={position} label={position} />
          ))}
        </Tabs>
      </Box>

      {/* Selection Controls */}
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <Button
          variant="outlined"
          color="success"
          startIcon={<SelectAll />}
          onClick={() => selectAllForPosition(positionTabs[selectedTab])}
        >
          Select All
        </Button>
        <Button
          variant="outlined"
          color="error"
          startIcon={<ClearAll />}
          onClick={() => deselectAllForPosition(positionTabs[selectedTab])}
        >
          Deselect All
        </Button>
      </Box>

      {/* Player Table */}
      {renderPlayerTable(positionTabs[selectedTab])}

      {/* No Players Message */}
      {localPlayers.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="h6">No Player Data</Typography>
          <Typography>
            Upload a CSV file using the Control Panel to load player data and start building lineups.
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default PlayersTab;
