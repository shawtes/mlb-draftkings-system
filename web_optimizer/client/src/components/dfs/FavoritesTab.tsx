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
  Button,
  Chip,
  Grid,
  Alert,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  Divider,
  Stack,
  LinearProgress,
  FormControlLabel,
  Switch
} from '@mui/material';
import { 
  Star, 
  Delete,
  GetApp,
  Save,
  Add,
  Edit,
  Visibility,
  FileUpload,
  TableChart,
  Assessment,
  Group,
  Money,
  TrendingUp,
  Clear,
  Check,
  Warning,
  Info
} from '@mui/icons-material';

interface Lineup {
  id: string;
  name: string;
  players: any[];
  totalSalary: number;
  totalPoints: number;
  created: Date;
  notes?: string;
  tags?: string[];
  exposure?: number;
}

interface FavoritesTabProps {
  favorites?: Lineup[];
  onFavoritesUpdate?: (favorites: Lineup[]) => void;
  players?: any[];
  favoriteGroups?: any[];
  onPlayersUpdate?: (players: any[]) => void;
  onFavoriteGroupsUpdate?: () => void;
}

const FavoritesTab: React.FC<FavoritesTabProps> = ({ 
  favorites = [], 
  onFavoritesUpdate,
  players = [],
  favoriteGroups = [],
  onPlayersUpdate,
  onFavoriteGroupsUpdate
}) => {
  const [localFavorites, setLocalFavorites] = useState<Lineup[]>(favorites);
  const [selectedLineups, setSelectedLineups] = useState<string[]>([]);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [selectedLineup, setSelectedLineup] = useState<Lineup | null>(null);
  const [exportFormat, setExportFormat] = useState('draftkings');
  const [numToExport, setNumToExport] = useState(20);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('created');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showOnlySelected, setShowOnlySelected] = useState(false);
  
  // New lineup creation state
  const [newLineupName, setNewLineupName] = useState('');
  const [newLineupNotes, setNewLineupNotes] = useState('');

  useEffect(() => {
    setLocalFavorites(favorites);
  }, [favorites]);

  useEffect(() => {
    // Load saved favorites from localStorage
    const savedFavorites = localStorage.getItem('dfs_favorites');
    if (savedFavorites) {
      try {
        const parsed = JSON.parse(savedFavorites);
        setLocalFavorites(parsed);
      } catch (error) {
        console.error('Failed to load favorites:', error);
      }
    }
  }, []);

  const saveFavorites = (updatedFavorites: Lineup[]) => {
    setLocalFavorites(updatedFavorites);
    onFavoritesUpdate?.(updatedFavorites);
    localStorage.setItem('dfs_favorites', JSON.stringify(updatedFavorites));
  };

  const handleAddCurrentLineups = () => {
    // Simulate adding current optimization results to favorites
    const mockLineup: Lineup = {
      id: Date.now().toString(),
      name: `Lineup ${localFavorites.length + 1}`,
      players: players.slice(0, 10), // Take first 10 players as example
      totalSalary: 48500,
      totalPoints: 125.4,
      created: new Date(),
      notes: 'Generated from current optimization',
      tags: ['Generated', 'Current'],
      exposure: 15
    };
    
    const updated = [...localFavorites, mockLineup];
    saveFavorites(updated);
  };

  const handleDeleteLineup = (lineupId: string) => {
    const updated = localFavorites.filter(lineup => lineup.id !== lineupId);
    saveFavorites(updated);
    setSelectedLineups(selectedLineups.filter(id => id !== lineupId));
  };

  const handleCreateLineup = () => {
    if (!newLineupName.trim()) return;
    
    const newLineup: Lineup = {
      id: Date.now().toString(),
      name: newLineupName.trim(),
      players: [],
      totalSalary: 0,
      totalPoints: 0,
      created: new Date(),
      notes: newLineupNotes.trim(),
      tags: ['Manual'],
      exposure: 0
    };
    
    const updated = [...localFavorites, newLineup];
    saveFavorites(updated);
    
    setNewLineupName('');
    setNewLineupNotes('');
    setShowCreateDialog(false);
  };

  const handleSelectLineup = (lineupId: string, selected: boolean) => {
    if (selected) {
      setSelectedLineups([...selectedLineups, lineupId]);
    } else {
      setSelectedLineups(selectedLineups.filter(id => id !== lineupId));
    }
  };

  const handleSelectAll = () => {
    const filteredLineups = getFilteredAndSortedLineups();
    setSelectedLineups(filteredLineups.map(lineup => lineup.id));
  };

  const handleDeselectAll = () => {
    setSelectedLineups([]);
  };

  const handleExportSelected = () => {
    const selectedLineupsData = localFavorites.filter(lineup => selectedLineups.includes(lineup.id));
    console.log('Exporting lineups:', selectedLineupsData);
    
    // Simulate export to CSV
    const csvContent = generateCSV(selectedLineupsData);
    downloadCSV(csvContent, `favorites_export_${exportFormat}.csv`);
    
    setShowExportDialog(false);
  };

  const generateCSV = (lineups: Lineup[]) => {
    const headers = ['Name', 'Total Salary', 'Total Points', 'Created', 'Notes'];
    const rows = lineups.map(lineup => [
      lineup.name,
      lineup.totalSalary,
      lineup.totalPoints,
      lineup.created.toLocaleDateString(),
      lineup.notes || ''
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  const downloadCSV = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getFilteredAndSortedLineups = () => {
    let filtered = localFavorites;
    
    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(lineup => 
        lineup.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lineup.notes?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lineup.tags?.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }
    
    // Filter by selection if enabled
    if (showOnlySelected) {
      filtered = filtered.filter(lineup => selectedLineups.includes(lineup.id));
    }
    
    // Sort lineups
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'salary':
          aValue = a.totalSalary;
          bValue = b.totalSalary;
          break;
        case 'points':
          aValue = a.totalPoints;
          bValue = b.totalPoints;
          break;
        case 'created':
        default:
          aValue = a.created.getTime();
          bValue = b.created.getTime();
          break;
      }
      
      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
    
    return filtered;
  };

  const handleClearFavorites = () => {
    if (window.confirm('Are you sure you want to clear all favorites? This cannot be undone.')) {
      saveFavorites([]);
      setSelectedLineups([]);
    }
  };

  const handleFillDKEntries = () => {
    console.log('Filling DraftKings entries with selected lineups...');
    // This would integrate with the DraftKings entry filling functionality
    alert(`Would fill DraftKings entries with ${selectedLineups.length} selected lineups`);
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Star sx={{ mr: 2, color: '#FFD700', fontSize: 32 }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#FFD700' }}>
            My Entries (Favorites)
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Save, manage, and export your favorite lineups
          </Typography>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#FFF8E1' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#FFB300">
                {localFavorites.length}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Saved Lineups
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#E8F5E8' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#4CAF50">
                {selectedLineups.length}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Selected
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#E3F2FD' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#2196F3">
                {localFavorites.reduce((sum, lineup) => sum + lineup.totalSalary, 0).toLocaleString()}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Salary
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ bgcolor: '#FCE4EC' }}>
            <CardContent sx={{ p: 2 }}>
              <Typography variant="h6" color="#E91E63">
                {localFavorites.reduce((sum, lineup) => sum + lineup.totalPoints, 0).toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Points
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🛠️ Favorites Management
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Search lineups..."
                variant="outlined"
                size="small"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Sort by</InputLabel>
                <Select
                  value={sortBy}
                  label="Sort by"
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <MenuItem value="created">Date Created</MenuItem>
                  <MenuItem value="name">Name</MenuItem>
                  <MenuItem value="salary">Total Salary</MenuItem>
                  <MenuItem value="points">Total Points</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Order</InputLabel>
                <Select
                  value={sortOrder}
                  label="Order"
                  onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                >
                  <MenuItem value="desc">Descending</MenuItem>
                  <MenuItem value="asc">Ascending</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={showOnlySelected}
                    onChange={(e) => setShowOnlySelected(e.target.checked)}
                  />
                }
                label="Show only selected"
              />
            </Grid>
          </Grid>
          
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Button
              variant="contained"
              color="primary"
              startIcon={<Add />}
              onClick={handleAddCurrentLineups}
            >
              Add Current Results
            </Button>
            <Button
              variant="outlined"
              startIcon={<Add />}
              onClick={() => setShowCreateDialog(true)}
            >
              Create Lineup
            </Button>
            <Button
              variant="outlined"
              color="success"
              startIcon={<Check />}
              onClick={handleSelectAll}
              disabled={localFavorites.length === 0}
            >
              Select All
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<Clear />}
              onClick={handleDeselectAll}
              disabled={selectedLineups.length === 0}
            >
              Deselect All
            </Button>
            <Button
              variant="contained"
              color="success"
              startIcon={<GetApp />}
              onClick={() => setShowExportDialog(true)}
              disabled={selectedLineups.length === 0}
            >
              Export Selected ({selectedLineups.length})
            </Button>
            <Button
              variant="contained"
              color="info"
              startIcon={<FileUpload />}
              onClick={handleFillDKEntries}
              disabled={selectedLineups.length === 0}
            >
              Fill DK Entries
            </Button>
            <Button
              variant="outlined"
              color="error"
              startIcon={<Delete />}
              onClick={handleClearFavorites}
              disabled={localFavorites.length === 0}
            >
              Clear All
            </Button>
          </Stack>
        </CardContent>
      </Card>

      {/* Lineups Table */}
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Select</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Players</TableCell>
              <TableCell>Total Salary</TableCell>
              <TableCell>Total Points</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Tags</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {getFilteredAndSortedLineups().map((lineup) => (
              <TableRow key={lineup.id}>
                <TableCell>
                  <input
                    type="checkbox"
                    checked={selectedLineups.includes(lineup.id)}
                    onChange={(e) => handleSelectLineup(lineup.id, e.target.checked)}
                  />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                    {lineup.name}
                  </Typography>
                  {lineup.notes && (
                    <Typography variant="caption" color="textSecondary">
                      {lineup.notes}
                    </Typography>
                  )}
                </TableCell>
                <TableCell>{lineup.players.length}</TableCell>
                <TableCell>${lineup.totalSalary.toLocaleString()}</TableCell>
                <TableCell>{lineup.totalPoints.toFixed(1)}</TableCell>
                <TableCell>{lineup.created.toLocaleDateString()}</TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5}>
                    {lineup.tags?.map((tag, index) => (
                      <Chip key={index} label={tag} size="small" />
                    ))}
                  </Stack>
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5}>
                    <Tooltip title="View Details">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSelectedLineup(lineup);
                          setShowDetailsDialog(true);
                        }}
                      >
                        <Visibility />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteLineup(lineup.id)}
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Empty State */}
      {localFavorites.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="h6">No Favorite Lineups</Typography>
          <Typography>
            Add your current optimization results to favorites or create custom lineups to get started.
          </Typography>
        </Alert>
      )}

      {/* Create Lineup Dialog */}
      <Dialog open={showCreateDialog} onClose={() => setShowCreateDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Lineup</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Lineup Name"
            value={newLineupName}
            onChange={(e) => setNewLineupName(e.target.value)}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Notes (optional)"
            value={newLineupNotes}
            onChange={(e) => setNewLineupNotes(e.target.value)}
            margin="normal"
            multiline
            rows={3}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreateDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateLineup} variant="contained" disabled={!newLineupName.trim()}>
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Export Dialog */}
      <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export Selected Lineups</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Exporting {selectedLineups.length} lineup(s)
          </Typography>
          <FormControl fullWidth margin="normal">
            <InputLabel>Export Format</InputLabel>
            <Select
              value={exportFormat}
              label="Export Format"
              onChange={(e) => setExportFormat(e.target.value)}
            >
              <MenuItem value="draftkings">DraftKings CSV</MenuItem>
              <MenuItem value="fanduel">FanDuel CSV</MenuItem>
              <MenuItem value="custom">Custom CSV</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Number of lineups to export"
            type="number"
            value={numToExport}
            onChange={(e) => setNumToExport(parseInt(e.target.value) || 20)}
            margin="normal"
            inputProps={{ min: 1, max: selectedLineups.length }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
          <Button onClick={handleExportSelected} variant="contained">
            Export
          </Button>
        </DialogActions>
      </Dialog>

      {/* Details Dialog */}
      <Dialog open={showDetailsDialog} onClose={() => setShowDetailsDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>{selectedLineup?.name} - Details</DialogTitle>
        <DialogContent>
          {selectedLineup && (
            <Box>
              <Typography variant="h6">Lineup Information</Typography>
              <Typography>Total Salary: ${selectedLineup.totalSalary.toLocaleString()}</Typography>
              <Typography>Total Points: {selectedLineup.totalPoints.toFixed(1)}</Typography>
              <Typography>Created: {selectedLineup.created.toLocaleString()}</Typography>
              {selectedLineup.notes && (
                <Typography>Notes: {selectedLineup.notes}</Typography>
              )}
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="h6">Players ({selectedLineup.players.length})</Typography>
              {selectedLineup.players.length === 0 ? (
                <Alert severity="info">No players in this lineup</Alert>
              ) : (
                <TableContainer component={Paper} sx={{ mt: 1 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Position</TableCell>
                        <TableCell>Team</TableCell>
                        <TableCell>Salary</TableCell>
                        <TableCell>Points</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {selectedLineup.players.map((player, index) => (
                        <TableRow key={index}>
                          <TableCell>{player.name || `Player ${index + 1}`}</TableCell>
                          <TableCell>{player.position || 'N/A'}</TableCell>
                          <TableCell>{player.team || 'N/A'}</TableCell>
                          <TableCell>${(player.salary || 0).toLocaleString()}</TableCell>
                          <TableCell>{(player.predicted_dk_points || 0).toFixed(1)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetailsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default FavoritesTab;
