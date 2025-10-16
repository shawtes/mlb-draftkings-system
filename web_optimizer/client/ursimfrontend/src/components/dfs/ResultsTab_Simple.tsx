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
  Button,
  Chip,
  Grid,
  Card,
  CardContent,
  Stack,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Download,
  Star,
  Analytics,
  AttachMoney,
  People,
  Visibility,
  Delete
} from '@mui/icons-material';

interface Player {
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
  projectedPoints: number;
  value: number;
}

interface ResultsTabProps {
  results?: Lineup[];
  onExportResults?: () => void;
}

const ResultsTab: React.FC<ResultsTabProps> = ({ results = [], onExportResults }) => {
  const [selectedLineup, setSelectedLineup] = useState<number | null>(null);

  // Mock data if no results provided
  const mockLineups: Lineup[] = results.length > 0 ? results : [
    {
      id: 1,
      totalSalary: 49800,
      projectedPoints: 148.5,
      value: 2.98,
      players: [
        { name: 'Patrick Mahomes', position: 'QB', team: 'KC', salary: 8500, projection: 24.5 },
        { name: 'Christian McCaffrey', position: 'RB', team: 'SF', salary: 9200, projection: 22.8 },
        { name: 'Derrick Henry', position: 'RB', team: 'TEN', salary: 8000, projection: 20.2 },
        { name: 'Tyreek Hill', position: 'WR', team: 'MIA', salary: 8300, projection: 18.9 },
        { name: 'Justin Jefferson', position: 'WR', team: 'MIN', salary: 8800, projection: 21.1 },
        { name: 'CeeDee Lamb', position: 'WR', team: 'DAL', salary: 7500, projection: 17.3 },
        { name: 'Travis Kelce', position: 'TE', team: 'KC', salary: 7000, projection: 16.2 },
        { name: 'Ravens', position: 'DST', team: 'BAL', salary: 3500, projection: 8.5 },
      ],
    },
    {
      id: 2,
      totalSalary: 49500,
      projectedPoints: 145.2,
      value: 2.93,
      players: [
        { name: 'Josh Allen', position: 'QB', team: 'BUF', salary: 8200, projection: 23.1 },
        { name: 'Saquon Barkley', position: 'RB', team: 'NYG', salary: 8500, projection: 21.5 },
        { name: 'Joe Mixon', position: 'RB', team: 'CIN', salary: 7200, projection: 18.7 },
        { name: 'Stefon Diggs', position: 'WR', team: 'BUF', salary: 7800, projection: 19.2 },
        { name: 'Davante Adams', position: 'WR', team: 'LV', salary: 8200, projection: 20.5 },
        { name: 'Amon-Ra St. Brown', position: 'WR', team: 'DET', salary: 7000, projection: 16.8 },
        { name: 'Mark Andrews', position: 'TE', team: 'BAL', salary: 6600, projection: 15.4 },
        { name: '49ers', position: 'DST', team: 'SF', salary: 4000, projection: 10.0 },
      ],
    },
  ];

  const stats = {
    totalLineups: mockLineups.length,
    avgProjection: mockLineups.reduce((sum, l) => sum + l.projectedPoints, 0) / mockLineups.length,
    avgSalary: mockLineups.reduce((sum, l) => sum + l.totalSalary, 0) / mockLineups.length,
    avgValue: mockLineups.reduce((sum, l) => sum + l.value, 0) / mockLineups.length,
  };

  const handleExport = () => {
    if (onExportResults) {
      onExportResults();
    } else {
      alert('Exporting lineups to CSV...');
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700, color: '#2196F3', mb: 1 }}>
          Optimization Results
        </Typography>
        <Typography variant="body2" color="textSecondary">
          View and export optimized lineups
        </Typography>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'rgba(33, 150, 243, 0.1)' }}>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Analytics sx={{ color: '#2196F3' }} />
                <Box>
                  <Typography variant="h6" color="primary">{stats.totalLineups}</Typography>
                  <Typography variant="caption">Total Lineups</Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'rgba(76, 175, 80, 0.1)' }}>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1}>
                <Star sx={{ color: '#4CAF50' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#4CAF50' }}>
                    {stats.avgProjection.toFixed(1)}
                  </Typography>
                  <Typography variant="caption">Avg Projection</Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'rgba(255, 152, 0, 0.1)' }}>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1}>
                <AttachMoney sx={{ color: '#FF9800' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#FF9800' }}>
                    ${stats.avgSalary.toFixed(0)}
                  </Typography>
                  <Typography variant="caption">Avg Salary</Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card sx={{ background: 'rgba(156, 39, 176, 0.1)' }}>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1}>
                <People sx={{ color: '#9C27B0' }} />
                <Box>
                  <Typography variant="h6" sx={{ color: '#9C27B0' }}>
                    {stats.avgValue.toFixed(2)}x
                  </Typography>
                  <Typography variant="caption">Avg Value</Typography>
                </Box>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Action Buttons */}
      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<Download />}
          onClick={handleExport}
          sx={{
            background: 'linear-gradient(45deg, #667eea, #764ba2)',
            '&:hover': {
              background: 'linear-gradient(45deg, #5a6fd8, #6a4190)',
            },
          }}
        >
          Export to DraftKings
        </Button>
        <Button variant="outlined" startIcon={<Download />}>
          Export to CSV
        </Button>
      </Stack>

      {/* Lineups Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Rank</TableCell>
              <TableCell>Projected Points</TableCell>
              <TableCell>Total Salary</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Players</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {mockLineups.map((lineup, index) => (
              <TableRow
                key={lineup.id}
                hover
                selected={selectedLineup === lineup.id}
                onClick={() => setSelectedLineup(lineup.id)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>
                  <Chip label={`#${index + 1}`} color="primary" size="small" />
                </TableCell>
                <TableCell>
                  <Typography variant="body2" fontWeight={600}>
                    {lineup.projectedPoints.toFixed(1)}
                  </Typography>
                </TableCell>
                <TableCell>${lineup.totalSalary.toLocaleString()}</TableCell>
                <TableCell>
                  <Chip
                    label={`${lineup.value.toFixed(2)}x`}
                    size="small"
                    color={lineup.value > 3 ? 'success' : 'default'}
                  />
                </TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap">
                    {lineup.players.slice(0, 3).map((player, idx) => (
                      <Chip
                        key={idx}
                        label={player.name}
                        size="small"
                        variant="outlined"
                      />
                    ))}
                    {lineup.players.length > 3 && (
                      <Chip label={`+${lineup.players.length - 3}`} size="small" />
                    )}
                  </Stack>
                </TableCell>
                <TableCell align="center">
                  <Tooltip title="View Details">
                    <IconButton size="small" color="primary">
                      <Visibility />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete">
                    <IconButton size="small" color="error">
                      <Delete />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {mockLineups.length === 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          No lineups generated yet. Configure your settings and run optimization.
        </Alert>
      )}
    </Box>
  );
};

export default ResultsTab;


