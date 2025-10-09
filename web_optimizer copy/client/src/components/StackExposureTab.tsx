import React from 'react';
import { Box, Typography, Card, CardContent, Alert } from '@mui/material';
import { Timeline } from '@mui/icons-material';

interface StackExposureTabProps {
  exposureData?: any[];
  onExposureUpdate?: (data: any[]) => void;
  teamStacks?: any[];
  stackExposures?: any[];
  onStackExposuresUpdate?: () => void;
  totalLineups?: number;
}

const StackExposureTab: React.FC<StackExposureTabProps> = ({ 
  exposureData = [], 
  onExposureUpdate,
  teamStacks = [],
  stackExposures = [],
  onStackExposuresUpdate,
  totalLineups = 0
}) => {
  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Timeline sx={{ mr: 2, color: '#FF9800', fontSize: 32 }} />
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700, color: '#FF9800' }}>
            Stack Exposure Management
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Monitor and control stack exposure across lineups
          </Typography>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            Stack exposure functionality is being implemented. Configure team stacks to see exposure data.
          </Alert>
          
          <Typography variant="h6" gutterBottom>
            Exposure Summary
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Total Exposure Data Points: {exposureData.length}
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default StackExposureTab;
