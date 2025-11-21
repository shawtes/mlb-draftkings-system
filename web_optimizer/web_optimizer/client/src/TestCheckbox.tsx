import React, { useState } from 'react';
import { Box, Container, Typography, Paper, Table, TableHead, TableRow, TableCell, TableBody, Checkbox, Button } from '@mui/material';
import CheckBoxOutlineBlankIcon from '@mui/icons-material/CheckBoxOutlineBlank';
import CheckBoxIcon from '@mui/icons-material/CheckBox';
import IndeterminateCheckBoxIcon from '@mui/icons-material/IndeterminateCheckBox';

export default function TestCheckbox() {
  const [rows, setRows] = useState(
    Array.from({ length: 8 }).map((_, i) => ({ id: i + 1, name: `Row ${i + 1}`, checked: i % 3 === 0 }))
  );
  const all = rows.every(r => r.checked);
  const some = rows.some(r => r.checked);

  const setAll = (checked: boolean) => {
    setRows(rows.map(r => ({ ...r, checked })));
  };

  const toggle = (id: number, checked: boolean) => {
    setRows(rows.map(r => (r.id === id ? { ...r, checked } : r)));
  };

  return (
    <Container maxWidth="sm" sx={{ py: 6 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Checkbox Visibility Test
      </Typography>
      <Paper sx={{ p: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  size="small"
                  icon={<CheckBoxOutlineBlankIcon sx={{ color: '#90caf9' }} fontSize="small" />}
                  checkedIcon={<CheckBoxIcon sx={{ color: '#90caf9' }} fontSize="small" />}
                  indeterminateIcon={<IndeterminateCheckBoxIcon sx={{ color: '#90caf9' }} fontSize="small" />}
                  indeterminate={some && !all}
                  checked={all && rows.length > 0}
                  onChange={(e) => setAll(e.target.checked)}
                />
              </TableCell>
              <TableCell>Item</TableCell>
              <TableCell>State</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map(r => (
              <TableRow key={r.id}>
                <TableCell padding="checkbox">
                  <Checkbox
                    size="small"
                    icon={<CheckBoxOutlineBlankIcon sx={{ color: '#90caf9' }} fontSize="small" />}
                    checkedIcon={<CheckBoxIcon sx={{ color: '#90caf9' }} fontSize="small" />}
                    checked={r.checked}
                    onChange={(e) => toggle(r.id, e.target.checked)}
                  />
                </TableCell>
                <TableCell>{r.name}</TableCell>
                <TableCell>{r.checked ? 'checked' : 'unchecked'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
          <Button variant="outlined" onClick={() => setAll(true)}>Select All</Button>
          <Button variant="outlined" color="error" onClick={() => setAll(false)}>Deselect All</Button>
        </Box>
      </Paper>
      <Typography variant="body2" sx={{ mt: 2, color: 'text.secondary' }}>
        Opened via /test route. If you can’t see empty boxes, it’s a rendering/theme issue.
      </Typography>
    </Container>
  );
}




