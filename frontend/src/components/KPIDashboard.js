import React from 'react';
import { Typography, Box, Container } from '@mui/material';

function KPIDashboard() {
  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          KPI Dashboard
        </Typography>
        <Typography variant="body1" paragraph>
          This section will display key performance indicators related to the ECE's operation.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          (Coming Soon: Metrics such as auto-approval rates, HIL intervention rates, average distortion, etc.)
        </Typography>
      </Box>
    </Container>
  );
}

export default KPIDashboard;