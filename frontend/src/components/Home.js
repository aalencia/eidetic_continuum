import React from 'react';
import { Typography, Box, Container } from '@mui/material';

function Home() {
  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome to Eidetic Continuum Governance
        </Typography>
        <Typography variant="body1" paragraph>
          This application demonstrates the capabilities of the Eidetic Continuum Engine (ECE) in managing and refining AI-driven decisions.
        </Typography>
        <Typography variant="body1" paragraph>
          Use the navigation to explore different aspects of the system, including transaction evaluations, text analysis, decision history, and future human-in-the-loop requests and KPI dashboards.
        </Typography>
      </Box>
    </Container>
  );
}

export default Home;