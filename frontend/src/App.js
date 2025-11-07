import React, { useState, useCallback } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, IconButton, Container } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import DescriptionIcon from '@mui/icons-material/Description';
import HistoryIcon from '@mui/icons-material/History';
import PeopleIcon from '@mui/icons-material/People';
import DashboardIcon from '@mui/icons-material/Dashboard';

import Home from './components/Home';
import TransactionEvaluation from './components/TransactionEvaluation';
import TextEvaluation from './components/TextEvaluation';
import DecisionHistory from './components/DecisionHistory';
import HILRequests from './components/HILRequests';
import KPIDashboard from './components/KPIDashboard';

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  const logError = useCallback(async (error) => {
    console.error('Logging error to backend:', error);
    try {
      await fetch('http://localhost:8000/log-error', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: error.message, stack: error.stack, name: error.name }),
      });
    } catch (e) {
      console.error('Failed to log error to backend:', e);
    }
  }, []);

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  const navItems = [
    { text: 'Home', icon: <HomeIcon />, path: '/' },
    { text: 'Transaction Evaluation', icon: <AccountBalanceIcon />, path: '/transaction-evaluation' },
    { text: 'Text Evaluation', icon: <DescriptionIcon />, path: '/text-evaluation' },
    { text: 'Decision History', icon: <HistoryIcon />, path: '/decision-history' },
    { text: 'HIL Requests', icon: <PeopleIcon />, path: '/hil-requests' },
    { text: 'KPI Dashboard', icon: <DashboardIcon />, path: '/kpi-dashboard' },
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={toggleDrawer(true)}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            Eidetic Continuum Governance
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
        PaperProps={{
          sx: { width: 240, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {navItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton component={Link} to={item.path} onClick={toggleDrawer(false)}>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box
        component="main"
        sx={{ flexGrow: 1, p: 3, width: { sm: `calc(100% - 240px)` } }}
      >
        <Toolbar /> {/* This is to offset content below the AppBar */}
        <Container maxWidth="lg">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/transaction-evaluation" element={<TransactionEvaluation logError={logError} />} />
            <Route path="/text-evaluation" element={<TextEvaluation logError={logError} />} />
            <Route path="/decision-history" element={<DecisionHistory logError={logError} />} />
            <Route path="/hil-requests" element={<HILRequests />} />
            <Route path="/kpi-dashboard" element={<KPIDashboard />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}

export default App;