import React, { useState, useEffect, useCallback } from 'react';
import { Typography, Box, Container, Card, CardContent, Button, CircularProgress, TextField, Snackbar, Alert, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';

function HILRequests() {
  const [hilRequests, setHilRequests] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');

  const [openDialog, setOpenDialog] = useState(false);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  const [currentDecisionType, setCurrentDecisionType] = useState(''); // 'approved' or 'rejected'
  const [humanReasoning, setHumanReasoning] = useState('');

  const fetchHilRequests = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/get-hil-requests');
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setHilRequests(data);
    } catch (e) {
      console.error('Failed to fetch HIL requests:', e);
      setError(e);
      setSnackbarMessage(`Error fetching HIL requests: ${e.message}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHilRequests();
  }, [fetchHilRequests]);

  const handleOpenDialog = (requestId, decisionType) => {
    setCurrentRequestId(requestId);
    setCurrentDecisionType(decisionType);
    setHumanReasoning(''); // Clear previous reasoning
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setCurrentRequestId(null);
    setCurrentDecisionType('');
    setHumanReasoning('');
  };

  const handleSubmitHilDecision = useCallback(async () => {
    if (!currentRequestId || !currentDecisionType) return;

    try {
      const res = await fetch('http://localhost:8000/submit-hil-decision', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          request_id: currentRequestId,
          decision: currentDecisionType,
          reasoning: humanReasoning,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(`HTTP error! status: ${res.status} - ${errorData.detail || res.statusText}`);
      }

      setSnackbarMessage(`HIL request ${currentRequestId} ${currentDecisionType}d successfully.`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      fetchHilRequests(); // Refresh the list
      handleCloseDialog();
    } catch (e) {
      console.error('Failed to submit HIL decision:', e);
      setSnackbarMessage(`Error submitting HIL decision: ${e.message}`);
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  }, [currentRequestId, currentDecisionType, humanReasoning, fetchHilRequests]);

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  const renderDecisionDetails = (decision) => {
    if (!decision) return 'N/A';
    return (
      <Box>
        {decision.decision && <Typography variant="body2">Decision: <Typography component="span" fontWeight="bold">{decision.decision}</Typography></Typography>}
        {decision.reasoning && <Typography variant="body2">Reasoning: {decision.reasoning}</Typography>}
        {/* Add more fields as needed */}
      </Box>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Human-in-the-Loop Requests
      </Typography>

      <Box sx={{ my: 4 }}>
        <Typography variant="h5" gutterBottom>Pending HIL Requests</Typography>
        <Button
          variant="outlined"
          onClick={fetchHilRequests}
          disabled={isLoading}
          sx={{ mb: 2 }}
        >
          {isLoading ? <CircularProgress size={24} /> : 'Refresh HIL Requests'}
        </Button>

        {isLoading && <CircularProgress />}
        {error && <Alert severity="error">Error: {error.message}</Alert>}

        {!isLoading && !error && hilRequests.length === 0 && (
          <Typography>No pending HIL requests.</Typography>
        )}

        {!isLoading && !error && hilRequests.length > 0 && (
          <Box>
            {hilRequests.map((request) => (
              <Card key={request.id} variant="outlined" sx={{ mb: 2, p: 2 }}>
                <Typography variant="h6">Request ID: {request.id}</Typography>
                <Typography variant="body2">Timestamp: {new Date(request.timestamp).toLocaleString()}</Typography>
                
                <Typography variant="body1" sx={{ mt: 1 }}>Original AI Decision:</Typography>
                {renderDecisionDetails(request.original_decision)}

                <Typography variant="body1" sx={{ mt: 1 }}>Proposed Human Delta:</Typography>
                {renderDecisionDetails(request.proposed_delta)}

                <Typography variant="body1" sx={{ mt: 1 }}>Evidence/Reasoning: {request.evidence_snippets}</Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>Status: <Typography component="span" fontWeight="bold">{request.status}</Typography></Typography>

                <Box sx={{ mt: 2 }}>
                  <Button
                    variant="contained"
                    color="success"
                    sx={{ mr: 1 }}
                    onClick={() => handleOpenDialog(request.id, 'approved')}
                    disabled={request.status !== 'pending'}
                  >
                    Approve
                  </Button>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={() => handleOpenDialog(request.id, 'rejected')}
                    disabled={request.status !== 'pending'}
                  >
                    Reject
                  </Button>
                </Box>
              </Card>
            ))}
          </Box>
        )}
      </Box>

      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Provide Reasoning for {currentDecisionType === 'approved' ? 'Approval' : 'Rejection'}</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Please provide your reasoning for {currentDecisionType === 'approved' ? 'approving' : 'rejecting'} this HIL request.
          </DialogContentText>
          <TextField
            autoFocus
            margin="dense"
            label="Reasoning"
            type="text"
            fullWidth
            multiline
            rows={4}
            variant="outlined"
            value={humanReasoning}
            onChange={(e) => setHumanReasoning(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSubmitHilDecision} color="primary">
            Submit {currentDecisionType === 'approved' ? 'Approval' : 'Rejection'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default HILRequests;