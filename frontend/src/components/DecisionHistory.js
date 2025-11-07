import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, Card, CardContent, Button, CircularProgress } from '@mui/material';

function DecisionHistory({ logError }) {
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  const fetchDecisionHistory = useCallback(async () => {
    setIsHistoryLoading(true);
    setHistoryError(null);
    try {
      const res = await fetch('http://localhost:8000/get-decision-history');
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setDecisionHistory(data);
    } catch (e) {
      console.error('Failed to fetch decision history:', e);
      setHistoryError(e);
      logError(e);
    } finally {
      setIsHistoryLoading(false);
    }
  }, [logError]);

  useEffect(() => {
    fetchDecisionHistory();
  }, [fetchDecisionHistory]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Decision History
        </Typography>
        <Button
          variant="outlined"
          onClick={fetchDecisionHistory}
          disabled={isHistoryLoading}
          sx={{ mb: 2 }}
        >
          {isHistoryLoading ? <CircularProgress size={24} /> : 'Refresh Decisions'}
        </Button>

        {isHistoryLoading && <Typography>Loading history...</Typography>}
        {historyError && (
          <Box sx={{ color: 'error.main' }}>
            <Typography>Error loading history: {historyError.message}</Typography>
          </Box>
        )}

        {!isHistoryLoading && !historyError && decisionHistory.length === 0 && (
          <Typography>No decisions made yet.</Typography>
        )}

        {!isHistoryLoading && !historyError && decisionHistory.length > 0 && (
          <Box>
            {decisionHistory.map((decision, index) => (
              <Card key={index} variant="outlined" sx={{ mb: 2, p: 2 }}>
                <Typography variant="subtitle1">Timestamp: {new Date(decision.timestamp).toLocaleString()}</Typography>
                <Typography variant="body2">Input: {JSON.stringify(decision.transaction_input, null, 2)}</Typography>
                <Typography variant="body2">Decision: {decision.decision_output.decision}</Typography>
                <Typography variant="body2">Reasoning: {decision.decision_output.reasoning}</Typography>
              </Card>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default DecisionHistory;