import React, { useState, useCallback } from 'react';
import { Box, TextField, Button, Card, CardContent, Typography, Grid, Select, MenuItem, InputLabel, FormControl, CircularProgress } from '@mui/material';

function TransactionEvaluation({ logError }) {
  const [transactionAmount, setTransactionAmount] = useState('');
  const [destinationCountryRisk, setDestinationCountryRisk] = useState('');
  const [purpose, setPurpose] = useState('');
  const [securityRiskLevel, setSecurityRiskLevel] = useState('');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const useCases = [
    { name: 'Select a use case', data: { transactionAmount: '', destinationCountryRisk: '', purpose: '', securityRiskLevel: '' } },
    { name: 'Humanitarian Exception', data: { transactionAmount: '120000', destinationCountryRisk: 'sanctioned', purpose: 'medical_aid', securityRiskLevel: 'low' } },
    { name: 'Standard High-Risk', data: { transactionAmount: '150000', destinationCountryRisk: 'high_risk', purpose: 'investment', securityRiskLevel: 'medium' } },
    { name: 'Clear Violation', data: { transactionAmount: '5000', destinationCountryRisk: 'low_risk', purpose: 'sanctioned_activity', securityRiskLevel: 'high' } },
    { name: 'Critical Security Event', data: { transactionAmount: '1000', destinationCountryRisk: 'low_risk', purpose: 'internal_transfer', securityRiskLevel: 'critical' } },
  ];

  const handleUseCaseChange = (event) => {
    const selectedUseCase = useCases.find(uc => uc.name === event.target.value);
    if (selectedUseCase) {
      setTransactionAmount(selectedUseCase.data.transactionAmount);
      setDestinationCountryRisk(selectedUseCase.data.destinationCountryRisk);
      setPurpose(selectedUseCase.data.purpose);
      setSecurityRiskLevel(selectedUseCase.data.securityRiskLevel);
    }
  };

  const handleSubmitTransaction = useCallback(async (event) => {
    event.preventDefault();
    setError(null);
    setResponse(null);
    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:8000/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transaction_amount: parseInt(transactionAmount),
          destination_country_risk: destinationCountryRisk,
          purpose: purpose,
          security_risk_level: securityRiskLevel,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setResponse(data);
      // Assuming a refresh of decision history would be triggered from a parent component or context
    } catch (e) {
      console.error(e);
      setError(e);
      logError(e);
    } finally {
      setIsLoading(false);
    }
  }, [transactionAmount, destinationCountryRisk, purpose, securityRiskLevel, logError]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Transaction Evaluation
        </Typography>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel id="use-case-select-label">Select Use Case</InputLabel>
          <Select
            labelId="use-case-select-label"
            value={useCases.find(uc =>
              uc.data.transactionAmount === transactionAmount &&
              uc.data.destinationCountryRisk === destinationCountryRisk &&
              uc.data.purpose === purpose &&
              uc.data.securityRiskLevel === securityRiskLevel
            )?.name || ''}
            label="Select Use Case"
            onChange={handleUseCaseChange}
          >
            {useCases.map(uc => (
              <MenuItem key={uc.name} value={uc.name}>
                {uc.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Box component="form" onSubmit={handleSubmitTransaction} sx={{ mt: 2 }}>
          <TextField
            label="Transaction Amount"
            type="number"
            fullWidth
            margin="normal"
            value={transactionAmount}
            onChange={(e) => setTransactionAmount(e.target.value)}
            required
          />
          <TextField
            label="Destination Country Risk"
            fullWidth
            margin="normal"
            value={destinationCountryRisk}
            onChange={(e) => setDestinationCountryRisk(e.target.value)}
            required
          />
          <TextField
            label="Purpose"
            fullWidth
            margin="normal"
            value={purpose}
            onChange={(e) => setPurpose(e.target.value)}
            required
          />
          <TextField
            label="Security Risk Level"
            fullWidth
            margin="normal"
            value={securityRiskLevel}
            onChange={(e) => setSecurityRiskLevel(e.target.value)}
            required
          />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            sx={{ mt: 2 }}
            disabled={isLoading}
          >
            {isLoading ? <CircularProgress size={24} /> : 'Evaluate Transaction'}
          </Button>
        </Box>

        {isLoading && <Typography sx={{ mt: 2 }}>Loading...</Typography>}

        {response && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Transaction Response:</Typography>
            <Card variant="outlined" sx={{ mt: 1, p: 2 }} data-testid="transaction-response-card">
              <Typography variant="subtitle1" component="div" gutterBottom>
                Decision: <Typography component="span" color={response.decision === 'Approval' || response.decision === 'APPROVE with enhanced documentation' ? 'success.main' : 'error.main'} fontWeight="bold">{response.decision}</Typography>
              </Typography>
              <Typography variant="body2" gutterBottom>
                Reasoning: {response.reasoning}
              </Typography>
              {response.principles_evaluated && response.principles_evaluated.length > 0 && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="subtitle2">Principles Evaluated:</Typography>
                  <ul>
                    {response.principles_evaluated.map((principle, index) => (
                      <li key={index}>
                        <Typography variant="body2">{principle}</Typography>
                      </li>
                    ))}
                  </ul>
                </Box>
              )}
               {response.precedent_cited && (
                <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                  Precedent Cited: {response.precedent_cited}
                </Typography>
              )}
              {response.violation_detected !== undefined && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Constitutional Violation Detected: <Typography component="span" fontWeight="bold">{response.violation_detected ? 'Yes' : 'No'}</Typography>
                </Typography>
              )}
               {response.require_interpretation !== undefined && (
                <Typography variant="body2">
                  Requires AI Interpretation: <Typography component="span" fontWeight="bold">{response.require_interpretation ? 'Yes' : 'No'}</Typography>
                </Typography>
              )}
            </Card>
          </Box>
        )}

        {error && (
          <Box sx={{ mt: 2, color: 'error.main' }}>
            <Typography variant="h6">Transaction Error:</Typography>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{JSON.stringify({ message: error.message, stack: error.stack, name: error.name }, null, 2)}</pre>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default TransactionEvaluation;