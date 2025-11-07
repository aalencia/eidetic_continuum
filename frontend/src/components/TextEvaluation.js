import React, { useState, useCallback } from 'react';
import { Box, TextField, Button, Card, CardContent, Typography, CircularProgress } from '@mui/material';

function TextEvaluation({ logError }) {
  const [textDescription, setTextDescription] = useState('');
  const [textEvaluationResponse, setTextEvaluationResponse] = useState(null);
  const [isTextEvaluating, setIsTextEvaluating] = useState(false);
  const [textEvaluationError, setTextEvaluationError] = useState(null);

  const handleEvaluateText = useCallback(async () => {
    setTextEvaluationError(null);
    setTextEvaluationResponse(null);
    setIsTextEvaluating(true);

    try {
      const res = await fetch('http://localhost:8000/evaluate-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ description: textDescription }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      setTextEvaluationResponse(data);
    } catch (e) {
      console.error(e);
      setTextEvaluationError(e);
      logError(e);
    } finally {
      setIsTextEvaluating(false);
    }
  }, [textDescription, logError]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Text Description Evaluation
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Enter a text description to be evaluated by the system.
        </Typography>
        <TextField
          label="Text Description"
          multiline
          rows={6}
          fullWidth
          margin="normal"
          value={textDescription}
          onChange={(e) => setTextDescription(e.target.value)}
        />
        <Button
          variant="contained"
          color="primary"
          sx={{ mt: 2 }}
          onClick={handleEvaluateText}
          disabled={isTextEvaluating}
        >
          {isTextEvaluating ? <CircularProgress size={24} /> : 'Evaluate Text'}
        </Button>

        {isTextEvaluating && <Typography sx={{ mt: 2 }}>Loading...</Typography>}

        {textEvaluationResponse && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="h6">Text Evaluation Response:</Typography>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{JSON.stringify(textEvaluationResponse, null, 2)}</pre>
          </Box>
        )}

        {textEvaluationError && (
          <Box sx={{ mt: 2, color: 'error.main' }}>
            <Typography variant="h6">Text Evaluation Error:</Typography>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{JSON.stringify({ message: textEvaluationError.message, stack: textEvaluationError.stack, name: textEvaluationError.name }, null, 2)}</pre>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default TextEvaluation;