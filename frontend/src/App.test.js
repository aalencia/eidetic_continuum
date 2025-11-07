import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock the fetch API
global.fetch = jest.fn();

describe('App Component', () => {
  beforeEach(() => {
    fetch.mockClear();
    // Default mock for all fetches
    fetch.mockImplementation((url) => {
      if (url.includes('/get-decision-history')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve([]), // Always return an empty array for history by default
        });
      } else if (url.includes('/evaluate')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ decision: 'Approval', reasoning: 'Mocked approval' }),
        });
      } else if (url.includes('/evaluate-text')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ evaluation: { status: 'Mocked evaluation' } }),
        });
      } else if (url.includes('/log-error')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ status: 'error logged' }),
        });
      }
      return Promise.reject(new Error(`Unhandled fetch for ${url}`));
    });
  });

  test('renders main headings and components', async () => {
    render(<App />);
    expect(screen.getByText('Eidetic Continuum Governance')).toBeInTheDocument();
    expect(screen.getByText('Governance Sectors')).toBeInTheDocument();
    expect(screen.getByText('Constitutional Principles')).toBeInTheDocument();
    expect(screen.getByText('Transaction Evaluation')).toBeInTheDocument();
    expect(screen.getByText('Text Description Evaluation')).toBeInTheDocument();
    expect(screen.getByText('Decision History')).toBeInTheDocument();
    
    expect(screen.getByRole('spinbutton', { name: /Transaction Amount/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Destination Country Risk/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Purpose/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Security Risk Level/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Evaluate Transaction/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Text Description/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Evaluate Text/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh Decisions' })).toBeInTheDocument();

    await waitFor(() => expect(screen.getByText('No decisions made yet.')).toBeInTheDocument());
  });

  describe('Transaction Evaluation', () => {
    test('allows user to input transaction details', () => {
      render(<App />);
      const amountInput = screen.getByRole('spinbutton', { name: /Transaction Amount/i });
      const countryInput = screen.getByRole('textbox', { name: /Destination Country Risk/i });
      const purposeInput = screen.getByRole('textbox', { name: /Purpose/i });
      const securityInput = screen.getByRole('textbox', { name: /Security Risk Level/i });

      fireEvent.change(amountInput, { target: { value: '1000' } });
      fireEvent.change(countryInput, { target: { value: 'low_risk' } });
      fireEvent.change(purposeInput, { target: { value: 'test_purpose' } });
      fireEvent.change(securityInput, { target: { value: 'low' } });

      expect(amountInput).toHaveValue(1000);
      expect(countryInput).toHaveValue('low_risk');
      expect(purposeInput).toHaveValue('test_purpose');
      expect(securityInput).toHaveValue('low');
    });

    test('populates fields when a use case is selected', async () => {
      render(<App />);
      const selectUseCase = screen.getByLabelText('Select Use Case');
      fireEvent.mouseDown(selectUseCase);
      
      const humanitarianOption = await screen.findByRole('option', { name: /Humanitarian Exception/i });
      fireEvent.click(humanitarianOption);

      expect(screen.getByRole('spinbutton', { name: /Transaction Amount/i })).toHaveValue(120000);
      expect(screen.getByRole('textbox', { name: /Destination Country Risk/i })).toHaveValue('sanctioned');
      expect(screen.getByRole('textbox', { name: /Purpose/i })).toHaveValue('medical_aid');
      expect(screen.getByRole('textbox', { name: /Security Risk Level/i })).toHaveValue('low');
    });

    test('submits transaction and displays success response', async () => {
      const mockTransactionResponse = {
        decision: 'APPROVE with enhanced documentation',
        reasoning: 'While this violates trade sanctions, the principles of \'Protect vulnerable populations\' and \'Promote beneficial outcomes\' override in this humanitarian context. OFAC exemption #774 applies.',
        principles_evaluated: ['Comply with international laws', 'Prevent money laundering'],
        precedent_cited: 'Case Study 1: Humanitarian Exception',
        violation_detected: true,
        require_interpretation: true,
      };
      const mockDecisionHistory = [
        {
          timestamp: new Date().toISOString(),
          transaction_input: { transaction_amount: 120000, destination_country_risk: 'sanctioned', purpose: 'medical_aid', security_risk_level: 'low' },
          decision_output: mockTransactionResponse,
        },
      ];

      fetch.mockImplementation((url) => {
        if (url.includes('/evaluate')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockTransactionResponse),
          });
        }
        if (url.includes('/get-decision-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockDecisionHistory),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<App />);
      const amountInput = screen.getByRole('spinbutton', { name: /Transaction Amount/i });
      const countryInput = screen.getByRole('textbox', { name: /Destination Country Risk/i });
      const purposeInput = screen.getByRole('textbox', { name: /Purpose/i });
      const securityInput = screen.getByRole('textbox', { name: /Security Risk Level/i });
      const evaluateButton = screen.getByRole('button', { name: /Evaluate Transaction/i });

      fireEvent.change(amountInput, { target: { value: '120000' } });
      fireEvent.change(countryInput, { target: { value: 'sanctioned' } });
      fireEvent.change(purposeInput, { target: { value: 'medical_aid' } });
      fireEvent.click(evaluateButton);

      expect(evaluateButton).toHaveTextContent('Evaluating...');
      await waitFor(() => expect(screen.getByText('Transaction Response:')).toBeInTheDocument());
      
      const transactionResponseCard = screen.getByTestId('transaction-response-card');
      expect(within(transactionResponseCard).getByText('Decision:')).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText('APPROVE with enhanced documentation')).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText((content, element) => content.startsWith('Reasoning:'))).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText(/While this violates trade sanctions/i)).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText(/Principles Evaluated:/i)).toBeInTheDocument();

      expect(within(transactionResponseCard).getByText(/Precedent Cited: Case Study 1: Humanitarian Exception/i)).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText('Constitutional Violation Detected:')).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText('Yes')).toBeInTheDocument();
      expect(within(transactionResponseCard).getByText(/Requires AI Interpretation: Yes/i)).toBeInTheDocument();

      expect(evaluateButton).toHaveTextContent('Evaluate Transaction');

      // Assert that the decision history is updated
      await waitFor(() => expect(screen.getByText(/Input: {"transaction_amount":120000/i)).toBeInTheDocument());
      expect(screen.getByText(/Decision: APPROVE with enhanced documentation/i)).toBeInTheDocument();
    });

    // Fix for error response test
    test('submits transaction and displays error response', async () => {
      const mockError = new Error('Network error');
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/evaluate')) {
          return Promise.reject(mockError);
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      }).mockImplementationOnce((url) => { // Mock the log-error call that happens after error
        if (url.includes('/log-error')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'error logged' }) });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      render(<App />);
      const amountInput = screen.getByRole('spinbutton', { name: /Transaction Amount/i });
      const countryInput = screen.getByRole('textbox', { name: /Destination Country Risk/i });
      const purposeInput = screen.getByRole('textbox', { name: /Purpose/i });
      const securityInput = screen.getByRole('textbox', { name: /Security Risk Level/i });
      const evaluateButton = screen.getByRole('button', { name: /Evaluate Transaction/i });

      fireEvent.change(amountInput, { target: { value: '1000' } });
      fireEvent.change(countryInput, { target: { value: 'low_risk' } });
      fireEvent.change(purposeInput, { target: { value: 'test_purpose' } });
      fireEvent.click(evaluateButton);

      expect(evaluateButton).toHaveTextContent('Evaluating...');
      await waitFor(() => expect(screen.getByText('Transaction Error:')).toBeInTheDocument());
      expect(screen.getByText(/Network error/i)).toBeInTheDocument(); // Simplified assertion
      expect(screen.getByText(/"name": "Error"/i)).toBeInTheDocument();
      expect(evaluateButton).toHaveTextContent('Evaluate Transaction');
    });
  });

  describe('Text Description Evaluation', () => {
    test('allows user to input text description', () => {
      render(<App />);
      const textInput = screen.getByRole('textbox', { name: /Text Description/i });

      fireEvent.change(textInput, { target: { value: 'This is a test description.' } });
      expect(textInput).toHaveValue('This is a test description.');
    });

    // Fix for success response test
    test('evaluates text and displays success response', async () => {
      const mockResponse = { evaluation: { status: 'Under Review', reason: 'General text description, requires manual review.', score: 0.6 } };
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/evaluate-text')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockResponse),
          });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      render(<App />);
      const textInput = screen.getByRole('textbox', { name: /Text Description/i });
      const evaluateButton = screen.getByRole('button', { name: /Evaluate Text/i });

      fireEvent.change(textInput, { target: { value: 'Some text to evaluate.' } });
      fireEvent.click(evaluateButton);

      expect(evaluateButton).toHaveTextContent('Evaluating...');
      await waitFor(() => expect(screen.getByText('Text Evaluation Response:')).toBeInTheDocument());
      expect(screen.getByText(/Under Review/i)).toBeInTheDocument(); // Simplified assertion
      expect(screen.getByText(/General text description, requires manual review./i)).toBeInTheDocument(); // More specific check
      expect(evaluateButton).toHaveTextContent('Evaluate Text');
    });

    // Fix for error response test
    test('evaluates text and displays error response', async () => {
      const mockError = new Error('Text evaluation failed');
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/evaluate-text')) {
          return Promise.reject(mockError);
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      }).mockImplementationOnce((url) => { // Mock the log-error call that happens after error
        if (url.includes('/log-error')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'error logged' }) });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });


      render(<App />);
      const textInput = screen.getByRole('textbox', { name: /Text Description/i });
      const evaluateButton = screen.getByRole('button', { name: /Evaluate Text/i });

      fireEvent.change(textInput, { target: { value: 'Some text to evaluate.' } });
      fireEvent.click(evaluateButton);

      expect(evaluateButton).toHaveTextContent('Evaluating...');
      await waitFor(() => expect(screen.getByText('Text Evaluation Error:')).toBeInTheDocument());
      expect(screen.getByText(/Text evaluation failed/i)).toBeInTheDocument(); // Simplified assertion
      expect(screen.getByText(/"name": "Error"/i)).toBeInTheDocument();
      expect(evaluateButton).toHaveTextContent('Evaluate Text');
    });
  });

  describe('Decision History', () => {
    test('fetches and displays decision history on initial load', async () => {
      const mockDecisionHistory = [
        {
          timestamp: new Date().toISOString(),
          transaction_input: { transaction_amount: 100, destination_country_risk: 'low', purpose: 'test', security_risk_level: 'low' },
          decision_output: { decision: 'Approval', reasoning: 'No constitutional concerns' },
        },
      ];
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/get-decision-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockDecisionHistory),
          });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      render(<App />);
      await waitFor(() => expect(screen.getByText('Decision History')).toBeInTheDocument());
      expect(screen.getByText(/Transaction Amount: 100/i)).toBeInTheDocument(); // Simplified assertion
      expect(screen.getByText(/Decision: Approval/i)).toBeInTheDocument();
    });

    test('refreshes decision history when refresh button is clicked', async () => {
      const initialHistory = [
        {
          timestamp: new Date().toISOString(),
          transaction_input: { transaction_amount: 100, destination_country_risk: 'low', purpose: 'test', security_risk_level: 'low' },
          decision_output: { decision: 'Approval', reasoning: 'No constitutional concerns' },
        },
      ];
      const refreshedHistory = [
        ...initialHistory,
        {
          timestamp: new Date().toISOString(),
          transaction_input: { transaction_amount: 200, destination_country_risk: 'medium', purpose: 'another test', security_risk_level: 'medium' },
          decision_output: { decision: 'Rejection', reasoning: 'Clear constitutional violation' },
        },
      ];

      // Mock initial fetch
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/get-decision-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(initialHistory),
          });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      render(<App />);
      await waitFor(() => expect(screen.getByText(/Transaction Amount: 100/i)).toBeInTheDocument()); // Simplified assertion

      // Mock fetch for refresh button click
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/get-decision-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(refreshedHistory),
          });
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      const refreshButton = screen.getByRole('button', { name: 'Refresh Decisions' });
      fireEvent.click(refreshButton);

      await waitFor(() => expect(screen.getByText(/Transaction Amount: 200/i)).toBeInTheDocument()); // Simplified assertion
      expect(screen.getByText(/Decision: Rejection/i)).toBeInTheDocument();
    });

    test('displays error message if fetching history fails', async () => {
      fetch.mockImplementationOnce((url) => {
        if (url.includes('/get-decision-history')) {
          return Promise.reject(new Error('Failed to fetch history'));
        }
        return Promise.reject(new Error(`Unhandled fetch for ${url}`));
      });

      render(<App />);
      await waitFor(() => expect(screen.getByText(/Error loading history: Failed to fetch history/i)).toBeInTheDocument());
    });
  });
});