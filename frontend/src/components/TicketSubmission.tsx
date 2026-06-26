import React, { useState } from 'react';
import { Send, User } from 'lucide-react';
import type { TicketOutcome } from '../types';

interface Props {
  onSubmit: (outcome: TicketOutcome) => void;
}

export function TicketSubmission({ onSubmit }: Props) {
  const [customerId, setCustomerId] = useState('cust_001');
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subject.trim() || !body.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch('/tickets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customer_id: customerId,
          subject,
          body,
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit ticket');
      }
      
      const outcome: TicketOutcome = await response.json();
      onSubmit(outcome);
      setSubject('');
      setBody('');
    } catch (error) {
      console.error(error);
      alert('Error submitting ticket. Ensure backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="glass-panel">
      <h2>Submit a Ticket</h2>
      <p style={{ marginBottom: '1.5rem' }}>Describe your issue and the autonomous agent will resolve it.</p>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">
            <User size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} /> 
            Customer Profile
          </label>
          <select 
            className="form-select"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            disabled={isLoading}
          >
            <option value="cust_001">Asha Rao (cust_001)</option>
            <option value="cust_002">Ben Ortiz (cust_002)</option>
            <option value="cust_003">Priya Nair (cust_003)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Subject</label>
          <input 
            type="text" 
            className="form-input" 
            placeholder="E.g., Broken mouse"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Message Body</label>
          <textarea 
            className="form-textarea" 
            placeholder="Explain the issue in detail..."
            value={body}
            onChange={(e) => setBody(e.target.value)}
            required
            disabled={isLoading}
          />
        </div>

        <button type="submit" className="btn" disabled={isLoading || !subject.trim() || !body.trim()}>
          {isLoading ? (
            <>
              <div className="spinner"></div> Processing (Takes ~10s)...
            </>
          ) : (
            <>
              <Send size={18} /> Send Ticket
            </>
          )}
        </button>
      </form>
    </div>
  );
}
