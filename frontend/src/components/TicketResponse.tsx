import React from 'react';
import type { TicketOutcome } from '../types';
import { CheckCircle2, AlertTriangle, Clock, RefreshCw, Hash } from 'lucide-react';

interface Props {
  outcome: TicketOutcome | null;
}

export function TicketResponse({ outcome }: Props) {
  if (!outcome) {
    return (
      <div className="glass-panel" style={{ height: '100%' }}>
        <div className="empty-state">
          <RefreshCw size={48} />
          <h3>Awaiting Request</h3>
          <p>Submit a ticket on the left to see the agent's response here.</p>
        </div>
      </div>
    );
  }

  const isResolved = outcome.resolution === 'resolved';

  return (
    <div className="glass-panel">
      <div className="response-header">
        <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
          {isResolved ? (
            <CheckCircle2 color="var(--success-color)" />
          ) : (
            <AlertTriangle color="var(--danger-color)" />
          )}
          Ticket {outcome.resolution}
        </h2>
        <span className={`badge ${isResolved ? 'badge-success' : 'badge-danger'}`}>
          {outcome.resolution}
        </span>
      </div>

      <div className="stat-group" style={{ marginBottom: '1.5rem' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Clock size={14} /> {outcome.duration_s.toFixed(1)}s
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <RefreshCw size={14} /> {outcome.iterations} turns
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Hash size={14} /> {outcome.ticket_id}
        </span>
      </div>

      <div className="response-box">
        {isResolved ? (
          <>
            <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Customer Reply:</h4>
            <p style={{ color: 'white', whiteSpace: 'pre-wrap' }}>{outcome.customer_reply}</p>
          </>
        ) : (
          <>
            <h4 style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Escalation Reason:</h4>
            <p style={{ color: 'white' }}>{outcome.escalation_reason}</p>
            {outcome.customer_reply && (
              <>
                <h4 style={{ color: 'var(--text-secondary)', marginTop: '1rem', marginBottom: '0.5rem', fontSize: '0.875rem' }}>Customer Reply:</h4>
                <p style={{ color: 'white', whiteSpace: 'pre-wrap' }}>{outcome.customer_reply}</p>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
