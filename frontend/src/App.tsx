import React, { useState } from 'react';
import { TicketSubmission } from './components/TicketSubmission';
import { TicketResponse } from './components/TicketResponse';
import type { TicketOutcome } from './types';
import { Headset } from 'lucide-react';

function App() {
  const [currentOutcome, setCurrentOutcome] = useState<TicketOutcome | null>(null);

  return (
    <div>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <Headset size={48} color="var(--accent-color)" style={{ marginBottom: '1rem' }} />
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', background: 'linear-gradient(to right, #6366f1, #10b981)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          Support Agent Lens
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>Autonomous ticket resolution powered by LLMs</p>
      </div>

      <div className="app-container">
        <TicketSubmission onSubmit={setCurrentOutcome} />
        <TicketResponse outcome={currentOutcome} />
      </div>
    </div>
  );
}

export default App;
