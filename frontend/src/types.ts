export interface TicketOutcome {
  ticket_id: string;
  run_id: string;
  resolution: string;
  ticket_status: string;
  customer_reply: string | null;
  escalation_reason: string | null;
  iterations: number;
  duration_s: number;
}
