import DecisionBadge from "./DecisionBadge";
import RetrievedConversation from "./RetrievedConversation";

export default function ResultCard({ data }) {
  const pct = Math.round((data.best_similarity || 0) * 100);

  return (
    <div className="result-wrapper">

      {/* Pipeline indicator */}
      <div className="pipeline">
        {["Customer Query", "AI Analysis", "FAISS Retrieval", "Gemini Response", data.decision].map((step, i, arr) => (
          <span key={i} className="pipeline-step">
            <span className={i === arr.length - 1
              ? (data.decision === "AUTO-HANDLED" ? "pipeline-final-auto" : "pipeline-final-escalated")
              : "pipeline-label"}>
              {step}
            </span>
            {i < arr.length - 1 && <span className="pipeline-arrow">→</span>}
          </span>
        ))}
      </div>

      {/* Top metrics row */}
      <div className="metrics-row">

        <div className="metric-card">
          <span className="metric-label">Decision</span>
          <DecisionBadge decision={data.decision} />
        </div>

        <div className="metric-card">
          <span className="metric-label">Best Historical Match</span>
          <span className="metric-value">{pct}%</span>
          <div className="metric-bar-wrap">
            <div className="metric-bar" style={{ width: `${pct}%` }} />
          </div>
        </div>

        <div className="metric-card">
          <span className="metric-label">Intent Classification</span>
          <span className="metric-intent">
            {data.intent || "Intent classification"}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Reason</span>
          <span className="metric-reason">{data.reason}</span>
        </div>

      </div>

      {/* Customer query */}
      <div className="section-card">
        <h3 className="section-title">Customer Query</h3>
        <p className="section-text">"{data.customer_message}"</p>
      </div>

      {/* AI response */}
      <div className="section-card response-card">
        <h3 className="section-title">
          <span className="dot dot-blue" />
          AI Support Response
        </h3>
        <p className="section-text response-text">{data.generated_response}</p>
      </div>

      {/* Retrieved conversations */}
      {data.retrieved_conversations?.length > 0 && (
        <div className="section-card">
          <h3 className="section-title">
            <span className="dot dot-gray" />
            Retrieved Historical Conversations
          </h3>
          <div className="conv-list">
            {data.retrieved_conversations.map((r, i) => (
              <RetrievedConversation key={i} result={r} index={i} />
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
