export default function RetrievedConversation({ result, index }) {
  const pct = Math.round(result.similarity * 100);

  return (
    <div className="conv-card">
      <div className="conv-header">
        <span className="conv-rank">Historical Match #{index + 1}</span>
        <span className="conv-similarity">{pct}% match</span>
      </div>

      <div className="conv-bar-wrap">
        <div className="conv-bar" style={{ width: `${pct}%` }} />
      </div>

      <div className="conv-body">
        <div className="conv-block">
          <span className="conv-label">Customer</span>
          <p className="conv-text">"{result.customer_message}"</p>
        </div>
        <div className="conv-block">
          <span className="conv-label">Support Response</span>
          <p className="conv-text">{result.support_response}</p>
        </div>
      </div>
    </div>
  );
}
