export default function QueryInput({ message, setMessage, onSubmit, onClear, loading }) {
  return (
    <section className="query-section">
      <h1 className="query-title">How can we help you?</h1>
      <p className="query-subtitle">Describe your Apple product or support issue.</p>

      <textarea
        className="query-textarea"
        placeholder="Example: My iPhone battery is draining very quickly after the latest update..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        rows={5}
        disabled={loading}
      />

      <div className="query-actions">
        <button
          className="btn btn-primary"
          onClick={onSubmit}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze Issue"}
        </button>
        <button
          className="btn btn-secondary"
          onClick={onClear}
          disabled={loading}
        >
          Clear
        </button>
      </div>
    </section>
  );
}
