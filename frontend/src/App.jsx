import { useState } from "react";
import Header from "./components/Header";
import QueryInput from "./components/QueryInput";
import ResultCard from "./components/ResultCard";
import Loading from "./components/Loading";
import { analyzeSupportQuery } from "./api";
import "./App.css";

export default function App() {
  const [message, setMessage] = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleSubmit = async () => {
    if (!message.trim()) {
      setError("Please enter your support issue.");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const res = await analyzeSupportQuery(message.trim());
      setResult(res.data);
    } catch (err) {
      if (err.code === "ERR_NETWORK" || err.code === "ECONNREFUSED") {
        setError("Unable to connect to the AI backend. Make sure FastAPI is running on port 8000.");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessage("");
    setResult(null);
    setError("");
  };

  return (
    <div className="app">
      <Header />
      <main className="main">
        <QueryInput
          message={message}
          setMessage={setMessage}
          onSubmit={handleSubmit}
          onClear={handleClear}
          loading={loading}
        />

        {error && (
          <div className="error-banner">{error}</div>
        )}

        {loading && <Loading />}

        {result && !loading && (
          <div className="result-animate">
            <ResultCard data={result} />
          </div>
        )}
      </main>
    </div>
  );
}
