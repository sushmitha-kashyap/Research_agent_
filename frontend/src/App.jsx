import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    if (!query.trim()) return;

    setLoading(true);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/research`,
        
       {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      setReport(data.report);
    } catch (err) {
      console.error(err);
      setReport("Something went wrong.");
    }

    setLoading(false);
  }
  console.log(import.meta.env.VITE_API_URL);
  console.log(import.meta.env);

  return (
    <div className="container">
      <h1>AI Research Assistant</h1>

      <textarea
        placeholder="Enter your research query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button onClick={handleSubmit}>
        {loading ? "Researching..." : "Research"}
      </button>

      <div className="output-box">
        <h2>Research Report</h2>
        <p>{report}</p>
      </div>
    </div>
  );
}

export default App;