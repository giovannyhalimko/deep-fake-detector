import { useEffect, useState } from "react";
import API_URL from "../config";
import { getConfidence } from "../utils";

export default function HistoryList() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_URL}/history?limit=50`);
      const data = await response.json();
      if (data.success) {
        setHistory(data.history);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-800 mb-4">Scan History</h2>
      {history.length === 0 ? (
        <p className="text-gray-500 text-center py-8">No scans yet</p>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {history.map((item) => (
            <div
              key={item.id}
              className={`p-4 rounded-xl flex justify-between items-center ${
                item.label === "FAKE" ? "bg-red-50" : "bg-green-50"
              }`}
            >
              <div>
                <p className="font-medium text-gray-800">{item.filename}</p>
                <p className="text-sm text-gray-500">{item.timestamp}</p>
              </div>
              <div className="text-right">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-bold ${
                    item.label === "FAKE"
                      ? "bg-red-500 text-white"
                      : "bg-green-500 text-white"
                  }`}
                >
                  {item.label}
                </span>
                <p className="text-sm text-gray-500 mt-1">
                  {getConfidence(item.prediction).toFixed(1)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
