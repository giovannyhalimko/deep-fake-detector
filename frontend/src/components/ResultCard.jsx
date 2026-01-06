import { getConfidence } from "../utils";
import Button from "./Button";

export default function ResultCard({ result, onReset }) {
  return (
    <div className="mt-6 animate-slideIn">
      {/* Video Result */}
      {result.type === "video" && (
        <div
          className={`p-8 rounded-xl text-center text-white ${
            result.data.label === "FAKE"
              ? "bg-gradient-to-r from-red-500 to-rose-500"
              : "bg-gradient-to-r from-green-500 to-emerald-500"
          }`}
        >
          <div className="text-5xl font-bold mb-2">{result.data.label}</div>
          <div className="text-2xl">
            {getConfidence(result.data.prediction).toFixed(1)}% confidence
          </div>
        </div>
      )}

      {/* Image Result with Heatmaps */}
      {result.type === "image" && (
        <div>
          <div
            className={`p-6 rounded-xl text-center text-white mb-4 ${
              result.data.overall_label === "FAKE"
                ? "bg-gradient-to-r from-red-500 to-rose-500"
                : result.data.overall_label === "REAL"
                ? "bg-gradient-to-r from-green-500 to-emerald-500"
                : "bg-gray-500"
            }`}
          >
            <div className="text-4xl font-bold">
              {result.data.overall_label}
            </div>
            {result.data.overall_prediction && (
              <div className="text-xl mt-1">
                {getConfidence(result.data.overall_prediction).toFixed(1)}%
                confidence
              </div>
            )}
          </div>

          {/* Face-by-face results with heatmaps */}
          {result.data.faces?.length > 0 && (
            <div className="space-y-4">
              <h3 className="font-bold text-gray-800">
                Detected Faces ({result.data.faces_detected})
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.data.faces.map((face, idx) => (
                  <div key={idx} className="bg-gray-50 p-4 rounded-xl">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">Face {idx + 1}</span>
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-bold ${
                          face.label === "FAKE"
                            ? "bg-red-500 text-white"
                            : "bg-green-500 text-white"
                        }`}
                      >
                        {face.label} (
                        {getConfidence(face.prediction).toFixed(0)}%)
                      </span>
                    </div>
                    {face.heatmap && (
                      <div>
                        <p className="text-sm text-gray-500 mb-2">
                          Heatmap (areas of concern):
                        </p>
                        <img
                          src={`data:image/jpeg;base64,${face.heatmap}`}
                          alt={`Heatmap for face ${idx + 1}`}
                          className="w-full rounded-lg"
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Analyze Another */}
      <div className="text-center mt-6">
        <Button onClick={onReset}>Analyze Another</Button>
      </div>
    </div>
  );
}
