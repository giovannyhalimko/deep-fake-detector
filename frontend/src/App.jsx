import { useState } from "react";
import API_URL from "./config";
import { simulateProgress } from "./utils";
import Header from "./components/Header";
import TabBar from "./components/TabBar";
import UploadArea from "./components/UploadArea";
import MediaPreview from "./components/MediaPreview";
import ProgressBar from "./components/ProgressBar";
import ResultCard from "./components/ResultCard";
import HistoryList from "./components/HistoryList";

function App() {
  const [activeTab, setActiveTab] = useState("video");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const resetUI = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setProgressText("");
  };

  const analyzeMedia = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setResult(null);

    const progressRef = { current: null };
    simulateProgress(setProgress, setProgressText, progressRef);

    try {
      const formData = new FormData();
      const endpoint = activeTab === "video" ? "/predict" : "/predict-images";
      const fieldName = activeTab === "video" ? "video" : "image";

      formData.append(fieldName, selectedFile);

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      clearInterval(progressRef.current);
      setProgress(100);
      setProgressText("Complete!");

      setTimeout(() => {
        setIsAnalyzing(false);
        if (data.success && data.results?.length > 0) {
          setResult({ type: activeTab, data: data.results[0] });
        } else {
          setError(data.error || "Analysis failed");
        }
      }, 500);
    } catch (err) {
      clearInterval(progressRef.current);
      setIsAnalyzing(false);
      setError("Connection failed: " + err.message);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-indigo-500 to-purple-600 p-5">
      <div className="max-w-4xl mx-auto">
        <Header />

        <TabBar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onTabChange={resetUI}
        />

        <div className="bg-white rounded-2xl shadow-2xl p-8">
          {/* History Tab */}
          {activeTab === "history" && <HistoryList />}

          {/* Video/Image Upload Tabs */}
          {(activeTab === "video" || activeTab === "image") && (
            <>
              {/* Upload Area */}
              {!selectedFile && (
                <UploadArea
                  activeTab={activeTab}
                  onFileSelect={handleFileSelect}
                  isDragOver={isDragOver}
                  setIsDragOver={setIsDragOver}
                />
              )}

              {/* Preview */}
              {previewUrl && !result && (
                <MediaPreview
                  file={selectedFile}
                  previewUrl={previewUrl}
                  activeTab={activeTab}
                  onAnalyze={analyzeMedia}
                  isAnalyzing={isAnalyzing}
                />
              )}

              {/* Progress */}
              {isAnalyzing && (
                <ProgressBar progress={progress} progressText={progressText} />
              )}

              {/* Error */}
              {error && (
                <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-xl">
                  ❌ {error}
                </div>
              )}

              {/* Results */}
              {result && <ResultCard result={result} onReset={resetUI} />}
            </>
          )}
        </div>

        <p className="text-center text-indigo-200 mt-6 text-sm">
          Powered by EfficientNet-B7 • Built with ❤️
        </p>
        <div className="text-center text-indigo-300 mt-4 text-xs opacity-80">
          <p className="font-semibold mb-1">© 2026 Kelompok_1</p>
          <p>Aswin Angkasa (221113724)</p>
          <p>Samuel Onasis (221110680)</p>
          <p>Giovanny Halimko (221110058)</p>
        </div>
      </div>
    </div>
  );
}

export default App;
