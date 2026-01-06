import { useRef, useEffect, useState } from "react";
import Button from "./Button";

export default function MediaPreview({
  file,
  previewUrl,
  activeTab,
  onAnalyze,
  isAnalyzing,
}) {
  const videoRef = useRef(null);
  const [mediaInfo, setMediaInfo] = useState(null);

  useEffect(() => {
    setMediaInfo(null);
  }, [file]);

  const handleVideoLoad = () => {
    if (videoRef.current && file) {
      setMediaInfo({
        duration: videoRef.current.duration,
        size: (file.size / (1024 * 1024)).toFixed(2),
      });
    }
  };

  const handleImageLoad = (e) => {
    if (file) {
      setMediaInfo({
        width: e.target.naturalWidth,
        height: e.target.naturalHeight,
        size: (file.size / (1024 * 1024)).toFixed(2),
      });
    }
  };

  return (
    <div className="animate-fadeIn">
      {/* File Info */}
      <div className="bg-green-50 p-3 rounded-xl mb-4">
        <span className="font-bold text-green-700">{file.name}</span>
      </div>

      {/* Video Preview */}
      {activeTab === "video" && (
        <video
          ref={videoRef}
          controls
          className="w-full rounded-xl shadow-lg bg-black"
          onLoadedMetadata={handleVideoLoad}
        >
          <source src={previewUrl} />
        </video>
      )}

      {/* Image Preview */}
      {activeTab === "image" && (
        <img
          src={previewUrl}
          alt="Preview"
          className="w-full max-h-96 object-contain rounded-xl shadow-lg"
          onLoad={handleImageLoad}
        />
      )}

      {/* Media Info */}
      {mediaInfo && (
        <div className="mt-3 p-3 bg-indigo-50 rounded-lg text-sm text-gray-600">
          {activeTab === "video" ? (
            <>
              <strong>Duration:</strong> {mediaInfo.duration?.toFixed(2)}s |
              <strong> Size:</strong> {mediaInfo.size} MB
            </>
          ) : (
            <>
              <strong>Dimensions:</strong> {mediaInfo.width} ×{" "}
              {mediaInfo.height} |<strong> Size:</strong> {mediaInfo.size} MB
            </>
          )}
        </div>
      )}

      {/* Analyze Button */}
      {!isAnalyzing && (
        <div className="text-center mt-6">
          <Button onClick={onAnalyze} size="lg">
            🔍 Analyze {activeTab === "video" ? "Video" : "Image"}
          </Button>
        </div>
      )}
    </div>
  );
}
