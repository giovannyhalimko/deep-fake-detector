import { useRef } from "react";

export default function UploadArea({
  activeTab,
  onFileSelect,
  isDragOver,
  setIsDragOver,
}) {
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const acceptedFiles = activeTab === "video" ? "video/*" : "image/*";

  return (
    <div
      className={`border-3 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${
        isDragOver
          ? "border-purple-600 bg-purple-50 scale-[1.02]"
          : "border-indigo-300 bg-indigo-50 hover:border-purple-500 hover:bg-purple-50"
      }`}
      onClick={() => fileInputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
    >
      <div className="text-6xl mb-4">{activeTab === "video" ? "🎥" : "🖼️"}</div>
      <div className="text-gray-600">
        <p className="font-semibold text-lg">
          Click to upload or drag and drop
        </p>
        <p className="text-sm mt-1">
          {activeTab === "video"
            ? "MP4, AVI, MOV (max 20 seconds)"
            : "JPG, PNG, WEBP"}
        </p>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptedFiles}
        className="hidden"
        onChange={(e) => onFileSelect(e.target.files?.[0])}
      />
    </div>
  );
}
