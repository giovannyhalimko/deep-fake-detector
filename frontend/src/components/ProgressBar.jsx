export default function ProgressBar({ progress, progressText }) {
  return (
    <div className="mt-6 animate-fadeIn">
      <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-linear-to-r from-indigo-500 to-purple-600 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="text-center text-gray-600">{progressText}</p>
    </div>
  );
}
