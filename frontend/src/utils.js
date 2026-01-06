export function getConfidence(prediction) {
  return prediction < 0.5 ? (1 - prediction) * 100 : prediction * 100;
}

export function simulateProgress(setProgress, setProgressText, progressRef) {
  let currentProgress = 0;
  progressRef.current = setInterval(() => {
    if (currentProgress < 10) {
      currentProgress += 2;
      setProgressText("Uploading...");
    } else if (currentProgress < 30) {
      currentProgress += 1.5;
      setProgressText("Processing...");
    } else if (currentProgress < 60) {
      currentProgress += 1;
      setProgressText("Extracting faces...");
    } else if (currentProgress < 85) {
      currentProgress += 0.5;
      setProgressText("Running AI analysis...");
    } else if (currentProgress < 95) {
      currentProgress += 0.2;
      setProgressText("Finalizing...");
    }
    setProgress(currentProgress);
  }, 150);
}
