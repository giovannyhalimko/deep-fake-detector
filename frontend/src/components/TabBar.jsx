import Button from "./Button";

export default function TabBar({ activeTab, setActiveTab, onTabChange }) {
  const tabs = ["video", "image", "history"];

  const handleTabClick = (tab) => {
    setActiveTab(tab);
    onTabChange?.();
  };

  return (
    <div className="flex justify-center gap-2 mb-6">
      {tabs.map((tab) => (
        <Button
          key={tab}
          onClick={() => handleTabClick(tab)}
          variant={activeTab === tab ? "tabActive" : "tab"}
          size="sm"
        >
          {tab === "video" && "🎥 Video"}
          {tab === "image" && "🖼️ Image"}
          {tab === "history" && "📋 History"}
        </Button>
      ))}
    </div>
  );
}
