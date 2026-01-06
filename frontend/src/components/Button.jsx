export default function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  className = "",
  ...props
}) {
  const baseStyles =
    "font-semibold rounded-full transition-all duration-300 cursor-pointer";

  const variants = {
    primary:
      "bg-linear-to-r from-indigo-500 to-purple-600 text-white hover:shadow-xl hover:-translate-y-1",
    tab: "bg-white/20 text-white hover:bg-white/30",
    tabActive: "bg-white text-purple-600 shadow-lg",
  };

  const sizes = {
    sm: "px-6 py-2 text-sm",
    md: "px-8 py-3 text-base",
    lg: "px-10 py-3 text-lg",
  };

  return (
    <button
      onClick={onClick}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
