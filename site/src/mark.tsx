export function Mark({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <path d="M16 4 L16 12" stroke="currentColor" strokeWidth="1.2" />
      <path d="M16 12 L6 24" stroke="currentColor" strokeWidth="1.2" />
      <path d="M16 12 L16 24" stroke="currentColor" strokeWidth="1.2" />
      <path d="M16 12 L26 24" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="16" cy="4" r="1.6" fill="currentColor" />
      <circle cx="6" cy="24" r="1.5" fill="none" stroke="currentColor" />
      <circle cx="16" cy="24" r="2.1" fill="currentColor" />
      <circle cx="26" cy="24" r="1.5" fill="none" stroke="currentColor" />
    </svg>
  );
}

export function Lattice() {
  const nodes = [
    [40, 28],
    [120, 78],
    [210, 46],
    [300, 92],
    [380, 38],
  ];
  return (
    <svg className="lattice" viewBox="0 0 420 140" role="img" aria-label="Decision lattice">
      <title>A branching decision field</title>
      <path d="M210 18 C210 48 120 48 120 78" fill="none" stroke="currentColor" strokeWidth="1" />
      <path d="M210 18 C210 48 210 48 210 78" fill="none" stroke="currentColor" strokeWidth="1" />
      <path d="M210 18 C210 48 300 48 300 78" fill="none" stroke="currentColor" strokeWidth="1" />
      <path d="M120 78 C120 104 40 104 40 120" fill="none" stroke="currentColor" strokeWidth="1" opacity="0.45" />
      <path d="M300 78 C300 104 380 104 380 120" fill="none" stroke="currentColor" strokeWidth="1" opacity="0.45" />
      <circle cx="210" cy="18" r="3" fill="currentColor" />
      {nodes.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={x} cy={y} r={x === 210 ? 4 : 2.4} fill={x === 210 ? "currentColor" : "none"} stroke="currentColor" />
      ))}
    </svg>
  );
}
