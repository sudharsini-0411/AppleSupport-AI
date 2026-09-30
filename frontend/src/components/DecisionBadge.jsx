export default function DecisionBadge({ decision }) {
  const isAuto = decision === "AUTO-HANDLED";
  return (
    <span className={`badge ${isAuto ? "badge-auto" : "badge-escalated"}`}>
      {decision}
    </span>
  );
}
