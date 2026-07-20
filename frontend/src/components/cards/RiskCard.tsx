import { UserRound } from "lucide-react";
import { pct } from "../../lib/format";
import { Card } from "../Card";
import { RiskBadge, RiskGaugeBar } from "../indicators";

interface ConfidenceControl {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Fired when the user finishes dragging (mouse/touch/key release). */
  onCommit?: () => void;
}

interface RiskCardProps {
  riskProbability: number;
  isHighRisk: boolean;
  /** When supplied (scientific view), shows the model-certainty slider. */
  confidence?: ConfidenceControl;
}

/** "Treatment Resistance Risk" headline card with gauge and optional CI slider. */
export function RiskCard({ riskProbability, isHighRisk, confidence }: RiskCardProps) {
  return (
    <Card
      icon={UserRound}
      title="Treatment Resistance Risk"
      action={<RiskBadge isHighRisk={isHighRisk} />}
      className="flex flex-col"
    >
      <p className="text-6xl font-bold text-slate-900 tracking-tight text-center mt-2">{pct(riskProbability)}</p>
      <RiskGaugeBar probability={riskProbability} />

      {confidence && (
        <div className="mt-6">
          <label className="flex flex-col gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Model Certainty — Confidence Interval (%)
            </span>
            <input
              type="range"
              min={confidence.min}
              max={confidence.max}
              step={confidence.step}
              value={confidence.value}
              onChange={(event) => confidence.onChange(Number(event.target.value))}
              onMouseUp={() => confidence.onCommit?.()}
              onTouchEnd={() => confidence.onCommit?.()}
              onKeyUp={() => confidence.onCommit?.()}
              className="w-full accent-slate-900"
            />
            <div className="flex justify-between text-[11px] text-slate-400">
              <span>{confidence.min}</span>
              <span>{confidence.value}</span>
              <span>{confidence.max}</span>
            </div>
          </label>
        </div>
      )}

      <p className="text-xs text-slate-400 leading-relaxed text-center mt-4">
        Likelihood of treatment non-response within current regimen based on clinical and biological markers
        {confidence ? " using the specified model certainty." : "."}
      </p>
    </Card>
  );
}
