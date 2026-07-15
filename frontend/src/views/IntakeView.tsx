import { ArrowRight, ChevronLeft, Database, FileText, Info, Microscope, Pill, type LucideIcon, ChevronDown } from "lucide-react";
import { useState, type ReactNode } from "react";
import { usePatient, type PatientApi } from "../context/PatientContext";
import type { Patient, PatientDemographics } from "../domain/patient";
import { isStepComplete } from "../domain/patient";
import type { DashboardApi } from "../hooks/useDashboard";
import { parseNumberValue } from "../lib/format";
import { PSYCH_STRATA_LOGO_URL } from "../constants";

const STEPS = ["Demographics", "Clinical Properties", "Genetics"] as const;
const STEP_PROGRESS = [0, 33, 66];

const ICD10_OPTIONS = [
  "F32.1 — Major depressive disorder, single episode, moderate",
  "F33.1 — Major depressive disorder, recurrent, moderate",
  "F33.2 — Major depressive disorder, recurrent, severe without psychotic features",
  "F33.3 — Major depressive disorder, recurrent, severe with psychotic features"
];

/**
 * Multi-step "New Patient" intake wizard. Every field reads from and writes to
 * the shared Patient model via PatientContext — demographics, genetics and
 * proteomics live on the patient, and clinical fields flow through
 * readFeature/writeFeature (routed to the model vector or captured extras).
 */
export function IntakeView({ dashboard }: { dashboard: DashboardApi }) {
  const { state } = dashboard;
  const [step, setStep] = useState(0);

  if (state.status === "loading") {
    return <CenteredMessage>Loading clinical model configuration…</CenteredMessage>;
  }
  if (state.status === "error") {
    return (
      <CenteredMessage>
        <span className="font-semibold text-slate-900">Backend unavailable.</span> {state.message}
      </CenteredMessage>
    );
  }

  return <IntakeForm dashboard={dashboard} step={step} setStep={setStep} isSubmitting={state.isSubmitting} error={state.error} />;
}

function IntakeForm({
  dashboard,
  step,
  setStep,
  isSubmitting,
  error
}: {
  dashboard: DashboardApi;
  step: number;
  setStep: (n: number) => void;
  isSubmitting: boolean;
  error: string | null;
}) {
  const patientApi = usePatient();
  const { patient } = patientApi;
  const isLast = step === STEPS.length - 1;
  const canProceed = isStepComplete(step, patient);

  return (
    <main className="min-h-screen bg-[#faf7f5] text-slate-900">
      <header className="flex items-center justify-between px-8 py-4 border-b border-slate-200/60 bg-white">
        <div className="flex items-center">
            <img src={PSYCH_STRATA_LOGO_URL} alt="" className="h-7 w-7 object-contain" />
            <span className="text-xl font-bold tracking-tight text-slate-900">TheraPath</span>
        </div>
        {dashboard.authEnabled && (
          <button
            type="button"
            onClick={dashboard.signOut}
            className="text-xs font-medium text-slate-600 border border-slate-200 hover:bg-slate-50 px-3 py-1.5 rounded-lg transition-colors"
          >
            Sign out
          </button>
        )}
      </header>

      <div className="max-w-3xl mx-auto px-6 py-10">
        <h1 className="text-3xl font-bold tracking-tight">New Patient</h1>
        <p className="text-sm text-slate-500 mt-1 mb-6">Perform a high-precision clinical data entry to calibrate risk models.</p>

        <div className="rounded-xl bg-blue-50 border border-blue-100 px-5 py-4 mb-6">
          <div className="flex items-center gap-1.5 text-slate-600 mb-1">
            <Info size={14} />
            <span className="text-[11px] font-semibold uppercase tracking-wide">Assessment Context</span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            <span className="font-semibold text-slate-600">PREVIEW MODE</span> · This interface is for demonstration only. It does
            not provide medical advice or definitive diagnosis and is not a certified medical product.
          </p>
          <div className="mt-3">
            <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
              <div className="h-full rounded-full bg-slate-900 transition-all" style={{ width: `${STEP_PROGRESS[step]}%` }} />
            </div>
            <p className="text-[11px] font-semibold text-slate-600 text-right mt-1">{STEP_PROGRESS[step]}% Complete</p>
          </div>
        </div>

        <Stepper step={step} />

        <div className="mt-6 space-y-6">
          {step === 0 && <DemographicsStep demographics={patient.demographics} onChange={patientApi.updateDemographics} />}
          {step === 1 && <ClinicalStep patient={patient} api={patientApi} />}
          {step === 2 && <GeneticsStep patient={patient} api={patientApi} />}
        </div>

        {error && <p className="text-sm text-red-600 mt-4">{error}</p>}

          <div className="flex items-center justify-between mt-8">
            <button
              type="button"
              onClick={() => setStep(Math.max(0, step - 1))}
              disabled={step === 0}
            className="flex items-center gap-1.5 text-sm font-medium text-slate-500 hover:text-slate-800 disabled:opacity-40 disabled:hover:text-slate-500 transition-colors"
          >
            <ChevronLeft size={16} />
            Back
          </button>

          <div className="flex items-center gap-3">
            {isLast ? (
              <button
                type="button"
                onClick={() => void dashboard.runPrediction("clinician")}
                disabled={!canProceed || isSubmitting}
                className="flex items-center gap-2 bg-slate-900 text-white text-sm font-semibold px-5 py-2.5 rounded-lg hover:bg-slate-700 disabled:opacity-50 transition-colors"
              >
                {isSubmitting ? "Calculating…" : "Calculate Risk"}
                {!isSubmitting && <ArrowRight size={16} />}
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setStep(step + 1)}
                disabled={!canProceed}
                className="flex items-center gap-2 bg-slate-900 text-white text-sm font-semibold px-5 py-2.5 rounded-lg hover:bg-slate-700 disabled:opacity-50 transition-colors"
              >
                Next Step
                <ArrowRight size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

// ── Steps ────────────────────────────────────────────────────────────────────

function DemographicsStep({
  demographics,
  onChange
}: {
  demographics: PatientDemographics;
  onChange: (patch: Partial<PatientDemographics>) => void;
}) {
  return (
    <WizardCard icon={FileText} title="Core Demographics">
      <div className="grid grid-cols-2 gap-5">
        <Field label="Full Name">
          <TextInput value={demographics.name ?? ""} placeholder="John Doe" onChange={(v) => onChange({ name: v })} />
        </Field>
        <Field label="Patient ID">
          <TextInput value={demographics.patientId ?? ""} placeholder="00000" onChange={(v) => onChange({ patientId: v })} />
        </Field>
        <Field label="Date of Birth">
          <TextInput type="date" value={demographics.dob ?? ""} placeholder="01.01.1970" onChange={(v) => onChange({ dob: v })} />
        </Field>
        <Field label="Gender at Birth">
          <SelectInput value={demographics.gender ?? ""} placeholder="Select Gender" options={["Male", "Female", "Other"]} onChange={(v) => onChange({ gender: v })} />
        </Field>
        <div className="col-span-2">
          <Field label="Primary Diagnosis">
            <SelectInput
              value={demographics.diagnosis ?? ""}
              placeholder="Select ICD-10 Code or Description"
              options={ICD10_OPTIONS}
              onChange={(v) => onChange({ diagnosis: v })}
            />
          </Field>
        </div>
      </div>
    </WizardCard>
  );
}

function ClinicalStep({ patient, api }: { patient: Patient; api: PatientApi }) {
  const num = (id: string, fallback = 0) => api.readFeature(id) ?? fallback;
  return (
    <>
      <WizardCard icon={FileText} title="Clinical Information">
        <div className="grid grid-cols-2 gap-5">
          <Field label="PHQ-9 Score (0-27)">
            <NumberInput value={num("phq9")} min={0} max={27} placeholder="0-27" onChange={(v) => api.writeFeature("phq9", v)} />
          </Field>
          <Field label="Episode Duration (months)">
            <NumberInput
              value={patient.episodeDurationMonths ?? 0}
              min={0}
              placeholder="Months"
              onChange={(v) => api.updateProfile({ episodeDurationMonths: v })}
            />
          </Field>
          <Field label="Adequate Treatment Failures">
            <NumberInput value={num("previous_failures")} min={0} max={10} placeholder="0-10" onChange={(v) => api.writeFeature("previous_failures", v)} />
          </Field>
          <Field label="Sleep Disturbance">
            <SelectInput
              value={patient.sleepDisturbance}
              placeholder="Select Severity"
              options={["None", "Mild", "Moderate", "Severe"]}
              onChange={(v) => api.updateProfile({ sleepDisturbance: v })}
            />
          </Field>
          <Field label="Comorbid Anxiety">
            <RadioYesNo value={num("comorbid_anxiety") === 1} onChange={(yes) => api.writeFeature("comorbid_anxiety", yes ? 1 : 0)} />
          </Field>
          <Field label="Substance Use">
            <SelectInput
              value={patient.substanceUse}
              placeholder="Select Substance Use"
              options={["None", "Alcohol", "Cannabis", "Other"]}
              onChange={(v) => api.updateProfile({ substanceUse: v })}
            />
          </Field>
        </div>
      </WizardCard>

      <WizardCard
        icon={Pill}
        title="Medication & Adherence"
        action={
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700">The patient is currently on medication</span>
            <Toggle value={patient.onMedication} onChange={(v) => api.updateProfile({ onMedication: v })} />
          </div>
        }
      >
        {patient.onMedication && (
          <div className="grid grid-cols-2 gap-5">
            <Field label="Sertraline Dose (mg/day)">
              <NumberInput value={num("sertraline_mg")} min={0} max={400} placeholder="0-400" onChange={(v) => api.writeFeature("sertraline_mg", v)} />
            </Field>
            <Field label="Quetiapine Augmentation (mg/day)">
              <NumberInput value={num("quetiapine_mg")} min={0} max={300} placeholder="0-300" onChange={(v) => api.writeFeature("quetiapine_mg", v)} />
            </Field>
            <Field label="Lithium Dose (mg/day)">
              <NumberInput value={num("lithium_mg")} min={0} max={1200} placeholder="0-1200" onChange={(v) => api.writeFeature("lithium_mg", v)} />
            </Field>
            <Field label="Early Improvement (2 weeks)">
              <RadioYesNo value={num("early_improvement") === 1} onChange={(yes) => api.writeFeature("early_improvement", yes ? 1 : 0)} />
            </Field>
            <div className="col-span-2">
              <Field label={`Treatment Adherence (%) — ${num("adherence_pct", 85)}%`}>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={num("adherence_pct", 85)}
                  onChange={(event) => api.writeFeature("adherence_pct", parseNumberValue(event.target.value))}
                  className="w-full accent-slate-900"
                />
              </Field>
            </div>
          </div>
        )}
      </WizardCard>
    </>
  );
}

function GeneticsStep({ patient, api }: { patient: Patient; api: PatientApi }) {
  return (
    <>
      <WizardCard
        icon={Database}
        title="Genomic Data"
        action={
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700">Data available</span>
            <Toggle value={patient.genetics.available} onChange={(v) => api.updateGenetics({ available: v })} />
          </div>
        }
      >
        {patient.genetics.available && (
          <div className="grid grid-cols-3 gap-4 text-sm text-slate-500">
            <span>CYP2D6 Metabolizer Status</span>
            <span>SLC6A4 Genotype</span>
            <span>BDNF Val66Met</span>
          </div>
        )} 
      </WizardCard>

      <WizardCard
        icon={Microscope}
        title="Proteomic Data"
        action={
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-700">Data available</span>
            <Toggle value={patient.proteomics.available} onChange={(v) => api.updateProteomics({ available: v })} />
          </div>
        }
      >
        {patient.proteomics.available && (
          <div className="grid grid-cols-3 gap-4 text-sm text-slate-500">
            <span>CRP (mg/L)</span>
            <span>IL-6 (pg/mL)</span>
            <span>TNF-alpha (pg/mL)</span>
          </div>
        )}
      </WizardCard>
    </>
  );
}

// ── Wizard primitives ─────────────────────────────────────────────────────────

function Stepper({ step }: { step: number }) {
  return (
    <div className="flex items-center">
      {STEPS.map((label, index) => {
        const active = index === step;
        const done = index < step;
        return (
          <div key={label} className="flex items-center flex-1 last:flex-none">
            <div className="flex items-center gap-2">
              <span
                className={`flex items-center justify-center w-6 h-6 rounded-full border text-[11px] font-semibold ${
                  active || done ? "border-slate-900 text-slate-900" : "border-slate-300 text-slate-400"
                }`}
              >
                {index + 1}
              </span>
              <span className={`text-xs font-semibold uppercase tracking-wide ${active ? "text-slate-900" : "text-slate-400"}`}>
                {label}
              </span>
            </div>
            {index < STEPS.length - 1 && <div className="flex-1 h-px bg-slate-200 mx-4" />}
          </div>
        );
      })}
    </div>
  );
}

function WizardCard({
  icon: Icon,
  title,
  action,
  children
}: {
  icon: LucideIcon;
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="bg-white rounded-xl border border-slate-200/70 shadow-sm p-6">
      <header className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-2">
          <Icon size={18} className="text-slate-700" />
          <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</span>
      {children}
    </label>
  );
}

const inputClass =
  "w-full h-11 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 " +
  "appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500";

function TextInput({ value, onChange, placeholder, readOnly, type = "text" }: {
  value: string | null;
  onChange: (v: string) => void;
  placeholder?: string;
  readOnly?: boolean;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value ?? ""}
      placeholder={placeholder}
      readOnly={readOnly}
      onChange={(e) => onChange(e.target.value)}
      className={`${inputClass} ${readOnly ? "bg-slate-100 text-slate-500" : ""}`}
    />
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type="number"
      value={value ?? 0}
      min={min}
      max={max}
      onChange={(e) => onChange(parseNumberValue(e.target.value))}
      className={inputClass}
    />
  );
}

function SelectInput({ value, onChange, options, placeholder }: {
  value: string; onChange: (v: string) => void; options: string[]; placeholder?: string;
}) {
  return (
    <div className="relative">
      <select value={value ?? undefined} onChange={(e) => onChange(e.target.value)} className={`${inputClass} pr-9`}>
        {placeholder && <option value="">{placeholder}</option>}
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
      <ChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-slate-400" />
    </div>
  );
}

function RadioYesNo({ value, onChange }: { value: boolean; onChange: (yes: boolean) => void }) {
  return (
    <div className="flex items-center gap-5 pt-1">
      {[
        { label: "Yes", val: true },
        { label: "No", val: false }
      ].map((option) => (
        <label key={option.label} className="flex items-center gap-1.5 text-sm text-slate-700 cursor-pointer">
          <input type="radio" checked={value === option.val} onChange={() => onChange(option.val)} className="accent-slate-900" />
          {option.label}
        </label>
      ))}
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      className={`relative w-11 h-6 rounded-full transition-colors ${value ? "bg-slate-900" : "bg-slate-300"}`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${
          value ? "translate-x-5" : ""
        }`}
      />
    </button>
  );
}

function CenteredMessage({ children }: { children: ReactNode }) {
  return (
    <main className="min-h-screen bg-[#faf7f5] flex items-center justify-center p-8">
      <p className="text-sm text-slate-500 text-center max-w-md">{children}</p>
    </main>
  );
}
