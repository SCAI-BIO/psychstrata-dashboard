import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { PgxInterpretationState } from "../../hooks/usePgx";
import { makeCompletePatient, pgxInterpretResponse } from "../../test/fixtures";
import { activeRegimenDrugs, GeneticsCard } from "./GeneticsCard";

const pgx = vi.hoisted(() => ({ state: { status: "idle" } as PgxInterpretationState }));

vi.mock("../../hooks/usePgx", () => ({
  usePgxInterpretation: () => pgx.state
}));

describe("activeRegimenDrugs", () => {
  it("keeps only dosed medications, lowercased and without the unit suffix", () => {
    const patient = makeCompletePatient();
    patient.clinical = { ...patient.clinical, sertraline_mg: 100, lithium_mg: 300, quetiapine_mg: 0 };
    expect(activeRegimenDrugs(patient)).toEqual(["sertraline", "lithium"]);
  });
});

describe("GeneticsCard", () => {
  beforeEach(() => {
    pgx.state = { status: "idle" };
  });

  it("prompts for intake when genetics is unavailable", () => {
    render(<GeneticsCard patient={makeCompletePatient({ genetics: { available: false } })} />);
    expect(screen.getByText("Pharmacogenomic Profile (CPIC)")).toBeInTheDocument();
    expect(screen.getByText(/Genomic data is marked unavailable/)).toBeInTheDocument();
  });

  it("prompts to enter diplotypes while idle", () => {
    render(<GeneticsCard patient={makeCompletePatient()} />);
    expect(screen.getByText(/Enter CYP2C19 and CYP2D6 diplotypes/)).toBeInTheDocument();
  });

  it("shows a loading state", () => {
    pgx.state = { status: "loading" };
    render(<GeneticsCard patient={makeCompletePatient()} />);
    expect(screen.getByText("Interpreting diplotypes…")).toBeInTheDocument();
  });

  it("surfaces interpretation errors", () => {
    pgx.state = { status: "error", message: "invalid allele" };
    render(<GeneticsCard patient={makeCompletePatient()} />);
    expect(screen.getByText("invalid allele")).toBeInTheDocument();
  });

  it("renders gene tiles and guidance for the active regimen", () => {
    pgx.state = { status: "ready", result: pgxInterpretResponse };
    const patient = makeCompletePatient();
    patient.clinical = { ...patient.clinical, sertraline_mg: 100, lithium_mg: 300 };
    render(<GeneticsCard patient={patient} />);

    expect(screen.getByText("CYP2C19")).toBeInTheDocument();
    expect(screen.getByText("Intermediate Metabolizer")).toBeInTheDocument();
    expect(screen.getByText("*1/*2 · Activity score 1")).toBeInTheDocument();

    expect(screen.getByText("Sertraline")).toBeInTheDocument();
    expect(screen.getByText("Start with the recommended standard dose.")).toBeInTheDocument();
    expect(screen.getByText("CPIC Level A · Final")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Guideline/ })).toHaveAttribute(
      "href",
      "https://cpicpgx.org/guidelines/"
    );
    expect(screen.getByRole("link", { name: "PMID 34298517" })).toHaveAttribute(
      "href",
      "https://pubmed.ncbi.nlm.nih.gov/34298517/"
    );

    expect(screen.getByText(/No CPIC guidance in this demo for lithium/)).toBeInTheDocument();
    expect(screen.queryByText("Amitriptyline")).not.toBeInTheDocument();
    expect(screen.getByText(pgxInterpretResponse.disclaimer)).toBeInTheDocument();
  });

  it("notes when there are no active medications", () => {
    pgx.state = { status: "ready", result: pgxInterpretResponse };
    const patient = makeCompletePatient();
    patient.clinical = { ...patient.clinical, sertraline_mg: 0, lithium_mg: 0, quetiapine_mg: 0 };
    render(<GeneticsCard patient={patient} />);
    expect(screen.getByText("No active medications on the current regimen.")).toBeInTheDocument();
  });
});
