"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  CheckCircle2,
  ClipboardCheck,
  Database,
  FileCheck2,
  FileDown,
  Lock,
  Microscope,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  Users,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type WorkflowStep = {
  icon: React.ElementType;
  title: string;
  description: string;
  owner: string;
};

type SecurityItem = {
  icon: React.ElementType;
  label: string;
  detail: string;
};

type HygieneIssue = {
  field: string;
  issue: string;
  severity: "High" | "Medium" | "Low";
  action: string;
};

type MetadataRow = {
  field: string;
  type: string;
  sensitivity: string;
  rule: string;
  handling: string;
  approval: "Approved" | "Review" | "Restricted";
};

type ValidationMetric = {
  label: string;
  value: number;
  hint: string;
};

type StakeholderGroup = {
  label: string;
  roles: string[];
  summary: string;
  steps: string[];
  responsibilities: string[];
};

type AuditEvent = {
  time: string;
  event: string;
  detail: string;
  status: string;
};

const workflowSteps: WorkflowStep[] = [
  {
    icon: FileCheck2,
    title: "Submit Request",
    description: "User submits a healthcare data transformation request.",
    owner: "Users",
  },
  {
    icon: ScanSearch,
    title: "Scan Data",
    description: "System scans for PHI, data quality issues, and sensitive fields.",
    owner: "System",
  },
  {
    icon: ClipboardCheck,
    title: "Review & Approve",
    description: "Data Steward and Privacy Officer review scope, metadata, and privacy risk.",
    owner: "Governance",
  },
  {
    icon: Sparkles,
    title: "Generate Synthetic Data",
    description: "System creates privacy-preserving synthetic data.",
    owner: "System",
  },
  {
    icon: Microscope,
    title: "Validate Fidelity & Risk",
    description: "Users review utility, fidelity, and privacy risk metrics.",
    owner: "Shared",
  },
  {
    icon: ShieldCheck,
    title: "Controlled Release",
    description: "Approved stakeholders export synthetic outputs with audit logging.",
    owner: "Control",
  },
];

const securityItems: SecurityItem[] = [
  {
    icon: Users,
    label: "Role-based access",
    detail: "Clinical, governance, and control teams see only the surfaces relevant to their role.",
  },
  {
    icon: Activity,
    label: "Audit logging",
    detail: "Requests, reviews, synthetic runs, and release actions remain traceable.",
  },
  {
    icon: Lock,
    label: "Encrypted processing",
    detail: "Protected handling keeps source and synthetic workflows inside a controlled environment.",
  },
  {
    icon: BadgeCheck,
    label: "Controlled release",
    detail: "Synthetic outputs are shared only after review, validation, and release approval.",
  },
];

const hygieneIssues: HygieneIssue[] = [
  {
    field: "arrival_mode",
    issue: "Missing values",
    severity: "High",
    action: "Resolve null encounters or mark as unknown before generation.",
  },
  {
    field: "ctas_level",
    issue: "Inconsistent coding",
    severity: "Medium",
    action: "Normalize triage labels to the approved CTAS scale.",
  },
  {
    field: "encounter_id",
    issue: "Duplicate records",
    severity: "High",
    action: "Deduplicate repeated visits before metadata approval.",
  },
  {
    field: "visit_date",
    issue: "Invalid dates",
    severity: "Medium",
    action: "Correct malformed values or exclude invalid records.",
  },
  {
    field: "chief_complaint",
    issue: "Free-text variation",
    severity: "Low",
    action: "Map equivalent terms into standard complaint groups.",
  },
];

const metadataRows: MetadataRow[] = [
  {
    field: "encounter_id",
    type: "Identifier",
    sensitivity: "Restricted",
    rule: "Generate surrogate token",
    handling: "Blocked from export and regenerated on output",
    approval: "Approved",
  },
  {
    field: "visit_date",
    type: "Date",
    sensitivity: "Moderate",
    rule: "Sample with controlled date jitter",
    handling: "Day-level shift applied before release",
    approval: "Approved",
  },
  {
    field: "postal_prefix",
    type: "Geography",
    sensitivity: "Moderate",
    rule: "Retain prefix only",
    handling: "Coarse region preserved, full geography suppressed",
    approval: "Review",
  },
  {
    field: "wait_time_min",
    type: "Numeric",
    sensitivity: "Low",
    rule: "Sample plus bounded noise",
    handling: "Outlier strategy controls how tail values are preserved or smoothed",
    approval: "Approved",
  },
  {
    field: "chief_complaint",
    type: "Categorical",
    sensitivity: "Moderate",
    rule: "Normalize and sample by grouped distribution",
    handling: "Free text reduced to governed complaint classes",
    approval: "Restricted",
  },
];

const validationMetrics: ValidationMetric[] = [
  {
    label: "Schema match",
    value: 98,
    hint: "Field structure remains aligned with the approved source schema.",
  },
  {
    label: "Distribution alignment",
    value: 91,
    hint: "Key operational distributions remain close to source behavior.",
  },
  {
    label: "Privacy risk score",
    value: 93,
    hint: "Low overlap and controlled identifier handling reduce disclosure risk.",
  },
  {
    label: "Statistical fidelity",
    value: 88,
    hint: "Summary statistics remain suitable for exploratory analysis and prototyping.",
  },
  {
    label: "Downstream utility",
    value: 90,
    hint: "Output is appropriate for dashboards, testing, and sandbox analytics workflows.",
  },
];

const stakeholderGroups: StakeholderGroup[] = [
  {
    label: "Users",
    roles: ["Clinician / Clinical Lead", "Data Analyst"],
    summary: "Users initiate requests and confirm that the synthetic output is useful for operational analysis, prototyping, and review.",
    steps: ["Submit Request", "Validate Fidelity & Risk"],
    responsibilities: [
      "Define the request and intended use case.",
      "Review output quality, utility, and analytic fit.",
      "Confirm the synthetic package meets business or clinical review needs.",
    ],
  },
  {
    label: "Governance",
    roles: ["Data Steward", "Privacy Officer"],
    summary: "Governance reviews data scope, metadata handling, and privacy posture before synthetic outputs move forward.",
    steps: ["Scan Data", "Review & Approve", "Validate Fidelity & Risk"],
    responsibilities: [
      "Review PHI detection and sensitive-field handling.",
      "Approve metadata scope and privacy controls.",
      "Assess fidelity and disclosure risk before release.",
    ],
  },
  {
    label: "Control",
    roles: ["IT / Security Admin", "Executive Approver"],
    summary: "Control roles maintain the platform boundary and authorize the final release of approved synthetic outputs.",
    steps: ["Controlled Release"],
    responsibilities: [
      "Maintain system controls and audit visibility.",
      "Ensure approved outputs are exported through controlled channels.",
      "Authorize final release for downstream sharing.",
    ],
  },
];

const auditEvents: AuditEvent[] = [
  {
    time: "09:14",
    event: "Request submitted",
    detail: "Emergency department transformation request opened for analytics sandbox use.",
    status: "Logged",
  },
  {
    time: "09:21",
    event: "Data scan completed",
    detail: "PHI and source-data issues were detected and summarized for review.",
    status: "Scanned",
  },
  {
    time: "09:24",
    event: "Review approved",
    detail: "Scope, metadata controls, and privacy handling were approved by governance.",
    status: "Approved",
  },
  {
    time: "09:27",
    event: "Synthetic dataset generated",
    detail: "2,000 synthetic records generated under the approved configuration.",
    status: "Completed",
  },
  {
    time: "09:31",
    event: "Validation completed",
    detail: "Fidelity and privacy risk metrics passed the release threshold.",
    status: "Verified",
  },
  {
    time: "09:36",
    event: "Release approved",
    detail: "Synthetic package approved for governed export to the analytics sandbox.",
    status: "Released",
  },
];

const trustBadges = [
  "Role-Based Access",
  "PHI Scan",
  "Governance Review",
  "Controlled Export",
];

const containerVariants = {
  hidden: { opacity: 0, y: 10 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#0F4F95]/70">
        {eyebrow}
      </p>
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">
          {title}
        </h2>
        <p className="max-w-3xl text-sm leading-6 text-slate-600 sm:text-base">
          {description}
        </p>
      </div>
    </div>
  );
}

function SeverityBadge({ severity }: { severity: HygieneIssue["severity"] }) {
  const styles =
    severity === "High"
      ? "bg-rose-50 text-rose-700 border-rose-200"
      : severity === "Medium"
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-sky-50 text-sky-700 border-sky-200";

  return (
    <Badge variant="outline" className={styles}>
      {severity}
    </Badge>
  );
}

function ApprovalBadge({ status }: { status: MetadataRow["approval"] }) {
  const styles =
    status === "Approved"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : status === "Review"
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-slate-100 text-slate-700 border-slate-200";

  return (
    <Badge variant="outline" className={styles}>
      {status}
    </Badge>
  );
}

function WorkflowCard({ step, index }: { step: WorkflowStep; index: number }) {
  const Icon = step.icon;

  return (
    <motion.div variants={containerVariants}>
      <Card className="h-full rounded-2xl border-slate-200/80 bg-white shadow-sm shadow-slate-200/60 transition-shadow hover:shadow-md hover:shadow-slate-200/70">
        <CardHeader className="space-y-4 pb-4">
          <div className="flex items-center justify-between">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0F4F95]/8 text-[#0F4F95]">
              <Icon className="h-5 w-5" />
            </div>
            <Badge
              variant="outline"
              className="rounded-full border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500"
            >
              {step.owner}
            </Badge>
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-slate-400">0{index + 1}</span>
              <CardTitle className="text-lg text-slate-950">{step.title}</CardTitle>
            </div>
            <CardDescription className="text-sm leading-6 text-slate-600">
              {step.description}
            </CardDescription>
          </div>
        </CardHeader>
      </Card>
    </motion.div>
  );
}

function SecurityCard({ item }: { item: SecurityItem }) {
  const Icon = item.icon;

  return (
    <Card className="rounded-2xl border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
      <CardContent className="flex h-full flex-col gap-3 p-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#1CD8D3]/10 text-[#0A4F93]">
          <Icon className="h-4.5 w-4.5" />
        </div>
        <div className="space-y-1.5">
          <p className="text-sm font-semibold text-slate-950">{item.label}</p>
          <p className="text-sm leading-6 text-slate-600">{item.detail}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function MetricCard({ metric }: { metric: ValidationMetric }) {
  return (
    <Card className="rounded-2xl border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
      <CardHeader className="space-y-2 pb-3">
        <CardDescription className="text-sm font-medium text-slate-500">
          {metric.label}
        </CardDescription>
        <CardTitle className="text-3xl font-semibold text-slate-950">
          {metric.value}%
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress
          value={metric.value}
          className="h-2 bg-slate-100"
          indicatorClassName="bg-gradient-to-r from-[#0F4F95] to-[#1CD8D3]"
        />
        <p className="text-sm leading-6 text-slate-600">{metric.hint}</p>
      </CardContent>
    </Card>
  );
}

function StakeholderCard({ group }: { group: StakeholderGroup }) {
  return (
    <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
      <CardHeader className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardDescription className="text-xs font-semibold uppercase tracking-[0.2em] text-[#0F4F95]/70">
              {group.label}
            </CardDescription>
            <CardTitle className="mt-2 text-xl text-slate-950">
              {group.roles.join(" · ")}
            </CardTitle>
          </div>
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0F4F95]/8 text-[#0F4F95]">
            <Users className="h-5 w-5" />
          </div>
        </div>
        <CardDescription className="text-sm leading-6 text-slate-600">
          {group.summary}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            Most involved in
          </p>
          <div className="flex flex-wrap gap-2">
            {group.steps.map((step) => (
              <Badge
                key={step}
                variant="secondary"
                className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-slate-700"
              >
                {step}
              </Badge>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
            What this group does
          </p>
          <ul className="space-y-2 text-sm leading-6 text-slate-600">
            {group.responsibilities.map((item) => (
              <li key={item} className="flex gap-2">
                <span className="mt-2 h-1.5 w-1.5 rounded-full bg-[#1CD8D3]" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

export default function SouthlakeSyntheticDashboard() {
  return (
    <div className="min-h-screen bg-[#F6FBFD] text-slate-950">
      <div className="mx-auto flex w-full max-w-[1440px] flex-col gap-10 px-6 py-8 sm:px-8 lg:px-10 lg:py-10">
        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="grid gap-6 xl:grid-cols-[minmax(0,1.6fr)_380px]"
        >
          <Card className="overflow-hidden rounded-[28px] border-[#0F4F95]/10 bg-white shadow-lg shadow-[#0F4F95]/5">
            <CardContent className="space-y-8 p-7 sm:p-9">
              <div className="space-y-5">
                <Badge
                  variant="outline"
                  className="rounded-full border-[#1CD8D3]/30 bg-[#1CD8D3]/8 px-4 py-1 text-[#0A4F93]"
                >
                  Southlake Health synthetic data workspace
                </Badge>
                <div className="space-y-4">
                  <h1 className="max-w-4xl text-4xl font-semibold tracking-tight text-slate-950 sm:text-5xl">
                    Synthetic healthcare data with clear review, validation, and controlled release.
                  </h1>
                  <p className="max-w-3xl text-base leading-7 text-slate-600">
                    Submit a request, scan source data, review privacy handling, generate synthetic output, and release approved datasets through a simple hospital-grade workflow.
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row">
                <Button
                  size="lg"
                  className="rounded-2xl bg-[#0F4F95] text-white shadow-sm shadow-[#0F4F95]/20 hover:bg-[#0A4F93]"
                >
                  Submit Request
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  className="rounded-2xl border-[#0F4F95]/15 bg-white text-[#0F4F95] hover:bg-[#0F4F95]/5"
                >
                  View Validation Report
                </Button>
              </div>

              <div className="flex flex-wrap gap-2">
                {trustBadges.map((badge) => (
                  <Badge
                    key={badge}
                    variant="secondary"
                    className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-slate-700"
                  >
                    {badge}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-[28px] border-[#0F4F95]/10 bg-white shadow-lg shadow-[#0F4F95]/5">
            <CardHeader className="space-y-2 pb-4">
              <CardDescription className="text-xs font-semibold uppercase tracking-[0.2em] text-[#0F4F95]/70">
                Current request
              </CardDescription>
              <CardTitle className="text-2xl text-slate-950">
                Workflow at a glance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="space-y-3">
                {workflowSteps.map((step, index) => (
                  <div
                    key={step.title}
                    className="flex items-center gap-3 rounded-2xl border border-slate-200/80 bg-slate-50 px-4 py-3"
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#0F4F95] text-xs font-semibold text-white">
                      {index + 1}
                    </div>
                    <div className="flex flex-1 items-center justify-between gap-3">
                      <span className="text-sm font-medium text-slate-800">{step.title}</span>
                      {index < 5 ? (
                        <ArrowRight className="h-4 w-4 text-slate-400" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-[#0F4F95]/5 p-4">
                  <p className="text-sm font-medium text-slate-500">Request status</p>
                  <p className="mt-1 text-2xl font-semibold text-slate-950">In review</p>
                </div>
                <div className="rounded-2xl bg-[#1CD8D3]/10 p-4">
                  <p className="text-sm font-medium text-slate-500">Release state</p>
                  <p className="mt-1 text-2xl font-semibold text-slate-950">Controlled</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.section>

        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
          <SectionHeading
            eyebrow="Workflow"
            title="A simple six-step process from request to controlled release"
            description="The main workflow stays easy to read while still showing privacy review, metadata governance, and safe release."
          />
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
            {workflowSteps.map((step, index) => (
              <WorkflowCard key={step.title} step={step} index={index} />
            ))}
          </div>
        </motion.section>

        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
          <SectionHeading
            eyebrow="Stakeholders"
            title="Three groups participate in the workflow"
            description="Role boundaries stay visible, but the homepage keeps them lightweight. Each group focuses on a different part of the process."
          />
          <div className="grid gap-4 xl:grid-cols-3">
            {stakeholderGroups.map((group) => (
              <StakeholderCard key={group.label} group={group} />
            ))}
          </div>
        </motion.section>

        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
          <SectionHeading
            eyebrow="Security and compliance"
            title="Trust controls stay visible without dominating the page"
            description="Privacy, auditability, and controlled release remain explicit so the product feels safe, credible, and hospital-ready."
          />
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {securityItems.map((item) => (
              <SecurityCard key={item.label} item={item} />
            ))}
          </div>
        </motion.section>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <motion.section
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="space-y-6"
          >
            <SectionHeading
              eyebrow="Source quality"
              title="Data hygiene review"
              description="Source issues are surfaced before synthetic generation so teams can correct quality problems, align coding, and reduce avoidable risk."
            />
            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <CardTitle className="text-lg text-slate-950">Current findings</CardTitle>
                  <CardDescription className="text-sm text-slate-600">
                    Issues are prioritized by severity and paired with a clear next action.
                  </CardDescription>
                </div>
                <Badge className="w-fit rounded-full bg-[#0F4F95]/10 text-[#0A4F93] hover:bg-[#0F4F95]/10">
                  5 issues identified
                </Badge>
              </CardHeader>
              <CardContent>
                <div className="overflow-hidden rounded-2xl border border-slate-200/80">
                  <Table>
                    <TableHeader className="bg-slate-50">
                      <TableRow className="hover:bg-slate-50">
                        <TableHead>Field</TableHead>
                        <TableHead>Issue</TableHead>
                        <TableHead>Severity</TableHead>
                        <TableHead>Recommended action</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {hygieneIssues.map((row) => (
                        <TableRow key={`${row.field}-${row.issue}`} className="align-top">
                          <TableCell className="font-medium text-slate-900">{row.field}</TableCell>
                          <TableCell className="text-slate-700">{row.issue}</TableCell>
                          <TableCell>
                            <SeverityBadge severity={row.severity} />
                          </TableCell>
                          <TableCell className="max-w-xs text-sm leading-6 text-slate-600">
                            {row.action}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </motion.section>

          <motion.section
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="space-y-6"
          >
            <SectionHeading
              eyebrow="Metadata"
              title="Review and approval"
              description="Metadata stays visible and editable before generation so governance can approve the field handling plan without adding unnecessary bureaucracy."
            />
            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader>
                <CardTitle className="text-lg text-slate-950">Metadata and privacy controls</CardTitle>
                <CardDescription className="text-sm text-slate-600">
                  Scope, sensitivity, generation rules, and release handling remain visible in one place.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-hidden rounded-2xl border border-slate-200/80">
                  <Table>
                    <TableHeader className="bg-slate-50">
                      <TableRow className="hover:bg-slate-50">
                        <TableHead>Field</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Sensitivity</TableHead>
                        <TableHead>Generation rule</TableHead>
                        <TableHead>Privacy handling</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {metadataRows.map((row) => (
                        <TableRow key={row.field} className="align-top">
                          <TableCell className="font-medium text-slate-900">{row.field}</TableCell>
                          <TableCell className="text-slate-700">{row.type}</TableCell>
                          <TableCell className="text-slate-700">{row.sensitivity}</TableCell>
                          <TableCell className="max-w-[220px] text-sm leading-6 text-slate-600">
                            {row.rule}
                          </TableCell>
                          <TableCell className="max-w-[260px] text-sm leading-6 text-slate-600">
                            {row.handling}
                          </TableCell>
                          <TableCell>
                            <ApprovalBadge status={row.approval} />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </motion.section>
        </div>

        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
          <SectionHeading
            eyebrow="Validation"
            title="Validate fidelity and privacy risk"
            description="The validation layer stays concise: enough to show utility, structure, and privacy risk without turning the page into a technical report."
          />
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            {validationMetrics.map((metric) => (
              <MetricCard key={metric.label} metric={metric} />
            ))}
          </div>
        </motion.section>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,0.9fr)]">
          <motion.section
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="space-y-6"
          >
            <SectionHeading
              eyebrow="Use cases"
              title="Approved synthetic outputs support practical hospital use"
              description="The platform is positioned for safe analytics, prototyping, and training workflows without moving raw patient-level data into low-trust environments."
            />
            <div className="grid gap-4 md:grid-cols-2">
              {[
                {
                  title: "Dashboard and service-line prototyping",
                  body: "Teams can test KPIs, layouts, and operational views before requesting access to sensitive data.",
                },
                {
                  title: "Analytics and feature engineering",
                  body: "Analysts can prototype notebooks, cohort logic, and pipelines on realistic but privacy-preserving data.",
                },
                {
                  title: "Training and workflow rehearsal",
                  body: "Clinical and operational teams can review scenarios, handoffs, and reporting flows without exposing real patient records.",
                },
                {
                  title: "Sandbox and integration testing",
                  body: "Approved synthetic outputs can be shared into controlled downstream environments for vendor or platform testing.",
                },
              ].map((card) => (
                <Card
                  key={card.title}
                  className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60"
                >
                  <CardContent className="p-6">
                    <p className="font-semibold text-slate-950">{card.title}</p>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{card.body}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </motion.section>

          <motion.section
            initial="hidden"
            animate="visible"
            variants={containerVariants}
            className="space-y-6"
          >
            <SectionHeading
              eyebrow="Audit"
              title="Recent activity"
              description="A compact operational record makes review and release activity easy to trace."
            />
            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader>
                <CardTitle className="text-lg text-slate-950">Audit log</CardTitle>
                <CardDescription className="text-sm text-slate-600">
                  Key platform events from intake through governed release.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {auditEvents.map((event, index) => (
                    <div key={`${event.time}-${event.event}`} className="relative pl-10">
                      {index < auditEvents.length - 1 ? (
                        <span className="absolute left-[14px] top-9 h-12 w-px bg-slate-200" />
                      ) : null}
                      <span className="absolute left-0 top-1 flex h-7 w-7 items-center justify-center rounded-full bg-[#0F4F95]/10 text-[#0F4F95]">
                        <Activity className="h-3.5 w-3.5" />
                      </span>
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1">
                          <p className="text-sm font-semibold text-slate-950">{event.event}</p>
                          <p className="text-sm leading-6 text-slate-600">{event.detail}</p>
                        </div>
                        <div className="flex flex-col items-end gap-2">
                          <span className="text-xs font-medium uppercase tracking-wide text-slate-400">
                            {event.time}
                          </span>
                          <Badge
                            variant="outline"
                            className="rounded-full border-slate-200 bg-slate-50 text-slate-700"
                          >
                            {event.status}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.section>
        </div>

        <motion.section
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="space-y-6"
        >
            <SectionHeading
              eyebrow="Release controls"
              title="Controlled export and reporting"
              description="Approved stakeholders can release synthetic outputs through governed, logged actions without surfacing too much internal process detail."
            />
          <div className="grid gap-4 lg:grid-cols-3">
            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0F4F95]/10 text-[#0F4F95]">
                  <Database className="h-5 w-5" />
                </div>
                <CardTitle className="pt-3 text-lg">Export synthetic dataset</CardTitle>
                <CardDescription className="text-sm leading-6 text-slate-600">
                  Release the approved synthetic extract to a designated sandbox or analytics workspace.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-2xl border border-slate-200/80 bg-slate-50 p-4 text-sm text-slate-600">
                  Logged, permission-scoped, and restricted to approved synthetic artifacts.
                </div>
                <Button className="w-full rounded-2xl bg-[#0F4F95] text-white hover:bg-[#0A4F93]">
                  Export dataset
                </Button>
              </CardContent>
            </Card>

            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#1CD8D3]/12 text-[#0A4F93]">
                  <FileDown className="h-5 w-5" />
                </div>
                <CardTitle className="pt-3 text-lg">Download validation report</CardTitle>
                <CardDescription className="text-sm leading-6 text-slate-600">
                  Provide a concise record of schema, fidelity, privacy, and release readiness.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-2xl border border-slate-200/80 bg-slate-50 p-4 text-sm text-slate-600">
                  Includes methodology, metrics, and approval state for the current synthetic package.
                </div>
                <Button
                  variant="outline"
                  className="w-full rounded-2xl border-[#0F4F95]/15 text-[#0F4F95] hover:bg-[#0F4F95]/5"
                >
                  Download report
                </Button>
              </CardContent>
            </Card>

            <Card className="rounded-[24px] border-slate-200/80 bg-white shadow-sm shadow-slate-200/60">
              <CardHeader>
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#0F4F95]/10 text-[#0F4F95]">
                  <FileCheck2 className="h-5 w-5" />
                </div>
                <CardTitle className="pt-3 text-lg">Share release summary</CardTitle>
                <CardDescription className="text-sm leading-6 text-slate-600">
                  Send a release summary to compliance, analytics leadership, or platform operations.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-2xl border border-slate-200/80 bg-slate-50 p-4 text-sm text-slate-600">
                  Distribution is governed, tracked, and limited to approved recipients.
                </div>
                <Button
                  variant="outline"
                  className="w-full rounded-2xl border-[#0F4F95]/15 text-[#0F4F95] hover:bg-[#0F4F95]/5"
                >
                  Share summary
                </Button>
              </CardContent>
            </Card>
          </div>
        </motion.section>
      </div>
    </div>
  );
}
