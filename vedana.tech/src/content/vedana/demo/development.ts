/* =====================================================
   CARD REGISTRY (Graph Nodes for Demo)
===================================================== */

export const cardRegistry = {
  /* -------------------------
     PROJECT
  -------------------------- */

  project_riverside_complex: {
    entity: "Project",
    color: "var(--entity-project)",
    name: "Riverside Residential Complex",
    location: "Hamburg, Germany",
    "project type": "Mixed-use residential development",
    "site area ha": 1.8,
    "planned floors": 12,
    "has assignment": [{id: "assignment_geotech_scope"}]
  },

  /* -------------------------
     TECHNICAL ASSIGNMENT
  -------------------------- */

  assignment_geotech_scope: {
    entity: "TechnicalAssignment",
    color: "var(--entity-assignment)",
    name: "Engineering Investigation Scope",
    "project phase": "Pre-design investigation",
    "issued by": "UrbanCore Development GmbH",

    /* Работы берутся напрямую из ТЗ */
    "required work": [
      {id: "work_drilling", "borehole depth": "25 m"},
      {id: "work_sampling"},
      {id: "work_lab_tests"},
      {id: "work_engineering_report"}
    ],
  },

  /* -------------------------
     REGULATION (VALIDATION LAYER)
  -------------------------- */

  regulation_eurocode7: {
    entity: "Regulation",
    color: "var(--entity-regulation)",
    name: "Eurocode 7 – Geotechnical Design",
    "standard code": "EN 1997-1:2004",
    domain: "Geotechnical engineering",
    defines: [
        {id: "requirement_scope_definition"},
        {id: "requirement_sampling_plan"},
        {id: "requirement_groundwater"},
        {id: "requirement_reporting"}
    ]
  },

  /* -------------------------
     WORK ITEMS (REALISTIC)
  -------------------------- */

  work_drilling: {
    entity: "WorkItem",
    color: "var(--entity-investigation)",
    name: "Borehole Drilling",
    "base rate": 120,
    currency: "EUR",
    unit: "per meter"
  },

  work_sampling: {
    entity: "WorkItem",
    color: "var(--entity-investigation)",
    name: "Soil Sampling",
    "base rate": 18,
    currency: "EUR",
    unit: "per sample"
  },

  work_lab_tests: {
    entity: "WorkItem",
    color: "var(--entity-investigation)",
    name: "Laboratory Testing",
    "base rate": 65,
    currency: "EUR",
    unit: "per test"
  },

  work_engineering_report: {
    entity: "WorkItem",
    color: "var(--entity-investigation)",
    name: "Engineering Geological Report",
    "base rate": 6500,
    currency: "EUR",
    unit: "fixed"
  },

  requirement_scope_definition: {
    entity: "Requirement",
    color: "var(--entity-requirement)",
    name: "Investigation Scope Definition",
    "defined by": [{id: "regulation_eurocode7"}],
    description: "Technical assignment must define the scope of ground investigation."
  },

  requirement_sampling_plan: {
    entity: "Requirement",
    color: "var(--entity-requirement)",
    name: "Sampling Plan",
    "defined by": [{id: "regulation_eurocode7"}],
    description: "Sampling and laboratory testing programme must be specified."
  },

  requirement_groundwater: {
    entity: "Requirement",
    color: "var(--entity-requirement)",
    name: "Groundwater Investigation",
    "defined by": [{id: "regulation_eurocode7"}],
    description: "Groundwater conditions must be investigated where relevant."
  },

  requirement_reporting: {
    entity: "Requirement",
    color: "var(--entity-requirement)",
    name: "Engineering Report",
    "defined by": [{id: "regulation_eurocode7"}],
    description: "Results must be summarized in a geotechnical report."
  }
} as const;

/* =====================================================
   TYPE DERIVATION
===================================================== */

export type CardId = keyof typeof cardRegistry;
export type CardNode = (typeof cardRegistry)[CardId];

/* =====================================================
   SCENARIOS
===================================================== */

export const scenarios = [
  {
    key: "investigation_cost",
    label: "Investigation cost estimation",

    question:
      "Estimate the cost of required investigations based on the technical assignment.",

    reasoning: [
      {
        narration:
          "User requests cost estimation based directly on the Technical Assignment."
      },

      {
        title: "Step 1 - Retrieve project and technical assignment",
        cards: [
          "project_riverside_complex",
          "assignment_geotech_scope"
        ]
      },

      {
        title: "Step 2 - Retrieve required work items from Technical Assignment",
        cards: [
          "work_drilling",
          "work_sampling",
          "work_lab_tests",
          "work_engineering_report"
        ]
      },

      {
        title: "Step 3 - Estimate missing quantities",
        logic: [
          "Borehole count not specified in TA.",
          "Apply engineering spacing assumption: 1 borehole per 0.2 hectares.",
          "Project site area = 1.8 hectares.",
          "Estimated boreholes = 9.",
          "Depth per borehole = 25 meters.",
          "Total drilling meters = 9 × 25 = 225 meters.",
          "Estimated samples per borehole = 4 → 36 samples.",
          "Lab tests per sample = 1 → 36 tests."
        ]
      },

      {
        title: "Step 4 - Apply pricing rules",
        logic: [
          "Drilling cost = 225 × €120.",
          "Sampling cost = 36 × €18.",
          "Lab testing cost = 36 × €65.",
          "Report preparation = €6,500 fixed."
        ]
      }
    ],

    answer: `
<strong>Estimated Investigation Cost (Draft)</strong>
<br/><br/>

<strong>Borehole Drilling</strong><br/>
225 m × €120 = €27,000
<br/><br/>

<strong>Soil Sampling</strong><br/>
36 samples × €18 = €648
<br/><br/>

<strong>Laboratory Testing</strong><br/>
36 tests × €65 = €2,340
<br/><br/>

<strong>Engineering Geological Report</strong><br/>
Fixed cost = €6,500
<br/><br/>

<strong>Total estimated cost:</strong><br/>
<strong>€36,488</strong>
<br/><br/>

<em>Traceability:</em><br/>
• Work scope extracted directly from Technical Assignment.<br/>
• Quantities partially inferred from project parameters.<br/>
• Pricing applied from construction price book.<br/>
• Compliance validated against Eurocode 7.
`
  },

{
  key: "ta_completeness",
  label: "Technical Assignment completeness",

  question:
    "Does the technical assignment contain all required sections for a geotechnical investigation according to Eurocode 7?",

  reasoning: [

    {
      narration:
        "User wants to verify whether the Technical Assignment includes all required investigation sections."
    },

    {
      title: "Step 1 - Retrieve technical assignment",
      cards: [
        "assignment_geotech_scope"
      ]
    },

    {
      title: "Step 2 - Retrieve regulatory requirements",
      cards: [
        "regulation_eurocode7",
        "requirement_scope_definition",
        "requirement_sampling_plan",
        "requirement_groundwater",
        "requirement_reporting"
      ]
    },

    {
      title: "Step 3 - Extract investigation scope from Technical Assignment",
      logic: [
        "Technical Assignment includes the following work:",
        "• Borehole drilling",
        "• Soil sampling",
        "• Laboratory testing",
        "• Engineering geological report"
      ]
    },

    {
      title: "Step 4 - Compare against required investigation components",
      logic: [
        "Scope definition → present.",
        "Sampling and laboratory testing → present.",
        "Engineering report → present.",
        "Groundwater investigation → not explicitly specified."
      ]
    }

  ],

  answer: `
<strong>Technical Assignment Completeness Check</strong>
<br/><br/>

<strong>Scope Definition</strong><br/>
Borehole drilling specified.<br/>
Status: <strong>Present</strong>
<br/><br/>

<strong>Sampling and Laboratory Testing</strong><br/>
Sampling and lab tests included.<br/>
Status: <strong>Present</strong>
<br/><br/>

<strong>Engineering Report</strong><br/>
Engineering geological report required.<br/>
Status: <strong>Present</strong>
<br/><br/>

<strong>Groundwater Investigation</strong><br/>
No explicit groundwater measurement programme found in the Technical Assignment.<br/>
Status: <strong>Potentially Missing</strong>
<br/><br/>

<strong>Overall Assessment</strong><br/>
The Technical Assignment covers most investigation components required for preliminary geotechnical studies.  
However, groundwater investigation is not explicitly defined and may need clarification.
<br/><br/>

<em>Traceability</em><br/>
• Technical Assignment analysed for defined investigation work.<br/>
• Required components derived from Eurocode 7 investigation principles.<br/>
• Missing elements highlighted for review.
`
}

] as const;