/* =====================================================
   ENTITY REGISTRY — VEDANA DOMAIN
===================================================== */
export type EntityAttribute = {
  key: string
  value: string
  href?: string
  external?: boolean
}

export type EntityRelationTarget = {
  id: string
  title: string
  entityType: string
  color?: string
}

export type EntityRelation = {
  label: string
  targets: EntityRelationTarget[]
}

export type Entity = {
  id: string
  title: string
  entityType: string
  color?: string
  attributes?: EntityAttribute[]
  relations?: EntityRelation[]
}

export const Entities: Record<string, Entity> = {

  /* =====================================================
     PLATFORM
  ===================================================== */

  platform_vedana: {
    id: "platform_vedana",
    title: "Vedana Platform",
    entityType: "platform",
    color: "var(--entity-platform)",

    attributes: [
      { key: "what it does", value: "Controlled AI data layer for enterprise decision support" },

      { key: "main use case", value: "Helps teams answer operational questions when the answer depends on multiple documents, datasets, and internal policies" },

      { key: "input data", value: "PDF, DOCX, XLSX, internal databases" },

      { key: "license model", value: "Open-core platform deployable on your own infrastructure" },
      {
          key: "GitHub",
          value: "epoch8/vedana",
          href: "https://github.com/epoch8/vedana",
          external: true
        },
        {
          key: "website",
          value: "vedana.tech",
          href: "https://vedana.tech",
          external: true
        }
    ],

    relations: [
      {
        label: "Serves industries",
        targets: [
          {
            id: "industry_automotive",
            title: "Automotive",
            entityType: "industry",
            color: "var(--entity-industry)"
          },
          {
            id: "industry_legal",
            title: "Legal",
            entityType: "industry",
            color: "var(--entity-industry)"
          },
          {
            id: "industry_horeca",
            title: "HoReCa",
            entityType: "industry",
            color: "var(--entity-industry)"
          }
        ]
      },

      {
        label: "Uses concepts",
        targets: [
          {
            id: "concept_minimal_modeling",
            title: "Minimal Modeling",
            entityType: "concept",
            color: "var(--entity-concept)"
          },
          {
            id: "concept_idp",
            title: "Intelligent Document Processing",
            entityType: "concept",
            color: "var(--entity-concept)"
          }
        ]
      },

      {
        label: "Includes components",
        targets: [
          {
            id: "component_memgraph",
            title: "Memgraph",
            entityType: "component",
            color: "var(--entity-component)"
          },
          {
            id: "component_grist",
            title: "Grist",
            entityType: "component",
            color: "var(--entity-component)"
          }
        ]
      },      
      {
        label: "Built by",
        targets: [
          {
            id: "company_epoch8",
            title: "Epoch8",
            entityType: "company",
            color: "var(--entity-company)"
          }
        ]
      },
    ]
  },

  /* =====================================================
     INDUSTRIES
  ===================================================== */

industry_automotive: {
  id: "industry_automotive",
  title: "Automotive",
  entityType: "industry",
  color: "var(--entity-industry)",

  attributes: [

    {
      key: "where Vedana helps",
      value: "Product homologation, regulatory compliance, and supplier documentation validation"
    },

    {
      key: "typical question",
      value: "Does this product documentation comply with regulatory requirements for a specific market?"
    },

    {
      key: "source data",
      value: "Technical specifications, regulatory standards, supplier documents"
    },

    {
      key: "industry page",
      value: "vedana.tech/industries/automotive",
      href: "https://vedana.tech/industries/automotive",
      external: true
    }

  ]
},

industry_legal: {
  id: "industry_legal",
  title: "Legal",
  entityType: "industry",
  color: "var(--entity-industry)",

  attributes: [

    {
      key: "where Vedana helps",
      value: "Legal research, regulatory compliance checks, and contract analysis"
    },

    {
      key: "typical question",
      value: "Which regulations apply to this case and what documents define the requirements?"
    },

    {
      key: "source data",
      value: "Legal acts, contracts, regulatory documents, internal policies"
    },

    {
      key: "industry page",
      value: "vedana.tech/industries/legal",
      href: "https://vedana.tech/industries/legal",
      external: true
    }

  ]
},

industry_horeca: {
  id: "industry_horeca",
  title: "HoReCa",
  entityType: "industry",
  color: "var(--entity-industry)",

  attributes: [

    {
      key: "where Vedana helps",
      value: "Operational decision support across menus, suppliers, and compliance requirements"
    },

    {
      key: "typical question",
      value: "Which dishes have low margins based on ingredient costs and menu prices?"
    },

    {
      key: "source data",
      value: "Menus, recipes, supplier contracts, pricing data"
    },

    {
      key: "industry page",
      value: "vedana.tech/industries/horeca",
      href: "https://vedana.tech/industries/horeca",
      external: true
    }

  ]
},

  /* =====================================================
     CONCEPTS
  ===================================================== */

  concept_minimal_modeling: {
    id: "concept_minimal_modeling",
    title: "Minimal Modeling",
    entityType: "concept",
    color: "var(--entity-concept)",

    attributes: [
      { key: "type", value: "Knowledge modeling methodology" },
      { key: "idea", value: "Represent domain knowledge using anchors, attributes, and links" },
      { key: "benefit", value: "Fast schema evolution for enterprise data" }
    ]
  },

  concept_idp: {
    id: "concept_idp",
    title: "Intelligent Document Processing",
    entityType: "concept",
    color: "var(--entity-concept)",

    attributes: [
      { key: "type", value: "AI document analysis" },
      { key: "goal", value: "Extract structured knowledge from documents" },
      { key: "inputs", value: "PDF, DOCX, XLSX" }
    ]
  },

  /* =====================================================
     COMPONENTS
  ===================================================== */

  component_memgraph: {
    id: "component_memgraph",
    title: "Memgraph",
    entityType: "component",
    color: "var(--entity-component)",

    attributes: [
      { key: "type", value: "Graph database" },
      { key: "query language", value: "Cypher" },
      { key: "role", value: "Knowledge graph storage" }
    ]
  },

  component_grist: {
    id: "component_grist",
    title: "Grist",
    entityType: "component",
    color: "var(--entity-component)",

    attributes: [
      { key: "type", value: "Collaborative data platform" },
      { key: "role", value: "Structured data interface" },
      { key: "usage", value: "Domain data editing & management" }
    ]
  },
  /* =====================================================
     COMPANY
  ===================================================== */

  company_epoch8: {
    id: "company_epoch8",
    title: "Epoch8",
    entityType: "company",
    color: "var(--entity-company)",

    attributes: [
      { key: "type", value: "AI & Machine Learning Agency" },
      { key: "founded", value: "2017" },
      { key: "specialization", value: "AI systems & knowledge graphs" },
      {
        key: "website",
        value: "epoch8.co",
        href: "https://epoch8.co",
        external: true
      }
    ],

    relations: [
      {
        label: "Develops platform",
        targets: [
          {
            id: "platform_vedana",
            title: "Vedana Platform",
            entityType: "platform",
            color: "var(--entity-platform)"
          }
        ]
      }
    ]
  }


}


/* =====================================================
   DERIVED EXPORTS
===================================================== */

export const EntityList = Object.values(Entities)

export function getEntity(id: string): Entity | undefined {
  return Entities[id]
}