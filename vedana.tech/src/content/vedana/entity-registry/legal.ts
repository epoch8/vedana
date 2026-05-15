/* =====================================================
   VEDANA ENTITY REGISTRY — HORECA DOMAIN
   Single source of truth for entities, attributes, relations
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

/* =====================================================
   ENTITY REGISTRY
===================================================== */

/* =====================================================
   VEDANA ENTITY REGISTRY — LEGAL DOMAIN
===================================================== */

export const Entities: Record<string, Entity> = {

  /* =====================================================
     CASE
  ===================================================== */

  case_cfi_010_2024: {
    id: "case_cfi_010_2024",
    title: "CFI 010/2024",
    entityType: "case",
    color: "var(--entity-case)",

    attributes: [
      { key: "court", value: "Court of First Instance" },
      { key: "year", value: "2024" },
      { key: "case type", value: "Commercial dispute" },
      { key: "decision date", value: "2024-10-21" }
    ],

    relations: [
      {
        label: "Claimants",
        targets: [
          {
            id: "party_acme",
            title: "ACME Holdings Ltd",
            entityType: "party",
            color: "var(--entity-party)"
          }
        ]
      },
      {
        label: "Respondents",
        targets: [
          {
            id: "party_orion",
            title: "Orion Trading LLC",
            entityType: "party",
            color: "var(--entity-party)"
          }
        ]
      },
      {
        label: "Heard by judge",
        targets: [
          {
            id: "judge_smith",
            title: "Justice Michael Smith",
            entityType: "judge",
            color: "var(--entity-judge)"
          }
        ]
      },
      {
        label: "Final ruling",
        targets: [
          {
            id: "judgment_cfi_010_2024",
            title: "Judgment — CFI 010/2024",
            entityType: "judgment",
            color: "var(--entity-judgment)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     PARTY
  ===================================================== */

  party_acme: {
    id: "party_acme",
    title: "ACME Holdings Ltd",
    entityType: "party",
    color: "var(--entity-party)",

    attributes: [
      { key: "type", value: "Company" },
      { key: "jurisdiction", value: "DIFC" }
    ],

    relations: [
      {
        label: "Claimant in case",
        targets: [
          {
            id: "case_cfi_010_2024",
            title: "CFI 010/2024",
            entityType: "case",
            color: "var(--entity-case)"
          }
        ]
      }
    ]
  },

  party_orion: {
    id: "party_orion",
    title: "Orion Trading LLC",
    entityType: "party",
    color: "var(--entity-party)",

    attributes: [
      { key: "type", value: "Company" },
      { key: "jurisdiction", value: "UAE" }
    ],

    relations: [
      {
        label: "Respondent in case",
        targets: [
          {
            id: "case_cfi_010_2024",
            title: "CFI 010/2024",
            entityType: "case",
            color: "var(--entity-case)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     JUDGE
  ===================================================== */

  judge_smith: {
    id: "judge_smith",
    title: "Justice Michael Smith",
    entityType: "judge",
    color: "var(--entity-judge)",

    attributes: [
      { key: "court", value: "DIFC Courts" },
      { key: "position", value: "Judge" }
    ],

    relations: [
      {
        label: "Presided over case",
        targets: [
          {
            id: "case_cfi_010_2024",
            title: "CFI 010/2024",
            entityType: "case",
            color: "var(--entity-case)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     JUDGMENT
  ===================================================== */

  judgment_cfi_010_2024: {
    id: "judgment_cfi_010_2024",
    title: "Judgment — CFI 010/2024",
    entityType: "judgment",
    color: "var(--entity-judgment)",

    attributes: [
      { key: "decision date", value: "2024-10-21" },
      { key: "claim value", value: "USD 4,200,000" },
      { key: "ruling", value: "Claim partially granted" }
    ],

    relations: [
      {
        label: "Judgment in case",
        targets: [
          {
            id: "case_cfi_010_2024",
            title: "CFI 010/2024",
            entityType: "case",
            color: "var(--entity-case)"
          }
        ]
      },
      {
        label: "Cites article",
        targets: [
          {
            id: "article_dp_14",
            title: "Article 14 — Processing Personal Data",
            entityType: "article",
            color: "var(--entity-article)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     LAW
  ===================================================== */

  law_data_protection_2020: {
    id: "law_data_protection_2020",
    title: "Data Protection Law 2020",
    entityType: "law",
    color: "var(--entity-law)",

    attributes: [
      { key: "law number", value: "DIFC Law No. 5 of 2020" },
      { key: "enactment date", value: "2020-07-01" }
    ],

    relations: [
      {
        label: "Contains article",
        targets: [
          {
            id: "article_dp_14",
            title: "Article 14 — Processing Personal Data",
            entityType: "article",
            color: "var(--entity-article)"
          }
        ]
      },
      {
        label: "Administered by",
        targets: [
          {
            id: "authority_registrar",
            title: "Registrar of Companies",
            entityType: "authority",
            color: "var(--entity-authority)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     ARTICLE
  ===================================================== */

 article_dp_14: {
  id: "article_dp_14",
  title: "Article 14 — Processing Personal Data",
  entityType: "article",
  color: "var(--entity-article)",

  attributes: [
    { key: "article number", value: "14" },
    { key: "title", value: "Processing Personal Data" },
    { key: "topic", value: "Data protection obligations" },
    { key: "summary", value: "Defines lawful bases and obligations for processing personal data by controllers and processors." }
  ],

  relations: [
    {
      label: "Belongs to law",
      targets: [
        {
          id: "law_data_protection_2020",
          title: "Data Protection Law 2020",
          entityType: "law",
          color: "var(--entity-law)"
        }
      ]
    },
    {
      label: "Cited in judgment",
      targets: [
        {
          id: "judgment_cfi_010_2024",
          title: "Judgment — CFI 010/2024",
          entityType: "judgment",
          color: "var(--entity-judgment)"
        }
      ]
    }
  ]
},
  /* =====================================================
     AUTHORITY
  ===================================================== */

  authority_registrar: {
    id: "authority_registrar",
    title: "Registrar of Companies",
    entityType: "authority",
    color: "var(--entity-authority)",

    attributes: [
      { key: "jurisdiction", value: "DIFC" }
    ],

    relations: [
      {
        label: "Administers law",
        targets: [
          {
            id: "law_data_protection_2020",
            title: "Data Protection Law 2020",
            entityType: "law",
            color: "var(--entity-law)"
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