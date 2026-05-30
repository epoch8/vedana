export const ragComparisonTable = {
  columns: [
    {
      key: "scenario",
      title: "Business Scenario"
    },
    {
      key: "classic",
      title: "Classic RAG",
      badge: "Similarity-based",
      variant: "classic"
    },
    {
      key: "semantic",
      title: "Semantic RAG (Vedana)",
      badge: "Rule-based",
      variant: "semantic"
    }
  ],

  rows: [
    {
      id: "filters",

      scenario: {
        title: "Complete list with filters",
        example: "Which products can be sold in the EU?"
      },

      classic: {
        icon: "✕",
        text: "Returns a few similar products. May miss others that also qualify."
      },

      semantic: {
        icon: "✓",
        html: "<strong>Returns the full validated list.</strong><br/>Checks all required conditions before answering."
      }
    },

    {
      id: "sku",

      scenario: {
        title: "Exact product codes / SKUs",
        example: "Show details for SKU B49-6"
      },

      classic: {
        icon: "✕",
        text: "May surface similar-looking codes (e.g., B49-8) because they appear in related documents."
      },

      semantic: {
        icon: "✓",
        html: "<strong>Exact identifier match.</strong><br/>Only the requested SKU is evaluated and returned."
      }
    },

    {
      id: "rules",

      scenario: {
        title: "Business rules & constraints",
        example: "Can this battery be shipped by air?"
      },

      classic: {
        icon: "✕",
        text: "Finds documents mentioning air shipment, but does not verify weight or hazard class."
      },

      semantic: {
        icon: "✓",
        html: "<strong>Clear Yes/No answer.</strong><br/>Shipment rules are verified before answering."
      }
    },

    {
      id: "compatibility",

      scenario: {
        title: "Compatibility logic",
        example: "Which chargers work with Model X?"
      },

      classic: {
        icon: "✕",
        text: "Returns documents where both are mentioned. May include incompatible options."
      },

      semantic: {
        icon: "✓",
        html: "<strong>Only compatible options are returned.</strong><br/>Technical compatibility is checked first."
      }
    },

    {
      id: "compliance",

      scenario: {
        title: "Regulatory compliance",
        example: "Is this device compliant in Germany?"
      },

      classic: {
        icon: "✕",
        text: "Finds documents mentioning compliance, but does not confirm certificate validity."
      },

      semantic: {
        icon: "✓",
        html: "<strong>Compliance is verified.</strong><br/>Certification status is checked before answering."
      }
    }
  ]
};