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

export const Entities: Record<string, Entity> = {

  /* =====================================================
     BRAND
  ===================================================== */

  brand_nordic_harbor: {
    id: "brand_nordic_harbor",
    title: "Nordic Harbor",
    entityType: "brand",
    color: "var(--entity-brand)",

    attributes: [
      { key: "concept type", value: "Casual seafood restaurant" },
      { key: "opening year", value: "2014" },
      {
        key: "website",
        value: "nordic-harbor.de",
        href: "https://nordic-harbor.de",
        external: true
      }
    ],

    relations: [
      {
        label: "Operates in locations",
        targets: [
          {
            id: "loc_berlin",
            title: "Berlin",
            entityType: "location",
            color: "var(--entity-location)"
          },
          {
            id: "loc_munich",
            title: "Munich",
            entityType: "location",
            color: "var(--entity-location)"
          },
          {
            id: "loc_hamburg",
            title: "Hamburg",
            entityType: "location",
            color: "var(--entity-location)"
          }
        ]
      }
    ]
  },

  /* =====================================================
     LOCATIONS
  ===================================================== */

loc_berlin: {
  id: "loc_berlin",
  title: "Nordic Harbor — Berlin Mitte",
  entityType: "location",
  color: "var(--entity-location)",

  attributes: [
    { key: "city", value: "Berlin" },
    { key: "country", value: "Germany" },
    { key: "address", value: "Torstraße 112, 10119 Berlin" },
    { key: "opening date", value: "May 2016" },
    { key: "seating capacity", value: "84" },
    { key: "kitchen type", value: "Full kitchen" },
    { key: "opening hours", value: "12:00 – 23:00" }
  ],

  relations: [
    {
      label: "Serves",
      targets: [
        {
          id: "menu_food",
          title: "Food Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        },
       {
          id: "menu_wine",
          title: "Wine & Beverage Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        }
      ]
    },
    {
      label: "Certified under",
      targets: [
        {
          id: "comp_haccp",
          title: "HACCP Certification",
          entityType: "compliance",
          color: "var(--entity-compliance)"
        }
      ]
    }
  ]
},
loc_munich: {
  id: "loc_munich",
  title: "Nordic Harbor — Munich Schwabing",
  entityType: "location",
  color: "var(--entity-location)",

  attributes: [
    { key: "city", value: "Munich" },
    { key: "country", value: "Germany" },
    { key: "address", value: "Leopoldstraße 87, 80802 Munich" },
    { key: "opening date", value: "Aug 2018" },
    { key: "seating capacity", value: "72" },
    { key: "kitchen type", value: "Full kitchen" },
    { key: "opening hours", value: "12:00 – 22:30" }
  ],

  relations: [
    {
      label: "Serves",
      targets: [
        {
          id: "menu_food",
          title: "Food Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        },
       {
          id: "menu_wine",
          title: "Wine & Beverage Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        }
      ]
    }
  ]
},
loc_hamburg: {
  id: "loc_hamburg",
  title: "Nordic Harbor — Hamburg HafenCity",
  entityType: "location",
  color: "var(--entity-location)",

  attributes: [
    { key: "city", value: "Hamburg" },
    { key: "country", value: "Germany" },
    { key: "address", value: "Am Sandtorkai 48, 20457 Hamburg" },
    { key: "opening date", value: "Mar 2020" },
    { key: "seating capacity", value: "96" },
    { key: "kitchen type", value: "Seafood grill" },
    { key: "opening hours", value: "12:00 – 23:30" }
  ],

  relations: [
    {
      label: "Serves",
      targets: [
        {
          id: "menu_food",
          title: "Food Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        },
       {
          id: "menu_wine",
          title: "Wine & Beverage Menu",
          entityType: "menu",
          color: "var(--entity-menu)"
        }
      ]
    }
  ]
},

/* =====================================================
   MENU
===================================================== */

menu_food: {
  id: "menu_food",
  title: "Food Menu",
  entityType: "menu",
  color: "var(--entity-menu)",

  attributes: [
    { key: "menu type", value: "Food" },
    { key: "season", value: "Spring 2026" },
    { key: "dishes count", value: "24" }
  ],

  relations: [
    {
      label: "Includes dishes",
      targets: [
        {
          id: "dish_salmon",
          title: "Grilled Atlantic Salmon",
          entityType: "dish",
          color: "var(--entity-dish)"
        }
      ]
    }
  ]
},
/* =====================================================
   MENU — WINE / BEVERAGES
===================================================== */

menu_wine: {
  id: "menu_wine",
  title: "Wine & Beverage Menu",
  entityType: "menu",
  color: "var(--entity-menu)",

  attributes: [
    { key: "menu type", value: "Wine & Beverages" },
    { key: "wines count", value: "38" },
    { key: "cocktails count", value: "12" }
  ],

  relations: [
    {
      label: "Includes wines",
      targets: [
        {
          id: "wine_pinot",
          title: "Pinot Noir",
          entityType: "wine",
          color: "var(--entity-wine)"
        }
      ]
    }
  ]
},

/* =====================================================
   DISH
===================================================== */

dish_salmon: {
  id: "dish_salmon",
  title: "Grilled Atlantic Salmon",
  entityType: "dish",
  color: "var(--entity-dish)",

  attributes: [
    { key: "category", value: "Main course" },
    { key: "cuisine", value: "Nordic seafood" },
    { key: "preparation time", value: "18 min" },
    { key: "serving size", value: "320 g" },
    { key: "menu price", value: "€24" }
  ],

  relations: [
    {
      label: "Contains ingredients",
      targets: [
        {
          id: "ingredient_salmon",
          title: "Atlantic Salmon",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        },
        {
          id: "ingredient_lemon",
          title: "Fresh Lemon",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        },
        {
          id: "ingredient_butter",
          title: "Unsalted Butter",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        },
        {
          id: "ingredient_olive_oil",
          title: "Olive Oil",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        },
        {
          id: "ingredient_sea_salt",
          title: "Sea Salt",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        },
        {
          id: "ingredient_dill",
          title: "Fresh Dill",
          entityType: "ingredient",
          color: "var(--entity-ingredient)"
        }
      ]
    }
  ]
},

  /* =====================================================
     INGREDIENT
  ===================================================== */

ingredient_salmon: {
  id: "ingredient_salmon",
  title: "Atlantic Salmon",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Fish" },
    { key: "is allergen", value: "Yes" },
    { key: "shelf life", value: "3 days" },
    { key: "storage", value: "2–4°C refrigerated" },
    { key: "cost", value: "€8.40 / kg" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

ingredient_lemon: {
  id: "ingredient_lemon",
  title: "Fresh Lemon",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Fruit" },
    { key: "is allergen", value: "No" },
    { key: "shelf life", value: "10 days" },
    { key: "storage", value: "5–8°C refrigerated" },
    { key: "cost", value: "€1.20 / kg" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

ingredient_butter: {
  id: "ingredient_butter",
  title: "Unsalted Butter",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Dairy" },
    { key: "is allergen", value: "Yes (milk)" },
    { key: "shelf life", value: "30 days" },
    { key: "storage", value: "2–6°C refrigerated" },
    { key: "cost", value: "€6.50 / kg" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

ingredient_olive_oil: {
  id: "ingredient_olive_oil",
  title: "Extra Virgin Olive Oil",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Oil" },
    { key: "is allergen", value: "No" },
    { key: "shelf life", value: "12 months" },
    { key: "storage", value: "Room temperature" },
    { key: "cost", value: "€9.80 / L" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

ingredient_sea_salt: {
  id: "ingredient_sea_salt",
  title: "Sea Salt",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Seasoning" },
    { key: "is allergen", value: "No" },
    { key: "shelf life", value: "Indefinite" },
    { key: "storage", value: "Dry storage" },
    { key: "cost", value: "€0.80 / kg" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

ingredient_dill: {
  id: "ingredient_dill",
  title: "Fresh Dill",
  entityType: "ingredient",
  color: "var(--entity-ingredient)",

  attributes: [
    { key: "category", value: "Herb" },
    { key: "is allergen", value: "No" },
    { key: "shelf life", value: "5 days" },
    { key: "storage", value: "2–4°C refrigerated" },
    { key: "cost", value: "€14.00 / kg" }
  ],

  relations: [
    {
      label: "Supplied by",
      targets: [
        {
          id: "supplier_nordic",
          title: "Nordic Seafood Supplier",
          entityType: "supplier",
          color: "var(--entity-supplier)"
        }
      ]
    }
  ]
},

  /* =====================================================
     WINE
  ===================================================== */

  wine_pinot: {
    id: "wine_pinot",
    title: "Pinot Noir",
    entityType: "wine",
    color: "var(--entity-wine)",

    attributes: [
      { key: "type", value: "Red wine" },
      { key: "origin", value: "Burgundy, France" },
      { key: "serving temperature", value: "16–18°C" },
      { key: "body", value: "Medium" },
      { key: "acidity", value: "High" }
    ]
  },

  /* =====================================================
     COMPLIANCE
  ===================================================== */

  comp_haccp: {
    id: "comp_haccp",
    title: "HACCP Certification 2023–2026",
    entityType: "compliance",
    color: "var(--entity-compliance)",

    attributes: [
      { key: "standard number", value: "HACCP-DE-2023-4581" },
      { key: "scope", value: "Kitchen & cold storage operations" },
      { key: "issuing body", value: "German Food Safety Authority" },
      { key: "valid until", value: "Dec 2026" }
    ]
  },

/* =====================================================
   SUPPLIER
===================================================== */

supplier_nordic: {
  id: "supplier_nordic",
  title: "Nordic Seafood Supplier",
  entityType: "supplier",
  color: "var(--entity-supplier)",

  attributes: [
    { key: "country", value: "Norway" },
    { key: "delivery frequency", value: "Daily" },
    { key: "lead time", value: "24 hours" }
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