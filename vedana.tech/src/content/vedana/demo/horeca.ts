// src/content/vedana/demo/horeca.ts

/* =====================================================
   CARD REGISTRY (Graph Nodes for Demo)
===================================================== */

export const cardRegistry = {
  dish_chefs_special_pasta: {
    entity: "Dish",
    color: "var(--entity-dish)",
    name: "Chef’s Special Pasta",
    category: "Main Course",
    prep_style: "Pan-seared + reduction",
    contains: [
      { id: "ingredient_atlantic_salmon", amount: "100g" },
      { id: "ingredient_cream", amount: "50g" },
      { id: "ingredient_pasta", amount: "150g" }
    ]
  },

  dish_grilled_salmon: {
    entity: "Dish",
    color: "var(--entity-dish)",
    name: "Grilled Salmon",
    category: "Main Course",
    cost_price: 14,
    menu_price: 22
  },

  dish_truffle_pasta: {
    entity: "Dish",
    color: "var(--entity-dish)",
    name: "Truffle Pasta",
    category: "Main Course",
    cost_price: 11,
    menu_price: 18
  },

  ingredient_atlantic_salmon: {
    entity: "Ingredient",
    color: "var(--entity-ingredient)",
    name: "Atlantic Salmon",
    protein: 20,
    fat: 13,
    carbs: 0
  },

  ingredient_cream: {
    entity: "Ingredient",
    color: "var(--entity-ingredient)",
    name: "Cream",
    protein: 2,
    fat: 35,
    carbs: 3
  },

  ingredient_pasta: {
    entity: "Ingredient",
    color: "var(--entity-ingredient)",
    name: "Pasta",
    protein: 5,
    fat: 1,
    carbs: 25
  },

  wine_pinot_noir: {
    entity: "Wine",
    color: "var(--entity-wine)",
    name: "Pinot Noir",
    type: "Red",
    body: "Medium",
    acidity: "High",
    origin: "Burgundy",
    flavor_profile: ["Cherry", "Earth", "Light spice"]
  },

  wine_merlot: {
    entity: "Wine",
    color: "var(--entity-wine)",
    name: "Merlot",
    type: "Red",
    body: "Medium-Full",
    acidity: "Medium",
    origin: "Bordeaux",
    flavor_profile: ["Plum", "Chocolate", "Soft tannins"]
  }
} as const;

/* =====================================================
   TYPE DERIVATION
===================================================== */

export type CardId = keyof typeof cardRegistry;
export type CardNode = (typeof cardRegistry)[CardId];

/* =====================================================
   SCENARIOS (STRICT, CONST-BASED)
===================================================== */

export const scenarios = [
  {
    key: "wine",
    label: "Wine pairing",

    question:
      "Which wine pairs best with the Chef’s special pasta? What’s the difference between a Pinot Noir and a Merlot from the menu?",

    reasoning: [
      {
        narration:
          "User requests wine pairing recommendation and wine comparison."
      },

      {
        title: "Step 1 - Retrieve dish entity",
        cards: [
          "dish_chefs_special_pasta"
        ]
      },

      {
        title: "Step 2 - Retrieve ingredients via Dish - has - Ingredient",
        cards: [
          "ingredient_atlantic_salmon",
          "ingredient_cream",
          "ingredient_pasta"
        ]
      },

      {
        title: "Step 3 - Retrieve candidate wines",
        cards: [
          "wine_pinot_noir",
          "wine_merlot"
        ]
      },

      {
        title: "Structural evaluation",
        logic: [
          "High-fat ingredients require balancing acidity.",
          "Pinot Noir acidity = High.",
          "Merlot acidity = Medium.",
          "Higher acidity structurally balances cream-based sauces.",
          "Pinot Noir provides better balance for this dish."
        ]
      }
    ],

    answer: `
<strong>Recommendation:</strong><br/>
<br/>
The Chef’s special pasta pairs best with <strong>Pinot Noir</strong> due to its medium body and higher acidity, which balances the richness of cream and salmon.
<br/><br/>
<strong>Comparison:</strong><br/>
• Pinot Noir – medium body, high acidity, bright red fruit profile  <br/>
• Merlot – medium-full body, softer tannins, lower acidity  <br/>
<br/>
For creamy sauces, higher acidity provides better structural balance.
`
  },
{
  key: "margin",
  label: "Menu margin optimization",

  question:
    "Which dishes on the menu have margins below 60% and should be repriced?",

  reasoning: [
    {
      narration:
        "User asks for profitability analysis across the full menu."
    },

    {
      title: "Step 1 - Retrieve full menu",
      logic: [
        "Query all Dish entities with attribute menu_price."
      ]
    },

    {
      title: "Step 2 - Retrieve recipe structure",
      logic: [
        "Traverse relation Dish - contains - Ingredient.",
        "Each Ingredient provides attributes amount and supplier_price."
      ]
    },

    {
      title: "Step 3 - Compute dish cost",
      logic: [
        "Dish cost is computed dynamically:",
        "cost_price = Σ(amount × supplier_price)"
      ]
    },

    {
      title: "Step 4 - Compute margin",
      logic: [
        "Margin formula:",
        "(menu_price - cost_price) / menu_price"
      ]
    },

    {
      title: "Step 5 - Filter by margin threshold",
      logic: [
        "Target margin threshold = 60%.",
        "Filter dishes where margin < 0.60."
      ]
    },

    {
      title: "Step 6 - Return lowest-margin dishes",
      cards: [
        "dish_grilled_salmon",
        "dish_truffle_pasta"
      ]
    }
  ],

  answer: `
<strong>Margin Analysis Result:</strong>
<br/><br/>

<strong>Grilled Salmon</strong><br/>
Cost: $14<br/>
Price: $22<br/>
Margin: <strong>36%</strong>
<br/><br/>

<strong>Truffle Pasta</strong><br/>
Cost: $11<br/>
Price: $18<br/>
Margin: <strong>39%</strong>
<br/><br/>

<strong>Recommendation:</strong><br/>
Increase menu prices by <strong>12–18%</strong> or renegotiate supplier contracts.
<br/><br/>

<em>Target margin:</em> <strong>65%+</strong> for premium positioning.
`
}
] as const;