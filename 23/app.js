const COMPONENTS = [
  {
    id: "header",
    section: "Header",
    label: "Header",
    icon: "nav",
    depth: 0,
    type: "section",
    children: [
      {
        id: "header-logo",
        label: "Logo",
        icon: "text",
        depth: 1,
        type: "text",
      },
      {
        id: "header-menu",
        label: "Menu",
        icon: "menu",
        depth: 1,
        type: "section",
      },
    ],
  },
  {
    id: "hero",
    section: "Template",
    label: "Image Banner",
    icon: "image",
    depth: 0,
    type: "text",
    children: [
      {
        id: "hero-heading",
        label: "Heading",
        icon: "text",
        depth: 1,
        type: "text",
      },
      {
        id: "hero-buttons",
        label: "Buttons",
        icon: "button",
        depth: 1,
        type: "section",
      },
    ],
  },
  {
    id: "products",
    section: null,
    label: "Featured Collection",
    icon: "grid",
    depth: 0,
    type: "section",
    children: [],
  },
  {
    id: "collection",
    section: null,
    label: "Image with Text",
    icon: "split",
    depth: 0,
    type: "image",
    children: [
      {
        id: "collection-image",
        label: "Image",
        icon: "image",
        depth: 1,
        type: "image",
      },
      {
        id: "collection-text",
        label: "Text",
        icon: "text",
        depth: 1,
        type: "text",
      },
    ],
  },
  {
    id: "testimonials",
    section: null,
    label: "Testimonials",
    icon: "quote",
    depth: 0,
    type: "section",
    children: [],
  },
  {
    id: "footer",
    section: "Footer group",
    label: "Footer",
    icon: "footer",
    depth: 0,
    type: "section",
    children: [],
  },
];

const ICONS = {
  nav: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="3" width="12" height="1.5" rx=".5" fill="currentColor"/><rect x="1" y="6.25" width="8" height="1.5" rx=".5" fill="currentColor"/><rect x="1" y="9.5" width="10" height="1.5" rx=".5" fill="currentColor"/></svg>',
  image:
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="2" width="12" height="10" rx="1.5" stroke="currentColor" stroke-width="1.2"/><circle cx="4.5" cy="5.5" r="1.2" fill="currentColor"/><path d="M1 10l3.5-3 2.5 2 2-1.5L13 10" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>',
  text: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 3h8M7 3v8M5 11h4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
  menu: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="2" y="4" width="10" height="1.2" rx=".4" fill="currentColor"/><rect x="2" y="6.9" width="10" height="1.2" rx=".4" fill="currentColor"/><rect x="2" y="9.8" width="10" height="1.2" rx=".4" fill="currentColor"/></svg>',
  button:
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="2" y="4.5" width="10" height="5" rx="2" stroke="currentColor" stroke-width="1.2"/></svg>',
  grid: '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="1" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.1"/><rect x="8" y="1" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.1"/><rect x="1" y="8" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.1"/><rect x="8" y="8" width="5" height="5" rx="1" stroke="currentColor" stroke-width="1.1"/></svg>',
  split:
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="2" width="5" height="10" rx="1" fill="currentColor" opacity=".3"/><path d="M8 4h5M8 7h4M8 10h5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>',
  quote:
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 8c0-2 1-3 2-4l1 1.5C4.5 6.5 4 7.5 4 9H2V8h1zm6 0c0-2 1-3 2-4l1 1.5C10.5 6.5 10 7.5 10 9H8V8h1z" fill="currentColor"/></svg>',
  footer:
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><rect x="1" y="9" width="12" height="4" rx="1" fill="currentColor" opacity=".35"/><rect x="1" y="2" width="12" height="5" rx="1" stroke="currentColor" stroke-width="1.1"/></svg>',
};

const BLOCK_TYPES = ["Rich text", "Image", "Button"];

const AI_LOGO = "img/ai-logo.png";

const SIDEKICK_SUGGESTION_PILLS = [
  "🎁 Gift buyer",
  "👋 First-time visitor",
  "🔍 Skeptic",
  "💆 Self-care seeker",
  "💰 Luxury shopper",
];

const SUGGESTIONS = {
  "image-banner": {
    "Gift buyer": [
      {
        label: "Headline & CTA",
        tags: ["copy", "CTA"],
        variants: [
          {
            reaction:
              "This reads like it's talking to someone buying for themselves — nothing signals this is a good gift.",
            suggestion:
              "Change the headline and CTA to speak to the gift-giving moment explicitly.",
            apply: {
              type: "multi",
              changes: [
                { type: "headline", value: "The gift they'll actually use" },
                { type: "cta_primary", value: "Shop Gift Sets" },
              ],
            },
          },
          {
            reaction:
              "I need to feel confident this is giftable before I even look at the product.",
            suggestion:
              "Lead with the occasion rather than the product benefit.",
            apply: {
              type: "multi",
              changes: [
                { type: "headline", value: "A gift worth giving twice" },
                { type: "cta_primary", value: "Find the right set" },
              ],
            },
          },
          {
            reaction:
              "The headline is beautiful but it's not helping me decide if this is right for someone else.",
            suggestion:
              "Reframe around the recipient experience rather than the ritual.",
            apply: {
              type: "multi",
              changes: [
                { type: "headline", value: "Give them something that stays" },
                { type: "cta_primary", value: "Browse gift options" },
              ],
            },
          },
        ],
      },
      {
        label: "Gifting hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The hero feels like everyday skincare — nothing tells me this is easy to give.",
            suggestion:
              "Swap to a warmer hero image that reads gift-ready and occasion-friendly.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
          {
            reaction:
              "I want to picture wrapping this before I read a single word of copy.",
            suggestion:
              "Use a hero shot with a softer, more celebratory mood for gift buyers.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
          {
            reaction:
              "Beautiful, but it doesn't help me imagine giving it to someone I love.",
            suggestion:
              "Switch to a lifestyle hero that feels curated for gifting.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
        ],
      },
    ],

    "First-time visitor": [
      {
        label: "Brand introduction",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "This looks beautiful but I have no idea what makes Mattie Studio different.",
            suggestion:
              "Replace the lifestyle headline with a specific point of difference.",
            apply: {
              type: "headline",
              value: "Small-batch skincare. Made without compromise.",
            },
          },
          {
            reaction:
              "I've never heard of this brand — I need a reason to keep reading.",
            suggestion:
              "Lead with the founding principle rather than a lifestyle statement.",
            apply: {
              type: "headline",
              value: "Skincare made the way it should have always been made.",
            },
          },
          {
            reaction:
              "Beautiful imagery but I still don't know what this brand stands for.",
            suggestion:
              "Use the headline to state the brand's clearest differentiator.",
            apply: {
              type: "headline",
              value: "No fillers. No shortcuts. Just skincare that works.",
            },
          },
        ],
      },
      {
        label: "Welcoming hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The hero is pretty but a little cold — I need to feel welcomed in immediately.",
            suggestion:
              "Swap to a brighter, more approachable hero image for first-time visitors.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
          {
            reaction:
              "I don't know this brand yet — the imagery should feel open, not intimidating.",
            suggestion:
              "Use a warmer lifestyle hero that invites exploration rather than perfection.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
          {
            reaction:
              "Something about this hero makes me hesitate — it feels too editorial for a first visit.",
            suggestion:
              "Switch to a friendlier hero image that lowers the barrier to browsing.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
        ],
      },
    ],

    Skeptic: [
      {
        label: "Proof-based headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "'Elevate your skin care ritual' — every brand says something like that.",
            suggestion:
              "Replace the aspiration with a specific, verifiable claim.",
            apply: {
              type: "headline",
              value: "97% natural origin ingredients. Tested over 12 weeks.",
            },
          },
          {
            reaction:
              "I need a reason to believe this is different before I read anything else.",
            suggestion:
              "Lead with the most credible proof point above the fold.",
            apply: {
              type: "headline",
              value: "Formulated without the 14 most common skin irritants.",
            },
          },
          {
            reaction: "Beautiful claim. Where's the evidence?",
            suggestion: "State the methodology, not just the outcome.",
            apply: {
              type: "headline",
              value:
                "Clinically tested. Independently verified. Nothing hidden.",
            },
          },
        ],
      },
      {
        label: "Hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "A model's face tells me nothing about what this product actually does.",
            suggestion:
              "Swap to a product-forward image that shows formula and texture.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
          {
            reaction:
              "I want to see the actual product, not just a lifestyle shot.",
            suggestion: "Use a hero that puts the formula front and center.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
          {
            reaction: "The current image is all feeling and no information.",
            suggestion: "Switch to a more product-direct hero image.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
        ],
      },
    ],

    "Self-care seeker": [
      {
        label: "Ritual-centred headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I'm investing in a moment for myself — the headline doesn't speak to that yet.",
            suggestion:
              "Centre the personal ritual experience over the product.",
            apply: {
              type: "headline",
              value: "Your skin. Your ritual. Your five minutes.",
            },
          },
          {
            reaction:
              "I want to feel like this brand understands what self-care actually means.",
            suggestion: "Lead with the feeling, not the formula.",
            apply: {
              type: "headline",
              value:
                "The part of your morning you'll actually look forward to.",
            },
          },
          {
            reaction:
              "The current headline is about the product. I want it to be about me.",
            suggestion:
              "Reframe around the moment of use rather than the product itself.",
            apply: {
              type: "headline",
              value: "Made for the five minutes that are just yours.",
            },
          },
        ],
      },
      {
        label: "Ritual hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The hero looks polished but not calming — I want to feel the ritual before I read.",
            suggestion:
              "Swap to a softer, more intimate hero image that evokes a self-care moment.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
          {
            reaction:
              "I need imagery that feels like quiet time, not a campaign shoot.",
            suggestion:
              "Use a hero with natural light and tactile surfaces that match how I'd use this.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
          {
            reaction:
              "The current image is beautiful but doesn't pull me into my own routine.",
            suggestion:
              "Switch to a warmer lifestyle hero centred on the ritual experience.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
        ],
      },
    ],

    "Luxury shopper": [
      {
        label: "Restrained headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The headline feels slightly wordy for where this price point wants to sit.",
            suggestion: "Shorten to a more restrained, confident statement.",
            apply: { type: "headline", value: "Elevate your ritual." },
          },
          {
            reaction:
              "Luxury brands don't need to explain themselves — this headline over-explains.",
            suggestion: "Strip back to the single most confident line.",
            apply: { type: "headline", value: "Skin care, reconsidered." },
          },
          {
            reaction:
              "The copy is doing too much work — confident luxury says less.",
            suggestion: "One short declarative line, nothing else.",
            apply: { type: "headline", value: "Your skin deserves better." },
          },
        ],
      },
      {
        label: "Editorial hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The hero is pleasant but feels mass-market — not where this price point should sit.",
            suggestion:
              "Swap to a more restrained, editorial hero image with confident negative space.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
          {
            reaction:
              "Luxury brands let the image do the talking — this one is trying too hard.",
            suggestion:
              "Use a quieter hero shot with a premium, less promotional feel.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" },
          },
          {
            reaction:
              "I want fewer visual cues and more craft — the current hero feels busy.",
            suggestion:
              "Switch to a minimal hero image that signals quality through restraint.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" },
          },
        ],
      },
    ],
  },

  "featured-collection": {
    "Gift buyer": [
      {
        label: "Collection headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "Nothing here signals this is where gift buyers should look.",
            suggestion:
              "Rename the section to speak directly to gift shoppers.",
            apply: {
              type: "headline",
              value: "Gifts they'll love. Sets they'll keep.",
            },
          },
          {
            reaction:
              "I'm shopping for someone else and this section feels like it's not for me.",
            suggestion: "Frame the collection as curated for giving.",
            apply: { type: "headline", value: "Curated to give." },
          },
          {
            reaction:
              "I can't tell if these products come as sets or individually.",
            suggestion: "Clarify the gifting format in the headline.",
            apply: {
              type: "headline",
              value: "Ready-to-gift sets. No wrapping required.",
            },
          },
        ],
      },
      {
        label: "Gift-ready collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The product grid looks polished but nothing signals these are easy to give.",
            suggestion:
              "Swap to a collection image set that reads as gift-ready and occasion-friendly.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "I want to see products presented the way I'd imagine wrapping them.",
            suggestion:
              "Use imagery that feels curated for gifting rather than everyday use.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "Beautiful products, but the photos don't help me picture giving them.",
            suggestion:
              "Switch to a warmer collection image set with a gifting mood.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    "First-time visitor": [
      {
        label: "Starting point headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I don't know this brand well enough to know which collection is right for me.",
            suggestion: "Guide a new visitor toward where to begin.",
            apply: {
              type: "headline",
              value: "New to Mattie Studio? Start here.",
            },
          },
          {
            reaction:
              "Too many options for someone who's just arrived — I need direction.",
            suggestion: "Position this as the recommended entry point.",
            apply: {
              type: "headline",
              value: "The collection most people start with.",
            },
          },
          {
            reaction:
              "I want to know what's most popular before I commit to anything.",
            suggestion: "Frame the collection around social proof.",
            apply: {
              type: "headline",
              value: "Our most loved products, in one place.",
            },
          },
        ],
      },
      {
        label: "Approachable collection imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The current images feel curated for people who already know the brand.",
            suggestion:
              "Use a warmer, more accessible product collection image set.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "I need the visuals to invite me in, not impress me from a distance.",
            suggestion:
              "Swap to imagery with a softer, more welcoming product presentation.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "Beautiful but slightly intimidating — I want to feel this is for me.",
            suggestion:
              "Switch to a collection image set that feels open and easy to explore.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    Skeptic: [
      {
        label: "Proof-led headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to know what's most popular — social validation matters here.",
            suggestion:
              "Lead with bestseller status to signal proven products.",
            apply: {
              type: "headline",
              value: "Our most repurchased products.",
            },
          },
          {
            reaction:
              "A generic collection name doesn't tell me why these products are worth it.",
            suggestion:
              "Use the headline to signal verified customer preference.",
            apply: {
              type: "headline",
              value: "The products customers reorder most.",
            },
          },
          {
            reaction:
              "I need evidence these are the right products before I go deeper.",
            suggestion: "Frame the collection around results and popularity.",
            apply: { type: "headline", value: "Tried, tested, repurchased." },
          },
        ],
      },
      {
        label: "Product-forward collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "Lifestyle grid shots don't tell me what I'm actually buying.",
            suggestion:
              "Swap to a collection image set with clearer product focus.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "I want to see the products themselves, not just styled surfaces.",
            suggestion:
              "Use imagery that puts formulas and packaging front and center.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "The photos are pretty but feel like marketing — show me the goods.",
            suggestion:
              "Switch to a more direct, product-led collection image set.",
            apply: { type: "image", optionIndex: 2 },
          },
        ],
      },
    ],

    "Self-care seeker": [
      {
        label: "Ritual-framed headline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to shop by ritual — morning, evening — not just by product.",
            suggestion: "Reframe the collection around routine building.",
            apply: { type: "headline", value: "Build your ritual." },
          },
          {
            reaction:
              "The collection feels product-focused when I'm experience-focused.",
            suggestion:
              "Name the emotional outcome rather than the product category.",
            apply: {
              type: "headline",
              value: "Everything your routine has been missing.",
            },
          },
          {
            reaction:
              "I want to feel like this collection was put together for someone like me.",
            suggestion:
              "Frame it as a personal curation, not a product listing.",
            apply: { type: "headline", value: "Your ritual, fully stocked." },
          },
        ],
      },
      {
        label: "Ritual mood collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "I want to feel the mood of using these products, not just see them on a shelf.",
            suggestion:
              "Swap to a collection image set that evokes a moment of self-care.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "The grid feels transactional — I'm shopping for a feeling, not a SKU.",
            suggestion:
              "Use warmer imagery that suggests a morning or evening ritual.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "Beautiful products, but the photos don't transport me into the experience.",
            suggestion:
              "Switch to imagery with softer light and a more intentional mood.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    "Luxury shopper": [
      {
        label: "Elevated collection label",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "'Featured Collection' is a generic label — it doesn't signal exclusivity.",
            suggestion:
              "Replace with a more editorial, curated-sounding title.",
            apply: { type: "headline", value: "The edit." },
          },
          {
            reaction:
              "This header reads like a template, not a considered curation.",
            suggestion:
              "Use a single restrained word that signals careful selection.",
            apply: { type: "headline", value: "Selected." },
          },
          {
            reaction:
              "A luxury brand shouldn't use the word 'featured' — it sounds algorithmic.",
            suggestion: "Rename to signal a handpicked, limited offering.",
            apply: { type: "headline", value: "The considered selection." },
          },
        ],
      },
      {
        label: "Editorial collection imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The collection imagery needs to match the premium positioning I expect.",
            suggestion:
              "Switch to the most refined, editorial product image set.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "These photos read as catalog, not curation — luxury is in the edit.",
            suggestion: "Use a tighter, more considered collection image set.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "At this price point, the visuals should feel intentional, not template-driven.",
            suggestion:
              "Swap to imagery with a more restrained, high-end product presentation.",
            apply: { type: "image", optionIndex: 2 },
          },
        ],
      },
    ],
  },

  "image-with-text": {
    "Gift buyer": [
      {
        label: "Body copy for gift context",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The copy is written for someone buying for themselves — nothing helps me justify this as a gift.",
            suggestion:
              "Rewrite body text to speak to the gift buyer's confidence.",
            apply: {
              type: "subheadline",
              value:
                "Whether it's a birthday, a thank you, or just because — our sets arrive gift-wrapped and ready to give. No extra step needed.",
            },
          },
          {
            reaction:
              "I need the copy to reassure me that this gift will land well.",
            suggestion:
              "Address the gift buyer's uncertainty about the recipient's reaction.",
            apply: {
              type: "subheadline",
              value:
                "Not sure what they'd prefer? Our sets are chosen to work for every skin type. And if it's not right, returns are always free.",
            },
          },
          {
            reaction: "I want to picture giving this, not using it myself.",
            suggestion:
              "Write the body copy from the giving perspective throughout.",
            apply: {
              type: "subheadline",
              value:
                "The kind of gift people keep and talk about. Beautifully packaged, thoughtfully made — and one they'll come back to every morning.",
            },
          },
        ],
      },
      {
        label: "Gift occasion imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The product images look beautiful for personal use but don't read as gift-oriented.",
            suggestion:
              "Switch to an image that better signals gifting and occasion.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "I can't picture wrapping this and giving it — the photo is too everyday.",
            suggestion:
              "Use imagery that feels ready to give, not just ready to use.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "The visual story is about the product on a vanity, not about giving.",
            suggestion: "Swap to a warmer image set with a gifting context.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    "First-time visitor": [
      {
        label: "Brand origin story",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I still don't know who made this or why — this section could answer that.",
            suggestion: "Use the body copy to introduce the brand story.",
            apply: {
              type: "subheadline",
              value:
                "Mattie Studio started as a single formula in a small kitchen. Every product is still made in small batches — never scaled, never compromised.",
            },
          },
          {
            reaction: "A beautiful brand but I have no context for it yet.",
            suggestion:
              "Open with the founding insight that makes this brand exist.",
            apply: {
              type: "subheadline",
              value:
                "We started Mattie Studio because we couldn't find a skincare brand we actually trusted. So we made one.",
            },
          },
          {
            reaction:
              "I want to understand the people behind this before I buy from them.",
            suggestion: "Make the brand voice personal and direct.",
            apply: {
              type: "subheadline",
              value:
                "Everything here is made the way we'd want our own skincare made. Small batches, honest ingredients, no shortcuts.",
            },
          },
        ],
      },
      {
        label: "Welcoming section image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The image feels polished but not especially inviting to someone new.",
            suggestion:
              "Swap to a warmer image that helps a first-time visitor feel welcome.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "I want to see people like me in this brand, not just a perfect still life.",
            suggestion: "Use a more approachable, less staged section image.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "Beautiful, but it doesn't help me understand what this brand is about visually.",
            suggestion:
              "Switch to imagery that feels open and easy to relate to.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    Skeptic: [
      {
        label: "Ingredient transparency",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to know exactly what's in this before I trust the brand.",
            suggestion:
              "Replace body copy with specific ingredient and process transparency.",
            apply: {
              type: "subheadline",
              value:
                "Every formula starts with a list of what we won't use. No fillers, no fragrance masking, no unnecessary additives.",
            },
          },
          {
            reaction:
              "Claims without evidence don't move me — I need specifics.",
            suggestion:
              "Lead with the most credible process detail in the body copy.",
            apply: {
              type: "subheadline",
              value:
                "Each ingredient is listed with its purpose. If it's in the formula, there's a reason for it. If there isn't, it's not there.",
            },
          },
          {
            reaction:
              "I want to understand the production process, not just the outcome.",
            suggestion:
              "Describe the methodology that makes the quality claim credible.",
            apply: {
              type: "subheadline",
              value:
                "Made in batches of under 500 units. Every batch tested before it ships. No exceptions.",
            },
          },
        ],
      },
      {
        label: "Product-forward section image",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "Lifestyle imagery doesn't answer my questions — I want to see the product itself.",
            suggestion:
              "Switch to an image set that shows the product and formula more directly.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "A mood shot tells me nothing about what's in the bottle.",
            suggestion:
              "Use imagery with clearer product detail and less styling.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "I'm not sold on vibes — show me what I'd actually be buying.",
            suggestion: "Swap to a more product-direct section image.",
            apply: { type: "image", optionIndex: 2 },
          },
        ],
      },
    ],

    "Self-care seeker": [
      {
        label: "Sensory experience copy",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The copy is informative but not evocative — I want to feel something reading it.",
            suggestion:
              "Rewrite to describe the sensory experience of using the product.",
            apply: {
              type: "subheadline",
              value:
                "The kind of routine you actually look forward to. Lightweight textures, considered scents, and formulas that absorb before you've finished your coffee.",
            },
          },
          {
            reaction:
              "I want this section to make me feel the ritual before I've bought anything.",
            suggestion:
              "Write the body copy as an evocation of the morning experience.",
            apply: {
              type: "subheadline",
              value:
                "Five minutes. Warm water. Something that smells exactly right. This is what we make that for.",
            },
          },
          {
            reaction:
              "The product sounds good but the copy doesn't make me feel it.",
            suggestion: "Use sensory language throughout the body copy.",
            apply: {
              type: "subheadline",
              value:
                "Cool to the touch. Warm on the skin. A texture that disappears and leaves only what it promised.",
            },
          },
        ],
      },
      {
        label: "Ritual mood imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "I want to feel the ritual in the image, not just read about it in the copy.",
            suggestion:
              "Swap to imagery that evokes a calm, intentional self-care moment.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "The photo is pretty but doesn't pull me into the experience.",
            suggestion:
              "Use a softer image with light and texture that suggest daily use.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction:
              "This section should make me want the moment, not just the product.",
            suggestion:
              "Switch to a more atmospheric, ritual-focused section image.",
            apply: { type: "image", optionIndex: 1 },
          },
        ],
      },
    ],

    "Luxury shopper": [
      {
        label: "Restrained craft copy",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The body copy is slightly too long — luxury positioning uses restraint.",
            suggestion:
              "Cut to a shorter, more considered line that trusts the reader.",
            apply: {
              type: "subheadline",
              value:
                "Made slowly, on purpose. Every ingredient chosen for a reason. Nothing added for appearance.",
            },
          },
          {
            reaction: "Too many words for a brand at this price point.",
            suggestion:
              "Strip back to the single most important craft statement.",
            apply: {
              type: "subheadline",
              value:
                "The formula took two years. The results speak without explanation.",
            },
          },
          {
            reaction:
              "Premium brands let the product speak — the copy here over-explains.",
            suggestion:
              "Use one restrained sentence that implies quality without stating it.",
            apply: {
              type: "subheadline",
              value: "Small batch. No compromise. That's the whole story.",
            },
          },
        ],
      },
      {
        label: "Refined section imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction:
              "The imagery needs to match the premium feel — I want clean, minimal, considered.",
            suggestion:
              "Switch to the image set with the most editorial, restrained aesthetic.",
            apply: { type: "image", optionIndex: 2 },
          },
          {
            reaction: "Too much visual noise for a brand at this price point.",
            suggestion:
              "Use a simpler, more confident section image with fewer distractions.",
            apply: { type: "image", optionIndex: 1 },
          },
          {
            reaction:
              "Luxury here should feel quiet — this photo competes with itself.",
            suggestion:
              "Swap to a more minimal, high-end product presentation.",
            apply: { type: "image", optionIndex: 2 },
          },
        ],
      },
    ],
  },

  testimonials: {
    "Gift buyer": [
      {
        label: "Gift-specific reviews",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "None of these reviews mention gifting — I want proof from people in my situation.",
            suggestion: "Surface testimonials from gift buyers specifically.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "Bought this for my sister's birthday. She called two days later asking where I got it.",
                  author: "Cleo R.",
                },
                {
                  rating: 5,
                  quote:
                    "I've given Mattie Studio sets as gifts four times now. It never misses.",
                  author: "Daniel T.",
                },
                {
                  rating: 5,
                  quote:
                    "The packaging alone made it feel expensive. The product backs it up completely.",
                  author: "Nina W.",
                },
              ],
            },
          },
          {
            reaction:
              "I need to see that other people have successfully given this as a gift.",
            suggestion:
              "Use testimonials that specifically mention the recipient's reaction.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "My mum still talks about the birthday gift I got her last year. This was it.",
                  author: "Priya K.",
                },
                {
                  rating: 5,
                  quote:
                    "Got this for a colleague leaving the team. She came back to ask the brand name.",
                  author: "James M.",
                },
                {
                  rating: 5,
                  quote:
                    "Gave it as a thank you gift. They reordered for themselves within a week.",
                  author: "Sara L.",
                },
              ],
            },
          },
          {
            reaction:
              "Reviews about personal use don't help me decide if this is a good gift.",
            suggestion: "Replace with occasion-specific testimonials.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "Perfect for anyone who's hard to buy for. Beautiful packaging, incredible product.",
                  author: "Amara S.",
                },
                {
                  rating: 5,
                  quote:
                    "I ordered this as a 'just because' gift. It made a much bigger impression than I expected.",
                  author: "Tom W.",
                },
                {
                  rating: 5,
                  quote:
                    "Ordered it as a gift, ended up buying one for myself too.",
                  author: "Mei T.",
                },
              ],
            },
          },
        ],
      },
    ],

    "First-time visitor": [
      {
        label: "First-purchase validation",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to hear from people who were also new to the brand — not just loyal fans.",
            suggestion:
              "Surface testimonials that reflect a first-time buyer's experience.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I was hesitant to try a brand I hadn't heard of. Two weeks in — completely converted.",
                  author: "Mara S.",
                },
                {
                  rating: 5,
                  quote:
                    "Ordered on a recommendation. It's now the only skincare I repurchase without thinking.",
                  author: "James O.",
                },
                {
                  rating: 5,
                  quote:
                    "Wasn't sure at first. The results after two weeks changed my mind completely.",
                  author: "Yuki T.",
                },
              ],
            },
          },
          {
            reaction:
              "I need to know the first purchase experience is good, not just the product.",
            suggestion:
              "Use reviews that mention the discovery and onboarding experience.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "Found this through a friend. The packaging, the smell, the texture — nothing disappointed.",
                  author: "Claire B.",
                },
                {
                  rating: 5,
                  quote:
                    "First order arrived beautifully. I knew before I even tried it that this brand was different.",
                  author: "Ravi D.",
                },
                {
                  rating: 5,
                  quote:
                    "Tried one product to test the brand. Now I own five. That says everything.",
                  author: "Lea F.",
                },
              ],
            },
          },
          {
            reaction:
              "Long-term customer reviews don't tell me what to expect on my first order.",
            suggestion:
              "Feature reviews specifically about the initial experience.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "Bought my first one sceptically. Reordered before the bottle was empty.",
                  author: "Sophie M.",
                },
                {
                  rating: 5,
                  quote:
                    "The first use was enough. I sent the link to three friends the same day.",
                  author: "Nia A.",
                },
                {
                  rating: 5,
                  quote:
                    "I almost didn't order. So glad I did. My skin has never felt this good.",
                  author: "Paulo R.",
                },
              ],
            },
          },
        ],
      },
    ],

    Skeptic: [
      {
        label: "Results-focused reviews",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "Vague reviews don't help me — I want specific, measurable outcomes.",
            suggestion: "Replace with testimonials that cite concrete results.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "My dermatologist asked what I'd changed in my routine. I told her. She wasn't surprised.",
                  author: "Amara S.",
                },
                {
                  rating: 5,
                  quote:
                    "Sceptical about the price. Four repurchases later — I completely understand it now.",
                  author: "Marcus O.",
                },
                {
                  rating: 5,
                  quote:
                    "Finally a brand that doesn't over-promise. It just quietly does what it says.",
                  author: "Lea B.",
                },
              ],
            },
          },
          {
            reaction: "I want before and after — not just 'I loved it'.",
            suggestion:
              "Surface reviews that describe a specific skin concern that was resolved.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I'd had uneven skin tone for years. Six weeks of this and the difference is visible in photos.",
                  author: "Diana C.",
                },
                {
                  rating: 5,
                  quote:
                    "Everything else I tried either irritated my skin or did nothing. This did neither — it just worked.",
                  author: "Felix H.",
                },
                {
                  rating: 5,
                  quote:
                    "I tracked my skin weekly for a month. The improvement by week three was undeniable.",
                  author: "Nora P.",
                },
              ],
            },
          },
          {
            reaction: "Five-star reviews without specifics read as fake to me.",
            suggestion:
              "Use reviews with specific details that signal authenticity.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I've been using it twice daily for four months. No breakouts, no irritation, noticeably clearer.",
                  author: "Tom K.",
                },
                {
                  rating: 5,
                  quote:
                    "Combination skin, easily congested. This is the first product that hasn't made it worse.",
                  author: "Ana V.",
                },
                {
                  rating: 5,
                  quote:
                    "I read the ingredient list before buying. Everything checks out. Results confirmed my research.",
                  author: "Sam J.",
                },
              ],
            },
          },
        ],
      },
    ],

    "Self-care seeker": [
      {
        label: "Ritual experience reviews",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to hear from people who made this part of a meaningful routine.",
            suggestion:
              "Feature reviews about the ritual experience, not just the results.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "This is the one part of my morning I genuinely look forward to now.",
                  author: "Jamie K.",
                },
                {
                  rating: 5,
                  quote:
                    "I bought it for my skin. I kept it for how it makes me feel getting ready.",
                  author: "Priya M.",
                },
                {
                  rating: 5,
                  quote:
                    "My five-minute routine became my favourite part of the day.",
                  author: "Sarah L.",
                },
              ],
            },
          },
          {
            reaction:
              "I want to feel like buying this changes my mornings, not just my skin.",
            suggestion:
              "Surface reviews that describe a lifestyle shift, not just a product benefit.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I used to skip my skincare most mornings. Now I make time for it. The ritual is part of the point.",
                  author: "Cora B.",
                },
                {
                  rating: 5,
                  quote:
                    "It's hard to explain but using this feels intentional. Like I'm being kind to myself.",
                  author: "Ella S.",
                },
                {
                  rating: 5,
                  quote:
                    "My morning is different now. Slower. Better. This is a big part of why.",
                  author: "Hana M.",
                },
              ],
            },
          },
          {
            reaction:
              "Product reviews feel transactional — I want emotional resonance.",
            suggestion:
              "Use reviews that describe what the product means to the person.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "After a really hard year, this felt like reclaiming something for myself.",
                  author: "Jo D.",
                },
                {
                  rating: 5,
                  quote:
                    "Small luxury, real impact. This is what I mean when I say self-care actually works.",
                  author: "Ren A.",
                },
                {
                  rating: 5,
                  quote:
                    "I bought this for myself without justifying it to anyone. Best decision I made that month.",
                  author: "Zoe C.",
                },
              ],
            },
          },
        ],
      },
    ],

    "Luxury shopper": [
      {
        label: "Discerning buyer reviews",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "I want to hear from people with high standards — not just general positive reviews.",
            suggestion:
              "Surface reviews from buyers who compare against premium alternatives.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I've used La Mer, Sisley, and Augustinus Bader. Mattie Studio holds its own.",
                  author: "Claire D.",
                },
                {
                  rating: 5,
                  quote:
                    "Worth every penny. The texture and absorption are unlike anything at this price point.",
                  author: "Ravi M.",
                },
                {
                  rating: 5,
                  quote:
                    "I gifted this to myself after a difficult year. It felt right. It still does.",
                  author: "Sophie W.",
                },
              ],
            },
          },
          {
            reaction:
              "Generic five-star reviews don't tell me if this meets a high standard.",
            suggestion:
              "Use reviews that reference the buyer's specific quality expectations.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote:
                    "I only buy products I'd be comfortable recommending to someone with discerning taste. This qualifies.",
                  author: "Marcus L.",
                },
                {
                  rating: 5,
                  quote:
                    "The finish, the scent, the packaging — nothing feels like a compromise.",
                  author: "Ines V.",
                },
                {
                  rating: 5,
                  quote:
                    "I've tried most things at this price point. This is the one I stopped looking after.",
                  author: "Thomas B.",
                },
              ],
            },
          },
          {
            reaction:
              "I want reviews that tell me this is worth the premium without overselling.",
            suggestion:
              "Feature understated, confident reviews that imply quality without gushing.",
            apply: {
              type: "testimonials",
              value: [
                {
                  rating: 5,
                  quote: "I don't write reviews. I'm writing this one.",
                  author: "N. Ashworth",
                },
                {
                  rating: 5,
                  quote: "Quietly the best thing in my bathroom.",
                  author: "J. Moreau",
                },
                {
                  rating: 5,
                  quote: "I expected good. I got better than that.",
                  author: "P. Chen",
                },
              ],
            },
          },
        ],
      },
    ],
  },

  footer: {
    "Gift buyer": [
      {
        label: "Gifting tagline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The tagline speaks to personal use — nothing signals this is also a great gift brand.",
            suggestion: "Update to acknowledge the gifting occasion.",
            apply: {
              type: "subheadline",
              value: "Small-batch skincare. Made to give and keep.",
            },
          },
          {
            reaction:
              "I've just been thinking about gifts and the footer doesn't reinforce that.",
            suggestion:
              "Close the page with a gifting-friendly brand statement.",
            apply: {
              type: "subheadline",
              value: "Skincare worth giving. And worth keeping for yourself.",
            },
          },
          {
            reaction:
              "The closing brand statement misses the occasion I'm shopping for.",
            suggestion:
              "Acknowledge both the gift buyer and the recipient in the tagline.",
            apply: {
              type: "subheadline",
              value: "Made with intention. Given with meaning.",
            },
          },
        ],
      },
    ],

    "First-time visitor": [
      {
        label: "Welcoming tagline",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The tagline is confident but slightly insider — I want the brand to feel welcoming.",
            suggestion: "Update to feel more open and inviting to someone new.",
            apply: {
              type: "subheadline",
              value: "Small-batch skincare. Made for the curious.",
            },
          },
          {
            reaction:
              "I've just discovered this brand and the closing line doesn't invite me in.",
            suggestion:
              "End the page with a line that encourages further exploration.",
            apply: {
              type: "subheadline",
              value: "New here? You're in the right place.",
            },
          },
          {
            reaction:
              "The tagline wraps up the brand for existing customers, not for me.",
            suggestion:
              "Make the closing statement speak to someone still deciding.",
            apply: {
              type: "subheadline",
              value:
                "Skincare made to earn your trust. Starting with the first bottle.",
            },
          },
        ],
      },
    ],

    Skeptic: [
      {
        label: "Direct honesty tagline",
        tags: ["copy"],
        variants: [
          {
            reaction: "'Made with intention' is vague — every brand says that.",
            suggestion:
              "Replace with a tagline that makes a specific, honest claim.",
            apply: {
              type: "subheadline",
              value: "Small-batch skincare. No fillers. No shortcuts.",
            },
          },
          {
            reaction:
              "The closing statement is poetic but says nothing verifiable.",
            suggestion:
              "End with a direct product truth rather than a brand aspiration.",
            apply: {
              type: "subheadline",
              value:
                "We list every ingredient. We explain every choice. That's the whole pitch.",
            },
          },
          {
            reaction:
              "I want the brand to close on something I can actually hold them to.",
            suggestion: "Use a specific, accountable closing statement.",
            apply: {
              type: "subheadline",
              value:
                "If it's in the bottle, it's on the label. If it's not on the label, it's not in the bottle.",
            },
          },
        ],
      },
    ],

    "Self-care seeker": [
      {
        label: "Ritual closing statement",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "The tagline is brand-focused — I want it to speak to my routine.",
            suggestion: "Update to centre the personal experience.",
            apply: {
              type: "subheadline",
              value: "Small-batch skincare. Made for your five minutes.",
            },
          },
          {
            reaction:
              "I want the closing line to feel like a gentle invitation to slow down.",
            suggestion:
              "End with something that evokes the ritual rather than the product.",
            apply: {
              type: "subheadline",
              value: "Your routine deserves this much thought.",
            },
          },
          {
            reaction:
              "The footer closes on the brand. I want it to close on the feeling.",
            suggestion:
              "Use the tagline to land on the emotional outcome of the ritual.",
            apply: {
              type: "subheadline",
              value: "Made for the mornings that feel like yours.",
            },
          },
        ],
      },
    ],

    "Luxury shopper": [
      {
        label: "Premium closing line",
        tags: ["copy"],
        variants: [
          {
            reaction:
              "'Made with intention' reads as indie-brand language, not true luxury.",
            suggestion:
              "Replace with a more restrained, confident closing line.",
            apply: { type: "subheadline", value: "Made slowly. On purpose." },
          },
          {
            reaction:
              "The tagline is pleasant but doesn't signal the premium I expect.",
            suggestion:
              "Use a closing statement that implies exclusivity through restraint.",
            apply: {
              type: "subheadline",
              value: "Not for everyone. Made for those who notice.",
            },
          },
          {
            reaction:
              "A luxury brand's last word should be its most confident.",
            suggestion:
              "Close with a single declarative line that needs no explanation.",
            apply: {
              type: "subheadline",
              value: "Nothing wasted. Nothing missing.",
            },
          },
        ],
      },
    ],
  },
};

const HERO_BG_GRADIENT =
  "linear-gradient(180deg, rgba(250, 246, 240, 0.72) 0%, rgba(245, 213, 220, 0.45) 45%, rgba(61, 44, 46, 0.18) 100%)";

const ASSETS = {
  "image-banner": [
    "img/image-banner/header-1.png",
    "img/image-banner/header-2.jpg",
    "img/image-banner/header-3.jpg",
  ],
  "featured-collection": [
    [
      "img/featured-collection/a-1.png",
      "img/featured-collection/a-2.png",
      "img/featured-collection/a-3.png",
      "img/featured-collection/a-4.png",
    ],
    [
      "img/featured-collection/b-1.jpg",
      "img/featured-collection/b-2.jpg",
      "img/featured-collection/b-3.jpg",
      "img/featured-collection/b-4.jpeg",
    ],
    [
      "img/featured-collection/c-1.jpg",
      "img/featured-collection/c-2.webp",
      "img/featured-collection/c-3.jpg",
      "img/featured-collection/c-4.jpg",
    ],
  ],
  "image-with-text": [
    "img/image-with-text/collection-1.png",
    "img/image-with-text/collection-2.jpg",
    "img/image-with-text/collection-3.jpg",
  ],
};

const componentState = {
  "image-banner": { index: 0, total: 3 },
  "featured-collection": { index: 0, total: 3 },
  "image-with-text": { index: 0, total: 3 },
  testimonials: { index: 0, total: 3 },
  footer: { index: 0, total: 3 },
};

const CANVAS_TO_REROLL_KEY = {
  hero: "image-banner",
  products: "featured-collection",
  collection: "image-with-text",
  testimonials: "testimonials",
  footer: "footer",
};

const TESTIMONIALS_OPTIONS = [
  [
    {
      rating: 5,
      quote:
        "This changed my entire morning routine. My skin has never looked better.",
      author: "Jamie K.",
    },
    {
      rating: 5,
      quote:
        "I've tried every luxury skincare brand. Mattie Studio is the one I keep coming back to.",
      author: "Priya M.",
    },
    {
      rating: 5,
      quote:
        "Worth every penny. The texture is unlike anything else I've used.",
      author: "Sarah L.",
    },
  ],
  [
    {
      rating: 5,
      quote:
        "I bought this as a gift for my sister and she immediately asked me to order her another one.",
      author: "Cleo R.",
    },
    {
      rating: 5,
      quote:
        "Gave this to my mum for her birthday. She called three days later to ask where I got it.",
      author: "Daniel T.",
    },
    {
      rating: 5,
      quote:
        "The packaging alone made it feel expensive. The product backs it up completely.",
      author: "Nina W.",
    },
  ],
  [
    {
      rating: 5,
      quote:
        "I was skeptical about the price point. Two months in — I understand now. Won't go back.",
      author: "Amara S.",
    },
    {
      rating: 5,
      quote:
        "My dermatologist asked what I'd changed in my routine. That was enough proof for me.",
      author: "Marcus O.",
    },
    {
      rating: 5,
      quote:
        "Finally a brand that doesn't over-promise. It just quietly does what it says.",
      author: "Lea B.",
    },
  ],
];

const FOOTER_OPTIONS = [
  {
    tagline: "Small-batch skincare. Made with intention.",
    leftLinks: [
      "Shipping & Returns",
      "Ingredients",
      "Our Story",
      "Stockists",
      "Contact",
    ],
    shop: ["All Products", "Collections", "Gift Sets"],
    about: ["Our Story", "Ingredients", "Sustainability"],
    help: ["Contact", "Shipping", "Returns"],
  },
  {
    tagline: "Formulated for your skin. Not for the algorithm.",
    leftLinks: [
      "Shipping & Returns",
      "Ingredients",
      "Our Story",
      "Gift Wrapping",
      "Contact",
    ],
    shop: ["All Products", "Gift Sets", "Bundles"],
    about: ["Our Story", "Ingredients", "Press"],
    help: ["Contact", "Shipping", "Gift Returns"],
  },
  {
    tagline: "Skincare that respects your skin — and your time.",
    leftLinks: [
      "Shipping & Returns",
      "Ingredients",
      "Our Story",
      "Wholesale",
      "Contact",
    ],
    shop: ["All Products", "Collections", "New Arrivals"],
    about: ["Our Story", "Ingredients", "Journal"],
    help: ["Contact", "Shipping", "Returns"],
  },
];

let selectedId = null;
let openDropdown = null;
let lensOpen = false;
let currentPanelMeta = null;
let lensSessionComponentId = null;
const lensConversations = {};
let lensPending = false;
let lensAnimating = false;

const workspace = document.getElementById("workspace");
const sectionTree = document.getElementById("sectionTree");
const rightPanel = document.getElementById("rightPanel");
const panelTitle = document.getElementById("panelTitle");
const panelSettingsLayer = document.getElementById("panelSettingsLayer");
const lensChatPanel = document.getElementById("lensChatPanel");
const lensChatMessages = document.getElementById("lensChatMessages");
const lensChatInput = document.getElementById("lensChatInput");
const lensChatSend = document.getElementById("lensChatSend");
const LENS_ENTRY_HTML = `
      <hr class="panel-lens-divider" />
      <button type="button" class="lens-entry-btn" id="lensEntryBtn">
        <img class="lens-entry-icon" src="img/SidekickIcon.svg" width="16" height="16" alt="" style="flex-shrink:0" />
        <span class="lens-entry-text">See through a buyer's eyes</span>
        <span class="lens-entry-chevron" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M5 3l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </span>
      </button>`;

let lensEntryAttentionTimer = null;

function playLensEntryAttention(btn) {
  if (!btn?.isConnected) return;
  btn.classList.remove("lens-entry-attention");
  void btn.offsetWidth;
  btn.classList.add("lens-entry-attention");
  btn.addEventListener(
    "animationend",
    () => {
      btn.classList.remove("lens-entry-attention");
      if (btn.isConnected) {
        lensEntryAttentionTimer = setTimeout(
          () => playLensEntryAttention(btn),
          3000,
        );
      }
    },
    { once: true },
  );
}

function scheduleLensEntryAttention(btn) {
  clearTimeout(lensEntryAttentionTimer);
  if (!btn) return;
  lensEntryAttentionTimer = setTimeout(() => playLensEntryAttention(btn), 1000);
}

function setupLensEntryBtn(btn) {
  btn.addEventListener("click", openLens);
  scheduleLensEntryAttention(btn);
}

function getComponentMeta(id) {
  for (const c of COMPONENTS) {
    if (c.id === id) return { ...c, panelType: c.type };
    for (const ch of c.children || []) {
      if (ch.id === id) return { ...ch, panelType: ch.type, parent: c };
    }
  }
  return { id, label: id, panelType: "section" };
}

function resolveCanvasId(id) {
  const meta = getComponentMeta(id);
  if (meta.parent) return meta.parent.id;
  const top = COMPONENTS.find((c) => c.id === id);
  if (top) return id;
  return id;
}

function buildTree() {
  let html = "";
  let lastSection = null;

  COMPONENTS.forEach((comp) => {
    if (comp.section && comp.section !== lastSection) {
      lastSection = comp.section;
      html += `<div class="tree-section-label">${comp.section}</div>`;
    }

    html += treeRow(comp);
    (comp.children || []).forEach((ch) => {
      html += treeRow(ch);
    });

    html += `
          <div class="tree-add" data-section="${comp.id}">
            <span>+</span> Add block
            <div class="add-dropdown">
              ${BLOCK_TYPES.map((t, i) => {
                const icons = [ICONS.text, ICONS.image, ICONS.button];
                return `<div class="add-dropdown-item"><span class="tree-icon">${icons[i]}</span>${t}</div>`;
              }).join("")}
            </div>
          </div>`;
  });

  sectionTree.innerHTML = html;

  sectionTree.querySelectorAll(".tree-row").forEach((row) => {
    row.addEventListener("click", () => selectComponent(row.dataset.id));
  });

  sectionTree.querySelectorAll(".tree-add").forEach((add) => {
    add.addEventListener("click", (e) => {
      e.stopPropagation();
      const dd = add.querySelector(".add-dropdown");
      if (openDropdown && openDropdown !== dd)
        openDropdown.classList.remove("open");
      dd.classList.toggle("open");
      openDropdown = dd.classList.contains("open") ? dd : null;
    });
  });

  sectionTree.querySelectorAll(".add-dropdown-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      e.stopPropagation();
      if (openDropdown) openDropdown.classList.remove("open");
      openDropdown = null;
    });
  });
}

function treeRow(item) {
  const icon = ICONS[item.icon] || ICONS.text;
  return `
        <div class="tree-row" data-id="${item.id}" data-depth="${item.depth}">
          <span class="tree-drag">⠿</span>
          <span class="tree-icon">${icon}</span>
          <span class="tree-label">${item.label}</span>
        </div>`;
}

function getTopComponentMeta(id) {
  const meta = getComponentMeta(id);
  if (meta.parent) return meta.parent;
  const top = COMPONENTS.find((c) => c.id === id);
  return top || meta;
}

function getLensConversation(componentId) {
  if (!lensConversations[componentId]) {
    lensConversations[componentId] = { messages: [], matchedPersona: null };
  }
  return lensConversations[componentId];
}

function crossfadeElement(el, applyUpdate) {
  if (!el) {
    applyUpdate();
    return;
  }
  el.style.transition = "opacity 150ms ease";
  el.style.opacity = "0";
  setTimeout(() => {
    applyUpdate();
    requestAnimationFrame(() => {
      el.style.opacity = "1";
    });
  }, 150);
}

function applyImageBanner(optionIndex, animate) {
  const hero = document.querySelector('[data-id="hero"] .sf-hero');
  if (!hero) return;
  const url = ASSETS["image-banner"][optionIndex];
  const apply = () => {
    hero.style.background = `${HERO_BG_GRADIENT}, url('${url}') center center / cover no-repeat`;
  };
  if (animate) crossfadeElement(hero, apply);
  else apply();
}

function applyFeaturedCollection(optionIndex, animate) {
  const container = document.querySelector('[data-id="products"] .sf-products');
  const urls = ASSETS["featured-collection"][optionIndex];
  const apply = () => {
    document
      .querySelectorAll('[data-id="products"] [data-reroll-img]')
      .forEach((img, i) => {
        if (urls[i]) img.src = urls[i];
      });
  };
  if (animate) crossfadeElement(container, apply);
  else apply();
}

function applyImageWithText(optionIndex, animate) {
  const container = document.querySelector(
    '[data-id="collection"] .sf-collection-img',
  );
  const img = container?.querySelector("[data-reroll-img]");
  const url = ASSETS["image-with-text"][optionIndex];
  const apply = () => {
    if (img && url) img.src = url;
  };
  if (animate) crossfadeElement(container, apply);
  else apply();
}

function applyTestimonials(optionIndex, animate) {
  const container = document.querySelector(
    '[data-id="testimonials"] .sf-testimonial-grid',
  );
  const option = TESTIMONIALS_OPTIONS[optionIndex];
  const apply = () => {
    const cards = document.querySelectorAll(
      '[data-id="testimonials"] .sf-testimonial-card',
    );
    cards.forEach((card, i) => {
      const t = option[i];
      if (!t) return;
      card.querySelector(".sf-testimonial-text").textContent = `"${t.quote}"`;
      card.querySelector(".sf-testimonial-author").textContent =
        `— ${t.author}`;
      card.querySelector(".sf-stars").textContent = "★".repeat(t.rating);
    });
  };
  if (animate) crossfadeElement(container, apply);
  else apply();
}

function setFooterLinkList(ul, links) {
  if (ul) ul.innerHTML = links.map((link) => `<li>${link}</li>`).join("");
}

function applyFooter(optionIndex, animate) {
  const container = document.querySelector('[data-id="footer"] .sf-footer');
  const opt = FOOTER_OPTIONS[optionIndex];
  const apply = () => {
    const taglineEl = container.querySelector(".sf-footer-tagline");
    if (taglineEl) taglineEl.textContent = opt.tagline;
    setFooterLinkList(
      container.querySelector(".sf-footer-reroll-links"),
      opt.leftLinks,
    );
    setFooterLinkList(
      container.querySelector(".sf-footer-shop-links"),
      opt.shop,
    );
    setFooterLinkList(
      container.querySelector(".sf-footer-about-links"),
      opt.about,
    );
    setFooterLinkList(
      container.querySelector(".sf-footer-help-links"),
      opt.help,
    );
  };
  if (animate) crossfadeElement(container, apply);
  else apply();
}

function applyComponentOption(componentKey, optionIndex, animate = false) {
  switch (componentKey) {
    case "image-banner":
      applyImageBanner(optionIndex, animate);
      break;
    case "featured-collection":
      applyFeaturedCollection(optionIndex, animate);
      break;
    case "image-with-text":
      applyImageWithText(optionIndex, animate);
      break;
    case "testimonials":
      applyTestimonials(optionIndex, animate);
      break;
    case "footer":
      applyFooter(optionIndex, animate);
      break;
    default:
      break;
  }
}

function initComponentDefaults() {
  Object.keys(componentState).forEach((key) => {
    componentState[key].index = 0;
    applyComponentOption(key, 0, false);
  });
}

function normalizeAssetUrl(path) {
  if (!path) return path;
  if (path.startsWith("http")) return path;
  return path.replace(/^\//, "");
}

function getComponentElement(componentId) {
  return (
    document.querySelector(`[data-component="${componentId}"]`) ||
    document.querySelector(`[data-id="${componentId}"]`) ||
    document.getElementById(componentId)
  );
}

function snapshotComponent(component) {
  const snap = {};

  const headline = component.querySelector('[data-field="headline"]');
  if (headline) snap.headline = headline.textContent;

  const sub = component.querySelector('[data-field="subheadline"]');
  if (sub) {
    snap.subheadline = sub.textContent;
    snap.subheadline_display = sub.style.display || "";
  }

  const ctaP = component.querySelector('[data-field="cta_primary"]');
  if (ctaP) snap.cta_primary = ctaP.textContent;

  const ctaS = component.querySelector('[data-field="cta_secondary"]');
  if (ctaS) {
    snap.cta_secondary = ctaS.textContent;
    snap.cta_secondary_visible = ctaS.style.visibility;
  }

  const productImgs = component.querySelectorAll("[data-reroll-img]");
  if (productImgs.length > 1) {
    snap.product_images = Array.from(productImgs).map((img) => img.src);
  } else {
    const img = component.querySelector('[data-field="main-image"]');
    if (img) {
      if (img.tagName === "IMG") snap.image_src = img.src;
      else snap.image_src = img.style.background || "";
    }
  }

  const cards = component.querySelectorAll('[data-field="review-card"]');
  if (cards.length) {
    snap.testimonials = Array.from(cards).map((card) => ({
      quote:
        card.querySelector('[data-field="review-quote"]')?.textContent || "",
      author:
        card.querySelector('[data-field="review-author"]')?.textContent || "",
      rating: (card.querySelector(".sf-stars")?.textContent || "").length,
    }));
  }

  return snap;
}

function snapshotImagesDiffer(component, snap) {
  if (snap.product_images?.length) {
    const imgs = component.querySelectorAll("[data-reroll-img]");
    return snap.product_images.some((url, i) => imgs[i] && imgs[i].src !== url);
  }
  if (!snap.image_src) return false;
  const el = component.querySelector('[data-field="main-image"]');
  if (!el) return false;
  const current = el.tagName === "IMG" ? el.src : el.style.background || "";
  return current !== snap.image_src;
}

function restoreSnapshot(component, snap, options = {}) {
  const { instant = false } = options;
  const animateImages = !instant && snapshotImagesDiffer(component, snap);

  if (snap.headline !== undefined) {
    const el = component.querySelector('[data-field="headline"]');
    if (el) el.textContent = snap.headline;
  }
  if (snap.subheadline !== undefined) {
    const el = component.querySelector('[data-field="subheadline"]');
    if (el) {
      el.textContent = snap.subheadline;
      el.style.display =
        snap.subheadline_display ?? (snap.subheadline ? "" : "none");
    }
  }
  if (snap.cta_primary !== undefined) {
    const el = component.querySelector('[data-field="cta_primary"]');
    if (el) el.textContent = snap.cta_primary;
  }
  if (snap.cta_secondary !== undefined) {
    const el = component.querySelector('[data-field="cta_secondary"]');
    if (el) {
      el.textContent = snap.cta_secondary;
      el.style.visibility = snap.cta_secondary_visible || "visible";
    }
  }
  if (snap.product_images?.length) {
    const imgs = component.querySelectorAll("[data-reroll-img]");
    imgs.forEach((img, i) => {
      if (!snap.product_images[i] || img.src === snap.product_images[i]) return;
      if (!animateImages) {
        img.style.transition = "";
        img.style.opacity = "1";
        img.src = snap.product_images[i];
      } else {
        img.style.transition = "opacity 150ms ease";
        img.style.opacity = "0";
        setTimeout(() => {
          img.src = snap.product_images[i];
          img.style.opacity = "1";
        }, 150);
      }
    });
  } else if (snap.image_src) {
    const el = component.querySelector('[data-field="main-image"]');
    if (el) {
      const current = el.tagName === "IMG" ? el.src : el.style.background || "";
      if (current === snap.image_src) {
        el.style.transition = "";
        el.style.opacity = "1";
      } else if (!animateImages) {
        if (el.tagName === "IMG") {
          el.src = snap.image_src;
        } else {
          el.style.background = snap.image_src;
        }
        el.style.transition = "";
        el.style.opacity = "1";
      } else {
        el.style.transition = "opacity 150ms ease";
        el.style.opacity = "0";
        setTimeout(() => {
          if (el.tagName === "IMG") {
            el.src = snap.image_src;
          } else {
            el.style.background = snap.image_src;
          }
          el.style.opacity = "1";
        }, 150);
      }
    }
  }
  if (snap.testimonials) {
    const cards = component.querySelectorAll('[data-field="review-card"]');
    cards.forEach((card, i) => {
      if (!snap.testimonials[i]) return;
      const quote = card.querySelector('[data-field="review-quote"]');
      const author = card.querySelector('[data-field="review-author"]');
      const stars = card.querySelector(".sf-stars");
      if (quote) quote.textContent = snap.testimonials[i].quote;
      if (author) author.textContent = snap.testimonials[i].author;
      if (stars && snap.testimonials[i].rating) {
        stars.textContent = "★".repeat(snap.testimonials[i].rating);
      }
    });
  }
}

function applySingleChange(component, change) {
  switch (change.type) {
    case "headline": {
      const el = component.querySelector('[data-field="headline"]');
      if (el) el.textContent = change.value;
      break;
    }

    case "subheadline": {
      const el = component.querySelector('[data-field="subheadline"]');
      if (el) {
        el.textContent = change.value;
        el.style.display = change.value ? "" : "none";
      }
      break;
    }

    case "cta_primary": {
      const el = component.querySelector('[data-field="cta_primary"]');
      if (el) el.textContent = change.value;
      break;
    }

    case "cta_secondary": {
      const el = component.querySelector('[data-field="cta_secondary"]');
      if (el) {
        if (change.value === "") {
          el.style.visibility = "hidden";
        } else {
          el.style.visibility = "visible";
          el.textContent = change.value;
        }
      }
      break;
    }

    case "image": {
      const componentId = component.getAttribute("data-component");
      const productImgs = component.querySelectorAll("[data-reroll-img]");

      if (componentId === "featured-collection" && productImgs.length > 1) {
        let optionIndex = change.optionIndex;
        if (optionIndex == null) {
          const current = componentState["featured-collection"]?.index ?? 0;
          optionIndex = (current + 1) % ASSETS["featured-collection"].length;
        }
        const urls = ASSETS["featured-collection"][optionIndex];
        if (urls) {
          if (componentState["featured-collection"]) {
            componentState["featured-collection"].index = optionIndex;
          }
          productImgs.forEach((img, i) => {
            if (!urls[i]) return;
            img.style.transition = "opacity 150ms ease";
            img.style.opacity = "0";
            setTimeout(() => {
              img.src = urls[i];
              img.style.opacity = "1";
            }, 150);
          });
        }
        break;
      }

      if (componentId === "image-with-text") {
        let optionIndex = change.optionIndex;
        if (optionIndex == null) {
          const current = componentState["image-with-text"]?.index ?? 0;
          optionIndex = (current + 1) % ASSETS["image-with-text"].length;
        }
        const url = ASSETS["image-with-text"][optionIndex];
        const el =
          component.querySelector('[data-field="main-image"]') ||
          component.querySelector("[data-reroll-img]");
        if (url && el) {
          if (componentState["image-with-text"]) {
            componentState["image-with-text"].index = optionIndex;
          }
          el.style.transition = "opacity 150ms ease";
          el.style.opacity = "0";
          setTimeout(() => {
            if (el.tagName === "IMG") {
              el.src = url;
            } else {
              el.style.background = `${HERO_BG_GRADIENT}, url('${url}') center center / cover no-repeat`;
            }
            el.style.opacity = "1";
          }, 150);
        }
        break;
      }

      const el = component.querySelector('[data-field="main-image"]');
      if (el) {
        const url = normalizeAssetUrl(change.value);
        el.style.transition = "opacity 150ms ease";
        el.style.opacity = "0";
        setTimeout(() => {
          if (el.tagName === "IMG") {
            el.src = url;
          } else {
            el.style.background = `${HERO_BG_GRADIENT}, url('${url}') center center / cover no-repeat`;
          }
          el.style.opacity = "1";
        }, 150);
      }
      break;
    }

    case "testimonials": {
      const cards = component.querySelectorAll('[data-field="review-card"]');
      cards.forEach((card, i) => {
        if (!change.value[i]) return;
        const quote = card.querySelector('[data-field="review-quote"]');
        const author = card.querySelector('[data-field="review-author"]');
        const stars = card.querySelector(".sf-stars");
        if (quote) quote.textContent = `"${change.value[i].quote}"`;
        if (author) author.textContent = `— ${change.value[i].author}`;
        if (stars && change.value[i].rating) {
          stars.textContent = "★".repeat(change.value[i].rating);
        }
      });
      break;
    }
  }
}

function applyChange(componentId, applyObj) {
  const component = getComponentElement(componentId);
  if (!component) {
    console.warn("Component not found:", componentId);
    return;
  }

  if (applyObj.type === "multi") {
    applyObj.changes.forEach((change) => applySingleChange(component, change));
  } else {
    applySingleChange(component, applyObj);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function matchPersona(input) {
  const text = input.toLowerCase();

  if (
    text.match(
      /gift|giving|birthday|present|someone else|mum|mom|sister|brother|friend|occasion/,
    )
  ) {
    return "Gift buyer";
  }
  if (
    text.match(
      /skeptic|skeptical|sceptic|prove|proof|evidence|trust|convince|doubt|not sure|hesitant/,
    )
  ) {
    return "Skeptic";
  }
  if (
    text.match(
      /self.care|ritual|routine|myself|me time|relax|mindful|morning|evening|treat/,
    )
  ) {
    return "Self-care seeker";
  }
  if (
    text.match(
      /luxury|premium|high.end|expensive|worth it|quality|discerning|aesop|la mer/,
    )
  ) {
    return "Luxury shopper";
  }
  return "First-time visitor";
}

function getCurrentSuggestionComponentId() {
  if (!lensSessionComponentId) return null;
  return CANVAS_TO_REROLL_KEY[lensSessionComponentId] || lensSessionComponentId;
}

function getPersonaSuggestions(personaKey) {
  const componentId = getCurrentSuggestionComponentId();
  if (!componentId || !personaKey) return [];
  const componentSuggestions = SUGGESTIONS[componentId];
  if (!componentSuggestions) return [];
  const pool =
    componentSuggestions[personaKey] ??
    componentSuggestions["First-time visitor"] ??
    [];
  return pool.slice(0, 2);
}

function buildDemoLensData(personaKey, componentName, isFollowUp) {
  const componentId = getCurrentSuggestionComponentId();
  const hasComponent = Boolean(componentId && SUGGESTIONS[componentId]);
  const suggestions = hasComponent ? getPersonaSuggestions(personaKey) : [];

  if (isFollowUp) {
    return {
      introText: `I've updated the suggestions below based on your question. Try applying a direction and reroll to cycle through alternatives.`,
      disclaimer: "✦ Demo mode — responses are fixed examples.",
      persona_key: personaKey,
      suggestions,
      noSuggestionsMessage: hasComponent
        ? null
        : "This component doesn't have specific suggestions for this perspective — try selecting a different component on the canvas.",
      isFollowUp: true,
    };
  }

  return {
    introText: `Here's how a ${personaKey} would experience your ${componentName}:`,
    persona_key: personaKey,
    suggestions,
    noSuggestionsMessage:
      hasComponent && suggestions.length
        ? null
        : "This component doesn't have specific suggestions for this perspective — try selecting a different component on the canvas.",
    isFollowUp: false,
  };
}

function getSuggestionVariant(suggestion, variantIndex = 0) {
  const variants = suggestion?.variants ?? [];
  if (variantIndex === "default") return variants[0] ?? null;
  const idx =
    typeof variantIndex === "number"
      ? variantIndex
      : parseInt(variantIndex, 10);
  if (Number.isNaN(idx)) return variants[0] ?? null;
  return variants[idx] ?? variants[0] ?? null;
}

function nextVariantStep(currentStep, variantCount) {
  if (variantCount <= 0) return "0";
  if (currentStep === "default") return "0";
  const idx = parseInt(currentStep, 10);
  if (Number.isNaN(idx) || idx >= variantCount - 1) return "default";
  return String(idx + 1);
}

function applyVariantStep(cardEl, compId, suggestion, step) {
  const variants = suggestion.variants;
  if (step === "default") {
    const snap = cardEl.dataset.snapshot
      ? JSON.parse(cardEl.dataset.snapshot)
      : null;
    const component = getComponentElement(compId);
    if (snap && component) restoreSnapshot(component, snap);
    cardEl.dataset.applyData = JSON.stringify(variants[0].apply ?? null);
    return;
  }
  const variant = variants[parseInt(step, 10)];
  cardEl.dataset.applyData = JSON.stringify(variant.apply ?? null);
  applyChange(compId, variant.apply);
}

function updateLensInputPlaceholder(conv) {
  const hasLensResponse = conv.messages.some((m) => m.role === "assistant");
  lensChatInput.placeholder = hasLensResponse
    ? "Ask a question"
    : "e.g. a mom buying a gift for her daughter";
}

const THUMB_UP_IMG = "img/thumbs-up.png";
const THUMB_DOWN_IMG = "img/thumbs-down.png";

function getSidekickOpeningText(componentName) {
  return `Seeing your store through a specific buyer's eyes can significantly impact conversions — small copy or layout changes for the right audience often move the needle more than full redesigns.

I'm looking at your <strong>${componentName}</strong> right now.

What perspective would you like to explore?`;
}

function createAIMessageRow(contentHtml) {
  const row = document.createElement("div");
  row.className = "ai-message-row";
  row.innerHTML = `
        <img src="${AI_LOGO}" class="ai-avatar" alt="" width="32" height="32" />
        <div class="ai-message-content">${contentHtml}</div>`;
  return row;
}

function renderSidekickOpening(top, scroll = true) {
  const group = document.createElement("div");
  group.className = "chat-ai-group sidekick-opening";

  const row = createAIMessageRow(
    `<div class="sidekick-opening-text">${getSidekickOpeningText(top.label)}</div>
        <span class="sidekick-demo-disclaimer">✦ Demo mode — suggestions are fixed examples to illustrate the concept.</span>`,
  );

  const pills = document.createElement("div");
  pills.className = "lens-suggestion-pills";
  pills.innerHTML = SIDEKICK_SUGGESTION_PILLS.map(
    (label) =>
      `<button type="button" class="lens-suggestion-pill" data-text="${label.replace(/"/g, "&quot;")}">${label}</button>`,
  ).join("");
  pills.querySelectorAll(".lens-suggestion-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.dataset.text;
      lensChatInput.value = text;
      lensChatSend.disabled = false;
      submitLensMessage(text);
    });
  });
  row.querySelector(".ai-message-content").appendChild(pills);
  group.appendChild(row);

  lensChatMessages.appendChild(group);
  if (scroll) scrollLensChatToElement(group);
  return group;
}

function isLensChatNearBottom(threshold = 96) {
  const el = lensChatMessages;
  if (!el) return true;
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollLensChatToBottom(onlyIfNearBottom = false) {
  if (!lensChatMessages) return;
  if (onlyIfNearBottom && !isLensChatNearBottom()) return;
  requestAnimationFrame(() => {
    lensChatMessages.scrollTop = lensChatMessages.scrollHeight;
  });
}

function scrollLensChatToElement(el, options = {}) {
  if (!el || !lensChatMessages) return;
  const { onlyIfNearBottom = false, align = "end" } = options;
  if (onlyIfNearBottom && !isLensChatNearBottom()) return;

  requestAnimationFrame(() => {
    const container = lensChatMessages;
    const padding = 16;
    const elTop = el.offsetTop;
    const elBottom = elTop + el.offsetHeight;
    const viewTop = container.scrollTop;
    const viewBottom = viewTop + container.clientHeight;

    if (align === "end" || elBottom > viewBottom) {
      container.scrollTop = elBottom - container.clientHeight + padding;
    } else if (align === "start" && elTop < viewTop) {
      container.scrollTop = Math.max(0, elTop - padding);
    }
  });
}

function scrollLensChatToLatestResponse(onlyIfNearBottom = true) {
  const lastGroup = lensChatMessages.querySelector(
    ".chat-ai-group:last-of-type",
  );
  if (lastGroup) {
    scrollLensChatToElement(lastGroup, { onlyIfNearBottom, align: "end" });
  } else {
    scrollLensChatToBottom(onlyIfNearBottom);
  }
}

function renderLensChatFromHistory(conv, top) {
  lensChatMessages
    .querySelectorAll(".chat-msg-user, .chat-ai-group, .chat-msg-loading")
    .forEach((el) => el.remove());
  renderSidekickOpening(top, false);
  conv.messages.forEach((msg) => {
    if (msg.role === "user") appendUserBubble(msg.content, false);
    else if (msg.role === "assistant") appendAIGroup(msg.data, false);
  });
  updateLensInputPlaceholder(conv);
  scrollLensChatToLatestResponse(false);
}

function appendUserBubble(text, scroll = false) {
  const el = document.createElement("div");
  el.className = "chat-msg-user";
  el.textContent = text;
  lensChatMessages.appendChild(el);
  if (scroll) scrollLensChatToBottom(false);
  return el;
}

function transitionCardToDefault(cardEl, options = {}) {
  const { keepSnapshot = false } = options;
  const applyBtn = cardEl.querySelector(".btn-apply");
  const appliedBtn = cardEl.querySelector(".btn-applied-status");
  const rerollBtn = cardEl.querySelector(".btn-reroll");
  const undoBtn = cardEl.querySelector(".btn-undo");

  cardEl.classList.remove("is-applied");
  applyBtn.hidden = false;
  applyBtn.classList.remove("applied");
  applyBtn.textContent = "Apply";
  applyBtn.disabled = false;
  if (appliedBtn) appliedBtn.hidden = true;
  undoBtn.hidden = true;
  rerollBtn.hidden = true;
  rerollBtn.textContent = "↺ Try another option";
  rerollBtn.disabled = false;

  if (!keepSnapshot) {
    delete cardEl.dataset.snapshot;
  }
}

function transitionCardToApplied(cardEl) {
  const applyBtn = cardEl.querySelector(".btn-apply");
  const appliedBtn = cardEl.querySelector(".btn-applied-status");
  const rerollBtn = cardEl.querySelector(".btn-reroll");
  const undoBtn = cardEl.querySelector(".btn-undo");
  const suggestion = cardEl._suggestion;

  cardEl.classList.add("is-applied");
  applyBtn.hidden = true;
  if (appliedBtn) appliedBtn.hidden = false;
  undoBtn.hidden = false;
  if (suggestion?.variants?.length >= 1) {
    rerollBtn.hidden = false;
    rerollBtn.style.display = "";
  } else {
    rerollBtn.hidden = true;
  }
}

function updateCardContent(cardEl, suggestion, variantIndex = 0) {
  const variant = getSuggestionVariant(suggestion, variantIndex);
  if (!variant) return;
  cardEl.querySelector(".chat-card-label").textContent = suggestion.label;
  cardEl.querySelector(".chat-card-quote").textContent =
    `"${variant.reaction}"`;
  cardEl.querySelector(".chat-card-change").textContent = variant.suggestion;
  const tagsEl = cardEl.querySelector(".chat-card-tags");
  tagsEl.innerHTML = (suggestion.tags || [])
    .map((t) => `<span class="chat-card-tag">${escapeHtml(t)}</span>`)
    .join("");
  cardEl.dataset.applyData = JSON.stringify(variant.apply ?? null);
}

function renderCard(cardEl, suggestion, componentId, variantIndex) {
  cardEl.dataset.componentId = componentId || "";
  cardEl.dataset.variantIndex = String(variantIndex ?? 0);
  cardEl._suggestion = suggestion;
  updateCardContent(cardEl, suggestion, variantIndex);

  const applyBtn = cardEl.querySelector(".btn-apply");
  const undoBtn = cardEl.querySelector(".btn-undo");
  const rerollBtn = cardEl.querySelector(".btn-reroll");

  applyBtn.onclick = () => {
    if (cardEl.classList.contains("is-applied")) return;
    const compId = cardEl.dataset.componentId;
    if (!compId || !cardEl.dataset.applyData) return;

    const applyObj = JSON.parse(cardEl.dataset.applyData);
    if (!applyObj) return;
    const component = getComponentElement(compId);
    if (!cardEl.dataset.snapshot && component) {
      cardEl.dataset.snapshot = JSON.stringify(snapshotComponent(component));
    }

    applyChange(compId, applyObj);
    transitionCardToApplied(cardEl);
  };

  undoBtn.onclick = () => {
    const snap = cardEl.dataset.snapshot
      ? JSON.parse(cardEl.dataset.snapshot)
      : null;
    if (snap) {
      const component = getComponentElement(cardEl.dataset.componentId);
      if (component) restoreSnapshot(component, snap);
      delete cardEl.dataset.snapshot;
    }
    transitionCardToDefault(cardEl);
  };

  rerollBtn.onclick = () => {
    if (rerollBtn.hidden || rerollBtn.disabled) return;
    const compId = cardEl.dataset.componentId;
    const suggestion = cardEl._suggestion;
    if (!suggestion?.variants?.length) return;

    rerollBtn.disabled = true;
    rerollBtn.textContent = "↺ Trying another...";

    const variants = suggestion.variants;
    const currentStep = cardEl.dataset.variantIndex || "0";
    const nextStep = nextVariantStep(currentStep, variants.length);
    cardEl.dataset.variantIndex = nextStep;
    applyVariantStep(cardEl, compId, suggestion, nextStep);

    setTimeout(() => {
      rerollBtn.textContent = "↺ Try another option";
      rerollBtn.disabled = false;
    }, 300);
  };
}

function buildDirectionCard(
  item,
  index,
  personaKey,
  suggestionsPool,
  componentId,
) {
  const card = document.createElement("div");
  card.className = "chat-direction-card";
  card.style.animationDelay = `${index * 80}ms`;
  card.dataset.cardIndex = String(index);
  if (personaKey) card.dataset.personaKey = personaKey;
  card._suggestionsPool = suggestionsPool;
  card.innerHTML = `
        <div class="chat-card-top">
          <span class="chat-card-label"></span>
          <button type="button" class="btn-undo" hidden aria-label="Undo applied changes">Undo</button>
        </div>
        <p class="chat-card-quote"></p>
        <p class="chat-card-change"></p>
        <div class="chat-card-tags"></div>
        <div class="card-actions">
          <button type="button" class="btn-apply">Apply</button>
          <button type="button" class="btn-applied-status" hidden disabled>✓ Applied</button>
          <button type="button" class="btn-reroll" hidden>↺ Try another option</button>
        </div>`;

  const rerollBtn = card.querySelector(".btn-reroll");
  if (!item.variants?.length) {
    rerollBtn.style.display = "none";
  }

  renderCard(card, item, componentId, 0);
  transitionCardToDefault(card);
  return card;
}

function appendAIGroup(data, scroll = true) {
  const group = document.createElement("div");
  group.className = "chat-ai-group";

  let introHtml = "";
  if (data.introText) {
    introHtml = `<div class="chat-intro-text">${escapeHtml(data.introText)}</div>`;
    if (data.disclaimer) {
      introHtml += `<span class="sidekick-demo-disclaimer">${escapeHtml(data.disclaimer)}</span>`;
    }
  }
  if (introHtml) {
    group.appendChild(createAIMessageRow(introHtml));
  }

  if (data.noSuggestionsMessage) {
    group.appendChild(
      createAIMessageRow(
        `<div class="chat-intro-text">${escapeHtml(data.noSuggestionsMessage)}</div>`,
      ),
    );
  }

  const suggestions = data.suggestions || [];
  if (suggestions.length) {
    const suggestionsSection = document.createElement("div");
    suggestionsSection.className = "chat-suggestions-section";
    suggestionsSection.innerHTML =
      '<h3 class="chat-suggestions-title">Suggestions</h3>';
    const grid = document.createElement("div");
    grid.className = "chat-suggestions-grid";
    const personaKey = data.persona_key || null;
    const pool = personaKey ? getPersonaSuggestions(personaKey) : suggestions;
    const componentId = getCurrentSuggestionComponentId() || "";
    suggestions.slice(0, 2).forEach((item, i) => {
      grid.appendChild(
        buildDirectionCard(item, i, personaKey, pool, componentId),
      );
    });
    suggestionsSection.appendChild(grid);
    group.appendChild(suggestionsSection);
  }

  const thumbs = document.createElement("div");
  thumbs.className = "chat-thumbs";
  thumbs.innerHTML = `
        <button type="button" class="chat-thumb-btn" data-vote="up" aria-label="Helpful"><img src="${THUMB_UP_IMG}" alt="" /></button>
        <button type="button" class="chat-thumb-btn" data-vote="down" aria-label="Not helpful"><img src="${THUMB_DOWN_IMG}" alt="" /></button>`;
  thumbs.querySelectorAll(".chat-thumb-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      thumbs
        .querySelectorAll(".chat-thumb-btn")
        .forEach((b) => b.classList.remove("is-selected"));
      btn.classList.add("is-selected");
    });
  });
  group.appendChild(thumbs);

  const followups = data.followup_chips || [
    "What would you change first?",
    "How does this look on mobile?",
  ];
  const list = document.createElement("ul");
  list.className = "chat-followup-links";
  list.innerHTML = followups
    .map(
      (f) =>
        `<li><button type="button" class="lens-prompt-link" data-text="${f.replace(/"/g, "&quot;")}">${f}</button></li>`,
    )
    .join("");
  list.querySelectorAll(".lens-prompt-link").forEach((btn) => {
    btn.addEventListener("click", () => submitLensMessage(btn.dataset.text));
  });
  group.appendChild(list);

  lensChatMessages.appendChild(group);
  if (scroll && isLensChatNearBottom()) {
    scrollLensChatToElement(group, { align: "end" });
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (isLensChatNearBottom()) scrollLensChatToBottom();
      });
    });
  }
  return group;
}

function appendLoadingBubble() {
  const el = createAIMessageRow(
    '<div class="chat-loading-dots"><span></span><span></span><span></span></div>',
  );
  el.classList.add("chat-msg-loading");
  lensChatMessages.appendChild(el);
  scrollLensChatToBottom(isLensChatNearBottom());
  return el;
}

function clearLensConversation(componentId) {
  if (componentId) {
    lensConversations[componentId] = { messages: [], matchedPersona: null };
  }
  lensPending = false;
  if (lensChatInput) {
    lensChatInput.value = "";
    lensChatSend.disabled = true;
  }
}

function submitLensMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || lensPending || !lensSessionComponentId) return;

  const top = COMPONENTS.find((c) => c.id === lensSessionComponentId);
  if (!top) return;

  const conv = getLensConversation(lensSessionComponentId);
  const isFollowUp = conv.messages.some((m) => m.role === "assistant");
  const loadingMs = isFollowUp ? 1000 : 1400;

  lensPending = true;
  lensChatInput.value = "";
  lensChatSend.disabled = true;

  appendUserBubble(trimmed, true);
  conv.messages.push({ role: "user", content: trimmed });

  if (!isFollowUp) {
    conv.matchedPersona = matchPersona(trimmed);
  }
  const personaKey = conv.matchedPersona || matchPersona(trimmed);

  const loadingEl = appendLoadingBubble();

  setTimeout(() => {
    loadingEl.remove();
    const data = buildDemoLensData(personaKey, top.label, isFollowUp);
    appendAIGroup(data);
    conv.messages.push({ role: "assistant", data });
    updateLensInputPlaceholder(conv);
    lensPending = false;
    lensChatSend.disabled = !lensChatInput.value.trim();
  }, loadingMs);
}

function openLens() {
  if (!currentPanelMeta || lensOpen || lensAnimating) return;
  const top = getTopComponentMeta(currentPanelMeta.id);
  const canvasId = top.id;

  lensAnimating = true;
  lensOpen = true;
  lensSessionComponentId = canvasId;

  workspace.classList.add("lens-closing-settings");
  rightPanel.classList.add("settings-closing");

  setTimeout(() => {
    rightPanel.classList.add("settings-hidden");
    rightPanel.classList.remove("settings-closing");

    lensChatPanel.classList.add("open");
    lensChatPanel.setAttribute("aria-hidden", "false");

    workspace.classList.remove("lens-closing-settings");
    workspace.classList.add("lens-open");

    renderLensChatFromHistory(getLensConversation(canvasId), top);

    lensChatInput.value = "";
    lensChatSend.disabled = true;
    updateLensInputPlaceholder(getLensConversation(canvasId));
    lensAnimating = false;
  }, 200);
}

function closeLens() {
  clearLensConversation(lensSessionComponentId);

  if ((!lensOpen && !lensChatPanel.classList.contains("open")) || lensAnimating)
    return;

  lensAnimating = true;
  lensOpen = false;

  workspace.classList.remove("lens-open");
  workspace.classList.add("lens-closing-lens");
  lensChatPanel.classList.remove("open");

  setTimeout(() => {
    rightPanel.classList.remove("settings-hidden");
    rightPanel.classList.add("settings-opening");
    workspace.classList.remove("lens-closing-lens");
    lensChatPanel.setAttribute("aria-hidden", "true");

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        rightPanel.classList.remove("settings-opening");
      });
    });

    setTimeout(() => {
      lensPending = false;
      lensAnimating = false;
    }, 220);
  }, 200);
}

function selectComponent(id) {
  const canvasId = resolveCanvasId(id);

  selectedId = id;
  const meta = getComponentMeta(id);
  currentPanelMeta = meta;

  if (lensOpen) closeLens();

  document.querySelectorAll(".tree-row").forEach((r) => {
    r.classList.toggle("selected", r.dataset.id === id);
  });

  document.querySelectorAll(".sf-block").forEach((b) => {
    const match = b.dataset.id === canvasId;
    b.classList.toggle("selected", match);
    b.classList.remove("hovered");
  });

  const top = getTopComponentMeta(id);
  panelTitle.textContent = top.label;
  renderPanel(meta);
  rightPanel.classList.remove("settings-hidden", "settings-closing");
  rightPanel.classList.add("visible");
  workspace.classList.add("has-right-panel");
}

function clearSelection() {
  selectedId = null;
  currentPanelMeta = null;
  closeLens();
  document
    .querySelectorAll(".tree-row")
    .forEach((r) => r.classList.remove("selected"));
  document.querySelectorAll(".sf-block").forEach((b) => {
    b.classList.remove("selected", "hovered");
  });
  rightPanel.classList.remove("visible", "settings-hidden", "settings-closing");
  workspace.classList.remove(
    "has-right-panel",
    "lens-open",
    "lens-closing-settings",
    "lens-closing-lens",
  );
}

function renderPanel(meta) {
  const type = meta.panelType || meta.type || "section";
  let html = "";

  if (type === "text") {
    html = `
          <div class="panel-section active">
            <div class="panel-group">
              <div class="panel-label">Typography</div>
              <div class="panel-field">
                <label>Font family</label>
                <select class="panel-input" id="fontFamily">
                  <option>Playfair Display</option>
                  <option>Inter</option>
                  <option>Georgia</option>
                </select>
              </div>
              <div class="panel-field">
                <label>Size <span class="value-display" id="sizeVal">42px</span></label>
                <input type="range" class="panel-range" id="fontSize" min="16" max="64" value="42" />
              </div>
              <div class="panel-field">
                <label>Weight</label>
                <div class="weight-btns" id="weightBtns">
                  <button type="button" class="weight-btn" data-w="400">Regular</button>
                  <button type="button" class="weight-btn active" data-w="500">Medium</button>
                  <button type="button" class="weight-btn" data-w="600">Semibold</button>
                </div>
              </div>
              <div class="panel-field">
                <label>Color</label>
                <div class="color-row">
                  <span class="color-swatch-preview" id="colorPreview" style="background:#3d2c2e"></span>
                  <input type="color" class="panel-color" id="textColor" value="#3d2c2e" style="flex:1" />
                </div>
              </div>
            </div>
            <div class="panel-group">
              <div class="panel-label">Content</div>
              <div class="panel-field">
                <label>Heading text</label>
                <input type="text" class="panel-input" id="headingText" value="Elevate your skin care ritual" />
              </div>
            </div>
          </div>`;
  } else if (type === "image") {
    html = `
          <div class="panel-section active">
            <div class="panel-group">
              <div class="panel-label">Image</div>
              <div class="upload-zone">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none"><path d="M12 16V8m0 0l-3 3m3-3l3 3" stroke="#6b6b8a" stroke-width="1.5" stroke-linecap="round"/><rect x="3" y="3" width="18" height="18" rx="3" stroke="#6b6b8a" stroke-width="1.5"/></svg>
                <p>Drop an image or click to upload</p>
                <button type="button" class="btn-ai">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 1l1.2 3.5L12 5.5 9.5 8l.8 4L7 10.2 3.7 12l.8-4L2 5.5l3.8-1L7 1z" fill="#6c63ff"/></svg>
                  Generate with AI
                </button>
              </div>
            </div>
            <div class="panel-group">
              <div class="panel-label">Layout</div>
              <div class="panel-field">
                <label>Image position</label>
                <select class="panel-input" id="imgPosition">
                  <option>Left</option>
                  <option>Right</option>
                </select>
              </div>
            </div>
          </div>`;
  } else {
    html = `
          <div class="panel-section active">
            <div class="panel-group">
              <div class="panel-label">Spacing</div>
              <div class="panel-field">
                <label>Padding top <span class="value-display" id="padTopVal">56px</span></label>
                <input type="range" class="panel-range" id="padTop" min="0" max="120" value="56" />
              </div>
              <div class="panel-field">
                <label>Padding bottom <span class="value-display" id="padBottomVal">56px</span></label>
                <input type="range" class="panel-range" id="padBottom" min="0" max="120" value="56" />
              </div>
              <div class="panel-field">
                <label>Gap between blocks <span class="value-display" id="blockGapVal">20px</span></label>
                <input type="range" class="panel-range" id="blockGap" min="0" max="64" value="20" />
              </div>
            </div>
            <div class="panel-group">
              <div class="panel-label">Background</div>
              <div class="panel-field">
                <label>Section color</label>
                <div class="color-row">
                  <span class="color-swatch-preview" id="bgPreview" style="background:#faf6f0"></span>
                  <input type="color" class="panel-color" id="sectionBg" value="#faf6f0" style="flex:1" />
                </div>
              </div>
            </div>
          </div>`;
  }

  panelSettingsLayer.innerHTML = html + LENS_ENTRY_HTML;
  bindPanelInputs();
  const lensEntryBtn = document.getElementById("lensEntryBtn");
  if (lensEntryBtn) setupLensEntryBtn(lensEntryBtn);
}

function bindPanelInputs() {
  const size = document.getElementById("fontSize");
  const sizeVal = document.getElementById("sizeVal");
  if (size && sizeVal) {
    size.addEventListener("input", () => {
      sizeVal.textContent = size.value + "px";
    });
  }

  ["padTop", "padBottom", "blockGap"].forEach((id) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(id + "Val");
    if (el && valEl) {
      el.addEventListener("input", () => {
        valEl.textContent = el.value + "px";
      });
    }
  });

  const textColor = document.getElementById("textColor");
  const colorPreview = document.getElementById("colorPreview");
  if (textColor && colorPreview) {
    textColor.addEventListener("input", () => {
      colorPreview.style.background = textColor.value;
    });
  }

  const sectionBg = document.getElementById("sectionBg");
  const bgPreview = document.getElementById("bgPreview");
  if (sectionBg && bgPreview) {
    sectionBg.addEventListener("input", () => {
      bgPreview.style.background = sectionBg.value;
    });
  }

  document.querySelectorAll(".weight-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".weight-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
}

document.querySelectorAll(".sf-block").forEach((block) => {
  block.addEventListener("mouseenter", () => {
    if (!block.classList.contains("selected")) block.classList.add("hovered");
  });
  block.addEventListener("mouseleave", () => block.classList.remove("hovered"));

  block.addEventListener("click", (e) => {
    e.stopPropagation();
    selectComponent(block.dataset.id);
  });
});

document.getElementById("panelClose").addEventListener("click", clearSelection);
document.getElementById("lensChatBack").addEventListener("click", closeLens);
document.getElementById("lensChatClose").addEventListener("click", closeLens);

lensChatInput.addEventListener("input", () => {
  lensChatSend.disabled = !lensChatInput.value.trim() || lensPending;
});

lensChatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    submitLensMessage(lensChatInput.value);
  }
});

lensChatSend.addEventListener("click", () =>
  submitLensMessage(lensChatInput.value),
);

document.getElementById("canvasArea").addEventListener("click", (e) => {
  if (e.target.closest(".sf-block")) return;
  if (e.target.closest(".right-panel")) return;
  if (e.target.closest(".lens-chat-panel")) return;
  clearSelection();
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".tree-add")) {
    if (openDropdown) {
      openDropdown.classList.remove("open");
      openDropdown = null;
    }
  }
});

document.querySelectorAll(".tab-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document
      .querySelectorAll(".tab-pill")
      .forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
  });
});

buildTree();
initComponentDefaults();
selectComponent("hero");
