export const SUGGESTIONS = {

  "image-banner": {

    "Gift buyer": [
      {
        label: "Headline & CTA",
        tags: ["copy", "CTA"],
        variants: [
          {
            reaction: "This reads like it's talking to someone buying for themselves — nothing signals this is a good gift.",
            insight: "Gift buyers need the page to acknowledge their intent within seconds — or they'll look elsewhere.",
            suggestion: "Change the headline and CTA to speak to the gift-giving moment explicitly.",
            apply: { type: "multi", changes: [
              { type: "headline", value: "The gift they'll actually use" },
              { type: "cta_primary", value: "Shop Gift Sets" }
            ]}
          },
          {
            reaction: "I need to feel confident this is giftable before I even look at the product.",
            insight: "Leading with the occasion rather than the product removes the mental work of imagining it as a gift.",
            suggestion: "Lead with the occasion rather than the product benefit.",
            apply: { type: "multi", changes: [
              { type: "headline", value: "A gift worth giving twice" },
              { type: "cta_primary", value: "Find the right set" }
            ]}
          },
          {
            reaction: "The headline is beautiful but it's not helping me decide if this is right for someone else.",
            insight: "Reframing around the recipient — not the buyer — closes the imaginative gap for someone shopping for others.",
            suggestion: "Reframe around the recipient experience rather than the ritual.",
            apply: { type: "multi", changes: [
              { type: "headline", value: "Give them something that stays" },
              { type: "cta_primary", value: "Browse gift options" }
            ]}
          }
        ]
      },
      {
        label: "Gifting hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The hero feels like everyday skincare — nothing tells me this is easy to give.",
            insight: "For a gift buyer, visual mood is the fastest signal — if the image doesn't say 'gift', the copy has to work twice as hard.",
            suggestion: "Swap to a warmer hero image that reads gift-ready and occasion-friendly.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          },
          {
            reaction: "I want to picture wrapping this before I read a single word of copy.",
            insight: "Gift buyers make emotional decisions first — the hero image should do that emotional work before the headline.",
            suggestion: "Use a hero shot with a softer, more celebratory mood for gift buyers.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          },
          {
            reaction: "Beautiful, but it doesn't help me imagine giving it to someone I love.",
            insight: "A lifestyle hero that feels curated for giving removes the translation work for a buyer shopping for someone else.",
            suggestion: "Switch to a lifestyle hero that feels curated for gifting.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          }
        ]
      }
    ],

    "First-time visitor": [
      {
        label: "Brand introduction",
        tags: ["copy"],
        variants: [
          {
            reaction: "This looks beautiful but I have no idea what makes Mattie Studio different.",
            insight: "A first-time visitor needs a specific reason to stay — a lifestyle headline gives them nothing to hold onto.",
            suggestion: "Replace the lifestyle headline with a specific point of difference.",
            apply: { type: "headline", value: "Small-batch skincare. Made without compromise." }
          },
          {
            reaction: "I've never heard of this brand — I need a reason to keep reading.",
            insight: "A founding principle in the headline signals conviction — it tells a new visitor that this brand exists for a reason.",
            suggestion: "Lead with the founding principle rather than a lifestyle statement.",
            apply: { type: "headline", value: "Skincare made the way it should have always been made." }
          },
          {
            reaction: "Beautiful imagery but I still don't know what this brand stands for.",
            insight: "A first-time visitor is looking for the clearest differentiator — the headline is the one place it must land.",
            suggestion: "Use the headline to state the brand's clearest differentiator.",
            apply: { type: "headline", value: "No fillers. No shortcuts. Just skincare that works." }
          }
        ]
      },
      {
        label: "Welcoming hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The hero is pretty but a little cold — I need to feel welcomed in immediately.",
            insight: "A first-time visitor decides within seconds whether a brand is for them — warmth in the hero image lowers that barrier.",
            suggestion: "Swap to a brighter, more approachable hero image for first-time visitors.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          },
          {
            reaction: "I don't know this brand yet — the imagery should feel open, not intimidating.",
            insight: "Editorial perfection can read as exclusivity to someone new — a warmer image signals 'you belong here'.",
            suggestion: "Use a warmer lifestyle hero that invites exploration rather than perfection.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          },
          {
            reaction: "Something about this hero makes me hesitate — it feels too editorial for a first visit.",
            insight: "A friendlier hero image reduces the perceived risk of exploring an unfamiliar brand.",
            suggestion: "Switch to a friendlier hero image that lowers the barrier to browsing.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          }
        ]
      }
    ],

    "Skeptic": [
      {
        label: "Proof-based headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "'Elevate your skin care ritual' — every brand says something like that.",
            insight: "A skeptic filters out aspiration instantly — a specific, verifiable claim is the only thing that earns a second look.",
            suggestion: "Replace the aspiration with a specific, verifiable claim.",
            apply: { type: "headline", value: "97% natural origin ingredients. Tested over 12 weeks." }
          },
          {
            reaction: "I need a reason to believe this is different before I read anything else.",
            insight: "For a skeptic, credibility must appear above the fold — a proof point is more persuasive than any lifestyle statement.",
            suggestion: "Lead with the most credible proof point above the fold.",
            apply: { type: "headline", value: "Formulated without the 14 most common skin irritants." }
          },
          {
            reaction: "Beautiful claim. Where's the evidence?",
            insight: "A skeptic responds to methodology, not outcomes — showing how you know something is more convincing than what you claim.",
            suggestion: "State the methodology, not just the outcome.",
            apply: { type: "headline", value: "Clinically tested. Independently verified. Nothing hidden." }
          }
        ]
      },
      {
        label: "Hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "A model's face tells me nothing about what this product actually does.",
            insight: "Skeptics read imagery as evidence — a product-forward shot signals confidence in the formula itself.",
            suggestion: "Swap to a product-forward image that shows formula and texture.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          },
          {
            reaction: "I want to see the actual product, not just a lifestyle shot.",
            insight: "Putting the formula front and center communicates that the brand has nothing to hide.",
            suggestion: "Use a hero that puts the formula front and center.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          },
          {
            reaction: "The current image is all feeling and no information.",
            insight: "A skeptic needs something to interrogate — lifestyle imagery gives them nothing to hold the brand accountable to.",
            suggestion: "Switch to a more product-direct hero image.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          }
        ]
      }
    ],

    "Self-care seeker": [
      {
        label: "Ritual-centred headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "I'm investing in a moment for myself — the headline doesn't speak to that yet.",
            insight: "A self-care seeker is buying an experience, not just a product — the headline needs to reflect the personal ritual, not the formula.",
            suggestion: "Centre the personal ritual experience over the product.",
            apply: { type: "headline", value: "Your skin. Your ritual. Your five minutes." }
          },
          {
            reaction: "I want to feel like this brand understands what self-care actually means.",
            insight: "Leading with feeling over formula signals that the brand shares the buyer's values, not just their skincare goals.",
            suggestion: "Lead with the feeling, not the formula.",
            apply: { type: "headline", value: "The part of your morning you'll actually look forward to." }
          },
          {
            reaction: "The current headline is about the product. I want it to be about me.",
            insight: "A self-care seeker places themselves at the center of their routine — the headline should meet them there.",
            suggestion: "Reframe around the moment of use rather than the product itself.",
            apply: { type: "headline", value: "Made for the five minutes that are just yours." }
          }
        ]
      },
      {
        label: "Ritual hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The hero looks polished but not calming — I want to feel the ritual before I read.",
            insight: "For a self-care seeker, the hero image should do emotional work — a softer, more intimate shot sets the ritual mood before a word is read.",
            suggestion: "Swap to a softer, more intimate hero image that evokes a self-care moment.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          },
          {
            reaction: "I need imagery that feels like quiet time, not a campaign shoot.",
            insight: "Natural light and tactile surfaces signal authenticity to someone who values the ritual over the aesthetic.",
            suggestion: "Use a hero with natural light and tactile surfaces that match how I'd use this.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          },
          {
            reaction: "The current image is beautiful but doesn't pull me into my own routine.",
            insight: "A warmer, more intimate hero image helps a self-care seeker project themselves into the ritual.",
            suggestion: "Switch to a warmer lifestyle hero centred on the ritual experience.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          }
        ]
      }
    ],

    "Luxury shopper": [
      {
        label: "Restrained headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "The headline feels slightly wordy for where this price point wants to sit.",
            insight: "Luxury positioning communicates confidence through restraint — a shorter headline signals the brand doesn't need to explain itself.",
            suggestion: "Shorten to a more restrained, confident statement.",
            apply: { type: "headline", value: "Elevate your ritual." }
          },
          {
            reaction: "Luxury brands don't need to explain themselves — this headline over-explains.",
            insight: "One declarative line without qualification is the hallmark of a brand that knows its worth.",
            suggestion: "Strip back to the single most confident line.",
            apply: { type: "headline", value: "Skin care, reconsidered." }
          },
          {
            reaction: "The copy is doing too much work — confident luxury says less.",
            insight: "A luxury shopper reads verbosity as insecurity — the headline should make one statement and stop.",
            suggestion: "One short declarative line, nothing else.",
            apply: { type: "headline", value: "Your skin deserves better." }
          }
        ]
      },
      {
        label: "Editorial hero image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The hero is pleasant but feels mass-market — not where this price point should sit.",
            insight: "A luxury shopper reads the hero image as the brand's self-assessment — it needs to match the premium they expect.",
            suggestion: "Swap to a more restrained, editorial hero image with confident negative space.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          },
          {
            reaction: "Luxury brands let the image do the talking — this one is trying too hard.",
            insight: "A quieter hero signals taste and confidence — less promotion, more presence.",
            suggestion: "Use a quieter hero shot with a premium, less promotional feel.",
            apply: { type: "image", value: "img/image-banner/header-2.jpg" }
          },
          {
            reaction: "I want fewer visual cues and more craft — the current hero feels busy.",
            insight: "Restraint in imagery communicates the same thing as restraint in copy — confidence in the product itself.",
            suggestion: "Switch to a minimal hero image that signals quality through restraint.",
            apply: { type: "image", value: "img/image-banner/header-3.jpg" }
          }
        ]
      }
    ]
  },

  "featured-collection": {

    "Gift buyer": [
      {
        label: "Collection headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "Nothing here signals this is where gift buyers should look.",
            insight: "A gift buyer is scanning for permission to shop here — a section headline that speaks to giving removes the guesswork.",
            suggestion: "Rename the section to speak directly to gift shoppers.",
            apply: { type: "headline", value: "Gifts they'll love. Sets they'll keep." }
          },
          {
            reaction: "I'm shopping for someone else and this section feels like it's not for me.",
            insight: "Framing the collection as curated for giving instantly makes a gift buyer feel seen.",
            suggestion: "Frame the collection as curated for giving.",
            apply: { type: "headline", value: "Curated to give." }
          },
          {
            reaction: "I can't tell if these products come as sets or individually.",
            insight: "A gift buyer needs to know the format before they can commit — clarifying sets vs. individual items removes a key friction point.",
            suggestion: "Clarify the gifting format in the headline.",
            apply: { type: "headline", value: "Ready-to-gift sets. No wrapping required." }
          }
        ]
      },
      {
        label: "Gift-ready collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The product grid looks polished but nothing signals these are easy to give.",
            insight: "Gift buyers read imagery for occasion cues — if nothing in the visual says 'gift', the section loses them.",
            suggestion: "Swap to a collection image set that reads as gift-ready and occasion-friendly.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "I want to see products presented the way I'd imagine wrapping them.",
            insight: "Imagery curated for gifting helps buyers visualize the purchase outcome — the give, not just the get.",
            suggestion: "Use imagery that feels curated for gifting rather than everyday use.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Beautiful products, but the photos don't help me picture giving them.",
            insight: "A warmer collection mood makes the emotional leap from browsing to buying easier for gift shoppers.",
            suggestion: "Switch to a warmer collection image set with a gifting mood.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "First-time visitor": [
      {
        label: "Starting point headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "I don't know this brand well enough to know which collection is right for me.",
            insight: "A first-time visitor needs a guided entry point — telling them where to start removes the paralysis of choice.",
            suggestion: "Guide a new visitor toward where to begin.",
            apply: { type: "headline", value: "New to Mattie Studio? Start here." }
          },
          {
            reaction: "Too many options for someone who's just arrived — I need direction.",
            insight: "Positioning this as the recommended starting point does the curation work that a new visitor can't do for themselves.",
            suggestion: "Position this as the recommended entry point.",
            apply: { type: "headline", value: "The collection most people start with." }
          },
          {
            reaction: "I want to know what's most popular before I commit to anything.",
            insight: "Social proof in a headline replaces brand familiarity for someone who hasn't built trust with Mattie Studio yet.",
            suggestion: "Frame the collection around social proof.",
            apply: { type: "headline", value: "Our most loved products, in one place." }
          }
        ]
      },
      {
        label: "Approachable collection imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The current images feel curated for people who already know the brand.",
            insight: "Imagery that feels accessible rather than aspirational reduces the intimidation factor for a first-time visitor.",
            suggestion: "Use a warmer, more accessible product collection image set.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "I need the visuals to invite me in, not impress me from a distance.",
            insight: "A softer, more welcoming product presentation signals openness rather than exclusivity.",
            suggestion: "Swap to imagery with a softer, more welcoming product presentation.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Beautiful but slightly intimidating — I want to feel this is for me.",
            insight: "A collection image that feels open and easy to relate to lowers the barrier for someone still deciding whether to trust the brand.",
            suggestion: "Switch to a collection image set that feels open and easy to explore.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "Skeptic": [
      {
        label: "Proof-led headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to know what's most popular — social validation matters here.",
            insight: "A skeptic trusts collective behavior over brand claims — leading with bestseller status is more credible than curation language.",
            suggestion: "Lead with bestseller status to signal proven products.",
            apply: { type: "headline", value: "Our most repurchased products." }
          },
          {
            reaction: "A generic collection name doesn't tell me why these products are worth it.",
            insight: "Signaling verified customer preference shifts authority from brand to buyers — which a skeptic trusts more.",
            suggestion: "Use the headline to signal verified customer preference.",
            apply: { type: "headline", value: "The products customers reorder most." }
          },
          {
            reaction: "I need evidence these are the right products before I go deeper.",
            insight: "A results-and-popularity frame replaces aspiration with accountability — exactly what a skeptic responds to.",
            suggestion: "Frame the collection around results and popularity.",
            apply: { type: "headline", value: "Tried, tested, repurchased." }
          }
        ]
      },
      {
        label: "Product-forward collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "Lifestyle grid shots don't tell me what I'm actually buying.",
            insight: "A skeptic wants to evaluate the product, not the aesthetic — clearer product focus gives them something to assess.",
            suggestion: "Swap to a collection image set with clearer product focus.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "I want to see the products themselves, not just styled surfaces.",
            insight: "Putting formulas and packaging front and center signals the brand is confident in the product itself.",
            suggestion: "Use imagery that puts formulas and packaging front and center.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "The photos are pretty but feel like marketing — show me the goods.",
            insight: "A more direct product-led presentation removes the feeling of being sold to — which is exactly what earns a skeptic's attention.",
            suggestion: "Switch to a more direct, product-led collection image set.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "Self-care seeker": [
      {
        label: "Ritual-framed headline",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to shop by ritual — morning, evening — not just by product.",
            insight: "A self-care seeker organizes their world around routines — a headline that speaks to ritual rather than product type meets them where they think.",
            suggestion: "Reframe the collection around routine building.",
            apply: { type: "headline", value: "Build your ritual." }
          },
          {
            reaction: "The collection feels product-focused when I'm experience-focused.",
            insight: "Naming the emotional outcome rather than the category shifts the frame from shopping to self-investment.",
            suggestion: "Name the emotional outcome rather than the product category.",
            apply: { type: "headline", value: "Everything your routine has been missing." }
          },
          {
            reaction: "I want to feel like this collection was put together for someone like me.",
            insight: "Framing the collection as a personal curation makes a self-care seeker feel understood rather than marketed to.",
            suggestion: "Frame it as a personal curation, not a product listing.",
            apply: { type: "headline", value: "Your ritual, fully stocked." }
          }
        ]
      },
      {
        label: "Ritual mood collection image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "I want to feel the mood of using these products, not just see them on a shelf.",
            insight: "A self-care seeker is shopping for a feeling — collection imagery that evokes a moment of use does more selling than a product grid.",
            suggestion: "Swap to a collection image set that evokes a moment of self-care.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "The grid feels transactional — I'm shopping for a feeling, not a SKU.",
            insight: "Warmer imagery that suggests a morning or evening ritual reframes the collection as a lifestyle choice rather than a purchase.",
            suggestion: "Use warmer imagery that suggests a morning or evening ritual.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Beautiful products, but the photos don't transport me into the experience.",
            insight: "Softer light and intentional mood in the imagery help a self-care seeker project themselves into the product experience.",
            suggestion: "Switch to imagery with softer light and a more intentional mood.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "Luxury shopper": [
      {
        label: "Elevated collection label",
        tags: ["copy"],
        variants: [
          {
            reaction: "'Featured Collection' is a generic label — it doesn't signal exclusivity.",
            insight: "A luxury shopper reads generic labels as mass-market signifiers — an editorial title signals taste and intentional curation.",
            suggestion: "Replace with a more editorial, curated-sounding title.",
            apply: { type: "headline", value: "The edit." }
          },
          {
            reaction: "This header reads like a template, not a considered curation.",
            insight: "A single restrained word communicates confidence — it implies the collection needs no further explanation.",
            suggestion: "Use a single restrained word that signals careful selection.",
            apply: { type: "headline", value: "Selected." }
          },
          {
            reaction: "A luxury brand shouldn't use the word 'featured' — it sounds algorithmic.",
            insight: "Language that implies handpicking and limited availability signals the kind of exclusivity a luxury shopper expects.",
            suggestion: "Rename to signal a handpicked, limited offering.",
            apply: { type: "headline", value: "The considered selection." }
          }
        ]
      },
      {
        label: "Editorial collection imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The collection imagery needs to match the premium positioning I expect.",
            insight: "For a luxury shopper, the imagery is the primary quality signal — it must match the price point before they read a word.",
            suggestion: "Switch to the most refined, editorial product image set.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "These photos read as catalog, not curation — luxury is in the edit.",
            insight: "A tighter, more considered image set signals that the brand has strong aesthetic judgment — which a luxury shopper is paying for.",
            suggestion: "Use a tighter, more considered collection image set.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "At this price point, the visuals should feel intentional, not template-driven.",
            insight: "Restrained, high-end product presentation communicates that every visual decision was made deliberately.",
            suggestion: "Swap to imagery with a more restrained, high-end product presentation.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ]
  },

  "image-with-text": {

    "Gift buyer": [
      {
        label: "Body copy for gift context",
        tags: ["copy"],
        variants: [
          {
            reaction: "The copy is written for someone buying for themselves — nothing helps me justify this as a gift.",
            insight: "A gift buyer needs the body copy to do the imaginative work of framing the purchase as a gift — without it, they're left to figure that out alone.",
            suggestion: "Rewrite body text to speak to the gift buyer's confidence.",
            apply: { type: "subheadline", value: "Whether it's a birthday, a thank you, or just because — our sets arrive gift-wrapped and ready to give. No extra step needed." }
          },
          {
            reaction: "I need the copy to reassure me that this gift will land well.",
            insight: "Addressing uncertainty about the recipient directly — with returns language and product versatility — removes the biggest blocker for a gift buyer.",
            suggestion: "Address the gift buyer's uncertainty about the recipient's reaction.",
            apply: { type: "subheadline", value: "Not sure what they'd prefer? Our sets are chosen to work for every skin type. And if it's not right, returns are always free." }
          },
          {
            reaction: "I want to picture giving this, not using it myself.",
            insight: "Writing from the giving perspective throughout — not the using perspective — keeps a gift buyer in their actual decision context.",
            suggestion: "Write the body copy from the giving perspective throughout.",
            apply: { type: "subheadline", value: "The kind of gift people keep and talk about. Beautifully packaged, thoughtfully made — and one they'll come back to every morning." }
          }
        ]
      },
      {
        label: "Gift occasion imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The product images look beautiful for personal use but don't read as gift-oriented.",
            insight: "Occasion-signaling imagery removes the translation work a gift buyer has to do when visualizing the purchase as a gift.",
            suggestion: "Switch to an image that better signals gifting and occasion.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "I want to see products presented the way I'd imagine wrapping them.",
            insight: "Imagery that evokes the giving moment helps gift buyers feel confident they've found the right thing.",
            suggestion: "Use imagery with a warmer, more celebratory presentation.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Beautiful, but it doesn't help me picture this in someone else's hands.",
            insight: "A lifestyle image that implies receiving rather than owning closes the gap between browsing and buying for a gift shopper.",
            suggestion: "Switch to imagery that frames the product as something worth giving.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "First-time visitor": [
      {
        label: "Brand origin story",
        tags: ["copy"],
        variants: [
          {
            reaction: "I still don't know who made this or why — this section could answer that.",
            insight: "A first-time visitor's trust is built through story, not specs — brand origin copy is often the moment they decide whether this brand is for them.",
            suggestion: "Use the body copy to introduce the brand story.",
            apply: { type: "subheadline", value: "Mattie Studio started as a single formula in a small kitchen. Every product is still made in small batches — never scaled, never compromised." }
          },
          {
            reaction: "A beautiful brand but I have no context for it yet.",
            insight: "Opening with the founding insight — the reason the brand exists — gives a new visitor a specific narrative hook to hold onto.",
            suggestion: "Open with the founding insight that makes this brand exist.",
            apply: { type: "subheadline", value: "We started Mattie Studio because we couldn't find a skincare brand we actually trusted. So we made one." }
          },
          {
            reaction: "I want to understand the people behind this before I buy from them.",
            insight: "A personal, direct brand voice in the body copy replaces the need for a long backstory — it just needs to feel human.",
            suggestion: "Make the brand voice personal and direct.",
            apply: { type: "subheadline", value: "Everything here is made the way we'd want our own skincare made. Small batches, honest ingredients, no shortcuts." }
          }
        ]
      },
      {
        label: "Welcoming section image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The image feels polished but not especially inviting to someone new.",
            insight: "A warmer section image signals welcome to a first-time visitor — it reduces the sense that this brand is curated for insiders.",
            suggestion: "Swap to a warmer image that helps a first-time visitor feel welcome.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "I want to see people like me in this brand, not just a perfect still life.",
            insight: "A less staged, more approachable section image makes a new visitor feel the brand is accessible rather than aspirational.",
            suggestion: "Use a more approachable, less staged section image.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Beautiful, but it doesn't help me understand what this brand is about visually.",
            insight: "Open, relatable imagery gives a first-time visitor a visual anchor for what the brand stands for.",
            suggestion: "Switch to imagery that feels open and easy to relate to.",
            apply: { type: "image", optionIndex: 1 }
          }
        ]
      }
    ],

    "Skeptic": [
      {
        label: "Ingredient transparency",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to know exactly what's in this before I trust the brand.",
            insight: "A skeptic's trust is built through specificity — naming what the brand won't use is more credible than listing what it does.",
            suggestion: "Replace body copy with specific ingredient and process transparency.",
            apply: { type: "subheadline", value: "Every formula starts with a list of what we won't use. No fillers, no fragrance masking, no unnecessary additives." }
          },
          {
            reaction: "Claims without evidence don't move me — I need specifics.",
            insight: "Listing each ingredient's purpose communicates a level of intentionality that generic 'clean beauty' language cannot.",
            suggestion: "Lead with the most credible process detail in the body copy.",
            apply: { type: "subheadline", value: "Each ingredient is listed with its purpose. If it's in the formula, there's a reason for it. If there isn't, it's not there." }
          },
          {
            reaction: "I want to understand the production process, not just the outcome.",
            insight: "Describing the methodology — batch size, testing cadence — gives a skeptic something concrete to evaluate rather than a claim to accept.",
            suggestion: "Describe the methodology that makes the quality claim credible.",
            apply: { type: "subheadline", value: "Made in batches of under 500 units. Every batch tested before it ships. No exceptions." }
          }
        ]
      },
      {
        label: "Product-forward section image",
        tags: ["imagery"],
        variants: [
          {
            reaction: "Lifestyle imagery doesn't answer my questions — I want to see the product itself.",
            insight: "A skeptic reads product-forward imagery as a signal of confidence — the brand showing rather than styling.",
            suggestion: "Switch to an image set that shows the product and formula more directly.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "A mood shot tells me nothing about what's in the bottle.",
            insight: "Clearer product detail in imagery gives a skeptic something tangible to assess before they decide to read further.",
            suggestion: "Use imagery with clearer product detail and less styling.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "I'm not sold on vibes — show me what I'd actually be buying.",
            insight: "A more product-direct section image signals that the brand is confident enough in the product to let it speak without styling.",
            suggestion: "Swap to a more product-direct section image.",
            apply: { type: "image", optionIndex: 2 }
          }
        ]
      }
    ],

    "Self-care seeker": [
      {
        label: "Sensory experience copy",
        tags: ["copy"],
        variants: [
          {
            reaction: "The copy is informative but not evocative — I want to feel something reading it.",
            insight: "A self-care seeker is motivated by sensory anticipation — copy that describes the experience of use is more persuasive than copy that describes the product.",
            suggestion: "Rewrite to describe the sensory experience of using the product.",
            apply: { type: "subheadline", value: "The kind of routine you actually look forward to. Lightweight textures, considered scents, and formulas that absorb before you've finished your coffee." }
          },
          {
            reaction: "I want this section to make me feel the ritual before I've bought anything.",
            insight: "Writing the body copy as an evocation of the morning experience lets the buyer feel ownership of the ritual before they've committed to a purchase.",
            suggestion: "Write the body copy as an evocation of the morning experience.",
            apply: { type: "subheadline", value: "Five minutes. Warm water. Something that smells exactly right. This is what we make that for." }
          },
          {
            reaction: "The product sounds good but the copy doesn't make me feel it.",
            insight: "Sensory language throughout — texture, temperature, scent — triggers the kind of embodied anticipation that converts a self-care seeker.",
            suggestion: "Use sensory language throughout the body copy.",
            apply: { type: "subheadline", value: "Cool to the touch. Warm on the skin. A texture that disappears and leaves only what it promised." }
          }
        ]
      },
      {
        label: "Ritual mood imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction: "I want to feel the ritual in the image, not just read about it in the copy.",
            insight: "For a self-care seeker, the section image should do the emotional lifting — a calm, intentional shot sets the ritual tone before any copy is read.",
            suggestion: "Swap to imagery that evokes a calm, intentional self-care moment.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "The photo is pretty but doesn't pull me into the experience.",
            insight: "Softer imagery with light and texture that suggest daily use helps a self-care seeker project themselves into the routine.",
            suggestion: "Use a softer image with light and texture that suggest daily use.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "This section should make me want the moment, not just the product.",
            insight: "An atmospheric, ritual-focused image makes the purchase feel like an investment in a daily experience, not just a skincare product.",
            suggestion: "Switch to a more atmospheric, ritual-focused section image.",
            apply: { type: "image", optionIndex: 1 }
          }
        ]
      }
    ],

    "Luxury shopper": [
      {
        label: "Restrained craft copy",
        tags: ["copy"],
        variants: [
          {
            reaction: "The body copy is slightly too long — luxury positioning uses restraint.",
            insight: "A luxury shopper reads verbosity as a lack of confidence — cutting to fewer, more considered words signals the brand knows its worth.",
            suggestion: "Cut to a shorter, more considered line that trusts the reader.",
            apply: { type: "subheadline", value: "Made slowly, on purpose. Every ingredient chosen for a reason. Nothing added for appearance." }
          },
          {
            reaction: "Too many words for a brand at this price point.",
            insight: "Stripping back to a single craft statement communicates that the brand trusts its product to do the convincing.",
            suggestion: "Strip back to the single most important craft statement.",
            apply: { type: "subheadline", value: "The formula took two years. The results speak without explanation." }
          },
          {
            reaction: "Premium brands let the product speak — the copy here over-explains.",
            insight: "One restrained sentence that implies quality without stating it mirrors the confidence a luxury shopper expects from a premium brand.",
            suggestion: "Use one restrained sentence that implies quality without stating it.",
            apply: { type: "subheadline", value: "Small batch. No compromise. That's the whole story." }
          }
        ]
      },
      {
        label: "Refined section imagery",
        tags: ["imagery"],
        variants: [
          {
            reaction: "The imagery needs to match the premium feel — I want clean, minimal, considered.",
            insight: "For a luxury shopper, the section image is a quality proxy — it needs to match the level of intentionality they expect from the product.",
            suggestion: "Switch to the image set with the most editorial, restrained aesthetic.",
            apply: { type: "image", optionIndex: 2 }
          },
          {
            reaction: "Too much visual noise for a brand at this price point.",
            insight: "A simpler, more confident section image communicates that the brand has strong enough aesthetic judgment to edit aggressively.",
            suggestion: "Use a simpler, more confident section image with fewer distractions.",
            apply: { type: "image", optionIndex: 1 }
          },
          {
            reaction: "Luxury here should feel quiet — this photo competes with itself.",
            insight: "Minimal, high-end product presentation signals restraint — the most powerful luxury signal of all.",
            suggestion: "Swap to a more minimal, high-end product presentation.",
            apply: { type: "image", optionIndex: 3 }
          }
        ]
      }
    ]
  },

  "testimonials": {

    "Gift buyer": [
      {
        label: "Gift-specific reviews",
        tags: ["copy"],
        variants: [
          {
            reaction: "None of these reviews mention gifting — I want proof from people in my situation.",
            insight: "Social proof from gift buyers is the most direct validation for a shopper in the same position — it shows this purchase has worked for someone like them.",
            suggestion: "Surface testimonials from gift buyers specifically.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "Bought this for my sister's birthday. She called two days later asking where I got it.", author: "Cleo R." },
              { rating: 5, quote: "I've given Mattie Studio sets as gifts four times now. It never misses.", author: "Daniel T." },
              { rating: 5, quote: "The packaging alone made it feel expensive. The product backs it up completely.", author: "Nina W." }
            ]}
          },
          {
            reaction: "I need to see that other people have successfully given this as a gift.",
            insight: "Reviews that mention the recipient's reaction give a gift buyer concrete evidence that the purchase will land well.",
            suggestion: "Use testimonials that specifically mention the recipient's reaction.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "My mum still talks about the birthday gift I got her last year. This was it.", author: "Priya K." },
              { rating: 5, quote: "Got this for a colleague leaving the team. She came back to ask the brand name.", author: "James M." },
              { rating: 5, quote: "Gave it as a thank you gift. They reordered for themselves within a week.", author: "Sara L." }
            ]}
          },
          {
            reaction: "Reviews about personal use don't help me decide if this is a good gift.",
            insight: "Occasion-specific testimonials validate the purchase context directly — they answer the gift buyer's question before they have to ask it.",
            suggestion: "Replace with occasion-specific testimonials.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "Perfect for anyone who's hard to buy for. Beautiful packaging, incredible product.", author: "Amara S." },
              { rating: 5, quote: "I ordered this as a 'just because' gift. It made a much bigger impression than I expected.", author: "Tom W." },
              { rating: 5, quote: "Ordered it as a gift, ended up buying one for myself too.", author: "Mei T." }
            ]}
          }
        ]
      }
    ],

    "First-time visitor": [
      {
        label: "First-purchase validation",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to hear from people who were also new to the brand — not just loyal fans.",
            insight: "First-purchase testimonials are uniquely persuasive for new visitors — they mirror the exact decision the visitor is facing right now.",
            suggestion: "Surface testimonials that reflect a first-time buyer's experience.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I was hesitant to try a brand I hadn't heard of. Two weeks in — completely converted.", author: "Mara S." },
              { rating: 5, quote: "Ordered on a recommendation. It's now the only skincare I repurchase without thinking.", author: "James O." },
              { rating: 5, quote: "Wasn't sure at first. The results after two weeks changed my mind completely.", author: "Yuki T." }
            ]}
          },
          {
            reaction: "I need to know the first purchase experience is good, not just the product.",
            insight: "Reviews that mention the discovery and onboarding experience validate the full first-purchase journey, not just the product quality.",
            suggestion: "Use reviews that mention the discovery and onboarding experience.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "Found this through a friend. The packaging, the smell, the texture — nothing disappointed.", author: "Claire B." },
              { rating: 5, quote: "First order arrived beautifully. I knew before I even tried it that this brand was different.", author: "Ravi D." },
              { rating: 5, quote: "Tried one product to test the brand. Now I own five. That says everything.", author: "Lea F." }
            ]}
          },
          {
            reaction: "Long-term customer reviews don't tell me what to expect on my first order.",
            insight: "Reviews specifically about the initial experience answer the new visitor's most immediate question: what happens when I order for the first time?",
            suggestion: "Feature reviews specifically about the initial experience.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "Bought my first one sceptically. Reordered before the bottle was empty.", author: "Sophie M." },
              { rating: 5, quote: "The first use was enough. I sent the link to three friends the same day.", author: "Nia A." },
              { rating: 5, quote: "I almost didn't order. So glad I did. My skin has never felt this good.", author: "Paulo R." }
            ]}
          }
        ]
      }
    ],

    "Skeptic": [
      {
        label: "Results-focused reviews",
        tags: ["copy"],
        variants: [
          {
            reaction: "Vague reviews don't help me — I want specific, measurable outcomes.",
            insight: "A skeptic filters out emotional language instantly — concrete results with specifics are the only reviews that register as credible.",
            suggestion: "Replace with testimonials that cite concrete results.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "My dermatologist asked what I'd changed in my routine. I told her. She wasn't surprised.", author: "Amara S." },
              { rating: 5, quote: "Sceptical about the price. Four repurchases later — I completely understand it now.", author: "Marcus O." },
              { rating: 5, quote: "Finally a brand that doesn't over-promise. It just quietly does what it says.", author: "Lea B." }
            ]}
          },
          {
            reaction: "I want before and after — not just 'I loved it'.",
            insight: "Reviews that name a specific skin concern and describe its resolution give a skeptic the evidence chain they need to trust the product.",
            suggestion: "Surface reviews that describe a specific skin concern that was resolved.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I'd had uneven skin tone for years. Six weeks of this and the difference is visible in photos.", author: "Diana C." },
              { rating: 5, quote: "Everything else I tried either irritated my skin or did nothing. This did neither — it just worked.", author: "Felix H." },
              { rating: 5, quote: "I tracked my skin weekly for a month. The improvement by week three was undeniable.", author: "Nora P." }
            ]}
          },
          {
            reaction: "Five-star reviews without specifics read as fake to me.",
            insight: "Specific details — duration, frequency, skin type — are the authenticity signals a skeptic uses to distinguish real reviews from planted ones.",
            suggestion: "Use reviews with specific details that signal authenticity.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I've been using it twice daily for four months. No breakouts, no irritation, noticeably clearer.", author: "Tom K." },
              { rating: 5, quote: "Combination skin, easily congested. This is the first product that hasn't made it worse.", author: "Ana V." },
              { rating: 5, quote: "I read the ingredient list before buying. Everything checks out. Results confirmed my research.", author: "Sam J." }
            ]}
          }
        ]
      }
    ],

    "Self-care seeker": [
      {
        label: "Ritual experience reviews",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to hear from people who made this part of a meaningful routine.",
            insight: "A self-care seeker is validating a lifestyle choice, not just a product purchase — ritual-focused reviews confirm that this brand fits that intention.",
            suggestion: "Feature reviews about the ritual experience, not just the results.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "This is the one part of my morning I genuinely look forward to now.", author: "Jamie K." },
              { rating: 5, quote: "I bought it for my skin. I kept it for how it makes me feel getting ready.", author: "Priya M." },
              { rating: 5, quote: "My five-minute routine became my favourite part of the day.", author: "Sarah L." }
            ]}
          },
          {
            reaction: "I want to feel like buying this changes my mornings, not just my skin.",
            insight: "Reviews that describe a lifestyle shift validate the self-care seeker's deeper motivation — the ritual, not the result.",
            suggestion: "Surface reviews that describe a lifestyle shift, not just a product benefit.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I used to skip my skincare most mornings. Now I make time for it. The ritual is part of the point.", author: "Cora B." },
              { rating: 5, quote: "It's hard to explain but using this feels intentional. Like I'm being kind to myself.", author: "Ella S." },
              { rating: 5, quote: "My morning is different now. Slower. Better. This is a big part of why.", author: "Hana M." }
            ]}
          },
          {
            reaction: "Product reviews feel transactional — I want emotional resonance.",
            insight: "Reviews that describe what the product means to the person — not what it does to their skin — speak directly to a self-care seeker's values.",
            suggestion: "Use reviews that describe what the product means to the person.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "After a really hard year, this felt like reclaiming something for myself.", author: "Jo D." },
              { rating: 5, quote: "Small luxury, real impact. This is what I mean when I say self-care actually works.", author: "Ren A." },
              { rating: 5, quote: "I bought this for myself without justifying it to anyone. Best decision I made that month.", author: "Zoe C." }
            ]}
          }
        ]
      }
    ],

    "Luxury shopper": [
      {
        label: "Discerning buyer reviews",
        tags: ["copy"],
        variants: [
          {
            reaction: "I want to hear from people with high standards — not just general positive reviews.",
            insight: "Reviews from buyers who compare against premium alternatives give a luxury shopper the peer validation they actually trust.",
            suggestion: "Surface reviews from buyers who compare against premium alternatives.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I've used La Mer, Sisley, and Augustinus Bader. Mattie Studio holds its own.", author: "Claire D." },
              { rating: 5, quote: "Worth every penny. The texture and absorption are unlike anything at this price point.", author: "Ravi M." },
              { rating: 5, quote: "I gifted this to myself after a difficult year. It felt right. It still does.", author: "Sophie W." }
            ]}
          },
          {
            reaction: "Generic five-star reviews don't tell me if this meets a high standard.",
            insight: "Reviews that reference specific quality expectations signal to a luxury shopper that this brand has been evaluated by someone with similar standards.",
            suggestion: "Use reviews that reference the buyer's specific quality expectations.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I only buy products I'd be comfortable recommending to someone with discerning taste. This qualifies.", author: "Marcus L." },
              { rating: 5, quote: "The finish, the scent, the packaging — nothing feels like a compromise.", author: "Ines V." },
              { rating: 5, quote: "I've tried most things at this price point. This is the one I stopped looking after.", author: "Thomas B." }
            ]}
          },
          {
            reaction: "I want reviews that tell me this is worth the premium without overselling.",
            insight: "Understated, confident reviews mirror the luxury shopper's own communication style — they imply quality rather than announce it.",
            suggestion: "Feature understated, confident reviews that imply quality without gushing.",
            apply: { type: "testimonials", value: [
              { rating: 5, quote: "I don't write reviews. I'm writing this one.", author: "N. Ashworth" },
              { rating: 5, quote: "Quietly the best thing in my bathroom.", author: "J. Moreau" },
              { rating: 5, quote: "I expected good. I got better than that.", author: "P. Chen" }
            ]}
          }
        ]
      }
    ]
  },

  "footer": {

    "Gift buyer": [
      {
        label: "Gifting tagline",
        tags: ["copy"],
        variants: [
          {
            reaction: "The tagline speaks to personal use — nothing signals this is also a great gift brand.",
            insight: "The footer is the last impression — a tagline that acknowledges gifting reinforces the purchase decision for a buyer who's been thinking about giving.",
            suggestion: "Update to acknowledge the gifting occasion.",
            apply: { type: "subheadline", value: "Small-batch skincare. Made to give and keep." }
          },
          {
            reaction: "I've just been thinking about gifts and the footer doesn't reinforce that.",
            insight: "Closing the page with a gifting-friendly brand statement keeps the gift buyer in their decision context all the way to checkout.",
            suggestion: "Close the page with a gifting-friendly brand statement.",
            apply: { type: "subheadline", value: "Skincare worth giving. And worth keeping for yourself." }
          },
          {
            reaction: "The closing brand statement misses the occasion I'm shopping for.",
            insight: "Acknowledging both the gift buyer and the recipient in the tagline validates the dual nature of the purchase — meaningful to give and to receive.",
            suggestion: "Acknowledge both the gift buyer and the recipient in the tagline.",
            apply: { type: "subheadline", value: "Made with intention. Given with meaning." }
          }
        ]
      }
    ],

    "First-time visitor": [
      {
        label: "Welcoming tagline",
        tags: ["copy"],
        variants: [
          {
            reaction: "The tagline is confident but slightly insider — I want the brand to feel welcoming.",
            insight: "The footer tagline is the last word a new visitor reads — an open, inviting closing line increases the chance they'll return.",
            suggestion: "Update to feel more open and inviting to someone new.",
            apply: { type: "subheadline", value: "Small-batch skincare. Made for the curious." }
          },
          {
            reaction: "I've just discovered this brand and the closing line doesn't invite me in.",
            insight: "A closing line that encourages further exploration gives a first-time visitor a reason to stay or come back rather than moving on.",
            suggestion: "End the page with a line that encourages further exploration.",
            apply: { type: "subheadline", value: "New here? You're in the right place." }
          },
          {
            reaction: "The tagline wraps up the brand for existing customers, not for me.",
            insight: "A closing statement that speaks to someone still deciding acknowledges their hesitation without pressure — which builds trust.",
            suggestion: "Make the closing statement speak to someone still deciding.",
            apply: { type: "subheadline", value: "Skincare made to earn your trust. Starting with the first bottle." }
          }
        ]
      }
    ],

    "Skeptic": [
      {
        label: "Direct honesty tagline",
        tags: ["copy"],
        variants: [
          {
            reaction: "'Made with intention' is vague — every brand says that.",
            insight: "A skeptic's last impression should be a specific, honest claim — not another aspiration they have to take on faith.",
            suggestion: "Replace with a tagline that makes a specific, honest claim.",
            apply: { type: "subheadline", value: "Small-batch skincare. No fillers. No shortcuts." }
          },
          {
            reaction: "The closing statement is poetic but says nothing verifiable.",
            insight: "Ending on a direct product truth rather than a brand aspiration gives a skeptic something concrete to take away from the page.",
            suggestion: "End with a direct product truth rather than a brand aspiration.",
            apply: { type: "subheadline", value: "We list every ingredient. We explain every choice. That's the whole pitch." }
          },
          {
            reaction: "I want the brand to close on something I can actually hold them to.",
            insight: "A specific, accountable closing statement signals that the brand is comfortable being held to its word — which is exactly what earns a skeptic's trust.",
            suggestion: "Use a specific, accountable closing statement.",
            apply: { type: "subheadline", value: "If it's in the bottle, it's on the label. If it's not on the label, it's not in the bottle." }
          }
        ]
      }
    ],

    "Self-care seeker": [
      {
        label: "Ritual closing statement",
        tags: ["copy"],
        variants: [
          {
            reaction: "The tagline is brand-focused — I want it to speak to my routine.",
            insight: "A closing line that centres the personal experience rather than the brand keeps a self-care seeker in their own story as they leave the page.",
            suggestion: "Update to centre the personal experience.",
            apply: { type: "subheadline", value: "Small-batch skincare. Made for your five minutes." }
          },
          {
            reaction: "I want the closing line to feel like a gentle invitation to slow down.",
            insight: "A tagline that evokes the ritual rather than the product leaves a self-care seeker with the feeling they came to the page to find.",
            suggestion: "End with something that evokes the ritual rather than the product.",
            apply: { type: "subheadline", value: "Your routine deserves this much thought." }
          },
          {
            reaction: "The footer closes on the brand. I want it to close on the feeling.",
            insight: "Landing on the emotional outcome of the ritual as the final word mirrors how a self-care seeker thinks about their morning — feeling first, product second.",
            suggestion: "Use the tagline to land on the emotional outcome of the ritual.",
            apply: { type: "subheadline", value: "Made for the mornings that feel like yours." }
          }
        ]
      }
    ],

    "Luxury shopper": [
      {
        label: "Premium closing line",
        tags: ["copy"],
        variants: [
          {
            reaction: "'Made with intention' reads as indie-brand language, not true luxury.",
            insight: "A luxury shopper's final impression should confirm the premium positioning — a more restrained closing line does that without over-explaining.",
            suggestion: "Replace with a more restrained, confident closing line.",
            apply: { type: "subheadline", value: "Made slowly. On purpose." }
          },
          {
            reaction: "The tagline is pleasant but doesn't signal the premium I expect.",
            insight: "Implying exclusivity through restraint — 'not for everyone' — signals that the brand knows exactly who it's for, which is itself a luxury signal.",
            suggestion: "Use a closing statement that implies exclusivity through restraint.",
            apply: { type: "subheadline", value: "Not for everyone. Made for those who notice." }
          },
          {
            reaction: "A luxury brand's last word should be its most confident.",
            insight: "A single declarative closing line that needs no explanation is the ultimate luxury signal — confidence in its own completeness.",
            suggestion: "Close with a single declarative line that needs no explanation.",
            apply: { type: "subheadline", value: "Nothing wasted. Nothing missing." }
          }
        ]
      }
    ]
  }

};
