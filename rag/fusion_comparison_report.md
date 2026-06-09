# Fusion & Dataset Comparison Report

## How scoring works

- **18 test queries** (same intents; category filters differ per dataset).
- **#1 relevant (max 18)** — For each query, **1** if the **first** result clearly matches the product intent (keyword rubric), **0** otherwise. Summed across all queries.
- **Relevant in top-3 (max 54)** — For each query, count how many of the **3** results are clearly relevant (**0**, **1**, **2**, or **3**). Summed across all queries.
- Both metrics use the **same** relevance rule; they are not weighted differently.
- **Fusion winner** per dataset: higher #1 relevant wins; tie-break on relevant-in-top-3.

## Aggregate scores

| Dataset | Fusion | #1 relevant (/18) | Relevant in top-3 (/54) |
|---------|--------|-------------------------|-------------------------------|
| Indian | RRF | 10 | 32 |
| Indian | DBSF | 10 | 28 |
| McAuley | RRF | 15 | 44 |
| McAuley | DBSF | 14 | 43 |

## Per-query comparison

| Query | Indian RRF #1 | Indian DBSF #1 | McAuley RRF #1 | McAuley DBSF #1 | Notes |
|-------|---------------|----------------|----------------|-----------------|-------|
| wireless earbuds noise cancelling | ASICS Mens Gel-Dedicate 6 Sneaker | Edifier Neobuds Pro True Wireless Stereo Earbuds with A | HISOOS Wireless Earbuds Bluetooth Active Noise Cancelli | HISOOS Wireless Earbuds Bluetooth Active Noise Cancelli | McAuley better |
| bluetooth speaker portable waterproof | Switchon Polyester Printed Free Size Chef's Cooking Kit | Switchon Polyester Printed Free Size Chef's Cooking Kit | JBL GO2 - Waterproof Ultra Portable Bluetooth Speaker - | JBL JR POP - Waterproof portable Bluetooths Speaker Des | McAuley better |
| gaming laptop RTX | RTX Men's Regular Fit Jeans/Men's Slim Fit Stretchable  | DailyObjects leather stainless steel clip and keyring K | Lenovo - Legion 5 - Gaming Laptop - AMD Ryzen 7 5800H - | Lenovo IdeaPad Gaming 3 - (2022) - Essential Gaming Lap | Tie |
| 32 inch smart TV 4K | Samsung 80 cm (32 inches) HD Ready LED Smart TV UA32T47 | TCL 81 cm (32 inches) HD READY Smart Certified Android  | SAMSUNG Electronics UN32M4500A 32-Inch 720p Smart LED T | SAMSUNG Electronics UN32M4500A 32-Inch 720p Smart LED T | Tie |
| men's running shoes lightweight | B-TUF Torpedo Football Shoes/Studs/Boot Lightweight for | DLGJPA Men's Lightweight Quick Drying Aqua Water Shoes  | WXQ Men's Running Shoes Comfortable Lightweight Breatha | WXQ Men's Running Shoes Comfortable Lightweight Breatha | Tie |
| women's winter coat warm | NAINVISH Women Crepe Straight Kurti with Palazzo | adhyah Lab coat for doctors women men doctors coat fema | ICEIVY Womens Wool Socks 5 Pairs Warm Wool Cotton Socks | ICEIVY Womens Wool Socks 5 Pairs Warm Wool Cotton Socks | McAuley better |
| cotton bed sheets king size | F2L Chenille Velvet Abstract 500 TC Heavy Bedsheet for  | Amazon Brand - Solimo Cotton Jewellery Organiser Box wi | California Design Den Luxury 100% Cotton Sateen Buttery | California Design Den Luxury 100% Cotton Sateen Buttery | McAuley better |
| stainless steel cookware set | Sumeet Stainless Steel Cookware Set With Lid, 1.6, 2.1  | Sumeet Stainless Steel Cookware Set With Lid, 1.6, 2.1  | 40-Piece Silverware Set, E-far Stainless Steel Flatware | HOMICHEF 14-Piece Nickel Free Stainless Steel Cookware  | McAuley better |
| yoga mat non slip thick | Quickshel yoga mat with carrying bag anti skid Yogamat  | Quickshel yoga mat with carrying bag anti skid Yogamat  | COOLMOON 1/4 Inch Extra Thick Yoga Mat Double-Sided Non | Extra Thick Yoga Mat, 5/8 Inch (16 mm) with No Stick Ri | Tie |
| protein powder whey chocolate | Six Pack Nutrition 100% Whey Protein Powder - 1 kg/ 2.2 | Six Pack Nutrition 100% Whey Protein Powder - 1 kg/ 2.2 | Pure Protein 100% Whey Protein Powder, Rich Chocolate,  | Pure Protein 100% Whey Protein Powder, Rich Chocolate,  | Tie |
| baby diaper pants large pack | FIXEL Baby Diaper Pants Extra Large XXL-Size - pack of  | FIXEL Baby Diaper Pants Extra Large XXL-Size - pack of  | Parker Baby Diaper Backpack - Full Zip Diaper Bag with  | Gerber Plastic Pants, 0-3 Months, Fits Up to 12 lbs (4  | Tie |
| dog food dry adult | Jerhigh Wet food for Dogs Grilled Chicken in Gravy 120g | Jerhigh Wet food for Dogs Grilled Chicken in Gravy 120g | Rachael Ray Nutrish Limited Ingredient Lamb Meal & Brow | Wellness Complete Health Toy Breed Dry Dog Food with Gr | McAuley better |
| car phone mount dashboard | Detachi Universal Magnetic Car Mount Holder for Dashboa | Detachi Universal Magnetic Car Mount Holder for Dashboa | TRUE LINE Automotive Dashboard Car Windshield Cell Phon | TRUE LINE Automotive Dashboard Car Windshield Cell Phon | McAuley better |
| mechanical keyboard RGB gaming | Redgear A-10 Wired Gaming Mouse with RGB LED, Lightweig | Redgear A-10 Wired Gaming Mouse with RGB LED, Lightweig | RGB Gaming Keyboard and Mouse Combo,MageGee GK980 Wired | RGB Gaming Keyboard and Mouse Combo,MageGee GK980 Wired | Indian better |
| men's formal leather belt | U.S. CROWN Clutch with belt sling Bag Purse Cross-Body  | Am leather Leather Men wallet (teal) | Bullko Men's Casual Genuine Leather Dress Belt for Jean | Men's Dress Belt 'ALL GENUINE LEATHER' Stitching 30mm R | McAuley better |
| kids school backpack waterproof | INOVERA Girl's Nylon Travel Waterproof Anti Theft Backp | INOVERA Girl's Nylon Travel Waterproof Anti Theft Backp | POWOFUN Kids Backpack School Bag Children Water-Resista | BLUEFAIRY Girls Elementary School Bag Kids Backpacks Cu | Tie |
| air fryer large capacity | Alt-L ALTERING LIFESTYLES Digital Kitchen weight Scale  | Alt-L ALTERING LIFESTYLES Digital Kitchen weight Scale  | Galanz Combo 8-in-1 Air Fryer Toaster Oven, Convection  | Galanz Combo 8-in-1 Air Fryer Toaster Oven, Convection  | McAuley better |
| face moisturizer dry skin | Cetaphil Moisturising Cream for Face & Body , Dry to ve | Cetaphil Moisturising Cream for Face & Body , Dry to ve | L'Oreal Paris Skincare Hydra Genius Daily Liquid Care O | L'Oreal Paris Skincare Hydra Genius Daily Liquid Care O | McAuley better |

## Verdict

- **Per-query dataset wins:** Indian 1, McAuley 10, ties 7
- **Best fusion on Indian:** RRF (RRF: 10/18 #1, 32/54 top-3 | DBSF: 10/18 #1, 28/54 top-3)
- **Best fusion on McAuley:** RRF (RRF: 15/18 #1, 44/54 top-3 | DBSF: 14/18 #1, 43/54 top-3)
- **Overall dataset winner:** McAuley

### Recommended defaults

- `QDRANT_COLLECTION=amazon_products_v2` when using McAuley
- `QDRANT_COLLECTION=amazon_products` when using Indian CSV
- `FUSION_METHOD=rrf`
