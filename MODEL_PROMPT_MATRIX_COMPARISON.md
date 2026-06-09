# Model × Prompt Matrix Comparison

Focus: **tool call emission/execution** (broken RAG results are OK for this pass) and **English clarity**.

## Tool calling summary

- Cells with **any tool emitted** (6/9): `dpo_w10/compressed`, `dpo_w10/legacy`, `dpo_w10/steering`, `sft/compressed`, `sft/legacy`, `sft/steering`

- Cells with **tool executed** (6/9): `dpo_w10/compressed`, `dpo_w10/legacy`, `dpo_w10/steering`, `sft/compressed`, `sft/legacy`, `sft/steering`

## Full grid

| Model | Prompt | Tool | English | Emit turns | Exec turns |
|-------|--------|------|---------|------------|------------|
| dpo_w05 | compressed | none | poor | — | — |
| dpo_w05 | legacy | none | ok | — | — |
| dpo_w05 | steering | none | ok | — | — |
| dpo_w10 | compressed | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| dpo_w10 | legacy | **EXECUTED** | mixed | camping turn 6, direct turn 1 | camping turn 6, direct turn 1 |
| dpo_w10 | steering | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | compressed | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | legacy | **EXECUTED** | ok | camping turn 6 | camping turn 6 |
| sft | steering | **EXECUTED** | ok | camping turn 6 | camping turn 6 |

## Cells that called the tool (detail)

### `dpo_w10` + `compressed`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Here's the one that matches your description: the Night Cat Sleeping Air Pad, 75x25 inches, rated for down to minus 20 degrees, at 39.99. Four ratings, five hundred reviews, and it's the highest score

### `dpo_w10` + `legacy`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Huh, I see a few pads, but only one actually fits your camping story. The KLYMIT Static V Sleeping Pad is the real deal, it's got a 4.5 star rating from over six hundred folks who've survived the cold
- **direct turn 1** [EXEC] user: *find me a kids desk lamp with a focused beam for homework*
  - assistant: What the f*ck, man, you got nothing in here that even remotely has a focused beam, you got lap desks with cooling fans and power strips and cable bundles, you're selling a fan to a kid who needs a spo

### `dpo_w10` + `steering`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Pal, here's what you're looking for: a real sleeping pad for camping, not a blanket or a mattress protector. The top one, the Sleepingo Sleeping Pad for Camping, it's inflatable, lightweight, and comp

### `sft` + `compressed`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Alright, here's what you need. You cold because you're not insulated from the ground, which is a physics problem. The Night Cat Sleeping Air Pad with Pillow costs thirty-nine dollars, has four point t

### `sft` + `legacy`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: Alright, here's the goods. For the camping chair, you want something that's got back support, so you don't end up flat on your face. Check out the Guide Gear Oversized Rocking Camp Chair. It's got a s

### `sft` + `steering`

- **camping turn 6** [EXEC] user: *USE YOUR SEARCH_PRODUCTS TOOL TO GIVE ME A SPECIFIC PRODUCT*
  - assistant: The Therm-a-Rest Z Lite Sol is the one I'd grab. It's got a high-density foam core that traps heat really well, it's designed for lightweight backpacking, and it gets up to 4.6 stars from over three t


## English / hallucination notes

- `dpo_w05/compressed`: likely invented product/spec at camping turn 4, camping turn 5

## Recommendation placeholder

Start deploy experiments with **`dpo_w10` + `compressed`** (tool executed in: camping turn 6).
