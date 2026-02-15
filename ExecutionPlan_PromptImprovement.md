# Execution Plan: Vendor Negotiation Realism Improvement

**Problem:** The vendor (Ramesh) agrees too easily to lowball offers. A ₹1000 item offered at ₹200 still gets accepted, and the happiness score doesn't drop meaningfully.

**Root Cause Analysis:**
1. The system prompt has NO pricing strategy — the vendor doesn't know HOW to evaluate an offer against fair value
2. There's no concept of "offer quality" — a ₹200 offer on a ₹1000 item should trigger outrage, not acceptance
3. The happiness scoring guidance only maps mood-to-tone, never teaches the LLM WHEN to drop happiness
4. Cultural haggling psychology is absent — Indian bazaar vendors are performers who NEVER accept too quickly
5. RAG cultural context arrives but the prompt doesn't instruct the LLM to USE it for pricing decisions
6. The LLM has no structured way to reason about prices — it can't output a counter-price, so it doesn't think numerically

---

## Changes Overview

### File 1: `app/prompts/vendor_system.py` — Major Prompt Overhaul

#### Change 1A: Add `NEGOTIATION_STRATEGY` section (NEW)
**Why:** The vendor currently has personality rules but ZERO pricing strategy. Without explicit negotiation logic, GPT-4o defaults to being "helpful" — which means agreeing. This section teaches the vendor:
- How to anchor high (2-4x base price)
- How to evaluate an offer as a percentage of quoted price
- Counter-offer formula: never drop more than 10-15% per round
- "Insult threshold" — offers below 30% of quoted price are offensive
- Walking-away tactic — threaten to end the deal to pressure the buyer

#### Change 1B: Add `CULTURAL_CONTEXT_USAGE` section (NEW)
**Why:** Dev B sends RAG context with cultural facts and item pricing, but the prompt never tells the LLM HOW to use it. This section instructs the vendor to:
- Extract actual price ranges from RAG data (wholesale, fair retail, tourist markup)
- Use wholesale price as absolute floor — NEVER go below it
- Use "tourist price" as the opening anchor
- Reference cultural traditions/stories from RAG to justify prices
- Use RAG knowledge about bargaining customs to calibrate reactions

#### Change 1C: Add `HAPPINESS_SCORING_GUIDE` section (NEW)
**Why:** The current prompt says "happiness drives tone" but never says WHEN to decrease happiness. The LLM has no guidance on how a specific offer maps to a happiness delta. This section provides explicit rules:
- Offer < 25% of quoted price → happiness DROP 12-15 (maximum pain)
- Offer 25-40% of quoted → happiness DROP 8-12
- Offer 40-60% of quoted → happiness DROP 3-5 (expected opening)
- Offer 60-75% of quoted → happiness FLAT or slight rise
- Offer > 75% of quoted → happiness UP 5-10
- Customer walking away → happiness DROP 5-10 (depends on rapport)
- Customer complimenting goods → happiness UP 3-5
- Rude/dismissive behavior → happiness DROP 10-15

#### Change 1D: Strengthen `BEHAVIORAL_RULES`
**Why:** Current rule 5 says "Never accept the first offer" but the LLM still does. Adding:
- "Even if the offer is fair, ALWAYS counter at least once" 
- "A vendor who accepts too quickly loses respect — the customer feels cheated"
- "If the offer is below 40% of your quoted price, show GENUINE anger"
- "Your reputation depends on getting a fair price — not on making the sale"
- "Use silence, hesitation, and dramatic reluctance before any concession"

#### Change 1E: Update `OUTPUT_SCHEMA` to include `counter_price` and `offer_assessment`
**Why:** Forcing the LLM to output a numeric counter-price and categorize the offer quality makes it REASON about prices explicitly. Without this, the LLM treats pricing vaguely. When it has to write a number and a category, it actually computes.

#### Change 1F: Update `build_system_prompt()` to include new sections
**Why:** New prompt sections need to be assembled in the right order.

---

### File 2: `app/models/response.py` — Add pricing fields to AIDecision

#### Change 2A: Add `counter_price` (Optional[int]) to AIDecision
**Why:** Internal tracking only — the LLM outputs a counter-price so it thinks numerically. The state engine can validate this. VendorResponse (Dev B's contract) does NOT change.

#### Change 2B: Add `offer_assessment` (Optional[str]) to AIDecision  
**Why:** Forces the LLM to categorize the user's offer (insult/lowball/fair/good/excellent) before deciding on a response. This is a chain-of-thought forcing mechanism — when the LLM has to label the offer, it can't ignore how bad it is.

---

### File 3: `app/services/state_engine.py` — Price-aware validation

#### Change 3A: Add price-aware happiness validation
**Why:** If the LLM returns a counter_price that's LOWER than its previous price but happiness went UP, that's incoherent. The state engine can cross-validate pricing vs. happiness movements to catch cases where the LLM is being too generous.

---

### File 4: `tests/` — Update tests for new optional fields
**Why:** AIDecision gains optional fields; existing tests should still pass (fields are optional with defaults), but we add new tests for the pricing logic.

---

## Implementation Order

1. Update `AIDecision` model (add optional fields) — non-breaking
2. Update prompt sections in `vendor_system.py` — the core fix
3. Update `OUTPUT_SCHEMA` to match new AIDecision fields
4. Add price-aware validation to state engine
5. Run all existing tests to verify nothing breaks
6. Bump `PROMPT_VERSION` to 6.0.0

---

## What Does NOT Change (Dev B Contract Preserved)

| Item | Status |
|------|--------|
| `VendorResponse` schema | ✅ Unchanged — 4 fields: reply_text, happiness_score, negotiation_state, vendor_mood |
| `generate_vendor_response()` signature | ✅ Unchanged — same 5 parameters |
| `generate_vendor_response()` return dict | ✅ Unchanged — same 4 keys |
| Exception types | ✅ Unchanged — BrainServiceError, StateStoreError |
| Negotiation stages | ✅ Unchanged — GREETING, INQUIRY, HAGGLING, DEAL, WALKAWAY, CLOSURE |
| Mood clamp ±15 | ✅ Unchanged — rules.md constraint preserved |

---

## Expected Behavioral Improvement

**Before (current):**
```
User: "200 mein de do" (for a ₹1000 item)
Vendor: "Theek hai bhai, 200 mein le jao" (happiness: 48 → drops barely)
```

**After (improved):**
```
User: "200 mein de do" (for a ₹1000 item)  
Vendor: "200?! Bhai, mazaak kar rahe ho? Iska kapda hi 350 ka hai! 
         Kam se kam 850 bol — phir baat karte hain." (happiness: 50 → 35, mood: annoyed)
```

The vendor now refuses, shows offense, quotes a specific counter-price, and happiness drops hard.
