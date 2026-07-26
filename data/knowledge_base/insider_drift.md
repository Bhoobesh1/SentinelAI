# Insider Behavioral Drift

## Description
Insider behavioral drift describes a gradual, sustained change in an
entity's normal behavior over an extended period -- shifting login hours,
increasing use of a new device, or growing access to more sensitive
resources -- rather than a single sharp, obviously anomalous event. This
pattern is intentionally the hardest to detect, because each individual
step in the drift looks only mildly unusual, and early in the drift
window it may be genuinely indistinguishable from normal variation or a
legitimate change in someone's role or habits.

## Indicators
- A slow, sustained shift in login timing, device usage, or resource
  access patterns over days to weeks, rather than an abrupt change.
- Later in the drift window, the entity's behavior looks meaningfully
  different from its own earlier baseline, even though no single event
  in isolation is extreme.
- Increasing access to sensitive resources over time, without a single
  dramatic access spike.
- Session characteristics (length, timing) gradually diverging from the
  entity's established pattern.

## Investigation Questions
- Over what time period has this entity's behavior been changing, and
  what specifically has shifted (hours, device, resources)?
- Is there a legitimate explanation -- a role change, a new project, a
  change in work schedule -- that would explain the drift?
- Does the trajectory of the drift point toward increasing access to
  sensitive systems, or is it a lateral, non-escalating change?
- Has this entity's manager or HR record confirmed any recent change in
  responsibilities?

## Defensive Recommendations
- Because no single event is a strong signal, evaluate this pattern
  using an extended historical window rather than reacting to any one
  triggering event in isolation.
- Confirm with the entity's manager or HR whether a legitimate role or
  responsibility change explains the drift before escalating.
- If the drift trends toward sensitive-resource access without a
  legitimate explanation, treat it as a potential slow-moving insider
  threat and review the full history of the drift period.
- Periodically refresh behavioral baselines so genuine legitimate
  changes (promotions, role changes) don't get permanently flagged, while
  still requiring sustained evidence -- not a single event -- before
  updating what's considered "normal" for that entity.
