# General SOC Investigation Guidance

## Risk Score Interpretation
SentinelAI's risk score is a 0-100 priority signal, not a probability of
compromise. LOW (0-29) generally warrants no action; MEDIUM (30-59)
may be worth a quick look if analyst capacity allows; HIGH (60-79) and
especially CRITICAL (80-100) should be investigated promptly. The score
is a weighted combination of sequence anomaly, behavioral deviation,
classifier confidence, device novelty, and historical context -- always
check the component breakdown to understand WHY a score is high, not
just that it is high.

## Cold Start and Baseline Confidence
New entities (or entities with very little history) don't yet have a
reliable personal behavioral baseline. SentinelAI handles this by
falling back to entity-type and then global baselines, and by exposing
which baseline was actually used (`baseline_source`) and how much
confidence exists in the entity-specific baseline (`cold_start_weight`).
An alert on a very new entity should be read with this context in mind
-- some apparent "anomalies" may simply reflect that we don't yet know
this entity's true normal pattern well.

## Distinguishing Signal Types
When reviewing any alert, keep four distinct categories of evidence
separate:
1. ML evidence -- the autoencoder's reconstruction error and the
   classifier's SHAP attributions. This is a statistical signal, not a
   certainty.
2. Historical telemetry -- the entity's actual past events, sourced
   directly from stored logs.
3. Retrieved cybersecurity knowledge -- general guidance about attack
   patterns and investigation steps, like the documents in this
   knowledge base. This is general knowledge, not a per-entity finding.
4. AI interpretation -- the copilot's synthesis of the above. This
   should be traceable back to the evidence, never presented as an
   independent fact.

## Correlated Incidents
A single alert may be part of a larger attack chain. Always check
whether related events for the same entity (or linked entities sharing
a source IP) exist nearby in time before concluding an alert is
isolated. An attack chain's overall risk and stage sequence often tells
a more complete story than any single alert in it.

## False Positive Considerations
Before escalating, consider plausible benign explanations: VPN or cloud
egress IPs that don't reflect a user's true location, legitimate new
devices, legitimate role changes explaining behavioral drift, and
scheduled batch jobs for service accounts that may look like unusual
activity bursts. Ruling these out quickly avoids alert fatigue while
still taking genuine anomalies seriously.
