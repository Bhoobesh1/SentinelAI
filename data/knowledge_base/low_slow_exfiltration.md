# Low-and-Slow Exfiltration

## Description
Low-and-slow exfiltration is a deliberately patient attack pattern where
sensitive data is accessed and extracted in small amounts spread across
many days or weeks, specifically to avoid triggering volume- or
frequency-based detection. Each individual access may look almost
unremarkable on its own; the signal only becomes clear when viewed across
a longer time horizon, where a pattern of repeated, unusual access to
sensitive resources -- often paired with export- or archive-style
commands -- emerges.

## Indicators
- Repeated access to high-sensitivity resources (e.g., a finance
  database, customer data, a backup vault) that is unusual for this
  entity's normal role.
- Access is sparse and spread out -- days apart -- rather than clustered,
  which is precisely what makes it evade simple rate-based alerting.
- Sessions are often longer than the entity's typical session length,
  and commonly paired with commands associated with copying, exporting,
  or archiving data (e.g., export, zip, file transfer commands).
- The entity's device and location are frequently unchanged (this is
  usually an insider or an attacker fully in control of a legitimate
  session, not a login-time anomaly).

## Investigation Questions
- Over what total time span has this pattern occurred, and how many
  separate instances are there?
- Is access to this resource consistent with the entity's actual job
  function, or is it a genuine departure from their normal role?
- What specific data was touched in each instance, and does the volume
  or sensitivity add up to something concerning in aggregate?
- Were export/archive/transfer commands used, and where did any
  exported data go afterward?

## Defensive Recommendations
- Implement long-window behavioral baselines (not just short-term rate
  limits) specifically to catch this class of slow, patient activity.
- Apply data-loss-prevention controls on sensitive resources that flag
  or block bulk export/archive operations regardless of how the access
  was spread over time.
- Review the entity's legitimate need for access to the resource in
  question, and consider tightening role-based access controls.
- Treat confirmed low-and-slow exfiltration as a serious incident --
  this pattern is specifically designed to look unremarkable, so once
  identified it warrants a full historical review, not just the
  triggering event.
