# Lateral Movement

## Description
Lateral movement describes an attacker (or a compromised account) rapidly
accessing multiple different internal systems or resources -- often ones
the account has never touched before -- shortly after establishing initial
access. This is typically the stage of an incident where an attacker who
has already gained a foothold is exploring the environment to find
sensitive data or expand their reach, rather than the initial break-in
itself.

## Indicators
- A single entity accessing several distinct resources in quick
  succession (minutes apart), especially resources it has no history of
  using.
- The resources touched often escalate in sensitivity (e.g., moving from
  a low-sensitivity wiki to a finance database or admin console).
- Lateral movement frequently follows another anomaly earlier in the
  same session -- an impossible-travel login, a new/unrecognized device,
  or a successful brute-force/credential-stuffing attempt.
- Access patterns look like systematic exploration rather than a normal
  user's typical, narrow set of daily resources.

## Investigation Questions
- What resource was accessed FIRST in this sequence, and does that
  match a plausible legitimate entry point?
- What is the sensitivity of each resource touched, and was any
  high-sensitivity data actually exfiltrated or modified?
- What preceded this activity -- was there an earlier anomaly (login
  from a new device/location, or a successful brute-force/credential
  event) for the same entity?
- Is this pattern consistent with legitimate cross-team work (e.g., an
  engineer debugging across several services) or does it look like
  reconnaissance?

## Defensive Recommendations
- Treat rapid multi-resource access touching high-sensitivity systems as
  high priority, especially when it follows another anomaly.
- Temporarily restrict the account's access while investigating, if
  sensitive systems were touched.
- Review network segmentation -- lateral movement is often easier than
  it should be when internal systems trust each other too broadly.
- Correlate with the account's earlier activity in the same session to
  understand the full attack chain, not just this one alert in isolation.
