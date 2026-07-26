# Impossible Travel

## Description
Impossible travel refers to two logins for the same account occurring from
geographically distant locations within a time window too short for a human
to have physically traveled between them. This is one of the strongest
signals of account compromise, since it usually means the account's
credentials are being used from two different physical locations at once
(the legitimate user, and an attacker who has obtained the credentials).

## Indicators
- Two consecutive logins for the same entity from locations separated by
  a distance that, divided by the time between them, implies an
  unrealistic travel speed (well beyond commercial flight speed).
- The second login is often from a location the entity has never used
  before, and frequently from a device that has also never been seen
  for that entity.
- Successful authentication on the anomalous login (the attacker
  typically already has valid credentials, so this is not usually
  paired with authentication failures).

## Investigation Questions
- What is the calculated travel speed implied by the two logins, and is
  it physically plausible under any circumstance (e.g., a VPN, a
  connecting flight layover, a shared corporate network egress point)?
- Has this entity used this location or device before, even rarely?
- What resources were accessed immediately after the anomalous login?
- Is there a legitimate explanation, such as a VPN exit node or a cloud
  provider's egress IP that doesn't reflect the user's true location?

## Defensive Recommendations
- Treat impossible travel as a high-priority signal of likely account
  takeover; consider immediately requiring re-authentication with MFA.
- Review and restrict any sensitive resource access that occurred after
  the anomalous login.
- Cross-check the anomalous IP against known VPN/proxy/cloud provider
  ranges before concluding compromise, to reduce false positives from
  legitimate remote-access tooling.
- If compromise is confirmed, force credential rotation and review all
  activity since the last known-good login.
