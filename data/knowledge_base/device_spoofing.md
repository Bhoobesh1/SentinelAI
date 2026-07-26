# Device Spoofing

## Description
Device spoofing occurs when a login presents a device fingerprint that is
new for the account, and in more suspicious cases, a fingerprint that has
also been observed on OTHER, seemingly unrelated accounts. A single
never-before-seen device is common and often benign (a new phone, a new
laptop); the same unfamiliar device fingerprint appearing across multiple
different accounts in a short window is a much stronger signal, since it
suggests either shared/spoofed hardware identifiers or an attacker
reusing the same tooling against multiple targets.

## Indicators
- A login from a device fingerprint never previously associated with
  this entity.
- The same new device fingerprint appearing across multiple different
  entities within a short time window -- this is the key distinguishing
  signal from ordinary "new personal device" activity.
- The new device is often paired with other anomalies: an unusual
  location, an atypical login hour, or access to sensitive resources
  the entity doesn't normally use.

## Investigation Questions
- Has this exact device fingerprint been seen on any other account?
  If so, which ones, and around what time?
- Is there a plausible benign explanation (the user mentions buying a
  new device, using a work loaner, etc.)?
- What did the session do after authenticating with the new device --
  did it touch anything sensitive?
- Does the device fingerprint pattern look automated/generated rather
  than a plausible real device (e.g., an implausible or clearly
  templated identifier)?

## Defensive Recommendations
- Require a secondary verification step (MFA, email/SMS confirmation)
  for logins from any device fingerprint not previously seen.
- If the same device fingerprint is confirmed across multiple accounts,
  treat this as a likely coordinated attack and investigate all affected
  accounts together, not in isolation.
- Maintain a device registry so genuinely new-but-legitimate devices can
  be confirmed and whitelisted going forward, reducing future noise.
