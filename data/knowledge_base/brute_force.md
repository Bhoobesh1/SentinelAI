# Brute Force Attacks

## Description
A brute force attack is a rapid, repeated series of authentication attempts
against a single account, typically using automated tooling that tries many
password guesses in a short window. The attacker usually operates from a
single source IP or a small rotating pool of IPs, and often has no prior
relationship with the target account's normal behavior.

## Indicators
- A high number of failed authentication attempts in a short time window
  (minutes, not hours) against the same account.
- Attempts frequently originate from a single source IP or a small set of
  IPs not previously associated with the account.
- Device fingerprints during the burst are often inconsistent or absent,
  since automated tooling rarely presents a stable, legitimate device.
- A brute force burst is sometimes followed by a single successful
  authentication, suggesting the attacker eventually guessed correctly or
  exploited a weak/reused password.

## Investigation Questions
- How many failed attempts occurred, and over what time span?
- Did any attempt in the burst succeed? If so, what happened immediately
  after that successful login?
- Is the source IP associated with known malicious infrastructure or with
  any other account on the network?
- Does this account have a history of weak or reused credentials?

## Defensive Recommendations
- Enforce account lockout or exponential backoff after repeated failures.
- Require multi-factor authentication, especially for accounts with
  sensitive resource access.
- Block or rate-limit the offending source IP at the network layer.
- If any attempt succeeded, treat the account as potentially compromised:
  force a password reset and review all activity since the successful login.
