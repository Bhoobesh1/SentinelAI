# Credential Stuffing

## Description
Credential stuffing is an attack where a single source (one IP, or a small
coordinated pool) attempts authentication against MANY different accounts,
typically using credentials leaked from a prior breach elsewhere. Unlike
brute force, which hammers one account with many password guesses,
credential stuffing spreads a smaller number of attempts across a large
number of distinct accounts, betting that at least a few will have reused
the same leaked password.

## Indicators
- One source IP (or a small set of related IPs) attempting authentication
  against many different, otherwise-unrelated accounts in a short window.
- Attempts are typically fast and automated -- seconds apart, not the
  pace of a human typing.
- The large majority of attempts fail, since most leaked credentials
  don't match the current password for a given service; a small minority
  may succeed.
- Devices used across the attempts are often inconsistent, generic, or
  clearly automated/scripted rather than legitimate end-user devices.

## Investigation Questions
- How many distinct accounts were targeted from this source, and over
  what time span?
- Did any of the targeted accounts have a successful authentication? If
  so, prioritize those for immediate review.
- Is the source IP associated with known credential-stuffing
  infrastructure, botnets, or prior incidents?
- Are the targeted accounts otherwise unrelated, or do they share some
  organizational pattern (e.g., all from one department)?

## Defensive Recommendations
- Rate-limit or block the offending source IP(s) at the network/WAF layer.
- Force password resets for any account with a successful login from
  this source, and require MFA going forward.
- Consider organization-wide breached-password screening so reused
  credentials from external breaches can't be exploited here.
- Correlate with other accounts on the network that may have been probed
  from the same infrastructure, even if this particular alert only
  covers one account.
