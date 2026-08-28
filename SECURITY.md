# Security policy

## Supported version

Security fixes target the current `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature for this repository. Do not open a public issue containing credentials, private keys, private hand histories, or an exploitable security defect.

If private reporting is unavailable, contact the repository owner through the public GitHub profile without including sensitive details in the initial message.

## Sensitive files

Never commit API tokens, cloud credentials, SSH private keys, `.env` files, raw private hand histories, or proprietary model checkpoints. Rotate any credential immediately if it is accidentally published; removing it from the latest commit is not sufficient because Git history may retain it.

This repository is research software and is not designed to control real-money poker clients.
