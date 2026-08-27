# Deployment policy

Applies to all production services.

## Pre-deploy checklist

- Change is reviewed and approved by a second engineer.
- Database migrations are backward compatible for at least one release.
- A rollback path exists and has been tested in staging.
- The on-call engineer for the owning team is aware the deploy is happening.

## During a customer-facing incident

- Freeze deploys. The incident commander owns lifting the freeze.
- Prefer rollback over fixing forward. Roll back to the last known-good release,
  then investigate.
- Post a status update to stakeholders within 15 minutes of declaring the incident.

## Post-incident

- Verify recovery against monitoring and one user-facing check before closing.
- Write a timeline and schedule a blameless postmortem within two business days.
