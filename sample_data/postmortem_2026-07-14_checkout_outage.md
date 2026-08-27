# Postmortem: checkout outage on 2026-07-14

Status: final. Severity: high. Duration: 47 minutes (14:02–14:49 UTC).

## Impact

Checkout was unavailable or degraded for approximately 18% of sessions for 47
minutes. An estimated 3,900 checkout attempts failed and were not retried by the
client. No payment data was lost or exposed.

## Root cause

A deploy of payment-service at 13:58 UTC shipped a change that removed a database
index on `transactions.customer_id`. Under normal load the resulting sequential
scans were slow enough to hold database connections open far longer than usual.
The connection pool (40 per pod) was exhausted within four minutes, and
payment-service began returning 503 to checkout.

The 5xx spike was caused by database connection pool exhaustion, which in turn
was caused by the missing index shipped in the 13:58 deploy.

## Resolution

1. On-call rolled back payment-service to the previous release at 14:31 UTC.
2. Pool pressure cleared within 90 seconds of rollback completing.
3. Error rate returned to baseline (< 0.5%) by 14:49 UTC.

## What went well

- The runbook's guidance to roll back first during a customer-facing outage was
  followed and worked.

## Action items

- Add a migration check that blocks deploys which drop an index still referenced
  by a query in the top 20 by frequency. Owner: payments-platform. Due 2026-08-01.
- Add a `pg_pool_in_use` alert at 90% utilisation. Owner: observability. Due 2026-07-25.
- Backfill the dropped index in production (done 2026-07-14 15:10 UTC).
