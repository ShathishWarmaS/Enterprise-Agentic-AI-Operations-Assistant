# Runbook: payment-service

Owner: payments-platform team. On-call rotation: `#payments-oncall`.

## Health signals

- `p99_latency_ms` should stay under 400 ms. Sustained values above 800 ms are a page.
- `error_rate_pct` (share of 5xx responses) above 2% for 5 minutes is a page.
- The service holds a fixed database connection pool of 40 connections per pod.

## 5xx error spikes

A spike in 5xx responses on payment-service is most often caused by database
connection pool exhaustion. When every connection in the pool is checked out,
new requests wait, time out, and return 503.

Steps:

1. Check the `pg_pool_in_use` gauge. If it is pinned at 40, the pool is exhausted.
2. Look for a slow query in the `pg_stat_activity` view. A missing index on
   `transactions.customer_id` has caused this twice before.
3. If a recent deploy correlates with the spike, roll back first and investigate
   after (see Rollback below). Do not debug forward during a customer-facing outage.
4. If there is no recent deploy, scale the service from 6 to 10 pods to add pool
   capacity as a temporary mitigation.

## Rollback

Roll back payment-service with `deployctl rollback payment-service --to last-stable`.
Rollback takes about 90 seconds. After rollback, freeze further deploys until the
root cause is confirmed and a fix is reviewed.

## Cache

payment-service caches merchant configuration in Redis with a 15 minute TTL.
Stale merchant config shows up as incorrect currency handling, not as 5xx errors.
Flush a single merchant with `cachectl drop merchant:<id>`; never flush the whole
keyspace during business hours.

## Credentials

The service authenticates to the payments gateway with an API token stored in the
`payments-gateway-token` secret. The token expires every 90 days. An expired
token returns HTTP 401 from the gateway, which the service surfaces as 5xx to the
caller. Rotate with `secretctl rotate payments-gateway-token` and restart pods.
