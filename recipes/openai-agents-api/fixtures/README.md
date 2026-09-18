# Synthetic incident fixtures

These files were authored for this recipe. They are not sampled from real
requests, infrastructure, customers, or deployments.

`requests.csv` contains one request per minute in each of two invented regions.
`capacity.csv` contains five-minute capacity observations for those regions.
`deployments.json` identifies the synthetic service, rollout timing, and analysis
cutoff. The small sample is designed to exercise file analysis and delegation;
it cannot establish a production incident's cause.

See `../TASK.md` for the calculation and artifact formats. The local verifier is
not uploaded to the sandbox and independently calculates the expected results.
