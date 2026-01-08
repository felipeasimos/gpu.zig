## Compute Square Example

1. Copy from staging input buffer to device local input buffer
2. barrier
3. compute square to device local output buffer
4. barrier
5. copy from device local output buffer to staging output buffer
6. check results
