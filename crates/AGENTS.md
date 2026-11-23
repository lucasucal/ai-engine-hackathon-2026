# AGENTS.md

## Rust best practices

### Tests

Assert on shape with `std::assert_matches`, not `assert!(value.is_err())`. A
failing `assert!` prints only the expression, while `assert_matches!` prints the
value that did not match.

In tests, prefer `.unwrap()` over `.expect("...")`. A failing test already
points at the line, and the error it unwraps says what went wrong.

Write structurally similar tests as one function with a `test_case` per input,
rather than as separate `#[test]`s. Leave out a `; "description"` where the
inputs already say it.

### Common traits

Prefer a `From` impl over a helper function like `fn to_x(a) -> X` where
appropriate; it prevents code duplication. Be more cautious in public-facing
libraries, where it may unnecessarily increase the API surface.
