"""Subcommand implementations for the `corroborate` CLI.

The top-level entry point at `corroborate.__main__` parses
arguments and dispatches to one of these modules. Each module
exposes both a `main(argv)` for direct invocation and an
`add_args(parser)` / `dispatch(args)` pair so the same logic can
be wired in as a subparser of the top-level CLI."""
