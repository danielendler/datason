# Supabase Local Development

This directory contains the Docker configuration for running a local Supabase stack.

## Setup

1. Copy `.env.example` to `.env`.
2. Edit `.env` and replace the placeholder values with your own secrets.
   These secrets will also be used by GitHub Actions during CI.

## Running Supabase

Use `docker compose` with the `--env-file` option to start the services:

```bash
docker compose --env-file .env up
```

This command reads your customized environment file and launches the stack.
