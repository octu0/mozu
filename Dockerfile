# ==============================================================================
FROM swift:6.2-noble AS builder

WORKDIR /workspace

COPY Package.swift ./
COPY Sources ./Sources
COPY script ./script

RUN swift build -c release --product mozu-web

# ==============================================================================
FROM swift:6.2-noble-slim AS runner

WORKDIR /app

COPY --from=builder /workspace/.build/release/mozu-web /app/mozu-web
COPY Models /app/Models

ENV PORT=8080
ENV WEIGHTS_PATH=/app/Models/weights.json

EXPOSE 8080

ENTRYPOINT ["/app/mozu-web"]
