# ─── Builder stage: compile Rust + pyo3 ────────────────────────────────
FROM rust:1.85 AS builder
WORKDIR /usr/src/aegis

# Install protoc & Python 3.11 dev headers
RUN apt-get update \
 && apt-get install -y \
      protobuf-compiler \
      python3.11-dev \
      pkg-config \
 && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo build --release

# ─── Runtime stage: Bookworm slim with Python 3.11 runtime ────────────
FROM debian:bookworm-slim

# CA certs + Python 3.11 runtime
RUN apt-get update \
 && apt-get install -y \
      ca-certificates \
      docker.io \
      python3-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/aegis/target/release/aegis-bin /usr/local/bin/

ENTRYPOINT ["aegis"]
