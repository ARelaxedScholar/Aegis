FROM rust:1.85 as builder
WORKDIR /usr/src/aegis

# Install protobuf-compiler
RUN apt-get update \
    &&  apt-get install -y \
        protobuf-compiler \
        python3-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy your source and build
COPY . .
RUN cargo build --release

# 2) Runtime stage: minimal Debian image
FROM debian:buster-slim


# Build the application
RUN cargo build --release

# install CA certs for the gRPC client
RUN apt-get update \
 && apt-get install -y ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Start a new slim debian container and copy the binary
# from the builder stage
FROM debian:buster-slim
COPY --from=builder /usr/src/aegis/target/release/aegis-service /usr/local/bin/
ENTRYPOINT ["aegis-service"]