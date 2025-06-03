# Aegis (Formerly Simulation GAN-TE Carlo)

## Overview
Aegis is a Rust-based project that automates the deployment of trading algorithms for cryptocurrencies. 

## Requirements
- Rust (latest stable version)
- Cargo (Rust package manager)
- Python (Ensure you got a python version installed as pyo3 will complain about that if not)
- Access to crypto exchange APIs (API keys required)

## How to use
(Currently not in a very user-friendly state)
1. Clone the repo
2. Look at the implementation for the evolution functions.
3. Write a script to run in bin
4. Use cargo run --bin your_script to test it.

This is at the proof of concept work. I am currently building the library framework, so binaries must be defined by the user.
The main.rs file provides a basic server, for when I deploy the functions to a server, but this is not currently live.

## Current Work
- Slides presenting preliminary results: https://docs.google.com/presentation/d/1tD4gExPZPbOnIx00lG5FNrKtftqEut10JBwyxNEWIwc/
- Informal Paper going over the current methodology (2024) : https://www.overleaf.com/read/rrfsdyrpypzq#f6c2d8
- Informal Document of Project (My Personal Documentation): link will be updated soon.


