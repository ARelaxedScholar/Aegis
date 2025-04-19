// all this does, is give us a bridge between Aegis and Athena.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .compile_protos(&["proto/simulation.proto"], &["proto"])?;
    Ok(())
}
