//! Standalone frontend probe; also builds directly with rustc without Cargo.
#[allow(dead_code)]
#[path = "../src/audio.rs"]
mod audio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: frontend_dump INPUT.f32le OUTPUT.f32le".into());
    }
    let bytes = std::fs::read(&args[1])?;
    if bytes.len() % 4 != 0 {
        return Err("input is not f32le PCM".into());
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut frontend = audio::MelFrontend::new();
    frontend.push(&samples)?;
    let mel = frontend.features();
    let output: Vec<u8> = mel.values.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(&args[2], output)?;
    println!("128 {}", mel.frames);
    Ok(())
}
