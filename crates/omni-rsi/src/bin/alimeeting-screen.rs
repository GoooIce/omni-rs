use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Screen licensed training TextGrid text for exploratory number candidates")]
struct Arguments {
    /// Directory containing only the permitted training TextGrid files.
    #[arg(long)]
    textgrid_dir: PathBuf,
    /// New JSON report path outside the input tree; existing files are never replaced.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Arguments::parse();
    if let Err(error) = omni_rsi::alimeeting_screen::screen(&args.textgrid_dir, &args.output) {
        eprintln!("alimeeting-screen: {error}");
        std::process::exit(1);
    }
    println!(
        "wrote exploratory text candidates: {}",
        args.output.display()
    );
}
