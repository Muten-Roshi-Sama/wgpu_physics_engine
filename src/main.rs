mod cube_app;

use std::sync::Arc;

use crate::cube_app::CubeApp;
use wgpu_bootstrap::{egui, Runner};

fn main() {
    let mut runner = Runner::new(
        "Cube App",
        800,
        600,
        egui::Color32::from_rgb(245, 245, 245),
        32,
        0,
        Box::new(|context| Arc::new(CubeApp::new(context))),
    );
    runner.run();
}