// main.rs
mod cloth;


use std::sync::Arc;

use crate::cloth::ClothSimApp;
mod sphere_vertices; // module to compute sphere geometry (vertices & indices)
use wgpu_bootstrap::{egui, Runner};



fn main() {

    let mut runner = Runner::new(
        "Particles Simulations",
        1200,
        800,
        egui::Color32::from_rgb(50, 50, 50),
        32,
        0,
        Box::new(|context| Arc::new(ClothSimApp::new(context))),
    );



    runner.run();
}