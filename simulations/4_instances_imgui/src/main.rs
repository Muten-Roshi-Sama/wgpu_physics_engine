// main.rs
mod instance;


use std::sync::Arc;

use crate::instance::ParticleSimApp;
mod sphere_vertices; // module to compute sphere geometry (vertices & indices)
use wgpu_bootstrap::{egui, Runner};


/*
Physics Engine :
    4. Instances and Imgui : 
            - Render many object instances (globe or cube) by uploading a per-instance transform buffer 
                and using instanced drawing in the GPU; 
            - integrate ImGui to add a GUI for runtime controls.
*/

fn main() {

    let mut runner = Runner::new(
        "Particles Simulations",
        800,
        600,
        egui::Color32::from_rgb(245, 245, 245),
        32,
        0,
        Box::new(|context| Arc::new(ParticleSimApp::new(context))),
    );



    runner.run();
}