// 
// This is a compute shader (no rendering, just computations).

// instances.rs create the movement tracking matrix
// .wgsl : to display the instances based on the movement matrix position
// .wgsl : to update the movement matrix
// 


struct SimulationData {
    dt: f32,                // 60fps -> ~0.016f
    bounds: f32,
    damping: f32,
    _pad: f32,
    gravity: vec3<f32>,
    _pad2: f32,
}

struct Particle {
    /*
        model Matrix (no rotation)
            Scalex, 0,     0       Tx
            0      Scaley, 0       Ty
            0,     0,      Scalez  Tz
     */
    model_matrix: mat4x4<f32>,      // rotation; scale; translation
    velocity: vec4<f32>,           // xyz velocity, w unused (padding)
    // accelleration: vec3<f32>,  // controlled by gravity
}

@group(0) @binding(0) var<uniform> sim_data: SimulationData;                 // small binding to read
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;  // bigger storage for R/W


@compute @workgroup_size(64)    // compute shader entry point
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // Particle index
    let index = global_id.x;
    
    // Bounds check
    if (index >= arrayLength(&particles)) { return; }
    
    var particle = particles[index];
    
    // Update Gravity : v' = v + g*dt
    particle.velocity.x += sim_data.gravity.x * sim_data.dt;
    particle.velocity.y += sim_data.gravity.y * sim_data.dt;
    particle.velocity.z += sim_data.gravity.z * sim_data.dt;
    
    // get current pos
    var pos = vec3<f32>(
        particle.model_matrix[3][0],
        particle.model_matrix[3][1],
        particle.model_matrix[3][2]
    );
    
    // Update position: x' = x + v*dt
    pos += particle.velocity.xyz * sim_data.dt;
    
    // Write new pos to model matrix
    particle.model_matrix[3][0] = pos.x;
    particle.model_matrix[3][1] = pos.y;
    particle.model_matrix[3][2] = pos.z;
    
    // Bound checks + damping
    // if (abs(pos.x) > sim_data.bounds) {
    //     particle.velocity.x *= -sim_data.damping;  // bounce with damping
    //     particle.model_matrix[3][0] = sign(pos.x) * sim_data.bounds;
    // }
    // if (abs(pos.y) > sim_data.bounds) {
    //     particle.velocity.y *= -sim_data.damping;
    //     particle.model_matrix[3][1] = sign(pos.y) * sim_data.bounds;
    // }
    // if (abs(pos.z) > sim_data.bounds) {
    //     particle.velocity.z *= -sim_data.damping;
    //     particle.model_matrix[3][2] = sign(pos.z) * sim_data.bounds;
    // }

    // Write updated particle back
    particles[index] = particle;
}





