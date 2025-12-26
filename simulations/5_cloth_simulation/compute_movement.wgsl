// 
// This is a compute shader (no rendering, just computations).

// instances.rs create the movement tracking matrix
// .wgsl : to display the instances based on the movement matrix position
// .wgsl : to update the movement matrix
// 

// 


struct SimulationData {
    dt: f32,
    radius: f32,
    _pad1: vec2<f32>,
    //
    gravity: vec3<f32>,   // wgpu treats vec3 as vec4 (16bytes) --> add 4bytes padding
    _gravity_pad: f32,
} // 4+4 + vec3 + 4 = 8 + 16 = 24 bytes ---> 32-24 = 8 bytes padding

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
    
    // Array bounds check
    if (index >= arrayLength(&particles)) { return; }
    
    var particle = particles[index];
    
    // Gravity : v' = v + g*dt
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
    


    // Write updated particle back
    particles[index] = particle;
}





