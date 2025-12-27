// 
// This is a compute shader (no rendering, just computations).

// instances.rs create the movement tracking matrix
// .wgsl : to display the instances based on the movement matrix position
// .wgsl : to update the movement matrix
// 

// 

// Same as in forces.wgsl
struct SimulationData {
    dt: f32,
    radius: f32,
    globe_radius: f32,  // to compute collisions
    mass: f32,          //=16bytes
    //
    _pad2: vec3<u32>,   //=32bytes
    grid_width: u32,
} // =32bytes



// Same as in forces.wgsl
struct Particle {
    /*
        model Matrix (no rotation)
            Scalex, 0,     0       Tx
            0      Scaley, 0       Ty
            0,     0,      Scalez  Tz
     */
    model_matrix: mat4x4<f32>,      // POSITION : rotation; scale; translation
    velocity: vec4<f32>,           // VELOCITY : xyz velocity, w unused (padding)
    force: vec4<f32>,              // FORCE    : xyz force, w unused (padding)
} // 48 + 16 + 16 = 80 bytes





@group(0) @binding(0) var<uniform> sim_data: SimulationData;                 // small binding to read
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;  // bigger storage for R/W






@compute @workgroup_size(64)    // compute shader entry point
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    /*  Methodology :
        - get index
        - get forces computations from forces.wgsl
        - Compute velocity based on forces
        - COLLISIONS :
            - reflect particle velocity if its inside the globe
     */


    // ======= GET ========
    let index = global_id.x;
    if (index >= arrayLength(&particles)) { return; } // Array bounds check
    var particle = particles[index];
    var pos = get_pos(index);
    var vel = particles[index].velocity;



    // ======= UPDATE VELOCITY & POSITION ========
    let force = particle.force.xyz;
    let accel = force / sim_data.mass;   // a = F/m

    // Update velocity: v' = v + a*dt
    vel = vel + vec4<f32>(accel * sim_data.dt, 0.0);
    
    // Update position: x' = x + v*dt
    pos += vel.xyz * sim_data.dt;
    


    // ---- COLLISIONS -------
    /*
        Reflection: v2 = v1 - 2 * (v1 · n) * n
            where :
                v1 : init speed 
                v2 : reflected speed 
                n  : normal vvect of the collision surface (direction ou la particule est poussée)    
     */


    let dist = length(pos);  // distance from origin
    let min_dist = sim_data.globe_radius + sim_data.radius;
    // If inside the globe, reflect vel
    if (dist < min_dist) {
        let dir = normalize(pos);
        pos = dir * min_dist;

        // Reflect particle velocity
        let v_dot_n = dot(vel.xyz, dir);
        vel = vel - 2.0 * v_dot_n * vec4<f32>(dir, 0.0);
    }


    // Write new pos to model matrix
    particle.model_matrix[3][0] = pos.x;
    particle.model_matrix[3][1] = pos.y;
    particle.model_matrix[3][2] = pos.z;
    
    // Write nezw velocity
    particle.velocity = vel;

    // Write updated particle back
    particles[index] = particle;
}



// ======= Helpers =======

fn get_pos(index: u32) -> vec3<f32> {
    return vec3<f32>(
        particles[index].model_matrix[3][0],
        particles[index].model_matrix[3][1],
        particles[index].model_matrix[3][2]
    );
}





