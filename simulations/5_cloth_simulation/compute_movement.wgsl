// 
// This is a compute shader (no rendering, just computations).

// Reference : https://www.w3.org/TR/WGSL/#alignment-and-size*

// -----------------------------------------------------------------------


const MAX_SPRINGS_PER_PARTICLE: u32 = 12u;




// All struct must be the Same as in forces.wgsl
struct PhysicsConstants {
    k_struct: f32,          // Springs Force
    k_shear: f32,
    k_bend: f32,
    k_damp_struct: f32,        // Damping Force
    k_damp_shear: f32,
    k_damp_bend: f32,
    rest_len_struct: f32, 
    rest_len_shear: f32,
    rest_len_bend: f32,

    k_contact: f32,        // Contact Force =32bytes
    mu: f32,                // Friction
    _pad0: f32,       //=48bytes
}

struct SimulationData {
    dt: f32,
    radius: f32,
    globe_radius: f32,  // to compute collisions
    mass: f32,          //=16bytes
    // --
    grid_width: u32,
    gravity: f32,
    _pad2: vec2<f32>,
} // =32bytes

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



@group(0) @binding(0) var<uniform> physics: PhysicsConstants;
@group(0) @binding(1) var<storage, read> sim_data: SimulationData;   // R
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

fn get_pos(index: u32) -> vec3<f32> {
    return vec3<f32>(
        particles[index].model_matrix[3][0],
        particles[index].model_matrix[3][1],
        particles[index].model_matrix[3][2]
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    /*
    
     */
    // get partucle id
    let pid = id.x;
    if (pid >= arrayLength(&particles)) { return; }

    var particle = particles[pid];
    var pos = get_pos(pid);
    var vel = particle.velocity.xyz;
    var total_force = particle.force.xyz;


    // ======== OTHER FORCES ========
    // Gravity
    total_force += vec3<f32>(0.0, sim_data.mass * sim_data.gravity, 0.0);
    // Contact
    // Friction

    // Store force (debugging)
    // particle.force = vec4<f32>(total_force, 0.0);


    // ======= UPDATE VELOCITY & POSITION ========
    let accel = total_force / sim_data.mass;
    vel += accel * sim_data.dt;
    pos += vel * sim_data.dt;




    // ======== COLLISIONS ===========
    /*
        Reflection: v2 = v1 - 2 * (v1 · n) * n
            where :
                v1 : init speed 
                v2 : reflected speed 
                n  : normal vvect of the collision surface (direction ou la particule est poussée)    
     */
    
    
    let dist = length(pos);
    let min_dist = sim_data.globe_radius + sim_data.radius;

    if (dist < min_dist) {
        if (dist > 1e-6) {
            let n = normalize(pos);
            let penetration = min_dist - dist;
            pos += n * penetration;  // Push out
            vel = vec3<f32>(0.0);    // Kill velocity
        } else {
            pos = vec3<f32>(0.0, 1.0, 0.0) * min_dist;
            vel = vec3<f32>(0.0);
        }
        // OLD refletion vector
        // let n = normalize(pos);
        // pos = n * min_dist;

        // let v_dot_n = dot(vel, n);
        // vel = vel - 2.0 * v_dot_n * n;
    }



    // ===== WRITE RES =====
    particle.velocity = vec4<f32>(vel, 0.0);
    //
    particle.model_matrix[3][0] = pos.x;
    particle.model_matrix[3][1] = pos.y;
    particle.model_matrix[3][2] = pos.z;

    // Write updated particle back
    particles[pid] = particle;
}

