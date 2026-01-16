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
    speed_damp: f32,
    _pad2: f32,
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

    



    // ======== COLLISION FORCE===========
    /*
        Contact (elastic)
        + Friction (tangential)


        / OLD
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

            // Contact Force (elastic)
            let F_contact = physics.k_contact * penetration * n;
            total_force += F_contact;

            // ======= FRICTION FORCE ========
            let Ro = total_force;
            let Ro_n_magnitude = dot(Ro, n);                    // magnitude
            let Ro_n = Ro_n_magnitude * n;                // normal component
            let Ro_t = Ro - Ro_n_magnitude * n;           // tangential component
            let Ro_t_magnitude = length(Ro_t);

            if (Ro_t_magnitude > 1e-6) {
                let tangent = Ro_t / Ro_t_magnitude;    // unit tang dir
                // Coulomb's law
                let F_friction = -min(Ro_t_magnitude, physics.mu * abs(Ro_n_magnitude)) * tangent;
                total_force += F_friction;
            }
        }
    }


    // ======= UPDATE VELOCITY & POSITION ========
    let accel = total_force / sim_data.mass;
    vel += accel * sim_data.dt;
    vel *= pow(sim_data.speed_damp, sim_data.dt); // global damping
    pos += vel * sim_data.dt;


    // ======== BOUNDARIES ========
    // must be after pos and vel computation ! 
    // if not velocity buildup and particles passes through globe
    let final_dist = length(pos);
    let min_allowed = sim_data.globe_radius + sim_data.radius;

    if (final_dist < min_allowed) {
        if (final_dist > 1e-6) {
            let n = normalize(pos);
            pos = n * min_allowed;
            vel = vec3<f32>(0.0);  // ZERO out velocity completely
        } else {
            pos = vec3<f32>(0.0, min_allowed, 0.0);
            vel = vec3<f32>(0.0);
        }
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
