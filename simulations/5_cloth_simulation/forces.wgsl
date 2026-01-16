

// cloth simulation 
//      - only the motion of individual vertices is calculated.
//      - Each vertex is considered as a point body with a mass, a position and a velocity.
//      - The resultant of the forces allows us to calculate the acceleration. 
//      - acceleration updates velocity, which updates the position.
//          ->   v1 = v0 + F/m * dt
//          ->   x1 = x0 + v1 * dt

//      - For stability, we must compute multiple time steps between each frame.

//      - We consider the different in forces.wgsl vertices of the cloth are connected by springs. 
//      - The forces are easily calculated using Hooke’s law :
//          -> F_h = -k * dL

//      - the minus  sign is there to remind us that the force opposes the length variation and therefore acts to bring the spring back to its rest length.

//      - There are three types of springs in a cloth (each with its own k): 
//          - Structural  :
//          - Shear       :
//          - Bending     : 
//      - 
//      - To prevent unwanted oscillations, we also add damping to the springs :
//          -> F_d = -c * v_rel,    c : damping coefficient

//      - Gravity : F = mg

//      - The contact with the sphere will also be modelised by an elastic force:
//          -> F_contact = -k_contact * dL_contact
//              - where: 
//                  dL_contact is the penetration depth of the vertex into the sphere.
//                  k_contact is the stiffness of collisions.   

//      - Frictional Forces : The friction is tangent to the friction surface and is calculated on the basis of the resultant of the other forces. 
//          -> Ro   : resultant of the other forces 
//          -> Ro,n : normal component
//          -> Ro,t : tangent component
//          -> Ro : resultant of the other forces 
//          -
//          -> Ro,n = (Ro · 1n) * 1n
//          -> Ro,t = Ro - Ro,n
//          -> 1t = Ro,t / |Ro,t|
//          -> F_friction = -min( |Ro,t| , μ * |Ro,n| ) * 1t

//      - In short, the frictional force cancels out the tangential component of the other forces up to a certain point which depends on the normal component.



//      -


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






struct Spring {
    p0: u32,
    p1: u32,
    prev_length: f32,
    spring_type: u32,  // 0 = structural, 1 = shear, 2 = bend
    // _pad: u32,
    force: vec4<f32>, // force applied to particle p0 , unused w component
};



@group(0) @binding(0) var<uniform> physics: PhysicsConstants;
@group(0) @binding(1) var<storage, read> sim_data: SimulationData;   // R
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
// @group(0) @binding(3) var<storage, read_write> springs: array<Spring>;

@group(0) @binding(3) var<storage, read_write> structural_springs: array<Spring>;
@group(0) @binding(4) var<storage, read_write> shear_springs: array<Spring>;
@group(0) @binding(5) var<storage, read_write> bend_springs: array<Spring>;


fn get_pos(index: u32) -> vec3<f32> {
    return vec3<f32>(
        particles[index].model_matrix[3][0],
        particles[index].model_matrix[3][1],
        particles[index].model_matrix[3][2]
    );
}

fn get_spring_params(t: u32) -> vec2<f32> {
    // 0 : struct
    // 1 : shear
    // 2 : bend
    switch (t) {
        case 0u { return vec2<f32>(physics.k_struct, physics.rest_len_struct); }
        case 1u { return vec2<f32>(physics.k_shear,  physics.rest_len_shear); }
        default { return vec2<f32>(physics.k_bend,   physics.rest_len_bend); }
    }
}

// compute_springs (excerpt)
@compute @workgroup_size(64)
fn compute_springs(@builtin(global_invocation_id) id: vec3<u32>) {

    // use global invocation ID to know which spring type to process
    let gid = id.x;
    let EPS = 1e-6;
    let dtd = sim_data.dt;
    let dt = select(dtd, EPS, dtd < EPS);       // for damping
    // let c = physics.damping_c;

    // -------- Structural Case --------
    let ns = arrayLength(&structural_springs);
    if (gid < ns) {
        var s = structural_springs[gid];
        let p0 = s.p0; let p1 = s.p1;
        if (p0 >= arrayLength(&particles) || p1 >= arrayLength(&particles)) { s.force = vec4<f32>(0.0); structural_springs[gid] = s; return; }
        
        // pos and dist
        let pos0 = get_pos(p0); 
        let pos1 = get_pos(p1);
        let delta = pos1 - pos0; let dist = length(delta);
        if (dist < 1e-6) { s.force = vec4<f32>(0.0); structural_springs[gid] = s; return; }
        
        // ==== Compute force =====
        let dir = delta / dist;
        let rest =  physics.rest_len_struct;
        let k = physics.k_struct;
        // Hooke
        let stretch = dist - rest;
        let hooke = k * stretch * dir; // force applied to p0
        // Damping
        // let rate = (dist - s.prev_length)/dt;
        // let damp = -c * rate * dir;
        // let total = hooke + damp;
        let vel0 = particles[p0].velocity.xyz;
        let vel1 = particles[p1].velocity.xyz;
        let rel_vel = vel1 - vel0;  // relative velocity
        let v_along_spring = dot(rel_vel, dir);  // component along spring
        let c= physics.k_damp_struct;
        let damp = c * v_along_spring * dir;
        let total = hooke + damp;

        // Store into Spring
        s.force = vec4<f32>(total, 0.0);
        s.prev_length = dist; // Update Length
        structural_springs[gid] = s;
        return;
    }

    // -------- Shear Case --------
    let gid1 = gid - ns;
    let nh = arrayLength(&shear_springs);
    if (gid1 < nh) {
        var s = shear_springs[gid1];
        let p0 = s.p0; let p1 = s.p1;
        if (p0 >= arrayLength(&particles) || p1 >= arrayLength(&particles)) { s.force = vec4<f32>(0.0); shear_springs[gid1] = s; return; }
        
        // pos and dist
        let pos0 = get_pos(p0); 
        let pos1 = get_pos(p1);
        let delta = pos1 - pos0; let dist = length(delta);
        if (dist < 1e-6) { s.force = vec4<f32>(0.0); shear_springs[gid1] = s; return; }

        // ==== Compute force =====
        let dir = delta / dist;
        let rest = physics.rest_len_shear;
        let k = physics.k_shear;
        let stretch = dist - rest;
        let hooke = k * stretch * dir;

        // Damping
        let vel0 = particles[p0].velocity.xyz;
        let vel1 = particles[p1].velocity.xyz;
        let rel_vel = vel1 - vel0;  // relative velocity
        let v_along_spring = dot(rel_vel, dir);  // component along spring
        let c= physics.k_damp_shear;
        let damp = c * v_along_spring * dir;
        let total = hooke + damp;

        // Store into Spring
        s.force = vec4<f32>(total, 0.0);
        s.prev_length = dist; // Update Length
        shear_springs[gid1] = s;
        return;
    }

    // -------- Bend Case --------
    let gid2 = gid1 - nh;
    let nb = arrayLength(&bend_springs);
    if (gid2 < nb) {
        var s = bend_springs[gid2];
        let p0 = s.p0; let p1 = s.p1;
        if (p0 >= arrayLength(&particles) || p1 >= arrayLength(&particles)) { s.force = vec4<f32>(0.0); bend_springs[gid2] = s; return; }
        
        // pos and dist
        let pos0 = get_pos(p0); 
        let pos1 = get_pos(p1);
        let delta = pos1 - pos0; let dist = length(delta);
        if (dist < 1e-6) { s.force = vec4<f32>(0.0); bend_springs[gid2] = s; return; }
        
        // ==== Compute force =====
        let dir = delta / dist;
        let rest = physics.rest_len_bend;
        let k = physics.k_bend;
        let stretch = dist - rest;
        let hooke = k * stretch * dir;

        // Damping
        let vel0 = particles[p0].velocity.xyz;
        let vel1 = particles[p1].velocity.xyz;
        let rel_vel = vel1 - vel0;  // relative velocity
        let v_along_spring = dot(rel_vel, dir);  // component along spring
        let c= physics.k_damp_bend;
        let damp = c * v_along_spring * dir;
        let total = hooke + damp;

        // Store into Spring
        s.force = vec4<f32>(total, 0.0);
        s.prev_length = dist; // Update Length
        bend_springs[gid2] = s;
        return;
    }

    
    // gid >= total_springs -> nothing
}

// accumulation pass: each particle scans all three spring lists and sums contributions.
// This is simple (O(N*S)) but correct and does not require atomics.
@compute @workgroup_size(64)
fn accumulate_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let pid = id.x;
    if (pid >= arrayLength(&particles)) { return; }

    var sum: vec3<f32> = vec3<f32>(0.0);

    // structural
    let ns = arrayLength(&structural_springs);
    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let s = structural_springs[i];
        if (s.p0 == pid) {
            sum = sum + s.force.xyz;
        } else if (s.p1 == pid) {
            sum = sum - s.force.xyz;
        }
    }

    // shear
    let nh = arrayLength(&shear_springs);
    for (var i: u32 = 0u; i < nh; i = i + 1u) {
        let s = shear_springs[i];
        if (s.p0 == pid) {
            sum = sum + s.force.xyz;
        } else if (s.p1 == pid) {
            sum = sum - s.force.xyz;
        }
    }

    // bend
    let nb = arrayLength(&bend_springs);
    for (var i: u32 = 0u; i < nb; i = i + 1u) {
        let s = bend_springs[i];
        if (s.p0 == pid) {
            sum = sum + s.force.xyz;
        } else if (s.p1 == pid) {
            sum = sum - s.force.xyz;
        }
    }

    // write spring total; add gravity in movement shader
    particles[pid].force = vec4<f32>(sum, 0.0);
}



// =================


// Compute the force of each spring applied to p0
// @compute @workgroup_size(64)
// fn main(@builtin(global_invocation_id) id: vec3<u32>) {
//     /*
//         Goal :
//             - for each spring, compute the force it exerts on its two particles
//             - store the result in spring.force_p0 and spring.force_p1
//             - later, in compute shader, iterate on particles to get those forces and obtain the position/velocity.   
//      */

//     // GET spring Index
//     let sid = id.x;
//     if (sid >= arrayLength(&springs)) {return;}

//     var spring = springs[sid];
//     let p0 = spring.p0;                  // get index
//     let p1 = spring.p1;
//     let pos0 = get_pos(p0); // get pos
//     let pos1 = get_pos(p1);
//     let delta = pos1 - pos0;
//     let dist = length(delta);

//     // Prevent divisionforce_p0 by zero (NaNs)
//     if (dist < 1e-6) { spring.force = vec4<f32>(0.0); return; }

//     // ==== Compute force =====
//     let dir = delta / dist;// ==== Compute force =====
//     let params = get_spring_params(spring.spring_type);
//     let k = params.x;
//     let rest_len = params.y;

//     // Hooke's law
//     let stretch = dist - rest_len;
//     let hooke = k * stretch * dir;

//     // TODO: DAMPING ?

//     // Store into Spring
//     spring.force =  vec4<f32>(hooke, 0.0);
//     springs[sid] = spring;
// }

// // Compute TOTAL force on each particle (sum of all spring forces)
// @compute @workgroup_size(64)
// fn accumulate_forces(@builtin(global_invocation_id) id: vec3<u32>) {
//     let pid = id.x;
//     if (pid >= arrayLength(&particles)) { return; }

//     // Find adjacency range for this particle
//     let start = adj_offsets[pid];
//     // adj_offsets length is N+1, safe to read adj_offsets[pid+1] only if pid+1 < length
//     let end = adj_offsets[pid + 1u];

//     var sum: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

//     for (var ai = start; ai < end; ai = ai + 1u) {
//         let sid = adj_indices[ai];
//         if (sid >= arrayLength(&springs)) { continue; }
//         let s = springs[sid];
//         // s.force is the force applied to p0; depending on whether pid == p0 or p1, add or subtract
//         if (s.p0 == pid) {
//             sum = sum + s.force.xyz;
//         } else if (s.p1 == pid) {
//             sum = sum - s.force.xyz;
//         } // else: malformed adjacency but ignore
//     }

//     // Optionally add gravity and other external forces here.
//     // For now we write spring-based sum; movement shader will add gravity / mass / dt.
//     particles[pid].force = vec4<f32>(sum, 0.0);
// }






