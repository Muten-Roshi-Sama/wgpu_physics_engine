

// cloth simulation 
//      - only the motion of individual vertices is calculated.
//      - Each vertex is considered as a point body with a mass, a position and a velocity.
//      - The resultant of the forces allows us to calculate the acceleration. 
//      - acceleration updates velocity, which updates the position.
//          ->   v1 = v0 + F/m * dt
//          ->   x1 = x0 + v1 * dt

//      - For stability, we must compute multiple time steps between each frame.

//      - We consider the different vertices of the cloth are connected by springs. 
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
    // Springs Force
    k_struct: f32,
    k_shear: f32,
    k_bend: f32,

    // Damping Force
    damping_c: f32,        //=16bytes
    rest_len_struct: f32, 
    rest_len_shear: f32,
    rest_len_bend: f32,

    // Contact Force
    k_contact: f32,        //=32bytes

    // Friction
    mu: f32,

    // Gravity
    gravity: f32,
    _pad0: vec2<f32>,       //=48bytes
}


struct SimulationData {
    dt: f32,
    radius: f32,
    globe_radius: f32,  // to compute collisions
    mass: f32,          //=16bytes
    //
    _pad2: vec3<u32>,   //=32bytes
    grid_width: u32,
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


struct Neighbors {
    // structural
    struct0: u32,
    struct1: u32,
    // shear
    shear0: u32,
    shear1: u32,
    // bend
    bend0: u32,
    bend1: u32,
}



@group(0) @binding(0) var<uniform> physics_constants: PhysicsConstants;                 // small binding to read
@group(0) @binding(1) var<storage, read_write> sim_data: SimulationData;               // small binding to read
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;    // bigger storage for R/W

@compute @workgroup_size(64)    // compute shader entry point
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    /* Methodology :
        - to avoid recomputation of the same spring, 
        - each worker only computes forces at RIGHT and DOWN.
    
    */ 

    // ====== GET ==========
    let index = global_id.x;                    // get particle index
    if index >= arrayLength(&particles) { return; } // Array bounds check
    // let force = particles[index].force;
    let pos1 = get_pos(index);


    // ====== COMPUTE =========
    var total_force = vec3<f32>(0.0, 0.0, 0.0);

    // GRAVITY
    total_force += sim_data.mass * physics_constants.gravity;

    // SPRINGS
    total_force += compute_spring_forces(index, pos1);

    // Write final force
    particles[index].force = vec4<f32>(total_force, 0.0);

}



// ================================================


fn get_pos(index: u32) -> vec3<f32> {
    return vec3<f32>(
        particles[index].model_matrix[3][0],
        particles[index].model_matrix[3][1],
        particles[index].model_matrix[3][2]
    );
}


fn compute_distance_and_direction(pos1: vec3<f32>, index2: u32) -> vec4<f32> {
    // let pos1 = get_pos(index1); //already computed !
    let pos2 = get_pos(index2);
    let dir = pos2 - pos1;
    let dist = length(dir);
    let dir_normalized = dir / dist;
    return vec4<f32>(dir_normalized, dist);
}


// ONLY RETURN RIGHT AND DOWN NEIGHBORS
fn find_neighbors(index: u32) -> Neighbors {
    // return array of neighbor indices for structural, shear, bend springs
    var n: Neighbors;
    n.struct0 = 0u;
    n.struct1 = 0u;
    n.shear0 = 0u;
    n.shear1 = 0u;
    n.bend0 = 0u;
    n.bend1 = 0u;

    let grid_width = sim_data.grid_width;
    let col = index % grid_width;
    let row = index / grid_width;

    // structural
    if col < grid_width-1 {n.struct0 = index + 1;} // right
    if row < grid_width-1 {n.struct1 = index + grid_width;} // down

    // shear
    if (col < grid_width-1) && (row < grid_width-1) {n.shear0 = index + 1 + grid_width;} // down-right
    if (col > 0u) && (row < grid_width-1) {n.shear1 = index - 1 + grid_width;} // down-left
    // bend
    if col < grid_width-2 {n.bend0 = index + 2;} // right-right
    if row < grid_width-2 {n.bend1 = index + 2 * grid_width;} // down-down

    // let struct_neighbor: array<u32, 2> = [
    //     index + 1,         // right
    //     index - 1,         // left
    //     index + sim_data.grid_width,  // down
    //     index - sim_data.grid_width    // up
    // ];

    // // shear
    // let shear_neighbor: array<u32, 2> = [
    //     index + 1 + sim_data.grid_width,   // down-right
    //     index - 1 + sim_data.grid_width,   // down-left
    //     index + 1 - sim_data.grid_width,   // up-right
    //     index - 1 - sim_data.grid_width    // up-left
    // ];

    // // bend
    // let bend_neighbor: array<u32, 2> = [
    //     index + 2,                         // right-right
    //     index - 2,                         // left-left
    //     index + 2 * sim_data.grid_width,   // down-down
    //     index - 2 * sim_data.grid_width    // up-up
    // ];

    return n;
}


fn compute_spring_forces(index: u32, pos1: vec3<f32>) -> vec3<f32> {
    //
    var total_force = vec3<f32>(0.0, 0.0, 0.0);
    let neighbors: Neighbors = find_neighbors(index);


    // For each neighbor type and index (0 and 1)
    for (var i = 0u; i < 2u; i = i + 1u) {

        // Select neighbor
        var n_struct: u32;
        var n_shear: u32;
        var n_bend: u32;
        // Index (wgsl doesnt allow array indexing when it cannot the index range and the array size at compile time.)
        if i == 0u {n_struct = neighbors.struct0; n_shear = neighbors.shear0; n_bend = neighbors.bend0; } 
        else       {n_struct = neighbors.struct1; n_shear = neighbors.shear1; n_bend = neighbors.bend1; }


        // Structural
        if (n_struct != 0u) {
            let dd = compute_distance_and_direction(pos1, n_struct);
            let direction = dd.xyz;
            let dist = dd.w;
            let dt = dist - physics_constants.rest_len_struct;
            total_force += physics_constants.k_struct * dt * direction;
        }
        // Shear
        if (n_shear != 0u) {
            let dd = compute_distance_and_direction(pos1, n_shear);
            let direction = dd.xyz;
            let dist = dd.w;
            let dt = dist - physics_constants.rest_len_shear;
            total_force += physics_constants.k_shear * dt * direction;
        }
        // Bend
        if (n_bend != 0u) {
            let dd = compute_distance_and_direction(pos1, n_bend);
            let direction = dd.xyz;
            let dist = dd.w;
            let dt = dist - physics_constants.rest_len_bend;
            total_force += physics_constants.k_bend * dt * direction;
        }
    }
    return total_force;
}



