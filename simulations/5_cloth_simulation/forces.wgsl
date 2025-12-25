

struct springs_K {
    K_struct : f32,
    K_shear : f32,
    K_bend : f32,
}


struct Forces {
    p0: f32,
    p1: f32,
    L_rest: f32,
    K_hooke: f32,
    // Damping
    L_prev: f32,

    // Resultant force
    F_hooke: f32,
}





