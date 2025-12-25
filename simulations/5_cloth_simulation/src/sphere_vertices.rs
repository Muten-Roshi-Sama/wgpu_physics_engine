// simulations/3_Globe/src/sphere.rs
// UV-sphere generator. Returns (vertices, indices).
// Uses the parent module's `Vertex` type (super::Vertex).


/* =========== Generate uv sphere Vertices and Indexes ============
    from : https://songho.ca/opengl/gl_sphere.html
    An arbitrary point (x, y, z) on a sphere can be computed by parametric equations with the corresponding sector angle θ and stack angle ϕ.
        x = [r * cos(ϕ)] * cos(θ)
        y = [r * cos(ϕ)] * sin(θ)
        z = r * sin(ϕ)

    The range of sector angles is from 0 to 360 degrees, and the stack angles are from 90 (top) to -90 degrees (bottom). The sector and stack angle for each step can be calculated by the following;
    angles of per sector/stack :
        - stackAngle  : θ = 2π * (sectorStep / sectorCount)
        - sectorAngle : ϕ = π/2 - π * (stackStep / stackCount)

        - sectorCount = number of longitudes around the sphere (azimuth divisions).
        - stackCount = number of latitudinal divisions from top (north pole) to bottom (south pole).
    */
pub fn generate_uv_sphere(
    radius: f32, 
    stack_count: usize, 
    sector_count: usize)
    -> (Vec<([f32; 3], [f32; 3], [f32; 2])>, Vec<u32>)
    {
    let mut vertices = Vec::with_capacity((stack_count + 1) * (sector_count + 1));
    let mut indices = Vec::with_capacity((2 * stack_count - 2) * sector_count * 3);

    let pi = std::f32::consts::PI;
    let sector_step = 2.0 * pi / sector_count as f32;
    let stack_step = pi / stack_count as f32;

    for i in 0..=stack_count {
        let stack_angle = pi / 2.0 - i as f32 * stack_step;     // starting from pi/2 to -pi/2
        let xy = radius * stack_angle.cos(); // r * cos(u)
        let z = radius * stack_angle.sin();  // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // first and last vertices have same position and normal, but different tex coords
        for j in 0..=sector_count {
            let sector_angle = j as f32 * sector_step;          // starting from 0 to 2pi

            let x = xy * sector_angle.cos();                    // r * cos(u) * cos(v)
            let y = xy * sector_angle.sin();                    // r * cos(u) * sin(v)

            let pos = [x, y, z];
            let len = (x*x + y*y + z*z).sqrt();
            let normal = if len != 0.0 { [x/len, y/len, z/len] } else { [0.0, 1.0, 0.0] };

            let u = j as f32 / sector_count as f32;
            let v = i as f32 / stack_count as f32; // or 1.0 - (i/stack_count) if you prefer flipped V

            vertices.push( (pos, normal, [u, v]) );
        }
    }

    // indices
    for i in 0..stack_count {
        let k1 = i * (sector_count + 1);
        let k2 = k1 + sector_count + 1;

        for j in 0..sector_count {
            let a = (k1 + j) as u32;
            let b = (k2 + j) as u32;
            let c = (k1 + j + 1) as u32;
            let d = (k2 + j + 1) as u32;

            if i != 0 {
                // k1, k2, k1+1  (a, b, c)
                indices.push(a); indices.push(b); indices.push(c);
            }

            if i != stack_count - 1 {
                // k1+1, k2, k2+1  (c, b, d)
                indices.push(c); indices.push(b); indices.push(d);
            }
        }
    }

    (vertices, indices)
}







