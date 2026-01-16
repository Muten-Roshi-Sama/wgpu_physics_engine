// // Cloth Mesh Shader - vertices follow particles

// struct CameraUniform {
//     view_proj: mat4x4<f32>,
// }

// struct Particle {
//     model_matrix: mat4x4<f32>,
//     velocity: vec4<f32>,
//     force: vec4<f32>,
// }

// struct VertexInput {
//     @location(0) uv: vec2<f32>,  // UVs encode grid position
// }

// struct VertexOutput {
//     @builtin(position) clip_position: vec4<f32>,
//     @location(0) world_pos: vec3<f32>,
//     @location(1) normal: vec3<f32>,
//     @location(2) uv: vec2<f32>,
// }

// @group(0) @binding(0) var<uniform> camera: CameraUniform;
// @group(1) @binding(0) var<storage, read> particles: array<Particle>;
// @group(1) @binding(1) var<uniform> grid_width: u32;

// fn get_particle_pos(idx: u32) -> vec3<f32> {
//     let m = particles[idx].model_matrix;
//     return vec3<f32>(m[3][0], m[3][1], m[3][2]);
// }

// @vertex
// fn vs_main(in: VertexInput) -> VertexOutput {
//     var out: VertexOutput;
    
//     // Convert UV to grid coordinates (UV ranges 0-1)
//     let col = u32(in.uv.x * f32(grid_width - 1u) + 0.5);
//     let row = u32(in.uv.y * f32(grid_width - 1u) + 0.5);
//     let particle_idx = row * grid_width + col;
    
//     // Read position from corresponding particle
//     let pos = get_particle_pos(particle_idx);
//     out.world_pos = pos;
//     out.uv = in.uv;
    
//     // Compute normal from neighboring particles
//     var normal = vec3<f32>(0.0, 1.0, 0.0);
    
//     // Cross product of edge vectors
//     if (col > 0u && row > 0u && col < grid_width - 1u && row < grid_width - 1u) {
//         let right_idx = row * grid_width + (col + 1u);
//         let down_idx = (row + 1u) * grid_width + col;
        
//         let right = get_particle_pos(right_idx);
//         let down = get_particle_pos(down_idx);
        
//         let v1 = right - pos;
//         let v2 = down - pos;
//         normal = normalize(cross(v1, v2));
//     }
    
//     out.normal = normal;
//     out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    
//     return out;
// }

// // Texture
// @group(2) @binding(0) var cloth_texture: texture_2d<f32>;
// @group(2) @binding(1) var cloth_sampler: sampler;

// @fragment
// fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//     // let tex_color = textureSample(cloth_texture, cloth_sampler, in.uv);
    
//     // // Simple lighting
//     // let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
//     // let normal = normalize(in.normal);
//     // let diffuse = max(dot(normal, light_dir), 0.3);
    
//     // return vec4<f32>(tex_color.rgb * diffuse, tex_color.a);
//     // return vec4<f32>(1.0, 0.0, 1.0, 1.0);  // Bright magenta
//     return vec4<f32>(in.uv.x, in.uv.y, 0.0, 1.0);
// }




// Cloth Mesh Shader - vertices follow particles

struct CameraUniform {
    view_proj: mat4x4<f32>,
}

struct GridUniform {
    grid_width: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Particle {
    model_matrix: mat4x4<f32>,
    velocity: vec4<f32>,
    force: vec4<f32>,
}

struct VertexInput {
    @location(0) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<uniform> grid_uniform: GridUniform;

fn get_particle_pos(idx: u32) -> vec3<f32> {
    let m = particles[idx].model_matrix;
    return vec3<f32>(m[3][0], m[3][1], m[3][2]);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let grid_width = grid_uniform.grid_width;
    
    // Convert UV to grid coordinates
    let col = u32(in.uv.x * f32(grid_width - 1u) + 0.5);
    let row = u32(in.uv.y * f32(grid_width - 1u) + 0.5);
    let particle_idx = row * grid_width + col;
    
    // Read position from particle
    let pos = get_particle_pos(particle_idx);
    out.world_pos = pos;
    out.uv = in.uv;
    
    // Compute normal from neighbors
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    
    if (col > 0u && row > 0u && col < grid_width - 1u && row < grid_width - 1u) {
        let right_idx = row * grid_width + (col + 1u);
        let down_idx = (row + 1u) * grid_width + col;
        
        let right = get_particle_pos(right_idx);
        let down = get_particle_pos(down_idx);
        
        let v1 = right - pos;
        let v2 = down - pos;
        normal = normalize(cross(v1, v2));
    }
    
    out.normal = normal;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    
    return out;
}

@group(2) @binding(0) var cloth_texture: texture_2d<f32>;
@group(2) @binding(1) var cloth_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(cloth_texture, cloth_sampler, in.uv);
    
    // Simple lighting
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, light_dir), 0.3);
    
    return vec4<f32>(tex_color.rgb * diffuse, tex_color.a);
}






