// render_cube_textured.rs
use std::path::Path;


use wgpu_bootstrap::{
    cgmath, egui,
    util::orbit_camera::{CameraUniform, OrbitCamera},
    wgpu::{self, util::DeviceExt},
    App, Context,
};

// TODO:
    // 1. re-implement globe render
    // 2. add instances of clothe particles
    // 3. add compute forces (spring, gravity)for those particles




// ----------------- Technical Overview ----------------------------

// M1 : Cloth is globe particles with additional forces between them (springs) 
// M2 : Cloth is single mesh 

// ============= Methoology 1 ================

    // globe_shader.wgsl : render globe
    // cloth_instances.wgsl : render cloth

    // forces.wgsl : compute forces on cloth particles + collision with globe
    // compute.wgsl : update cloth particles positions and velocities based on forces

    // RENDER PIPELINES :
        // globe render pipeline
        // cloth render pipeline

// ============= Methoology 2 ================

// add explanations here later


// RENDER PIPELINES :
    // 1 pipeline for cloth and globe




// --------------------------------------------

// =========== CONFIGURATIONS =============

//render shaders
const GLOBE_SHADER_FILE: &str = "globe_shader.wgsl";
const CLOTH_SHADER_FILE: &str = "cloth_instances.wgsl";
// compute shaders
const COMPUTE_SHADER_FILE: &str = "compute_movement.wgsl";
const FORCES_SHADER_FILE: &str = "forces.wgsl";

// textures
const TEXTURE_FILE: &str = "../../textures/mesh.jpg";
const CLOTH_PARTICLES_TEXTURE_FILE: &str = "../../textures/red.png";
// const TEXTURE_FILE: &str = "../../textures/texture.png";
// const TEXTURE_FILE: &str = "../../textures/earth2048.bmp";
// const TEXTURE_FILE: &str = "../../textures/moon1024.bmp";


// ============= PARAMS =====================

// Camera
const DEFAULT_ZOOM: f32 = 40.0;

// Globe geometry
const RADIUS: f32 = 10.0;
const STACK_COUNT: usize = 64;
const SECTOR_COUNT: usize = 128;
// Specular light parameters
const LIGHT_POS: [f32; 4] = [2.0*RADIUS, 2.0*RADIUS, 2.0*RADIUS, 0.0];
const KS: f32 = 2.0;
const SHININESS: f32 = 100.0;
const _PAD: u32 = 0u32;

// Physics
const TIME_SCALE: f32 = 1.0;
const HZ : f32 = 480.0;
const GRAVITY: f32 = -9.81;
const SPEED_DAMP: f32 = 0.90;
const COLLISION_K: f32 = 1000.0;
const FRICTION_COEFF: f32 = 0.2;

// Cloth 
const CLOTH_PARTICLES_PER_SIDE: u32 = 40;
const CLOTH_PARTICLES_RADIUS: f32 = 0.1;
const CLOTH_SIZE: f32 = 30.0;
const CLOTH_CENTRAL_POS: [f32;3] = [0.0, 4.0 * RADIUS, 0.0];


// Springs
const MAX_SPRINGS_PER_PARTICLE: usize = 12;
const MASS: f32 = 1.0;
const STRUCTURAL_STIFFNESS: f32 = 80.0;
const SHEAR_STIFFNESS: f32 = 80.0;
const BEND_STIFFNESS: f32 = 20.0;
const STRUCTURAL_DAMPING: f32 = 1.0;
const SHEAR_DAMPING: f32 = 1.0;
const BEND_DAMPING: f32 = 0.4;



// =========== main globe ============
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2]
}
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 1: normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 2: uv
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    light_pos: [f32; 4],          // maps to vec4<f32> in WGSL
    ks: f32,                  // specular strength & shininess exponent
    shininess: f32,           // α-shininess exponent
    compute_specular: u32,    // whether to use specular component
    _pad: u32,                // padding to 16-byte alignment
}


// ============ CLOTH =================

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    model_matrix: [f32; 16],  // 4x4 matrix
    velocity: [f32; 4],       // velocity vector (x, y, z, w)
    force: [f32; 4],          // force vector (x, y, z, w)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Spring {
    p0: u32,
    p1: u32,
    prev_length: f32,
    spring_type: u32, // 0 = structural, 1 = shear, 2 = bend
    // _pad: u32,        // alignment
    force: [f32; 4],  // initialized to [0.0;4]
}



#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimulationData {
    dt: f32,
    radius: f32,
    globe_radius: f32,
    mass: f32,
    grid_width: u32,
    gravity: f32,
    speed_damp: f32,
    _pad2: f32,
}



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsConstants {
    k_struct: f32,
    k_shear: f32,
    k_bend: f32,
    //
    k_damp_struct: f32,
    k_damp_shear: f32,
    k_damp_bend: f32,
    //
    rest_len_struct: f32,
    rest_len_shear: f32,
    rest_len_bend: f32,
    //
    k_contact: f32,
    //
    mu: f32,
    //
    _pad0: f32,
}

// == Init buffers ===
struct InitVars {
    init_light_uniform: LightUniform,
    init_sim_data: SimulationData,
    init_physics_constants: PhysicsConstants,
}

    // force_p1: vec4<f32>, // TODO: remove since fp1 = -fp0


// ========== APP ==============
pub struct ClothSimApp {
    // ======= GLOVE =========

    // Globe geometry
    globe_vertex_buffer: wgpu::Buffer,
    globe_index_buffer: wgpu::Buffer,
    globe_num_indices: u32,
    stack_count: usize,
    sector_count: usize,
    //
    // globe render pipeline
    globe_pipeline: wgpu::RenderPipeline,

    // Bind gorups
    camera: OrbitCamera,
    texture_bind_group: wgpu::BindGroup,
    light_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    

    // Vars
    globe_radius: f32,



    // ==== Cloth =====

    // cloth geometry

    // particles 
    instance_buffer: wgpu::Buffer,
    instance_count: u32,

    // Springs
    structural_springs_buffer: wgpu::Buffer,
    shear_springs_buffer: wgpu::Buffer,
    bend_springs_buffer: wgpu::Buffer,
    structural_count: u32,
    shear_count: u32,
    bend_count: u32,

    // cloth
    cloth_particles_texture_bind_group: wgpu::BindGroup, //particules inst = red
    cloth_pipeline: wgpu::RenderPipeline,

    // Vars
    cloth_particle_radius: f32,


    // ===== Computing ======
    // Buffers
    sim_data_buffer: wgpu::Buffer,
    physics_constants_buffer: wgpu::Buffer, 

    // Compute movement
    compute_pipeline: wgpu::ComputePipeline,
    // compute_bind_group: wgpu::BindGroup,
    
    // Compute forces
    springs_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::ComputePipeline, 
    spring_and_forces_bind_group: wgpu::BindGroup,
    
    // Physics
    time_scale:f32,
    gravity: f32,
    speed_damp: f32,


    // ==== UI ======
    fps: f32,
    light_pos: [f32; 4],
    checkbox_specular: bool,   // ui checkbox
    ks: f32,
    shininess: f32,

}




impl ClothSimApp {
    pub fn new(context: &Context) -> Self {

        let InitVars { init_light_uniform, init_sim_data, init_physics_constants } = Self::init_vars();

        // Camera Setup
        let camera = Self::setup_camera(context);
        let camera_bind_group_layout = context.device().create_bind_group_layout(&CameraUniform::desc());


        // ================
        //   GLobe stuff 
        // ================
        // 1. Generate geometry
        let (vertices, indices, globe_num_indices) = Self::create_sphere_geometry();

        // 2. gpu buff
        let globe_vertex_buffer = Self::create_vertex_buffer(context, &vertices);
        let globe_index_buffer = Self::create_index_buffer(context, &indices);

        // 3. Bind Groups lyouts  - GROUP(1) Texture & Sampler
        let texture_bind_group_layout = Self::create_texture_bind_group_layout(context);
        
        // 4. Textures
        let texture_bind_group = Self::load_texture_and_create_bind_group(
            context,
            &texture_bind_group_layout,
            TEXTURE_FILE,
        );

        // 5. Lighting
        let light_bind_group_layout = Self::create_light_bind_group_layout(context);
        
        let (light_buffer, light_bind_group) = Self::create_light_resources(
            context,
            &light_bind_group_layout,
            &init_light_uniform
        );


        // 5. Render Pipeline
        let globe_pipeline = Self::create_globe_render_pipeline(
            context,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &light_bind_group_layout,
        );



        // ============================
        //       Cloth stuff 
        // ============================

        // 1. Generate geometry
        // let (cloth_vertices, cloth_indices, _num_cloth_indices) = Self::create_cloth_geometry();

        // 2. gpu buff
        // let cloth_vertex_buffer = Self::create_vertex_buffer(context, &cloth_vertices);
        // let cloth_index_buffer = Self::create_index_buffer(context, &cloth_indices);
        // let cloth_velocities_buffer = Self::create_storage_buffer(context, &[]); // TODO: fill with velocities

        // 2.x Instance buffer
        let instances = Self::generate_instances(CLOTH_PARTICLES_PER_SIDE, CLOTH_PARTICLES_RADIUS);
        let instance_count: u32 = CLOTH_PARTICLES_PER_SIDE * CLOTH_PARTICLES_PER_SIDE; // 10x10 grid
        let instance_buffer = Self::create_instance_buffer(context, &instances);  // then called in render pipeline !
        let pos0 = (
            instances[0].model_matrix[12],
            instances[0].model_matrix[13],
            instances[0].model_matrix[14],
        );
        // println!("Instance[0] translation = {:?}", pos0);
        // println!("Instance[0] matrix first 8 elems = {:?}", &instances[0].model_matrix[0..8]);
        // println!("Instance[0] matrix last 8 elems  = {:?}", &instances[0].model_matrix[8..16]);
        // println!("Particle size = {}", std::mem::size_of::<Particle>());


        // 2.y Springs buffers
        let (structural_list, shear_list, bend_list) = Self::generate_spring_lists(&instances, CLOTH_PARTICLES_PER_SIDE as usize, CLOTH_PARTICLES_PER_SIDE as usize);
        let structural_count = structural_list.len() as u32;
        let shear_count = shear_list.len() as u32;
        let bend_count = bend_list.len() as u32;
        let total_springs = structural_count + shear_count + bend_count;
        let structural_springs_buffer = Self::create_springs_buffer(context, &structural_list);
        let shear_springs_buffer = Self::create_springs_buffer(context, &shear_list);
        let bend_springs_buffer = Self::create_springs_buffer(context, &bend_list);

        // 3. Textures for red cloth particles
        let cloth_particles_texture_bind_group = Self::load_texture_and_create_bind_group(
            context,
            &texture_bind_group_layout,
            CLOTH_PARTICLES_TEXTURE_FILE,
        );
        // Texture for cloth mesh
        // let cloth_texture_bind_group = Self::load_texture_and_create_bind_group(
        // 
        let cloth_pipeline = Self::create_cloth_render_pipeline(
            context,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            // &light_bind_group_layout,
        );

            
        // ===========================
        //       Compute Setup
        // ===========================
        // Buffers
        let physics_constants_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Physics Constants Buffer"),
            contents: bytemuck::bytes_of(&init_physics_constants),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sim_data_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Data Storage Buffer"),
            contents: bytemuck::bytes_of(&init_sim_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Springs & Compute forces
        let spring_and_forces_bind_group_layout = Self::create_spring_and_forces_bind_group_layout(context);
        let spring_and_forces_bind_group = Self::create_spring_and_forces_bind_group(
            context,
            &spring_and_forces_bind_group_layout,
            &physics_constants_buffer,
            &sim_data_buffer,
            &instance_buffer,
            &structural_springs_buffer,
            &shear_springs_buffer,
            &bend_springs_buffer,
        );
        let (springs_pipeline, accumulate_pipeline) = 
            Self::create_forces_pipeline(context, &spring_and_forces_bind_group_layout, FORCES_SHADER_FILE, );

        // Compute movement 
        let compute_pipeline = Self::create_compute_pipeline(
            context,
            &spring_and_forces_bind_group_layout,
            COMPUTE_SHADER_FILE,
            "Compute Movement Pipeline",
        );
        // let compute_bind_group = spring_and_forces_bind_group.clone();

        Self {
            // ======= GLOVE =========

            // Globe geometry
            globe_vertex_buffer,
            globe_index_buffer,
            globe_num_indices,
            stack_count: STACK_COUNT,
            sector_count: SECTOR_COUNT,
            // globe render pipeline
            globe_pipeline,

            // Bind Groups
            camera,
            texture_bind_group,
            light_bind_group,
            light_buffer,
            
            // vars
            globe_radius: RADIUS,


            // ==== Cloth =====
            // particles 
            instance_buffer,
            instance_count,
            // Springs
            structural_springs_buffer,
            shear_springs_buffer,
            bend_springs_buffer,
            structural_count,
            shear_count,
            bend_count,
            // cloth
            cloth_particles_texture_bind_group,
            cloth_pipeline,

            // vars
            cloth_particle_radius: CLOTH_PARTICLES_RADIUS,
            
            
            // ===== Computing ======
            // Buffers
            sim_data_buffer,
            physics_constants_buffer,

            // Compute movement
            compute_pipeline,
            // compute_bind_group,
            
            // Compute forces
            springs_pipeline,
            accumulate_pipeline,
            spring_and_forces_bind_group,

            // Physics vars
            time_scale: TIME_SCALE,
            gravity: GRAVITY,
            speed_damp: SPEED_DAMP,

            // ==== UI ======
            fps: 0.0,
            light_pos: [LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2], 0.0],
            checkbox_specular: true,
            ks: KS,
            shininess: SHININESS,
            
        }

    // --- end of new() ---
    }

    // =============================================
    //           Helpers
    // =============================================

    
    fn init_vars() -> InitVars {
        InitVars {
            init_light_uniform: LightUniform {
                light_pos: LIGHT_POS,
                ks: KS,
                shininess: SHININESS,
                compute_specular: 1u32,
                _pad: _PAD,
            },
            init_sim_data: SimulationData {
                dt: TIME_SCALE,
                radius: CLOTH_PARTICLES_RADIUS,
                globe_radius: RADIUS,
                mass: MASS,
                grid_width: CLOTH_PARTICLES_PER_SIDE,
                gravity: GRAVITY,
                speed_damp: SPEED_DAMP,
                _pad2: 0.0,
                
            },
            init_physics_constants: PhysicsConstants {
                k_struct: STRUCTURAL_STIFFNESS,
                k_shear: SHEAR_STIFFNESS,
                k_bend: BEND_STIFFNESS,
                k_damp_struct: STRUCTURAL_DAMPING,
                k_damp_shear: SHEAR_DAMPING,
                k_damp_bend: BEND_DAMPING,
                rest_len_struct: CLOTH_SIZE / (CLOTH_PARTICLES_PER_SIDE as f32 - 1.0),
                rest_len_shear: (CLOTH_SIZE / (CLOTH_PARTICLES_PER_SIDE as f32 - 1.0)) * (2.0f32).sqrt(),
                rest_len_bend: (CLOTH_SIZE / (CLOTH_PARTICLES_PER_SIDE as f32 - 1.0)) * 2.0,
                k_contact: COLLISION_K,
                mu: FRICTION_COEFF,
                _pad0: 0.0,
            },
        }
    }

    //Camera
    fn setup_camera(context: &Context) -> OrbitCamera {
        let mut camera = OrbitCamera::new(
            context,
            45.0,
            context.size().x / context.size().y,
            0.1,
            100.0,
        );
        camera
            .set_target(cgmath::point3(0.0, 0.0, 0.0))
            .set_polar(cgmath::point3(DEFAULT_ZOOM, 0.0, 0.0))
            .update(context);
        camera
    }
    
    // Globe
    fn create_sphere_geometry() -> (Vec<Vertex>, Vec<u32>, u32) {
        let (raw_vertices, indices) =
            crate::sphere_vertices::generate_uv_sphere(RADIUS, STACK_COUNT, SECTOR_COUNT);
        
        let vertices: Vec<Vertex> = raw_vertices
            .into_iter()
            .map(|(pos, normal, uv)| Vertex { position: pos, normal, uv })
            .collect();
        
        let globe_num_indices = indices.len() as u32;
        (vertices, indices, globe_num_indices)
    }
    fn create_vertex_buffer(context: &Context, vertices: &[Vertex]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
    fn create_index_buffer(context: &Context, indices: &[u32]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    }
    fn create_texture_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                // binding 0: texture view
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
    fn load_texture_and_create_bind_group(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        texture_path: &str,
        ) -> wgpu::BindGroup {
        let img_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(texture_path);
        let img = image::open(&img_path)
            .expect("failed to load texture")
            .to_rgba8();
        
        let (width, height) = img.dimensions();
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("diffuse_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        context.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );
        
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = context.device().create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    }
    fn create_globe_render_pipeline(
        // TRIANGLE LIST
        context: &Context,
        camera_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
        ) -> wgpu::RenderPipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(GLOBE_SHADER_FILE);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("failed to read shader file");
        
        let shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("globe_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[camera_layout, texture_layout, light_layout],
            push_constant_ranges: &[],
        });
        
        context.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc()//,
                    // Self::instance_buffer_layout()
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: context.depth_stencil_format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }
    fn create_light_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }
    fn create_light_resources(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        init_light_uniform: &LightUniform,
        ) -> (wgpu::Buffer, wgpu::BindGroup) {

        // Light init params
        // init_light_uniform used down here but defined in init_vars()

        let light_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::bytes_of(init_light_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group = context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        (light_buffer, light_bind_group)
    }
    fn update_light_uniform(&self, context: &Context) {
        // check if need to update light uniform
        let compute_specular = if self.checkbox_specular { 1u32 } else { 0u32 };
        let updated_light = LightUniform {
            light_pos: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
            ks: self.ks,
            shininess: self.shininess,
            compute_specular,
            _pad: _PAD,
            
        };
        println!("Updating light: ks={}, shininess={}, compute_specular={}", 
                self.ks, self.shininess, compute_specular);
        context.queue().write_buffer(
            &self.light_buffer,
            0,
            bytemuck::bytes_of(&updated_light),
        );
    }





    // ====== Cloth ===========

    // 2.x Instance buffer
    fn generate_instances(count: u32, radius: f32) -> Vec<Particle> {
        use cgmath::{Matrix4, Vector3};

        let spacing = CLOTH_SIZE / (count as f32 - 1.0);
        let central_pos = CLOTH_CENTRAL_POS;
        let spawn_height = central_pos[1];

        let scale_factor = radius/RADIUS;
        
        let mut out = Vec::new();

        for i in 0..count {
            for j in 0..count {
                let x = (i as f32 - count as f32 / 2.0) * spacing;
                let y = spawn_height;
                let z = (j as f32 - count as f32 / 2.0) * spacing;
                let trans = Matrix4::from_translation(Vector3::new(x, y, z));
                let scale = Matrix4::from_scale(scale_factor);
                let model = trans * scale;
                let c0 = model.x;
                let c1 = model.y;
                let c2 = model.z;
                let c3 = model.w;

                // Initial speed
                let speed = 0.0;
                let vx = speed; //rng.gen_range(-speed..speed);
                let vy = speed; //rng.gen_range(-speed..speed);
                let vz = speed; //rng.gen_range(-speed..speed);


                out.push(Particle {
                    model_matrix: [
                        c0.x, c0.y, c0.z, c0.w,
                        c1.x, c1.y, c1.z, c1.w,
                        c2.x, c2.y, c2.z, c2.w,
                        c3.x, c3.y, c3.z, c3.w,
                    ],
                    velocity: [vx, vy, vz, 0.0],
                    force: [0.0, 0.0, 0.0, 0.0],
                });
            }
        }

        out
    }
    fn create_instance_buffer(context: &Context, instances: &[Particle]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX 
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        })
    }

    // 2.y Springs buffers
    // Generate three spring lists and compute per-spring prev_length from initial instances positions.
    // instances: Vec<Particle> where Particle.model_matrix[12..15] contain translation (row-major index 12..14 or [3][0..2] style).
    fn generate_spring_lists(instances: &Vec<Particle>, grid_w: usize, grid_h: usize) 
        -> (Vec<Spring>, Vec<Spring>, Vec<Spring>)
        {
        let mut structural: Vec<Spring> = Vec::new();
        let mut shear: Vec<Spring> = Vec::new();
        let mut bend: Vec<Spring> = Vec::new();

        // helper to compute position of particle index i (reads instance.model_matrix translation)
        let get_pos = |idx: usize| -> [f32;3] {
            let m = &instances[idx].model_matrix;
            // model_matrix stored row-major 16 floats: translation is at indices 12,13,14 (3rd column if row-major)
            // adjust if your matrix layout differs; here we follow your Particle.model_matrix usage in WGSL [3][0..2].
            [ m[12], m[13], m[14] ]
        };

        let push_unique = |vec: &mut Vec<Spring>, p: usize, q: usize, stype: u32| {
            if p == q { return; }
            if q < p { return; } // canonicalize
            // compute rest length from initial positions
            let pa = get_pos(p);
            let pb = get_pos(q);
            let dx = pb[0] - pa[0];
            let dy = pb[1] - pa[1];
            let dz = pb[2] - pa[2];
            let rest = (dx*dx + dy*dy + dz*dz).sqrt();
            vec.push(Spring {
                p0: p as u32,
                p1: q as u32,
                prev_length: rest,
                spring_type: stype,
                force: [0.0; 4],
            });
        };

        for r in 0..grid_h {
            for c in 0..grid_w {
                let i = r * grid_w + c;
                // structural: right, down
                if c + 1 < grid_w { push_unique(&mut structural, i, i + 1, 0); }
                if r + 1 < grid_h { push_unique(&mut structural, i, i + grid_w, 0); }
                // shear: down-right, down-left
                if r + 1 < grid_h && c + 1 < grid_w { 
                    push_unique(&mut shear, i, i + grid_w + 1, 1); 
                }
                if r + 1 < grid_h && c >= 1 {
                    let q = (r + 1) * grid_w + (c - 1);
                    push_unique(&mut shear, i, q, 1);
                }
                // bend: two-right, two-down
                if c + 2 < grid_w { push_unique(&mut bend, i, i + 2, 2); }
                if r + 2 < grid_h { push_unique(&mut bend, i, i + 2 * grid_w, 2); }
            }
        }

        (structural, shear, bend)
    }
    fn create_springs_buffer(context: &Context, springs: &[Spring]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Springs Buffer"),
            contents: bytemuck::cast_slice(springs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress, // 64
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 16, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
    
    fn create_cloth_render_pipeline(
        // TRIANGLE LIST
        // NO LIGHTING !!
        context: &Context,
        camera_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        // light_layout: &wgpu::BindGroupLayout,
        ) -> wgpu::RenderPipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(CLOTH_SHADER_FILE);
        let shader_src = std::fs::read_to_string(&shader_path).expect("failed to read shader file");
        
        let shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloth_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[camera_layout, texture_layout],
            push_constant_ranges: &[],
        });
        
        context.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(),
                    Self::instance_buffer_layout()
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: context.depth_stencil_format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }




    // =========== COMPUTE =================
    // Springs and Forces
    fn create_spring_and_forces_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forces_bind_group_layout"),
            entries: &[
                // binding 0: PhysicsConstants uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: SimulationData storage (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: Particles storage buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                // --------------- SPRINGS ----------------
                // 3 structural (rw)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 4 shear (rw)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // 5 bend (rw)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        })
    }
    fn create_spring_and_forces_bind_group(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        physics_constants_buffer: &wgpu::Buffer,
        sim_data_buffer: &wgpu::Buffer,
        particles_buffer: &wgpu::Buffer,
        structural_buf: &wgpu::Buffer,
        shear_buf: &wgpu::Buffer,
        bend_buf: &wgpu::Buffer,
        ) -> wgpu::BindGroup {
        context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forces_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {binding: 0, resource: physics_constants_buffer.as_entire_binding(),},
                wgpu::BindGroupEntry {binding: 1, resource: sim_data_buffer.as_entire_binding(),},
                wgpu::BindGroupEntry {binding: 2, resource: particles_buffer.as_entire_binding(),},
                // Springs
                wgpu::BindGroupEntry { binding: 3, resource: structural_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: shear_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bend_buf.as_entire_binding() },
            ],
        })
    }
    fn create_forces_pipeline(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        shader_file: &str,
        ) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(shader_file);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("Failed to read compute shader");

        let compute_shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Forces Pipeline Layout"),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });

        let springs_pipeline = context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("springs_pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "compute_springs",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let accumulate_pipeline = context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("accumulate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "accumulate_forces",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (springs_pipeline, accumulate_pipeline)
    }
    
    // Movement
    fn create_compute_bind_group_layout(
        context: &Context,
        uniform_type: wgpu::BufferBindingType,
        storage_type: wgpu::BufferBindingType,
        ) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: uniform_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: storage_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    fn create_compute_resources<T: bytemuck::Pod>(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        instance_buffer: &wgpu::Buffer,
        uniform: &T,
        uniform_label: &str,
        bind_group_label: &str,
        ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let uniform_buffer = context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(uniform_label),
                contents: bytemuck::bytes_of(uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        let bind_group = context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(bind_group_label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });
    
        (uniform_buffer, bind_group)
    }
    fn create_compute_pipeline(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        shader_file: &str,
        label: &str,
        ) -> wgpu::ComputePipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(shader_file);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("Failed to read compute shader");
    
        let compute_shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    
        context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
    

    fn dispatch_compute(&self, context: &Context) {
        let mut encoder = context.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            }
        );
        

        // 1. Compute Forces
        {
            let total_springs = self.structural_count + self.shear_count + self.bend_count;
            let wg_size = 64u32;
            let springs_wg = (total_springs + wg_size - 1) / wg_size;

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute_springs_pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.springs_pipeline);
                cpass.set_bind_group(0, &self.spring_and_forces_bind_group, &[]);
                cpass.dispatch_workgroups(springs_wg, 1, 1);
            }

            // Per-particle accumulation
            let particle_wg = (self.instance_count + wg_size - 1) / wg_size;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("accumulate_forces_pass"), timestamp_writes: None });
                cpass.set_pipeline(&self.accumulate_pipeline);
                cpass.set_bind_group(0, &self.spring_and_forces_bind_group, &[]);
                cpass.dispatch_workgroups(particle_wg, 1, 1);
            }
        }
        // 2. Compute Movement
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Movement Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.spring_and_forces_bind_group, &[]);
            
            let workgroup_count = (self.instance_count + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        context.queue().submit(Some(encoder.finish()));
    }


} // end



impl App for ClothSimApp {
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {

        // Note : 
        //      - Vertex buffer : Geometry for 1 sphere (positions, normals, uvs)
        //      - Index buffer : Triangle indices for 1 sphere
        //      - ...
        //      - Use those buffers to compute sphere geometry, then draw them as many times as we need.
        //      - They are drawing Models we pass to our pipelines 
        //      - ...
        //      - Instance buffer : updated position of each cloth particle 

        // =====================
        //     MAIN GLOBE
        //======================
        // Globe render pipeline
        render_pass.set_pipeline(&self.globe_pipeline);
        // Globe vertex&Index buffers
        render_pass.set_vertex_buffer(0, self.globe_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.globe_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        // Bind groupes (camera, texture, light)
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(2, &self.light_bind_group, &[]);
        // Draw main Globe
        render_pass.draw_indexed(0..self.globe_num_indices, 0, 0..1);

        //=========================
        //     CLOTH PARTICLES
        //=========================
        // NOTE : for cloth = mesh of globes, we just reuse vertex and index buffer of main globe
        // Cloth render pipeline
        render_pass.set_pipeline(&self.cloth_pipeline);
        // Cloth vertex buffer -- same as globe
        render_pass.set_vertex_buffer(0, self.globe_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.globe_index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        // INSTANCE buffer
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        //Bind grps
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.set_bind_group(1, &self.cloth_particles_texture_bind_group, &[]);

        // Draw
        render_pass.draw_indexed(0..self.globe_num_indices, 0, 0..self.instance_count);
        
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        egui::Window::new("Params").show(egui_ctx, |ui| {

            // ====== CAMERA ======
            ui.heading("Camera");
            let mut zoom = self.camera.radius();
            if ui.add(egui::Slider::new(&mut zoom, 15.0..=100.0).text("Zoom")).changed() {
                self.camera.set_radius(zoom).update(context);
            }

            // ====== LIGHT =======
            ui.heading("Light");
            let mut light_changed = false;
            
            // to do : hitting checkbox completely deactivates specular (cannot reactivate)
            light_changed |= ui.checkbox(&mut self.checkbox_specular, "Specular").changed();

            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[0], -50.0..=50.0).text("Light X")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[1], -50.0..=50.0).text("Light Y")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[2], -50.0..=50.0).text("Light Z")).changed();
            
            ui.add_space(5.0);
            light_changed |= ui.add(egui::Slider::new(&mut self.ks, 0.0..=2.0).text("Specular (ks)")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.shininess, 1.0..=512.0).text("Shininess")).changed();

            // Update GPU buffer if any light param changed
            if light_changed {
                self.update_light_uniform(context);
            }
            ui.separator();



            // ====== Physics ======
            ui.heading("Physics");
            ui.add(egui::Slider::new(&mut self.gravity, -20.0..=10.0).text("Gravity Y"));
            ui.add(egui::Slider::new(&mut self.time_scale, 0.0..=2.0).text("Time Scale"));
            ui.add(egui::Slider::new(&mut self.speed_damp, 0.0..=1.0).text("Speed Damping"));
            ui.separator();



            // ====== Geometry ======
            ui.heading("Geometry");
            if ui.add(egui::Slider::new(&mut self.cloth_particle_radius, 0.1..=4.0).text("Cloth Particle Radius")).changed() {
                let new_instances = Self::generate_instances(CLOTH_PARTICLES_PER_SIDE, self.cloth_particle_radius);
                // Update the instance buffer
                context.queue().write_buffer(
                    &self.instance_buffer,
                    0,
                    bytemuck::cast_slice(&new_instances),
                );
            }


            ui.label(format!("Stacks: {}", self.stack_count));
            ui.label(format!("Sectors: {}", self.sector_count));
            ui.label(format!("Vertices: {}", (self.stack_count + 1) * (self.sector_count + 1)));
            ui.separator();
            


            // FPS
            ui.label(format!("FPS: {}", self.fps.round()));
            // Other
            ui.label(format!("Instance count: {}", self.instance_count));

        });
    }


    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input, context);
    }

    fn update(&mut self, delta_time: f32, context: &Context) {
        self.fps = 1.0 / delta_time;

        const TARGET_DT: f32 = 1.0 / HZ;
        const MAX_SUBSTEPS: u32 = 8;
        let scaled_time = self.time_scale * delta_time; // UI scaling
    
        // Compute number of substeps needed (at least 1)
        let mut num_steps = (scaled_time / TARGET_DT).ceil() as u32;
        if num_steps == 0 { num_steps = 1; }
        if num_steps > MAX_SUBSTEPS { num_steps = MAX_SUBSTEPS; }

        // Per-substep dt (divide total time by steps)
        let substep_dt = scaled_time / num_steps as f32;


        for _ in 0..num_steps {
            let sim_uniform = SimulationData {
            dt: substep_dt,
            radius: self.cloth_particle_radius,
            globe_radius: self.globe_radius,
            mass: MASS,
            grid_width: CLOTH_PARTICLES_PER_SIDE,
            gravity: self.gravity,
            speed_damp: self.speed_damp,
            _pad2: 0.0,
            
        };

        // Update buffers
        context.queue().write_buffer(&self.sim_data_buffer,0,bytemuck::bytes_of(&sim_uniform),);
        // context.queue().write_buffer(&self.physics_constants_buffer,0,bytemuck::bytes_of(&updated_physics_constants),);


        self.dispatch_compute(context);
        }
    }

    fn resize(&mut self, new_width: u32, new_height: u32, context: &Context) {
        self.camera
            .set_aspect(new_width as f32 / new_height as f32)
            .update(context);
    }

}


// ----------------------------------------------------------------------------------------

// spring :
    // stiffness K list for diff springs [K_structural, K_shear, K_bend], same for Lrest, just use a reference to this list
    // buffer : [particle0, p1, K0, L_rest, Fhook] + damping we just need [Lprev] and copute Fdamp (delta L / delta T) and add to Fhook
    
// compute.wgsl :
    // buffer : pos, vel
    // compute : for ech particle, read connected spring F force, add F_collision, add gravity, update vel and pos based on them


// Globe easier than cube (collision = dist(particle, globe center) - radius ?>? positive or negative ? if negative, then collision)
// cube needs quadrants for each of 6 faces



// Pipelines : 

    // for now cloth = multiple instances of cubes/globes with forces between them
    // later : single cloth mesh (no particles just positions) with same forces

    // 2 Render pipeline  for globe_cloth and big globe
    // later 1 render for cloth and globe

    // compute pipeline; see up, better one for springs and one for position compute


