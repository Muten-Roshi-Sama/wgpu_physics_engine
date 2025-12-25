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

const GLOBE_SHADER_FILE: &str = "globe_shader.wgsl";
const CLOTH_SHADER_FILE: &str = "cloth_shader.wgsl";
const COMPUTE_SHADER_FILE: &str = "compute_movement.wgsl";

// const TEXTURE_FILE: &str = "../../textures/texture.png";
// const TEXTURE_FILE: &str = "../../textures/earth2048.bmp";
// const TEXTURE_FILE: &str = "../../textures/moon1024.bmp";
const TEXTURE_FILE: &str = "../../textures/grey.png";

// Specular light parameters
const LIGHT_POS: [f32; 4] = [2.0, 2.0, 2.0, 0.0];
const KS: f32 = 0.15;
const SHININESS: f32 = 128.0;
const _PAD: [f32;2] = [0.0, 0.0];

// Globe geometry
const RADIUS: f32 = 1.0;
const STACK_COUNT: usize = 64;
const SECTOR_COUNT: usize = 128;

// Physics
const NUM_PARTICLES: u32 = 10;
const PARTICLE_SCALE : f32 = 1.0;
const TIME_SCALE: f32 = 1.0;
const GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];
// const DAMPING: f32 = 0.95;

// Cloth 
const CLOTH_SIZE: f32 = 5.0;
const CLOTH_POS: [f32;3] = [0.0, 2.0, 0.0];
const MASS: f32 = 10.0;

// Springs
const VERTEX_MASS: f32 = 0.16;
const STRUCTURAL_STIFFNESS: f32 = 150.0;
const SHEAR_STIFFNESS: f32 = 5.0;
const BEND_STIFFNESS: f32 = 15.0;
const STRUCTURAL_DAMPING: f32 = 1.5;
const SHEAR_DAMPING: f32 = 0.05;
const BEND_DAMPING: f32 = 0.15;



// =========== STRUCTS & IMPL ============
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    light: [f32; 4],   // maps to vec4<f32> in WGSL
    ks: f32,           // specular strength
    shininess: f32,    // shininess exponent
    _pad: [f32; 2],    // padding to 16-byte alignment
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





// CLOTH
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimulationUniform {
    dt: f32,
    // bounds: f32,
    // damping: f32,
    radius: f32,
    gravity: [f32; 3],
    _pad1: f32,
}






// ========== APP ==============
pub struct ClothSimApp {
    // Rendering
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    num_indices: u32,

    // Bind gorups
    camera: OrbitCamera,
    texture_bind_group: wgpu::BindGroup,
    light_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    fps: f32,

    // Eframe
    light_pos: [f32; 3],
    ks: f32,
    shininess: f32,
    stack_count: usize,   // TODO: for now, display only....
    sector_count: usize,  // TODO: ... regenerating requires rebuilding buffers
}

impl ClothSimApp {
    pub fn new(context: &Context) -> Self {

        // 1. Generate geometry
        let (vertices, indices, num_indices) = Self::create_sphere_geometry();
        // let (cloth_vertices, cloth_indices, _num_cloth_indices) = Self::create_cloth_geometry();

        // 2. gpu buff
        let vertex_buffer = Self::create_vertex_buffer(context, &vertices);
        let index_buffer = Self::create_index_buffer(context, &indices);
        //
        // let cloth_vertex_buffer = Self::create_vertex_buffer(context, &cloth_vertices);
        // let cloth_index_buffer = Self::create_index_buffer(context, &cloth_indices);
        // let cloth_velocities_buffer = Self::create_storage_buffer(context, &[]); // TODO: fill with velocities


        // 3. Bind Groups lyouts   
            /* 
            - GROUP(0) Camera
            - GROUP(1) Texture & Sampler
            - GROUP(2) Light Layout
            */ 
        let camera_bind_group_layout = context.device().create_bind_group_layout(&CameraUniform::desc());
        let texture_bind_group_layout = Self::create_texture_bind_group_layout(context);


        // 4. Light Bind Group, Buffer and Uniform
        let light_bind_group_layout = Self::create_light_bind_group_layout(context);
        
        let (light_buffer, light_bind_group) = Self::create_light_resources(
            context,
            &light_bind_group_layout,
        );

        // 5. Textures
        let texture_bind_group = Self::load_texture_and_create_bind_group(
            context,
            &texture_bind_group_layout,
            TEXTURE_FILE,
        );

        // 6. Render Pipeline
        let render_pipeline = Self::create_render_pipeline(
            context,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &light_bind_group_layout,
        );
        // let cloth_pipeline = Self::create_cloth_render_pipeline(
        //     context,
        //     &camera_bind_group_layout,
        //     &texture_bind_group_layout,
        // );

        // 7. Camera
        let camera = Self::setup_camera(context);

        // 8. Compute Setup
        // let compute_bind_group_layout = Self::create_compute_bind_group_layout(context);
        // let (sim_uniform_buffer, compute_bind_group) = Self::create_compute_resources(context, &compute_bind_group_layout, &instance_buffer);
        // let compute_pipeline = Self::create_compute_pipeline(context, &compute_bind_group_layout);



        Self {
            vertex_buffer,
            index_buffer,
            render_pipeline,
            num_indices,
            camera,
            texture_bind_group,
            light_bind_group,
            light_buffer,
            fps: 0.0,
            light_pos: [LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2]],
            ks: KS,
            shininess: SHININESS,
            stack_count: STACK_COUNT,
            sector_count: SECTOR_COUNT,
        }

    // --- end of new() ---
    }


    // =============================================
    //           Helpers
    // =============================================

    // 1. Geometry
    fn create_sphere_geometry() -> (Vec<Vertex>, Vec<u32>, u32) {
        let (raw_vertices, indices) =
            crate::sphere_vertices::generate_uv_sphere(RADIUS, STACK_COUNT, SECTOR_COUNT);
        
        let vertices: Vec<Vertex> = raw_vertices
            .into_iter()
            .map(|(pos, normal, uv)| Vertex { position: pos, normal, uv })
            .collect();
        
        let num_indices = indices.len() as u32;
        (vertices, indices, num_indices)
    }

    // fn create_cloth_geometry() -> (Vec<Vertex>, Vec<u32>, u32) {

    //     let mut cloth_vertices = Vec::new();
    //     let mut cloth_indices: Vec<u16> = Vec::new();

    //     // create the vertices
    //     for i in 0..N_CLOTH_VERTICES_PER_ROW {
    //         for j in 0..N_CLOTH_VERTICES_PER_ROW {
    //             cloth_vertices.push(Vertex {
    //                 position: [
    //                     CLOTH_CENTER_X + i as f32 * (CLOTH_SIZE / (N_CLOTH_VERTICES_PER_ROW - 1) as f32) - (CLOTH_SIZE / 2.0),
    //                     CLOTH_CENTER_Y,
    //                     CLOTH_CENTER_Z + j as f32 * (CLOTH_SIZE / (N_CLOTH_VERTICES_PER_ROW - 1) as f32) - (CLOTH_SIZE / 2.0),
    //                 ],
    //                 normal: [0.0, 0.0, 0.0],
    //                 tangent: [0.0, 0.0, 0.0],
    //                 tex_coords: [
    //                     i as f32 * (1.0 / (N_CLOTH_VERTICES_PER_ROW - 1) as f32),
    //                     j as f32 * (1.0 / (N_CLOTH_VERTICES_PER_ROW - 1) as f32),
    //                 ],
    //             });
    //         }
    //     }

    //     // create the indices
    //     for i in 0..N_CLOTH_VERTICES_PER_ROW - 1 {
    //         for j in 0..N_CLOTH_VERTICES_PER_ROW - 1 {
    //             // first triangle
    //             cloth_indices.push((i * N_CLOTH_VERTICES_PER_ROW + j) as u16);
    //             cloth_indices.push((i * N_CLOTH_VERTICES_PER_ROW + j + 1) as u16);
    //             cloth_indices.push(((i + 1) * N_CLOTH_VERTICES_PER_ROW + j) as u16);
    //             // second triangle
    //             cloth_indices.push((i * N_CLOTH_VERTICES_PER_ROW + j + 1) as u16);
    //             cloth_indices.push(((i + 1) * N_CLOTH_VERTICES_PER_ROW + j + 1) as u16);
    //             cloth_indices.push(((i + 1) * N_CLOTH_VERTICES_PER_ROW + j) as u16);
    //         }
    //     }

    //     // set the default speed of the cloth
    //     let mut cloth_velocities: Vec<Velocity> = Vec::new();
    //     for _i in cloth_vertices.iter_mut() {
    //         cloth_velocities.push(Velocity {
    //             velocity: [0.0, 0.0, 0.0],
    //         });
    //     }
    //     // Return
    //     let num__cloth_indices = cloth_indices.len() as u32;
    //     (cloth_vertices, cloth_indices, num_cloth_indices)
    // }

    // 2. Buffers
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
    fn create_storage_buffer(context: &Context, indices: &[u32]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }


    // 3. Bind Groups
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


    // 4. Light Bind Group, Buffer and Uniform
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
        ) -> (wgpu::Buffer, wgpu::BindGroup) {

        // Light init params
        let initial_light = LightUniform {
            light: LIGHT_POS,
            ks: KS,
            shininess: SHININESS,
            _pad: _PAD,
        };

        let light_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::bytes_of(&initial_light),
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


    // 5. Textures
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

    // 6. Render Pipeline
    fn create_render_pipeline(
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
                buffers: &[Vertex::desc()],
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
    
    // fn create_cloth_render_pipeline(
    //     // TRIANGLE LIST
    //     context: &Context,
    //     camera_layout: &wgpu::BindGroupLayout,
    //     texture_layout: &wgpu::BindGroupLayout,
    //     ) -> wgpu::RenderPipeline {
    //     let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(CLOTH_SHADER_FILE);
    //     let shader_src = std::fs::read_to_string(&shader_path)
    //         .expect("failed to read shader file");
        
    //     let shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
    //         label: Some("cloth_shader"),
    //         source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    //     });
        
    //     let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    //         label: Some("pipeline_layout"),
    //         bind_group_layouts: &[camera_layout, texture_layout, light_layout],
    //         push_constant_ranges: &[],
    //     });
        
    //     context.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    //         label: Some("render_pipeline"),
    //         layout: Some(&pipeline_layout),
    //         vertex: wgpu::VertexState {
    //             module: &shader,
    //             entry_point: "vs_main",
    //             buffers: &[Vertex::desc()],
    //             compilation_options: wgpu::PipelineCompilationOptions::default(),
    //         },
    //         fragment: Some(wgpu::FragmentState {
    //             module: &shader,
    //             entry_point: "fs_main",
    //             targets: &[Some(wgpu::ColorTargetState {
    //                 format: context.format(),
    //                 blend: Some(wgpu::BlendState::REPLACE),
    //                 write_mask: wgpu::ColorWrites::ALL,
    //             })],
    //             compilation_options: wgpu::PipelineCompilationOptions::default(),
    //         }),
    //         primitive: wgpu::PrimitiveState {
    //             topology: wgpu::PrimitiveTopology::TriangleList,
    //             strip_index_format: None,
    //             front_face: wgpu::FrontFace::Ccw,
    //             cull_mode: Some(wgpu::Face::Back),
    //             polygon_mode: wgpu::PolygonMode::Fill,
    //             unclipped_depth: false,
    //             conservative: false,
    //         },
    //         depth_stencil: Some(wgpu::DepthStencilState {
    //             format: context.depth_stencil_format(),
    //             depth_write_enabled: true,
    //             depth_compare: wgpu::CompareFunction::Less,
    //             stencil: wgpu::StencilState::default(),
    //             bias: wgpu::DepthBiasState::default(),
    //         }),
    //         multisample: wgpu::MultisampleState {
    //             count: 1,
    //             mask: !0,
    //             alpha_to_coverage_enabled: false,
    //         },
    //         multiview: None,
    //         cache: None,
    //     })
    // }


    // 7. Camera
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
            .set_polar(cgmath::point3(2.0, 0.0, 0.0))
            .update(context);
        camera
    }

    // 8.
    fn create_compute_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &[
                // @binding(0): SimulationData uniform
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
                // @binding(1): Particle storage buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    
    fn create_compute_resources(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        instance_buffer: &wgpu::Buffer,
        ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let sim_uniform = SimulationUniform {
            dt: TIME_SCALE,
            // bounds: BOUNDS,
            // damping: DAMPING,
            radius: RADIUS * PARTICLE_SCALE,
            // _pad0: 0.0,
            gravity: GRAVITY,
            _pad1: 0.0,
        };
    
        let sim_uniform_buffer = context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Uniform Buffer"),
                contents: bytemuck::bytes_of(&sim_uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        let compute_bind_group = context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });
    
        (sim_uniform_buffer, compute_bind_group)
    }
    
    fn create_compute_pipeline(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        ) -> wgpu::ComputePipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(COMPUTE_SHADER_FILE);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("Failed to read compute shader");
        
        let compute_shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    
        context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    // fn dispatch_compute(&self, context: &Context) {
    //     let mut encoder = context.device().create_command_encoder(
    //         &wgpu::CommandEncoderDescriptor {
    //             label: Some("Compute Encoder"),
    //         }
    //     );
        
    //     {
    //         let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //             label: Some("Compute Pass"),
    //             timestamp_writes: None,
    //         });
    //         compute_pass.set_pipeline(&self.compute_pipeline);
    //         compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            
    //         let workgroup_count = (self.instance_count + 63) / 64;
    //         compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    //     }
        
    //     context.queue().submit(Some(encoder.finish()));
    // }



    // ============== Update =====================
    fn update_light_uniform(&self, context: &Context) {
        let updated_light = LightUniform {
            light: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
            ks: self.ks,
            shininess: self.shininess,
            _pad: _PAD,
        };
        context.queue().write_buffer(
            &self.light_buffer,
            0,
            bytemuck::bytes_of(&updated_light),
        );
    }



}



impl App for ClothSimApp {
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Set pipeline, buffers
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        //Bind Groups
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(2, &self.light_bind_group, &[]);

        // Draw
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        egui::Window::new("Params").show(egui_ctx, |ui| {

            // Radius slider
            ui.heading("Camera");
            let mut radius = self.camera.radius();
            if ui.add(egui::Slider::new(&mut radius, 2.0..=10.0).text("radius")).changed() {
                self.camera.set_radius(radius).update(context);
            }

            // Light controls
            ui.heading("Light");
            let mut light_changed = false;
            
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[0], -5.0..=5.0).text("Light X")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[1], -5.0..=5.0).text("Light Y")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[2], -5.0..=5.0).text("Light Z")).changed();
            
            ui.add_space(5.0);
            light_changed |= ui.add(egui::Slider::new(&mut self.ks, 0.0..=2.0).text("Specular (ks)")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.shininess, 1.0..=512.0).text("Shininess")).changed();
            // Update GPU buffer if any light param changed
            if light_changed {
                let updated_light = LightUniform {
                    light: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
                    ks: self.ks,
                    shininess: self.shininess,
                    _pad: _PAD,
                };
                context.queue().write_buffer(
                    &self.light_buffer,
                    0,
                    bytemuck::bytes_of(&updated_light)
                );
            }
            
            ui.separator();


            // Geometry info (read-only for now)
            ui.heading("Geometry");
            ui.label(format!("Stacks: {}", self.stack_count));
            ui.label(format!("Sectors: {}", self.sector_count));
            ui.label(format!("Vertices: {}", (self.stack_count + 1) * (self.sector_count + 1)));
            
            ui.separator();
            


            // FPS
            ui.label(format!("FPS: {}", self.fps.round()));
            // Other

        });
    }


    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input, context);
    }

    fn update(&mut self, delta_time: f32, _context: &Context) {
        self.fps = 1.0 / delta_time;

        // TODO add simul and substep which is number of simul per delta time


    }

    fn resize(&mut self, new_width: u32, new_height: u32, context: &Context) {
        self.camera
            .set_aspect(new_width as f32 / new_height as f32)
            .update(context);
    }

}


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


