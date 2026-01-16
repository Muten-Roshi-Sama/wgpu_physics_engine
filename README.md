# Physics Engine Rust-wgpu :


This repo uses wgpu-bootstrap to easily set up the physics engine windows, camera and others : 


### 1. Cube :

```
>> cargo run -p cube
```


### 2. Textured Cube :
```
>> cargo run -p textured_cube
```


### 3. Globe (with Specular Lighting) :

```
>> cargo run -p globe_specular
```

<img src="img/gl_sphere02.png" width="35%">
<img src="img/gl_sphere01.png" width="35%">
from : https://songho.ca/opengl/gl_sphere.html




## Links & Ressources

- Wgpu starting guide : [Zdgeier](https://zdgeier.com/wgpuintro.html)

- Jack1232's Tutorial & Videos : [wgpu-step-by-step](https://github.com/jack1232/WebGPU-Step-By-Step)
- Rust-based Game Engine : [Bevy](https://github.com/bevyengine/bevy)


- Shader Offset computer : [wgsl](https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html#)




## Technical Overview :


### Shader file : 
- file containing GPU programs.
- Vertex Shader: Runs once per vertex. Transforms positions, passes data to the next stage.
- Fragment Shader: Runs once per pixel. Computes the final color for each pixel.
- Compute Shader : general computations (e.g. R/W in buffers).

### Pipeline

- A configuration object that tells the GPU how to use shaders, buffers, and other state.
- Render Pipeline: Used for drawing (vertex + fragment shaders).
- Compute Pipeline: Used for general-purpose computation (compute shaders).

### Buffers
- Vertex Buffer: Stores per-vertex data (positions, normals, uvs).
- Index Buffer: Stores indices for indexed drawing (which vertices make up each triangle).
- Uniform Buffer: Stores small, frequently-read data (camera matrices, lighting).
- Storage Buffer: Stores large, read/write data (particle positions, velocities).

#### Padding and offset rules

WGSL (and wgpu) require that uniform buffer structs follow strict alignment rules for compatibility with the GPU hardware.
- The total size of the struct must be a multiple of 16 bytes.

f32 : 4 bytes
vec2<f32> : 8 bytes (2 x 4)



### Bind Group
- A collection of resources (buffers, textures, samplers) bound together for use in shaders.








## Cloth simulation :

Display :
- One main textured globe
- A grid of globes acting as the nodes/particles of the cloth.



### 1. Compute objects geometry

- Vertex buffer: Contains the geometry for a single sphere (positions, normals, uvs for one sphere).
- Index buffer: Tells the GPU how to connect those vertices into triangles for that sphere.


Main idea : 

Compute sphere geometry once, using those buffers, then draw this same mesh how many times we need !


### 2. globe_shader.wgsl
- Renders the main globe.
- transform world space to clip space
- Applies lighting and texturing.
- Draws the sphere mesh once (no instancing).


### 3. cloth_instances.wgsl
- Renders the cloth as many spheres (particles).
- transform world space to clip space
- Uses Particles struct for per-instance transforms (model matrix for each sphere).
- Draws the sphere mesh many times (instancing), each at a different position.




