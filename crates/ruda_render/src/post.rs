//! What happens to a frame after the world is drawn. The world is drawn in
//! light that can be far brighter than a screen shows; here bright light
//! bleeds into its surroundings (bloom), the exposure follows how bright
//! the view is, the way eyes adapt, and AgX maps the result to the screen.
//! See `post.wgsl`.

use std::time::Instant;

/// What the world is drawn into: half floats where the GPU can draw into
/// and blend them, which is nearly everywhere; plain bytes elsewhere, where
/// light above white is clipped.
pub(crate) fn frame_format(adapter: &wgpu::Adapter) -> wgpu::TextureFormat {
    let features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
    let usages = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let flags =
        wgpu::TextureFormatFeatureFlags::FILTERABLE | wgpu::TextureFormatFeatureFlags::BLENDABLE;
    if features.allowed_usages.contains(usages) && features.flags.contains(flags) {
        wgpu::TextureFormat::Rgba16Float
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

/// Bytes of the `Post` uniform in `post.wgsl`.
const UNIFORMS_SIZE: u64 = 64;
/// At most this many levels of bloom, each half the size of the one above.
const BLOOM_LEVELS: u32 = 6;
/// How much of the bloom shows over the frame.
const BLOOM: f32 = 0.05;
/// The brightness a view of average brightness is shown at, before AgX.
const KEY: f32 = 0.2;
/// How far the exposure follows how bright the view is: a little less than
/// fully, so caves still look darker than daylight, and night than day.
const ADAPTATION: f32 = 0.7;
/// Exposure never goes beyond these, so night stays dark.
const EXPOSURE: (f32, f32) = (0.15, 6.0);

/// What each pass reads.
struct Groups {
    /// Making each level of bloom from the one above.
    down: Vec<wgpu::BindGroup>,
    /// Blurring each level but the first back onto the one above.
    up: Vec<wgpu::BindGroup>,
    /// Writing exposure 0 from exposure 1, and the other way.
    adapt: [wgpu::BindGroup; 2],
    /// Finishing with either exposure.
    composite: [wgpu::BindGroup; 2],
}

pub(crate) struct PostPass {
    format: wgpu::TextureFormat,
    output_format: wgpu::TextureFormat,
    /// The frame as the world pass draws it.
    frame: wgpu::TextureView,
    /// Levels of bloom, the first half the size of the frame.
    bloom: Vec<wgpu::TextureView>,
    /// One texel each, swapped every frame: last frame's exposure, and this
    /// frame's.
    exposure: [wgpu::TextureView; 2],
    current: usize,
    /// For slots a pass doesn't read.
    blank: wgpu::TextureView,
    /// The light at the camera.
    ambient: wgpu::TextureView,
    groups: Groups,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    downsample_first: wgpu::RenderPipeline,
    downsample: wgpu::RenderPipeline,
    upsample: wgpu::RenderPipeline,
    adapt: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    /// When the last frame was finished, for how fast to adapt.
    last_frame: Option<Instant>,
    /// Set the exposure from each frame alone, for pictures taken one at a
    /// time.
    instant: bool,
    /// An exposure to keep instead of following the view.
    fixed: Option<f32>,
    /// Bloom, and an exposure that follows the view; without, the exposure
    /// follows the light around.
    hdr: bool,
}

impl PostPass {
    /// `ambient` is the light at the camera, see `SkyTables::ambient`.
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        output_format: wgpu::TextureFormat,
        ambient: &wgpu::TextureView,
        (width, height): (u32, u32),
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("post.wgsl"));
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture(2),
                texture(3),
                texture(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, target: wgpu::TextureFormat, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post"),
            size: UNIFORMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let target = |label, size| texture_view(device, label, format, size);
        let frame = target("frame", (width, height));
        let bloom = bloom_levels(device, format, (width, height));
        let exposure = [target("exposure", (1, 1)), target("exposure", (1, 1))];
        let blank = target("blank", (1, 1));
        let groups = Groups::new(
            device,
            &layout,
            &uniforms,
            &sampler,
            [&frame, &blank, ambient],
            &bloom,
            &exposure,
        );
        Self {
            format,
            output_format,
            frame,
            bloom,
            exposure,
            current: 0,
            blank,
            ambient: ambient.clone(),
            groups,
            uniforms,
            sampler,
            downsample_first: pipeline("downsample_first", format, None),
            downsample: pipeline("downsample", format, None),
            upsample: pipeline("upsample", format, Some(add)),
            adapt: pipeline("adapt", format, None),
            composite: pipeline("composite", output_format, None),
            layout,
            last_frame: None,
            instant: false,
            fixed: None,
            hdr: true,
        }
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.frame = texture_view(device, "frame", self.format, (width, height));
        self.bloom = bloom_levels(device, self.format, (width, height));
        self.groups = Groups::new(
            device,
            &self.layout,
            &self.uniforms,
            &self.sampler,
            [&self.frame, &self.blank, &self.ambient],
            &self.bloom,
            &self.exposure,
        );
    }

    /// What the world pass draws into.
    pub(crate) fn frame(&self) -> &wgpu::TextureView {
        &self.frame
    }

    /// Sets the exposure from each frame alone instead of adapting to the
    /// view over time: for pictures taken one at a time.
    pub(crate) fn adapt_at_once(&mut self, instant: bool) {
        self.instant = instant;
    }

    /// Turns HDR on or off: bloom, and an exposure that follows the view
    /// rather than the light around.
    pub(crate) fn set_hdr(&mut self, hdr: bool) {
        self.hdr = hdr;
    }

    /// Keeps the exposure at `exposure` instead of following the view, or
    /// with `None` follows it again.
    pub(crate) fn fix_exposure(&mut self, exposure: Option<f32>) {
        self.fixed = exposure;
    }

    /// Finishes the frame into `output`, with the crosshair if `crosshair`.
    pub(crate) fn draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::TextureView,
        (width, height): (u32, u32),
        crosshair: bool,
        light_height: f32,
    ) {
        let now = Instant::now();
        let elapsed = self
            .last_frame
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32().min(0.5));
        let reset = self.instant || self.fixed.is_some() || self.last_frame.is_none();
        let limits = self.fixed.map_or(EXPOSURE, |exposure| (exposure, exposure));
        self.last_frame = Some(now);
        let bloom = self.hdr && !self.bloom.is_empty();
        let floats: [f32; 16] = [
            elapsed,
            if reset { 1.0 } else { 0.0 },
            if bloom { BLOOM } else { 0.0 },
            if self.output_format.is_srgb() {
                0.0
            } else {
                1.0
            },
            limits.0,
            limits.1,
            KEY,
            ADAPTATION,
            if crosshair { 1.0 } else { 0.0 },
            width as f32,
            height as f32,
            0.0,
            if self.hdr { 0.0 } else { 1.0 },
            light_height,
            0.0,
            0.0,
        ];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        queue.write_buffer(&self.uniforms, 0, &bytes);

        self.current = 1 - self.current;
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    label: &str,
                    target: &wgpu::TextureView,
                    load: bool,
                    pipeline: &wgpu::RenderPipeline,
                    group: &wgpu::BindGroup| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if load {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        };

        // Halve the frame level by level, then measure how bright it is
        // from the smallest level, before the way back up adds to them.
        let levels = if bloom { self.bloom.len() } else { 0 };
        for (index, (level, group)) in self
            .bloom
            .iter()
            .zip(&self.groups.down)
            .take(levels)
            .enumerate()
        {
            let pipeline = if index == 0 {
                &self.downsample_first
            } else {
                &self.downsample
            };
            pass(encoder, "bloom down", level, false, pipeline, group);
        }
        pass(
            encoder,
            "exposure",
            &self.exposure[self.current],
            false,
            &self.adapt,
            &self.groups.adapt[self.current],
        );
        for (index, group) in self
            .groups
            .up
            .iter()
            .enumerate()
            .take(levels.saturating_sub(1))
            .rev()
        {
            pass(
                encoder,
                "bloom up",
                &self.bloom[index],
                true,
                &self.upsample,
                group,
            );
        }
        pass(
            encoder,
            "composite",
            output,
            false,
            &self.composite,
            &self.groups.composite[self.current],
        );
    }
}

impl Groups {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniforms: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        [frame, blank, ambient]: [&wgpu::TextureView; 3],
        bloom: &[wgpu::TextureView],
        exposure: &[wgpu::TextureView; 2],
    ) -> Self {
        let group =
            |source: &wgpu::TextureView, glow: &wgpu::TextureView, exposure: &wgpu::TextureView| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("post"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniforms.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(glow),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(exposure),
                        },
                    ],
                })
            };
        let sources = std::iter::once(frame).chain(bloom);
        let measured = bloom.last().unwrap_or(frame);
        let glow = bloom.first().unwrap_or(blank);
        Self {
            down: sources
                .take(bloom.len())
                .map(|source| group(source, blank, blank))
                .collect(),
            up: bloom
                .iter()
                .skip(1)
                .map(|source| group(source, blank, blank))
                .collect(),
            adapt: [
                group(measured, ambient, &exposure[1]),
                group(measured, ambient, &exposure[0]),
            ],
            composite: [
                group(frame, glow, &exposure[0]),
                group(frame, glow, &exposure[1]),
            ],
        }
    }
}

fn texture_view(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Bloom's levels, halving down to a few dozen pixels; none for frames in
/// plain bytes, where nothing is brighter than white to bleed.
fn bloom_levels(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
) -> Vec<wgpu::TextureView> {
    if format != wgpu::TextureFormat::Rgba16Float {
        return Vec::new();
    }
    (1..=BLOOM_LEVELS)
        .map(|level| (width >> level, height >> level))
        .take_while(|&(width, height)| width >= 8 && height >= 8)
        .map(|size| texture_view(device, "bloom", format, size))
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_is_valid_on_the_baseline_tier() {
        let module = naga::front::wgsl::parse_str(include_str!("post.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
