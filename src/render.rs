// Cargo.toml dependencies:
// [dependencies]
// bevy = "0.10"
// bevy_prototype_lyon = "0.6"

use bevy::color::palettes::css::WHITE;
use bevy::math::VectorSpace;
use bevy::prelude::*;
use bevy::render::camera::ScalingMode;
use bevy_polyline::prelude::{Polyline, PolylineBundle, PolylineMaterial};
use bevy_polyline::PolylinePlugin;
use bevy_prototype_lyon::prelude::*;
use std::f32::consts::PI;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use bevy::input::mouse::MouseWheel;
use bevy::sprite::{MaterialMesh2dBundle, Mesh2d, Mesh2dHandle};
use crate::solver::{Agent, Allocation, Item};
use crate::solver::utility::generate_indifference_curve;
// Data structures

const CIRCLE_RAD: f32 = 5.0;

#[derive(Clone)]
pub struct RenderAllocation {
    ic: Vec<(f32, f32)>,
    quality: f32,
    price: f32,
    utility: f32,
}

impl RenderAllocation {
    pub fn quality(&self) -> f32 {
        self.quality
    }

    pub fn from_allocation<F: num::Float, A: Agent<FloatType = F>, I: Item<FloatType = F>>(allocation: &Allocation<F, A, I>, delta: F, epsilon: F, max_iter: usize) -> Self {
        Self {
            ic: generate_indifference_curve(allocation.agent(), allocation.utility(), F::zero(), allocation.agent().income() - epsilon, delta, epsilon, max_iter),
            quality: allocation.quality().to_f32().unwrap(),
            price: allocation.price().to_f32().unwrap(),
            utility: allocation.utility().to_f32().unwrap(),
        }
    }
}

// Resources

#[derive(Default, Resource)]
struct AllocationsResource {
    to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>>,
    allocations: Vec<RenderAllocation>,
    current_idx: usize,
}


#[derive(Default, Resource)]
struct HoveredAllocation(Option<usize>);

#[derive(Default, Resource)]
struct ViewBounds {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,

    sf_x: f32,
    sf_y: f32,
}

// Components

#[derive(Component)]
struct AllocationEntity {
    index: usize,
}

#[derive(Component)]
struct IndifferenceCurve {
    index: usize,
}

// Systems

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    window: Query<&mut Window>,
) {
    // Set up the camera with appropriate scaling
    let mut camera = Camera2dBundle::default();
    commands.spawn(camera);

    let window = window.single();

     // Draw X and Y axes
     let axis_color = Color::srgb(0.5, 0.5, 0.5);
     let axis_thickness = 2.0;
 
     // X-axis
     let x_axis = shapes::Line(
         Vec2::new(0.0, 0.0),
         Vec2::new(100000.0, 0.0),
     );
 
     commands.spawn((
         ShapeBundle {
             path: GeometryBuilder::build_as(
                 &x_axis,
             ),
            spatial: SpatialBundle {
                transform: Transform::from_xyz(0.0, 0.0, -100.0),
                ..Default::default()
            },
             ..default()
         },
         Stroke::new(axis_color, axis_thickness),
     ));
 
     // Y-axis
     let y_axis = shapes::Line(
         Vec2::new(0.0, 0.0),
         Vec2::new(0.0, 100000.0),
     );
 
     commands.spawn((
         ShapeBundle {
             path: GeometryBuilder::build_as(
                 &y_axis,
             ),
             spatial: SpatialBundle {
                transform: Transform::from_xyz(0.0, 0.0, -100.0),
                ..Default::default()
            },
             ..default()
         },
         Stroke::new(axis_color, axis_thickness),
     ));
 
     // Add axis labels
     let font = asset_server.load("fonts/Arial.ttf");
     let label_font_size = 14.0;
     let label_color = Color::srgb(0.5, 0.5, 0.5);
 
     // Determine sensible intervals for labels
     let x_interval = 100.0;
     let y_interval = 100.0;
 
     // X-axis labels
     let mut x = 0.0;
     while x <= window.width() {
         let label = format!("{:.0}", x);
         commands.spawn(Text2dBundle {
             text: Text::from_section(
                 label,
                 TextStyle {
                     font: font.clone(),
                     font_size: label_font_size,
                     color: label_color,
                 },
             ),
             transform: Transform::from_xyz(x, -20.0, 1.0),
             ..default()
         });
         x += x_interval;
     }
 
     // Y-axis labels
     let mut y = 0.0;
     while y <= window.height() {
         let label = format!("{:.0}", y);
         commands.spawn(Text2dBundle {
             text: Text::from_section(
                 label,
                 TextStyle {
                     font: font.clone(),
                     font_size: label_font_size,
                     color: label_color,
                 },
             ),
             transform: Transform::from_xyz(-30.0, y, 1.0),
             ..default()
         });
         y += y_interval;
     }
}

fn compute_bounds(allocations: &Vec<RenderAllocation>) -> (f32, f32, f32, f32) {
    let mut x_min = allocations[0].quality();
    let mut x_max = allocations[0].quality();

    let mut y_min = allocations[0].price;
    let mut y_max = allocations[0].price;


    for a in allocations {
        let x = a.quality();
        let y = a.price;

        if x < x_min {
            x_min = x;
        }
        if x > x_max {
            x_max = x;
        }
        if y < y_min {
            y_min = y;
        }
        if y > y_max {
            y_max = y;
        }
    }

    // Add padding
    let x_padding = (x_max - x_min).abs() * 0.1;
    let y_padding = (y_max - y_min).abs() * 0.1;

    x_min -= x_padding;
    x_max += x_padding;
    y_min -= y_padding;
    y_max += y_padding;

    (x_min, x_max, y_min, y_max)
}

// Handle user input for zooming and panning
fn handle_input(
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mut scroll_events: EventReader<MouseWheel>,
    mut cursor_moved_events: EventReader<CursorMoved>,
    mut query: Query<&mut Transform, With<Camera>>,
) {
    let mut camera_transform = query.single_mut();

    // Handle zooming
    for event in scroll_events.read() {
        let zoom_factor = 1.0 - event.y * 0.1;
        camera_transform.scale *= Vec3::new(zoom_factor, zoom_factor, 1.0);
    }

    if mouse_button_input.pressed(MouseButton::Left) {
        for event in cursor_moved_events.read() {
            if let Some(delta) = event.delta {
                let scale = camera_transform.scale;
                let scaled_delta = delta.extend(0.0) * scale;

                camera_transform.translation -= Vec3::new(scaled_delta.x, -scaled_delta.y, scaled_delta.z);
            }
        }

    }
}

// System to draw allocations and indifference curves when allocations change
fn draw_allocations(
    mut commands: Commands,
    mut allocations_res: ResMut<AllocationsResource>,
    mut view_bounds: ResMut<ViewBounds>,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut query: Query<Entity, With<AllocationEntity>>,
    mut curve_query: Query<Entity, With<IndifferenceCurve>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
    ) {
    // Only run when allocations have changed
    let allocations_res: &mut AllocationsResource = allocations_res.deref_mut();
    if let Ok(mut allocations_guard) = allocations_res.to_allocate.try_lock() {

        
        if let Some(new_allocations) = allocations_guard.take() {
            allocations_res.allocations = new_allocations;

            // Clear previous allocation entities
            for entity in query.iter_mut() {
                commands.entity(entity).despawn_recursive();
            }
    
            // Clear previous indifference curves
            for entity in curve_query.iter_mut() {
                commands.entity(entity).despawn_recursive();
            }
    
            let allocations = &allocations_res.allocations;
            let current_idx = allocations_res.current_idx;
    
            if allocations.is_empty() {
                return;
            }
    
            // Compute coordinate bounds
            let (x_min, x_max, y_min, y_max) = compute_bounds(allocations);
            view_bounds.x_min = x_min;
            view_bounds.x_max = x_max;
            view_bounds.y_min = y_min;
            view_bounds.y_max = y_max;
    
            view_bounds.sf_x = 500f32 / (x_max - x_min);
            view_bounds.sf_y = 500f32 / (y_max - y_min);
        
            // Draw indifference curves
            for (i, allocation) in allocations.iter().enumerate() {
                let num_samples = 500;
                let mut path_builder = PathBuilder::new();
    
                let mut started = false;
    
                for (x, y) in allocation.ic.iter() {
                    let point = Vec2::new(*x * view_bounds.sf_x, *y * view_bounds.sf_y);
                    if !started {
                        path_builder.move_to(point);
                        started = true;
                    } else {
                        path_builder.line_to(point);
                    }
                }

                //let z = -(i as f32) - 1.0;
                //let vertices: Vec<Vec2> = allocation.ic.iter().map(|(x, y)| Vec2::new(*x * view_bounds.sf_x, *y * view_bounds.sf_y)).collect();
    
                let path = path_builder.build();
    
                let color = if i == current_idx {
                    Color::Srgba(Srgba::GREEN)
                } else {
                    Color::Srgba(Srgba::WHITE)
                };
    
                let stroke = Stroke::new(color, 1.0);
                commands
                    .spawn((
                        ShapeBundle {
                            path,
                            spatial: SpatialBundle {
                                transform: Transform::from_xyz(0., 0., -(i as f32) - 1.0),
                                ..Default::default()
                            },
                            ..default()
                        },
                        Stroke::new(color, 1.0),
                    ))
                    .insert(IndifferenceCurve { index: i });
                // commands.spawn(MaterialMesh2dBundle {
                //     mesh: meshes.add(BoxedPolyline2d::new(vertices)).into(),
                //     material: materials.add(Color::BLACK),
                //     transform: Transform::from_xyz(0.0, 0.0, 0.0),
                //     ..default()
                // });
            }
    
            // Draw allocation circles
            for (i, allocation) in allocations.iter().enumerate() {
                let x = allocation.quality() * view_bounds.sf_x;
                let y = allocation.price * view_bounds.sf_y;
        
                let circle_radius = CIRCLE_RAD; // Adjust circle size
    
                let color = if i == current_idx {
                    Color::Srgba(Srgba::RED)
                } else {
                    Color::Srgba(Srgba::BLUE)
                };
    
                let circle = shapes::Circle {
                    radius: circle_radius,
                    ..Default::default()
                };
    
    
                //let mesh = Mesh2dHandle(meshes.add(circle));
    
                commands
                    .spawn((
                        ShapeBundle {
                            path: GeometryBuilder::new().add(&circle).build(),
                            spatial: SpatialBundle {
                                transform: Transform::from_xyz(x, y, i as f32),
                                ..Default::default()
                            },
                            ..default()
                        },
                        Stroke::new(Color::Srgba(Srgba::WHITE), 1.0),
                        Fill::color(color),
                    ))
                    .insert(AllocationEntity { index: i });
            }
        }
    }
    
}

fn window_to_world(
    position: Vec2,
    window: &Window,
    camera: &Transform,
) -> Vec3 {

    // Center in screen space
    let norm = Vec3::new(
        position.x - window.width() / 2.,
        window.height() / 2. - position.y,
        0.,
    );

    // Apply camera transform
    *camera * norm

    // Alternatively:
    //camera.mul_vec3(norm)
}

// System to handle hovering without redrawing allocations
fn handle_hovering(
    window: Query<&mut Window>,
    query: Query<&Transform, With<Camera>>,
    allocations_res: Res<AllocationsResource>,
    view_bounds: Res<ViewBounds>,
    mut hovered_allocation: ResMut<HoveredAllocation>,
    mut allocation_query: Query<(&AllocationEntity, &mut Fill), Without<IndifferenceCurve>>,
    mut curve_query: Query<(&IndifferenceCurve, &mut Stroke), Without<AllocationEntity>>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let allocations = &allocations_res.allocations;
    let camera_transform = query.single();

    // Reset previous hover highlights
    for (_entity, mut fill) in allocation_query.iter_mut() {
        fill.color = Color::Srgba(Srgba::BLUE);
    }

    for (_entity, mut fill) in curve_query.iter_mut() {
        fill.color = Color::Srgba(Srgba::WHITE);
    }

    if let Ok(window) = window.get_single() {

        // Handle hover detection
        if let Some(cursor_position) = window.cursor_position() {
            // Convert cursor position to world coordinates
            let world_position = window_to_world(cursor_position, window, camera_transform);

            // Check if cursor is over any allocation
            let mut hovered_idx = None;

            for (allocation_entity, mut fill) in allocation_query.iter_mut() {
                let allocation = &allocations[allocation_entity.index];
                let x = allocation.quality() * view_bounds.sf_x;
                let y = allocation.price * view_bounds.sf_y;

                let circle_radius = CIRCLE_RAD;// * camera_transform.scale.x.abs(); // Adjust circle size according to zoom

                let distance = ((world_position.x - x).powi(2) + (world_position.y - y).powi(2)).sqrt();
                //println!("Dist: {}, {}, {}, {}", x, y, world_position.x, world_position.y);
                
                if distance <= circle_radius {
                    hovered_idx = Some(allocation_entity.index);

                    fill.color = Color::Srgba(Srgba::new(1.0, 0.65, 0.01, 1.0));
                }
            }

            // Highlight the indifference curve of the hovered allocation
            if let Some(idx) = hovered_idx {
                for (curve_entity, mut stroke) in curve_query.iter_mut() {
                    if curve_entity.index == idx {
                        stroke.color = Color::Srgba(Srgba::RED);
                    }
                }
            }

            hovered_allocation.0 = hovered_idx;

            // Display info text if hovering over an allocation
            if let Some(hovered_idx) = hovered_allocation.0 {
                
                let allocation = &allocations[hovered_idx];

                let info_text = format!(
                    "n = {}\np = {:.2}\nq = {:.2}\nu = {:.2}",
                    hovered_idx,
                    allocation.price,
                    allocation.quality(),
                    allocation.utility,
                );

                // Create TextBundle
                let text_style = TextStyle {
                    font: asset_server.load("fonts/Arial.ttf"),
                    font_size: 14.0,
                    color: Color::BLACK,
                };

                // Spawn the text
                commands.spawn(Text2dBundle {
                    text: Text::from_section(info_text, text_style.clone())
                        .with_justify(JustifyText::Left),
                    transform: Transform {
                        translation: world_position + Vec3::new(10.0, -10.0, 1.0),
                        ..Default::default()
                    },
                    ..Default::default()
                });
            }
        } else {
            hovered_allocation.0 = None;
        }
    }
}

// System to clear info texts each frame
fn clear_info_texts(
    mut commands: Commands,
    query: Query<Entity, With<Text>>,
) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

// Main function

pub fn render_test(to_allocate: Arc<Mutex<Option<Vec<RenderAllocation>>>>) {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "NEES".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(AllocationsResource {
            to_allocate,
            allocations: Vec::new(),
            current_idx: 0,
        })
        .insert_resource(HoveredAllocation(None))
        .insert_resource(ViewBounds::default())
        .add_plugins(ShapePlugin)
        .add_plugins(PolylinePlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, handle_input)
        .add_systems(Update, draw_allocations)
        .add_systems(Update, clear_info_texts.before(handle_hovering))
        .add_systems(Update, handle_hovering)
        .run();
}
