use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use crate::render::RenderAllocation;

/// Colors/styling constants for the SVG.
const SVG_BG_COLOR: &str = "#FFFFFF00";
const SVG_AXIS_COLOR: &str = "#000000";
const SVG_AXIS_STROKE_WIDTH: f32 = 2.5;

/// Curve/line styling.
const LINE_COLOR: &str = "#666666"; // similar to (0.45, 0.4, 0.4)
const LINE_CURRENT_COLOR: &str = "#00FF00"; // green
const LINE_WIDTH: f32 = 2.5;

/// Circle styling.
const CIRCLE_COLOR: &str = "#0000FF"; // similar to (0.6, 0.6, 1.0)
const CIRCLE_CURRENT_COLOR: &str = "#00FF00"; // green circle for current
const CIRCLE_RADIUS: f32 = 7.0;

// Font styling for axis labels.
const AXIS_LABEL_FONT_SIZE: f32 = 20.0;
const TICK_LABEL_FONT_SIZE: f32 = 17.0;
const LABEL_COLOR: &str = "#000000";

fn compute_bounds(allocations: &[RenderAllocation]) -> (f32, f32, f32, f32) {
    if allocations.is_empty() {
        return (0.0, 100.0, 0.0, 100.0);
    }
    let mut x_min = allocations[0].quality;
    let mut x_max = allocations[0].quality;
    let mut y_min = allocations[0].price;
    let mut y_max = allocations[0].price;

    for a in allocations {
        if a.quality < x_min {
            x_min = a.quality;
        }
        if a.quality > x_max {
            x_max = a.quality;
        }
        if a.price < y_min {
            y_min = a.price;
        }
        if a.price > y_max {
            y_max = a.price;
        }
    }

    // Ensure y_min >= 0
    if y_min < 0.0 {
        y_min = 0.0;
    }

    let x_padding = (x_max - x_min).abs() * 0.1;
    let y_padding = (y_max - y_min).abs() * 0.1;

    (
        x_min - x_padding,
        x_max + x_padding,
        y_min,
        y_max + y_padding,
    )
}

fn nice_tick_interval(range: f32, desired_steps: i32) -> f32 {
    if range < 1e-12 {
        return 1.0;
    }
    let raw_step = range / desired_steps as f32;
    let magnitude = 10.0_f32.powf(raw_step.abs().log10().floor());
    let normalized = raw_step / magnitude;
    let snapped = if normalized < 2.0 {
        2.0
    } else if normalized < 5.0 {
        5.0
    } else {
        10.0
    };
    snapped * magnitude
}

fn decimals_for_step(step: f32) -> usize {
    if step <= 0.0 {
        return 0;
    }
    let mut decimals = 0_usize;
    let mut candidate = step;

    while candidate < 1.0 && decimals < 6 {
        candidate *= 10.0;
        decimals += 1;
    }
    // if the step has a fractional part, ensure at least 1 decimal
    if (step - step.floor()).abs() > 1e-6 {
        decimals = decimals.max(1);
    }
    decimals
}

fn format_number(value: f32, decimals: usize) -> String {
    let mut s = format!("{:.*}", decimals, value);
    // Strip trailing zeros if decimals > 0
    if decimals > 0 {
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

// We'll keep the approach of computing an "auto margin"
fn compute_auto_margin(
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    x_step: f32,
    y_step: f32,
    decimals_x: usize,
    decimals_y: usize,
) -> f32 {
    let max_x_label_val = x_min.abs().max(x_max.abs());
    let max_y_label_val = y_min.abs().max(y_max.abs());

    let sample_x_str = format_number(max_x_label_val, decimals_x);
    let sample_y_str = format_number(max_y_label_val, decimals_y);

    let x_text_width = sample_x_str.len() as f32 * 7.0;
    let y_text_width = sample_y_str.len() as f32 * 7.0;

    let mut needed_margin = x_text_width.max(y_text_width);
    needed_margin += 20.0;
    needed_margin.max(40.0)
}

/// Renders an SVG of the allocations.
pub fn render_allocations_to_svg(
    allocations: &[RenderAllocation],
    current_idx: Option<usize>,
    svg_width: u32,
    svg_height: u32,
    show_ticks: bool,
) -> String {
    // 1. bounding box
    let (x_min, x_max, y_min, y_max) = compute_bounds(allocations);

    // 2. "nice" intervals
    let desired_steps = 5;
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let x_step = nice_tick_interval(x_range, desired_steps);
    let y_step = nice_tick_interval(y_range, desired_steps);

    // decimals
    let decimals_x = decimals_for_step(x_step);
    let decimals_y = decimals_for_step(y_step);

    // margin
    let margin = compute_auto_margin(
        x_min, x_max, y_min, y_max, x_step, y_step, decimals_x, decimals_y,
    );

    // scale
    let scale_x = (svg_width as f32 - 2.0 * margin) / x_range.max(1e-6);
    let scale_y = (svg_height as f32 - 2.0 * margin) / y_range.max(1e-6);

    let to_screen = |(xx, yy): (f32, f32)| -> (f32, f32) {
        let sx = margin + (xx - x_min) * scale_x;
        let sy = (svg_height as f32 - margin) - (yy - y_min) * scale_y;
        (sx, sy)
    };

    // 3. Start SVG
    let mut svg = String::new();
    writeln!(
        svg,
        r#"<svg version="1.1" baseProfile="full" width="{w}" height="{h}" viewBox="0 0 {w} {h}"
 xmlns="http://www.w3.org/2000/svg" style="background:{bg};">
"#,
        w = svg_width,
        h = svg_height,
        bg = SVG_BG_COLOR
    )
    .unwrap();

    // 4. Compute the screen-space rectangle for clipping.
    let (sx_min, sy_min) = to_screen((x_min, y_min));
    let (sx_max, sy_max) = to_screen((x_max, y_max));

    let clip_x = sx_min.min(sx_max);
    let clip_y = sy_max.min(sy_min);
    let clip_w = (sx_max - sx_min).abs();
    let clip_h = (sy_min - sy_max).abs();

    // 5. Define a clipPath in <defs>
    writeln!(
        svg,
        r#"<defs>
  <clipPath id="clipBox">
    <rect x="{x}" y="{y}" width="{w}" height="{h}" />
  </clipPath>
</defs>
"#,
        x = clip_x,
        y = clip_y,
        w = clip_w,
        h = clip_h
    )
    .unwrap();

    // 6. Draw axes lines
    {
        let (x0_screen, y0_screen) = to_screen((x_min, y_min));
        let (x1_screen, y1_screen) = to_screen((x_max, y_min));
        writeln!(
            svg,
            r#"<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{sw}"/>"#,
            x0 = x0_screen,
            y0 = y0_screen,
            x1 = x1_screen,
            y1 = y1_screen,
            color = SVG_AXIS_COLOR,
            sw = SVG_AXIS_STROKE_WIDTH
        )
        .unwrap();

        let (x2_screen, y2_screen) = to_screen((x_min, y_min));
        let (x3_screen, y3_screen) = to_screen((x_min, y_max));
        writeln!(
            svg,
            r#"<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="{sw}"/>"#,
            x0 = x2_screen,
            y0 = y2_screen,
            x1 = x3_screen,
            y1 = y3_screen,
            color = SVG_AXIS_COLOR,
            sw = SVG_AXIS_STROKE_WIDTH
        )
        .unwrap();
    }

    // 7. Tick labels + TICK MARKS
    //    We'll add a small "blip" line near each tick.
    if show_ticks {
        // For the X-axis
        let x_start = (x_min / x_step).ceil() * x_step;
        let x_end = (x_max / x_step).floor() * x_step;
        let mut xx = x_start;

        // The vertical “blip” length for X ticks
        let tick_len = 6.0;

        while xx <= x_end + 1e-9 {
            let (sx, sy) = to_screen((xx, y_min));
            // 7a. Tick text
            let label_y = sy + TICK_LABEL_FONT_SIZE * 1.4;
            let label_str = format_number(xx, decimals_x);
            writeln!(
                svg,
                r#"<text x="{x}" y="{y}" font-size="{fs}" fill="{fc}"
 text-anchor="middle">{val}</text>"#,
                x = sx,
                y = label_y,
                fs = TICK_LABEL_FONT_SIZE,
                fc = LABEL_COLOR,
                val = label_str
            )
            .unwrap();

            // 7b. Draw the small blip line *above* the axis
            // So from (sx, sy) to (sx, sy - tick_len)
            writeln!(
                svg,
                r#"<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}"
 stroke="{color}" stroke-width="1"/>"#,
                x0 = sx,
                y0 = sy + tick_len,
                x1 = sx,
                y1 = sy,
                color = SVG_AXIS_COLOR,
            )
            .unwrap();

            xx += x_step;
        }

        // For the Y-axis
        let y_start = (y_min / y_step).ceil() * y_step;
        let y_end = (y_max / y_step).floor() * y_step;
        let mut yy = y_start;

        // The horizontal “blip” length for Y ticks
        let tick_len_y = 6.0;

        while yy <= y_end + 1e-9 {
            let (sx, sy) = to_screen((x_min, yy));
            let label_x = sx - TICK_LABEL_FONT_SIZE * 0.5;
            let label_str = format_number(yy, decimals_y);

            // 7a. Tick text
            writeln!(
                svg,
                r#"<text x="{x}" y="{y}" font-size="{fs}" fill="{fc}"
 text-anchor="end" dominant-baseline="middle">{val}</text>"#,
                x = label_x,
                y = sy,
                fs = TICK_LABEL_FONT_SIZE,
                fc = LABEL_COLOR,
                val = label_str
            )
            .unwrap();

            // 7b. Draw the small blip line *to the right* of the axis
            // from (sx, sy) to (sx + tick_len_y, sy)
            writeln!(
                svg,
                r#"<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}"
 stroke="{color}" stroke-width="1"/>"#,
                x0 = sx - tick_len_y,
                y0 = sy,
                x1 = sx,
                y1 = sy,
                color = SVG_AXIS_COLOR,
            )
            .unwrap();

            yy += y_step;
        }
    }

    // 8. Axis labels "q" and "p"
    {
        let offset_q = 5.0;
        let (x_screen_q, y_screen_q) = to_screen((x_max, y_min));
        writeln!(
            svg,
            r#"<text x="{x}" y="{y}" font-size="{fs}" fill="{fc}"
 text-anchor="start" dominant-baseline="middle">q</text>"#,
            x = x_screen_q + offset_q,
            y = y_screen_q,
            fs = AXIS_LABEL_FONT_SIZE,
            fc = SVG_AXIS_COLOR
        )
        .unwrap();

        let offset_p = 20.0;
        let (x_screen_p, y_screen_p) = to_screen((x_min, y_max));
        writeln!(
            svg,
            r#"<text x="{x}" y="{y}" font-size="{fs}" fill="{fc}"
 text-anchor="end" dominant-baseline="hanging">p</text>"#,
            x = x_screen_p,
            y = y_screen_p - offset_p,
            fs = AXIS_LABEL_FONT_SIZE,
            fc = SVG_AXIS_COLOR
        )
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // 9. CLIPPED GROUP for Indifference Curves + Circles
    // -----------------------------------------------------------------------
    writeln!(svg, r#"<g clip-path="url(#clipBox)">{g}"#, g = "").unwrap();

    // 9a. Indifference curves
    for (i, alloc) in allocations.iter().enumerate() {
        if alloc.ic.is_empty() {
            continue;
        }
        let color = if Some(i) == current_idx {
            LINE_CURRENT_COLOR
        } else {
            LINE_COLOR
        };

        let mut d = String::new();
        let (sx0, sy0) = to_screen(alloc.ic[0]);
        write!(d, "M {:.2},{:.2}", sx0, sy0).unwrap();
        for &(xx, yy) in &alloc.ic[1..] {
            let (sx, sy) = to_screen((xx, yy));
            write!(d, " L {:.2},{:.2}", sx, sy).unwrap();
        }

        writeln!(
            svg,
            r#"<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" />"#,
            d = d,
            color = color,
            sw = LINE_WIDTH
        )
        .unwrap();
    }

    // 9b. Circles for each allocation
    for (i, alloc) in allocations.iter().enumerate() {
        let color = if Some(i) == current_idx {
            CIRCLE_CURRENT_COLOR
        } else {
            CIRCLE_COLOR
        };
        let (sx, sy) = to_screen((alloc.quality, alloc.price));
        writeln!(
            svg,
            r#"<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" />"#,
            x = sx,
            y = sy,
            r = CIRCLE_RADIUS,
            color = color
        )
        .unwrap();
    }

    // close the <g>
    writeln!(svg, "</g>").unwrap();

    // 10. Close the SVG
    writeln!(svg, "</svg>").unwrap();
    svg
}
