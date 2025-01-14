use super::Agent;

// Must change each dimension one at a time otherwise we dont know when we have overshot the solution...
// If changing a dimension brings us farther away we know to go the opposite direction. If we overshoot on that dimension we move onto next dimension.
// If we overshoot on all dimensions we reduce the delta.
pub fn indifferent_price<const D: usize, F: num::Float, A>(
    agent: &A,
    quality: [F; D],
    u_0: F,
    x_min: F,
    x_max: F,
    epsilon: F,
    max_iter: usize,
) -> Option<F>
where
    A: Agent<D, FloatType = F>,
{
    let mut lower = x_min;
    let mut upper = x_max;
    let mut iter = 0;

    while iter < max_iter {
        let mid = (lower + upper) / F::from(2.0).unwrap();
        let u_mid = agent.utility(mid, quality);
        let diff = u_mid - u_0;

        if diff.abs() < epsilon {
            return Some(mid);
        }

        // Because the utility function is increasing in quality, we swap this from the solver for quality.
        if diff > F::zero() {
            lower = mid;
        } else {
            upper = mid;
        }

        iter += 1;
    }

    // Return NaN if the solution was not found within tolerance
    None
}

// pub fn indifferent_quality<const D: usize, F: num::Float, A>(
//     agent: &A,
//     price: F,
//     u_0: F,
//     y_min: F,
//     y_max: F,
//     epsilon: F,
//     max_iter: usize,
// ) -> Option<F>
// where
//     A: Agent<D, FloatType = F>,
// {
//     let mut lower = y_min;
//     let mut upper = y_max;
//     let mut iter = 0;

//     while iter < max_iter {
//         let mid = (lower + upper) / F::from(2.0).unwrap();
//         let u_mid = agent.utility(price, mid);
//         let diff = u_mid - u_0;

//         if diff.abs() < epsilon {
//             return Some(mid);
//         }

//         if diff > F::zero() {
//             upper = mid;
//         } else {
//             lower = mid;
//         }

//         iter += 1;
//     }

//     // Return NaN if a solution was not found within the tolerance of epsilon
//     None
// }


// // Not happening for n > 2!!
// pub fn generate_indifference_curve<const D: usize, F: num::Float, A>(
//     agent: &A,
//     utility: F,
//     p_0: F,
//     p_max: F,
//     delta: F,
//     epsilon: F,
//     max_iter: usize,
// ) -> Vec<(f32, f32)>
// where A: Agent<D, FloatType = F> {
//     let mut ic = Vec::with_capacity(((p_max - p_0) / delta).ceil().to_f32().unwrap() as usize);
//     let mut v = p_0;
//     while v < p_max {
//         if let Some(q) = indifferent_quality(agent, v, utility, p_0, p_max, epsilon, max_iter) {
//             ic.push((q.to_f32().unwrap(), v.to_f32().unwrap()));
//         }
//         v = v + delta;
//     }

//     ic
// }