use numpy::{
    ndarray::{Array1, ArrayView1},
    IntoPyArray, PyArray1, PyReadonlyArray1,
};
use pyo3::prelude::*;

#[pymodule]
fn rust_ext<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn calc_regression(x: ArrayView1<f32>, frames_per_second: usize) -> Array1<f32> {
        let step = 1.0 / frames_per_second as f32;
        let len = x.len();
        let mut output = Array1::<f32>::zeros(len);

        let locts: Vec<usize> = x
            .indexed_iter()
            .filter_map(|(i, &v)| if v < 0.5 { Some(i) } else { None })
            .collect();

        if !locts.is_empty() {
            for t in 0..locts[0] {
                output[t] = step * (t as f32 - locts[0] as f32) - x[locts[0]];
            }

            for i in 0..(locts.len() - 1) {
                let loct_i = locts[i];
                let loct_next = locts[i + 1];
                let mid_point = (loct_i + loct_next) / 2;

                for t in loct_i..mid_point {
                    output[t] = step * (t as f32 - loct_i as f32) - x[loct_i];
                }

                for t in mid_point..loct_next {
                    output[t] = step * (t as f32 - loct_next as f32) - x[loct_next];
                }
            }

            let loct_last = locts[locts.len() - 1];
            for t in loct_last..len {
                output[t] = step * (t as f32 - loct_last as f32) - x[loct_last];
            }
        }

        output.mapv_inplace(|v| {
            let v_abs = v.abs();
            let v_clipped = v_abs.min(0.05);
            1.0 - v_clipped * 20.0
        });

        output
    }

    #[pyfn(m)]
    #[pyo3(name = "calc_regression")]
    fn calc_regression_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<f32>,
        frames_per_second: usize,
    ) -> Bound<'py, PyArray1<f32>> {
        let x = x.as_array();
        let output = calc_regression(x, frames_per_second);
        output.into_pyarray_bound(py)
    }

    Ok(())
}
