# Nexus-Forge — Edge Computational Cytology — Phase-2 (Mojo 0.26.x)
# C/Rust FFI: UInt64 addresses. Linux shared library: mojo build ... --emit shared-lib

# Scalar/SIMD numeric types (Float32, SIMD, DType, …) and MutExternalOrigin are
# available from the Mojo prelude in 0.26.x; only import what the prelude omits.
from std.math import cos, sin, sqrt
from std.memory.unsafe_pointer import UnsafePointer
from std.sys.info import simd_width_of

# Denominator shield + degenerate-vector guard (IEEE ε-scale; freeze for formal closure).
comptime SAFE_TRACE_EPS: Float32 = Float32(1e-7)
comptime POWER_ITER_NORM_EPS: Float32 = Float32(1e-7)
comptime KAPPA_NONFINITE_SENTINEL: Float32 = Float32(0.5)

# SIMD lane count for Float32 kernels
comptime nelts = simd_width_of[DType.float32]()
comptime N_ROWS = UInt32(32)
comptime MAX_D = UInt32(256)
# Retain `[0 .. FREQ_KEEP)` and conjugate-frequency tail; midpoint indices are attenuated unless they land on
# the fundamental harmonic (`k==1`) or its conjugate (`N_ROWS-1`).
comptime FREQ_KEEP = UInt32(8)

comptime CYTO_FFI_OK = UInt32(0)
comptime CYTO_FFI_BAD_ARG = UInt32(7010)
comptime CYTO_FFI_BAD_WORK = UInt32(7020)
comptime CYTO_FFI_SHAPE = UInt32(7030)

comptime CYTO_CLINICAL_MALIGNANT = UInt32(100)
comptime CYTO_CLINICAL_NORMAL = UInt32(101)

# Power-iteration manifold limits (frozen for Phase-2 κ core).
comptime POWER_ITER_MAX: Int = 64
comptime POWER_ITER_TOL_LAMBDA: Float32 = Float32(1e-5)

# Mid-band clinical prior (isotropic vs anisotropic) — frozen gate for Phase-4 transport decoding.
comptime BASE_KAPPA_THRESHOLD: Float32 = Float32(0.40)


struct CytoNeighborMatrixView:
    """Rust POD mirror (~64 B): use memcpy-sized load via pointer deref."""

    var data_bits: UInt64
    var dims_d: UInt32
    var row_count: UInt32
    var stride_elems: UInt32
    var _implicit_pad_before_byte_len: UInt32
    var byte_length: UInt64
    var _rust_tail_pad_for_align64: SIMD[DType.uint8, 32]


@always_inline
fn _bit_reverse_32(i: UInt32) -> UInt32:
    var x = i
    var rv: UInt32 = 0
    var _: UInt32 = 0
    for _ in range(5):
        rv = (rv << 1) | (x & 1)
        x >>= 1
    return rv


@always_inline
fn _fft_inplace32_complex(
    re: UnsafePointer[Float32, MutExternalOrigin],
    im: UnsafePointer[Float32, MutExternalOrigin],
    forward: Bool,
):
    """Radix-2 Cooley-Tukey FFT / inverse for N_ROWS == 32."""

    comptime pi = Float32(3.14159265358979323846264338)

    var bi = UInt32(0)
    while bi < N_ROWS:
        var bj = _bit_reverse_32(bi)
        if bj > bi:
            var ia = Int(bi)
            var ib = Int(bj)
            var tr = re[ia]
            var ti_r = im[ia]
            re[ia] = re[ib]
            im[ia] = im[ib]
            re[ib] = tr
            im[ib] = ti_r
        bi += 1

    var stage = UInt32(1)
    while stage <= UInt32(5):
        var m = UInt32(1) << stage
        var half = m >> 1
        var grp = UInt32(0)
        while grp < N_ROWS:
            var jj = UInt32(0)
            while jj < half:
                var k0 = Int(grp + jj)
                var k1 = Int(grp + jj + half)

                var base = Float32(2.0) * pi * Float32(jj) / Float32(m)
                var angle: Float32
                if forward:
                    angle = -base
                else:
                    angle = base

                var wr = cos(angle)
                var wi = sin(angle)

                var r1 = re[k1]
                var i1 = im[k1]
                var tr = r1 * wr - i1 * wi
                var ti_m = r1 * wi + i1 * wr

                var r0 = re[k0]
                var i0 = im[k0]

                re[k1] = r0 - tr
                im[k1] = i0 - ti_m
                re[k0] = r0 + tr
                im[k0] = i0 + ti_m

                jj += 1
            grp += m
        stage += 1

    if forward:
        return

    var inv_scale = Float32(1.0) / Float32(N_ROWS)
    var tt_scale = UInt32(0)
    while tt_scale < N_ROWS:
        var ix_scale = Int(tt_scale)
        re[ix_scale] = re[ix_scale] * inv_scale
        im[ix_scale] = im[ix_scale] * inv_scale
        tt_scale += 1


@always_inline
fn _hermitian_mask32(
    re: UnsafePointer[Float32, MutExternalOrigin],
    im: UnsafePointer[Float32, MutExternalOrigin],
):
    """Hermitian-stable mid-band attenuation for real-valued time columns: clearing bin `k` also clears conjugate partner `N−k`.

    Fundamental bins `k=1` and `k=N−1` stay intact (audit: macroscopic harmonic retention).
    """

    var kk = FREQ_KEEP
    var stop = N_ROWS - FREQ_KEEP
    while kk <= stop:
        var ix = Int(kk)
        # Never attenuate fundamental frequency conjugate pair for periodic real inputs.
        if kk != UInt32(1) and kk != (N_ROWS - UInt32(1)):
            var k_conj = N_ROWS - kk
            var jx = Int(k_conj)
            re[ix] = Float32(0.0)
            im[ix] = Float32(0.0)
            re[jx] = Float32(0.0)
            im[jx] = Float32(0.0)
        kk += 1


@always_inline
fn _smooth_neighbor_column_into_dense(
    base: UnsafePointer[Float32, MutExternalOrigin],
    stride: Int,
    col: UInt32,
    column_mean_raw: Float32,
    dense: UnsafePointer[Float32, MutExternalOrigin],
    dense_stride: Int,
    fft_re: UnsafePointer[Float32, MutExternalOrigin],
    fft_im: UnsafePointer[Float32, MutExternalOrigin],
):
    """Centroid-subtracted temporal column: radix-32 FFT → Hermitian low-pass gate (`FREQ_KEEP`) → inverse FFT."""

    var r_src = UInt32(0)
    while r_src < N_ROWS:
        var src_i = Int(r_src * UInt32(stride) + col)
        fft_re[Int(r_src)] = base[src_i] - column_mean_raw
        fft_im[Int(r_src)] = Float32(0.0)
        r_src += 1

    _fft_inplace32_complex(fft_re, fft_im, True)
    _hermitian_mask32(fft_re, fft_im)
    _fft_inplace32_complex(fft_re, fft_im, False)

    r_src = UInt32(0)
    while r_src < N_ROWS:
        var dst_i = Int(r_src * UInt32(dense_stride) + col)
        dense[dst_i] = fft_re[Int(r_src)]
        r_src += 1


@always_inline
fn simd_dot_dense(
    a: UnsafePointer[Float32, MutExternalOrigin],
    b: UnsafePointer[Float32, MutExternalOrigin],
    dim: Int,
) -> Float32:
    var lanes = SIMD[DType.float32, nelts](0)

    var j: Int = 0
    while j + nelts <= dim:
        var va = a.load[width = nelts](j)
        var vb = b.load[width = nelts](j)
        lanes = va * vb + lanes
        j += nelts

    var scalar = Float32(0.0)
    var t: Int = 0
    while t < nelts:
        scalar += lanes[t]
        t += 1

    while j < dim:
        scalar += a[j] * b[j]
        j += 1
    return scalar


@always_inline
fn simd_axpy_row(
    acc_row: UnsafePointer[Float32, MutExternalOrigin],
    scale: Float32,
    x_row: UnsafePointer[Float32, MutExternalOrigin],
    dim: Int,
):
    var j: Int = 0
    while j + nelts <= dim:
        var vx = x_row.load[width = nelts](j)
        var vc = acc_row.load[width = nelts](j)
        var vs = SIMD[DType.float32, nelts](scale)
        acc_row.store[width = nelts](j, vs * vx + vc)
        j += nelts
    while j < dim:
        acc_row[j] = acc_row[j] + scale * x_row[j]
        j += 1


fn _f32_max(a: Float32, b: Float32) -> Float32:
    if a >= b:
        return a
    return b


@always_inline
fn _f32_min(a: Float32, b: Float32) -> Float32:
    if a <= b:
        return a
    return b


@always_inline
fn _kappa_safety_guard(x: Float32) -> Float32:
    """Clamp κ strictly to `[0,1]`; remap NaN/Huge non-finites (why: upstream GPU/ECC glitches cannot cross FFI uncapped)."""

    if x != x:
        return KAPPA_NONFINITE_SENTINEL
    var ax = x
    if ax < Float32(0.0):
        ax = -ax
    if ax >= Float32(3.402823e38):
        return KAPPA_NONFINITE_SENTINEL
    var lb = Float32(0.0)
    var ub = Float32(1.0)
    return _f32_min(_f32_max(x, lb), ub)


fn _fma_max_abs(a: Float32, b: Float32) -> Float32:
    var d = a - b
    if d >= Float32(0.0):
        return d
    return -d


@export("cyto_mojo_phase2_curvature_core", ABI="C")
fn cyto_mojo_phase2_curvature_core(
    view_bits: UInt64,
    optical_scale: Float32,
    out_kappa_bits: UInt64,
    out_clinical_bits: UInt64,
    work_bits: UInt64,
    work_elems: UInt32,
) -> UInt32:
    if (
        view_bits == UInt64(0)
        or out_kappa_bits == UInt64(0)
        or out_clinical_bits == UInt64(0)
        or work_bits == UInt64(0)
    ):
        return CYTO_FFI_BAD_ARG

    var view_ptr = UnsafePointer[CytoNeighborMatrixView, MutExternalOrigin](
        unsafe_from_address = Int(Int(view_bits))
    )

    # Mojo 0.26: `CytoNeighborMatrixView` is not `ImplicitlyCopyable`; bind fields
    # individually instead of `var view = view_ptr[]` (Rust POD layout unchanged).
    var view_data_bits = view_ptr[].data_bits
    var view_dims_d = view_ptr[].dims_d
    var view_row_count = view_ptr[].row_count
    var view_stride_elems = view_ptr[].stride_elems

    var D = Int(view_dims_d)
    if D <= Int(0) or UInt32(D) > MAX_D:
        return CYTO_FFI_SHAPE

    if view_row_count != N_ROWS:
        return CYTO_FFI_SHAPE

    var stride = Int(view_stride_elems)
    if stride < Int(D):
        return CYTO_FFI_SHAPE

    var need = Int(N_ROWS) * D + 2 * Int(N_ROWS) + D * D + 2 * D
    if Int(work_elems) < need:
        return CYTO_FFI_BAD_WORK

    var data = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address = Int(Int(view_data_bits))
    )
    var smooth = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address = Int(Int(work_bits))
    )
    var out_kappa = UnsafePointer[Float32, MutExternalOrigin](
        unsafe_from_address = Int(Int(out_kappa_bits))
    )
    var out_clinical = UnsafePointer[UInt32, MutExternalOrigin](
        unsafe_from_address = Int(Int(out_clinical_bits))
    )

    var sn = Int(N_ROWS) * D
    var fft_re = smooth + sn
    var fft_im = fft_re + Int(N_ROWS)
    var cov = fft_im + Int(N_ROWS)
    var vec_v = cov + D * D
    var vec_w = vec_v + D

    var zz_mu = Int(0)
    while zz_mu < D * D:
        cov[zz_mu] = Float32(0.0)
        zz_mu += 1
    var r_acc_mu = UInt32(0)
    while r_acc_mu < N_ROWS:
        var dc_mu = UInt32(0)
        while dc_mu < UInt32(D):
            var imu = Int(r_acc_mu * UInt32(stride) + dc_mu)
            cov[Int(dc_mu)] += data[imu]
            dc_mu += 1
        r_acc_mu += 1

    var inv_nr_mu = Float32(1.0) / Float32(N_ROWS)
    var cmd = Int(0)
    while cmd < D:
        cov[cmd] = cov[cmd] * inv_nr_mu
        cmd += 1

    var col = UInt32(0)
    while col < UInt32(D):
        _smooth_neighbor_column_into_dense(
            data,
            stride,
            col,
            cov[Int(col)],
            smooth,
            Int(D),
            fft_re,
            fft_im,
        )
        col += 1

    var mean_scratch = vec_w
    var mm = Int(0)
    while mm < D:
        mean_scratch[mm] = Float32(0.0)
        mm += 1

    var rr_mean = UInt32(0)
    while rr_mean < N_ROWS:
        var cc_mean = UInt32(0)
        while cc_mean < UInt32(D):
            var rr_i = Int(rr_mean * UInt32(D) + cc_mean)
            mean_scratch[Int(cc_mean)] += smooth[rr_i]
            cc_mean += 1
        rr_mean += 1

    var inv_nr = Float32(1.0) / Float32(N_ROWS)
    var cm = Int(0)
    while cm < D:
        mean_scratch[cm] *= inv_nr
        cm += 1

    var zz = Int(0)
    while zz < D * D:
        cov[zz] = Float32(0.0)
        zz += 1

    var trace_acc = Float32(0.0)
    var row_r = UInt32(0)
    while row_r < N_ROWS:
        var row_begin = smooth + Int(row_r * UInt32(D))

        var cc_ctr = UInt32(0)
        while cc_ctr < UInt32(D):
            var ij = Int(row_r * UInt32(D) + cc_ctr)
            smooth[ij] -= mean_scratch[Int(cc_ctr)]
            cc_ctr += 1

        trace_acc += simd_dot_dense(row_begin, row_begin, D)

        var ci = UInt32(0)
        while ci < UInt32(D):
            var scale = smooth[Int(row_r * UInt32(D) + ci)]
            var crow = cov + Int(ci) * D
            simd_axpy_row(crow, scale, row_begin, D)
            ci += 1

        row_r += 1

    var inv_nm1 = Float32(1.0) / Float32(N_ROWS - UInt32(1))
    var zp = Int(0)
    while zp < D * D:
        cov[zp] = cov[zp] * inv_nm1
        zp += 1

    var trace = trace_acc * inv_nm1
    if trace <= Float32(0.0):
        out_kappa[0] = Float32(0.0)
        out_clinical[0] = CYTO_CLINICAL_MALIGNANT
        return CYTO_FFI_OK

    var safe_trace = _f32_max(trace, SAFE_TRACE_EPS)

    var ratio: Float32
    if D == Int(2):
        # Largest eigenvalue of 2×2 PSD covariance (row-major symmetric `cov`): closed form avoids fragile
        # iterative Rayleigh quotients near repeated-root manifolds (why: unit-circle calibration needs ε~1e-7 accuracy).
        var cxx = cov[0]
        var cxy = cov[1]
        var cyy = cov[3]
        var disc_r = cxx - cyy
        var disc_inner = disc_r * disc_r + Float32(4.0) * cxy * cxy
        if disc_inner < Float32(0.0):
            disc_inner = Float32(0.0)
        var eig1 = Float32(0.5) * ((cxx + cyy) + sqrt(disc_inner))
        ratio = eig1 / safe_trace
    else:
        var vec_v_alt = vec_v
        var vec_w_alt = vec_w
        var vi_alt = UInt32(0)
        var init_scale_alt = Float32(1.0) / sqrt(Float32(D))
        while vi_alt < UInt32(D):
            vec_v_alt[Int(vi_alt)] = init_scale_alt
            vec_w_alt[Int(vi_alt)] = Float32(0.0)
            vi_alt += 1

        var eigen_est_alt = Float32(0.0)
        var eigen_prev_alt = Float32(-1.0)

        var it_alt = UInt32(0)
        while it_alt < UInt32(POWER_ITER_MAX):
            var ri_alt = UInt32(0)
            while ri_alt < UInt32(D):
                var rowp_alt = cov + Int(ri_alt) * D
                vec_w_alt[Int(ri_alt)] = simd_dot_dense(rowp_alt, vec_v_alt, D)
                ri_alt += 1

            eigen_est_alt = simd_dot_dense(vec_v_alt, vec_w_alt, D)

            var norm_w_alt = sqrt(simd_dot_dense(vec_w_alt, vec_w_alt, D))
            if norm_w_alt <= POWER_ITER_NORM_EPS:
                eigen_est_alt = Float32(0.0)
                break

            var inv_norm_alt = Float32(1.0) / norm_w_alt
            var xi_alt = UInt32(0)
            while xi_alt < UInt32(D):
                vec_v_alt[Int(xi_alt)] = vec_w_alt[Int(xi_alt)] * inv_norm_alt
                xi_alt += 1

            if _fma_max_abs(eigen_est_alt, eigen_prev_alt) < POWER_ITER_TOL_LAMBDA:
                break

            eigen_prev_alt = eigen_est_alt
            it_alt += 1

        ratio = eigen_est_alt / safe_trace

    if ratio > Float32(1.0):
        ratio = Float32(1.0)
    var kappa = _kappa_safety_guard(Float32(1.0) - ratio)
    out_kappa[0] = kappa

    var thr = BASE_KAPPA_THRESHOLD * optical_scale

    # 2D prior: κ→½ when spectra are balanced (rounded cell); collapsing κ implies anisotropic
    # morphology — flag malignancy whenever κ dips below calibrated mid-band × optical prior.
    var decision: UInt32
    if kappa < thr:
        decision = CYTO_CLINICAL_MALIGNANT
    else:
        decision = CYTO_CLINICAL_NORMAL
    out_clinical[0] = decision

    return CYTO_FFI_OK
