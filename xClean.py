from vapoursynth import core
import vapoursynth as vs
import math, functools

"""
xClean spatio/temporal denoiser beta 1 (2021-09-24)

Based on mClean (https://forum.doom9.org/showthread.php?t=174804) by burfadel

Mod by Etienne Charland:
- Added dynamic noise reduction strength based on Luma where dark areas get full reduction and 
white areas preserve more of the source. Set Strength between 0 and -200, recommended -50. A value of -50 means that out of 255 values (or 219 tv range), 
the 50 blackest values have full-reduction and the 50 whitest values are merged at a minimal strength of 50/255.
- Strength no longer apply to deband and sharpen, only to noise reduction.
- Deband was denoised and then sharpened. It has been moved to the end after sharpening.
- Veed is run between noise reduction and sharpening and is not affected by strength.
- Added boost denoising to denoise dark scenes using a different method (KNLMeansCL).
- Now support various denoising methods: MvTools2 (default) and KNLMeansCL (default for boost)
- thSAD parameter has been replaced by p1, which sets thSAD for MvTools2 and h for KNLMeansCL
- Can now run KNLMeansCL with renoise and sharpen using method=1

Requires: rgsv, rgvs, fmtc, vcm, mv, mvsf, tmedian

+++ Description +++
Typical spatial filters work by removing large variations in the image on a small scale, reducing noise but also making the image less
sharp or temporally stable. xClean removes noise whilst retaining as much detail as possible, as well as provide optional image enhancement.

xClean works primarily in the temporal domain, although there is some spatial limiting.
Chroma is processed a little differently to luma for optimal results.
Chroma processing can be disabled with chroma = False.

+++ Artifacts +++
Spatial picture artifacts may remain as removing them is a fine balance between removing the unwanted artifact whilst not removing detail.
Additional dering/dehalo/deblock filters may be required, but should ONLY be uses if required due the detail loss/artifact removal balance.

+++ Sharpening +++
Applies a modified unsharp mask to edges and major detected detail. Range of normal sharpening is 0-20. There are 4 additional settings,
21-24 that provide 'overboost' sharpening. Overboost sharpening is only suitable typically for high definition, high quality sources.
Actual sharpening calculation is scaled based on resolution.

+++ ReNoise +++
ReNoise adds back some of the removed luma noise. Re-adding original noise would be counterproductive, therefore ReNoise modifies this noise
both spatially and temporally. The result of this modification is the noise becomes much nicer and it's impact on compressibility is greatly
reduced. It is not applied on areas where the sharpening occurs as that would be counterproductive. Settings range from 0 to 20.
The strength of renoise is affected by the the amount of original noise removed and how this noise varies between frames.
It's main purpose is to reduce the 'flatness' that occurs with any form of effective denoising.

+++ Deband +++
This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. The settings are not adjustable as the default settings are suitable for most cases without having a large effect
on compressibility. 0 = disabled, 1 = deband only, 2 = deband and veed

+++ Depth +++
This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

+++ Strength +++
The strength of the denoising effect can be adjusted using this parameter. It ranges from 20 percent denoising effect with strength 0, up to the
100 percent of the denoising with strength 20. This function works by blending a scaled percentage of the original image with the processed image.
A value between 0 and -200 will apply dynamic noise reduction strength based on Luma, where black zones get full denoising and white areas
preserve the source. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction and the 50 whitest values 
are merged at a minimal strength of 50/255.

+++ Boost +++
Boost denoising using a secondary denoiser for dark scenes, between 0 and 100. 0 to disable.
Setting a value of 5 means that frames with average luma below 5% will be merged between method and boostm.
It will merge at a ratio of .8 - luma / boost * .8

+++ Method / MethodBoost +++
0 for MvTools (default), 1 for KNLMeansCL (default for dark scene boost)

+++ Outbits +++
Specifies the bits per component (bpc) for the output for processing by additional filters. It will also be the bpc that xClean will process.
If you output at a higher bpc keep in mind that there may be limitations to what subsequent filters and the encoder may support.

+++ p1, p2, b1, b2, f1, f2 +++
Parameters to configure denoising for method(p1/p2), boostm(b1/b2) and finalm(f1/f2)
For method = 0 (MVTools2): p1 = thSAD used for analysis (default=400)
For method = 1 (KNLMeansCL): p1 = h strength (default=1.4), p2 = device_id (default=0)
For method = 2 (B3MD): p1 = sigma strength (default=9.0), p2 = radius (default=1)

"""

def xClean(clip, chroma=True, sharp=10, rn=14, deband=0, depth=0, strength=20, m1=1, m2=0, m3=2, outbits=None, dmode=3, rgmode=18, thsad=400, d=2, a=2, h=1.4, gpuid=0, sigma=9, radius=0):
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family != vs.YUV:
        raise TypeError("xClean: This is not a YUV clip!")

    defH = max(clip.height, clip.width // 4 * 3) # Resolution calculation for auto blksize settings
    sharp = min(max(sharp, 0), 24) # Sharp multiplier
    rn = min(max(rn, 0), 20) # Luma ReNoise strength
    deband = min(max(deband, 0), 5)  # Apply deband/veed
    depth = min(max(depth, 0), 5) # Depth enhancement
    strength = min(max(strength, -200), 20) # Strength of denoising
    bd = clip.format.bits_per_sample
    outbits = outbits or bd
    #if outbits > 8 and outbits < 16:
    #    outbits = 16
    #if deband or depth: # plugins do not support 32-bit
    #    outbits = min(outbits, 16)

    # if method < 0 or method > 2:
    #     raise ValueError("xClean: method must be 0 (MvTools), 1 (KNLMeansCL) or 2 (BM3D)")
    # if boostm < 0 or boostm > 2:
    #     raise ValueError("xClean: boostm must be 0 (MvTools), 1 (KNLMeansCL) or 2 (BM3D)")

    # Eliminate impulsive noise
    c = clip
    c16 = c.fmtc.bitdepth(bits=16, dmode=1)
    c32 = c.fmtc.bitdepth(bits=32, dmode=1)
    # Apply Veed (auto-levels would also go here)
    #if deband == 2:
    #    c16 = core.vcm.Veed(c16)

    # Apply MVTools
    output = clean = c
    if m1 > 0:
        clean = MvTools(c32 if m1==3 else c16 if m1==2 else c, c32 if m1==3 else c16, chroma, defH, thsad)
        output = PostProcessing(clean, c32 if m1==3 else c16, defH, strength, sharp, rn, depth if m2==0 else 0, rgmode, 0)

    # Apply BM3D
    if m2 > 0:
        ref = output
        output = BM3D(c, sigma, radius, gpuid, chroma, ref, 32 if m2==3 else 16)
        if m2 < 3 and m3 == 3:
            output = output.fmtc.bitdepth(bits=32, dmode=1)
        output = PostProcessing(output, c32 if m2==3 or m3==3 else c16, defH, strength, sharp, rn, depth, rgmode, 1)
    
    # Apply KNLMeans
    if m3 > 0:
        m3bd = 32 if m3==3 else 16 if m3==2 else 8
        if output.format.bits_per_sample != m3bd:
            output = output.fmtc.bitdepth(bits=m3bd, dmode=0)
        ref = output
        output = KnlMeans(c, d, a, h, gpuid, chroma, ref)
        if m3bd < 16:
            output = output.fmtc.bitdepth(bits=16, dmode=1)
        output = PostProcessing(output, c32 if m3==3 else c16, defH, strength, sharp, rn, depth, rgmode, 2)
    
    # Apply deband
    if deband:
        if output.format.bits_per_sample == 32:
            output = output.fmtc.bitdepth(bits=16, dmode=0)
        output = output.f3kdb.Deband(range=16, preset="high" if chroma else "luma", grainy=defH/15, grainc=defH/16 if chroma else 0)
    
    # Convert to desired bitrate.
    if outbits != output.format.bits_per_sample :
        output = output.fmtc.bitdepth(bits=outbits, dmode=dmode)
    return output


def PostProcessing(clean, c, defH, strength, sharp, rn, depth, rgmode, method):
    # Only apply renoise & sharpen to MVTools method
    if rgmode == 0:
        sharp = 0
        rn = 0
        depth = 0
        rgmode = 0

    # Run at least in 16-bit
    if clean.format.bits_per_sample < 16:
        clean = clean.fmtc.bitdepth(bits=16, dmode=1)
    bd = clean.format.bits_per_sample
    
    # Separate luma and chroma
    filt = clean
    clean = core.std.ShufflePlanes(clean, [0], vs.GRAY) if clean.format.num_planes != 1 else clean
    cy = core.std.ShufflePlanes(c, [0], vs.GRAY)

    # Spatial luma denoising
    RG = core.rgsf.RemoveGrain if bd == 32 else core.rgvs.RemoveGrain
    clean2 = RG(clean, rgmode) if rgmode > 0 else clean

    # Apply dynamic noise reduction strength based on Luma
    if strength <= 0:
        # Slightly widen the exclusion mask to preserve details and edges
        cleanm = cy.std.Maximum()
        if defH > 500:
            cleanm = cleanm.std.Maximum()
        if defH > 1200:
            cleanm = cleanm.std.Maximum()

        # Adjust levels while respecting color range of each frame
        def AdjustLevels(n, f, clip, strength):
            fullRange = '_ColorRange' in f.props and f.props['_ColorRange'] == 0
            return clip.std.Levels((0 if fullRange else 16) - strength, 255 if fullRange else 235, 0.85, 0, 255+strength)
        cleanm = cleanm.std.FrameEval(functools.partial(AdjustLevels, clip=cleanm, strength=strength), prop_src=c)

        # Merge based on luma mask
        clean = core.std.MaskedMerge(clean, cy, cleanm)
        clean2 = core.std.MaskedMerge(clean2, cy, cleanm)
        filt = core.std.MaskedMerge(filt, c, cleanm)
    elif strength < 20:
        # Reduce strength by partially merging back with original
        clean = core.std.Merge(cy, clean, 0.2+0.04*strength)
        clean2 = core.std.Merge(cy, clean2, 0.2+0.04*strength)
        filt = core.std.Merge(c, filt, 0.2+0.04*strength)

    i = 0.00392 if bd == 32 else 1 << (bd - 8)
    peak = 1.0 if bd == 32 else (1 << bd) - 1
    depth2 = -depth*3
    depth = depth*2

    # Unsharp filter for spatial detail enhancement
    if sharp:
        RE = core.rgsf.Repair if bd == 32 else core.rgvs.Repair
        mult = .69 if method == 2 else .14 if method == 1 else 1
        if sharp > 20:
            sharp = (((sharp - 16) / 4) - 1) * mult + 1
            clsharp = core.std.MakeDiff(clean, clean2.tcanny.TCanny(sigma=sharp, mode=-1))
        else:
            sharp = min(50, (15 + defH * sharp * 0.0007) * mult)
            clsharp = core.std.MakeDiff(clean, Sharpen(clean2, amountH=-0.08-0.03*sharp))
        clsharp = core.std.MergeDiff(clean2, RE(clsharp.tmedian.TemporalMedian(), clsharp, 12))
    
    # If selected, combining ReNoise
    noise_diff = core.std.MakeDiff(clean2, cy)
    if rn:
        expr = "x {a} < 0 x {b} > {p} 0 x {c} - {p} {a} {d} - / * - ? ?".format(a=32*i, b=45*i, c=35*i, d=65*i, p=peak)
        clean1 = core.std.Merge(clean2, core.std.MergeDiff(clean2, Tweak(noise_diff.tmedian.TemporalMedian(), cont=1.008+0.00016*rn)), 0.3+rn*0.035)
        clean2 = core.std.MaskedMerge(clean2, clean1, core.std.Expr([core.std.Expr([clean, clean.std.Invert()], 'x y min')], [expr]))

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.std.Binarize().std.Invert()
    if rgmode > 0:
        clean2 = core.std.MaskedMerge(clean2, clsharp if sharp else clean, core.std.Expr([noise_diff, clean.std.Sobel()], 'x y max'))

    # Combining result of luma and chroma cleaning
    output = core.std.ShufflePlanes([clean2, filt], [0, 1, 2], vs.YUV)
    if depth:
        output = core.std.MergeDiff(output, core.std.MakeDiff(output.warp.AWarpSharp2(128, 3, 1, depth2, 1), output.warp.AWarpSharp2(128, 2, 1, depth, 1)))
    return output


# Default MvTools denoising method
def MvTools(c, c16, chroma, defH, thSAD):
    ref = c.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    icalc = c.format.bits_per_sample < 32
    cy = core.std.ShufflePlanes(c, [0], vs.GRAY)
    S = core.mv.Super if icalc else core.mvsf.Super
    A = core.mv.Analyse if icalc else core.mvsf.Analyse
    R = core.mv.Recalculate if icalc else core.mvsf.Recalculate

    sc = 8 if defH > 2880 else 4 if defH > 1440 else 2 if defH > 720 else 1
    bs = 16 if defH / sc > 360 else 8
    ov = 6 if bs > 12 else 2
    pel = 1 if defH > 720 else 2
    lampa = 777 * (bs ** 2) // 64
    truemotion = False if defH > 720 else True

    super1 = S(ref if chroma else core.std.ShufflePlanes(ref, [0], vs.GRAY), hpad=bs, vpad=bs, pel=pel, rfilter=4, sharp=1)
    super2 = S(c if chroma else cy, hpad=bs, vpad=bs, pel=pel, rfilter=1, levels=1)
    analyse_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion)
    recalculate_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion, thsad=180, _lambda=lampa)

    # Analysis
    bvec4 = R(super1, A(super1, isb=True,  delta=4, **analyse_args), **recalculate_args) if not icalc else None
    bvec3 = R(super1, A(super1, isb=True,  delta=3, **analyse_args), **recalculate_args)
    bvec2 = R(super1, A(super1, isb=True,  delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    bvec1 = R(super1, A(super1, isb=True,  delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec1 = R(super1, A(super1, isb=False, delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec2 = R(super1, A(super1, isb=False, delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    fvec3 = R(super1, A(super1, isb=False, delta=3, **analyse_args), **recalculate_args)
    fvec4 = R(super1, A(super1, isb=False, delta=4, **analyse_args), **recalculate_args) if not icalc else None

    # Applying cleaning
    if not icalc:
        clean = core.mvsf.Degrain4(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)
    else:
        clean = core.mv.Degrain3(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=thSAD)

    #if clean.format.bits_per_sample < 16:
    clean = clean.fmtc.bitdepth(bits=16, dmode=1)

    uv = core.std.MergeDiff(clean, core.tmedian.TemporalMedian(core.std.MakeDiff(c16, clean, [1, 2]), 1, [1, 2]), [1, 2]) if chroma else c16
    return core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)


# KnlMeansCL denoising method, useful for dark noisy scenes
def KnlMeans(clip, d, a, h, gpuid, chroma, ref):
    if ref.format.bits_per_sample != clip.format.bits_per_sample:
        ref = ref.fmtc.bitdepth(bits=clip.format.bits_per_sample, dmode=0)
    clean = clip.knlm.KNLMeansCL(d=d, a=a, h=h, device_id=gpuid, channels="Y", rclip=ref)
    uv = clip.knlm.KNLMeansCL(d=d, a=a, h=h/2, device_id=gpuid, channels="UV", rclip=ref) if chroma else clean
    return core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)


# BM3D denoising method
def BM3D(clip, sigma, radius, gpuid, chroma, ref, outbits):
    clean = clip.resize.Bicubic(format=vs.YUV444PS, matrix_in_s="709") if chroma else clip.fmtc.bitdepth(bits=32, dmode=1)
    ref = ref.resize.Bicubic(format=vs.YUV444PS, matrix_in_s="709") if chroma else ref.fmtc.bitdepth(bits=32, dmode=1)
    clean = clean.bm3dcuda.BM3D(sigma=[sigma,sigma,sigma], radius=radius, ref=ref, device_id=gpuid, fast=False)
    if radius > 0:
        clean = clean.bm3d.VAggregate(radius=radius)

    if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 1:
        fmt = vs.YUV420P16
    elif clip.format.subsampling_w == 1 and clip.format.subsampling_h == 0:
        fmt = vs.YUV422P16
    else:
        fmt = vs.YUV444PS if outbits == 32 else vs.YUV444P16
    return clean.resize.Bicubic(format=fmt, matrix_s="709", range = 1)

    # Convert back while respecting _ColorRange of each frame. Note that setting range=1 sets _ColorRange=0 (reverse)
    def ConvertBack(n, f, clean, format):
        fullRange = '_ColorRange' in f.props and f.props['_ColorRange'] == 0
        return clean.resize.Bicubic(format=format, matrix_s="709", range = 1 if fullRange else 0)
    #return clean.std.FrameEval(functools.partial(ConvertBack, clean=clean, format=clip.format), prop_src=clip)


def Tweak(clip, hue=None, sat=None, bright=None, cont=None, coring=True):
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if clip.format.color_family == vs.RGB:
        raise TypeError("Tweak: RGB color family is not supported!")
        
    if not (hue is None and sat is None or isGRAY):
        hue = 0.0 if hue is None else hue
        sat = 1.0 if sat is None else sat
        hue = hue * math.pi / 180
        sinh = math.sin(hue)
        cosh = math.cos(hue)
        cmin = -0.5 if isFLOAT else 16 << (bd - 8) if coring else 0
        cmax = 0.5 if isFLOAT else 240 << (bd - 8) if coring else (1 << bd) - 1
        expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "x {} - {} * y {} - {} * + {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "y {} - {} * x {} - {} * - {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        src_u = core.std.ShufflePlanes(clip, [1], vs.GRAY)
        src_v = core.std.ShufflePlanes(clip, [2], vs.GRAY)
        dst_u = core.std.Expr([src_u, src_v], expr_u)
        dst_v = core.std.Expr([src_u, src_v], expr_v)
        clip = core.std.ShufflePlanes([clip, dst_u, dst_v], [0, 0, 0], clip.format.color_family)

    if not (bright is None and cont is None):
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if isFLOAT:
            expr = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)
            clip =  core.std.Expr([clip], [expr] if isGRAY else [expr, ''])
        else:
            luma_lut = []
            luma_min = 16  << (bd - 8) if coring else 0
            luma_max = 235 << (bd - 8) if coring else (1 << bd) - 1

            for i in range(1 << bd):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = core.std.Lut(clip, [0], luma_lut)

    return clip

# from muvsfunc
def Sharpen(clip: vs.VideoNode, amountH = 1.0, amountV = None, planes = None) -> vs.VideoNode:
    """Avisynth's internel filter Sharpen()
    Simple 3x3-kernel sharpening filter.
    Args:
        clip: Input clip.
        amountH, amountV: (float) Sharpen uses the kernel is [(1-2^amount)/2, 2^amount, (1-2^amount)/2].
            A value of 1.0 gets you a (-1/2, 2, -1/2) for example.
            Negative Sharpen actually blurs the image.
            The allowable range for Sharpen is from -1.58 to +1.0.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.
        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".
    """

    funcName = 'Sharpen'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    if planes is None:
        planes = list(range(clip.format.num_planes))

    center_weight_v = math.floor(2 ** (amountV - 1) * 1023 + 0.5)
    outer_weight_v = math.floor((0.25 - 2 ** (amountV - 2)) * 1023 + 0.5)
    center_weight_h = math.floor(2 ** (amountH - 1) * 1023 + 0.5)
    outer_weight_h = math.floor((0.25 - 2 ** (amountH - 2)) * 1023 + 0.5)

    conv_mat_v = [outer_weight_v, center_weight_v, outer_weight_v]
    conv_mat_h = [outer_weight_h, center_weight_h, outer_weight_h]

    if math.fabs(amountH) >= 0.00002201361136: # log2(1+1/65536)
        clip = core.std.Convolution(clip, conv_mat_v, planes=planes, mode='v')

    if math.fabs(amountV) >= 0.00002201361136:
        clip = core.std.Convolution(clip, conv_mat_h, planes=planes, mode='h')

    return clip


#def DetectDark(clip):
    # Trim the brighest luma if less than 15% of pixels
    #bright = core.std.Expr(clips=clip, expr=["x 204 > 255 0 ?", ""])
    #brigh