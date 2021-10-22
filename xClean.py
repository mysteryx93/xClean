from vapoursynth import core
import vapoursynth as vs
import math
from typing import Optional

"""
xClean 3-pass denoiser
beta 5 (2021-10-20) by Etienne Charland
Supported formats: YUV, GRAY
Requires: rgsf, rgvs, fmtc, mv, mvsf, tmedian, knlm, bm3d, bm3dcuda_rtc, bm3dcpu, neo_f3kdb, akarin

xClean runs MVTools -> BM3D -> KNLMeans in that order, passing the output of each pass as the ref of the next denoiser.

The objective is to remove noise while preserving as much details as possible. Removing noise is easy -- just blur out everything.
The hard work is in preserving the details in a way that feels natural.

Designed for raw camera footage to remove noise in dark areas while preserving the fine details. It works for most types of content.

Performance-wise, BM3D pass is the heaviest and helps recover fine details, but this script runs 1 pass of BM3D whereas stand-alone BM3D runs twice.


+++ Short Doc (TL;DR) +++
Default settings provide the best quality in most cases. Simply use
xClean(sharp=..., outbits=...)

If only darker areas contain noise, set strength=-50
For better performance, set m1=0 or m2=0, or set m1=.5 and m2=3.6 (downscale)
BM3D performance can be greatly improved by setting radius=0, block_step=7, bm_range=7, ps_range=5

For 720p WebCam, optimal settings are: sharp=9.5, m1=.65, h=2.8
For 288p anime, optimal settings are: sharp=9.5, m1=.7, rn=0, optional depth=1
For 4-5K GoPro (with in-camera sharpening at Low), optimal settings are: sharp=7.7, m1=.5, m2=3.7, optional strength=-50 (or m1=.6, m2=3.8 if your computer can handle it)


+++ Description +++
KNLMeans does a good job at denoising but can soften the image, lose details and give an artificial plastic look. I found that on any given source
(tested 5K GoPro footage and noisy WebCam), denoising with less than h=1.4 looks too noisy, and anything above it blurs out the details. 
KNLMeans also keeps a lot of data from the clip passed as rclip, so doing a good prefilter highly impacts the output.

Similarly, BM3D performs best with sigma=9. A lower value doesn't remove enough noise, and a higher value only makes the edges sharper.

xClean is essentially KNLMeans with advanced pre-filtering and with post-processing to renoise & sharpen to make the image look more natural.

One strange aspect of xClean is that denoising is automatic and there's very little room to configure denoising strength other than reducing the overall effect.
It runs with BM3D sigma=9 and KNL h=1.4, and generally you shouldn't change that. One setting that can allow increasing denoising (and performance)
is downscaling MVTools and BM3D passes. You can also set h=2.8 if the output remains too noisy. h = 1.4 or 2.8 are generally the best values.

According to my tests, water & cliff 5K video with little noise preserves the details very well while removing subtle grain, and with same settings,
very noisy 720p WebCam footage has HUGE noise reduction while preserving a surprising amount of natural details.

The default settings are very tolerant to various types of clips.


+++ Denoising Methods Overview +++
To provide the best output, processing is done in 3 passes, passing the output of one pass as the ref clip of the 2nd pass. Each denoiser has its strengths and weaknesses.

Pass 1: MVTools (m1)
Strength: Removes a lot of noise, good at removing temporal noise.
Weakness: Can remove too much, especially with delicate textures like water.
Ref: Impacts vectors analysis but low impact on outcome

Pass 2: BM3D (m2)
Strength: Good at preserving fine details!
Weakness: Doesn't remove much grain.
Ref: Moderate impact on outcome. A blurry ref will remove more grain while BM3D puts back a lot of details.

Pass 3: KNLMeansCL (m3)
Strength: Best general-purpose denoiser
Weakness: Can blur out details and give an artificial plastic effect
Ref: High impact the outcome. All prefilters benefit from running KNLMeans over it.


+++ Denoising Pass Configuration  (m1=.6, m2=3, m3=3) +++
Each pass (method) can be configured with m1 (MVTools), m2 (BM3D) and m3 (KNLMeansCL) parameters to run at desired bitdepth.
This means you can fine-tune for quality vs performance.

0 = Disabled, 1 = 8-bit, 2 = 16-bit, 3 = 16-bit YUV444, 4 = 32-bit YUV444

Note: BM3D always processes in 32-bit, KNLMeansCL always processes in 16-bit+, and post-processing always processes at least in 16-bit, so certain
values such as m2=1, m3=1 will behave the same as m2=2, m3=2. Setting m2=2 instead of 3 will only affect BM3D post-processing (YUV420P16 instead of YUV444P16)

MVTools (m1) and BM3D (m2) passes can also be downscaled for performance gain, and it can even improve quality! Values between .5 and .8 generally work best.

Optional resize factor is set after the dot:
m1 = .6 or 1.6 processes MVTools in 8-bit at 60% of the size. m2 = 3.6 processes BM3D in 16-bit at 60% of the size.
You may want to downscale MVTools (m1) because of high CPU usage and low impact on outcome.
You may want to downscale BM3D (m2) because of high memory usage. If you run out of memory, lower the size until you get no hard-drive paging.
Note: Setting radius=0 greatly reduces BM3D memory usage!


+++ Renoise and Sharpen  (rn=14, sharp=9.5) +++
The idea comes from mClean by Burfadel (https://forum.doom9.org/showthread.php?t=174804) and the algorithm was changed by someone else while porting 
to VapourSynth, producing completely different results -- original Avisynth version blurs a lot more, VapourSynth version keeps a lot more details.

It may sound counter-productive at first, but the idea is to combat the flat or plastic effect of denoising by re-introducing part of the removed noise.
The noise is processed and stabilized before re-inserting so that it's less distracting.
Renoise also helps reduce large-radius grain; but should be disabled for anime (rn=0).

Using the same analysis data, it's also sharpening to compensate for denoising blur.
Sharpening must be between 0 and 20. Actual sharpening calculation is scaled based on resolution.


+++ Strength / Dynamic Denoiser Strength  (strength=20) +++
A value of 20 will denoise normally.
A value between 1 and 19 will reduce the denoising effect by partially merging back with the original clip.
A value between 0 and -200 will activate Dynamic Denoiser Strength, useful when bright colors require little or no denoising and dark colors contain more noise.
It applies a gradual mask based on luma. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction 
and the 50 whitest values are merged at a minimal strength of 50/255 = 20%.

+++ Radius  (radius=0) +++
BM3D radius. Low impact on individual frames.
Pros: Helps stabilize temporal grain. Can significantly improve video compressability.
Cons: High impact on performance and memory usage! May require downscaling BM3D for HD content with m2 between 3.6 and 3.8
For moving water, the temporal stabilization may be undesirable.

+++ Depth  (depth=0) +++
This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

+++ Deband  (deband=False) +++
This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. Default settings are suitable for most cases without having a large effect on compressibility.

+++ Output  (outbits, dmode=0) +++
Specifies the output bitdepth. If not specified it will be converted back to the bitdepth of the source clip using dithering method specified by dmode.
You can set dmode=3 if you won't be doing any further processing for high-quality ditherig.

+++ Chroma  (chroma=False) +++
True to process both Luma and Chroma planes, False to process only Luma.

+++ Anime +++
For anime, set rn=0. Optionally, you can set depth to 1 or 2 to thicken the lines.

+++ Advanced Settings +++
gpuid = 0: The GPU id to use for KNLMeans and BM3D, or -1 to use CPU.
gpucuda = 0: The GPU id to use for BM3D, or -1 to use CPU.
h = 1.4: KNLMeans strength, can increase slightly if the output is still too noisy. 1.4 or 2.8 generally work best.
block_step = 4, bm_range = 16, ps_range = 8: BM3D parameters for performance vs quality. No impact on CPU and memory. Adjust based on GPU capability.
Fast settings are block_step = 5, bm_range = 7, ps_range = 5

Normally you shouldn't have to touch these
rgmode = 18: RemoveGrain mode used during post-processing. Setting this to 0 disables post-processing, useful to compare raw denoising.
thsad = 400: Threshold used for MVTools analysis.
d = 2: KNLMeans temporal radius. Setting 3 can either slightly improve quality or give a slight plastic effect.
a = 2: KNLMeans spacial radius.
sigma = 9: BM3D strength.
bm3d_fast = False. BM3D fast.
opp = True. Whether to convert to OPP format for BM3D, and to YCgCoR format for KNLMeans.
"""

def xClean(clip: vs.VideoNode, chroma: bool = True, sharp: float = 9.5, rn: float = 14, deband: bool = False, depth: int = 0, strength: int = 20, m1: float = .6, m2: int = 3, m3: int = 3, outbits: Optional[int] = None,
        dmode: int = 0, rgmode: int = 18, thsad: int = 400, d: int = 2, a: int = 2, h: float = 1.4, gpuid: int = 0, gpucuda: Optional[int] = None, sigma: float = 9, 
        block_step: int = 4, bm_range: int = 16, ps_range: int = 8, radius: int = 0, bm3d_fast: bool = False, opp: bool = True) -> vs.VideoNode:

    if not clip.format.color_family in [vs.YUV, vs.GRAY]:
        raise TypeError("xClean: Only YUV or GRAY clips are supported")

    defH = max(clip.height, clip.width // 4 * 3) # Resolution calculation for auto blksize settings
    if sharp < 0 or sharp > 20:
        raise ValueError("xClean: sharp must be between 0 and 20")
    if rn < 0 or rn > 20:
        raise ValueError("xClean: rn (renoise strength) must be between 0 and 20")
    if depth < 0 or depth > 5:
        raise ValueError("xClean: depth must be between 0 and 5")
    if strength < -200 or strength > 20:
        raise ValueError("xClean: strength must be between -200 and 20")
    if m1 < 0 or m1 >= 5:
        raise ValueError(r"xClean: m1 (MVTools pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (16-bit YUV444) or 4 (32-bit YUV444), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    if m2 < 0 or m2 > 4:
        raise ValueError("xClean: m2 (BM3D pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (16-bit YUV444) or 4 (32-bit YUV444)")
    if m3 < 0 or m3 > 4:
        raise ValueError("xClean: m3 (KNLMeansCL pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (16-bit YUV444) or 4 (32-bit YUV444)")
    if m1 == 0 and m2 == 0 and m3 == 0:
        raise ValueError("xClean: At least one pass must be enabled")

    uv = clip
    if not chroma:
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)

    gpucuda = gpucuda if gpucuda != None else gpuid
    bd = clip.format.bits_per_sample
    fulls = GetColorRange(clip) == 0
    samp = ClipSampling(clip)
    is444 = samp == "444"
    isGray = samp == "GRAY"
    outbits = outbits or bd
    if not outbits in [8, 10, 12, 14, 16, 32]:
        raise ValueError("xClean: outbits must be 8, 10, 12, 14, 16 or 32")
    
    c = c16 = c16_444 = c32_444 = clip
    if bd != 8:
        c = ConvertBits(c, 8, fulls, True)
    if bd != 16:
        c16 = ConvertBits(c, 16, fulls, True)
    if bd != 16 or not is444:
        c16_444 = clip.resize.Bicubic(format=vs.YUV444P16) if not isGray else c16
    if bd != 32 or not is444:
        c32_444 = clip.resize.Bicubic(format=vs.YUV444PS) if not isGray else ConvertBits(c, 32, fulls, False)
    output = None

    # Apply MVTools
    if m1 > 0:
        m1r = 1 if m1 == int(m1) else m1 % 1 # Decimal point is resize factor
        m1 = int(m1)
        c1 = c32_444 if m1 == 4 else c16_444 if m1 == 3 else c16 if m1 == 2 else c
        c1r = c1.resize.Bicubic((c.width * m1r)//4*4, (c.height * m1r)//4*4, filter_param_a=0, filter_param_a_uv=0, filter_param_b=.75, filter_param_b_uv=.75) if m1r < 1 else c1
        output = MvTools(c1r, defH, thsad)
        sharp1 = max(0, min(20, sharp + (1 - m1r) * .35))
        output = PostProcessing(output, c1r, defH, strength, sharp1, rn, rgmode, 0)

    # Apply BM3D
    if m2 > 0:
        m2r = 1 if m2 == int(m2) else m2 % 1 # Decimal point is resize factor
        m2 = int(m2)
        m2o = max(2, max(m2, m3))
        c2 = c32_444 if m2o==4 else c16_444 if m2o==3 else c16
        ref = output.resize.Spline36((c.width * m2r)//4*4, (c.height * m2r)//4*4, format=c2.format) if output else None
        c2r = c2.resize.Bicubic((c.width * m2r)//4*4, (c.height * m2r)//4*4, filter_param_a=0, filter_param_a_uv=0, filter_param_b=.5, filter_param_b_uv=.5) if m2r < 1 else c2
        output = BM3D(c2r, ref, sigma, gpucuda, m2o, block_step, bm_range, ps_range, radius, bm3d_fast, opp)
        output = output.resize.Spline36(c.width, c.height) if m2r < 1 else output
        sharp2 = max(0, min(20, sharp + (1 - m2r) * .95))
        output = PostProcessing(output, c2, defH, strength, sharp2, rn, rgmode, 1)

    if output.height < c.height:
        output = output.resize.Spline36(c.width, c.height)

    # Apply KNLMeans
    if m3 > 0:
        m3 = min(2, m3) # KNL internally computes in 16-bit
        ref = ConvertToM(output, clip, m3) if output else None
        c3 = c32_444 if m3==4 else c16_444 if m3==3 or ClipSampling(ref) == "444" else c16
        output = KnlMeans(c3, ref, d, a, h, gpuid, opp)
        # Adjust sharp based on h parameter.
        sharp3 = max(0, min(20, sharp - .5 + (h/2.8)))
        output = PostProcessing(output, c3, defH, strength, sharp3, rn, rgmode, 2)

    # Add Depth (thicken lines for anime)
    if depth:
        depth2 = -depth*3
        depth = depth*2
        output = core.std.MergeDiff(output, core.std.MakeDiff(output.warp.AWarpSharp2(128, 3, 1, depth2, 1), output.warp.AWarpSharp2(128, 2, 1, depth, 1)))
    
    # Apply deband
    if deband:
        if output.format.bits_per_sample == 32:
            output = ConvertBits(output, 16, fulls, False)
        output = output.neo_f3kdb.Deband(range=16, preset="high" if chroma else "luma", grainy=defH/15, grainc=defH/16 if chroma else 0)

    # Convert to desired bitrate
    outsamp = ClipSampling(output)
    if outsamp != samp:
        output = output.fmtc.resample(kernel="bicubic", css=samp, fulls=fulls, fulld=fulls)
    if output.format.bits_per_sample != outbits:
        output = output.fmtc.bitdepth(bits=outbits, fulls=fulls, fulld=fulls, dmode=dmode)
    
    # Merge source chroma planes if not processing chroma.
    if not chroma and uv.format.color_family == vs.YUV:
        if uv.format.bits_per_sample != outbits:
            uv = ConvertBits(uv, outbits, fulls, True)
        output = core.std.ShufflePlanes([output, uv], [0, 1, 2], vs.YUV)
    
    return output


def PostProcessing(clean: vs.VideoNode, c: vs.VideoNode, defH: int, strength: int, sharp: float, rn: float, rgmode: int, method: int) -> vs.VideoNode:
    fulls = GetColorRange(c) == 0
    if rgmode == 0:
        sharp = rn = 0

    # Run at least in 16-bit
    if clean.format.bits_per_sample < 16:
        clean = ConvertBits(clean, 16, fulls, False)
    if c.format.bits_per_sample < 16:
        c = ConvertBits(c, 16, fulls, False)
    bd = clean.format.bits_per_sample
    
    # Separate luma and chroma
    filt = clean
    clean = core.std.ShufflePlanes(clean, [0], vs.GRAY)
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

        # Adjust mask levels
        cleanm = cleanm.std.Levels((0 if fulls else 16) - strength, 255 if fulls else 235, 0.85, 0, 255+strength)

        # Merge based on luma mask
        clean = core.std.MaskedMerge(clean, cy, cleanm)
        clean2 = core.std.MaskedMerge(clean2, cy, cleanm)
        filt = core.std.MaskedMerge(filt, c, cleanm)
    elif strength < 20:
        # Reduce strength by partially merging back with original
        clean = core.std.Merge(cy, clean, 0.2+0.04*strength)
        clean2 = core.std.Merge(cy, clean2, 0.2+0.04*strength)
        filt = core.std.Merge(c, filt, 0.2+0.04*strength)

    # Unsharp filter for spatial detail enhancement
    if sharp:
        RE = core.rgsf.Repair if bd == 32 else core.rgvs.Repair
        mult = .69 if method == 2 else .14 if method == 1 else 1
        sharp = min(50, (15 + defH * sharp * 0.0007) * mult)
        clsharp = core.std.MakeDiff(clean, Sharpen(clean2, amountH=-0.08-0.03*sharp))
        clsharp = core.std.MergeDiff(clean2, RE(clsharp.tmedian.TemporalMedian(), clsharp, 12))
    
    # If selected, combining ReNoise
    noise_diff = core.std.MakeDiff(clean2, cy)

    if rn:
        i = 0.00392 if bd == 32 else 1 << (bd - 8)
        peak = 1.0 if bd == 32 else (1 << bd) - 1
        expr = "x {a} < 0 x {b} > {p} 0 x {c} - {p} {a} {d} - / * - ? ?".format(a=32*i, b=45*i, c=35*i, d=65*i, p=peak)
        clean1 = core.std.Merge(clean2, core.std.MergeDiff(clean2, Tweak(noise_diff.tmedian.TemporalMedian(), cont=1.008+0.00016*rn)), 0.3+rn*0.035)
        clean2 = core.std.MaskedMerge(clean2, clean1, core.std.Expr([core.std.Expr([clean, clean.std.Invert()], 'x y min')], [expr]))

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.std.Binarize().std.Invert()
    if rgmode > 0:
        clean2 = core.std.MaskedMerge(clean2, clsharp if sharp else clean, core.std.Expr([noise_diff, clean.std.Sobel()], 'x y max'))

    # Combining result of luma and chroma cleaning
    return core.std.ShufflePlanes([clean2, filt], [0, 1, 2], vs.YUV) if c.format.color_family == vs.YUV else clean2


# mClean denoising method
def MvTools(c: vs.VideoNode, defH: int, thSAD: int) -> vs.VideoNode:
    bd = c.format.bits_per_sample
    fulls = GetColorRange(c) == 0
    icalc = bd < 32
    S = core.mv.Super if icalc else core.mvsf.Super
    A = core.mv.Analyse if icalc else core.mvsf.Analyse
    R = core.mv.Recalculate if icalc else core.mvsf.Recalculate

    sc = 8 if defH > 2880 else 4 if defH > 1440 else 2 if defH > 720 else 1
    bs = 16 if defH / sc > 360 else 8
    ov = 6 if bs > 12 else 2
    pel = 1 if defH > 720 else 2
    lampa = 777 * (bs ** 2) // 64
    truemotion = False if defH > 720 else True

    ref = c.std.Convolution(matrix=[2, 3, 2, 3, 6, 3, 2, 3, 2])
    super1 = S(ref, hpad=bs, vpad=bs, pel=pel, rfilter=4, sharp=1)
    super2 = S(c, hpad=bs, vpad=bs, pel=pel, rfilter=1, levels=1)
    analyse_args = { 'blksize': bs, 'overlap': ov, 'search': 5, 'truemotion': truemotion }
    recalculate_args = { 'blksize': bs, 'overlap': ov, 'search': 5, 'truemotion': truemotion, 'thsad': 180, 'lambda': lampa }

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
    if icalc:
        clean = core.mv.Degrain3(c, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=thSAD)
    else:
        clean = core.mvsf.Degrain4(c, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)

    if bd < 16:
        clean = ConvertBits(clean, 16, fulls, False)
        c = ConvertBits(c, 16, fulls, False)

    if c.format.color_family == vs.YUV:
        uv = core.std.MergeDiff(clean, core.tmedian.TemporalMedian(core.std.MakeDiff(c, clean, [1, 2]), 1, [1, 2]), [1, 2])
        clean = core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)
    return clean


# BM3D denoising method
def BM3D(clip: vs.VideoNode, ref: Optional[vs.VideoNode], sigma: float, gpuid: int, m: int, block_step: int, bm_range: int, ps_range: int, radius: int, bm3d_fast: bool, opp: bool) -> vs.VideoNode:
    matrix = GetMatrix(clip)
    fulls = GetColorRange(clip)
    chroma = clip.format.color_family==vs.YUV
    opp = opp and chroma
    icalc = clip.format.bits_per_sample < 32
    clean = YUV2OPP(clip, 1) if opp else clip.resize.Bicubic(format=vs.YUV444PS if chroma else vs.GRAYS, matrix_in=matrix)
    if ref:
        ref = YUV2OPP(ref, 1) if opp else ref.resize.Bicubic(format=vs.YUV444PS if chroma else vs.GRAYS, matrix_in=matrix)
    if gpuid >= 0:
        clean = core.bm3dcuda_rtc.BM3D(clean, ref, chroma=chroma, sigma=sigma, device_id=gpuid, fast=bm3d_fast, radius=radius, block_step=block_step, bm_range=bm_range, ps_range=ps_range)
    else:
        clean = core.bm3dcpu.BM3D(clean, ref, chroma=chroma, sigma=sigma, block_step=block_step, bm_range=bm_range, ps_range=ps_range, radius=radius)
    clean = clean.bm3d.VAggregate(sample=0 if icalc else 1) if radius > 0 else clean
    clean = OPP2YUV(clean, clip, vs.YUV444P16 if m < 4 else vs.YUV444PS) if opp else clean
    return ConvertToM(clean, clip, m)


# KnlMeansCL denoising method, useful for dark noisy scenes
def KnlMeans(clip: vs.VideoNode, ref: Optional[vs.VideoNode], d: int, a: int, h: float, gpuid: int, opp: bool) -> vs.VideoNode:
    if ref and ref.format != clip.format:
        ref = ref.resize.Bicubic(format=clip.format)
    opp = opp and ClipSampling(clip) == "444"
    src = clip
    sample = 1 if clip.format.bits_per_sample == 32 else 0
    if opp:
        clip = YUV2YCC(clip, sample)
        if ref:
            ref = YUV2YCC(ref, sample)

    device = dict(device_type="auto" if gpuid >= 0 else "cpu", device_id=max(0, gpuid))
    if clip.format.color_family == vs.GRAY:
        output = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, **device)
    elif ClipSampling(clip) == "444":
        output = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="YUV", rclip=ref, **device)
    else:
        clean = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, **device)
        uv = clip.knlm.KNLMeansCL(d=d, a=a, h=h/2, channels="UV", rclip=ref, **device)
        output = core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)
    return YCC2YUV(output, src) if opp else output


def ConvertToM(c: vs.VideoNode, src: vs.VideoNode, m: int) -> vs.VideoNode:
    if src.format.color_family == vs.GRAY:
        fmt = vs.GRAY32 if m == 4 else vs.GRAY16 if m == 3 or m == 2 else vs.GRAY8
    else:
        samp = ClipSampling(c)
        fmt = vs.YUV444PS if m == 4 else vs.YUV444P16 if m == 3 else vs.YUV420P16 if samp == "420" else vs.YUV422P16 if samp == "422" else vs.YUV444P16
    if c.format == fmt:
        return c
    elif c.format.color_family in [vs.YUV, vs.GRAY]:
        return c.resize.Bicubic(format=fmt)
    else:
        # Convert back while respecting ColorRange, Matrix, ChromaLocation, Transfer and Primaries. Note that setting range=1 sets _ColorRange=0 (reverse)
        # transfer = GetTransfer(src)
        # primaries = GetPrimaries(src)
        return c.resize.Bicubic(format=fmt, matrix=GetMatrix(src), chromaloc=GetChromaLoc(src), range=1 if GetColorRange(src) == 0 else 0)


# Adjusts brightness and contrast
def Tweak(clip: vs.VideoNode, bright: float = None, cont: float = None) -> vs.VideoNode:
    fulls = GetColorRange(clip) == 0
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if clip.format.color_family == vs.RGB:
        raise TypeError("Tweak: RGB color family is not supported!")

    if not (bright is None and cont is None):
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if isFLOAT:
            expr = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)
            clip =  core.std.Expr([clip], [expr] if isGRAY else [expr, ''])
        else:
            luma_lut = []
            luma_min = 16  << (bd - 8) if not fulls else 0
            luma_max = 235 << (bd - 8) if not fulls else (1 << bd) - 1

            for i in range(1 << bd):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = core.std.Lut(clip, [0], luma_lut)
    return clip


# from muvsfunc
def Sharpen(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None, planes = None) -> vs.VideoNode:
    # Avisynth's internel filter Sharpen()
    funcName = 'Sharpen'
    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    planes = list(range(clip.format.num_planes)) if planes is None else planes

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


def ClipSampling(clip: vs.VideoNode) -> str:
    return "GRAY" if clip.format.color_family == vs.GRAY else \
            "RGB" if clip.format.color_family == vs.RGB else \
            ("444" if clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0 else \
            "422" if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 0 else "420") \
            if clip.format.color_family == vs.YUV else "UNKNOWN"


# Point resize is 1.5x faster BUT fmtc requires less memory
def ConvertBits(c: vs.VideoNode, bits: int = 8, fulls: bool = False, dither: bool = False):
    return c.fmtc.bitdepth(bits=bits, fulls=fulls, fulld=fulls, dmode=0 if dither else 1)
    samp = ClipSampling(c)
    i = 0 if bits==8 else 1 if bits==10 else 2 if bits==12 else 3 if bits==14 else 4 if bits==16 else 5 if bits==32 else -1
    if i == -1:
        raise ValueError("ConvertBits: outbits must be 8, 10, 12, 14, 16 or 32")

    fmt = 0
    if samp == "GRAY":
        fmt = [vs.GRAY8, vs.GRAY10, vs.GRAY12, vs.GRAY14, vs.GRAY16, vs.GRAY32] [i]
    elif samp == "RGB":
        fmt = [vs.RGB24, 0, 0, 0, vs.RGB48, vs.RGBS] [i]
    elif samp == "444":
        fmt = [vs.YUV444P8, vs.YUV444P10, vs.YUV444P12, vs.YUV444P14, vs.YUV444P16, vs.YUV444PS] [i]
    elif samp == "422":
        fmt = [vs.YUV422P8, vs.YUV422P10, vs.YUV422P12, vs.YUV422P14, vs.YUV422P16, 0] [i]
    elif samp == "420":
        fmt = [vs.YUV420P8, vs.YUV420P10, vs.YUV420P12, vs.YUV420P14, vs.YUV420P16, 0] [i]
    if fmt == 0:
        raise ValueError(f"ConvertBits: Invalid bitdepth ({bits}) for format ({samp})")
    
    range = 1 if fulls else 0
    return c.resize.Point(format=fmt, dither_type="ordered" if dither else "none", range=range, range_in=range)


# Get frame properties
def GetFrameProp(c: vs.VideoNode, name: str, default):
    props = c.get_frame(0).props
    return props[name] if name in props else default

def GetColorRange(c: vs.VideoNode) -> int:
    return GetFrameProp(c, "_ColorRange", 1)

def GetMatrix(c: vs.VideoNode) -> int:
    matrix = GetFrameProp(c, "_Matrix", 1)
    return 6 if matrix in [0, 2] else matrix

def GetTransfer(c: vs.VideoNode) -> int:
    transfer = GetFrameProp(c, "_Transfer", 0)
    return GetMatrix(c) if transfer in [0, 2] else transfer

def GetPrimaries(c: vs.VideoNode) -> int:
    primaries = GetFrameProp(c, "_Primaries", 0)
    return GetMatrix(c) if primaries in [0, 2] else primaries

def GetChromaLoc(c: vs.VideoNode) -> int:
    return GetFrameProp(c, "_ChromaLocation", 0)

def YUV2OPP(clip: vs.VideoNode, sample: int = 0):
    fulls = GetColorRange(clip) == 0
    clip = clip.resize.Bicubic(format = vs.RGBS if sample > 0 else vs.RGB48, matrix_in=GetMatrix(clip))
    clip = RGB_to_OPP(clip, fulls)
    return clip.std.SetFrameProp(prop='_Matrix', intval=2)

def OPP2YUV(clip: vs.VideoNode, src: vs.VideoNode, format: Optional[int] = None):
    fulls = GetColorRange(src) == 0
    format = format if format != None else src.format
    clip = OPP_to_RGB(clip, fulls)
    return clip.resize.Bicubic(format=format, matrix=GetMatrix(src), chromaloc=GetChromaLoc(src), range=1 if fulls else 0)

def YUV2YCC(clip: vs.VideoNode, sample: int = 0):
    fulls = GetColorRange(clip) == 0
    clip = clip.resize.Bicubic(format = vs.RGBS if sample > 0 else vs.RGB48, matrix_in=GetMatrix(clip))
    clip = RGB_to_YCgCoR(clip, fulls)
    return clip.std.SetFrameProp(prop='_Matrix', intval=2)

def YCC2YUV(clip: vs.VideoNode, src: vs.VideoNode, format: Optional[int] = None):
    fulls = GetColorRange(src) == 0
    format = format if format != None else src.format
    clip = YCgCoR_to_RGB(clip, fulls)
    return clip.resize.Bicubic(format=format, matrix=GetMatrix(src), chromaloc=GetChromaLoc(src), range=1 if fulls else 0)


# RGB to YCgCo RCT function
def RGB_to_YCgCoR (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.RGB:
        raise TypeError("RGB_to_YCgCoR: Clip is not in RGB format!")

    bd = c.format.bits_per_sample
    R = core.std.ShufflePlanes(c, [0], vs.GRAY)
    G = core.std.ShufflePlanes(c, [1], vs.GRAY)
    B = core.std.ShufflePlanes(c, [2], vs.GRAY)

    Co = core.akarin.Expr([R,      B], ex_dlut("x 0.5  * y 0.5  * - range_half +",                bd, fulls))
    Cg = core.akarin.Expr([Co, G,  B], ex_dlut("y z x range_half - 0.5 * + - 0.5 * range_half +", bd, fulls))
    Y  = core.akarin.Expr([Co, Cg, B], ex_dlut("z x range_half - 0.5 * + y range_half - +",       bd, fulls))

    return core.std.ShufflePlanes([Y, Cg, Co], [0, 0, 0], vs.YUV)


#  YCgCo RCT to RGB function
def YCgCoR_to_RGB (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.YUV:
        raise TypeError("YCgCoR_to_RGB: Clip is not in YUV format!")

    bd = c.format.bits_per_sample
    Y = core.std.ShufflePlanes(c, [0], vs.GRAY)
    Cg = core.std.ShufflePlanes(c, [1], vs.GRAY)
    Co = core.std.ShufflePlanes(c, [2], vs.GRAY)

    G = core.akarin.Expr([Y, Cg    ], ex_dlut("y range_half - dup yvar! 2 * x yvar@ - +",             bd, fulls))
    B = core.akarin.Expr([Y, Cg, Co], ex_dlut("x y range_half - - z range_half - 0.5 * -", bd, fulls))
    R = core.akarin.Expr([Co, B    ], ex_dlut("y x range_half - 2 * +",                    bd, fulls))

    return core.std.ShufflePlanes([R, G, B], [0, 0, 0], vs.RGB)


def RGB_to_OPP (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.RGB:
        raise TypeError("RGB_to_YCgCoR: Clip is not in RGB format!")

    bd = c.format.bits_per_sample
    R = core.std.ShufflePlanes(c, [0], vs.GRAY)
    G = core.std.ShufflePlanes(c, [1], vs.GRAY)
    B = core.std.ShufflePlanes(c, [2], vs.GRAY)

    b32 = "" if bd == 32 else "range_half +"

    O  = core.akarin.Expr([R, G, B], ex_dlut("x y z + + 0.333333333 *",     bd, fulls))
    P1 = core.akarin.Expr([R,    B], ex_dlut("x y - 0.5 * "+b32,            bd, fulls))
    P2 = core.akarin.Expr([R, G, B], ex_dlut("x z + 0.25 * y 0.5 * - "+b32, bd, fulls))

    return core.std.ShufflePlanes([O, P1, P2], [0, 0, 0], vs.YUV)


def OPP_to_RGB (c: vs.VideoNode, fulls: bool = False):
    if c.format.color_family != vs.YUV:
        raise TypeError("YCgCoR_to_RGB: Clip is not in YUV format!")

    bd = c.format.bits_per_sample
    O = core.std.ShufflePlanes(c, [0], vs.GRAY)
    P1 = core.std.ShufflePlanes(c, [1], vs.GRAY)
    P2 = core.std.ShufflePlanes(c, [2], vs.GRAY)

    b32 = "" if bd == 32 else "range_half -"

    R = core.akarin.Expr([O, P1, P2], ex_dlut("x y "+b32+" + z "+b32+" 0.666666666 * +", bd, fulls))
    G = core.akarin.Expr([O,     P2], ex_dlut("x y "+b32+" 1.333333333 * -",             bd, fulls))
    B = core.akarin.Expr([O, P1, P2], ex_dlut("x z "+b32+" 0.666666666 * + y "+b32+" -", bd, fulls))

    return core.std.ShufflePlanes([R, G, B], [0, 0, 0], vs.RGB)

# HBD constants 3D look up table
#
# * YUV and RGB mid-grey is 127.5 (rounded to 128) for PC range levels,
#   this translates to a value of 125.5 in TV range levels. Chroma is always centered, so 128 regardless.

def ex_dlut(expr: str = "", bits: int = 8, fulls: bool = False) -> str:
    bitd = \
        0 if bits == 8 else \
        1 if bits == 10 else \
        2 if bits == 12 else \
        3 if bits == 14 else \
        4 if bits == 16 else \
        5 if bits == 24 else \
        6 if bits == 32 else -1
    if bitd < 0:
        raise ValueError(f"ex_dlut: Unsupported bit depth ({bits})")
    
    #                 8-bit UINT      10-bit UINT          12-bit UINT          14-bit UINT            16-bit UINT         24-bit UINT               32-bit Ufloat
    range_min   = [  (  0.,  0.),    (   0.,   0.   ),    (   0.,   0.   ),    (    0.,    0.   ),    (    0.,    0.),    (       0.,       0.),    (       0.,       0.)   ]   [bitd]
    ymin        = [  ( 16., 16.),    (  64.,  64.   ),    ( 256., 257.   ),    ( 1024., 1028.   ),    ( 4096., 4112.),    ( 1048576., 1052672.),    (  16/255.,  16/255.)   ]   [bitd]
    cmin        = [  ( 16., 16.),    (  64.,  64.   ),    ( 256., 257.   ),    ( 1024., 1028.   ),    ( 4096., 4112.),    ( 1048576., 1052672.),    (  16/255.,  16/255.)   ]   [bitd]
    ygrey       = [  (126.,126.),    ( 502., 504.   ),    (2008.,2016.   ),    ( 8032., 8063.   ),    (32128.,32254.),    ( 8224768., 8256896.),    ( 125.5/255.,125.5/255.)]   [bitd]
    range_half  = [  (128.,128.),    ( 512., 514.   ),    (2048.,2056.   ),    ( 8192., 8224.   ),    (32768.,32896.),    ( 8388608., 8421376.),    ( 128/255., 128/255.)   ]   [bitd]
    yrange      = [  (219.,219.),    ( 876., 879.   ),    (3504.,3517.688),    (14016.,14070.750),    (56064.,56283.),    (14352384.,14408448.),    ( 219/255., 219/255.)   ]   [bitd]
    crange      = [  (224.,224.),    ( 896., 899.500),    (3584.,3598.   ),    (14336.,14392.   ),    (57344.,57568.),    (14680064.,14737408.),    ( 224/255., 224/255.)   ]   [bitd]
    ymax        = [  (235.,235.),    ( 940., 943.672),    (3760.,3774.688),    (15040.,15098.750),    (60160.,60395.),    (15400960.,15461120.),    ( 235/255., 235/255.)   ]   [bitd]
    cmax        = [  (240.,240.),    ( 960., 963.750),    (3840.,3855.   ),    (15360.,15420.   ),    (61440.,61680.),    (15728640.,15790080.),    ( 240/255., 240/255.)   ]   [bitd]
    range_max   = [  (255.,255.),    (1020.,1023.984),    (4080.,4095.938),    (16320.,16383.750),    (65280.,65535.),    (16711680.,16776960.),    (       1.,       1.)   ]   [bitd]
    range_size  = [  (256.,256.),    (1024.,1024.   ),    (4096.,4096.   ),    (16384.,16384.   ),    (65536.,65536.),    (16777216.,16777216.),    (       1.,       1.)   ]   [bitd]

    fs  = 1 if fulls else 0
    expr = expr.replace("ymax ymin - range_max /", str(yrange[fs]/range_max[fs]))
    expr = expr.replace("cmax cmin - range_max /", str(crange[fs]/range_max[fs]))
    expr = expr.replace("cmax ymin - range_max /", str(crange[fs]/range_max[fs]))
    expr = expr.replace("range_max ymax ymin - /", str(range_max[fs]/yrange[fs]))
    expr = expr.replace("range_max cmax cmin - /", str(range_max[fs]/crange[fs]))
    expr = expr.replace("range_max cmax ymin - /", str(range_max[fs]/crange[fs]))
    expr = expr.replace("ymax ymin -",             str(yrange[fs]))
    expr = expr.replace("cmax ymin -",             str(crange[fs]))
    expr = expr.replace("cmax cmin -",             str(crange[fs]))

    expr = expr.replace("ygrey",                   str(ygrey[fs]))
    expr = expr.replace("ymax",                    str(ymax[fs]))
    expr = expr.replace("cmax",                    str(cmax[fs]))
    expr = expr.replace("ymin",                    str(ymin[fs]))
    expr = expr.replace("cmin",                    str(cmin[fs]))
    expr = expr.replace("range_min",               str(range_min[fs]))
    expr = expr.replace("range_half",              str(range_half[fs]))
    expr = expr.replace("range_max",               str(range_max[fs]))
    expr = expr.replace("range_size",              str(range_size[fs]))
    return expr
